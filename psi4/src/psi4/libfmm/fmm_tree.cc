#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_shell_pair.h"
#include "psi4/libfmm/fmm_box.h"
#include "psi4/libfmm/fmm_tree.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/multipoles.h"
#include "psi4/libmints/overlap.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#define bohr2ang 0.52917720859

namespace psi {

// determine most positive and most negative molecular coordinates 
std::pair<double, double> parse_molecular_dims(std::shared_ptr<Molecule> molecule) {
    // define molecule params
    double min_dim_mol = molecule->x(0);
    double max_dim_mol = molecule->x(0);

    for (int atom = 0; atom < molecule->natom(); atom++) {
        double x = molecule->x(atom);
        double y = molecule->y(atom);
        double z = molecule->z(atom);
        min_dim_mol = std::min(x, min_dim_mol);
        min_dim_mol = std::min(y, min_dim_mol);
        min_dim_mol = std::min(z, min_dim_mol);
        max_dim_mol = std::max(x, max_dim_mol);
        max_dim_mol = std::max(y, max_dim_mol);
        max_dim_mol = std::max(z, max_dim_mol);
    }

    max_dim_mol += 0.05; // Add a small buffer to the box
    min_dim_mol -= 0.05; // Add a small buffer to the box
    
    return { max_dim_mol, min_dim_mol };
}

void CFMMTree::generate_per_level_info() {
    level_to_box_count_.fill(0);
    level_to_shell_count_.fill(0);
    int level_prev = 0;
    for (int bi = 0; bi < tree_.size(); bi++) {
        std::shared_ptr<CFMMBox> box = tree_[bi];
        auto sp = box->get_shell_pairs();
        int nshells = sp.size();
        int level = box->get_level();
        if (nshells > 0) {
            ++level_to_box_count_[level];
            level_to_shell_count_[level] += nshells;
        }
    }
}
 
CFMMTree::CFMMTree(std::shared_ptr<BasisSet> basis, Options& options) 
                    : basisset_(basis), options_(options) {

    //== baseline param setup ==//
    molecule_ = basisset_->molecule();
   
    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    print_ = options_.get_int("PRINT");
    bench_ = options_.get_int("BENCH");

    density_screening_ = (options_.get_str("SCREENING") == "DENSITY");
    ints_tolerance_ = options_.get_double("INTS_TOLERANCE");

    //== cfmm-specific param setup ==//
    lmax_ = options_.get_int("CFMM_ORDER");

    mpole_coefs_ = std::make_shared<HarmonicCoefficients>(lmax_, Regular);
    double cfmm_extent_tol = options.get_double("CFMM_EXTENT_TOLERANCE");

    // ints engine 
    auto factory = std::make_shared<IntegralFactory>(basisset_);
    auto shellpair_int = std::shared_ptr<TwoBodyAOInt>(factory->eri());

    auto& ints_shell_pairs = shellpair_int->shell_pairs();
    size_t nshell_pairs = ints_shell_pairs.size();
    shell_pairs_.resize(nshell_pairs);

#pragma omp parallel for
    for (size_t pair_index = 0; pair_index < nshell_pairs; pair_index++) {
        const auto& pair = ints_shell_pairs[pair_index];
        shell_pairs_[pair_index] = std::make_shared<CFMMShellPair>(basisset_, pair, mpole_coefs_, cfmm_extent_tol);
    }

    // ==> time to define the number of lowest-level boxes in the CFMM tree! <== // 
    int grain = options_.get_int("CFMM_GRAIN");
    
    // compute M_target based on multipole order as defined by White 1996
    int M_target_computed = 5*lmax_ - 5;  

    M_target_ = options_["CFMM_TARGET_NSHP"].has_changed() ? options_.get_int("CFMM_TARGET_NSHP") : M_target_computed * M_target_computed;
    
    // CFMM_GRAIN < -1 is invalid 
    if (grain < -1) { 
        std::string error_message = "CFMM grain set to below -1! If you meant to use adaptive CFMM, please set CFMM_GRAIN to exactly -1 or 0.";
        
        throw PSIEXCEPTION(error_message);
 
    // CFMM_GRAIN = -1 or 0 enables adaptive CFMM 
    } else if (grain == -1 || grain == 0) { 
        // Eq. 1 of White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
        N_target_ = ceil(static_cast<double>(nshell_pairs) / (static_cast<double>(M_target_) * g_)); 
        outfile->Printf("N_target: %d, %d, %f -> %d \n", nshell_pairs, M_target_, g_, N_target_);
   
    // CFMM_GRAIN > n (s.t. n > 0) uses n lowest-level boxes in the CFMM tree
    } else { 
        N_target_ = grain;
        outfile->Printf("N_target: %d \n", N_target_);
    }

    // Modified form of Eq. 2 of White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
    // further modifications arise from differences between regular and Continuous FMM
    //nlevels_ = 1 + ceil( log(N_target_) / ( (1 + static_cast<double>(dimensionality_)) * log(2) ));

    // root box
    nlevels_ = 1;
    int num_lowest_level_boxes = 1;

    // split root box into 8 parts spatially
    nlevels_ += 1;
    num_lowest_level_boxes *= std::pow(2, dimensionality_);

    // split box into 8 parts spatially, and 2 parts on WS
    while (num_lowest_level_boxes < N_target_) {
        nlevels_ += 1;
        num_lowest_level_boxes *= 2 * std::pow(2, dimensionality_);
        //num_lowest_level_boxes *= std::pow(2, dimensionality_);
    }
    outfile->Printf("nlevels: %d \n", nlevels_);
 
    if (nlevels_ <= 2) {
        std::string error_message = "Too few tree levels ("; 
        error_message += std::to_string(nlevels_);
        error_message += ")! Why do you wanna do CFMM with Direct SCF?";

        throw PSIEXCEPTION(error_message);
    } else if (nlevels_ >= max_nlevels_) {
        std::string error_message = "Too many tree levels ("; 
        error_message += std::to_string(nlevels_);
        error_message += ")! You memory hog.";

        throw PSIEXCEPTION(error_message);
    }

    int num_boxes = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;
    tree_.resize(num_boxes);

    timer_on("CFMMTree: Setup");

    for (int iter = 0; iter != 10; ++iter) {
        bool converged = false;
        outfile->Printf("CFMMTree: Setup Iter %d\n", iter);
        if (tree_[0] == nullptr) {
            outfile->Printf("  Make Root Node... ");
            make_root_node();
            outfile->Printf("  Done! \n");
        } else {
            outfile->Printf("  Regenerate Root Node... ");
            //int num_boxes = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;
            //tree_.resize(num_boxes);

            shellpair_list_.clear();
            sorted_leaf_boxes_.clear();
            converged = regenerate_root_node();
            //make_root_node();
            outfile->Printf("  Done! \n");
        }

        outfile->Printf("  Make Children... ");
        make_children();
        outfile->Printf("  Done! \n");

        outfile->Printf("  Sort Leaf Boxes... ");
        sort_leaf_boxes();
        outfile->Printf("  Done! \n");
    
        outfile->Printf("  Setup Regions... ");
        setup_regions();
        outfile->Printf("  Done! \n");
    
        outfile->Printf("  Setup Local Far Field Task Pairs... ");
        setup_local_far_field_task_pairs();
        outfile->Printf("  Done! \n");
    
        outfile->Printf("  Setup Shellpair Info... ");
        setup_shellpair_info();
        outfile->Printf("  Done! \n");
        
        if (converged) break;
    }

    if (print_ >= 2) print_out();
    
    outfile->Printf("  Calculate Shellpair Multipoles... ");
    calculate_shellpair_multipoles();
    outfile->Printf("  Done! \n");
    outfile->Printf("Completed CFMMTree: Setup! \n");

    timer_off("CFMMTree: Setup");

    // if (print_ >= 2) print_out();
}

void CFMMTree::sort_leaf_boxes() {

    // Starting and ending leaf node box indices
    int start = (nlevels_ == 1) ? 0 : (0.5 * std::pow(16, nlevels_-1) + 7) / 15;
    int end = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;

    for (int bi = start; bi < end; bi++) {
        std::shared_ptr<CFMMBox> box = tree_[bi];
        if (box->nshell_pair() > 0) sorted_leaf_boxes_.push_back(box);
    }

    auto box_compare = [](const std::shared_ptr<CFMMBox> &a, 
                                const std::shared_ptr<CFMMBox> &b) { return a->nshell_pair() > b->nshell_pair(); };

    std::sort(sorted_leaf_boxes_.begin(), sorted_leaf_boxes_.end(), box_compare);

}

void CFMMTree::make_root_node() {
    auto [max_dim, min_dim] = parse_molecular_dims(molecule_);

    Vector3 origin = Vector3(min_dim, min_dim, min_dim);
    double length = (max_dim - min_dim);

    // Scale root CFMM box for adaptive CFMM
    // Logic follows Eq. 3 of White 1996, adapted for CFMM 
    outfile->Printf("Original box volume (bohr): %f \n", std::pow(length, 3));
    outfile->Printf("Original box volume (ang): %f \n", std::pow(length * bohr2ang, 3));
    outfile->Printf("Original box origin: (%f, %f, %f) \n\n", origin[0] * bohr2ang, origin[1] * bohr2ang, origin[2] * bohr2ang);
    
    int num_lowest_level_boxes = 8 * std::pow(2,  (1 + dimensionality_) * (nlevels_ - 2)); 
    //int num_lowest_level_boxes = std::pow(2,  dimensionality_ * (nlevels_ - 1)); 
   // int num_lowest_level_boxes_per_branch = static_cast<double>(num_lowest_level_boxes) / std::pow(2, (nlevels_ - 2)); 
    //double N_target_per_branch = static_cast<double>(N_target_) / std::pow(2, (nlevels_ - 2)); 

    double f = static_cast<double>(N_target_) / static_cast<double>(num_lowest_level_boxes); 
    outfile->Printf("f scaling factor: %f\n", f);
    if (f > 1.0) throw PSIEXCEPTION("Bad f scaling factor value!");

    //f = std::pow(f, 1.0 / dimensionality_); // account for CFMM tree structure
 
    double length_tmp = length;
    //length = length_tmp / std::pow(f, 1.0 / (1.0 + dimensionality_));
    length = length_tmp / std::pow(f, 1.0 / dimensionality_);
 
    min_dim -= (length - length_tmp) / 2.0;
    Vector3 origin_new = Vector3(min_dim, min_dim, min_dim);

    outfile->Printf("New box volume (bohr): %f, %f -> %f, %f\n\n", length_tmp, f, length, length*length*length);
    outfile->Printf("New box volume (ang): %f, %f -> %f, %f\n\n", length_tmp * bohr2ang, f, length * bohr2ang, std::pow(length * bohr2ang, 3));
    outfile->Printf("New box origin: (%f, %f, %f) \n\n", origin_new[0] * bohr2ang, origin_new[1] * bohr2ang, origin_new[2] * bohr2ang); 
   
    tree_[0] = std::make_shared<CFMMBox>(nullptr, shell_pairs_, origin_new, length, 0, lmax_, 2);
    //tree_[0] = std::make_shared<CFMMBox>(nullptr, shell_pairs_, origin, length, 0, lmax_, 2);
}

bool CFMMTree::regenerate_root_node() {
    bool converged = false;

    //outfile->Printf("    Define tree params\n");
    Vector3 origin_old = tree_[0]->origin();
    double length_old = tree_[0]->length();

    double min_dim_old = origin_old[0];

    // define molecule params
    auto [max_dim_mol, min_dim_mol] = parse_molecular_dims(molecule_);

    generate_per_level_info();

    //outfile->Printf("    Determine scaling factor\n");
    double nshp_per_box = static_cast<double>(level_to_shell_count_[this->nlevels() - 1]) / static_cast<double>(level_to_box_count_[this->nlevels() - 1]); 

    double scaling_factor = static_cast<double>(nshp_per_box)/static_cast<double>(M_target_); 
    // double scaling_factor = 1.0;

    if (0.975 <= scaling_factor && scaling_factor <= 1.025) {
        converged = true;
    } 
    
    // Scale root CFMM box for adaptive CFMM
    // Logic follows Eq. 3 of White 1996, adapted for CFMM 
    //outfile->Printf("    Do scaling\n");
    outfile->Printf("Original box volume (bohr): %f \n", std::pow(length_old, 3));
    outfile->Printf("Original box volume (ang): %f \n", std::pow(length_old * bohr2ang, 3));
    outfile->Printf("Original box origin: (%f, %f, %f) \n\n", origin_old[0] * bohr2ang, origin_old[1] * bohr2ang, origin_old[2] * bohr2ang);
    
    double length_tmp = length_old;
    double length = length_tmp / std::pow(scaling_factor, 1.0 / dimensionality_);

    double min_dim = min_dim_old - (length - length_tmp) / 2.0;
    Vector3 origin_new = Vector3(min_dim, min_dim, min_dim);

    // damp new origin
    origin_new = (origin_new + origin_old) / 2.0;

    outfile->Printf("New box volume (bohr): %f, %f -> %f, %f\n\n", length_tmp, scaling_factor, length, length*length*length);
    outfile->Printf("New box volume (ang): %f, %f -> %f, %f\n\n", length_tmp * bohr2ang, scaling_factor, length * bohr2ang, std::pow(length * bohr2ang, 3));
    outfile->Printf("New box origin: (%f, %f, %f) \n\n", origin_new[0] * bohr2ang, origin_new[1] * bohr2ang, origin_new[2] * bohr2ang); 

    //outfile->Printf("    Create root\n");
    tree_.clear();

    int num_boxes = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;
    tree_.resize(num_boxes);

    tree_[0] = std::make_shared<CFMMBox>(nullptr, shell_pairs_, origin_new, length, 0, lmax_, 2);
    //tree_[0] = std::make_shared<CFMMBox>(nullptr, shell_pairs_, origin, length, 0, lmax_, 2);
    return converged;
}

void CFMMTree::make_children() {

#pragma omp parallel
    {
        for (int level = 0; level <= nlevels_ - 2; level += 1) {
            int start, end;
            if (level == 0) {
                start = 0;
                end = 1;
            } else {
                start = (0.5 * std::pow(16, level) + 7) / 15;
                end = (0.5 * std::pow(16, level+1) + 7) / 15;
            }

#pragma omp for
            for (int bi = start; bi < end; bi++) {
                tree_[bi]->make_children();
                auto children = tree_[bi]->get_children();

                for (int ci = 0; ci < children.size(); ci++) {
                    int ti = (level == 0) ? ci + 1 : bi * 16 - 7 + ci;
                    tree_[ti] = children[ci];
                }
            }
        }
    }

}

void CFMMTree::setup_regions() {

#pragma omp parallel
    {
        for (int level = 0; level <= nlevels_ - 1; level += 1) {
            int start, end;
            if (level == 0) {
                start = 0;
                end = 1;
            } else {
                start = (0.5 * std::pow(16, level) + 7) / 15;
                end = (0.5 * std::pow(16, level+1) + 7) / 15;
            }

#pragma omp for
            for (int bi = start; bi < end; bi++) {
                if (tree_[bi]->nshell_pair() == 0) continue;
                tree_[bi]->set_regions();
            }
        }
    }

}

void CFMMTree::setup_shellpair_info() {

    size_t nsh = basisset_->nshell(); 

    shellpair_list_.resize(nsh);

    for (int P = 0; P != nsh; ++P) { 
        shellpair_list_[P].resize(nsh);
    }

    for (int i = 0; i < sorted_leaf_boxes_.size(); i++) {
        std::shared_ptr<CFMMBox> curr = sorted_leaf_boxes_[i];
        auto& shellpairs = curr->get_shell_pairs();
        auto& nf_boxes = curr->near_field_boxes();

        for (auto& sp : shellpairs) {
            auto [P, Q] = sp->get_shell_pair_index();

            shellpair_list_[P][Q] = { sp, curr };
        }
    }
}

bool CFMMTree::shell_significant(int P, int Q, int R, int S, std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                             const std::vector<SharedMatrix>& D) {
    if (density_screening_) {
        double D_PQ = 0.0;
        double D_RS = 0.0;

        double prefactor = (D.size() == 1) ? 4.0 : 2.0;

        for (int i = 0; i < D.size(); i++) {
            D_PQ += ints[0]->shell_pair_max_density(i, P, Q);
            D_RS += ints[0]->shell_pair_max_density(i, R, S);
        }

        double screen_val = prefactor * std::max(D_PQ, D_RS) * std::sqrt(ints[0]->shell_ceiling2(P, Q, R, S));

        if (screen_val >= ints_tolerance_) return true;
        else return false;

    } else {
        return ints[0]->shell_significant(P, Q, R, S);
    }
}

void CFMMTree::setup_local_far_field_task_pairs() {

    // First access is level, second access is the list of local far field per box for that box
    lff_task_pairs_per_level_.resize(nlevels_);

    // Build the task pairs
    for (int level = 0; level < nlevels_; level++) {
        int start, end;
        if (level == 0) {
            start = 0;
            end = 1;
        } else {
            start = (0.5 * std::pow(16, level) + 7) / 15;
            end = (0.5 * std::pow(16, level+1) + 7) / 15;
        }

        for (int bi = start; bi < end; bi++) {
            std::shared_ptr<CFMMBox> box = tree_[bi];
            if (box->nshell_pair() == 0) continue;
            for (auto& lff : box->local_far_field_boxes()) {
                if (lff->nshell_pair() == 0) continue;
                lff_task_pairs_per_level_[level].emplace_back(box, lff);
            }
        }
    }
}

void CFMMTree::calculate_shellpair_multipoles() {

    timer_on("CFMMTree: Shell-Pair Multipole Ints");

    std::vector<std::shared_ptr<OneBodyAOInt>> sints;
    std::vector<std::shared_ptr<OneBodyAOInt>> mpints;

    auto int_factory = std::make_shared<IntegralFactory>(basisset_);
    for (int thread = 0; thread < nthread_; thread++) {
        sints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_overlap()));
        mpints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_multipoles(lmax_)));
    }

#pragma omp parallel for collapse(2) schedule(guided)
    for (int Ptask = 0; Ptask < shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < shellpair_list_.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(shellpair_list_[Ptask][Qtask]);
            if (shellpair == nullptr) continue;
            
            std::shared_ptr<CFMMBox> box = std::get<1>(shellpair_list_[Ptask][Qtask]);
            
            auto [P, Q] = shellpair->get_shell_pair_index();
    
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
 
            mpints[thread]->set_origin(box->center());
            shellpair->calculate_mpoles(box->center(), sints[thread], mpints[thread], lmax_);
        }  
    }

    timer_off("CFMMTree: Shell-Pair Multipole Ints");
}

void CFMMTree::calculate_multipoles(const std::vector<SharedMatrix>& D) {
    timer_on("CFMMTree: Box Multipoles");

    // Compute mpoles for leaf nodes
#pragma omp parallel
    {
#pragma omp for
        for (int bi = 0; bi < sorted_leaf_boxes_.size(); bi++) {
            sorted_leaf_boxes_[bi]->compute_multipoles(basisset_, D);
        }

        // Calculate mpoles for higher level boxes
        for (int level = nlevels_ - 2; level >= 0; level -= 1) {
            int start, end;
            if (level == 0) {
                start = 0;
                end = 1;
            } else {
                start = (0.5 * std::pow(16, level) + 7) / 15;
                end = (0.5 * std::pow(16, level+1) + 7) / 15;
            }

#pragma omp for
            for (int bi = start; bi < end; bi++) {
                if (tree_[bi]->nshell_pair() == 0) continue;
                tree_[bi]->compute_mpoles_from_children();
            }
        }
    }

    timer_off("CFMMTree: Box Multipoles");
}

void CFMMTree::compute_far_field() {

    timer_on("CFMMTree: Far Field Vector");

#pragma omp parallel
    {
        for (int level = 0; level < nlevels_; level++) {
            const auto& all_box_pairs = lff_task_pairs_per_level_[level];
#pragma omp for
            for (int box_pair = 0; box_pair < all_box_pairs.size(); box_pair++) {
                std::shared_ptr<CFMMBox> box1 = all_box_pairs[box_pair].first;
                std::shared_ptr<CFMMBox> box2 = all_box_pairs[box_pair].second;
                box1->compute_far_field_contribution(box2);
            }

            int start, end;
            if (level == 0) {
                start = 0;
                end = 1;
            } else {
                start = (0.5 * std::pow(16, level) + 7) / 15;
                end = (0.5 * std::pow(16, level+1) + 7) / 15;
            }

#pragma omp for
            for (int bi = start; bi < end; bi++) {
                if (tree_[bi]->nshell_pair() == 0) continue;
                tree_[bi]->add_parent_far_field_contribution();
            }
        }
    }

    timer_off("CFMMTree: Far Field Vector");

}

void CFMMTree::build_nf_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                          const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("CFMMTree: Near Field J");

    int nshell = basisset_->nshell();
    int natom = molecule_->natom();

    // Maximum space (r_nbf * s_nbf) to allocate per task
    size_t max_alloc = 0;

    // A map of the function (num_r * num_s) offsets per shell-pair in a box pair
    std::unordered_map<int, int> offsets;
    
    int start = (nlevels_ == 1) ? 0 : (0.5 * std::pow(16, nlevels_-1) + 7) / 15;
    int end = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;

    for (int bi = start; bi < end; bi++) {
        auto& RSshells = tree_[bi]->get_shell_pairs();
        int RSoff = 0;
        for (int RSind = 0; RSind < RSshells.size(); RSind++) {
            auto [R, S] = RSshells[RSind]->get_shell_pair_index();
            offsets[R * nshell + S] = RSoff;
            int Rfunc = basisset_->shell(R).nfunction();
            int Sfunc = basisset_->shell(S).nfunction();
            RSoff += Rfunc * Sfunc;
        }
        max_alloc = std::max((size_t) RSoff, max_alloc);
    }

    // Make intermediate buffers (for threading purposes and take advantage of 8-fold perm sym)
    std::vector<std::vector<std::vector<double>>> JT;
    for (int thread = 0; thread < nthread_; thread++) {
        std::vector<std::vector<double>> J2;
        for (size_t N = 0; N <D.size(); N++) {
            std::vector<double> temp(2 * max_alloc);
            J2.push_back(temp);
        }
        JT.push_back(J2);
    }

    // Benchmark Number of Computed Shells
    num_computed_shells_ = 0L;
    size_t computed_shells = 0L;

#pragma omp parallel for collapse(2) schedule(guided) reduction(+ : computed_shells)
    for (int Ptask = 0; Ptask < shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < shellpair_list_.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(shellpair_list_[Ptask][Qtask]);
            
            if (shellpair == nullptr) continue;
       
            std::shared_ptr<CFMMBox> box = std::get<1>(shellpair_list_[Ptask][Qtask]);
            std::vector<std::shared_ptr<CFMMBox>> nf_boxes = box->near_field_boxes(); 
 
            auto [P, Q] = shellpair->get_shell_pair_index();
            
            const GaussianShell& Pshell = basisset_->shell(P);
            const GaussianShell& Qshell = basisset_->shell(Q);

            int p_start = Pshell.start();
            int num_p = Pshell.nfunction();

            int q_start = Qshell.start();
            int num_q = Qshell.nfunction();

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
        
            for (const auto& nf_box : nf_boxes) {
                auto& RSshells = nf_box->get_shell_pairs();
                
                bool touched = false;
     
                for (const auto& RSsh : RSshells) {
                    auto [R, S] = RSsh->get_shell_pair_index();
                
                    if (R * nshell + S > P * nshell + Q) continue;
                    if (!shell_significant(P, Q, R, S, ints, D)) continue;
            
                    if (ints[thread]->compute_shell(P, Q, R, S) == 0) continue;
                    computed_shells++;

                    const GaussianShell& Rshell = basisset_->shell(R);
                    const GaussianShell& Sshell = basisset_->shell(S);

                    int r_start = Rshell.start();
                    int num_r = Rshell.nfunction();
    
                    int s_start = Sshell.start();
                    int num_s = Sshell.nfunction();
    
                    double prefactor = 1.0;
                    if (P != Q) prefactor *= 2;
                    if (R != S) prefactor *= 2;
                    if (P == R && Q == S) prefactor *= 0.5;
    
                    int RSoff = offsets[R * nshell + S];
    
                    const double* pqrs = ints[thread]->buffer();
    
                    for (int N = 0; N <D.size(); N++) {
                        double** Jp = J[N]->pointer();
                        double** Dp =D[N]->pointer();
                        double* JTp = JT[thread][N].data();
                        const double* pqrs2 = pqrs;
                        
                        if (!touched) {
                            ::memset((void*)(&JTp[0L * max_alloc]), '\0', max_alloc * sizeof(double));
                            ::memset((void*)(&JTp[1L * max_alloc]), '\0', max_alloc * sizeof(double));
                        }
    
                        // Contraction into box shell pairs to improve parallel performance
                        double* J1p = &JTp[0L * max_alloc];
                        double* J2p = &JTp[1L * max_alloc];
    
                        for (int p = p_start; p < p_start + num_p; p++) {
                            int dp = p - p_start;
                            for (int q = q_start; q < q_start + num_q; q++) {
                                int dq = q - q_start;
                                for (int r = r_start; r < r_start + num_r; r++) {
                                    int dr = r - r_start;
                                    for (int s = s_start; s < s_start + num_s; s++) {
                                        int ds = s - s_start;
    
                                        int pq = dp * num_q + dq;
                                        int rs = RSoff + dr * num_s + ds;
    
                                        J1p[pq] += prefactor * (*pqrs2) * Dp[r][s];
                                        J2p[rs] += prefactor * (*pqrs2) * Dp[p][q];
                                        pqrs2++;
                                    } // end s
                                } // end r
                            } // end q
                        } // end p
                    } // end N
                    touched = true;
                } // end RSshells
                if (!touched) continue;
    
                // = > Stripeout < = //
                for (int N = 0; N < D.size(); N++) {
                    double** Jp = J[N]->pointer();
                    double** Dp = D[N]->pointer();
                    double* JTp = JT[thread][N].data();
    
                    double* J1p = &JTp[0L * max_alloc];
                    double* J2p = &JTp[1L * max_alloc];
    
                    for (int p = p_start; p < p_start + num_p; p++) {
                        int dp = p - p_start;
                        for (int q = q_start; q < q_start + num_q; q++) {
                            int dq = q - q_start;
                                
                            int pq = dp * num_q + dq;
#pragma omp atomic
                            Jp[p][q] += J1p[pq];
                        }
                    }
            
                    for (const auto& RSsh : RSshells) {
                        auto [R, S] = RSsh->get_shell_pair_index();
    
                        int RSoff = offsets[R * nshell + S];
                    
                        const GaussianShell& Rshell = basisset_->shell(R);
                        const GaussianShell& Sshell = basisset_->shell(S);
                    
                        int r_start = Rshell.start();
                        int num_r = Rshell.nfunction();
    
                        int s_start = Sshell.start();
                        int num_s = Sshell.nfunction();
                    
                        for (int r = r_start; r < r_start + num_r; r++) {
                            int dr = r - r_start;
                            for (int s = s_start; s < s_start + num_s; s++) {
                                int ds = s - s_start;
                                int rs = RSoff + dr * num_s + ds;
#pragma omp atomic
                                Jp[r][s] += J2p[rs];
                            }
                        }
                    }
                }
                // => End Stripeout <= //
            } // end nf_box
        } // end Qtasks
    } // end Ptasks
    
    num_computed_shells_ = computed_shells; 

    timer_off("CFMMTree: Near Field J");
}

void CFMMTree::build_ff_J(std::vector<SharedMatrix>& J) {

    timer_on("CFMMTree: Far Field J");

#pragma omp parallel for collapse(2) schedule(guided)
    for (int Ptask = 0; Ptask < shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < shellpair_list_.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(shellpair_list_[Ptask][Qtask]);
            if (shellpair == nullptr) continue;
    
            auto [P, Q] = shellpair->get_shell_pair_index();
    
            const auto& Vff = std::get<1>(shellpair_list_[Ptask][Qtask])->far_field_vector();
                
            const auto& shellpair_mpoles = shellpair->get_mpoles();
     
            double prefactor = (P == Q) ? 1.0 : 2.0;
        
            const GaussianShell& Pshell = basisset_->shell(P);
            const GaussianShell& Qshell = basisset_->shell(Q);
    
            int p_start = Pshell.start();
            int num_p = Pshell.nfunction();
    
            int q_start = Qshell.start();
            int num_q = Qshell.nfunction();
    
            for (int p = p_start; p < p_start + num_p; p++) {
                int dp = p - p_start;
                for (int q = q_start; q < q_start + num_q; q++) {
                    int dq = q - q_start;
                    for (int N = 0; N < J.size(); N++) {
                        double** Jp = J[N]->pointer();
                        // Far field multipole contributions
#pragma omp atomic
                        Jp[p][q] += prefactor * Vff[N]->dot(shellpair_mpoles[dp * num_q + dq]);
                    } // end N
                } // end q
            } // end p
        } // end Qtasks
    } // end Ptasks

    timer_off("CFMMTree: Far Field J");
}

void CFMMTree::build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                        const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("CFMMTree: J");
    
    std::vector<SharedMatrix> nf_J, ff_J;
    
    // Zero the J matrix
    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->zero();
        
        nf_J.push_back(std::make_shared<Matrix>(J[ind]->clone()));
        ff_J.push_back(std::make_shared<Matrix>(J[ind]->clone()));
    }

    // Update the densities
    for (int thread = 0; thread < nthread_; thread++) {
        ints[thread]->update_density(D);
    }

    // Compute multipoles and far field
    calculate_multipoles(D);
    compute_far_field();

    // Compute near field J and far field J
    build_nf_J(ints, D, nf_J);
    /*
    outfile->Printf("#========================# \n");
    outfile->Printf("#== Start Near-Field J ==# \n");
    outfile->Printf("#========================# \n\n");
    for (int ind = 0; ind < D.size(); ind++) {
         outfile->Printf("  Ind = %d \n", ind);
         outfile->Printf("  -------- \n");

         nf_J[ind]->print_out();
         outfile->Printf("\n");
    }
    outfile->Printf("#========================# \n");
    outfile->Printf("#==  End Near-Field J  ==# \n");
    outfile->Printf("#========================# \n");
    */

    build_ff_J(ff_J);
    /*
    outfile->Printf("#=======================# \n");
    outfile->Printf("#== Start Far-Field J ==# \n");
    outfile->Printf("#=======================# \n\n");
    for (int ind = 0; ind < D.size(); ind++) {
         outfile->Printf("  Ind = %d \n", ind);
         outfile->Printf("  -------- \n");
 
         ff_J[ind]->print_out();
         outfile->Printf("\n");
    }
    outfile->Printf("#=======================# \n");
    outfile->Printf("#==  End Far-Field J  ==# \n");
    outfile->Printf("#=======================# \n\n");
    */

    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->add(nf_J[ind]);
        J[ind]->add(ff_J[ind]);
    }
    
    // Hermitivitize J matrix afterwards
    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->hermitivitize();
    }

    timer_off("CFMMTree: J");
}

void CFMMTree::print_out() {
    // recursive print-out per box
    // outfile->Printf("CFMM BOX INFO:\n");
    // tree_[0]->print_out();  

    // overall tree summary 
    level_info();

    outfile->Printf("CFMM TREE LEVEL INFO:\n");
    int ilevel = 0;
    while (level_to_box_count_[ilevel] > 0) {
        outfile->Printf("  LEVEL: %d, BOXES: %d NSHP/BOX: %f \n", ilevel, level_to_box_count_[ilevel], static_cast<double>(level_to_shell_count_[ilevel]) / static_cast<double>(level_to_box_count_[ilevel]) );
        ++ilevel;
    }
} 

} // end namespace psi
