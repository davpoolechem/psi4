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

#include <optional>
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
    std::array<double, 3> min_dims_mol = { molecule->x(0), molecule->y(0), molecule->z(0) };
    std::array<double, 3> max_dims_mol = { molecule->x(0), molecule->y(0), molecule->z(0) };
    
    for (int atom = 0; atom < molecule->natom(); atom++) {
        double x = molecule->x(atom);
        double y = molecule->y(atom);
        double z = molecule->z(atom);
        min_dims_mol[0] = std::min(x, min_dims_mol[0]);
        min_dims_mol[1] = std::min(y, min_dims_mol[1]);
        min_dims_mol[2] = std::min(z, min_dims_mol[2]);
        max_dims_mol[0] = std::max(x, max_dims_mol[0]);
        max_dims_mol[1] = std::max(y, max_dims_mol[1]);
        max_dims_mol[2] = std::max(z, max_dims_mol[2]);
    }

    // define molecule params
    double min_dim_mol = *std::min_element(std::begin(min_dims_mol), std::end(min_dims_mol)); 
    double max_dim_mol = *std::max_element(std::begin(max_dims_mol), std::end(max_dims_mol));

/*
    outfile->Printf("Min dims:\n");
    outfile->Printf("  Min dim x: %f\n", min_dims_mol[0]);
    outfile->Printf("  Min dim y: %f\n", min_dims_mol[1]);
    outfile->Printf("  Min dim z: %f\n", min_dims_mol[2]);
    outfile->Printf("  Min dim total: %f\n", min_dim_mol);
    outfile->Printf("Max dims:\n");
    outfile->Printf("  Max dim x: %f\n", max_dims_mol[0]);
    outfile->Printf("  Max dim y: %f\n", max_dims_mol[1]);
    outfile->Printf("  Max dim z: %f\n", max_dims_mol[2]);
    outfile->Printf("  Max dim total: %f\n", max_dim_mol);
    outfile->Printf("Lengths:\n");
    outfile->Printf("  Length x: %f\n", max_dims_mol[0] - min_dims_mol[0]);
    outfile->Printf("  Length y: %f\n", max_dims_mol[1] - min_dims_mol[1]);
    outfile->Printf("  Length z: %f\n", max_dims_mol[2] - min_dims_mol[2]);
    outfile->Printf("\n");
*/
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
        auto sp = box->get_primary_shell_pairs();
        int nshells = sp.size();
        int level = box->get_level();
        if (nshells > 0) {
            ++level_to_box_count_[level];
            level_to_shell_count_[level] += nshells;
        }
    }
}

/*
void CFMMTree::generate_per_level_info_union() {
    level_to_box_count_.fill(0);
    level_to_shell_count_.fill(0);
    int level_prev = 0;
    for (int bi = 0; bi < tree_.size(); bi++) {
        std::shared_ptr<CFMMBox> box = tree_[bi];
        auto sp = box->get_shell_pairs();
        int nshells = sp.size();
        int level = box->get_level();
        int ws = box->get_ws();
        auto center = box->center();

        if (nshells > 0) {
            ++level_to_box_count_[level];
            level_to_shell_count_[level] += nshells;
        }
    }
}
 */

void CFMMTree::set_nlevels(int nlevels) {
    // sanity check on number of levels in CFMM tree
    // shouldnt be too small (CFMM has no effect)...
    if (nlevels <= 2) {
        std::string error_message = "Too few tree levels ("; 
        error_message += std::to_string(nlevels_);
        error_message += ")! Why do you wanna do CFMM with Direct SCF?";

        throw PSIEXCEPTION(error_message);
    
    // ...nor too large (too much memory usage) 
    } else if (nlevels >= max_nlevels_) {
        std::string error_message = "Too many tree levels ("; 
        error_message += std::to_string(nlevels_);
        error_message += ")! You memory hog.";

        throw PSIEXCEPTION(error_message);
    
    // otherwise we are okay
    } else {
        nlevels_ = nlevels;
    }
}

CFMMTree::CFMMTree(std::shared_ptr<BasisSet> primary, Options& options) 
                    : primary_(primary), options_(options) {
    timer_on("CFMMTree: Setup");

    // ==> ---------------- <== //
    // ==> setup parameters <== //
    // ==> ---------------- <== //
    
    // baseline params
    molecule_ = primary_->molecule();
   
    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    print_ = options_.get_int("PRINT");
    bench_ = options_.get_int("BENCH");

    density_screening_ = (options_.get_str("SCREENING") == "DENSITY");
    ints_tolerance_ = options_.get_double("INTS_TOLERANCE");

    // CFMM-specific params 
    lmax_ = options_.get_int("CFMM_ORDER");

    mpole_coefs_ = std::make_shared<HarmonicCoefficients>(lmax_, Regular);

    // compute M_target_ based on multipole order as defined by White 1996
    // M_target_ is the average number of distributions (shell pairs) per
    // occupied lowest-level box for adaptive CFMM
    int M_target_computed = 5*lmax_ - 5;  

    M_target_ = options_["CFMM_TARGET_NSHP"].has_changed() ? options_.get_int("CFMM_TARGET_NSHP") : M_target_computed * M_target_computed;
    
    timer_off("CFMMTree: Setup");
}

//CFMMTree::CFMMTree(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options), { CFMMTree(primary, options) };

void CFMMTree::make_tree(int nshell_pairs) {
    timer_on("CFMMTree: Make Tree");
    
    // ==> define the number of lowest-level boxes in the CFMM tree! <== // 
    int grain = options_.get_int("CFMM_GRAIN");
  
    // CFMM_GRAIN < -1 is invalid 
    if (grain < -1) { 
        std::string error_message = "CFMM grain set to below -1! If you meant to use adaptive CFMM, please set CFMM_GRAIN to exactly -1 or 0.";
        
        throw PSIEXCEPTION(error_message);
 
    // CFMM_GRAIN = -1 or 0 enables adaptive CFMM 
    } else if (grain == -1 || grain == 0) { 
        // nshell_pairs must be set to something if using adaptive CFMM
        if (nshell_pairs <= 0) {
            throw PSIEXCEPTION("CFMMTree::make_tree called, but nshell_pairs is set to 0 or less!");
        }

        // Eq. 1 of White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
        g_ = 1.0; // perfectly spherical molecule for now 
        
        N_target_ = ceil(static_cast<double>(nshell_pairs) / (static_cast<double>(M_target_) * g_)); 
        //outfile->Printf("N_target: %d, %d, %f -> %d \n", nshell_pairs, M_target_, g_, N_target_);
   
    // CFMM_GRAIN > n (s.t. n > 0) uses n lowest-level boxes in the CFMM tree
    } else { 
        N_target_ = grain;
        //outfile->Printf("N_target: %d \n", N_target_);
    }

    // ==> Determine number of levels in CFMM tree <== //
    
    // Modified form of Eq. 2 of White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
    // further modifications arise from differences between regular and Continuous FMM
    //nlevels_ = 1 + ceil( log(N_target_) / ( (1 + static_cast<double>(dimensionality_)) * log(2) ));

    // root box
    nlevels_ = 1;
    num_boxes_ = 1;
    int num_lowest_level_boxes = 1;

    // split root box into 8 parts spatially
    nlevels_ += 1;
    num_lowest_level_boxes *= std::pow(2, dimensionality_);
    num_boxes_ += num_lowest_level_boxes;

    // split box into 8 parts spatially, and 2 parts on WS
    while (num_lowest_level_boxes < N_target_) {
        set_nlevels(nlevels_ + 1);
        num_lowest_level_boxes *= 2 * std::pow(2, dimensionality_);
        //num_lowest_level_boxes *= std::pow(2, dimensionality_);
        num_boxes_ += num_lowest_level_boxes;
    }
    
    set_nlevels(nlevels_); // here to sanity-check in case above while-loop is skipped
    //outfile->Printf("nlevels: %d \n", nlevels_);

    assert(num_boxes_ == (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15);
    num_boxes_ = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;
    tree_.resize(num_boxes_);

    // ==> ------------------------------ <== //
    // ==> Actually create CFMM tree now! <== //
    // ==> ------------------------------ <== //
    outfile->Printf("  ==> CFMM Tree Setup <== \n\n");
   
    // do "adaptive" CFMM scheme if desired...
    if (grain <= 0) {
        int niter = options_.get_int("CFMM_TREE_MAXITER");

        bool converged = false;
        for (int iter = 0; iter != niter; ++iter) {
            if (print_ >= 2) outfile->Printf("  Iteration: %i\n", iter);

            if (tree_[0] == nullptr) {
                make_root_node();
                outfile->Printf("  Tree length #0b: %f\n", tree_[0]->length());
            } else {
                primary_shellpair_list_.clear();
                sorted_leaf_boxes_.clear();
             
                bool changed_level;   
                std::tie(converged, changed_level) = regenerate_root_node();
                
                if (changed_level) continue;
            }

            outfile->Printf("  Tree length #1: %f\n", tree_[0]->length());
            make_children();
            outfile->Printf("  Tree length #2: %f\n", tree_[0]->length());
            sort_leaf_boxes();
            outfile->Printf("  Tree length #3: %f\n", tree_[0]->length());
            setup_regions();
            outfile->Printf("  Tree length #4: %f\n", tree_[0]->length());
            setup_local_far_field_task_pairs();
            outfile->Printf("  Tree length #5: %f\n", tree_[0]->length());
            setup_shellpair_info();
            outfile->Printf("  Tree length #6: %f\n\n", tree_[0]->length());
            
            if (converged) break;
        }
    
        generate_per_level_info();

        if (!converged) {
            std::string warning_message = "    WARNING: CFMM tree box scaling did not converge! Using ";
            warning_message += std::to_string(distributions()); 
            warning_message += " shell pairs per box instead of target (";
            warning_message += std::to_string(M_target_); 
            warning_message += ")\n\n";

            outfile->Printf(warning_message);
            //std::string error_message = "Adaptive CFMM tree box scaling did not converge!";
   
            //throw PSIEXCEPTION(error_message);
        }
    
    // ... otherwise just specify tree according to CFMM_GRAIN
    } else {
        make_root_node();
        make_children();
        sort_leaf_boxes();
        setup_regions();
        setup_local_far_field_task_pairs();
        setup_shellpair_info();
    }

    // early kill?
    //if (options_.get_bool("CFMM_TREE_DEBUG")) throw PSIEXCEPTION("Early kill for CFMM debugging!");
    
    timer_off("CFMMTree: Make Tree");
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

std::tuple<Vector3, double> CFMMTree::make_root_node_kernel() {
    // get basic molecular bounds
    auto [max_dim, min_dim] = parse_molecular_dims(molecule_);

    Vector3 origin = Vector3(min_dim, min_dim, min_dim);
    double length = (max_dim - min_dim);

    if (print_ >= 2) { 
        outfile->Printf("    Original box volume (bohr): %f \n", std::pow(length, 3));
        outfile->Printf("    Original box volume (ang): %f \n", std::pow(length * bohr2ang, 3));
        outfile->Printf("    Original box origin: (%f, %f, %f) \n\n", origin[0] * bohr2ang, origin[1] * bohr2ang, origin[2] * bohr2ang);
    }
    
    // Scale root CFMM box for adaptive CFMM
    // Logic follows Eq. 3 of White 1996, adapted for CFMM 
    int num_lowest_level_boxes = 8 * std::pow(2,  (1 + dimensionality_) * (nlevels_ - 2)); 
    //int num_lowest_level_boxes = std::pow(2,  dimensionality_ * (nlevels_ - 1)); 

    f_ = static_cast<double>(N_target_) / static_cast<double>(num_lowest_level_boxes); 
    if (f_ > 1.0) {
        std::string error_message = "Bad f scaling factor value! ";
        error_message += std::to_string(N_target_);
        error_message += " N_target_ vs. ";
        error_message += std::to_string(num_lowest_level_boxes);
        error_message += " num_lowest_level_boxes.";

        throw PSIEXCEPTION(error_message);
    } 

    double length_tmp = length;
    length = length_tmp / std::pow(f_, 1.0 / dimensionality_);
 
    min_dim -= (length - length_tmp) / 2.0;
    Vector3 origin_new = Vector3(min_dim, min_dim, min_dim);

    if (print_ >= 2) {
        outfile->Printf("    f scaling factor: %d, %d -> %f\n", N_target_, num_lowest_level_boxes, f_);
        outfile->Printf("    New box volume (bohr): %f, %f -> %f, %f\n", length_tmp, f_, length, length * length * length);
        outfile->Printf("    New box volume (ang): %f, %f -> %f, %f\n", length_tmp * bohr2ang, f_, length * bohr2ang, std::pow(length * bohr2ang, 3));
        outfile->Printf("    New box origin: (%f, %f, %f) \n\n", origin_new[0] * bohr2ang, origin_new[1] * bohr2ang, origin_new[2] * bohr2ang); 
    }

    return std::tie(origin_new, length);
}

std::tuple<Vector3, double, bool, bool> CFMMTree::regenerate_root_node_kernel() {
    // some basic variables used throughout 
    bool converged = false; // have we converged the box scaling?
    bool changed_level = false; // do we need to change the number of levels in the CFMM tree? 
    
    Vector3 origin_old = tree_[0]->origin();
    double length_old = tree_[0]->length();

    Vector3 origin_new; 
    double length_new;

    // determine how much box needs to be scaled to reach target distribution value
    //outfile->Printf("    Determine scaling factor\n");
    generate_per_level_info();
    
    size_t num_lowest_level_boxes = level_to_box_count_[nlevels() - 1]; 
    size_t num_lowest_level_shells = level_to_shell_count_[nlevels() - 1]; 
    double nshp_per_box = static_cast<double>(num_lowest_level_shells) / static_cast<double>(num_lowest_level_boxes); 

    double scaling_factor = static_cast<double>(nshp_per_box)/static_cast<double>(M_target_); 
    
    // if here, we have converged the adaptive CFMM scheme, no need for more....
    double percent_error = static_cast<double>(options_.get_int("CFMM_TREE_PRECISION")) / 100.0; 
    if ((1.0 - percent_error) <= scaling_factor && scaling_factor <= (1.0 + percent_error)) { 
        origin_new = origin_old; 
        length_new = length_old; 
 
        converged = true;
    
    // ... otherwise, we need to rescale the root CFMM box again 
    } else { 
        // Scale root CFMM box for adaptive CFMM
        // Logic follows Eq. 3 of White 1996, adapted for CFMM 
        //outfile->Printf("    Do scaling\n");
        if (print_ >= 2) {
            outfile->Printf("    Original f scaling factor: %f\n", f_);
            outfile->Printf("    Original box volume (bohr): %f \n", std::pow(length_old, 3));
            outfile->Printf("    Original box volume (ang): %f \n", std::pow(length_old * bohr2ang, 3));
            outfile->Printf("    Original box origin: (%f, %f, %f) \n\n", origin_old[0] * bohr2ang, origin_old[1] * bohr2ang, origin_old[2] * bohr2ang);
        }

        double min_dim_old = origin_old[0];
        
        double length_tmp = length_old;
        length_new = length_tmp / std::pow(scaling_factor, 1.0 / dimensionality_);

        double min_dim = min_dim_old - (length_new - length_tmp) / 2.0;
   
        // damp expansion of CFMM box
        double expansion_thresh = scaling_factor >= 1.0 ? 1.5 - (1.5 / scaling_factor) : (1.5 / scaling_factor) - 1.5;
        //double expansion_thresh = scaling_factor >= 1.0 ? 1.0 - (1.0 / scaling_factor) : (1.0 / scaling_factor) - 1.0;
        expansion_thresh = std::min(1.0, std::abs(expansion_thresh)); // box can shrink/expand by 1 Bohr at most 
     
        if (print_ >= 2) outfile->Printf("    %f -> %f\n", min_dim_old * bohr2ang, min_dim * bohr2ang);
        
        if ((min_dim - min_dim_old) > expansion_thresh) {
            if (print_ >= 2) outfile->Printf("    INFO: Damping box shrinkage to %f A\n", expansion_thresh * bohr2ang);
            min_dim = min_dim_old + expansion_thresh; 

            length_new = length_old - expansion_thresh;
        } else if ((min_dim - min_dim_old) < -expansion_thresh) {
            if (print_ >= 2) outfile->Printf("    INFO: Damping box expansion to %f A\n", expansion_thresh * bohr2ang);
            min_dim = min_dim_old - expansion_thresh; 

            length_new = length_old + expansion_thresh;
        }
 
        origin_new = Vector3(min_dim, min_dim, min_dim);

        // if the root box becomes smaller than the molecule, increase the tree level and reboot the process... 
        auto [max_dim_mol, min_dim_mol] = parse_molecular_dims(molecule_);
        if (std::abs(origin_new[0]) < std::abs(min_dim_mol)) {
            if (print_ >= 2) {
                outfile->Printf("    WARNING: Root CFMM box has iterated to a size smaller than the molecule. \n");

                std::string message = "    Changing number of CFMM tree levels from ";
                message += std::to_string(nlevels_);
                message += " to ";
                message += std::to_string(nlevels_ + 1);
                message += ".\n\n";

                outfile->Printf(message);
            }
            //outfile->Printf("\n");

            set_nlevels(nlevels_ + 1);

            int prev_level_boxes = 8 * std::pow(2,  (1 + dimensionality_) * (nlevels_ - 3)); 
            int current_level_boxes = 8 * std::pow(2,  (1 + dimensionality_) * (nlevels_ - 2)); 

            N_target_ *= current_level_boxes/prev_level_boxes; 
            
            changed_level = true;  
        
        // ... otherwise regenerate tree as per normal
        } else {
            if (print_ >= 2) {
                outfile->Printf("    Lowest level info: %d, %d\n", num_lowest_level_shells, num_lowest_level_boxes); 
                outfile->Printf("    New scaling factor: %f, %d -> %f\n", nshp_per_box, M_target_, scaling_factor);
                outfile->Printf("    New f scaling factor: %f, %f -> %f\n", f_, scaling_factor, scaling_factor * f_);
                outfile->Printf("    New box volume (bohr): %f, %f -> %f, %f\n", length_tmp, scaling_factor, length_new, length_new * length_new * length_new);
                outfile->Printf("    New box volume (ang): %f, %f -> %f, %f\n", length_tmp * bohr2ang, scaling_factor, length_new * bohr2ang, std::pow(length_new * bohr2ang, 3));
                outfile->Printf("    New box origin: (%f, %f, %f) \n\n", origin_new[0] * bohr2ang, origin_new[1] * bohr2ang, origin_new[2] * bohr2ang); 
            }
        } 
    }

    return std::tie(origin_new, length_new, converged, changed_level);
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

    size_t nsh = primary_->nshell(); 

    primary_shellpair_list_.resize(nsh);

    for (int P = 0; P != nsh; ++P) { 
        primary_shellpair_list_[P].resize(nsh);
    }

    for (int i = 0; i < sorted_leaf_boxes_.size(); i++) {
        std::shared_ptr<CFMMBox> curr = sorted_leaf_boxes_[i];
        auto& shellpairs = curr->get_primary_shell_pairs();
        auto& nf_boxes = curr->near_field_boxes();

        for (auto& sp : shellpairs) {
            auto [P, Q] = sp->get_shell_pair_index();

            primary_shellpair_list_[P][Q] = { sp, curr };
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

void CFMMTree::calculate_multipoles(const std::vector<SharedMatrix>& D) {
    timer_on("CFMMTree: Box Multipoles");

    // Compute mpoles for leaf nodes
#pragma omp parallel
    {
#pragma omp for
        for (int bi = 0; bi < sorted_leaf_boxes_.size(); bi++) {
            sorted_leaf_boxes_[bi]->compute_multipoles(D, std::nullopt);
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

void CFMMTree::J_build_kernel(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                        const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
			bool do_incfock_iter, const std::vector<double>& Jmet_max) {
   
    std::vector<SharedMatrix> nf_J, ff_J;
 
    // Zero the J matrix
    auto zero_mat = J[0]->clone();
    zero_mat->zero(); 
    for (int ind = 0; ind < D.size(); ind++) {
        if (!do_incfock_iter) 
        {
          J[ind]->zero();
        }

        nf_J.push_back(std::make_shared<Matrix>(zero_mat->clone()));
        ff_J.push_back(std::make_shared<Matrix>(zero_mat->clone()));
    }

    // Update the densities
    //if (density_screening_) {
    //    for (int thread = 0; thread < nthread_; thread++) {
    //        ints[thread]->update_density(D);
    //    }
    //}

    // Compute multipoles and far field
    calculate_multipoles(D);
    compute_far_field();

    // Compute near field J and far field J
    build_nf_J(ints, D, nf_J, Jmet_max);
    
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

    build_ff_J(ff_J);
    
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

    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->add(nf_J[ind]);
        J[ind]->add(ff_J[ind]);
    }
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

void CFMMTree::print_out() {
    // recursive print-out per box
    if (print_ >= 2) {
        outfile->Printf("  ==> CFMM: Box Information <==\n\n");
        tree_[0]->print_out();  
        outfile->Printf("\n");
    }

    // overall tree summary 
    generate_per_level_info();

    outfile->Printf("  ==> CFMM: Tree Information <==\n\n");
    int ilevel = 0;
    while (level_to_box_count_[ilevel] > 0) {
        outfile->Printf("    Tree Level: %d, Num. Boxes: %d -> Avg. Shell Pairs/Box: %f \n", ilevel, level_to_box_count_[ilevel], static_cast<double>(level_to_shell_count_[ilevel]) / static_cast<double>(level_to_box_count_[ilevel]) );
        ++ilevel;
    }
    outfile->Printf("\n");
} 

} // end namespace psi
