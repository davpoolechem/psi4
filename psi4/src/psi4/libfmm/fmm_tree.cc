#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
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

namespace psi {

double sign(double x) {
    if (std::abs(x) < 1.0e-8) {
        return 0.0;
    } else if (x < 0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

ShellPair::ShellPair(std::shared_ptr<BasisSet>& basisset, std::pair<int, int> pair_index, 
                     std::shared_ptr<HarmonicCoefficients>& mpole_coefs, double cfmm_extent_tol) {
    basisset_ = basisset;
    pair_index_ = pair_index;

    const GaussianShell& Pshell = basisset_->shell(pair_index.first);
    const GaussianShell& Qshell = basisset_->shell(pair_index.second);

    Vector3 pcenter = Pshell.center();
    Vector3 qcenter = Qshell.center();

    center_ = Vector3(0.0, 0.0, 0.0);
    exp_ = INFINITY;

    int nprim_p = Pshell.nprimitive();
    int nprim_q = Qshell.nprimitive();
    for (int pp = 0; pp < nprim_p; pp++) {
        double pcoef = Pshell.coef(pp);
        double pexp = Pshell.exp(pp);
        for (int qp = 0; qp < nprim_q; qp++) {
            double qcoef = Qshell.coef(qp);
            double qexp = Qshell.exp(qp);

            const double pq_exp = std::abs(pexp + qexp);
            Vector3 pq_center = (pexp * pcenter + qexp * qcenter) / pq_exp;

            center_ += pq_center;
            exp_ = std::min(exp_, pq_exp);
        }
    }
    center_ /= (nprim_p * nprim_q);
    extent_ = std::sqrt(-2.0 * std::log(cfmm_extent_tol) / exp_);

    mpole_coefs_ = mpole_coefs;
}

void ShellPair::calculate_mpoles(Vector3 box_center, std::shared_ptr<OneBodyAOInt>& s_ints, 
                                 std::shared_ptr<OneBodyAOInt>& mpole_ints, int lmax) {
    
    // number of total multipoles to compute, -1 since the overlap is not computed
    int nmpoles = (lmax + 1) * (lmax + 2) * (lmax + 3) / 6 - 1;

    int P = pair_index_.first;
    int Q = pair_index_.second;

    // Calculate the overlap integrals (Order 0 multipole integrals)
    s_ints->compute_shell(P, Q);
    const double* sbuffer = s_ints->buffers()[0];

    // Calculate the multipole integrals
    mpole_ints->compute_shell(P, Q);
    const double* mbuffer = mpole_ints->buffers()[0];

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

            std::shared_ptr<RealSolidHarmonics> pq_mpoles = std::make_shared<RealSolidHarmonics>(lmax, box_center, Regular);

            pq_mpoles->add(0, 0, sbuffer[dp * num_q + dq]);

            int running_index = 0;
            for (int l = 1; l <= lmax; l++) {
                int l_ncart = ncart(l);
                for (int m = -l; m <= l; m++) {
                    int mu = m_addr(m);
                    std::unordered_map<int, double>& mpole_terms = mpole_coefs_->get_terms(l, mu);

                    int powdex = 0;
                    for (int ii = 0; ii <= l; ii++) {
                        int a = l - ii;
                        for (int jj = 0; jj <= ii; jj++) {
                            int b = ii - jj;
                            int c = jj;
                            int ind = a * l_ncart * l_ncart + b * l_ncart + c;

                            if (mpole_terms.count(ind)) {
                                double coef = mpole_terms[ind];
                                int abcindex = powdex + running_index;
                                pq_mpoles->add(l, mu, pow(-1.0, (double) l+1) * coef * mbuffer[abcindex * num_p * num_q + dp * num_q + dq]);
                            }
                            powdex += 1;
                        } // end jj
                    } // end ii
                } // end m loop
                running_index += l_ncart;
            } // end l
            mpoles_.push_back(pq_mpoles);
        } // end q
    } // end p

}

CFMMBox::CFMMBox(std::shared_ptr<CFMMBox> parent, std::vector<std::shared_ptr<ShellPair>> shell_pairs, 
              Vector3 origin, double length, int level, int lmax, int ws) {
                  
    parent_ = parent;
    shell_pairs_ = shell_pairs;
    origin_ = origin;
    center_ = origin_ + 0.5 * Vector3(length, length, length);
    length_ = length;
    level_ = level;
    lmax_ = lmax;
    ws_ = ws;
    ws_max_ = ws;

    nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

}

void CFMMBox::make_children() {

    int nchild = (level_ > 0) ? 16 : 8;
    std::vector<std::vector<std::shared_ptr<ShellPair>>> child_shell_pair_buffer(nchild);

    // Max WS at the child's level
    int child_level_max_ws = std::max(2, (int) std::pow(2, level_+1));
    int diffuse_child_max_ws = child_level_max_ws;

    // Fill order (ws,z,y,x) (0)000 (0)001 (0)010 (0)011 (0)100 (0)101 (0)110 (0)111
    // (1)000 (1)001 (1)010 (1)011 (1)100 (1)101 (1)110 (1)111
    for (std::shared_ptr<ShellPair> shell_pair : shell_pairs_) {
        Vector3 sp_center = shell_pair->get_center();
        double x = sp_center[0];
        double y = sp_center[1];
        double z = sp_center[2];
        double extent = shell_pair->get_extent();
        int ws = std::max(2, 2 * (int)std::ceil(extent / length_));

        int xbit = (x < center_[0]) ? 0 : 1;
        int ybit = (y < center_[1]) ? 0 : 1;
        int zbit = (z < center_[2]) ? 0 : 1;
        int rbit = (level_ == 0 || ws < 2 * ws_) ? 0 : 1;

        int boxind = 8 * rbit + 4 * zbit + 2 * ybit + 1 * xbit;
        child_shell_pair_buffer[boxind].push_back(shell_pair);

        int child_ws = std::max(2, (int)std::ceil(2.0 * extent / length_));
        if (child_ws > diffuse_child_max_ws) diffuse_child_max_ws = child_ws;
    }

    // Make the children
    for (int boxind = 0; boxind < nchild; boxind++) {
        int xbit = boxind % 2;
        int ybit = (boxind / 2) % 2;
        int zbit = (boxind / 4) % 2;
        int rbit = (boxind / 8) % 2;
        Vector3 new_origin = origin_ + Vector3(xbit * 0.5 * length_, ybit * 0.5 * length_, zbit * 0.5 * length_);
        int child_ws = 2 * ws_ - 2 + 2 * rbit;
        children_.push_back(std::make_shared<CFMMBox>(this->get(), child_shell_pair_buffer[boxind], new_origin, 
                                                          0.5 * length_, level_ + 1, lmax_, child_ws));
        if (child_ws == child_level_max_ws) children_[boxind]->set_ws_max(diffuse_child_max_ws);
    }
}

void CFMMBox::set_regions() {

    // Creates a temporary parent shared pointer
    std::shared_ptr<CFMMBox> parent = parent_.lock();
    // Maximum possible WS for a box at this level
    int max_ws = std::max(2, (int)std::pow(2, level_));

    // Parent is not a nullpointer
    if (parent) {
        // Near field or local far fields are from children of parents
        // and children of parent's near field
        for (std::shared_ptr<CFMMBox> parent_nf : parent->near_field_) {
            for (std::shared_ptr<CFMMBox> child : parent_nf->children_) {
                if (child->nshell_pair() == 0) continue;
                // WS Max formulation takes the most diffuse branch into account
                int ref_ws = (ws_max_ + child->ws_max_) / 2;

                Vector3 Rab = child->center_ - center_;
                double dx = sign(Rab[0]);
                double dy = sign(Rab[1]);
                double dz = sign(Rab[2]);

                Rab = Rab - length_ * Vector3(dx, dy, dz);
                double rab = std::sqrt(Rab.dot(Rab));
                
                if (rab <= length_ * ref_ws) {
                    near_field_.push_back(child);
                } else {
                    local_far_field_.push_back(child);
                }
            }
        }
    } else {
        near_field_.push_back(this->get());
    }
}

void CFMMBox::compute_far_field_contribution(std::shared_ptr<CFMMBox> lff_box) {
    for (int N = 0; N < Vff_.size(); N++) {
        std::shared_ptr<RealSolidHarmonics> far_field = lff_box->mpoles_[N]->far_field_vector(center_);
        Vff_[N]->add(far_field);
    }
}

void CFMMBox::add_parent_far_field_contribution() {
    // Temporary parent shared pointer object
    std::shared_ptr<CFMMBox> parent = parent_.lock();

    if (parent) {
        // Add the parent's far field contribution
        for (int N = 0; N < Vff_.size(); N++) {
            auto parent_cont = parent->Vff_[N]->translate(center_);
            Vff_[N]->add(parent_cont);
        }
    }
}

void CFMMBox::compute_multipoles(std::shared_ptr<BasisSet>& basisset, const std::vector<SharedMatrix>& D) {

    if (mpoles_.size() == 0) {
        mpoles_.resize(D.size());
        Vff_.resize(D.size());
    }

    // Create multipoles and far field vectors for each density matrix
    for (int N = 0; N < D.size(); N++) {
        mpoles_[N] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
        Vff_[N] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);
    }

    // Contract the multipoles with the density matrix to get box multipoles
    for (const auto& sp : shell_pairs_) {

        std::vector<std::shared_ptr<RealSolidHarmonics>>& sp_mpoles = sp->get_mpoles();
        
        std::pair<int, int> PQ = sp->get_shell_pair_index();
        int P = PQ.first;
        int Q = PQ.second;

        double prefactor = (P == Q) ? 1.0 : 2.0;

        const GaussianShell& Pshell = basisset->shell(P);
        const GaussianShell& Qshell = basisset->shell(Q);

        int p_start = Pshell.start();
        int num_p = Pshell.nfunction();

        int q_start = Qshell.start();
        int num_q = Qshell.nfunction();

        for (int N = 0; N < D.size(); N++) {
            for (int p = p_start; p < p_start + num_p; p++) {
                int dp = p - p_start;
                for (int q = q_start; q < q_start + num_q; q++) {
                    int dq = q - q_start;
                    std::shared_ptr<RealSolidHarmonics> basis_mpole = sp_mpoles[dp * num_q + dq]->copy();
                    
                    basis_mpole->scale(prefactor * D[N]->get(p, q));
                    mpoles_[N]->add(basis_mpole);
                } // end q
            } // end p
        } // end N
    }

}

void CFMMBox::compute_mpoles_from_children() {

    int nmat = 0;
    for (std::shared_ptr<CFMMBox> child : children_) {
        nmat = std::max(nmat, (int)child->mpoles_.size());
    }

    if (mpoles_.size() == 0) {
        mpoles_.resize(nmat);
        Vff_.resize(nmat);
    }

    // Create multipoles and far field vectors for each density matrix
    for (int N = 0; N < nmat; N++) {
        mpoles_[N] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
        Vff_[N] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);
    }

    for (std::shared_ptr<CFMMBox> child : children_) {
        if (child->nshell_pair() == 0) continue;
        for (int N = 0; N < nmat; N++) {
            std::shared_ptr<RealSolidHarmonics> child_mpoles = child->mpoles_[N]->translate(center_);
            mpoles_[N]->add(child_mpoles);
        }
    }
}

CFMMTree::CFMMTree(std::shared_ptr<BasisSet> basis, Options& options) 
                    : basisset_(basis), options_(options) {

    molecule_ = basisset_->molecule();
   
   nthread_ = 1;
#ifdef _OPENMP
    nthread_ = Process::environment.get_n_threads();
#endif

    print_ = options_.get_int("PRINT");
    bench_ = options_.get_int("BENCH");

    density_screening_ = (options_.get_str("SCREENING") == "DENSITY");
    ints_tolerance_ = options_.get_double("INTS_TOLERANCE");

    lmax_ = options_.get_int("CFMM_ORDER");

    mpole_coefs_ = std::make_shared<HarmonicCoefficients>(lmax_, Regular);
    double cfmm_extent_tol = options.get_double("CFMM_EXTENT_TOLERANCE");

    auto factory = std::make_shared<IntegralFactory>(basisset_);
    auto shellpair_int = std::shared_ptr<TwoBodyAOInt>(factory->eri());

    auto& ints_shell_pairs = shellpair_int->shell_pairs();
    size_t nshell_pairs = ints_shell_pairs.size();
    shell_pairs_.resize(nshell_pairs);

#pragma omp parallel for
    for (size_t pair_index = 0; pair_index < nshell_pairs; pair_index++) {
        const auto& pair = ints_shell_pairs[pair_index];
        shell_pairs_[pair_index] = std::make_shared<ShellPair>(basisset_, pair, mpole_coefs_, cfmm_extent_tol);
    }

    // ==> time to define the number of lowest-level boxes in the CFMM tree! <== // 
    int grain = options_.get_int("CFMM_GRAIN");
    int M_target = options_.get_int("CFMM_TARGET_NSHP");
    
    // CFMM_GRAIN < -1 is invalid 
    if (grain < -1) { 
        std::string error_message = "CFMM grain set to below -1! If you meant to use adaptive CFMM, please set CFMM_GRAIN to exactly -1 or 0.";
        
        throw PSIEXCEPTION(error_message);
 
    // CFMM_GRAIN = -1 or 0 enables adaptive CFMM 
    } else if (grain == -1 || grain == 0) { 
        // Eq. 1 of White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
        N_target_ = ceil(static_cast<double>(nshell_pairs) / (static_cast<double>(M_target) * g_)); 
        outfile->Printf("N_target: %d, %d, %f -> %d \n", nshell_pairs, M_target, g_, N_target_);
   
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
    } else if (nlevels_ >= 6) {
        std::string error_message = "Too many tree levels ("; 
        error_message += std::to_string(nlevels_);
        error_message += ")! You memory hog.";

        throw PSIEXCEPTION(error_message);
    }

    int num_boxes = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;
    tree_.resize(num_boxes);

    timer_on("CFMMTree: Setup");

    outfile->Printf("CFMMTree: Setup \n");
    outfile->Printf("  Make Root Node... ");
    make_root_node();
    outfile->Printf("  Done! \n");

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
    
    outfile->Printf("  Calculate Shellpair Multipoles... ");
    calculate_shellpair_multipoles();
    outfile->Printf("  Done! \n");
    outfile->Printf("Completed CFMMTree: Setup! \n");

    timer_off("CFMMTree: Setup");

    if (print_ >= 2) print_out();
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
    double min_dim = molecule_->x(0);
    double max_dim = molecule_->x(0);

    for (int atom = 0; atom < molecule_->natom(); atom++) {
        double x = molecule_->x(atom);
        double y = molecule_->y(atom);
        double z = molecule_->z(atom);
        min_dim = std::min(x, min_dim);
        min_dim = std::min(y, min_dim);
        min_dim = std::min(z, min_dim);
        max_dim = std::max(x, max_dim);
        max_dim = std::max(y, max_dim);
        max_dim = std::max(z, max_dim);
    }

    max_dim += 0.1; // Add a small buffer to the box

    Vector3 origin = Vector3(min_dim, min_dim, min_dim);
    double length = (max_dim - min_dim);

    // Scale root CFMM box for adaptive CFMM
    // Logic follows Eq. 3 of White 1996, adapted for CFMM 
    outfile->Printf("Original box volume: %f \n", length*length*length);
    outfile->Printf("Original box origin: %f \n\n", origin[0]);
    
    int num_lowest_level_boxes = 8 * std::pow(2,  (1 + dimensionality_) * (nlevels_ - 2)); 
    // int num_lowest_level_boxes_per_branch = std::pow(2,  (dimensionality_) * (nlevels_ - 1)); 

    double f = static_cast<double>(N_target_) / static_cast<double>(num_lowest_level_boxes); 
    outfile->Printf("f scaling factor: %f\n", f);
    if (f > 1.0) throw PSIEXCEPTION("Bad f scaling factor value!");

    //f = std::pow(f, 1.0 / dimensionality_); // account for CFMM tree structure
 
    double length_tmp = length;
    //length = length_tmp / std::pow(f, 1.0 / (1.0 + dimensionality_));
    length = length_tmp / std::pow(f, 1.0 / dimensionality_);
 
    min_dim -= (length - length_tmp) / 2.0;
    Vector3 origin_new = Vector3(min_dim, min_dim, min_dim);

    outfile->Printf("New box volume: %f, %f -> %f, %f\n\n", length_tmp, f, length, length*length*length);
    outfile->Printf("New box origin: %f \n\n", origin_new[0]); 

    tree_[0] = std::make_shared<CFMMBox>(nullptr, shell_pairs_, origin, length, 0, lmax_, 2);
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

    /*
    size_t nsh = basisset_->nshell(); 

    shellpair_list_.resize(nsh);

    for (int P = 0; P != nsh; ++P) { 
        shellpair_list_[P].resize(nsh);
    }
    */

    nshp_ = 0;
    for (int i = 0; i < sorted_leaf_boxes_.size(); i++) {
        std::shared_ptr<CFMMBox> curr = sorted_leaf_boxes_[i];
        auto& shellpairs = curr->get_shell_pairs();
        auto& nf_boxes = curr->near_field_boxes();

        for (auto& sp : shellpairs) {
            auto PQ = sp->get_shell_pair_index();
            int P = PQ.first;
            int Q = PQ.second;

            //shellpair_list_[P][Q] = { sp, curr, {} };
            shellpair_list_.push_back({ sp, curr, {} });

            //auto shellpair_to_nf_boxes = std::get<2>(shellpair_list_[P][Q]);
            auto shellpair_to_nf_boxes = std::get<2>(shellpair_list_[nshp_]);
            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                shellpair_to_nf_boxes.push_back(neighbor);
            }
            nshp_ += 1;
        }
    }
   
    /* 
    for (int P = 0; P != shellpair_list_.size(); ++P) { 
        auto new_end = std::remove_if(
          shellpair_list_[P].begin(), shellpair_list_[P].end(),
          [](std::tuple<std::shared_ptr<ShellPair>, std::shared_ptr<CFMMBox>, 
                        std::vector<std::shared_ptr<CFMMBox>>
                       > tuple
            ) { return std::get<0>(tuple) == nullptr; }
        );
        shellpair_list_[P].erase(new_end, shellpair_list_[P].end());
    }
    */
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

#pragma omp parallel for schedule(dynamic)
    for (int ishp = 0; ishp < shellpair_list_.size(); ishp++) {
        std::pair<int, int> PQ = std::get<0>(shellpair_list_[ishp])->get_shell_pair_index();
        int P = PQ.first;
        int Q = PQ.second;

        std::shared_ptr<ShellPair> shellpair = std::get<0>(shellpair_list_[ishp]);
        std::shared_ptr<CFMMBox> box = std::get<1>(shellpair_list_[ishp]);

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        mpints[thread]->set_origin(box->center());
        shellpair->calculate_mpoles(box->center(), sints[thread], mpints[thread], lmax_);
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
            std::pair<int, int> RS = RSshells[RSind]->get_shell_pair_index();
            int R = RS.first;
            int S = RS.second;
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
    size_t computed_shells = 0L;

#pragma omp parallel for schedule(dynamic) reduction(+ : computed_shells)
    for (int ishp = 0; ishp < shellpair_list_.size(); ishp++) {
        std::pair<int, int> PQ = std::get<0>(shellpair_list_[ishp])->get_shell_pair_index();
        int P = PQ.first;
        int Q = PQ.second;
            
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
        
        for (const auto& nf_box : std::get<2>(shellpair_list_[ishp])) {
            auto& RSshells = nf_box->get_shell_pairs();

            bool touched = false;
            for (const auto& RSsh : RSshells) {
                auto RS = RSsh->get_shell_pair_index();
                int R = RS.first;
                int S = RS.second;
                
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
                    std::pair<int, int> RS = RSsh->get_shell_pair_index();
                    int R = RS.first;
                    int S = RS.second;

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
   } // end tasks
     
   if (bench_) {
        auto mode = std::ostream::app;
        auto printer = PsiOutStream("bench.dat", mode);
        size_t ntri = nshell * (nshell + 1L) / 2L;
        size_t possible_shells = ntri * (ntri + 1L) / 2L;
        double computed_fraction = ((double) computed_shells) / possible_shells;
        printer.Printf("CFMM Near Field: Computed %20zu Shell Quartets out of %20zu, (%11.3E ratio)\n", 
                    computed_shells, possible_shells, computed_fraction);
    }

    timer_off("CFMMTree: Near Field J");
}

void CFMMTree::build_ff_J(std::vector<SharedMatrix>& J) {

    timer_on("CFMMTree: Far Field J");

#pragma omp parallel for schedule(dynamic)
    for (int ishp = 0; ishp < shellpair_list_.size(); ishp++) {
        std::pair<int, int> PQ = std::get<0>(shellpair_list_[ishp])->get_shell_pair_index();
        int P = PQ.first;
        int Q = PQ.second;

        std::shared_ptr<ShellPair> shellpair = std::get<0>(shellpair_list_[ishp]);
        //std::shared_ptr<CFMMBox> box = std::get<1>(shp);

        const auto& Vff = std::get<1>(shellpair_list_[ishp])->far_field_vector();
            
        const auto& shellpair_mpoles = shellpair->get_mpoles();
        assert(!(shellpair_mpoles.empty()));
 
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
                    Jp[p][q] += prefactor * Vff[N]->dot(shellpair_mpoles.at(dp * num_q + dq));
                } // end N
            } // end q
        } // end p
    } // end tasks

    timer_off("CFMMTree: Far Field J");
}

void CFMMTree::build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                        const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {

    timer_on("CFMMTree: J");

    // Zero the J matrix
    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->zero();
    }

    // Update the densities
    for (int thread = 0; thread < nthread_; thread++) {
        ints[thread]->update_density(D);
    }

    // Compute multipoles and far field
    calculate_multipoles(D);
    compute_far_field();

    // Compute near field J and far field J
    build_nf_J(ints, D, J);
    build_ff_J(J);

    // Hermitivitize J matrix afterwards
    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->hermitivitize();
    }

    timer_off("CFMMTree: J");
}

void CFMMTree::print_out() {
    std::vector<int> level_to_box_count(6, 0);
    std::vector<int> level_to_shell_count(6, 0);
    for (int bi = 0; bi < tree_.size(); bi++) {
        std::shared_ptr<CFMMBox> box = tree_[bi];
        auto sp = box->get_shell_pairs();
        int nshells = sp.size();
        int level = box->get_level();
        int ws = box->get_ws();
        if (nshells > 0) {
            outfile->Printf("  BOX INDEX: %d, LEVEL: %d, WS: %d, NSHP: %d\n", bi, level, ws, nshells);
            ++level_to_box_count[level];
            level_to_shell_count[level] += nshells;
        }
    }
    
    outfile->Printf("CFMM TREE LEVEL INFO:\n");
    int ilevel = 0;
    while (level_to_box_count[ilevel] > 0) {
        outfile->Printf("  LEVEL: %d, BOXES: %d NSHP/BOX: %f \n", ilevel, level_to_box_count[ilevel], static_cast<double>(level_to_shell_count[ilevel]) / static_cast<double>(level_to_box_count[ilevel]) );
        ++ilevel;
    }
}

} // end namespace psi
