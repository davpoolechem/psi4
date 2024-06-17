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

#include <optional>
#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {

DFCFMMTree::DFCFMMTree(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options) 
                    : CFMMTree(primary, options), auxiliary_(auxiliary) {

    if (primary && auxiliary) {
        contraction_type_ = ContractionType::DF_AUX_PRI;
    } else if (auxiliary) {
        contraction_type_ = ContractionType::METRIC;
    } else {
        throw PSIEXCEPTION("No auxiliary basis sets inputted into CFMMTree constructor!");
    }

    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();

    std::shared_ptr<IntegralFactory> factory;
    if (contraction_type_ == ContractionType::DF_AUX_PRI) {
        factory = std::make_shared<IntegralFactory>(auxiliary_, zero, primary_, primary_);
    } else if (contraction_type_ == ContractionType::METRIC) {
        factory = std::make_shared<IntegralFactory>(auxiliary_, zero, auxiliary_, zero);
    }

    std::shared_ptr<TwoBodyAOInt> shellpair_int = std::shared_ptr<TwoBodyAOInt>(factory->eri());

    const auto& ints_bra_shell_pairs = shellpair_int->shell_pairs_bra();
    size_t bra_nshell_pairs = ints_bra_shell_pairs.size();
    auxiliary_shell_pairs_.resize(bra_nshell_pairs);

    double cfmm_extent_tol = options.get_double("CFMM_EXTENT_TOLERANCE");

#pragma omp parallel for
    for (size_t pair_index = 0; pair_index < bra_nshell_pairs; pair_index++) {
        const auto& pair = ints_bra_shell_pairs[pair_index];
        auxiliary_shell_pairs_[pair_index] = std::make_shared<CFMMShellPair>(auxiliary_, zero, pair, mpole_coefs_, cfmm_extent_tol);
    }

    // TODO: is this okay?
    if (contraction_type_ == ContractionType::DF_AUX_PRI) {
        const auto& ints_ket_shell_pairs = shellpair_int->shell_pairs_ket();
        size_t ket_nshell_pairs = ints_ket_shell_pairs.size();
        primary_shell_pairs_.resize(ket_nshell_pairs);

#pragma omp parallel for 
        for (size_t pair_index = 0; pair_index < ket_nshell_pairs; pair_index++) {
            const auto& pair = ints_ket_shell_pairs[pair_index];
            primary_shell_pairs_[pair_index] = std::make_shared<CFMMShellPair>(primary_, primary_, pair, mpole_coefs_, cfmm_extent_tol);
        }
    }

    make_tree(primary_shell_pairs_.size()); // TODO: maybe???    

    if (contraction_type_ == ContractionType::DF_AUX_PRI) calculate_shellpair_multipoles(true);
    if (contraction_type_ == ContractionType::METRIC || contraction_type_ == ContractionType::DF_AUX_PRI) calculate_shellpair_multipoles(false);

    if (print_ >= 2) print_out();
}

//DFCFMMTree::DFCFMMTree(std::shared_ptr<BasisSet> primary, Options& options) {
//    throw PSIEXCEPTION("DFCFMMTree requires an auxiliary basis!");
//}

void DFCFMMTree::calculate_multipoles(const std::vector<SharedMatrix>& D) {
    timer_on("CFMMTree: Box Multipoles");

    // Compute mpoles for leaf nodes
#pragma omp parallel
    {
#pragma omp for
        for (int bi = 0; bi < sorted_leaf_boxes_.size(); bi++) {
            sorted_leaf_boxes_[bi]->compute_multipoles(D, std::make_optional<ContractionType>(contraction_type_));
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

void DFCFMMTree::make_root_node() {
    // base algorithm for constructing CFMM tree node
    auto [origin, length] = make_root_node_kernel();
    
    // create top-level box
    //outfile->Printf("BUILDING TREE_[0]\n");
    tree_[0] = std::make_shared<CFMMBox>(nullptr, primary_shell_pairs_, auxiliary_shell_pairs_, origin, length, 0, lmax_, 2); 
    //outfile->Printf("  Tree length #0a: %f, %f\n", tree_[0]->length());
}

std::tuple<bool, bool> DFCFMMTree::regenerate_root_node() {
    // base algorithm for resizing CFMM tree node
    //Vector3 origin;
    //double length;
    //bool converged, changed_level;

    //std::tie(origin, length, converged, changed_level) = regenerate_root_node_kernel();
    
    auto [origin, length, converged, changed_level] = regenerate_root_node_kernel();

    // actually regenerate tree
    tree_.clear();

    num_boxes_ = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;
    tree_.resize(num_boxes_);

    if (!changed_level) tree_[0] = std::make_shared<CFMMBox>(nullptr, primary_shell_pairs_, auxiliary_shell_pairs_, origin, length, 0, lmax_, 2);

    // we are done
    return std::tie(converged, changed_level);
}

void DFCFMMTree::calculate_shellpair_multipoles(bool is_primary) {
    timer_on("DFCFMMTree: Shell-Pair Multipole Ints");

    // Build the multipole integrals of the bra basis
    std::vector<std::shared_ptr<OneBodyAOInt>> sints;
    std::vector<std::shared_ptr<OneBodyAOInt>> mpints;

    std::shared_ptr<IntegralFactory> int_factory;

    if (is_primary) {
        int_factory = std::make_shared<IntegralFactory>(primary_);
    } else {
        auto zero = BasisSet::zero_ao_basis_set();
        int_factory = std::make_shared<IntegralFactory>(auxiliary_, zero, auxiliary_, zero);
    }

    for (int thread = 0; thread < nthread_; thread++) {
        sints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_overlap()));
        mpints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_multipoles(lmax_)));
    }

    auto shellpair_list = (is_primary) ? primary_shellpair_list_ : auxiliary_shellpair_list_;

#pragma omp parallel for collapse(2) schedule(guided)
    for (int Ptask = 0; Ptask < shellpair_list.size(); Ptask++) {
        for (int Qtask = 0; Qtask < shellpair_list.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(shellpair_list[Ptask][Qtask]);
            if (shellpair == nullptr) continue;

            std::shared_ptr<CFMMBox> box = std::get<1>(shellpair_list[Ptask][Qtask]);

            auto [P, Q] = shellpair->get_shell_pair_index();

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            mpints[thread]->set_origin(box->center());
            shellpair->calculate_mpoles(box->center(), sints[thread], mpints[thread], lmax_);
        }
    }

    timer_off("DFCFMMTree: Shell-Pair Multipole Ints");
}


void DFCFMMTree::set_contraction(ContractionType contraction_type) {
    if (contraction_type_ != ContractionType::DF_PRI_AUX && contraction_type_ != ContractionType::DF_AUX_PRI) {
        throw PSIEXCEPTION("Cannot reset contraction type on non 3-index DF CFMM Trees");
    }
    if (contraction_type == ContractionType::DIRECT || contraction_type == ContractionType::METRIC) {
        throw PSIEXCEPTION("Cannot reset DF contraction type to Non-DF contraction");
    }
    contraction_type_ = contraction_type;
}

// TODO: THIS
void DFCFMMTree::setup_shellpair_info() {
    //outfile->Printf("  Start DFCFMMTree::setup_shellpair_info()\n");
    size_t nsh = primary_->nshell(); 
    primary_shellpair_list_.resize(nsh);
    primary_shellpair_to_nf_boxes_.resize(nsh);
    for (int P = 0; P != nsh; ++P) { 
        primary_shellpair_list_[P].resize(nsh);
        primary_shellpair_to_nf_boxes_[P].resize(nsh);
    }
    
    size_t naux = auxiliary_->nshell(); 
    auxiliary_shellpair_list_.resize(naux);
    auxiliary_shellpair_to_nf_boxes_.resize(naux);
    for (int P = 0; P != naux; ++P) { 
        auxiliary_shellpair_list_[P].resize(naux);
        auxiliary_shellpair_to_nf_boxes_[P].resize(naux);
    }

    int primary_task_index = 0;
    int auxiliary_task_index = 0;
    for (int i = 0; i < sorted_leaf_boxes_.size(); i++) {
        std::shared_ptr<CFMMBox> curr = sorted_leaf_boxes_[i];
        auto& primary_shellpairs = curr->get_primary_shell_pairs();
        auto& auxiliary_shellpairs = curr->get_auxiliary_shell_pairs();
        auto& nf_boxes = curr->near_field_boxes();
        //outfile->Printf("    Box %i params: %i, %i, %i\n", i, 
          //primary_shellpairs.size(), auxiliary_shellpairs.size(), nf_boxes.size()
        //);

        for (auto& primary_sp : primary_shellpairs) {
            auto [P, Q] = primary_sp->get_shell_pair_index();

            primary_shellpair_list_[P][Q] = { primary_sp, curr };

            primary_shellpair_to_nf_boxes_[P][Q] = std::vector<std::shared_ptr<CFMMBox> >();
            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                primary_shellpair_to_nf_boxes_[P][Q].push_back(neighbor);
            }
        }

        for (auto& auxiliary_sp : auxiliary_shellpairs) {
            auto [R, S] = auxiliary_sp->get_shell_pair_index();

            auxiliary_shellpair_list_[R][S] = { auxiliary_sp, curr };

            auxiliary_shellpair_to_nf_boxes_[R][S] = std::vector<std::shared_ptr<CFMMBox> >();
            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                auxiliary_shellpair_to_nf_boxes_[R][S].push_back(neighbor);
            }
        }
    }
    //outfile->Printf("  End DFCFMMTree::setup_shellpair_info()\n");
}

/*
void DFCFMMTree::setup_shellpair_info() {
    //outfile->Printf("  Start DFCFMMTree::setup_shellpair_info()\n");
    size_t nsh = primary_->nshell(); 
    primary_shellpair_list_.resize(nsh);
    for (int P = 0; P != nsh; ++P) { 
        primary_shellpair_list_[P].resize(nsh);
    }
    
    size_t naux = auxiliary_->nshell(); 
    auxiliary_shellpair_list_.resize(naux);
    for (int P = 0; P != naux; ++P) { 
        auxiliary_shellpair_list_[P].resize(naux);
    }

    int primary_task_index = 0;
    int auxiliary_task_index = 0;
    for (int i = 0; i < sorted_leaf_boxes_.size(); i++) {
        std::shared_ptr<CFMMBox> curr = sorted_leaf_boxes_[i];
        auto& primary_shellpairs = curr->get_primary_shell_pairs();
        auto& auxiliary_shellpairs = curr->get_auxiliary_shell_pairs();
        auto& nf_boxes = curr->near_field_boxes();
        //outfile->Printf("    Box %i params: %i, %i, %i\n", i, 
          //primary_shellpairs.size(), auxiliary_shellpairs.size(), nf_boxes.size()
        //);

        for (auto& primary_sp : primary_shellpairs) {
            auto [P, Q] = primary_sp->get_shell_pair_index();

            primary_shellpair_list_[P][Q] = { primary_sp, curr };

            primary_shellpair_to_nf_boxes_.push_back({});
            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                primary_shellpair_to_nf_boxes_[primary_task_index].push_back(neighbor);
            }
            primary_task_index += 1;
        }

        for (auto& auxiliary_sp : auxiliary_shellpairs) {
            auto [R, S] = auxiliary_sp->get_shell_pair_index();

            auxiliary_shellpair_list_[R][S] = { auxiliary_sp, curr };

            auxiliary_shellpair_to_nf_boxes_.push_back({});
            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                auxiliary_shellpair_to_nf_boxes_[auxiliary_task_index].push_back(neighbor);
            }
            auxiliary_task_index += 1;
        }
    }
    //outfile->Printf("  End DFCFMMTree::setup_shellpair_info()\n");
}
*/

/*
void CFMMTree::setup_shellpair_info() {

    int primary_task_index = 0;
    int auxiliary_task_index = 0;
    for (int i = 0; i < sorted_leaf_boxes_.size(); i++) {
        std::shared_ptr<CFMMBox> curr = sorted_leaf_boxes_[i];
        auto& primary_shellpairs = curr->get_primary_shell_pairs();
        auto& auxiliary_shellpairs = curr->get_auxiliary_shell_pairs();
        auto& nf_boxes = curr->near_field_boxes();

        for (auto& primary_sp : primary_shellpairs) {
            auto PQ = primary_sp->get_shell_pair_index();
            int P = PQ.first;
            int Q = PQ.second;

            primary_shellpair_tasks_.emplace_back(P, Q);
            primary_shellpair_list_.push_back(primary_sp);
            primary_shellpair_to_box_.push_back(curr);
            primary_shellpair_to_nf_boxes_.push_back({});

            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                primary_shellpair_to_nf_boxes_[primary_task_index].push_back(neighbor);
            }
            primary_task_index += 1;
        }

        for (auto& auxiliary_sp : auxiliary_shellpairs) {
            auto RS = auxiliary_sp->get_shell_pair_index();
            int R = RS.first;
            int S = RS.second;

            auxiliary_shellpair_tasks_.emplace_back(R, S);
            auxiliary_shellpair_list_.push_back(auxiliary_sp);
            auxiliary_shellpair_to_box_.push_back(curr);
            auxiliary_shellpair_to_nf_boxes_.push_back({});

            for (int nfi = 0; nfi < nf_boxes.size(); nfi++) {
                std::shared_ptr<CFMMBox> neighbor = nf_boxes[nfi];
                if (neighbor->nshell_pair() == 0) continue;
                auxiliary_shellpair_to_nf_boxes_[auxiliary_task_index].push_back(neighbor);
            }
            auxiliary_task_index += 1;
        }
    }
}
*/

/*
void CFMMTree::calculate_shellpair_multipoles(bool is_primary) {

    timer_on("CFMMTree: Shell-Pair Multipole Ints");

    // Build the multipole integrals of the bra basis
    std::vector<std::shared_ptr<OneBodyAOInt>> sints;
    std::vector<std::shared_ptr<OneBodyAOInt>> mpints;

    std::shared_ptr<IntegralFactory> int_factory;

    if (is_primary) {
        int_factory = std::make_shared<IntegralFactory>(primary_);
    } else {
        auto zero = BasisSet::zero_ao_basis_set();
        int_factory = std::make_shared<IntegralFactory>(auxiliary_, zero, auxiliary_, zero);
    }

    for (int thread = 0; thread < nthread_; thread++) {
        sints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_overlap()));
        mpints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_multipoles(lmax_)));
    }

    std::vector<std::pair<int, int>>& shellpair_tasks = (is_primary) ? primary_shellpair_tasks_ : auxiliary_shellpair_tasks_;
    std::vector<std::shared_ptr<ShellPair>>& shellpair_list = (is_primary) ? primary_shellpair_list_ : auxiliary_shellpair_list_;
    std::vector<std::shared_ptr<CFMMBox>>& shellpair_to_box = (is_primary) ? primary_shellpair_to_box_ : auxiliary_shellpair_to_box_;

#pragma omp parallel for schedule(guided)
    for (int task = 0; task < shellpair_tasks.size(); task++) {
        std::shared_ptr<ShellPair> shellpair = shellpair_list[task];
        std::shared_ptr<CFMMBox> box = shellpair_to_box[task];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        mpints[thread]->set_origin(box->center());
        shellpair->calculate_mpoles(box->center(), sints[thread], mpints[thread], lmax_);
    }

    timer_off("CFMMTree: Shell-Pair Multipole Ints");

}
*/


void DFCFMMTree::build_nf_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                          const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
			  const std::vector<double>& Jmet_max) { 
    //outfile->Printf("  CALL build_nf_J\n");
    if (contraction_type_ == ContractionType::DF_AUX_PRI) build_nf_gamma_P(ints, D, J, Jmet_max);
    else if (contraction_type_ == ContractionType::DF_PRI_AUX) build_nf_df_J(ints, D, J, Jmet_max);
    //else if (contraction_type_ == ContractionType::METRIC) build_nf_metric(ints, D, J);
    //outfile->Printf("  END build_nf_J\n");
}

void DFCFMMTree::build_nf_gamma_P(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
		      const std::vector<double>& Jmet_max) {
    timer_on("DFCFMM: Near Field Gamma P");
    //outfile->Printf("   CALL build_nf_gammaP\n");

    // => Sizing <= //
    int pri_nshell = primary_->nshell();
    int aux_nshell = auxiliary_->nshell();
    int nmat = D.size();

    // check that Jmet_max is not empty is density screening is enabled
    if (density_screening_ && Jmet_max.empty()) {
        throw PSIEXCEPTION("CFMMTree::build_nf_gamma_P was called with density screening enabled, but Jmet is NULL. Check your arguments to build_J.");
    }

    // maximum values of Density matrix for primary shell pair block UV
    // TODO: Integrate this more smoothly into current density screening framework
    Matrix D_max(pri_nshell, pri_nshell);
    auto D_maxp = D_max.pointer();

    if (density_screening_) {
#pragma omp parallel for
        for(size_t UV = 0; UV < pri_nshell * pri_nshell; UV++) {
            size_t U = UV / pri_nshell;
            size_t V = UV % pri_nshell;

            int u_start = primary_->shell(U).start();
            int num_u = primary_->shell(U).nfunction();
            
	    int v_start = primary_->shell(V).start();
            int num_v = primary_->shell(V).nfunction();

            for(size_t i = 0; i < D.size(); i++) {
                auto Dp = D[i]->pointer();
                for(size_t u = u_start; u < u_start + num_u; u++) {
                    for(size_t v = v_start; v < v_start + num_v; v++) {
                        D_maxp[U][V] = std::max(D_maxp[U][V], std::abs(Dp[u][v]));
                    }
                }
            }
        }
    }

#pragma omp parallel for num_threads(nthread_) schedule(guided)
    for (int Ptask = 0; Ptask < auxiliary_shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < auxiliary_shellpair_list_.size(); Qtask++) { //TODO: maybe need to restrict this based on Ptask
            
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(auxiliary_shellpair_list_[Ptask][Qtask]);
            if (shellpair == nullptr) {
                continue;
            //} else {
            //    outfile->Printf("      Processing task (%i, %i)...\n", Ptask, Qtask);
            }
            
            std::shared_ptr<CFMMBox> box = std::get<1>(auxiliary_shellpair_list_[Ptask][Qtask]);
            
            auto PQ = shellpair->get_shell_pair_index();
            int P = PQ.first;
            int Q = PQ.second;

            //outfile->Printf("        Bra Shp indices: (%i, %i)...\n", P, Q);

            const GaussianShell& Pshell = auxiliary_->shell(P);

            int p_start = Pshell.start();
            int num_p = Pshell.nfunction();

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            
            //outfile->Printf("        Start NF Box Loop\n");
            //outfile->Printf("        NF Box Loop Size: %i\n", auxiliary_shellpair_to_nf_boxes_[P][Q].size());
            for (const auto& nf_box : auxiliary_shellpair_to_nf_boxes_[P][Q]) {
                auto& UVshells = nf_box->get_primary_shell_pairs();

                //outfile->Printf("          Start UVshells Loop\n");
                //outfile->Printf("          UVshells Loop Size: %i\n", UVshells.size());
                for (const auto& UVsh : UVshells) {
                    auto UV = UVsh->get_shell_pair_index();
                    int U = UV.first;
                    int V = UV.second;

                    if (density_screening_) {
                        double screen_val = D_maxp[U][V] * D_maxp[U][V] * Jmet_max[P] * ints[thread]->shell_pair_value(U,V);
                        //outfile->Printf("            Screening (%i, %i)... %f (%f, %f, %f, %f), %f\n", U, V, screen_val, D_maxp[U][V], D_maxp[U][V], Jmet_max[P], ints[thread]->shell_pair_value(U,V), ints_tolerance_*ints_tolerance_);
                        if (screen_val < ints_tolerance_*ints_tolerance_) continue;
                    }
	    
                    //outfile->Printf("            Ket Shp indices: (%i, %i)...\n", U, V);
                    int u_start = primary_->shell(U).start();
                    int num_u = primary_->shell(U).nfunction();

                    int v_start = primary_->shell(V).start();
                    int num_v = primary_->shell(V).nfunction();

                    ints[thread]->compute_shell(P, 0, U, V);
                    const double* buffer = ints[thread]->buffer();

                    double prefactor = 2.0;
                    if (U == V) prefactor *= 0.5;

                    for (int i = 0; i < D.size(); i++) {
                        double** Dp = D[i]->pointer();
                        double *Puv = const_cast<double *>(buffer);
                        double *gamp = J[i]->pointer()[0];

                        /*
                        for (int p = p_start; p < p_start + num_p; p++) {
                            int dp = p - p_start;
                            for (int u = u_start; u < u_start + num_u; u++) {
                                int du = u - u_start;
                                for (int v = v_start; v < v_start + num_v; v++) {
                                    int dv = v - v_start;
                                    gamp[p] += prefactor * (*Puv) * Dp[u][v];
                                    Puv++;
                                }
                            }
                        }
                        */ // TODO FIX THIS, NEEDS BACKSLASH AT END

                        std::vector<double> Dbuff(num_u * num_v, 0.0);
                        double* Dbp = Dbuff.data();

                        for (int u = u_start; u < u_start + num_u; u++) {
                            for (int v = v_start; v < v_start + num_v; v++) {
                                *(Dbp) = Dp[u][v];
                                Dbp++;
                            }
                        }
                        C_DGEMV('N', num_p, num_u * num_v, prefactor, Puv, num_u * num_v, Dbuff.data(), 1, 1.0, &(gamp[p_start]), 1);

                    } // end i
                } // UV shells
                //outfile->Printf("          End UVshells Loop\n");
            } // NF Boxes
            //outfile->Printf("        End NF Box Loop\n\n");
        } // Qtask
    } // Ptask

    //outfile->Printf("    END build_nf_gammaP\n");
    timer_off("DFCFMM: Near Field Gamma P");
}

void DFCFMMTree::build_nf_df_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
		      const std::vector<double>& Jmet_max) {
    timer_on("DFCFMM: Near Field J");
    //outfile->Printf("  START build_nf_df_J\n");

    // => Sizing <= //

    int pri_nshell = primary_->nshell();
    int aux_nshell = auxiliary_->nshell();
    int nbf = primary_->nbf();
    int nmat = D.size();

    int max_nbf_per_shell = 0;
    for (int P = 0; P < pri_nshell; P++) {
        max_nbf_per_shell = std::max(max_nbf_per_shell, primary_->shell(P).nfunction());
    }

    // => J buffers (to satisfy DGEMM)
    std::vector<std::vector<SharedMatrix>> JT;

    for (int thread = 0; thread < nthread_; thread++) {
        std::vector<SharedMatrix> J2;
        for (size_t ind = 0; ind < nmat; ind++) {
            J2.push_back(std::make_shared<Matrix>(max_nbf_per_shell, max_nbf_per_shell));
            J2[ind]->zero();
        }
        JT.push_back(J2);
    }

    // check that Jmet_max is not empty is density screening is enabled
    if (density_screening_ && Jmet_max.empty()) {
        throw PSIEXCEPTION("CFMMTree::build_nf_gamma_P was called with density screening enabled, but Jmet is NULL. Check your arguments to build_J.");
    }

    // set up D_max for screening purposes
    std::vector<double> D_max(aux_nshell, 0.0);

    if (density_screening_) {
#pragma omp parallel for
        for (int P = 0; P < aux_nshell; P++) {
            int p_start = auxiliary_->shell(P).start();
            int num_p = auxiliary_->shell(P).nfunction();
            for (size_t i = 0; i < D.size(); i++) {
                double* Dp = D[i]->pointer()[0];
                for (int p = p_start; p < p_start + num_p; p++) {
                    D_max[P] = std::max(D_max[P], std::abs(Dp[p]));
                }
            }
        }

    }

    //outfile->Printf("    Start UVTask loop\n");
    //outfile->Printf("    UVTask loop size: %i\n", primary_shellpair_list_.size());
#pragma omp parallel for collapse(2) schedule(guided)
    for (int Utask = 0; Utask < primary_shellpair_list_.size(); Utask++) {
        for (int Vtask = 0; Vtask < primary_shellpair_list_.size(); Vtask++) {
            //outfile->Printf("     Processing task (%i, %i)...\n", Utask, Vtask);
 
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(primary_shellpair_list_[Utask][Vtask]);
            if (shellpair == nullptr) continue;

            auto [U, V] = shellpair->get_shell_pair_index();
            //outfile->Printf("      Bra Shp indices: (%i, %i)...\n", U, V);
 
            const GaussianShell& Ushell = primary_->shell(U);
            const GaussianShell& Vshell = primary_->shell(V);

            int u_start = Ushell.start();
            int num_u = Ushell.nfunction();

            int v_start = Vshell.start();
            int num_v = Vshell.nfunction();

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            double prefactor = 2.0;
            if (U == V) prefactor *= 0.5;

            //outfile->Printf("        Start NF box loop\n");
            //outfile->Printf("          NF Box loop size: %i\n", primary_shellpair_to_nf_boxes_[U][V].size());
            for (const auto& nf_box : primary_shellpair_to_nf_boxes_[U][V]) {
                auto& Qshells = nf_box->get_auxiliary_shell_pairs();

                //outfile->Printf("          Start Qsh loop\n");
                //outfile->Printf("            Qsh loop size: %i\n", Qshells.size());
 
                for (const auto& Qsh : Qshells) {

                    int Q = Qsh->get_shell_pair_index().first;

                    if (density_screening_) {
	    	        double screen_val = D_max[Q] * D_max[Q] * Jmet_max[Q] * ints[thread]->shell_pair_value(U,V);
                        //outfile->Printf("            Screening (%i)... %f (%f, %f, %f, %f), %f\n", Q, screen_val, D_max[Q], D_max[Q], Jmet_max[Q], ints[thread]->shell_pair_value(U,V), ints_tolerance_*ints_tolerance_);
                        if (screen_val < ints_tolerance_*ints_tolerance_) continue;
                    }
                    //outfile->Printf("            Ket Shp indices: (%i,)...\n", Q);
	    
                    int q_start = auxiliary_->shell(Q).start();
                    int num_q = auxiliary_->shell(Q).nfunction();

                    ints[thread]->compute_shell(Q, 0, U, V);

                    const double* buffer = ints[thread]->buffer();

                    for (int i = 0; i < D.size(); i++) {
                        double* JTp = JT[thread][i]->pointer()[0];
                        double* Dp = D[i]->pointer()[0];
                        double* Quv = const_cast<double *>(buffer);

                        /*
                        for (int q = q_start; q < q_start + num_q; q++) {
                            int dq = q - q_start;
                            for (int u = u_start; u < u_start + num_u; u++) {
                                int du = u - u_start;
                                for (int v = v_start; v < v_start + num_v; v++) {
                                    int dv = v - v_start;
                                    JTp[du * num_v + dv] += prefactor * (*Quv) * Dp[q];
                                    Quv++;
                                }
                            }
                        }
                        */ //TODO FIX THIS, NEEDS BACKSLASH AT END

                        C_DGEMV('T', num_q, num_u * num_v, prefactor, Quv, num_u * num_v, &(Dp[q_start]), 1, 1.0, JTp, 1);

                    } // end i
                } // end Qsh
                //outfile->Printf("          End Qsh loop\n");
            } // end nf box
            //outfile->Printf("        End NF box loop\n\n");

            // => Stripeout >= //

            for (int i = 0; i < D.size(); i++) {
                double* JTp = JT[thread][i]->pointer()[0];
                double** Jp = J[i]->pointer();
                for (int u = u_start; u < u_start + num_u; u++) {
                    int du = u - u_start;
                    for (int v = v_start; v < v_start + num_v; v++) {
                        int dv = v - v_start;

                        Jp[u][v] += JTp[du * num_v + dv];
                    }
                }
                JT[thread][i]->zero();
            }
        } // end Vtask
    } // end Utask
    //outfile->Printf("      End UVTask loop\n");

    //outfile->Printf("    END build_nf_df_J\n");
    timer_off("DFCFMM: Near Field J");
}

/*
void CFMMTree::build_nf_df_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
		      const std::vector<double>& Jmet_max) {
    timer_on("DF CFMM: Near Field J");

    // => Sizing <= //

    int pri_nshell = primary_->nshell();
    int aux_nshell = auxiliary_->nshell();
    int nbf = primary_->nbf();
    int nmat = D.size();

    int max_nbf_per_shell = 0;
    for (int P = 0; P < pri_nshell; P++) {
        max_nbf_per_shell = std::max(max_nbf_per_shell, primary_->shell(P).nfunction());
    }

    // => J buffers (to satisfy DGEMM)
    std::vector<std::vector<SharedMatrix>> JT;

    for (int thread = 0; thread < nthread_; thread++) {
        std::vector<SharedMatrix> J2;
        for (size_t ind = 0; ind < nmat; ind++) {
            J2.push_back(std::make_shared<Matrix>(max_nbf_per_shell, max_nbf_per_shell));
            J2[ind]->zero();
        }
        JT.push_back(J2);
    }

    // check that Jmet_max is not empty is density screening is enabled
    if (density_screening_ && Jmet_max.empty()) {
        throw PSIEXCEPTION("CFMMTree::build_nf_gamma_P was called with density screening enabled, but Jmet is NULL. Check your arguments to build_J.");
    }

    // set up D_max for screening purposes
    std::vector<double> D_max(aux_nshell, 0.0);

    if (density_screening_) {
#pragma omp parallel for
        for (int P = 0; P < aux_nshell; P++) {
            int p_start = auxiliary_->shell(P).start();
            int num_p = auxiliary_->shell(P).nfunction();
            for (size_t i = 0; i < D.size(); i++) {
                double* Dp = D[i]->pointer()[0];
                for (int p = p_start; p < p_start + num_p; p++) {
                    D_max[P] = std::max(D_max[P], std::abs(Dp[p]));
                }
            }
        }

    }

#pragma omp parallel for schedule(guided)
    for (int task = 0; task < primary_shellpair_tasks_.size(); task++) {

        int U = primary_shellpair_tasks_[task].first;
        int V = primary_shellpair_tasks_[task].second;

        const GaussianShell& Ushell = primary_->shell(U);
        const GaussianShell& Vshell = primary_->shell(V);

        int u_start = Ushell.start();
        int num_u = Ushell.nfunction();

        int v_start = Vshell.start();
        int num_v = Vshell.nfunction();

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        double prefactor = 2.0;
        if (U == V) prefactor *= 0.5;

        for (const auto& nf_box : primary_shellpair_to_nf_boxes_[task]) {
            auto& Qshells = nf_box->get_auxiliary_shell_pairs();

            for (const auto& Qsh : Qshells) {

                int Q = Qsh->get_shell_pair_index().first;

                if (density_screening_) {
		            double screen_val = D_max[Q] * D_max[Q] * Jmet_max[Q] * ints[thread]->shell_pair_value(U,V);
                    if (screen_val < ints_tolerance_*ints_tolerance_) continue;
                }
	
                int q_start = auxiliary_->shell(Q).start();
                int num_q = auxiliary_->shell(Q).nfunction();

                ints[thread]->compute_shell(Q, 0, U, V);

                const double* buffer = ints[thread]->buffer();

                for (int i = 0; i < D.size(); i++) {
                    double* JTp = JT[thread][i]->pointer()[0];
                    double* Dp = D[i]->pointer()[0];
                    double* Quv = const_cast<double *>(buffer);

                    /*
                    for (int q = q_start; q < q_start + num_q; q++) {
                        int dq = q - q_start;
                        for (int u = u_start; u < u_start + num_u; u++) {
                            int du = u - u_start;
                            for (int v = v_start; v < v_start + num_v; v++) {
                                int dv = v - v_start;
                                JTp[du * num_v + dv] += prefactor * (*Quv) * Dp[q];
                                Quv++;
                            }
                        }
                    }
                    * //TODO FIX THIS, NEEDS BACKSLASH AT END

                    C_DGEMV('T', num_q, num_u * num_v, prefactor, Quv, num_u * num_v, &(Dp[q_start]), 1, 1.0, JTp, 1);

                } // end i
            } // end Qsh
        } // end nf box

        // => Stripeout >= //

        for (int i = 0; i < D.size(); i++) {
            double* JTp = JT[thread][i]->pointer()[0];
            double** Jp = J[i]->pointer();
            for (int u = u_start; u < u_start + num_u; u++) {
                int du = u - u_start;
                for (int v = v_start; v < v_start + num_v; v++) {
                    int dv = v - v_start;

                    Jp[u][v] += JTp[du * num_v + dv];
                }
            }
            JT[thread][i]->zero();
        }
    } // end task

    timer_off("DF CFMM: Near Field J");
}
*/

/*
void CFMMTree::build_nf_metric(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J) {
    throw PSIEXCEPTION("Not implemented. Leave me alone. I'm tired. -Andy Jiang");
}
*/

void DFCFMMTree::build_ff_J(std::vector<SharedMatrix>& J) {

    timer_on("DFCFMMTree: Far Field J");
    //outfile->Printf("    Start DFCFMMTree: Far Field J\n");

    bool is_primary = (contraction_type_ == ContractionType::DF_PRI_AUX);
    int nbf = (is_primary) ? primary_->nbf() : 1;

    std::shared_ptr<BasisSet>& ref_basis = (is_primary) ? primary_ : auxiliary_;
    const auto& shellpair_list = (is_primary) ? primary_shellpair_list_ : auxiliary_shellpair_list_;

    //outfile->Printf("      ShPair List size: %i\n", shellpair_list.size());
    //outfile->Printf("      Begin ShPair List Loop\n");
#pragma omp parallel for collapse(2) schedule(guided)
    for (int Ptask = 0; Ptask < shellpair_list.size(); Ptask++) {
        for (int Qtask = 0; Qtask < shellpair_list.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(shellpair_list[Ptask][Qtask]);
            if (shellpair == nullptr) continue;

            auto [P, Q] = shellpair->get_shell_pair_index();
            //outfile->Printf("        Processing shell pair (%i, %i)...\n", P, Q);

            const auto& Vff = std::get<1>(shellpair_list[Ptask][Qtask])->far_field_vector();

            const auto& shellpair_mpoles = shellpair->get_mpoles();

            double prefactor = (!is_primary || P == Q) ? 1.0 : 2.0;
            //double prefactor = (P == Q) ? 1.0 : 2.0;
            if (is_primary) prefactor /= 2.0;             

            const GaussianShell& Pshell = shellpair->bs1()->shell(P);
            const GaussianShell& Qshell = shellpair->bs2()->shell(Q);

            int p_start = Pshell.start();
            int num_p = Pshell.nfunction();

            int q_start = Qshell.start();
            int num_q = Qshell.nfunction();

            for (int p = p_start; p < p_start + num_p; p++) {
                int dp = p - p_start;
                for (int q = q_start; q < q_start + num_q; q++) {
                    int dq = q - q_start;
                    for (int N = 0; N < J.size(); N++) {
                        double* Jp = J[N]->pointer()[0];
                        // Far field multipole contributions
                        auto contribution = prefactor * Vff[N]->dot(shellpair_mpoles[dp * num_q + dq]);
                        //outfile->Printf("          Jp[%i][%i] <- %f. Contribution: %f, %f, %f\n", p, q, contribution, prefactor, Vff[N]->get_multipoles()[0][0], shellpair_mpoles[dp * num_q + dq]->get_multipoles()[0][0]);

#pragma omp atomic
                        Jp[p * nbf + q] += contribution; 
                    } // end N
                } // end q
            } // end p
        } // end Qtasks
    } // end Ptasks
    //outfile->Printf("      End ShPair List Loop\n");

    //outfile->Printf("    End DFCFMMTree: Far Field J\n");
    timer_off("DFCFMMTree: Far Field J");
}

void DFCFMMTree::build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                        const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
			bool do_incfock_iter, const std::vector<double>& Jmet_max) {

    timer_on("DFCFMMTree: J");

    // actually build J
    J_build_kernel(ints, D, J, do_incfock_iter, Jmet_max);
   
    // Hermitivitize J matrix afterwards
    if (contraction_type_ == ContractionType::DF_PRI_AUX) {
        for (int ind = 0; ind < D.size(); ind++) {
            J[ind]->hermitivitize();
        }
    }

    timer_off("DFCFMMTree: J");
}

} // end namespace psi
