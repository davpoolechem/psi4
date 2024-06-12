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

DirectCFMMTree::DirectCFMMTree(std::shared_ptr<BasisSet> primary, Options& options) 
                    : CFMMTree(primary, options) {
    timer_on("DirectCFMMTree: Setup");

    // ints engine 
    auto factory = std::make_shared<IntegralFactory>(primary_);
    auto shellpair_int = std::shared_ptr<TwoBodyAOInt>(factory->eri());

    // shell pair info
    auto& ints_shell_pairs = shellpair_int->shell_pairs();
    size_t nshell_pairs = ints_shell_pairs.size();
    primary_shell_pairs_.resize(nshell_pairs);
    
    double cfmm_extent_tol = options.get_double("CFMM_EXTENT_TOLERANCE");

#pragma omp parallel for
    for (size_t pair_index = 0; pair_index < nshell_pairs; pair_index++) {
        const auto& pair = ints_shell_pairs[pair_index];
        primary_shell_pairs_[pair_index] = std::make_shared<CFMMShellPair>(primary_, primary_, pair, mpole_coefs_, cfmm_extent_tol);
    }

    make_tree(nshell_pairs);

    calculate_shellpair_multipoles(true);
    if (print_ >= 1) print_out();

    timer_off("DirectCFMMTree: Setup");
}

void DirectCFMMTree::make_root_node() {
    // base algorithm for constructing CFMM tree node
    auto [origin, length] = make_root_node_kernel();
    
    // create top-level box
    outfile->Printf("BUILDING TREE_[0]\n");
    tree_[0] = std::make_shared<CFMMBox>(nullptr, primary_shell_pairs_, origin, length, 0, lmax_, 2); // TODO: Separate this out, so DF-CFMM can pass auxiliary_shell_pairs_
    outfile->Printf("  Tree length #0a: %f, %f\n", tree_[0]->length());
}

//DirectCFMMTree::DirectCFMMTree(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options) {
//    DirectCFMMTree(primary, options);
//}  

void DirectCFMMTree::calculate_shellpair_multipoles(bool is_primary) {

    timer_on("DirectCFMMTree: Shell-Pair Multipole Ints");

    if (!is_primary) throw PSIEXCEPTION("is_primary = false only applies to DFCFMMTree, not DirectCFMMTree!");

    std::vector<std::shared_ptr<OneBodyAOInt>> sints;
    std::vector<std::shared_ptr<OneBodyAOInt>> mpints;

    auto int_factory = std::make_shared<IntegralFactory>(primary_);
    for (int thread = 0; thread < nthread_; thread++) {
        sints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_overlap()));
        mpints.push_back(std::shared_ptr<OneBodyAOInt>(int_factory->ao_multipoles(lmax_)));
    }

#pragma omp parallel for collapse(2) schedule(guided)
    for (int Ptask = 0; Ptask < primary_shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < primary_shellpair_list_.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(primary_shellpair_list_[Ptask][Qtask]);
            if (shellpair == nullptr) continue;
            
            std::shared_ptr<CFMMBox> box = std::get<1>(primary_shellpair_list_[Ptask][Qtask]);
            
            auto [P, Q] = shellpair->get_shell_pair_index();
    
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
 
            mpints[thread]->set_origin(box->center());
            shellpair->calculate_mpoles(box->center(), sints[thread], mpints[thread], lmax_);
        }  
    }

    timer_off("DirectCFMMTree: Shell-Pair Multipole Ints");
}

void DirectCFMMTree::build_nf_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                          const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
                          const std::vector<double>& Jmet_max) {

    timer_on("DirectCFMMTree: Near Field J");

    int nshell = primary_->nshell();
    int natom = molecule_->natom();

    // Maximum space (r_nbf * s_nbf) to allocate per task
    size_t max_alloc = 0;

    // A map of the function (num_r * num_s) offsets per shell-pair in a box pair
    std::unordered_map<int, int> offsets;
    
    int start = (nlevels_ == 1) ? 0 : (0.5 * std::pow(16, nlevels_-1) + 7) / 15;
    int end = (nlevels_ == 1) ? 1 : (0.5 * std::pow(16, nlevels_) + 7) / 15;

    for (int bi = start; bi < end; bi++) {
        auto& RSshells = tree_[bi]->get_primary_shell_pairs();
        int RSoff = 0;
        for (int RSind = 0; RSind < RSshells.size(); RSind++) {
            auto [R, S] = RSshells[RSind]->get_shell_pair_index();
            offsets[R * nshell + S] = RSoff;
            int Rfunc = primary_->shell(R).nfunction();
            int Sfunc = primary_->shell(S).nfunction();
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
    for (int Ptask = 0; Ptask < primary_shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < primary_shellpair_list_.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(primary_shellpair_list_[Ptask][Qtask]);
            
            if (shellpair == nullptr) continue;
       
            std::shared_ptr<CFMMBox> box = std::get<1>(primary_shellpair_list_[Ptask][Qtask]);
            std::vector<std::shared_ptr<CFMMBox>> nf_boxes = box->near_field_boxes(); 
 
            auto [P, Q] = shellpair->get_shell_pair_index();
            
            const GaussianShell& Pshell = primary_->shell(P);
            const GaussianShell& Qshell = primary_->shell(Q);

            int p_start = Pshell.start();
            int num_p = Pshell.nfunction();

            int q_start = Qshell.start();
            int num_q = Qshell.nfunction();

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
        
            for (const auto& nf_box : nf_boxes) {
                auto& RSshells = nf_box->get_primary_shell_pairs();
                
                bool touched = false;
     
                for (const auto& RSsh : RSshells) {
                    auto [R, S] = RSsh->get_shell_pair_index();
                
                    if (R * nshell + S > P * nshell + Q) continue;
                    if (!shell_significant(P, Q, R, S, ints, D)) continue;
            
                    if (ints[thread]->compute_shell(P, Q, R, S) == 0) continue;
                    computed_shells++;

                    const GaussianShell& Rshell = primary_->shell(R);
                    const GaussianShell& Sshell = primary_->shell(S);

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
                    
                        const GaussianShell& Rshell = primary_->shell(R);
                        const GaussianShell& Sshell = primary_->shell(S);
                    
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

    timer_off("DirectCFMMTree: Near Field J");
}

void DirectCFMMTree::build_ff_J(std::vector<SharedMatrix>& J) {

    timer_on("DirectCFMMTree: Far Field J");
    outfile->Printf("    Start DirectCFMMTree: Far Field J\n");
   
    outfile->Printf("      ShPair List size: %i\n", primary_shellpair_list_.size());
    outfile->Printf("      Begin ShPair List Loop\n");
#pragma omp parallel for collapse(2) schedule(guided)
    for (int Ptask = 0; Ptask < primary_shellpair_list_.size(); Ptask++) {
        for (int Qtask = 0; Qtask < primary_shellpair_list_.size(); Qtask++) {
            std::shared_ptr<CFMMShellPair> shellpair = std::get<0>(primary_shellpair_list_[Ptask][Qtask]);
            if (shellpair == nullptr) {
                continue;
            }

            auto [P, Q] = shellpair->get_shell_pair_index();
            outfile->Printf("        Processing shell pair (%i, %i)...\n", P, Q);
    
            const auto& Vff = std::get<1>(primary_shellpair_list_[Ptask][Qtask])->far_field_vector();
                
            const auto& shellpair_mpoles = shellpair->get_mpoles();
     
            double prefactor = (P == Q) ? 1.0 : 2.0;
        
            const GaussianShell& Pshell = primary_->shell(P);
            const GaussianShell& Qshell = primary_->shell(Q);
    
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
                        auto contribution = prefactor * Vff[N]->dot(shellpair_mpoles[dp * num_q + dq]);
                        outfile->Printf("          Jp[%i][%i] <- %f. Contribution: %f, %f, %f\n", p, q, contribution, prefactor, Vff[N]->get_multipoles()[0][0], shellpair_mpoles[dp * num_q + dq]->get_multipoles()[0][0]);
#pragma omp atomic
                        Jp[p][q] += contribution; 
                    } // end N
                } // end q
            } // end p
        } // end Qtasks
    } // end Ptasks
    outfile->Printf("      End ShPair List Loop\n");

    outfile->Printf("    End DirectCFMMTree: Far Field J\n");
    timer_off("DirectCFMMTree: Far Field J");
}

void DirectCFMMTree::build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                        const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, 
                        bool do_incfock_iter, const std::vector<double>& Jmet_max) {

    timer_on("DirectCFMMTree: J");
    outfile->Printf("  Start DirectCFMMTree: J\n");
    
    // actually build J 
    J_build_kernel(ints, D, J, do_incfock_iter, Jmet_max);
 
    // Hermitivitize J matrix afterwards
    for (int ind = 0; ind < D.size(); ind++) {
        J[ind]->hermitivitize();
    }

    outfile->Printf("  End DirectCFMMTree: J\n");
    timer_off("DirectCFMMTree: J");
}

} // end namespace psi
