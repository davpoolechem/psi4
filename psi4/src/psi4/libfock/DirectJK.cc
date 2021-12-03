/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2021 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/lib3index/3index.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/aiohandler.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libiwl/iwl.hpp"
#include "jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/integral.h"
#include "psi4/lib3index/cholesky.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <map>
#include <unordered_set>
#include "psi4/libpsi4util/PsiOutStream.h"
#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

#ifdef USING_BrianQC

#include <use_brian_wrapper.h>
#include <brian_macros.h>
#include <brian_common.h>
#include <brian_scf.h>
#include <brian_cphf.h>

extern void checkBrian();
extern BrianCookie brianCookie;
extern bool brianEnable;
extern bool brianEnableDFT;
extern bool brianCPHFFlag;
extern bool brianCPHFLeftSideFlag;
extern brianInt brianRestrictionType;

#endif

using namespace psi;

namespace psi {

DirectJK::DirectJK(std::shared_ptr<BasisSet> primary, Options& options) : JK(primary), options_(options) { common_init(); }
DirectJK::~DirectJK() {}
void DirectJK::common_init() {
    df_ints_num_threads_ = 1;

#ifdef _OPENMP
    df_ints_num_threads_ = Process::environment.get_n_threads();
#endif

    incfock_ = options_.get_bool("INCFOCK");
    incfock_count_ = 0;
    do_incfock_iter_ = false;
    if (options_.get_int("INCFOCK_FULL_FOCK_EVERY") <= 0) {
        throw PSIEXCEPTION("Invalid input for option INCFOCK_FULL_FOCK_EVERY (<= 0)");
    }
    density_screening_ = options_.get_str("SCREENING") == "DENSITY";
    linK_ = options_.get_bool("DO_LINK");
    linK_ints_cutoff_ = options_.get_double("LINK_INTS_TOLERANCE");
    set_cutoff(options_.get_double("INTS_TOLERANCE"));
}
size_t DirectJK::memory_estimate() {
    return 0;  // Effectively
}
void DirectJK::print_header() const {
    std::string screen_type = options_.get_str("SCREENING");
    if (print_) {
        outfile->Printf("  ==> DirectJK: Integral-Direct J/K Matrices <==\n\n");

        outfile->Printf("    J tasked:          %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:          %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:         %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_) outfile->Printf("    Omega:             %11.3E\n", omega_);
        outfile->Printf("    Integrals threads: %11d\n", df_ints_num_threads_);
        // outfile->Printf( "    Memory [MiB]:      %11ld\n", (memory_ *8L) / (1024L * 1024L));
        outfile->Printf("    Screening Type:    %11s\n", screen_type.c_str());
        outfile->Printf("    Screening Cutoff:  %11.0E\n", cutoff_);
        outfile->Printf("    Incremental Fock:  %11s\n", incfock_ ? "Yes" : "No");
        outfile->Printf("    LinK:              %11s\n\n", linK_ ? "Yes" : "No");
    }
}
void DirectJK::preiterations() {

#ifdef USING_BrianQC
    if (brianEnable) {
        double threshold = cutoff_ * (brianCPHFFlag ? 1e-3 : 1e-0); // CPHF needs higher precision
        brianCOMSetPrecisionThresholds(&brianCookie, &threshold);
        checkBrian();
    }
#endif
}

void DirectJK::incfock_setup() {

    // The prev_D_ao_ condition is used to handle stability analysis case
    if (initial_iteration_ || prev_D_ao_.size() != D_ao_.size()) {
        initial_iteration_ = true;

        prev_D_ao_.clear();
        delta_D_ao_.clear();

        if (do_wK_) {
            prev_wK_ao_.clear();
            delta_wK_ao_.clear();
        }

        if (do_J_) {
            prev_J_ao_.clear();
            delta_J_ao_.clear();
        }

        if (do_K_) {
            prev_K_ao_.clear();
            delta_K_ao_.clear();
        }
    
        for (size_t N = 0; N < D_ao_.size(); N++) {
            prev_D_ao_.push_back(std::make_shared<Matrix>("D Prev", D_ao_[N]->nrow(), D_ao_[N]->ncol()));
            delta_D_ao_.push_back(std::make_shared<Matrix>("Delta D", D_ao_[N]->nrow(), D_ao_[N]->ncol()));

            if (do_wK_) {
                prev_wK_ao_.push_back(std::make_shared<Matrix>("wK Prev", wK_ao_[N]->nrow(), wK_ao_[N]->ncol()));
                delta_wK_ao_.push_back(std::make_shared<Matrix>("Delta wK", wK_ao_[N]->nrow(), wK_ao_[N]->ncol()));
            }
                
            if (do_J_) {
                prev_J_ao_.push_back(std::make_shared<Matrix>("J Prev", J_ao_[N]->nrow(), J_ao_[N]->ncol()));
                delta_J_ao_.push_back(std::make_shared<Matrix>("Delta J", J_ao_[N]->nrow(), J_ao_[N]->ncol()));
            }
        
            if (do_K_) {
                prev_K_ao_.push_back(std::make_shared<Matrix>("K Prev", K_ao_[N]->nrow(), K_ao_[N]->ncol()));
                delta_K_ao_.push_back(std::make_shared<Matrix>("Delta K", K_ao_[N]->nrow(), K_ao_[N]->ncol()));
            }
        }
    } else {
        for (size_t N = 0; N < D_ao_.size(); N++) {
            delta_D_ao_[N]->copy(D_ao_[N]);
            delta_D_ao_[N]->subtract(prev_D_ao_[N]);
        }
    }
}
void DirectJK::incfock_postiter() {
    if (do_incfock_iter_) {
        for (size_t N = 0; N < D_ao_.size(); N++) {

            if (do_wK_) {
                prev_wK_ao_[N]->add(delta_wK_ao_[N]);
                wK_ao_[N]->copy(prev_wK_ao_[N]);
            }

            if (do_J_) {
                prev_J_ao_[N]->add(delta_J_ao_[N]);
                J_ao_[N]->copy(prev_J_ao_[N]);
            }

            if (do_K_) {
                prev_K_ao_[N]->add(delta_K_ao_[N]);
                K_ao_[N]->copy(prev_K_ao_[N]);
            }

            prev_D_ao_[N]->copy(D_ao_[N]);
        }
    } else {
        for (size_t N = 0; N < D_ao_.size(); N++) {
            if (do_wK_) prev_wK_ao_[N]->copy(wK_ao_[N]);
            if (do_J_) prev_J_ao_[N]->copy(J_ao_[N]);
            if (do_K_) prev_K_ao_[N]->copy(K_ao_[N]);
            prev_D_ao_[N]->copy(D_ao_[N]);
        }
    }
}

void DirectJK::compute_JK() {
#ifdef USING_BrianQC
    if (brianEnable) {
        brianBool computeCoulomb = (do_J_ ? BRIAN_TRUE : BRIAN_FALSE);
        brianBool computeExchange = ((do_K_ || do_wK_) ? BRIAN_TRUE : BRIAN_FALSE);

        if (do_wK_ and not brianEnableDFT) {
            throw PSIEXCEPTION("Currently, BrianQC cannot compute range-separated exact exchange when Psi4 is handling the DFT terms");
        }

        if (not brianCPHFFlag) {
            if (!lr_symmetric_) {
                throw PSIEXCEPTION("Currently, BrianQC's non-CPHF Fock building only works with symmetric densities");
            }

            // BrianQC only computes the sum of all Coulomb contributions.
            // For ROHF, the matrices are not the alpha and beta densities, but
            // the doubly and singly occupied densities, and the weight of
            // the first Coulomb contribution must be two. Currently, we
            // achieve this by scaling the doubly occupied density
            // before building, and doing the reverse for the results.
            // We also restore the original density in case it is still needed.
            if (brianRestrictionType == BRIAN_RESTRICTION_TYPE_ROHF) {
                D_ao_[0]->scale(2.0);
            }

            double* exchangeAlpha = nullptr;
            double* exchangeBeta = nullptr;
            if (do_K_) {
                exchangeAlpha = K_ao_[0]->get_pointer();
                exchangeBeta = (D_ao_.size() > 1) ? K_ao_[1]->get_pointer() : nullptr;
            } else if (do_wK_) {
                exchangeAlpha = wK_ao_[0]->get_pointer();
                exchangeBeta = (D_ao_.size() > 1) ? wK_ao_[1]->get_pointer() : nullptr;
            }

            brianSCFBuildFockRepulsion(&brianCookie,
                &computeCoulomb,
                &computeExchange,
                D_ao_[0]->get_pointer(0),
                (D_ao_.size() > 1 ? D_ao_[1]->get_pointer() : nullptr),
                (do_J_ ? J_ao_[0]->get_pointer() : nullptr),
                exchangeAlpha,
                exchangeBeta
            );
            checkBrian();

            // BrianQC computes the sum of all Coulomb contributions into
            // J_ao_[0], so all other contributions must be zeroed out for
            // the sum to be correct. For RHF/RKS, Psi4 expects J_ao_[0]
            // to contain the alpha contribution instead of the total, so
            // we halve it.
            if (do_J_) {
                if (brianRestrictionType == BRIAN_RESTRICTION_TYPE_RHF) {
                    J_ao_[0]->scale(0.5);
                }

                for (size_t ind = 1; ind < J_ao_.size(); ind++) {
                    J_ao_[ind]->zero();
                }
            }

            if (brianRestrictionType == BRIAN_RESTRICTION_TYPE_ROHF) {
                D_ao_[0]->scale(0.5);

                if (do_J_) {
                    J_ao_[0]->scale(0.5);
                }

                if (do_K_) {
                    K_ao_[0]->scale(0.5);
                }

                if (do_wK_) {
                    wK_ao_[0]->scale(0.5);
                }
            }
        } else {
            brianInt maxSegmentSize;
            brianCPHFMaxSegmentSize(&brianCookie, &maxSegmentSize);

            brianInt densityCount = (brianRestrictionType == BRIAN_RESTRICTION_TYPE_RHF) ? 1 : 2;
            if (D_ao_.size() % densityCount != 0) {
                throw PSIEXCEPTION("Invalid number of density matrices for CPHF");
            }

            brianInt derivativeCount = D_ao_.size() / densityCount;

            for (brianInt segmentStartIndex = 0; segmentStartIndex < derivativeCount; segmentStartIndex += maxSegmentSize) {
                brianInt segmentSize = std::min(maxSegmentSize, derivativeCount - segmentStartIndex);

                std::vector<std::vector<SharedMatrix>> pseudoDensitySymmetrized(densityCount);
                std::vector<std::vector<const double*>> pseudoDensityPointers(densityCount);
                std::vector<std::vector<double*>> pseudoExchangePointers(densityCount);
                for (brianInt densityIndex = 0; densityIndex < densityCount; densityIndex++) {
                    pseudoDensitySymmetrized[densityIndex].resize(segmentSize);
                    pseudoDensityPointers[densityIndex].resize(segmentSize, nullptr);
                    pseudoExchangePointers[densityIndex].resize(segmentSize, nullptr);
                    for (brianInt i = 0; i < segmentSize; i++) {
                        // Psi4's code computing the left- and right-hand side CPHF terms use different indexing conventions
                        brianInt psi4Index = brianCPHFLeftSideFlag ? (densityIndex * derivativeCount + segmentStartIndex + i) : ((segmentStartIndex + i) * densityCount + densityIndex);

                        pseudoDensitySymmetrized[densityIndex][i] = D_ao_[psi4Index]->clone();
                        pseudoDensitySymmetrized[densityIndex][i]->add(D_ao_[psi4Index]->transpose());
                        pseudoDensitySymmetrized[densityIndex][i]->scale(0.5);
                        pseudoDensityPointers[densityIndex][i] = pseudoDensitySymmetrized[densityIndex][i]->get_pointer();

                        if (do_K_) {
                            pseudoExchangePointers[densityIndex][i] = K_ao_[psi4Index]->get_pointer();
                        } else if (do_wK_) {
                            pseudoExchangePointers[densityIndex][i] = wK_ao_[psi4Index]->get_pointer();
                        }
                    }
                }

                std::vector<double*> pseudoCoulombPointers(segmentSize, nullptr);
                for (brianInt i = 0; i < segmentSize; i++) {
                    if (do_J_) {
                        // we always write the total coulomb into the densityIndex == 0 matrix, and later divide it if necessary
                        brianInt psi4Index = brianCPHFLeftSideFlag ? (0 * derivativeCount + segmentStartIndex + i) : ((segmentStartIndex + i) * densityCount + 0);
                        pseudoCoulombPointers[i] = J_ao_[psi4Index]->get_pointer();
                    }
                }

                brianCPHFBuildRepulsion(&brianCookie,
                    &computeCoulomb,
                    &computeExchange,
                    &segmentSize,
                    pseudoDensityPointers[0].data(),
                    (densityCount > 1) ? pseudoDensityPointers[1].data() : nullptr,
                    pseudoCoulombPointers.data(),
                    pseudoExchangePointers[0].data(),
                    (densityCount > 1) ? pseudoExchangePointers[1].data() : nullptr
                );
                checkBrian();

                // BrianQC computes the sum of all Coulomb contributions into
                // J_ao_[0], so all other contributions must be zeroed out for
                // the sum to be correct. For RHF/RKS, Psi4 expects J_ao_[0]
                // to contain the alpha contribution instead of the total, so
                // we halve it.
                if (do_J_) {
                    for (brianInt i = 0; i < segmentSize; i++) {
                        if (brianRestrictionType == BRIAN_RESTRICTION_TYPE_RHF) {
                            brianInt psi4Index = brianCPHFLeftSideFlag ? (0 * derivativeCount + segmentStartIndex + i) : ((segmentStartIndex + i) * densityCount + 0);
                            J_ao_[psi4Index]->scale(0.5);
                        }

                        for (brianInt densityIndex = 1; densityIndex < densityCount; densityIndex++) {
                            brianInt psi4Index = brianCPHFLeftSideFlag ? (densityIndex * derivativeCount + segmentStartIndex + i) : ((segmentStartIndex + i) * densityCount + densityIndex);
                            J_ao_[psi4Index]->zero();
                        }
                    }
                }
            }
        }

        return;
    }
#endif

    if (incfock_) {
        timer_on("DirectJK: INCFOCK Preprocessing");
        incfock_setup();
        int reset = options_.get_int("INCFOCK_FULL_FOCK_EVERY");
        double dconv = options_.get_double("D_CONVERGENCE");
        double Dnorm = Process::environment.globals["SCF D NORM"];
        // Do IFB on this iteration?
        do_incfock_iter_ = (Dnorm >= dconv) && !initial_iteration_ && (incfock_count_ % reset != reset - 1);
        
        if (!initial_iteration_ && (Dnorm >= dconv)) incfock_count_ += 1;
        timer_off("DirectJK: INCFOCK Preprocessing");
    }

    auto factory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
    
    std::vector<SharedMatrix>& D_ref = (do_incfock_iter_ ? delta_D_ao_ : D_ao_);
    std::vector<SharedMatrix>& J_ref = (do_incfock_iter_ ? delta_J_ao_ : J_ao_);
    std::vector<SharedMatrix>& K_ref = (do_incfock_iter_ ? delta_K_ao_ : K_ao_);
    std::vector<SharedMatrix>& wK_ref = (do_incfock_iter_ ? delta_wK_ao_ : wK_ao_);

    if (do_wK_) {
        std::vector<std::shared_ptr<TwoBodyAOInt>> ints;
        for (int thread = 0; thread < df_ints_num_threads_; thread++) {
            ints.push_back(std::shared_ptr<TwoBodyAOInt>(factory->erf_eri(omega_)));
            if (density_screening_) ints[thread]->update_density(D_ref);
        }
        // TODO: Fast K algorithm
        if (do_J_) {
            build_JK_matrices(ints, D_ref, J_ref, wK_ref, true, true, true, true);
        } else {
            build_JK_matrices(ints, D_ref, J_ref, wK_ref, false, true, true, true);
        }
    }
    
    if (do_J_ || do_K_) {
        std::vector<std::shared_ptr<TwoBodyAOInt>> ints;
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
        if (density_screening_) ints[0]->update_density(D_ref);
        for (int thread = 1; thread < df_ints_num_threads_; thread++) {
            ints.push_back(std::shared_ptr<TwoBodyAOInt>(ints[0]->clone()));
        }
        if (do_J_ && do_K_) {
            if (initial_iteration_ || !linK_) {
                build_JK_matrices(ints, D_ref, J_ref, K_ref, true, true, true, true);
            } else {
                build_linK(ints, D_ref, J_ref, K_ref);
                build_JK_matrices(ints, D_ref, J_ref, K_ref, true, false, false, false);
            }
        } else if (do_J_) {
            build_JK_matrices(ints, D_ref, J_ref, K_ref, true, false, true, false);
        } else {
            if (initial_iteration_ || !linK_) {
                build_JK_matrices(ints, D_ref, J_ref, K_ref, false, true, false, true);
            } else {
                build_linK(ints, D_ref, J_ref, K_ref);
            }
        }
    }

    if (incfock_) {
        timer_on("DirectJK: INCFOCK Postprocessing");
        incfock_postiter();
        timer_off("DirectJK: INCFOCK Postprocessing");
    }

    if (initial_iteration_) initial_iteration_ = false;
}
void DirectJK::postiterations() {}

void DirectJK::build_JK_matrices(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, const std::vector<SharedMatrix>& D,
                        std::vector<SharedMatrix>& J, std::vector<SharedMatrix>& K,
                        bool build_J, bool build_K, bool reset_J, bool reset_K) {
    
    timer_on("build_JK_matrices()");

    // Are we also performing linK on this iteration?
    bool linK_iter = linK_ && !initial_iteration_;

    // => Zeroing <= //
    if (build_J && reset_J) {
        for (auto Jmat : J) {
            Jmat->zero();
        }
    }
    
    if (build_K && reset_K) {
        for (auto Kmat : K) {
            Kmat->zero();
        }
    }
    
    // => Sizing <= //

    int nshell = primary_->nshell();
    int nthread = df_ints_num_threads_;

    // => Task Blocking <= //

    std::vector<int> task_shells;
    std::vector<int> task_starts;

    // > Atomic Blocking < //

    int atomic_ind = -1;
    for (int P = 0; P < nshell; P++) {
        if (primary_->shell(P).ncenter() > atomic_ind) {
            task_starts.push_back(P);
            atomic_ind++;
        }
        task_shells.push_back(P);
    }
    task_starts.push_back(nshell);

    // < End Atomic Blocking > //

    size_t ntask = task_starts.size() - 1;

    std::vector<int> task_offsets;
    task_offsets.push_back(0);
    for (int P2 = 0; P2 < primary_->nshell(); P2++) {
        task_offsets.push_back(task_offsets[P2] + primary_->shell(task_shells[P2]).nfunction());
    }

    size_t max_task = 0L;
    for (size_t task = 0; task < ntask; task++) {
        size_t size = 0L;
        for (int P2 = task_starts[task]; P2 < task_starts[task + 1]; P2++) {
            size += primary_->shell(task_shells[P2]).nfunction();
        }
        max_task = (max_task >= size ? max_task : size);
    }

    if (debug_) {
        outfile->Printf("  ==> DirectJK: Task Blocking <==\n\n");
        for (size_t task = 0; task < ntask; task++) {
            outfile->Printf("  Task: %3d, Task Start: %4d, Task End: %4d\n", task, task_starts[task],
                            task_starts[task + 1]);
            for (int P2 = task_starts[task]; P2 < task_starts[task + 1]; P2++) {
                int P = task_shells[P2];
                int size = primary_->shell(P).nfunction();
                int off = primary_->shell(P).function_index();
                int off2 = task_offsets[P2];
                outfile->Printf("    Index %4d, Shell: %4d, Size: %4d, Offset: %4d, Offset2: %4d\n", P2, P, size, off,
                                off2);
            }
        }
        outfile->Printf("\n");
    }

    // => Significant Task Pairs (PQ|-style <= //

    std::vector<std::pair<int, int> > task_pairs;
    for (size_t Ptask = 0; Ptask < ntask; Ptask++) {
        for (size_t Qtask = 0; Qtask < ntask; Qtask++) {
            if (Qtask > Ptask) continue;
            bool found = false;
            for (int P2 = task_starts[Ptask]; P2 < task_starts[Ptask + 1]; P2++) {
                for (int Q2 = task_starts[Qtask]; Q2 < task_starts[Qtask + 1]; Q2++) {
                    int P = task_shells[P2];
                    int Q = task_shells[Q2];
                    if (ints[0]->shell_pair_significant(P, Q)) {
                        found = true;
                        task_pairs.push_back(std::pair<int, int>(Ptask, Qtask));
                        break;
                    }
                }
                if (found) break;
            }
        }
    }
    size_t ntask_pair = task_pairs.size();
    size_t ntask_pair2 = ntask_pair * ntask_pair;

    // => Intermediate Buffers <= //

    // Intermediate buffer for J
    std::vector<std::vector<SharedMatrix>> JT;
    if (build_J) {
        for (int thread = 0; thread < nthread; thread++) {
            std::vector<SharedMatrix> J2;
            for (size_t ind = 0; ind < D.size(); ind++) {
                J2.push_back(std::make_shared<Matrix>("JT", 2 * max_task, max_task));
            }
            JT.push_back(J2);
        }
    }

    // Intermediate buffer for K
    std::vector<std::vector<SharedMatrix>> KT;
    if (build_K) {
        for (int thread = 0; thread < nthread; thread++) {
            std::vector<SharedMatrix > K2;
            for (size_t ind = 0; ind < D.size(); ind++) {
                K2.push_back(std::make_shared<Matrix>("KT", (lr_symmetric_ ? 4 : 8) * max_task, max_task));
            }
            KT.push_back(K2);
        }
    }
    
    // => Benchmarks <= //

    size_t computed_shells = 0L;

// ==> Master Task Loop <== //

#pragma omp parallel for num_threads(nthread) schedule(dynamic) reduction(+ : computed_shells)
    for (size_t task = 0L; task < ntask_pair2; task++) {
        size_t task1 = task / ntask_pair;
        size_t task2 = task % ntask_pair;

        int Ptask = task_pairs[task1].first;
        int Qtask = task_pairs[task1].second;
        int Rtask = task_pairs[task2].first;
        int Stask = task_pairs[task2].second;

        // GOTCHA! Thought this should be RStask > PQtask, but
        // H2/3-21G: Task (10|11) gives valid quartets (30|22) and (31|22)
        // This is an artifact that multiple shells on each task allow
        // for for the Ptask's index to possibly trump any RStask pair,
        // regardless of Qtask's index
        if (Rtask > Ptask) continue;

        // printf("Task: %2d %2d %2d %2d\n", Ptask, Qtask, Rtask, Stask);

        int nPtask = task_starts[Ptask + 1] - task_starts[Ptask];
        int nQtask = task_starts[Qtask + 1] - task_starts[Qtask];
        int nRtask = task_starts[Rtask + 1] - task_starts[Rtask];
        int nStask = task_starts[Stask + 1] - task_starts[Stask];

        int P2start = task_starts[Ptask];
        int Q2start = task_starts[Qtask];
        int R2start = task_starts[Rtask];
        int S2start = task_starts[Stask];

        int dPsize = task_offsets[P2start + nPtask] - task_offsets[P2start];
        int dQsize = task_offsets[Q2start + nQtask] - task_offsets[Q2start];
        int dRsize = task_offsets[R2start + nRtask] - task_offsets[R2start];
        int dSsize = task_offsets[S2start + nStask] - task_offsets[S2start];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // => Master shell quartet loops <= //

        bool touched = false;
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
                if (Q2 > P2) continue;
                int P = task_shells[P2];
                int Q = task_shells[Q2];
                if (!ints[0]->shell_pair_significant(P, Q)) continue;
                for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
                    for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                        if (S2 > R2) continue;
                        int R = task_shells[R2];
                        int S = task_shells[S2];
                        if (R2 * nshell + S2 > P2 * nshell + Q2) continue;
                        if (!ints[0]->shell_pair_significant(R, S)) continue;
                        if (!ints[0]->shell_significant(P, Q, R, S)) continue;

                        // Avoid integral recomputation in the linK case
                        if (linK_iter) {
                            eri_index index = ((eri_index) P) * nshell * nshell * nshell;
                            index += ((eri_index) Q) * nshell * nshell;
                            index += ((eri_index) R) * nshell;
                            index += (eri_index) S;
                            bool is_computed = false;

                            for (int i = 0; i < computed_integrals_.size(); i++) {
                                if (computed_integrals_[i].count(index)) {
                                    is_computed = true;
                                }
                            }
                            if (is_computed) continue;
                        }

                        // printf("Quartet: %2d %2d %2d %2d\n", P, Q, R, S);
                        // if (thread == 0) timer_on("JK: Ints");
                        if (ints[thread]->compute_shell(P, Q, R, S) == 0)
                            continue;  // No integrals in this shell quartet
                        computed_shells++;
                        // if (thread == 0) timer_off("JK: Ints");

                        const double* buffer = ints[thread]->buffer();

                        int Psize = primary_->shell(P).nfunction();
                        int Qsize = primary_->shell(Q).nfunction();
                        int Rsize = primary_->shell(R).nfunction();
                        int Ssize = primary_->shell(S).nfunction();

                        int Poff = primary_->shell(P).function_index();
                        int Qoff = primary_->shell(Q).function_index();
                        int Roff = primary_->shell(R).function_index();
                        int Soff = primary_->shell(S).function_index();

                        int Poff2 = task_offsets[P2] - task_offsets[P2start];
                        int Qoff2 = task_offsets[Q2] - task_offsets[Q2start];
                        int Roff2 = task_offsets[R2] - task_offsets[R2start];
                        int Soff2 = task_offsets[S2] - task_offsets[S2start];

                        // if (thread == 0) timer_on("JK: GEMV");
                        for (size_t ind = 0; ind < D.size(); ind++) {
                            double** Dp = D[ind]->pointer();
                            double** JTp; 
                            if (build_J) JTp = JT[thread][ind]->pointer();
                            double** KTp;
                            if (build_K) KTp = KT[thread][ind]->pointer();
                            const double* buffer2 = buffer;

                            if (!touched) {
                                if (build_J) {
                                    ::memset((void*)JTp[0L * max_task], '\0', dPsize * dQsize * sizeof(double));
                                    ::memset((void*)JTp[1L * max_task], '\0', dRsize * dSsize * sizeof(double));
                                }

                                if (build_K) {
                                    ::memset((void*)KTp[0L * max_task], '\0', dPsize * dRsize * sizeof(double));
                                    ::memset((void*)KTp[1L * max_task], '\0', dPsize * dSsize * sizeof(double));
                                    ::memset((void*)KTp[2L * max_task], '\0', dQsize * dRsize * sizeof(double));
                                    ::memset((void*)KTp[3L * max_task], '\0', dQsize * dSsize * sizeof(double));
                                    if (!lr_symmetric_) {
                                        ::memset((void*)KTp[4L * max_task], '\0', dRsize * dPsize * sizeof(double));
                                        ::memset((void*)KTp[5L * max_task], '\0', dSsize * dPsize * sizeof(double));
                                        ::memset((void*)KTp[6L * max_task], '\0', dRsize * dQsize * sizeof(double));
                                        ::memset((void*)KTp[7L * max_task], '\0', dSsize * dQsize * sizeof(double));
                                    }
                                }
                            }

                            // Intermediate Contraction Pointers
                            double* J1p;
                            double* J2p;
                            double* K1p;
                            double* K2p;
                            double* K3p;
                            double* K4p;
                            double* K5p;
                            double* K6p;
                            double* K7p;
                            double* K8p;

                            if (build_J) {
                                J1p = JTp[0L * max_task];
                                J2p = JTp[1L * max_task];
                            }

                            if (build_K) {
                                K1p = KTp[0L * max_task];
                                K2p = KTp[1L * max_task];
                                K3p = KTp[2L * max_task];
                                K4p = KTp[3L * max_task];
                                if (!lr_symmetric_) {
                                    K5p = KTp[4L * max_task];
                                    K6p = KTp[5L * max_task];
                                    K7p = KTp[6L * max_task];
                                    K8p = KTp[7L * max_task];
                                }
                            }

                            double prefactor = 1.0;
                            if (P == Q) prefactor *= 0.5;
                            if (R == S) prefactor *= 0.5;
                            if (P == R && Q == S) prefactor *= 0.5;

                            for (int p = 0; p < Psize; p++) {
                                for (int q = 0; q < Qsize; q++) {
                                    for (int r = 0; r < Rsize; r++) {
                                        for (int s = 0; s < Ssize; s++) {
                                            if (build_J) {
                                                J1p[(p + Poff2) * dQsize + q + Qoff2] +=
                                                    prefactor * (Dp[r + Roff][s + Soff] + Dp[s + Soff][r + Roff]) *
                                                    (*buffer2);
                                                J2p[(r + Roff2) * dSsize + s + Soff2] +=
                                                    prefactor * (Dp[p + Poff][q + Qoff] + Dp[q + Qoff][p + Poff]) *
                                                    (*buffer2);
                                            }

                                            if (build_K) {
                                                K1p[(p + Poff2) * dRsize + r + Roff2] +=
                                                    prefactor * (Dp[q + Qoff][s + Soff]) * (*buffer2);
                                                K2p[(p + Poff2) * dSsize + s + Soff2] +=
                                                    prefactor * (Dp[q + Qoff][r + Roff]) * (*buffer2);
                                                K3p[(q + Qoff2) * dRsize + r + Roff2] +=
                                                    prefactor * (Dp[p + Poff][s + Soff]) * (*buffer2);
                                                K4p[(q + Qoff2) * dSsize + s + Soff2] +=
                                                    prefactor * (Dp[p + Poff][r + Roff]) * (*buffer2);
                                                if (!lr_symmetric_) {
                                                    K5p[(r + Roff2) * dPsize + p + Poff2] +=
                                                        prefactor * (Dp[s + Soff][q + Qoff]) * (*buffer2);
                                                    K6p[(s + Soff2) * dPsize + p + Poff2] +=
                                                        prefactor * (Dp[r + Roff][q + Qoff]) * (*buffer2);
                                                    K7p[(r + Roff2) * dQsize + q + Qoff2] +=
                                                        prefactor * (Dp[s + Soff][p + Poff]) * (*buffer2);
                                                    K8p[(s + Soff2) * dQsize + q + Qoff2] +=
                                                        prefactor * (Dp[r + Roff][p + Poff]) * (*buffer2);
                                                }
                                            }
                                            buffer2++;
                                        }
                                    }
                                }
                            }
                        }
                        touched = true;
                        // if (thread == 0) timer_off("JK: GEMV");
                    }
                }
            }
        }  // End Shell Quartets

        if (!touched) continue;

        // => Stripe out <= //

        // if (thread == 0) timer_on("JK: Atomic");
        for (size_t ind = 0; ind < D.size(); ind++) {
            double** JTp;
            double** KTp;
            double** Jp;
            double** Kp;

            if (build_J) {
                JTp = JT[thread][ind]->pointer();
                Jp = J[ind]->pointer();
            }
            
            if (build_K) {
                KTp = KT[thread][ind]->pointer();
                Kp = K[ind]->pointer();
            }

            double* J1p;
            double* J2p;
            double* K1p;
            double* K2p;
            double* K3p;
            double* K4p;
            double* K5p;
            double* K6p;
            double* K7p;
            double* K8p;

            if (build_J) {
                J1p = JTp[0L * max_task];
                J2p = JTp[1L * max_task];
            }

            if (build_K) {
                K1p = KTp[0L * max_task];
                K2p = KTp[1L * max_task];
                K3p = KTp[2L * max_task];
                K4p = KTp[3L * max_task];
                if (!lr_symmetric_) {
                    K5p = KTp[4L * max_task];
                    K6p = KTp[5L * max_task];
                    K7p = KTp[6L * max_task];
                    K8p = KTp[7L * max_task];
                }
            }

            if (build_J) {

                // > J_PQ < //

                for (int P2 = 0; P2 < nPtask; P2++) {
                    for (int Q2 = 0; Q2 < nQtask; Q2++) {
                        int P = task_shells[P2start + P2];
                        int Q = task_shells[Q2start + Q2];
                        int Psize = primary_->shell(P).nfunction();
                        int Qsize = primary_->shell(Q).nfunction();
                        int Poff = primary_->shell(P).function_index();
                        int Qoff = primary_->shell(Q).function_index();
                        int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                        int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                        for (int p = 0; p < Psize; p++) {
                            for (int q = 0; q < Qsize; q++) {
#pragma omp atomic
                                Jp[p + Poff][q + Qoff] += J1p[(p + Poff2) * dQsize + q + Qoff2];
                            }
                        }
                    }
                }

                // > J_RS < //

                for (int R2 = 0; R2 < nRtask; R2++) {
                    for (int S2 = 0; S2 < nStask; S2++) {
                        int R = task_shells[R2start + R2];
                        int S = task_shells[S2start + S2];
                        int Rsize = primary_->shell(R).nfunction();
                        int Ssize = primary_->shell(S).nfunction();
                        int Roff = primary_->shell(R).function_index();
                        int Soff = primary_->shell(S).function_index();
                        int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                        int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                        for (int r = 0; r < Rsize; r++) {
                            for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                                Jp[r + Roff][s + Soff] += J2p[(r + Roff2) * dSsize + s + Soff2];
                            }
                        }
                    }
                }
            }

            if (build_K) {

                // > K_PR < //

                for (int P2 = 0; P2 < nPtask; P2++) {
                    for (int R2 = 0; R2 < nRtask; R2++) {
                        int P = task_shells[P2start + P2];
                        int R = task_shells[R2start + R2];
                        int Psize = primary_->shell(P).nfunction();
                        int Rsize = primary_->shell(R).nfunction();
                        int Poff = primary_->shell(P).function_index();
                        int Roff = primary_->shell(R).function_index();
                        int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                        int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                        for (int p = 0; p < Psize; p++) {
                            for (int r = 0; r < Rsize; r++) {
#pragma omp atomic
                                Kp[p + Poff][r + Roff] += K1p[(p + Poff2) * dRsize + r + Roff2];
                                if (!lr_symmetric_) {
#pragma omp atomic
                                    Kp[r + Roff][p + Poff] += K5p[(r + Roff2) * dPsize + p + Poff2];
                                }
                            }
                        }
                    }
                }

                // > K_PS < //

                for (int P2 = 0; P2 < nPtask; P2++) {
                    for (int S2 = 0; S2 < nStask; S2++) {
                        int P = task_shells[P2start + P2];
                        int S = task_shells[S2start + S2];
                        int Psize = primary_->shell(P).nfunction();
                        int Ssize = primary_->shell(S).nfunction();
                        int Poff = primary_->shell(P).function_index();
                        int Soff = primary_->shell(S).function_index();
                        int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                        int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                        for (int p = 0; p < Psize; p++) {
                            for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                                Kp[p + Poff][s + Soff] += K2p[(p + Poff2) * dSsize + s + Soff2];
                                if (!lr_symmetric_) {
#pragma omp atomic
                                    Kp[s + Soff][p + Poff] += K6p[(s + Soff2) * dPsize + p + Poff2];
                                }
                            }
                        }
                    }
                }

                // > K_QR < //

                for (int Q2 = 0; Q2 < nQtask; Q2++) {
                    for (int R2 = 0; R2 < nRtask; R2++) {
                        int Q = task_shells[Q2start + Q2];
                        int R = task_shells[R2start + R2];
                        int Qsize = primary_->shell(Q).nfunction();
                        int Rsize = primary_->shell(R).nfunction();
                        int Qoff = primary_->shell(Q).function_index();
                        int Roff = primary_->shell(R).function_index();
                        int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                        int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                        for (int q = 0; q < Qsize; q++) {
                            for (int r = 0; r < Rsize; r++) {
#pragma omp atomic
                                Kp[q + Qoff][r + Roff] += K3p[(q + Qoff2) * dRsize + r + Roff2];
                                if (!lr_symmetric_) {
#pragma omp atomic
                                    Kp[r + Roff][q + Qoff] += K7p[(r + Roff2) * dQsize + q + Qoff2];
                                }
                            }
                        }
                    }
                }

                // > K_QS < //

                for (int Q2 = 0; Q2 < nQtask; Q2++) {
                    for (int S2 = 0; S2 < nStask; S2++) {
                        int Q = task_shells[Q2start + Q2];
                        int S = task_shells[S2start + S2];
                        int Qsize = primary_->shell(Q).nfunction();
                        int Ssize = primary_->shell(S).nfunction();
                        int Qoff = primary_->shell(Q).function_index();
                        int Soff = primary_->shell(S).function_index();
                        int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                        int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                        for (int q = 0; q < Qsize; q++) {
                            for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                                Kp[q + Qoff][s + Soff] += K4p[(q + Qoff2) * dSsize + s + Soff2];
                                if (!lr_symmetric_) {
#pragma omp atomic
                                    Kp[s + Soff][q + Qoff] += K8p[(s + Soff2) * dQsize + q + Qoff2];
                                }
                            }
                        }
                    }
                }
            }

        }  // End stripe out
        // if (thread == 0) timer_off("JK: Atomic");

    }  // End master task list

    if (build_J) {
        for (auto Jmat : J) {
            Jmat->scale(2.0);
            Jmat->hermitivitize();
        }
    }

    if (build_K && lr_symmetric_) {
        for (auto Kmat : K) {
            Kmat->scale(2.0);
            Kmat->hermitivitize();
        }
    }

    if (bench_) {
        auto mode = std::ostream::app;
        auto printer = PsiOutStream("bench.dat", mode);
        size_t ntri = nshell * (nshell + 1L) / 2L;
        size_t possible_shells = ntri * (ntri + 1L) / 2L;
        printer.Printf("Computed %20zu Shell Quartets out of %20zu, (%11.3E ratio)\n", computed_shells,
                        possible_shells, computed_shells / (double)possible_shells);
    }

    for (int i = 0; i < computed_integrals_.size(); i++) {
        computed_integrals_[i].clear();
    }

    timer_off("build_JK_matrices()");
}

bool linK_sort_helper(const std::tuple<int, double>& t1, const std::tuple<int, double>& t2) {
    return std::get<1>(t1) > std::get<1>(t2);
}

void DirectJK::build_linK(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, const std::vector<SharedMatrix>& D,
                  std::vector<SharedMatrix>& J, std::vector<SharedMatrix>& K) {

    timer_on("build_linK()");

    // => Zeroing <= //

    for (auto Jmat : J) {
        Jmat->zero();
    }

    for (auto Kmat : K) {
        Kmat->zero();
    }

    // => Sizing <= //

    int nshell = primary_->nshell();
    int nthread = df_ints_num_threads_;

    // => Setting up computed integrals map <= //
    computed_integrals_.resize(nthread);

    // => Task Blocking <= //

    std::vector<int> task_shells;
    std::vector<int> task_starts;

    // => Atomic Blocking <= //

    int atomic_ind = -1;
    for (int P = 0; P < nshell; P++) {
        if (primary_->shell(P).ncenter() > atomic_ind) {
            task_starts.push_back(P);
            atomic_ind++;
        }
        task_shells.push_back(P);
    }
    task_starts.push_back(nshell);

    // => End Atomic Blocking <= //

    size_t ntask = task_starts.size() - 1;

    std::vector<int> task_offsets;
    task_offsets.push_back(0);
    for (int P2 = 0; P2 < primary_->nshell(); P2++) {
        task_offsets.push_back(task_offsets[P2] + primary_->shell(task_shells[P2]).nfunction());
    }

    size_t max_task = 0L;
    for (size_t task = 0; task < ntask; task++) {
        size_t size = 0L;
        for (int P2 = 0; P2 < task_starts[task + 1]; P2++) {
            size += primary_->shell(task_shells[P2]).nfunction();
        }
        max_task = (max_task >= size ? max_task : size);
    }

    if (debug_) {
        outfile->Printf("  ==> DirectJK: Task Blocking <==\n\n");
        for (size_t task = 0; task < ntask; task++) {
            outfile->Printf("  Task: %3d, Task Start: %4d, Task End: %4d\n", task, task_starts[task],
                            task_starts[task + 1]);
            for (int P2 = task_starts[task]; P2 < task_starts[task + 1]; P2++) {
                int P = task_shells[P2];
                int size = primary_->shell(P).nfunction();
                int off = primary_->shell(P).function_index();
                int off2 = task_offsets[P2];
                outfile->Printf("    Index %4d, Shell: %4d, Size: %4d, Offset: %4d, Offset2: %4d\n", P2, P, size, off,
                                off2);
            }
        }
        outfile->Printf("\n");
    }

    // => Significant Task Pairs (PQ|-style <= //

    std::vector<std::pair<int, int>> task_pairs;
    for (size_t Ptask = 0; Ptask < ntask; Ptask++) {
        for (size_t Qtask = 0; Qtask < ntask; Qtask++) {
            if (Qtask > Ptask) continue;
            bool found = false;
            for (int P2 = task_starts[Ptask]; P2 < task_starts[Ptask + 1]; P2++) {
                for (int Q2 = task_starts[Qtask]; Q2 < task_starts[Qtask + 1]; Q2++) {
                    int P = task_shells[P2];
                    int Q = task_shells[Q2];
                    if (ints[0]->shell_pair_significant(P, Q)) {
                        found = true;
                        task_pairs.push_back(std::pair<int, int>(Ptask, Qtask));
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    size_t ntask_pair = task_pairs.size();
    size_t ntask_pair2 = ntask_pair * ntask_pair;

    // => Intermediate Buffers <= //

    std::vector<std::vector<SharedMatrix>> JT;
    for (int thread = 0; thread < nthread; thread++) {
        std::vector<SharedMatrix> J2;
        for (size_t ind = 0; ind < D.size(); ind++) {
            J2.push_back(std::make_shared<Matrix>("JT (linK)", 2 * max_task, max_task));
        }
        JT.push_back(J2);
    }

    std::vector<std::vector<SharedMatrix>> KT;
    for (int thread = 0; thread < nthread; thread++) {
        std::vector<SharedMatrix> K2;
        for (size_t ind = 0; ind < D.size(); ind++) {
            K2.push_back(std::make_shared<Matrix>("KT (linK)", (lr_symmetric_ ? 4 : 8) * max_task, max_task));
        }
        KT.push_back(K2);
    }

    // => Benchmarks <= //

    size_t computed_shells = 0L;

// ==> Master Task Loop <== //

#pragma omp parallel for num_threads(nthread) schedule(dynamic) reduction(+ : computed_shells)
    for (size_t task = 0L; task < ntask_pair2; task++) {
        size_t task1 = task / ntask_pair;
        size_t task2 = task % ntask_pair;

        int Ptask = task_pairs[task1].first;
        int Qtask = task_pairs[task1].second;
        int Rtask = task_pairs[task2].first;
        int Stask = task_pairs[task2].second;

        // GOTCHA! Thought this should be RStask > PQtask, but
        // H2/3-21G: Task (10|11) gives valid quartets (30|22) and (31|22)
        // This is an artifact that multiple shells on each task allow
        // for for the Ptask's index to possibly trump any RStask pair,
        // regardless of Qtask's index
        if (Rtask > Ptask) continue;

        // printf("Task: %2d %2d %2d %2d\n", Ptask, Qtask, Rtask, Stask);

        int nPtask = task_starts[Ptask + 1] - task_starts[Ptask];
        int nQtask = task_starts[Qtask + 1] - task_starts[Qtask];
        int nRtask = task_starts[Rtask + 1] - task_starts[Rtask];
        int nStask = task_starts[Stask + 1] - task_starts[Stask];

        int P2start = task_starts[Ptask];
        int Q2start = task_starts[Qtask];
        int R2start = task_starts[Rtask];
        int S2start = task_starts[Stask];

        int Pstart = task_shells[P2start];
        int Qstart = task_shells[Q2start];
        int Rstart = task_shells[R2start];
        int Sstart = task_shells[S2start];

        // Number of basis functions per atom
        int dPsize = task_offsets[P2start + nPtask] - task_offsets[P2start];
        int dQsize = task_offsets[Q2start + nQtask] - task_offsets[Q2start];
        int dRsize = task_offsets[R2start + nRtask] - task_offsets[R2start];
        int dSsize = task_offsets[S2start + nStask] - task_offsets[S2start];

        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // Used in the first pre-screening iteration loops
        std::vector<std::vector<int>> PR_sig_shells(nPtask);
        std::vector<std::vector<int>> PS_sig_shells(nPtask);
        std::vector<std::vector<int>> QR_sig_shells(nQtask);
        std::vector<std::vector<int>> QS_sig_shells(nQtask);

        // Shell ceilings are max value of (U*|U*) for shell U
        std::vector<double> shell_ceiling_P(nPtask, 0.0);
        std::vector<double> shell_ceiling_Q(nQtask, 0.0);
        std::vector<double> shell_ceiling_R(nRtask, 0.0);
        std::vector<double> shell_ceiling_S(nStask, 0.0);

        // P and Q shell ceilings max(P*|P*) and max(Q*|Q*)
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            int P = task_shells[P2];
            int dP = P - Pstart;
            for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
                int Q = task_shells[Q2];
                int dQ = Q - Qstart;
                double val = std::sqrt(ints[0]->shell_ceiling2(P, Q, P, Q));
                shell_ceiling_P[dP] = std::max(shell_ceiling_P[dP], val);
                shell_ceiling_Q[dQ] = std::max(shell_ceiling_Q[dQ], val);
            }
        }

        // R and S shell ceilings max(R*|R*) and max(S*|S*)
        for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
            int R = task_shells[R2];
            int dR = R - Rstart;
            for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                int S = task_shells[S2];
                int dS = S - Sstart;
                double val = std::sqrt(ints[0]->shell_ceiling2(R, S, R, S));
                shell_ceiling_R[dR] = std::max(shell_ceiling_R[dR], val);
                shell_ceiling_S[dS] = std::max(shell_ceiling_S[dS], val);
            }
        }

        // PR significant shells (joined by density matrix)
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            int P = task_shells[P2];
            int dP = P - Pstart;
            std::vector<std::tuple<int, double>> temp;
            for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
                int R = task_shells[R2];
                int dR = R - Rstart;
                double D_max = std::max(D_max, ints[0]->shell_pair_max_density(P, R));
                double screen_val = D_max * shell_ceiling_P[dP] * shell_ceiling_R[dR];
                if (screen_val >= linK_ints_cutoff_) {
                    temp.push_back(std::tuple<int, double>(R, screen_val));
                }
            }

            std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &shell = temp[ind];
                PR_sig_shells[dP].push_back(std::get<0>(shell));
            }
        }

        // PS significant shells
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            int P = task_shells[P2];
            int dP = P - Pstart;
            std::vector<std::tuple<int, double>> temp;
            for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                int S = task_shells[S2];
                int dS = S - Sstart;
                double D_max = std::max(D_max, ints[0]->shell_pair_max_density(P, S));
                double screen_val = D_max * shell_ceiling_P[dP] * shell_ceiling_S[dS];
                if (screen_val > linK_ints_cutoff_) {
                    temp.push_back(std::tuple<int, double>(S, screen_val));
                }
            }

            std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &shell = temp[ind];
                PS_sig_shells[dP].push_back(std::get<0>(shell));
            }
        }

        // QR significant shells
        for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
            int Q = task_shells[Q2];
            int dQ = Q - Qstart;
            std::vector<std::tuple<int, double>> temp;
            for (int R2 = R2start; R2 < R2start + nRtask; R2++) {
                int R = task_shells[R2];
                int dR = R - Rstart;
                double D_max = std::max(D_max, ints[0]->shell_pair_max_density(Q, R));
                double screen_val = D_max * shell_ceiling_Q[dQ] * shell_ceiling_R[dR];
                if (screen_val > linK_ints_cutoff_) {
                    temp.push_back(std::tuple<int, double>(R, screen_val));
                }
            }

            std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &shell = temp[ind];
                QR_sig_shells[dQ].push_back(std::get<0>(shell));
            }
        }

        // QS significant shells
        for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {
            int Q = task_shells[Q2];
            int dQ = Q - Qstart;
            std::vector<std::tuple<int, double>> temp;
            for (int S2 = S2start; S2 < S2start + nStask; S2++) {
                int S = task_shells[S2];
                int dS = S - Sstart;
                double D_max = std::max(D_max, ints[0]->shell_pair_max_density(Q, S));
                double screen_val = D_max * shell_ceiling_Q[dQ] * shell_ceiling_S[dS];
                if (screen_val > linK_ints_cutoff_) {
                    temp.push_back(std::tuple<int, double>(S, screen_val));
                }
            }

            std::sort(&(temp[0]), &(temp[0]) + temp.size(), linK_sort_helper);

            for (int ind = 0; ind < temp.size(); ind++) {
                const std::tuple<int, double> &shell = temp[ind];
                QS_sig_shells[dQ].push_back(std::get<0>(shell));
            }
        }

        // Used in the stripeout of J
        std::vector<std::unordered_set<int>> PQ_stripeout(nPtask);
        std::vector<std::unordered_set<int>> RS_stripeout(nRtask);

        // Used in the second pre-screening iteration loop (and later used in the stripeout)
        std::vector<std::unordered_set<int>> PR_stripeout(nPtask);
        std::vector<std::unordered_set<int>> PS_stripeout(nPtask);
        std::vector<std::unordered_set<int>> QR_stripeout(nQtask);
        std::vector<std::unordered_set<int>> QS_stripeout(nQtask);

        bool touched = false;
        for (int P2 = P2start; P2 < P2start + nPtask; P2++) {
            for (int Q2 = Q2start; Q2 < Q2start + nQtask; Q2++) {

                if (Q2 > P2) continue;

                int P = task_shells[P2];
                int Q = task_shells[Q2];

                int dP = P - Pstart;
                int dQ = Q - Qstart;

                // Significant ket shell pairs RS for bra shell pair PQ
                std::unordered_set<int> PQ_sig_RS;

                // Form ML_P (Oschenfeld Fig. 1)
                for (int R2 = 0; R2 < PR_sig_shells[dP].size(); R2++) {
                    int count = 0;
                    for (int S2 = 0; S2 < PS_sig_shells[dP].size(); S2++) {
                        int R = PR_sig_shells[dP][R2];
                        int S = PS_sig_shells[dP][S2];
                        int dR = R - Rstart;
                        int dS = S - Sstart;

                        double screen_val = ints[0]->shell_pair_max_density(P, R) * std::sqrt(ints[0]->shell_ceiling2(P, Q, R, S));

                        if (screen_val >= linK_ints_cutoff_) {
                            count += 1;
                            if (S > R) continue;
                            if (R * nshell + S > P * nshell + Q) continue;

                            // Bra Stripeouts
                            if (!PQ_stripeout[dP].count(Q)) PQ_stripeout[dP].emplace(Q);
                            if (!RS_stripeout[dR].count(S)) RS_stripeout[dR].emplace(S);

                            // Ket Stripeouts
                            if (!PR_stripeout[dP].count(R)) PR_stripeout[dP].emplace(R);
                            if (!PS_stripeout[dP].count(S)) PS_stripeout[dP].emplace(S);

                            if (PQ_sig_RS.count(R * nshell + S)) continue;
                            PQ_sig_RS.emplace(R * nshell + S);
                        }
                        else break;
                    }
                    if (count == 0) break;
                }

                // Form ML_Q
                for (int R2 = 0; R2 < QR_sig_shells[dQ].size(); R2++) {
                    int count = 0;
                    for (int S2 = 0; S2 < QS_sig_shells[dQ].size(); S2++) {
                        int R = QR_sig_shells[dQ][R2];
                        int S = QS_sig_shells[dQ][S2];
                        int dR = R - Rstart;
                        int dS = S - Sstart;

                        double screen_val = ints[0]->shell_pair_max_density(Q, R) * std::sqrt(ints[0]->shell_ceiling2(P, Q, R, S));

                        if (screen_val >= linK_ints_cutoff_) {
                            count += 1;
                            if (S > R) continue;
                            if (R * nshell + S > P * nshell + Q) continue;

                            // Bra Stripeouts
                            if (!PQ_stripeout[dP].count(Q)) PQ_stripeout[dP].emplace(Q);
                            if (!RS_stripeout[dR].count(S)) RS_stripeout[dR].emplace(S);

                            // Ket Stripeouts
                            if (!QR_stripeout[dQ].count(R)) QR_stripeout[dQ].emplace(R);
                            if (!QS_stripeout[dQ].count(S)) QS_stripeout[dQ].emplace(S);

                            if (PQ_sig_RS.count(R * nshell + S)) continue;
                            PQ_sig_RS.emplace(R * nshell + S);
                        }
                        else break;
                    }
                    if (count == 0) break;
                }

                // Loop over significant RS pairs
                for (const int RS2 : PQ_sig_RS) {

                    int R = RS2 / nshell;
                    int S = RS2 % nshell;
                    int R2 = R;
                    int S2 = S;

                    if (S > R) continue;
                    if (R * nshell + S > P * nshell + Q) continue;

                    if (!ints[0]->shell_pair_significant(R, S)) continue;
                    if (!ints[0]->shell_significant(P, Q, R, S)) continue;

                    // printf("Quartet: %2d %2d %2d %2d\n", P, Q, R, S);
                    // timer_on("compute_shell(P, Q, R, S)");
                    // if (thread == 0) timer_on("JK: Ints");
                    if (ints[thread]->compute_shell(P, Q, R, S) == 0)
                        continue;  // No integrals in this shell quartet
                    computed_shells++;
                    // if (thread == 0) timer_off("JK: Ints");
                    // timer_off("compute_shell(P, Q, R, S)");

                    // Adding computed integrals to buffer
                    eri_index index = ((eri_index) P) * nshell * nshell * nshell;
                    index += ((eri_index) Q) * nshell * nshell;
                    index += ((eri_index) R) * nshell;
                    index += (eri_index) S;
                    computed_integrals_[thread].emplace(index);

                    const double* buffer = ints[thread]->buffer();

                    int Psize = primary_->shell(P).nfunction();
                    int Qsize = primary_->shell(Q).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Ssize = primary_->shell(S).nfunction();

                    int Poff = primary_->shell(P).function_index();
                    int Qoff = primary_->shell(Q).function_index();
                    int Roff = primary_->shell(R).function_index();
                    int Soff = primary_->shell(S).function_index();

                    int Poff2 = task_offsets[P2] - task_offsets[P2start];
                    int Qoff2 = task_offsets[Q2] - task_offsets[Q2start];
                    int Roff2 = task_offsets[R2] - task_offsets[R2start];
                    int Soff2 = task_offsets[S2] - task_offsets[S2start];

                    // if (thread == 0) timer_on("JK: GEMV");
                    for (size_t ind = 0; ind < D.size(); ind++) {
                        double** Dp = D[ind]->pointer();
                        double** JTp = JT[thread][ind]->pointer();
                        double** KTp = KT[thread][ind]->pointer();
                        const double* buffer2 = buffer;

                        if (!touched) {
                            ::memset((void*)JTp[0L * max_task], '\0', dPsize * dQsize * sizeof(double));
                            ::memset((void*)JTp[1L * max_task], '\0', dRsize * dSsize * sizeof(double));
                            ::memset((void*)KTp[0L * max_task], '\0', dPsize * dRsize * sizeof(double));
                            ::memset((void*)KTp[1L * max_task], '\0', dPsize * dSsize * sizeof(double));
                            ::memset((void*)KTp[2L * max_task], '\0', dQsize * dRsize * sizeof(double));
                            ::memset((void*)KTp[3L * max_task], '\0', dQsize * dSsize * sizeof(double));
                            if (!lr_symmetric_) {
                                ::memset((void*)KTp[4L * max_task], '\0', dRsize * dPsize * sizeof(double));
                                ::memset((void*)KTp[5L * max_task], '\0', dSsize * dPsize * sizeof(double));
                                ::memset((void*)KTp[6L * max_task], '\0', dRsize * dQsize * sizeof(double));
                                ::memset((void*)KTp[7L * max_task], '\0', dSsize * dQsize * sizeof(double));
                            }
                        }

                        double* J1p = JTp[0L * max_task];
                        double* J2p = JTp[1L * max_task];
                        double* K1p = KTp[0L * max_task];
                        double* K2p = KTp[1L * max_task];
                        double* K3p = KTp[2L * max_task];
                        double* K4p = KTp[3L * max_task];
                        double* K5p;
                        double* K6p;
                        double* K7p;
                        double* K8p;
                        if (!lr_symmetric_) {
                            K5p = KTp[4L * max_task];
                            K6p = KTp[5L * max_task];
                            K7p = KTp[6L * max_task];
                            K8p = KTp[7L * max_task];
                        }

                        double prefactor = 1.0;
                        if (P == Q) prefactor *= 0.5;
                        if (R == S) prefactor *= 0.5;
                        if (P == R && Q == S) prefactor *= 0.5;

                        for (int p = 0; p < Psize; p++) {
                            for (int q = 0; q < Qsize; q++) {
                                for (int r = 0; r < Rsize; r++) {
                                    for (int s = 0; s < Ssize; s++) {

                                        J1p[(p + Poff2) * dQsize + q + Qoff2] +=
                                            prefactor * (Dp[r + Roff][s + Soff] + Dp[s + Soff][r + Roff]) *
                                            (*buffer2);
                                        J2p[(r + Roff2) * dSsize + s + Soff2] +=
                                            prefactor * (Dp[p + Poff][q + Qoff] + Dp[q + Qoff][p + Poff]) *
                                            (*buffer2);
                                        K1p[(p + Poff2) * dRsize + r + Roff2] +=
                                            prefactor * (Dp[q + Qoff][s + Soff]) * (*buffer2);
                                        K2p[(p + Poff2) * dSsize + s + Soff2] +=
                                            prefactor * (Dp[q + Qoff][r + Roff]) * (*buffer2);
                                        K3p[(q + Qoff2) * dRsize + r + Roff2] +=
                                            prefactor * (Dp[p + Poff][s + Soff]) * (*buffer2);
                                        K4p[(q + Qoff2) * dSsize + s + Soff2] +=
                                            prefactor * (Dp[p + Poff][r + Roff]) * (*buffer2);
                                        if (!lr_symmetric_) {
                                            K5p[(r + Roff2) * dPsize + p + Poff2] +=
                                                prefactor * (Dp[s + Soff][q + Qoff]) * (*buffer2);
                                            K6p[(s + Soff2) * dPsize + p + Poff2] +=
                                                prefactor * (Dp[r + Roff][q + Qoff]) * (*buffer2);
                                            K7p[(r + Roff2) * dQsize + q + Qoff2] +=
                                                prefactor * (Dp[s + Soff][p + Poff]) * (*buffer2);
                                            K8p[(s + Soff2) * dQsize + q + Qoff2] +=
                                                prefactor * (Dp[r + Roff][p + Poff]) * (*buffer2);
                                        }

                                        buffer2++;
                                    }
                                }
                            }
                        }
                    }
                    touched = true;
                // if (thread == 0) timer_off("JK: GEMV");
                }
            }
        }

        // => Master shell quartet loops <= //

        if (!touched) continue;

        // => Stripe out <= //

        // if (thread == 0) timer_on("JK: Atomic");
        for (size_t ind = 0; ind < D.size(); ind++) {
            double** JTp = JT[thread][ind]->pointer();
            double** Jp = J[ind]->pointer();
            double** KTp = KT[thread][ind]->pointer();
            double** Kp = K[ind]->pointer();

            double* J1p = JTp[0L * max_task];
            double* J2p = JTp[1L * max_task];
            double* K1p = KTp[0L * max_task];
            double* K2p = KTp[1L * max_task];
            double* K3p = KTp[2L * max_task];
            double* K4p = KTp[3L * max_task];
            double* K5p;
            double* K6p;
            double* K7p;
            double* K8p;
            if (!lr_symmetric_) {
                K5p = KTp[4L * max_task];
                K6p = KTp[5L * max_task];
                K7p = KTp[6L * max_task];
                K8p = KTp[7L * max_task];
            }
                
            // > J_PQ < //

            for (int P2 = 0; P2 < nPtask; P2++) {
                for (const int Q : PQ_stripeout[P2]) {
                    int P = task_shells[P2start + P2];
                    int Q2 = Q - Qstart;

                    int Psize = primary_->shell(P).nfunction();
                    int Qsize = primary_->shell(Q).nfunction();
                    int Poff = primary_->shell(P).function_index();
                    int Qoff = primary_->shell(Q).function_index();
                    int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                    int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                    for (int p = 0; p < Psize; p++) {
                        for (int q = 0; q < Qsize; q++) {
#pragma omp atomic
                            Jp[p + Poff][q + Qoff] += J1p[(p + Poff2) * dQsize + q + Qoff2];
                        }
                    }
                }
            }

            // > J_RS < //

            for (int R2 = 0; R2 < nRtask; R2++) {
                for (const int S : RS_stripeout[R2]) {
                    int R = task_shells[R2start + R2];
                    int S2 = S - Sstart;

                    int Rsize = primary_->shell(R).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Roff = primary_->shell(R).function_index();
                    int Soff = primary_->shell(S).function_index();
                    int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                    int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                    for (int r = 0; r < Rsize; r++) {
                        for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                            Jp[r + Roff][s + Soff] += J2p[(r + Roff2) * dSsize + s + Soff2];
                        }
                    }
                }
            }

            // > K_PR < //

            for (int P2 = 0; P2 < nPtask; P2++) {
                for (const int R : PR_stripeout[P2]) {
                    int P = task_shells[P2start + P2];
                    // int R = task_shells[R2start + R2];
                    int R2 = R - Rstart;

                    int Psize = primary_->shell(P).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Poff = primary_->shell(P).function_index();
                    int Roff = primary_->shell(R).function_index();

                    int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                    int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                    for (int p = 0; p < Psize; p++) {
                        for (int r = 0; r < Rsize; r++) {
#pragma omp atomic
                            Kp[p + Poff][r + Roff] += K1p[(p + Poff2) * dRsize + r + Roff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[r + Roff][p + Poff] += K5p[(r + Roff2) * dPsize + p + Poff2];
                            }
                        }
                    }
                }
            }

            // > K_PS < //

            for (int P2 = 0; P2 < nPtask; P2++) {
                for (const int S : PS_stripeout[P2]) {
                    int P = task_shells[P2start + P2];
                    // int S = task_shells[S2start + S2];
                    int S2 = S - Sstart;

                    int Psize = primary_->shell(P).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Poff = primary_->shell(P).function_index();
                    int Soff = primary_->shell(S).function_index();

                    int Poff2 = task_offsets[P2 + P2start] - task_offsets[P2start];
                    int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                    for (int p = 0; p < Psize; p++) {
                        for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                            Kp[p + Poff][s + Soff] += K2p[(p + Poff2) * dSsize + s + Soff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[s + Soff][p + Poff] += K6p[(s + Soff2) * dPsize + p + Poff2];
                            }
                        }
                    }
                }
            }

            // > K_QR < //

            for (int Q2 = 0; Q2 < nQtask; Q2++) {
                for (const int R : QR_stripeout[Q2]) {
                    int Q = task_shells[Q2start + Q2];
                    // int R = task_shells[R2start + R2];
                    int R2 = R - Rstart;

                    int Qsize = primary_->shell(Q).nfunction();
                    int Rsize = primary_->shell(R).nfunction();
                    int Qoff = primary_->shell(Q).function_index();
                    int Roff = primary_->shell(R).function_index();

                    int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                    int Roff2 = task_offsets[R2 + R2start] - task_offsets[R2start];
                    for (int q = 0; q < Qsize; q++) {
                        for (int r = 0; r < Rsize; r++) {
#pragma omp atomic
                            Kp[q + Qoff][r + Roff] += K3p[(q + Qoff2) * dRsize + r + Roff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[r + Roff][q + Qoff] += K7p[(r + Roff2) * dQsize + q + Qoff2];
                            }
                        }
                    }
                }
            }

            // > K_QS < //

            for (int Q2 = 0; Q2 < nQtask; Q2++) {
                for (const int S : QS_stripeout[Q2]) {
                    int Q = task_shells[Q2start + Q2];
                    // int S = task_shells[S2start + S2];
                    int S2 = S - Sstart;

                    int Qsize = primary_->shell(Q).nfunction();
                    int Ssize = primary_->shell(S).nfunction();
                    int Qoff = primary_->shell(Q).function_index();
                    int Soff = primary_->shell(S).function_index();

                    int Qoff2 = task_offsets[Q2 + Q2start] - task_offsets[Q2start];
                    int Soff2 = task_offsets[S2 + S2start] - task_offsets[S2start];
                    for (int q = 0; q < Qsize; q++) {
                        for (int s = 0; s < Ssize; s++) {
#pragma omp atomic
                            Kp[q + Qoff][s + Soff] += K4p[(q + Qoff2) * dSsize + s + Soff2];
                            if (!lr_symmetric_) {
#pragma omp atomic
                                Kp[s + Soff][q + Qoff] += K8p[(s + Soff2) * dQsize + q + Qoff2];
                            }
                        }
                    }
                }
            }

        }  // End stripe out
        // if (thread == 0) timer_off("JK: Atomic");

    }  // End master task list

    if (lr_symmetric_) {
        for (auto Kmat : K) {
            Kmat->scale(2.0);
            Kmat->hermitivitize();
        }
    }

    if (bench_) {
        auto mode = std::ostream::app;
        auto printer = PsiOutStream("bench.dat", mode);
        size_t ntri = nshell * (nshell + 1L) / 2L;
        size_t possible_shells = ntri * (ntri + 1L) / 2L;
        printer.Printf("(linK) Computed %20zu Shell Quartets out of %20zu, (%11.3E ratio)\n", computed_shells,
                        possible_shells, computed_shells / (double)possible_shells);
    }
    timer_off("build_linK()");
}

}  // namespace psi
