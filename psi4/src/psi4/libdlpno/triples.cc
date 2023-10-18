/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
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

#include "dlpno.h"
#include "sparse.h"

#include "psi4/lib3index/3index.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/local.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/orthog.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"

#include <ctime>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {
namespace dlpno {

DLPNOCCSD_T::DLPNOCCSD_T(SharedWavefunction ref_wfn, Options &options) : DLPNOCCSD(ref_wfn, options) {}
DLPNOCCSD_T::~DLPNOCCSD_T() {}

void DLPNOCCSD_T::print_header() {
    std::string triples_algorithm = (options_.get_bool("T0_APPROXIMATION")) ? "SEMICANONICAL (T0)" : "ITERATIVE (T)";
    std::string scale_t0 = (options_.get_bool("SCALE_T0") ? "TRUE" : "FALSE");
    double t_cut_tno_pre = options_.get_double("T_CUT_TNO_PRE");
    double t_cut_tno = options_.get_double("T_CUT_TNO");
    double t_cut_tno_strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
    double t_cut_tno_weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");

    outfile->Printf("   --------------------------------------------\n");
    outfile->Printf("                    DLPNO-CCSD(T)              \n");
    outfile->Printf("                    by Andy Jiang              \n");
    outfile->Printf("   --------------------------------------------\n\n");
    outfile->Printf("  DLPNO convergence set to %s.\n\n", options_.get_str("PNO_CONVERGENCE").c_str());
    outfile->Printf("  Detailed DLPNO thresholds and cutoffs:\n");
    outfile->Printf("    ALGORITHM    = %6s   \n", triples_algorithm.c_str());
    outfile->Printf("    T_CUT_TNO_PRE (T0)   = %6.3e \n", t_cut_tno_pre);
    outfile->Printf("    T_CUT_TNO (T0)       = %6.3e \n", t_cut_tno);
    outfile->Printf("    T_CUT_TNO_STRONG (T) = %6.3e \n", t_cut_tno * t_cut_tno_strong_scale);
    outfile->Printf("    T_CUT_TNO_WEAK (T)   = %6.3e \n", t_cut_tno * t_cut_tno_weak_scale);
    outfile->Printf("    F_CUT_T      = %6.3e \n", options_.get_double("F_CUT_T"));
    outfile->Printf("    T0_SCALING?  = %6s   \n\n", scale_t0.c_str());
    outfile->Printf("\n");
}

SharedMatrix matmul_3d(SharedMatrix A, SharedMatrix X, int dim_old, int dim_new) {
    /*
    Performs the operation A'[i,j,k] = A[I,J,K] * X[i,I] * X[j,J] * X[k,K] for cube 3d tensors
    */

    SharedMatrix A_new = linalg::doublet(X, A, false, false);
    A_new->reshape(dim_new * dim_old, dim_old);
    A_new = linalg::doublet(A_new, X, false, true);

    SharedMatrix A_T = std::make_shared<Matrix>(dim_new * dim_new, dim_old);
    for (int ind = 0; ind < dim_new * dim_new * dim_old; ++ind) {
        int a = ind / (dim_new * dim_old), b = (ind / dim_old) % dim_new, c = ind % dim_old;
        (*A_T)(a *dim_new + b, c) = (*A_new)(a * dim_old + c, b);
    }
    A_T = linalg::doublet(A_T, X, false, true);

    A_new = std::make_shared<Matrix>(dim_new, dim_new * dim_new);

    for (int ind = 0; ind < dim_new * dim_new * dim_new; ++ind) {
        int a = ind / (dim_new * dim_new), b = (ind / dim_new) % dim_new, c = ind % dim_new;
        (*A_new)(a, b *dim_new + c) = (*A_T)(a * dim_new + c, b);
    }

    return A_new;
}

void DLPNOCCSD_T::recompute_pnos() {
    timer_on("Recompute PNOs");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();
    int npao = C_pao_->colspi(0);

#pragma omp parallel for schedule(guided, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        int ji = ij_to_ji_[ij];

        if (i > j || n_pno_[ij] == 0) continue;

        // number of PAOs in the pair domain (before removing linear dependencies)
        int npao_ij = lmopair_to_paos_[ij].size();

        //                                      //
        // ==> Canonicalize PAOs of pair ij <== //
        //                                      //

        auto S_pao_ij = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);
        auto F_pao_ij = submatrix_rows_and_cols(*F_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);

        SharedMatrix X_pao_ij;  // canonical transformation of this domain's PAOs to
        SharedVector e_pao_ij;  // energies of the canonical PAOs
        std::tie(X_pao_ij, e_pao_ij) = orthocanonicalizer(S_pao_ij, F_pao_ij);

        F_pao_ij = linalg::triplet(X_pao_ij, F_pao_ij, X_pao_ij, true, false, false);

        int npao_can_ij = X_pao_ij->colspi(0);

        auto T_pao_ij = T_iajb_[ij]->clone();
        for (int a = 0; a < n_pno_[ij]; ++a) {
            for (int b = 0; b < n_pno_[ij]; ++b) {
                (*T_pao_ij)(a, b) =
                    (*T_pao_ij)(a, b) * (-(*e_pno_[ij])(a) - (*e_pno_[ij])(b) + (*F_lmo_)(i, i) + (*F_lmo_)(j, j));
            }
        }
        auto S_ij = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);
        S_ij = linalg::triplet(X_pno_[ij], S_ij, X_pao_ij, true, false, false);
        T_pao_ij = linalg::triplet(S_ij, T_pao_ij, S_ij, true, false, false);
        for (int a = 0; a < npao_can_ij; ++a) {
            for (int b = 0; b < npao_can_ij; ++b) {
                (*T_pao_ij)(a, b) =
                    (*T_pao_ij)(a, b) / (-(*e_pao_ij)(a) - (*e_pao_ij)(b) + (*F_lmo_)(i, i) + (*F_lmo_)(j, j));
            }
        }

        size_t nvir_ij = X_pao_ij->colspi(0);

        auto Tt_pao_ij = T_pao_ij->clone();
        Tt_pao_ij->scale(2.0);
        Tt_pao_ij->subtract(T_pao_ij->transpose());

        auto D_ij = linalg::doublet(Tt_pao_ij, T_pao_ij, false, true);
        D_ij->add(linalg::doublet(Tt_pao_ij, T_pao_ij, true, false));

        auto X_pno_ij = std::make_shared<Matrix>("eigenvectors", nvir_ij, nvir_ij);
        Vector pno_occ("eigenvalues", nvir_ij);
        D_ij->diagonalize(*X_pno_ij, pno_occ, descending);

        double t_cut_scale = (i == j) ? T_CUT_PNO_DIAG_SCALE_ : 1.0;

        int nvir_ij_final = 0;
        for (size_t a = 0; a < nvir_ij; ++a) {
            if (fabs(pno_occ.get(a)) >= t_cut_scale * T_CUT_TNO_) {
                nvir_ij_final++;
            }
        }

        Dimension zero(1);
        Dimension dim_final(1);
        dim_final.fill(nvir_ij_final);

        // This transformation gives orbitals that are orthonormal but not canonical
        X_pno_ij = X_pno_ij->get_block({zero, X_pno_ij->rowspi()}, {zero, dim_final});
        pno_occ = pno_occ.get_block({zero, dim_final});

        SharedMatrix pno_canon;
        SharedVector e_pno_ij;
        std::tie(pno_canon, e_pno_ij) = canonicalizer(X_pno_ij, F_pao_ij);

        X_pno_ij = linalg::doublet(X_pno_ij, pno_canon, false, false);

        auto T_pno_ij = linalg::triplet(X_pno_ij, T_pao_ij, X_pno_ij, true, false, false);
        auto Tt_pno_ij = linalg::triplet(X_pno_ij, Tt_pao_ij, X_pno_ij, true, false, false);

        X_pno_ij = linalg::doublet(X_pao_ij, X_pno_ij, false, false);

        // Recompute singles amplitudes
        if (i == j) {
            auto S_ii = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmopair_to_paos_[ij]);
            S_ii = linalg::triplet(X_pno_ij, S_ii, X_pno_[ij], true, false, false);
            T_ia_[i] = linalg::doublet(S_ii, T_ia_[i], false, false);
        }

        T_iajb_[ij] = T_pno_ij;
        Tt_iajb_[ij] = Tt_pno_ij;
        X_pno_[ij] = X_pno_ij;
        e_pno_[ij] = e_pno_ij;
        n_pno_[ij] = X_pno_ij->colspi(0);

        if (i < j) {
            T_iajb_[ji] = T_pno_ij->transpose();
            Tt_iajb_[ji] = Tt_pno_ij->transpose();
            X_pno_[ji] = X_pno_ij;
            e_pno_[ji] = e_pno_ij;
            n_pno_[ji] = X_pno_ij->colspi(0);
        }
    }

    timer_off("Recompute PNOs");
}

void DLPNOCCSD_T::triples_sparsity(bool prescreening) {
    timer_on("Triples Sparsity");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();
    int npao = C_pao_->colspi(0);

    if (prescreening) {
        int ijk = 0;
        // Every pair contains at least two strong pairs
        for (int ij = 0; ij < n_lmo_pairs; ij++) {
            int i, j;
            std::tie(i, j) = ij_to_i_j_[ij];
            if (i > j) continue;
            for (int k : lmopair_to_lmos_[ij]) {
                if (i > k || j > k) continue;
                if (i == j && j == k) continue;
                int ij_weak = i_j_to_ij_weak_[i][j], ik_weak = i_j_to_ij_weak_[i][k], kj_weak = i_j_to_ij_weak_[k][j];

                int weak_pair_count = 0;
                if (ij_weak != -1) weak_pair_count += 1;
                if (ik_weak != -1) weak_pair_count += 1;
                if (kj_weak != -1) weak_pair_count += 1;

                if (weak_pair_count > 1) continue;

                ijk_to_i_j_k_.push_back(std::make_tuple(i, j, k));
                i_j_k_to_ijk_[i * naocc * naocc + j * naocc + k] = ijk;
                i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j] = ijk;
                i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k] = ijk;
                i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i] = ijk;
                i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j] = ijk;
                i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i] = ijk;
                ++ijk;
            }
        }
    } else {
        std::unordered_map<int, int> i_j_k_to_ijk_new;
        std::vector<std::tuple<int, int, int>> ijk_to_i_j_k_new;

        double t_cut_triples_weak = options_.get_double("T_CUT_TRIPLES_WEAK");
        de_lccsd_t_screened_ = 0.0;

        int ijk_new = 0;
        for (int ijk = 0; ijk < ijk_to_i_j_k_.size(); ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

            if (std::fabs(e_ijk_[ijk]) >= t_cut_triples_weak) {
                ijk_to_i_j_k_new.push_back(std::make_tuple(i, j, k));
                i_j_k_to_ijk_new[i * naocc * naocc + j * naocc + k] = ijk_new;
                i_j_k_to_ijk_new[i * naocc * naocc + k * naocc + j] = ijk_new;
                i_j_k_to_ijk_new[j * naocc * naocc + i * naocc + k] = ijk_new;
                i_j_k_to_ijk_new[j * naocc * naocc + k * naocc + i] = ijk_new;
                i_j_k_to_ijk_new[k * naocc * naocc + i * naocc + j] = ijk_new;
                i_j_k_to_ijk_new[k * naocc * naocc + j * naocc + i] = ijk_new;
                ++ijk_new;
            } else {
                de_lccsd_t_screened_ += e_ijk_[ijk];
            }
        }
        i_j_k_to_ijk_ = i_j_k_to_ijk_new;
        ijk_to_i_j_k_ = ijk_to_i_j_k_new;
    }

    int n_lmo_triplets = ijk_to_i_j_k_.size();
    int natom = molecule_->natom();
    int nbf = basisset_->nbf();

    tno_scale_.clear();
    tno_scale_.resize(n_lmo_triplets, 1.0);

    // => Local density fitting domains <= //

    SparseMap lmo_to_ribfs(naocc);
    SparseMap lmo_to_riatoms(naocc);

    double t_cut_mkn_triples = (prescreening) ? options_.get_double("T_CUT_MKN_TRIPLES_PRE") : options_.get_double("T_CUT_MKN_TRIPLES");

    for (size_t i = 0; i < naocc; ++i) {
        // atomic mulliken populations for this orbital
        std::vector<double> mkn_pop(natom, 0.0);

        auto P_i = reference_wavefunction_->S()->clone();

        for (size_t u = 0; u < nbf; u++) {
            P_i->scale_row(0, u, C_lmo_->get(u, i));
            P_i->scale_column(0, u, C_lmo_->get(u, i));
        }

        for (size_t u = 0; u < nbf; u++) {
            int centerU = basisset_->function_to_center(u);
            double p_uu = P_i->get(u, u);

            for (size_t v = 0; v < nbf; v++) {
                int centerV = basisset_->function_to_center(v);
                double p_vv = P_i->get(v, v);

                // off-diag pops (p_uv) split between u and v prop to diag pops
                double p_uv = P_i->get(u, v);
                mkn_pop[centerU] += p_uv * ((p_uu) / (p_uu + p_vv));
                mkn_pop[centerV] += p_uv * ((p_vv) / (p_uu + p_vv));
            }
        }

        // if non-zero mulliken pop on atom, include atom in the LMO's fitting domain
        for (size_t a = 0; a < natom; a++) {
            if (fabs(mkn_pop[a]) > t_cut_mkn_triples) {
                lmo_to_riatoms[i].push_back(a);

                // each atom's aux orbitals are all-or-nothing for each LMO
                for (int u : atom_to_ribf_[a]) {
                    lmo_to_ribfs[i].push_back(u);
                }
            }
        }
    }

    // => PAO domains <= //

    SparseMap lmo_to_paos(naocc);

    double t_cut_do_triples = (prescreening) ? options_.get_double("T_CUT_DO_TRIPLES_PRE") : options_.get_double("T_CUT_DO_TRIPLES");

    for (size_t i = 0; i < naocc; ++i) {
        // PAO domains determined by differential overlap integral
        std::vector<int> lmo_to_paos_temp;
        for (size_t u = 0; u < nbf; ++u) {
            if (fabs(DOI_iu_->get(i, u)) > t_cut_do_triples) {
                lmo_to_paos_temp.push_back(u);
            }
        }

        // if any PAO on an atom is in the list, we take all of the PAOs on that atom
        lmo_to_paos[i] = contract_lists(lmo_to_paos_temp, atom_to_bf_);
    }

    if (!prescreening) {
        lmotriplet_to_ribfs_.clear();
        lmotriplet_to_lmos_.clear();
        lmotriplet_to_paos_.clear();

        lmotriplet_lmo_to_riatom_lmo_.clear();
        lmotriplet_pao_to_riatom_pao_.clear();
    }

    lmotriplet_to_ribfs_.resize(n_lmo_triplets);
    lmotriplet_to_lmos_.resize(n_lmo_triplets);
    lmotriplet_to_paos_.resize(n_lmo_triplets);

    lmotriplet_lmo_to_riatom_lmo_.resize(n_lmo_triplets);
    lmotriplet_pao_to_riatom_pao_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        lmotriplet_to_ribfs_[ijk] = merge_lists(lmo_to_ribfs[i], merge_lists(lmo_to_ribfs[j], lmo_to_ribfs[k]));
        for (int l = 0; l < naocc; ++l) {
            int il = i_j_to_ij_[i][l], jl = i_j_to_ij_[j][l], kl = i_j_to_ij_[k][l];
            if (il != -1 && jl != -1 && kl != -1) lmotriplet_to_lmos_[ijk].push_back(l);
        }
        lmotriplet_to_paos_[ijk] = merge_lists(lmo_to_paos[i], merge_lists(lmo_to_paos[j], lmo_to_paos[k]));

        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();
        int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        int npao_ijk = lmotriplet_to_paos_[ijk].size();

        lmotriplet_lmo_to_riatom_lmo_[ijk].resize(naux_ijk);
        lmotriplet_pao_to_riatom_pao_[ijk].resize(naux_ijk);

        for (int q_ijk = 0; q_ijk < naux_ijk; q_ijk++) {
            int q = lmotriplet_to_ribfs_[ijk][q_ijk];
            int centerq = ribasis_->function_to_center(q);

            lmotriplet_lmo_to_riatom_lmo_[ijk][q_ijk].resize(nlmo_ijk);
            lmotriplet_pao_to_riatom_pao_[ijk][q_ijk].resize(npao_ijk);

            for (int m_ijk = 0; m_ijk < nlmo_ijk; m_ijk++) {
                int m = lmotriplet_to_lmos_[ijk][m_ijk];
                int m_sparse = riatom_to_lmos_ext_dense_[centerq][m];
                lmotriplet_lmo_to_riatom_lmo_[ijk][q_ijk][m_ijk] = m_sparse;
            }

            for (int a_ijk = 0; a_ijk < npao_ijk; a_ijk++) {
                int a = lmotriplet_to_paos_[ijk][a_ijk];
                int a_sparse = riatom_to_paos_ext_dense_[centerq][a];
                lmotriplet_pao_to_riatom_pao_[ijk][q_ijk][a_ijk] = a_sparse;
            }
        }
    }


    timer_off("Triples Sparsity");
}

void DLPNOCCSD_T::sort_triplets(double e_total) {
    timer_on("Sort Triplets");

    outfile->Printf("  ==> Sorting Triplets <== \n\n");

    int n_lmo_triplets = ijk_to_i_j_k_.size();
    std::vector<std::pair<int, double>> ijk_e_pairs(n_lmo_triplets);

#pragma omp parallel for
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        ijk_e_pairs[ijk] = std::make_pair(ijk, e_ijk_[ijk]);
    }

    std::sort(ijk_e_pairs.begin(), ijk_e_pairs.end(), [&](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return (std::fabs(a.second) > std::fabs(b.second));
    });

    double e_curr = 0.0;
    double strong_scale = options_.get_double("T_CUT_TNO_STRONG_SCALE");
    double weak_scale = options_.get_double("T_CUT_TNO_WEAK_SCALE");
    is_strong_triplet_.resize(n_lmo_triplets, false);
    tno_scale_.clear();
    tno_scale_.resize(n_lmo_triplets, weak_scale);

    int strong_count = 0;
    for (int idx = 0; idx < n_lmo_triplets; ++idx) {
        is_strong_triplet_[ijk_e_pairs[idx].first] = true;
        tno_scale_[ijk_e_pairs[idx].first] = strong_scale;
        e_curr += ijk_e_pairs[idx].second;
        ++strong_count;
        if (e_curr / e_total > 0.9) break;
    }

    outfile->Printf("    Number of Strong Triplets: %6d, Total Triplets: %6d, Ratio: %.4f\n\n", strong_count, n_lmo_triplets, 
                            (double) strong_count / n_lmo_triplets);

    timer_off("Sort Triplets");
}

void DLPNOCCSD_T::tno_transform(bool scale_triples, double t_cut_tno) {
    timer_on("TNO transform");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    X_tno_.clear();
    e_tno_.clear();
    n_tno_.clear();

    X_tno_.resize(n_lmo_triplets);
    e_tno_.resize(n_lmo_triplets);
    n_tno_.resize(n_lmo_triplets);

    ijk_scale_.resize(n_lmo_triplets, 1.0);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        // number of PAOs in the triplet domain (before removing linear dependencies)
        int npao_ijk = lmotriplet_to_paos_[ijk].size();

        // number of auxiliary basis in the domain
        int naux_ijk = lmotriplet_to_ribfs_[ijk].size();

        auto i_qa = std::make_shared<Matrix>("Three-index Integrals", naux_ijk, npao_ijk);
        auto j_qa = std::make_shared<Matrix>("Three-index Integrals", naux_ijk, npao_ijk);
        auto k_qa = std::make_shared<Matrix>("Three-index Integrals", naux_ijk, npao_ijk);

        for (int q_ijk = 0; q_ijk < naux_ijk; q_ijk++) {
            int q = lmotriplet_to_ribfs_[ijk][q_ijk];
            int centerq = ribasis_->function_to_center(q);
            for (int a_ijk = 0; a_ijk < npao_ijk; a_ijk++) {
                int a = lmotriplet_to_paos_[ijk][a_ijk];
                i_qa->set(q_ijk, a_ijk,
                          qia_[q]->get(riatom_to_lmos_ext_dense_[centerq][i], riatom_to_paos_ext_dense_[centerq][a]));
                j_qa->set(q_ijk, a_ijk,
                          qia_[q]->get(riatom_to_lmos_ext_dense_[centerq][j], riatom_to_paos_ext_dense_[centerq][a]));
                k_qa->set(q_ijk, a_ijk,
                          qia_[q]->get(riatom_to_lmos_ext_dense_[centerq][k], riatom_to_paos_ext_dense_[centerq][a]));
            }
        }

        auto A_solve = submatrix_rows_and_cols(*full_metric_, lmotriplet_to_ribfs_[ijk], lmotriplet_to_ribfs_[ijk]);
        A_solve->power(0.5, 1.0e-14);
        C_DGESV_wrapper(A_solve->clone(), i_qa);
        C_DGESV_wrapper(A_solve->clone(), j_qa);
        C_DGESV_wrapper(A_solve->clone(), k_qa);

        auto K_pao_ij = linalg::doublet(i_qa, j_qa, true, false);
        auto K_pao_jk = linalg::doublet(j_qa, k_qa, true, false);
        auto K_pao_ik = linalg::doublet(i_qa, k_qa, true, false);

        //                                          //
        // ==> Canonicalize PAOs of triplet ijk <== //
        //                                          //

        auto S_pao_ijk = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijk]);
        auto F_pao_ijk = submatrix_rows_and_cols(*F_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijk]);

        SharedMatrix X_pao_ijk;
        SharedVector e_pao_ijk;
        std::tie(X_pao_ijk, e_pao_ijk) = orthocanonicalizer(S_pao_ijk, F_pao_ijk);

        F_pao_ijk = linalg::triplet(X_pao_ijk, F_pao_ijk, X_pao_ijk, true, false, false);
        K_pao_ij = linalg::triplet(X_pao_ijk, K_pao_ij, X_pao_ijk, true, false, false);
        K_pao_jk = linalg::triplet(X_pao_ijk, K_pao_jk, X_pao_ijk, true, false, false);
        K_pao_ik = linalg::triplet(X_pao_ijk, K_pao_ik, X_pao_ijk, true, false, false);

        // number of PAOs in the domain after removing linear dependencies
        int npao_can_ijk = X_pao_ijk->colspi(0);
        auto T_pao_ij = K_pao_ij->clone();
        auto T_pao_jk = K_pao_jk->clone();
        auto T_pao_ik = K_pao_ik->clone();
        for (int a = 0; a < npao_can_ijk; ++a) {
            for (int b = 0; b < npao_can_ijk; ++b) {
                T_pao_ij->set(a, b,
                              T_pao_ij->get(a, b) /
                                  (-e_pao_ijk->get(b) + -e_pao_ijk->get(a) + F_lmo_->get(i, i) + F_lmo_->get(j, j)));
                T_pao_jk->set(a, b,
                              T_pao_ij->get(a, b) /
                                  (-e_pao_ijk->get(b) + -e_pao_ijk->get(a) + F_lmo_->get(j, j) + F_lmo_->get(k, k)));
                T_pao_ik->set(a, b,
                              T_pao_ij->get(a, b) /
                                  (-e_pao_ijk->get(b) + -e_pao_ijk->get(a) + F_lmo_->get(i, i) + F_lmo_->get(k, k)));
            }
        }

        // Save amplitudes as MP2 versions for TNO truncation scaling
        auto T_pao_ij_mp2 = T_pao_ij->clone();
        auto T_pao_jk_mp2 = T_pao_jk->clone();
        auto T_pao_ik_mp2 = T_pao_ik->clone();

        // Create antisymmetrized ERIs
        auto L_pao_ij = K_pao_ij->clone();
        L_pao_ij->scale(2.0);
        L_pao_ij->subtract(K_pao_ij->transpose());

        auto L_pao_jk = K_pao_jk->clone();
        L_pao_jk->scale(2.0);
        L_pao_jk->subtract(K_pao_jk->transpose());

        auto L_pao_ik = K_pao_ik->clone();
        L_pao_ik->scale(2.0);
        L_pao_ik->subtract(K_pao_ik->transpose());

        // Compute non-truncated energies
        double e_ij_non_trunc = T_pao_ij_mp2->vector_dot(L_pao_ij);
        double e_jk_non_trunc = T_pao_jk_mp2->vector_dot(L_pao_jk);
        double e_ik_non_trunc = T_pao_ik_mp2->vector_dot(L_pao_ik);

        if (n_pno_[ij] > 0) {
            T_pao_ij = T_iajb_[ij]->clone();
            for (int a = 0; a < n_pno_[ij]; ++a) {
                for (int b = 0; b < n_pno_[ij]; ++b) {
                    (*T_pao_ij)(a, b) =
                        (*T_pao_ij)(a, b) * (-(*e_pno_[ij])(a) - (*e_pno_[ij])(b) + (*F_lmo_)(i, i) + (*F_lmo_)(j, j));
                }
            }
            auto S_ij = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ij], lmotriplet_to_paos_[ijk]);
            S_ij = linalg::triplet(X_pno_[ij], S_ij, X_pao_ijk, true, false, false);
            T_pao_ij = linalg::triplet(S_ij, T_pao_ij, S_ij, true, false, false);
            for (int a = 0; a < npao_can_ijk; ++a) {
                for (int b = 0; b < npao_can_ijk; ++b) {
                    (*T_pao_ij)(a, b) =
                        (*T_pao_ij)(a, b) / (-(*e_pao_ijk)(a) - (*e_pao_ijk)(b) + (*F_lmo_)(i, i) + (*F_lmo_)(j, j));
                }
            }
        }

        if (n_pno_[jk] > 0) {
            T_pao_jk = T_iajb_[jk]->clone();
            for (int a = 0; a < n_pno_[jk]; ++a) {
                for (int b = 0; b < n_pno_[jk]; ++b) {
                    (*T_pao_jk)(a, b) =
                        (*T_pao_jk)(a, b) * (-(*e_pno_[jk])(a) - (*e_pno_[jk])(b) + (*F_lmo_)(j, j) + (*F_lmo_)(k, k));
                }
            }
            auto S_jk = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[jk], lmotriplet_to_paos_[ijk]);
            S_jk = linalg::triplet(X_pno_[jk], S_jk, X_pao_ijk, true, false, false);
            T_pao_jk = linalg::triplet(S_jk, T_pao_jk, S_jk, true, false, false);
            for (int a = 0; a < npao_can_ijk; ++a) {
                for (int b = 0; b < npao_can_ijk; ++b) {
                    (*T_pao_jk)(a, b) =
                        (*T_pao_jk)(a, b) / (-(*e_pao_ijk)(a) - (*e_pao_ijk)(b) + (*F_lmo_)(j, j) + (*F_lmo_)(k, k));
                }
            }
        }

        if (n_pno_[ik] > 0) {
            T_pao_ik = T_iajb_[ik]->clone();
            for (int a = 0; a < n_pno_[ik]; ++a) {
                for (int b = 0; b < n_pno_[ik]; ++b) {
                    (*T_pao_ik)(a, b) =
                        (*T_pao_ik)(a, b) * (-(*e_pno_[ik])(a) - (*e_pno_[ik])(b) + (*F_lmo_)(i, i) + (*F_lmo_)(k, k));
                }
            }
            auto S_ik = submatrix_rows_and_cols(*S_pao_, lmopair_to_paos_[ik], lmotriplet_to_paos_[ijk]);
            S_ik = linalg::triplet(X_pno_[ik], S_ik, X_pao_ijk, true, false, false);
            T_pao_ik = linalg::triplet(S_ik, T_pao_ik, S_ik, true, false, false);
            for (int a = 0; a < npao_can_ijk; ++a) {
                for (int b = 0; b < npao_can_ijk; ++b) {
                    (*T_pao_ik)(a, b) =
                        (*T_pao_ik)(a, b) / (-(*e_pao_ijk)(a) - (*e_pao_ijk)(b) + (*F_lmo_)(i, i) + (*F_lmo_)(k, k));
                }
            }
        }

        //                                           //
        // ==> Canonical PAOs  to Canonical TNOs <== //
        //                                           //

        size_t nvir_ijk = F_pao_ijk->rowspi(0);

        auto Tt_pao_ij = T_pao_ij->clone();
        Tt_pao_ij->scale(2.0);
        Tt_pao_ij->subtract(T_pao_ij->transpose());

        auto Tt_pao_jk = T_pao_jk->clone();
        Tt_pao_jk->scale(2.0);
        Tt_pao_jk->subtract(T_pao_jk->transpose());

        auto Tt_pao_ik = T_pao_ik->clone();
        Tt_pao_ik->scale(2.0);
        Tt_pao_ik->subtract(T_pao_ik->transpose());

        // Construct pair densities from amplitudes
        auto D_ij = linalg::doublet(Tt_pao_ij, T_pao_ij, false, true);
        D_ij->add(linalg::doublet(Tt_pao_ij, T_pao_ij, true, false));

        auto D_jk = linalg::doublet(Tt_pao_jk, T_pao_jk, false, true);
        D_jk->add(linalg::doublet(Tt_pao_jk, T_pao_jk, true, false));

        auto D_ik = linalg::doublet(Tt_pao_ik, T_pao_ik, false, true);
        D_ik->add(linalg::doublet(Tt_pao_ik, T_pao_ik, true, false));

        // Construct triplet density from pair densities
        auto D_ijk = D_ij->clone();
        D_ijk->add(D_jk);
        D_ijk->add(D_ik);
        D_ijk->scale(1.0 / 3.0);

        // Diagonalization of triplet density gives TNOs (in basis of LMO's virtual domain)
        // as well as TNO occ numbers
        auto X_tno_ijk = std::make_shared<Matrix>("eigenvectors", nvir_ijk, nvir_ijk);
        Vector tno_occ("eigenvalues", nvir_ijk);
        D_ijk->diagonalize(*X_tno_ijk, tno_occ, descending);

        double tno_scale = tno_scale_[ijk];

        int nvir_ijk_final = 0;
        for (size_t a = 0; a < nvir_ijk; ++a) {
            if (fabs(tno_occ.get(a)) >= tno_scale * t_cut_tno) {
                nvir_ijk_final++;
            }
        }

        nvir_ijk_final = std::max(1, nvir_ijk_final);

        Dimension zero(1);
        Dimension dim_final(1);
        dim_final.fill(nvir_ijk_final);

        // This transformation gives orbitals that are orthonormal but not canonical
        X_tno_ijk = X_tno_ijk->get_block({zero, X_tno_ijk->rowspi()}, {zero, dim_final});
        tno_occ = tno_occ.get_block({zero, dim_final});

        SharedMatrix tno_canon;
        SharedVector e_tno_ijk;
        std::tie(tno_canon, e_tno_ijk) = canonicalizer(X_tno_ijk, F_pao_ijk);

        X_tno_ijk = linalg::doublet(X_tno_ijk, tno_canon, false, false);

        if (scale_triples) {
            // Compute truncated energies and scaling factors
            auto T_pno_ij_mp2 = linalg::triplet(X_tno_ijk, T_pao_ij_mp2, X_tno_ijk, true, false, false);
            auto T_pno_jk_mp2 = linalg::triplet(X_tno_ijk, T_pao_jk_mp2, X_tno_ijk, true, false, false);
            auto T_pno_ik_mp2 = linalg::triplet(X_tno_ijk, T_pao_ik_mp2, X_tno_ijk, true, false, false);

            auto L_pno_ij = linalg::triplet(X_tno_ijk, L_pao_ij, X_tno_ijk, true, false, false);
            auto L_pno_jk = linalg::triplet(X_tno_ijk, L_pao_jk, X_tno_ijk, true, false, false);
            auto L_pno_ik = linalg::triplet(X_tno_ijk, L_pao_ik, X_tno_ijk, true, false, false);
        
            double e_ij_trunc = T_pno_ij_mp2->vector_dot(L_pno_ij);
            double e_jk_trunc = T_pno_jk_mp2->vector_dot(L_pno_jk);
            double e_ik_trunc = T_pno_ik_mp2->vector_dot(L_pno_ik);
        
            ijk_scale_[ijk] = (e_ij_non_trunc / e_ij_trunc + e_jk_non_trunc / e_jk_trunc
                                + e_ik_non_trunc / e_ik_trunc) / 3.0;
        }

        X_tno_ijk = linalg::doublet(X_pao_ijk, X_tno_ijk, false, false);

        X_tno_[ijk] = X_tno_ijk;
        e_tno_[ijk] = e_tno_ijk;
        n_tno_[ijk] = X_tno_ijk->colspi(0);
    }

    int tno_count_total = 0, tno_count_min = C_pao_->colspi(0), tno_count_max = 0;
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        tno_count_total += n_tno_[ijk];
        tno_count_min = std::min(tno_count_min, n_tno_[ijk]);
        tno_count_max = std::max(tno_count_max, n_tno_[ijk]);
    }

    int n_total_possible = (naocc + 2) * (naocc + 1) * (naocc) / 6 - naocc;

    outfile->Printf("  \n");
    outfile->Printf("    Number of (Unique) Local MO triplets: %d\n", n_lmo_triplets);
    outfile->Printf("    Max Number of Possible (Unique) LMO Triplets: %d (Ratio: %.4f)\n", n_total_possible,
                    (double)n_lmo_triplets / n_total_possible);
    outfile->Printf("    Natural Orbitals per Local MO triplet:\n");
    outfile->Printf("      Avg: %3d NOs \n", tno_count_total / n_lmo_triplets);
    outfile->Printf("      Min: %3d NOs \n", tno_count_min);
    outfile->Printf("      Max: %3d NOs \n", tno_count_max);
    outfile->Printf("  \n");

    timer_off("TNO transform");
}

void DLPNOCCSD_T::estimate_memory() {
    outfile->Printf("  ==> DLPNO-(T) Memory Estimate <== \n\n");

    int n_lmo_triplets = ijk_to_i_j_k_.size();

    size_t tno_total_memory = 0;
#pragma omp parallel for reduction(+ : tno_total_memory)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        tno_total_memory += n_tno_[ijk] * n_tno_[ijk] * n_tno_[ijk];
    }
    size_t total_memory = qij_memory_ + qia_memory_ + qab_memory_ + 3 * tno_total_memory;

    outfile->Printf("    (q | i j) integrals    : %.3f [GiB]\n", qij_memory_ * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (q | i a) integrals    : %.3f [GiB]\n", qia_memory_ * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    (q | a b) integrals    : %.3f [GiB]\n", qab_memory_ * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    W_{ijk}^{abc}          : %.3f [GiB]\n", tno_total_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    V_{ijk}^{abc}          : %.3f [GiB]\n", tno_total_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    T_{ijk}^{abc}          : %.3f [GiB]\n", tno_total_memory * pow(2.0, -30) * sizeof(double));
    outfile->Printf("    Total Memory Given     : %.3f [GiB]\n", memory_ * pow(2.0, -30));
    outfile->Printf("    Total Memory Required  : %.3f [GiB]\n\n", total_memory * pow(2.0, -30) * sizeof(double));

    if (3 * tno_total_memory * sizeof(double) > 0.8 * (memory_ - qab_memory_ * sizeof(double))) {
        write_amplitudes_ = true;
        outfile->Printf("    Writing all X_{ijk}^{abc} quantities to disk...\n\n");
    } else {
        write_amplitudes_ = false;
        outfile->Printf("    Keeping all X_{ijk}^{abc} quantities in core...\n\n");
    }
}

double DLPNOCCSD_T::compute_lccsd_t0(bool store_amplitudes) {
    timer_on("LCCSD(T0)");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    double E_T0 = 0.0;

    if (write_qab_pao_) {
        psio_->open(PSIF_DLPNO_QAB_PAO, PSIO_OPEN_OLD);
    }

    if (store_amplitudes) {
        W_iajbkc_.resize(n_lmo_triplets);
        V_iajbkc_.resize(n_lmo_triplets);
        T_iajbkc_.resize(n_lmo_triplets);
        if (write_amplitudes_) psio_->open(PSIF_DLPNO_TRIPLES, PSIO_OPEN_NEW);
    }

    e_ijk_.clear();
    e_ijk_.resize(n_lmo_triplets, 0.0);

#pragma omp parallel for schedule(dynamic) reduction(+ : E_T0)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];

        if (ntno_ijk == 0) continue;

        // => Step 1: Compute all necessary integrals

        // number of LMOs in the triplet domain
        const int nlmo_ijk = lmotriplet_to_lmos_[ijk].size();
        // number of PAOs in the triplet domain (before removing linear dependencies)
        const int npao_ijk = lmotriplet_to_paos_[ijk].size();
        // number of auxiliary functions in the triplet domain
        const int naux_ijk = lmotriplet_to_ribfs_[ijk].size();

        /// => Build (i a_ijk | b_ijk d_ijk) and (k c_ijk | j l) integrals <= ///

        auto q_iv = std::make_shared<Matrix>(naux_ijk, ntno_ijk);
        auto q_jv = std::make_shared<Matrix>(naux_ijk, ntno_ijk);
        auto q_kv = std::make_shared<Matrix>(naux_ijk, ntno_ijk);

        auto q_io = std::make_shared<Matrix>(naux_ijk, nlmo_ijk);
        auto q_jo = std::make_shared<Matrix>(naux_ijk, nlmo_ijk);
        auto q_ko = std::make_shared<Matrix>(naux_ijk, nlmo_ijk);

        auto q_vv = std::make_shared<Matrix>(naux_ijk, ntno_ijk * ntno_ijk);

        for (int q_ijk = 0; q_ijk < naux_ijk; q_ijk++) {
            const int q = lmotriplet_to_ribfs_[ijk][q_ijk];
            const int centerq = ribasis_->function_to_center(q);

            const int i_sparse = riatom_to_lmos_ext_dense_[centerq][i];
            const std::vector<int> i_slice(1, i_sparse);
            const int j_sparse = riatom_to_lmos_ext_dense_[centerq][j];
            const std::vector<int> j_slice(1, j_sparse);
            const int k_sparse = riatom_to_lmos_ext_dense_[centerq][k];
            const std::vector<int> k_slice(1, k_sparse);

            auto q_iv_tmp = submatrix_rows_and_cols(*qia_[q], i_slice, lmotriplet_pao_to_riatom_pao_[ijk][q_ijk]);
            q_iv_tmp = linalg::doublet(q_iv_tmp, X_tno_[ijk], false, false);
            C_DCOPY(ntno_ijk, &(*q_iv_tmp)(0, 0), 1, &(*q_iv)(q_ijk, 0), 1);

            auto q_jv_tmp = submatrix_rows_and_cols(*qia_[q], j_slice, lmotriplet_pao_to_riatom_pao_[ijk][q_ijk]);
            q_jv_tmp = linalg::doublet(q_jv_tmp, X_tno_[ijk], false, false);
            C_DCOPY(ntno_ijk, &(*q_jv_tmp)(0, 0), 1, &(*q_jv)(q_ijk, 0), 1);

            auto q_kv_tmp = submatrix_rows_and_cols(*qia_[q], k_slice, lmotriplet_pao_to_riatom_pao_[ijk][q_ijk]);
            q_kv_tmp = linalg::doublet(q_kv_tmp, X_tno_[ijk], false, false);
            C_DCOPY(ntno_ijk, &(*q_kv_tmp)(0, 0), 1, &(*q_kv)(q_ijk, 0), 1);

            auto q_io_tmp = submatrix_rows_and_cols(*qij_[q], i_slice, lmotriplet_lmo_to_riatom_lmo_[ijk][q_ijk]);
            C_DCOPY(nlmo_ijk, &(*q_io_tmp)(0, 0), 1, &(*q_io)(q_ijk, 0), 1);

            auto q_jo_tmp = submatrix_rows_and_cols(*qij_[q], j_slice, lmotriplet_lmo_to_riatom_lmo_[ijk][q_ijk]);
            C_DCOPY(nlmo_ijk, &(*q_jo_tmp)(0, 0), 1, &(*q_jo)(q_ijk, 0), 1);

            auto q_ko_tmp = submatrix_rows_and_cols(*qij_[q], k_slice, lmotriplet_lmo_to_riatom_lmo_[ijk][q_ijk]);
            C_DCOPY(nlmo_ijk, &(*q_ko_tmp)(0, 0), 1, &(*q_ko)(q_ijk, 0), 1);

            SharedMatrix q_vv_tmp;
            if (write_qab_pao_) {
                std::stringstream toc_entry;
                toc_entry << "QAB (PAO) " << q;
                int npao_q = riatom_to_paos_ext_[centerq].size();
                q_vv_tmp = std::make_shared<Matrix>(toc_entry.str(), npao_q, npao_q);
#pragma omp critical
                q_vv_tmp->load(psio_, PSIF_DLPNO_QAB_PAO, psi::Matrix::LowerTriangle);
                q_vv_tmp = submatrix_rows_and_cols(*q_vv_tmp, lmotriplet_pao_to_riatom_pao_[ijk][q_ijk],
                                                lmotriplet_pao_to_riatom_pao_[ijk][q_ijk]);
            } else {
                q_vv_tmp = submatrix_rows_and_cols(*qab_[q], lmotriplet_pao_to_riatom_pao_[ijk][q_ijk],
                                                lmotriplet_pao_to_riatom_pao_[ijk][q_ijk]);
            }
            q_vv_tmp = linalg::triplet(X_tno_[ijk], q_vv_tmp, X_tno_[ijk], true, false, false);
                
            C_DCOPY(ntno_ijk * ntno_ijk, &(*q_vv_tmp)(0, 0), 1, &(*q_vv)(q_ijk, 0), 1);
        }

        auto A_solve = submatrix_rows_and_cols(*full_metric_, lmotriplet_to_ribfs_[ijk], lmotriplet_to_ribfs_[ijk]);
        A_solve->power(0.5, 1.0e-14);

        C_DGESV_wrapper(A_solve->clone(), q_iv);
        C_DGESV_wrapper(A_solve->clone(), q_jv);
        C_DGESV_wrapper(A_solve->clone(), q_kv);
        C_DGESV_wrapper(A_solve->clone(), q_io);
        C_DGESV_wrapper(A_solve->clone(), q_jo);
        C_DGESV_wrapper(A_solve->clone(), q_ko);
        C_DGESV_wrapper(A_solve->clone(), q_vv);

        // W integrals
        auto K_ivvv = linalg::doublet(q_iv, q_vv, true, false);
        auto K_jvvv = linalg::doublet(q_jv, q_vv, true, false);
        auto K_kvvv = linalg::doublet(q_kv, q_vv, true, false);

        auto K_iojv = linalg::doublet(q_io, q_jv, true, false);
        auto K_joiv = linalg::doublet(q_jo, q_iv, true, false);
        auto K_kojv = linalg::doublet(q_ko, q_jv, true, false);
        auto K_jokv = linalg::doublet(q_jo, q_kv, true, false);
        auto K_iokv = linalg::doublet(q_io, q_kv, true, false);
        auto K_koiv = linalg::doublet(q_ko, q_iv, true, false);

        // V integrals
        auto K_jk = linalg::doublet(q_jv, q_kv, true, false);
        auto K_ik = linalg::doublet(q_iv, q_kv, true, false);
        auto K_ij = linalg::doublet(q_iv, q_jv, true, false);

        // => Step 1: Compute W_ijk <= //

        std::stringstream w_name;
        w_name << "W " << (ijk);
        auto W_ijk = std::make_shared<Matrix>(w_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
        W_ijk->zero();

        std::vector<std::tuple<int, int, int>> perms = {std::make_tuple(i, j, k), std::make_tuple(i, k, j),
                                                        std::make_tuple(j, i, k), std::make_tuple(j, k, i),
                                                        std::make_tuple(k, i, j), std::make_tuple(k, j, i)};
        std::vector<SharedMatrix> Wperms(perms.size());

        std::vector<SharedMatrix> K_ovvv_list = {K_ivvv, K_ivvv, K_jvvv, K_jvvv, K_kvvv, K_kvvv};
        std::vector<SharedMatrix> K_ooov_list = {K_jokv, K_kojv, K_iokv, K_koiv, K_iojv, K_joiv};

        for (int idx = 0; idx < perms.size(); ++idx) {
            int i, j, k;
            std::tie(i, j, k) = perms[idx];

            int ii = i_j_to_ij_[i][i];
            int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];
            int kj = ij_to_ji_[jk];

            Wperms[idx] = std::make_shared<Matrix>(ntno_ijk, ntno_ijk * ntno_ijk);
            Wperms[idx]->zero();

            if (n_pno_[kj] > 0) {
                // Compute overlap between TNOs of triplet ijk and PNOs of pair kj
                auto S_ijk_kj = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[kj]);
                S_ijk_kj = linalg::triplet(X_tno_[ijk], S_ijk_kj, X_pno_[kj], true, false, false);

                auto T_kj = linalg::doublet(S_ijk_kj, T_iajb_[kj], false, false);

                auto K_ovvv = K_ovvv_list[idx]->clone();

                K_ovvv->reshape(ntno_ijk * ntno_ijk, ntno_ijk);
                K_ovvv = linalg::doublet(K_ovvv, S_ijk_kj, false, false);
                Wperms[idx]->add(linalg::doublet(K_ovvv, T_kj, false, true));
            }

            for (int l_ijk = 0; l_ijk < lmotriplet_to_lmos_[ijk].size(); ++l_ijk) {
                int l = lmotriplet_to_lmos_[ijk][l_ijk];
                int il = i_j_to_ij_[i][l];

                if (il == -1 || n_pno_[il] == 0) continue;

                auto S_ijk_il = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[il]);
                S_ijk_il = linalg::triplet(X_tno_[ijk], S_ijk_il, X_pno_[il], true, false, false);

                auto T_il = linalg::triplet(S_ijk_il, T_iajb_[il], S_ijk_il, false, false, true);

                for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                        for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                            (*Wperms[idx])(a_ijk, b_ijk *ntno_ijk + c_ijk) -=
                                (*T_il)(a_ijk, b_ijk) * (*K_ooov_list[idx])(l_ijk, c_ijk);
                        }
                    }
                }  // end a_ijk
            }      // end l_ijk
        }

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*W_ijk)(a_ijk, b_ijk *ntno_ijk + c_ijk) =
                        (*Wperms[0])(a_ijk, b_ijk * ntno_ijk + c_ijk) + (*Wperms[1])(a_ijk, c_ijk * ntno_ijk + b_ijk) +
                        (*Wperms[2])(b_ijk, a_ijk * ntno_ijk + c_ijk) + (*Wperms[3])(b_ijk, c_ijk * ntno_ijk + a_ijk) +
                        (*Wperms[4])(c_ijk, a_ijk * ntno_ijk + b_ijk) + (*Wperms[5])(c_ijk, b_ijk * ntno_ijk + a_ijk);
                }
            }
        }

        // => Step 2: Compute V_ijk <= //

        auto V_ijk = W_ijk->clone();
        std::stringstream v_name;
        v_name << "V " << (ijk);
        V_ijk->set_name(v_name.str());

        // Compute overlap between TNOs of triplet ijk and PNOs of pair ii, jj, and kk
        int ii = i_j_to_ij_[i][i];
        auto S_ijk_ii = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ii]);
        S_ijk_ii = linalg::triplet(X_tno_[ijk], S_ijk_ii, X_pno_[ii], true, false, false);

        int jj = i_j_to_ij_[j][j];
        auto S_ijk_jj = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[jj]);
        S_ijk_jj = linalg::triplet(X_tno_[ijk], S_ijk_jj, X_pno_[jj], true, false, false);

        int kk = i_j_to_ij_[k][k];
        auto S_ijk_kk = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[kk]);
        S_ijk_kk = linalg::triplet(X_tno_[ijk], S_ijk_kk, X_pno_[kk], true, false, false);

        auto T_i = linalg::doublet(S_ijk_ii, T_ia_[i], false, false);
        auto T_j = linalg::doublet(S_ijk_jj, T_ia_[j], false, false);
        auto T_k = linalg::doublet(S_ijk_kk, T_ia_[k], false, false);

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*V_ijk)(a_ijk, b_ijk *ntno_ijk + c_ijk) += (*T_i)(a_ijk, 0) * (*K_jk)(b_ijk, c_ijk) +
                                                                (*T_j)(b_ijk, 0) * (*K_ik)(a_ijk, c_ijk) +
                                                                (*T_k)(c_ijk, 0) * (*K_ij)(a_ijk, b_ijk);
                }
            }
        }

        // Step 3: Compute T0 energy through amplitudes
        auto T_ijk = W_ijk->clone();
        std::stringstream t_name;
        t_name << "T " << (ijk);
        T_ijk->set_name(t_name.str());

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*T_ijk)(a_ijk, b_ijk *ntno_ijk + c_ijk) =
                        -(*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) /
                        (e_tno_[ijk]->get(a_ijk) + e_tno_[ijk]->get(b_ijk) + e_tno_[ijk]->get(c_ijk) - (*F_lmo_)(i, i) -
                         (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                }
            }
        }

        double prefactor = ijk_scale_[ijk];
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        e_ijk_[ijk] += 8.0 * prefactor * V_ijk->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(V_ijk, k, j, i)->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(V_ijk, i, k, j)->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(V_ijk, j, i, k)->vector_dot(T_ijk);
        e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(V_ijk, j, k, i)->vector_dot(T_ijk);
        e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(V_ijk, k, i, j)->vector_dot(T_ijk);

        E_T0 += e_ijk_[ijk];

        // Step 4: Save Matrices (if doing full (T))
        if (store_amplitudes && !write_amplitudes_) {
            W_iajbkc_[ijk] = W_ijk;
            V_iajbkc_[ijk] = V_ijk;
            T_iajbkc_[ijk] = T_ijk;
        } else if (store_amplitudes && write_amplitudes_) {
#pragma omp critical
            W_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
#pragma omp critical
            V_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
#pragma omp critical
            T_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
        }
    }

    outfile->Printf("\n");
    outfile->Printf("  * DLPNO-CCSD(T0) Correlation Energy: %16.12f \n", e_lccsd_ + E_T0);
    outfile->Printf("  * DLPNO-(T0) Contribution:           %16.12f \n\n", E_T0);

    timer_off("LCCSD(T0)");

    return E_T0;
}

double DLPNOCCSD_T::compute_t_iteration_energy() {
    timer_on("Compute (T) Energy");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    double E_T = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(+ : E_T)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        int kji = i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i];
        int ikj = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j];
        int jik = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k];
        int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
        int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];

        double prefactor = ijk_scale_[ijk];
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        SharedMatrix V_ijk;
        SharedMatrix T_ijk;

        if (write_amplitudes_) {
            std::stringstream v_name;
            v_name << "V " << (ijk);
            V_ijk = std::make_shared<Matrix>(v_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
            V_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);

            std::stringstream t_name;
            t_name << "T " << (ijk);
            T_ijk = std::make_shared<Matrix>(t_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
            T_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
        } else {
            V_ijk = V_iajbkc_[ijk];
            T_ijk = T_iajbkc_[ijk];
        }

        e_ijk_[ijk] = 8.0 * prefactor * V_ijk->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(V_ijk, k, j, i)->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(V_ijk, i, k, j)->vector_dot(T_ijk);
        e_ijk_[ijk] -= 4.0 * prefactor * triples_permuter(V_ijk, j, i, k)->vector_dot(T_ijk);
        e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(V_ijk, j, k, i)->vector_dot(T_ijk);
        e_ijk_[ijk] += 2.0 * prefactor * triples_permuter(V_ijk, k, i, j)->vector_dot(T_ijk);

        E_T += e_ijk_[ijk];
    }

    timer_off("Compute (T) Energy");

    return E_T;
}

inline SharedMatrix DLPNOCCSD_T::triples_permuter(const SharedMatrix &X, int i, int j, int k, bool reverse) {
    SharedMatrix Xperm = X->clone();
    int ntno_ijk = X->rowspi(0);

    int perm_idx;
    if (i <= j && j <= k && i <= k) {
        perm_idx = 0;
    } else if (i <= k && k <= j && i <= j) {
        perm_idx = 1;
    } else if (j <= i && i <= k && j <= k) {
        perm_idx = 2;
    } else if (j <= k && k <= i && j <= i) {
        perm_idx = 3;
    } else if (k <= i && i <= j && k <= j) {
        perm_idx = 4;
    } else {
        perm_idx = 5;
    }

    for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
        for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
            for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                if (perm_idx == 0)
                    (*Xperm)(a_ijk, b_ijk *ntno_ijk + c_ijk) = (*X)(a_ijk, b_ijk * ntno_ijk + c_ijk);
                else if (perm_idx == 1)
                    (*Xperm)(a_ijk, b_ijk *ntno_ijk + c_ijk) = (*X)(a_ijk, c_ijk * ntno_ijk + b_ijk);
                else if (perm_idx == 2)
                    (*Xperm)(a_ijk, b_ijk *ntno_ijk + c_ijk) = (*X)(b_ijk, a_ijk * ntno_ijk + c_ijk);
                else if ((perm_idx == 3 && !reverse) || (perm_idx == 4 && reverse))
                    (*Xperm)(a_ijk, b_ijk *ntno_ijk + c_ijk) = (*X)(b_ijk, c_ijk * ntno_ijk + a_ijk);
                else if ((perm_idx == 4 && !reverse) || (perm_idx == 3 && reverse))
                    (*Xperm)(a_ijk, b_ijk *ntno_ijk + c_ijk) = (*X)(c_ijk, a_ijk * ntno_ijk + b_ijk);
                else
                    (*Xperm)(a_ijk, b_ijk *ntno_ijk + c_ijk) = (*X)(c_ijk, b_ijk * ntno_ijk + a_ijk);
            }
        }
    }

    return Xperm;
}

double DLPNOCCSD_T::lccsd_t_iterations() {
    timer_on("LCCSD(T) Iterations");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    outfile->Printf("\n  ==> Local CCSD(T) <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                         Corr. Energy    Delta E     Max R     Time (s)\n");

    int iteration = 1, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_prev = 0.0, r_curr = 0.0;
    bool e_converged = false, r_converged = false;

    double F_CUT = options_.get_double("F_CUT_T");
    double T_CUT_ITER = options_.get_double("T_CUT_ITER");

    std::vector<double> e_ijk_old(n_lmo_triplets, 0.0);

    while (!(e_converged && r_converged)) {
        // RMS of residual per single LMO, for assesing convergence
        std::vector<double> R_iajbkc_rms(n_lmo_triplets, 0.0);

        std::time_t time_start = std::time(nullptr);

#pragma omp parallel for schedule(dynamic)
        for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

            int ntno_ijk = n_tno_[ijk];

            if (std::fabs(e_ijk_[ijk] - e_ijk_old[ijk]) < std::fabs(e_ijk_old[ijk] * T_CUT_ITER)) continue;

            auto R_ijk = std::make_shared<Matrix>("R_ijk", ntno_ijk, ntno_ijk * ntno_ijk);

            if (ntno_ijk > 0) {
                SharedMatrix W_ijk;
                SharedMatrix T_ijk;

                if (write_amplitudes_) {
                    std::stringstream w_name;
                    w_name << "W " << (ijk);
                    W_ijk = std::make_shared<Matrix>(w_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
                    W_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);

                    std::stringstream t_name;
                    t_name << "T " << (ijk);
                    T_ijk = std::make_shared<Matrix>(t_name.str(), ntno_ijk, ntno_ijk * ntno_ijk);
#pragma omp critical
                    T_ijk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                } else {
                    W_ijk = W_iajbkc_[ijk];
                    T_ijk = T_iajbkc_[ijk];
                }
                R_ijk->copy(W_ijk);

                for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; ++b_ijk) {
                        for (int c_ijk = 0; c_ijk < ntno_ijk; ++c_ijk) {
                            (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) += (*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) *
                                ((*e_tno_[ijk])(a_ijk) + (*e_tno_[ijk])(b_ijk) + (*e_tno_[ijk])(c_ijk) 
                                    - (*F_lmo_)(i, i) - (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                        }
                    }
                }

                for (int l = 0; l < naocc; l++) {
                    int ijl_dense = i * naocc * naocc + j * naocc + l;
                    if (l != k && i_j_k_to_ijk_.count(ijl_dense)) {
                        int ijl = i_j_k_to_ijk_[ijl_dense];
                        if (n_tno_[ijl] == 0 || std::fabs((*F_lmo_)(l, k)) < F_CUT) continue;

                        auto S_ijk_ijl =
                            submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijl]);
                        S_ijk_ijl = linalg::triplet(X_tno_[ijk], S_ijk_ijl, X_tno_[ijl], true, false, false);

                        SharedMatrix T_ijl;
                        if (write_amplitudes_) {
                            std::stringstream t_name;
                            t_name << "T " << (ijl);
                            T_ijl = std::make_shared<Matrix>(t_name.str(), n_tno_[ijl], n_tno_[ijl] * n_tno_[ijl]);
#pragma omp critical
                            T_ijl->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                        } else {
                            T_ijl = T_iajbkc_[ijl];
                        }

                        auto T_temp1 =
                            matmul_3d(triples_permuter(T_ijl, i, j, l), S_ijk_ijl, n_tno_[ijl], n_tno_[ijk]);
                        C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, k), &(*T_temp1)(0, 0), 1,
                                &(*R_ijk)(0, 0), 1);
                    }

                    int ilk_dense = i * naocc * naocc + l * naocc + k;
                    if (l != j && i_j_k_to_ijk_.count(ilk_dense)) {
                        int ilk = i_j_k_to_ijk_[ilk_dense];
                        if (n_tno_[ilk] == 0 || std::fabs((*F_lmo_)(l, j)) < F_CUT) continue;

                        auto S_ijk_ilk =
                            submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ilk]);
                        S_ijk_ilk = linalg::triplet(X_tno_[ijk], S_ijk_ilk, X_tno_[ilk], true, false, false);

                        SharedMatrix T_ilk;
                        if (write_amplitudes_) {
                            std::stringstream t_name;
                            t_name << "T " << (ilk);
                            T_ilk = std::make_shared<Matrix>(t_name.str(), n_tno_[ilk], n_tno_[ilk] * n_tno_[ilk]);
#pragma omp critical
                            T_ilk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                        } else {
                            T_ilk = T_iajbkc_[ilk];
                        }

                        auto T_temp1 =
                            matmul_3d(triples_permuter(T_ilk, i, l, k), S_ijk_ilk, n_tno_[ilk], n_tno_[ijk]);
                        C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, j), &(*T_temp1)(0, 0), 1,
                                &(*R_ijk)(0, 0), 1);
                    }

                    int ljk_dense = l * naocc * naocc + j * naocc + k;
                    if (l != i && i_j_k_to_ijk_.count(ljk_dense)) {
                        int ljk = i_j_k_to_ijk_[ljk_dense];
                        if (n_tno_[ljk] == 0 || std::fabs((*F_lmo_)(l, i)) < F_CUT) continue;

                        auto S_ijk_ljk =
                            submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ljk]);
                        S_ijk_ljk = linalg::triplet(X_tno_[ijk], S_ijk_ljk, X_tno_[ljk], true, false, false);

                        SharedMatrix T_ljk;
                        if (write_amplitudes_) {
                            std::stringstream t_name;
                            t_name << "T " << (ljk);
                            T_ljk = std::make_shared<Matrix>(t_name.str(), n_tno_[ljk], n_tno_[ljk] * n_tno_[ljk]);
#pragma omp critical
                            T_ljk->load(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                        } else {
                            T_ljk = T_iajbkc_[ljk];
                        }

                        auto T_temp1 =
                            matmul_3d(triples_permuter(T_ljk, l, j, k), S_ijk_ljk, n_tno_[ljk], n_tno_[ijk]);
                        C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l, i), &(*T_temp1)(0, 0), 1,
                                &(*R_ijk)(0, 0), 1);
                    }
                }

                // => Update T3 Amplitudes <= //
                for (int a_ijk = 0; a_ijk < ntno_ijk; ++a_ijk) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; ++b_ijk) {
                        for (int c_ijk = 0; c_ijk < ntno_ijk; ++c_ijk) {
                            (*T_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) -= (*R_ijk)(a_ijk, b_ijk * ntno_ijk + c_ijk) /
                                ((*e_tno_[ijk])(a_ijk) + (*e_tno_[ijk])(b_ijk) + (*e_tno_[ijk])(c_ijk) 
                                    - (*F_lmo_)(i, i) - (*F_lmo_)(j, j) - (*F_lmo_)(k, k));
                        }
                    }
                }

                if (write_amplitudes_) {
#pragma omp critical
                    T_ijk->save(psio_, PSIF_DLPNO_TRIPLES, psi::Matrix::Full);
                }

            }
            R_iajbkc_rms[ijk] = R_ijk->rms();
        }

        // evaluate convergence
        e_prev = e_curr;
        e_ijk_old = e_ijk_;
        // Compute LCCSD(T) energy
        e_curr = compute_t_iteration_energy();

        double r_curr = *max_element(R_iajbkc_rms.begin(), R_iajbkc_rms.end());

        r_converged = fabs(r_curr) < options_.get_double("R_CONVERGENCE");
        e_converged = fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE");

        std::time_t time_stop = std::time(nullptr);

        outfile->Printf("  @LCCSD(T) iter %3d: %16.12f %10.3e %10.3e %8d\n", iteration, e_curr, e_curr - e_prev, r_curr, (int)time_stop - (int)time_start);

        iteration++;

        if (iteration > max_iteration) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }

    timer_off("LCCSD(T) Iterations");

    return e_curr;
}

double DLPNOCCSD_T::compute_energy() {
    timer_on("DLPNO-CCSD(T)");

    // Run DLPNO-CCSD
    double e_dlpno_ccsd = DLPNOCCSD::compute_energy();

    // Clear CCSD integrals
    K_mnij_.clear();
    K_bar_.clear();
    L_bar_.clear();
    J_ijab_.clear();
    L_iajb_.clear();
    M_iajb_.clear();
    K_tilde_chem_.clear();
    K_tilde_phys_.clear();
    L_tilde_.clear();
    Qab_ij_.clear();
    S_pno_ij_kj_.clear();

    bool scale_triples = options_.get_bool("SCALE_T0");

    print_header();

    double t_cut_tno_pre = options_.get_double("T_CUT_TNO_PRE");
    double t_cut_tno = options_.get_double("T_CUT_TNO");

    // Step 1: Recompute PNOs with converged CCSD amplitudes
    recompute_pnos();

    // Step 2: Perform the prescreening
    triples_sparsity(true);
    tno_transform(scale_triples, t_cut_tno_pre);
    double E_T0_pre = compute_lccsd_t0();

    // Step 3: Compute DLPNO-CCSD(T0) energy with surviving triplets
    triples_sparsity(false);
    tno_transform(scale_triples, t_cut_tno);
    double E_T0 = compute_lccsd_t0();
    e_lccsd_t_ = e_lccsd_ + E_T0 + de_lccsd_t_screened_;

    // Step 4: Compute full DLPNO-CCSD(T) energy if NOT using T0 approximation

    if (!options_.get_bool("T0_APPROXIMATION")) {
        outfile->Printf("\n\n  ==> Computing Full Iterative (T) <==\n\n");

        sort_triplets(E_T0);
        tno_transform(false, t_cut_tno);
        estimate_memory();

        double E_T0_crude = compute_lccsd_t0(true);
        double E_T_crude = lccsd_t_iterations();
        double dE_T = E_T_crude - E_T0_crude;

        outfile->Printf("  * Iterative (T) Contribution: %16.12f\n\n", dE_T);

        e_lccsd_t_ += dE_T;
    }

    double e_scf = reference_wavefunction_->energy();
    double e_ccsd_t_corr = e_lccsd_t_ + de_lmp2_weak_ + de_lmp2_eliminated_ + de_dipole_ + de_pno_total_;
    double e_ccsd_t_total = e_scf + e_ccsd_t_corr;

    set_scalar_variable("CCSD(T) CORRELATION ENERGY", e_ccsd_t_corr);
    set_scalar_variable("CURRENT CORRELATION ENERGY", e_ccsd_t_corr);
    set_scalar_variable("CCSD(T) TOTAL ENERGY", e_ccsd_t_total);
    set_scalar_variable("CURRENT ENERGY", e_ccsd_t_total);

    print_results();

    if (write_qab_pao_) {
        // Bye bye, you won't be missed
        psio_->close(PSIF_DLPNO_QAB_PAO, 0);
        psio_->close(PSIF_DLPNO_TRIPLES, 0);
    }

    timer_off("DLPNO-CCSD(T)");

    return e_ccsd_t_total;
}

void DLPNOCCSD_T::print_results() {
    double e_dlpno_ccsd = e_lccsd_ + de_lmp2_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_;
    double e_total = e_lccsd_t_ + de_lmp2_weak_ + de_lmp2_eliminated_ + de_pno_total_ + de_dipole_;
    outfile->Printf("  \n");
    outfile->Printf("  Total DLPNO-CCSD(T) Correlation Energy: %16.12f \n", e_total);
    outfile->Printf("    DLPNO-CCSD Contribution:              %16.12f \n", e_dlpno_ccsd);
    outfile->Printf("    DLPNO-(T) Contribution:               %16.12f \n", e_lccsd_t_ - e_lccsd_ - de_lccsd_t_screened_);
    outfile->Printf("    Screened Triplets Contribution:       %16.12f \n", de_lccsd_t_screened_);
    outfile->Printf("    Andy Jiang... FOR THREEEEEEEEEEE!!!\n\n\n");
    outfile->Printf("  @Total DLPNO-CCSD(T) Energy: %16.12f \n",
                    variables_["SCF TOTAL ENERGY"] + de_lmp2_weak_ + de_lmp2_eliminated_ + e_lccsd_t_ + de_pno_total_ + de_dipole_);
}

}  // namespace dlpno
}  // namespace psi