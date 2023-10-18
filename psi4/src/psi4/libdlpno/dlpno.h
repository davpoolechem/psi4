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

#ifndef PSI4_SRC_DLPNO_H_
#define PSI4_SRC_DLPNO_H_

#include "sparse.h"

#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsio/psio.h"
#include "psi4/psifiles.h"

#include <map>
#include <tuple>
#include <string>
#include <unordered_map>

namespace psi {
namespace dlpno {

enum VirtualStorage { CORE, DIRECT };

enum AlgorithmType { MP2, CCSD, CCSD_T };

// Equations refer to Pinski et al. (JCP 143, 034108, 2015; DOI: 10.1063/1.4926879)

class DLPNOBase : public Wavefunction {
    protected:
      /// what quantum chemistry module are we running
      AlgorithmType algorithm_;
      /// threshold for PAO domain size
      double T_CUT_DO_;
      /// threshold for PNO truncation
      double T_CUT_PNO_;
      /// tolerance to separate pairs into CCSD and MP2 pairs
      double T_CUT_PAIRS_;
      /// tolerance to separate MP2 pairs in between crude and refined prescreening
      double T_CUT_PAIRS_MP2_;
      /// tolerance for energy of a pair for it to be considered a "dipole pair"
      double T_CUT_PRE_;
      /// tolerance for local density fitting (by Mulliken population)
      double T_CUT_MKN_;
      /// tolerance for eigenvalue decomposition of (Q|u_t v_t) integrals
      double T_CUT_EIG_;
      /// tolerance for singular value decomposition of ERI quantities
      double T_CUT_SVD_;
      /// T_CUT_PNO scaling factor for diagonal PNOs
      double T_CUT_PNO_DIAG_SCALE_;
      /// Tolerance for TNO truncation (by occupation number)
      double T_CUT_TNO_;

      /// auxiliary basis
      std::shared_ptr<BasisSet> ribasis_;
      SharedMatrix full_metric_;

      /// localized molecular orbitals (LMOs)
      SharedMatrix C_lmo_;
      SharedMatrix F_lmo_;
      SharedMatrix H_lmo_;

      /// projected atomic orbitals (PAOs)
      SharedMatrix C_pao_;
      SharedMatrix F_pao_;
      SharedMatrix S_pao_;
      SharedMatrix H_pao_;

      // LMO/PAO Hamiltonian (Used in T1-Hamiltonian CCSD)
      SharedMatrix H_lmo_pao_;
      SharedMatrix F_lmo_pao_;

      /// differential overlap integrals (EQ 4)
      SharedMatrix DOI_ij_; // LMO/LMO
      SharedMatrix DOI_iu_; // LMO/PAO

      // approximate LMO/LMO pair energies from dipole integrals (EQ 17)
      // used to screen out and estimate weakly interacting LMO/LMO pairs
      SharedMatrix dipole_pair_e_; ///< actual approximate pair energy (used in final energy calculation)
      SharedMatrix dipole_pair_e_bound_; ///< upper bound to approximate pair energies (used for screening)

      /// How much memory is used by storing each of the DF integral types
      size_t qij_memory_;
      size_t qia_memory_;
      size_t qab_memory_;
      /// Write (Q | u v) integrals to disk?
      bool write_qab_pao_;

      /// LMO/LMO three-index integrals
      std::vector<SharedMatrix> qij_;
      /// LMO/PAO three-index integrals
      std::vector<SharedMatrix> qia_;
      /// PAO/PAO three-index integrals
      std::vector<SharedMatrix> qab_;

      /// pair natural orbitals (PNOs)
      std::vector<SharedMatrix> K_iajb_;  ///< exchange operators (i.e. (ia|jb) integrals)
      std::vector<SharedMatrix> T_iajb_;  ///< amplitudes
      std::vector<SharedMatrix> Tt_iajb_; ///< antisymmetrized amplitudes
      std::vector<SharedMatrix> X_pno_;   ///< global PAO -> canonical PNO transforms
      std::vector<SharedVector> e_pno_;   ///< PNO orbital energies
      std::vector<int> n_pno_;       ///< number of pnos
      std::vector<double> de_pno_;   ///< PNO truncation energy error
      std::vector<double> de_pno_os_;   ///< opposite-spin contributions to de_pno_
      std::vector<double> de_pno_ss_;   ///< same-spin contributions to de_pno_

      /// pre-screening energies
      double de_dipole_; ///< energy correction for distant (LMO, LMO) pairs
      double e_lmp2_non_trunc_; ///< LMP2 energy in a pure PAO basis (Strong and Weak Pairs Only)
      double e_lmp2_trunc_; ///< LMP2 energy computed with (truncated) PNOs (Strong Pairs Only)
      double de_lmp2_eliminated_; ///< LMP2 correction for eliminated pairs (surviving pairs after dipole screening that
      // are neither weak nor strong)
      double de_lmp2_weak_; ///< LMP2 correction for weak pairs (only for CC)
      double de_pno_total_; ///< energy correction for PNO truncation
      double de_pno_total_os_; ///< energy correction for PNO truncation
      double de_pno_total_ss_; ///< energy correction for PNO truncation

      // => Sparse Maps <= //

      // orbital / aux bases
      SparseMap atom_to_bf_; ///< which orbital BFs are on a given atom?
      SparseMap atom_to_ribf_; ///< which aux BFs are on a given atom?
      SparseMap atom_to_shell_; ///< which orbital basis shells are on a given atom?
      SparseMap atom_to_rishell_; ///< which aux basis shells are on a given atom?

      // AO to LMO/PAO
      SparseMap lmo_to_bfs_;
      SparseMap lmo_to_atoms_;
      SparseMap pao_to_bfs_;
      SparseMap pao_to_atoms_;

      // LMO domains
      SparseMap lmo_to_ribfs_; ///< which aux BFs are needed for density-fitting a LMO?
      SparseMap lmo_to_riatoms_; ///< aux BFs on which atoms are needed for density-fitting a LMO?
      SparseMap lmo_to_paos_; ///< which PAOs span the virtual space of a LMO?
      SparseMap lmo_to_paoatoms_; ///< PAOs on which atoms span the virtual space of a LMO?
      std::vector<std::vector<int>> i_j_to_ij_; ///< LMO indices (i, j) to significant LMO pair index (ij); insignificant (i, j) maps to -1
      std::vector<std::pair<int,int>> ij_to_i_j_; ///< LMO pair index (ij) to both LMO indices (i, j)
      std::vector<int> ij_to_ji_; ///< LMO pair index (ij) to LMO pair index (ji)

      // LMO Pair Domains
      SparseMap lmopair_to_ribfs_; ///< which aux BFs are needed for density-fitting a pair of LMOs?
      SparseMap lmopair_to_riatoms_; ///< aux BFs on which atoms are needed for density-fitting a pair of LMOs?
      SparseMap lmopair_to_paos_; ///< which PAOs span the virtual space of a pair of LMOs?
      SparseMap lmopair_to_paoatoms_; ///< PAOs on which atoms span the virtual space of a pair of LMOs?
      SparseMap lmopair_to_lmos_; ///< Which LMOs "interact" with an LMO pair (determined by DOI integrals)

      // Extended LMO Domains 
      SparseMap lmo_to_riatoms_ext_; ///< aux BFs on which atoms are needed for density-fitting a LMO and all connected LMOs
      SparseMap riatom_to_lmos_ext_; ///< the extended DF domains of which LMOs include aux BFs on an atom
      SparseMap riatom_to_paos_ext_; ///< the extended DF domains of which PAOs include aux BFs on an atom
      SparseMap riatom_to_atoms1_; ///< orbital BFs on which atoms are needed for DF int transform (first index)
      SparseMap riatom_to_shells1_; ///< which shells of orbital BFs are needed for DF int transform (first index)
      SparseMap riatom_to_bfs1_; ///< which orbital BFs are needed for DF int transform (first index)
      SparseMap riatom_to_atoms2_; ///< orbital BFs on which atoms are needed for DF int transform (second index)
      SparseMap riatom_to_shells2_; ///< which shells of orbital BFs are needed for DF int transform (second index)
      SparseMap riatom_to_bfs2_; ///< which orbital BFs are needed for DF int transform (second index)

      // Dense analogues of some sparse maps for quick lookup
      std::vector<std::vector<int>> riatom_to_lmos_ext_dense_;
      std::vector<std::vector<int>> riatom_to_paos_ext_dense_;
      std::vector<std::vector<bool>> riatom_to_atoms1_dense_;
      std::vector<std::vector<bool>> riatom_to_atoms2_dense_;
      std::vector<std::vector<int>> lmopair_to_lmos_dense_;

      /// Useful for generating DF integrals
      std::vector<std::vector<std::vector<int>>> lmopair_lmo_to_riatom_lmo_;
      std::vector<std::vector<std::vector<int>>> lmopair_pao_to_riatom_pao_;

      /// PSIO object (to help with reading/writing large tensors)
      std::shared_ptr<PSIO> psio_;

      void common_init();

      // Helper functions
      void C_DGESV_wrapper(SharedMatrix A, SharedMatrix B);

      std::pair<SharedMatrix, SharedVector> canonicalizer(SharedMatrix C, SharedMatrix F);
      std::pair<SharedMatrix, SharedVector> orthocanonicalizer(SharedMatrix S, SharedMatrix F);

      SharedVector flatten_mats(const std::vector<SharedMatrix>& mat_list);

      void copy_flat_mats(SharedVector flat, std::vector<SharedMatrix>& mat_list);

      /// Form LMOs, PAOs, etc.
      void setup_orbitals();

      /// Compute approximate MP2 pair energies for distant LMOs using dipole integrals (EQ 17)
      void compute_dipole_ints();

      /// Compute differential overlap integrals between LMO/LMO and LMO/PAO pairs (EQ 4), DOI_ij and DOI_iu
      void compute_overlap_ints();

      /// Use dipole and overlap integrals to assess sparsity relationships between LMOs and estimate
      /// energy contribution of weakly interacting LMO pairs. Additionally, the overlap integrals
      /// and LMO sparsity are used to construct domains of PAOs, RI basis functions, and orbital
      /// basis functions for each LMO. These domains are necessary for efficient evaluation of
      /// three-index integrals.
      void prep_sparsity(bool initial, bool last);

      /// Compute the auxiliary metric (P|Q)
      void compute_metric();
      /// Compute three-index integrals in LMO/LMO basis with linear scaling
      void compute_qij();
      /// Compute three-index integrals in LMO/PAO basis with linear scaling (EQ 11)
      void compute_qia();
      /// Compute three-index integrals in PAO/PAO basis with linear scaling
      void compute_qab();

      /// form pair exch operators (EQ 15) and SC amplitudes (EQ 18); transform to PNO basis
      void pno_transform();

      /// Printing
      void print_aux_domains();
      void print_pao_domains();
      void print_lmo_domains();
      void print_aux_pair_domains();
      void print_pao_pair_domains();

    public:
      DLPNOBase(SharedWavefunction ref_wfn, Options& options);
      ~DLPNOBase() override;

      double compute_energy() override;
};

class DLPNOMP2 : public DLPNOBase {
   protected:
    /// PNO overlap integrals
    std::vector<std::vector<SharedMatrix>> S_pno_ij_kj_; ///< pno overlaps
    std::vector<std::vector<SharedMatrix>> S_pno_ij_ik_; ///< pno overlaps

    double e_lmp2_; ///< raw (uncorrected) local MP2 correlation energy
    double e_lmp2_ss_; ///< same-spin component of e_lmp2_
    double e_lmp2_os_; ///< opposite-spin component of e_lmp2_

    /// compute PNO/PNO overlap matrices for DLPNO-MP2
    void compute_pno_overlaps();
    
    /// compute MP2 correlation energy w/ current amplitudes (EQ 14)
    double compute_iteration_energy(const std::vector<SharedMatrix> &R_iajb);

    /// iteratively solve local MP2 equations  (EQ 13)
    void lmp2_iterations();

    void print_header();
    void print_results();
    void print_integral_sparsity();

   public:
    DLPNOMP2(SharedWavefunction ref_wfn, Options& options);
    ~DLPNOMP2() override;

    double compute_energy() override;
};

class DLPNOCCSD : public DLPNOBase {
   protected:
    /// Write (Q_ij | a_ij b_ij) integrals to disk?
    bool write_qab_pno_;

    /// Number of svd functions for PNO pair ij in rank-reduced (Q_ij |a_ij b_ij)
    std::vector<int> n_svd_;

    /// PNO overlap integrals
    std::vector<std::vector<SharedMatrix>> S_pno_ij_kj_; ///< pno overlaps
    std::vector<std::vector<SharedMatrix>> S_pno_ij_mn_; ///< pno overlaps

    /// Coupled-cluster amplitudes
    std::vector<SharedMatrix> T_ia_; ///< singles amplitudes
    std::vector<SharedMatrix> T_n_ij_; ///< singles amplitudes of LMO n_ij in PNO basis of ij (dim: n_lmo_pairs * nlmo_ij * npno_ij)

    // => Strong and Weak Pair Info <=//

    std::vector<std::vector<int>> i_j_to_ij_strong_;
    std::vector<std::pair<int,int>> ij_to_i_j_strong_;
    std::vector<int> ij_to_ji_strong_;

    std::vector<std::vector<int>> i_j_to_ij_weak_;
    std::vector<std::pair<int,int>> ij_to_i_j_weak_;
    std::vector<int> ij_to_ji_weak_;

    // => CCSD Integrals <= //

    /// (4 occupied, 0 virtual)
    std::vector<SharedMatrix> K_mnij_; /// (m i | n j)
    /// (3 occupied, 1 virtual)
    std::vector<SharedMatrix> K_bar_; /// (m i | b_ij j) [aka K_bar]
    std::vector<SharedMatrix> K_bar_chem_; /// (i j | m b_ij)
    std::vector<SharedMatrix> L_bar_; /// 2.0 * K_mbij - K_mbji
    /// (2 occupied, 2 virtual)
    std::vector<SharedMatrix> J_ijab_; /// (i j | a_ij b_ij)
    std::vector<SharedMatrix> L_iajb_; /// 2.0 * (i a_ij | j b_ij) - (i b_ij | j a_ij)
    std::vector<SharedMatrix> M_iajb_; /// 2.0 * (i a_ij | j b_ij) - (i j | b_ij a_ij)
    /// (1 occupied, 3 virtual)
    std::vector<SharedMatrix> K_tilde_chem_; /// (i e_ij | a_ij f_ij) [aka K_tilde] (stored as (e, a*f)) [Chemist's Notation]
    std::vector<SharedMatrix> K_tilde_phys_; /// (i e_ij | a_ij f_ij) [aka K_tilde] (stored as (a, e*f)) [Physicist's Notation]
    std::vector<SharedMatrix> L_tilde_; /// 2.0 * K_tilde_chem - K_tilde_phys
    /// (0 occupied, 4 virtual)

    // DF Integrals (Used in DLPNO-T1-CCSD)
    std::vector<std::vector<SharedMatrix>> Qma_ij_; // (q_ij | m_ij a_ij)
    std::vector<std::vector<SharedMatrix>> Qab_ij_; // (q_ij | a_ij b_ij)

    std::vector<SharedMatrix> i_Qk_ij_;   // (q_ij | k_ij i)
    std::vector<SharedMatrix> i_Qa_ij_;   // (q_ij | a_ij i)
    std::vector<SharedMatrix> i_Qk_t1_;   // (q_ij | k_ij i) [T1-dressed]
    std::vector<SharedMatrix> i_Qa_t1_;   // (q_ij | a_ij i) [T1-dressed]

    // Dressed Fock matrices (used in DLPNO-T1-CCSD)
    SharedMatrix Fkj_;
    std::vector<SharedMatrix> Fkc_;
    std::vector<SharedMatrix> Fai_;

    double e_lccsd_; ///< raw (uncorrected) local CCSD correlation energy

    /// Returns the appropriate overlap matrix given two LMO pairs
    inline SharedMatrix S_PNO(const int ij, const int mn);

    /// Determine which pairs are strong and weak pairs
    template<bool crude> void pair_prescreening(); // Encapsulates crude/refined prescreening step in Riplinger 2016
    template<bool crude> std::vector<double> compute_pair_energies();
    template<bool crude> std::pair<double, double> filter_pairs(const std::vector<double>& e_ijs);

    /// Runs preceeding DLPNO-MP2 computation before DLPNO-CCSD iterations
    void pno_lmp2_iterations();

    /// compute PNO/PNO overlap matrices for DLPNO-CCSD
    void compute_pno_overlaps();

    // => Computing integrals <= //

    /// A function to estimate integral memory costs
    void estimate_memory();
    /// Compute four-center integrals for CC computations
    void compute_cc_integrals();

    // => CCSD intermediates <= //

    /// compute Fmi intermediate (Madriaga Eq. 40)
    SharedMatrix compute_Fmi(const std::vector<SharedMatrix>& tau_tilde);
    /// compute Fbe intermediate (of diagonal LMO pair ii) (Madriaga Eq. 39)
    std::vector<SharedMatrix> compute_Fbe(const std::vector<SharedMatrix>& tau_tilde);
    /// compute Fme intermediate (of diagonal LMO pair mm) (Madriaga Eq. 41)
    std::vector<SharedMatrix> compute_Fme();
    /// compute Wmnij intermediate (Madriaga Eq. 43)
    std::vector<SharedMatrix> compute_Wmnij(const std::vector<SharedMatrix>& tau);
    /// compute Wmbej intermediate (Madriaga Eq. 44)
    std::vector<SharedMatrix> compute_Wmbej(const std::vector<SharedMatrix>& tau_bar);
    /// compute Wmbje intermediate (Madriaga Eq. 45)
    std::vector<SharedMatrix> compute_Wmbje(const std::vector<SharedMatrix>& tau_bar);

    // => T1-CCSD intermediates
    std::vector<SharedMatrix> compute_B_tilde();
    std::vector<SharedMatrix> compute_C_tilde();
    std::vector<SharedMatrix> compute_D_tilde();
    std::vector<SharedMatrix> compute_E_tilde();
    SharedMatrix compute_G_tilde();

    /// iteratively solve local CCSD equations
    void lccsd_iterations();

    /// compute T1-dressed DF integrals
    void t1_ints();
    /// compute T1-dressed Fock matrix intermediates
    void t1_fock();
    /// local CCSD equations (with the T1-transformation)
    void t1_lccsd_iterations();

    void print_header();
    void print_results();
    void print_integral_sparsity();
    
   public:
    DLPNOCCSD(SharedWavefunction ref_wfn, Options& options);
    ~DLPNOCCSD() override;

    double compute_energy() override;
};

class DLPNOCCSD_T : public DLPNOCCSD {
   private:
    // Sparsity information
    // WARNING: Only unique triplets are used
    SparseMap lmotriplet_to_ribfs_; ///< which ribfs are on an LMO triplet (i, j, k)
    SparseMap lmotriplet_to_lmos_; ///< which LMOs l form a significant pair with (i, j, or k)
    SparseMap lmotriplet_to_paos_; ///< which PAOs span the virtual space of a triplet of LMOs?
    std::unordered_map<int, int> i_j_k_to_ijk_; ///< LMO indices (i, j, k) to significant LMO triplet index (ijk), -1 if not found
    std::vector<std::tuple<int, int, int>> ijk_to_i_j_k_; ///< LMO triplet index (ijk) to LMO index tuple (i, j, k)

    std::vector<std::vector<std::vector<int>>> lmotriplet_lmo_to_riatom_lmo_;
    std::vector<std::vector<std::vector<int>>> lmotriplet_pao_to_riatom_pao_;

    /// triplet natural orbitals (TNOs)
    std::vector<SharedMatrix> W_iajbkc_; ///< W3 intermediate for each lmo triplet
    std::vector<SharedMatrix> V_iajbkc_; ///< V3 intermeidate for each lmo triplet
    std::vector<SharedMatrix> T_iajbkc_; ///< Triples amplitude for each lmo triplet
    std::vector<SharedMatrix> X_tno_; ///< global PAO -> canonical TNO transforms
    std::vector<SharedVector> e_tno_; ///< TNO orbital energies
    std::vector<int> n_tno_; ///< number of tnos per triplet domain
    std::vector<double> e_ijk_; ///< energy of triplet ijk (used for pre-screening and convergence purposes)
    std::vector<double> ijk_scale_; ///< scaling factor to apply to triplet energy ijk (based on MP2 scaling)
    std::vector<double> tno_scale_; ///< scaling factor to apply to each triplet to account for TNO truncation error
    std::vector<bool> is_strong_triplet_; ///< whether or not triplet is strong

    /// Write amplitudes to disk?
    bool write_amplitudes_ = false;

    /// final energies
    double de_lccsd_t_screened_; ///< energy contribution from screened triplets
    double e_lccsd_t_; ///< local (T) correlation energy

    /// Recompute PNOs (Pair Natural Orbitals) using CCSD densities
    void recompute_pnos();
    /// Create sparsity maps for triples
    void triples_sparsity(bool prescreening);
    /// Create TNOs (Triplet Natural Orbitals) for DLPNO-(T)
    void tno_transform(bool scale_triples, double tno_tolerance);
    /// Sort triplets to split between "strong" and "weak" triplets (for (T) iterations)
    void sort_triplets(double e_total);

    /// Returns a symmetrized version of that matrix (in i <= j <= k ordering)
    inline SharedMatrix triples_permuter(const SharedMatrix& X, int i, int j, int k, bool reverse=false);
    /// compute (T) iteration energy
    double compute_t_iteration_energy();

    /// L_CCSD(T0) energy
    double compute_lccsd_t0(bool store_amplitudes=false);
    /// A function to estimate Full-(T) memory costs
    void estimate_memory();
    /// L_CCSD(T) iterations
    double lccsd_t_iterations();

    void print_header();

    void print_results();

   public:
    DLPNOCCSD_T(SharedWavefunction ref_wfn, Options& options);
    ~DLPNOCCSD_T() override;

    double compute_energy() override;
};

}  // namespace dlpno
}  // namespace psi

#endif //PSI4_SRC_DLPNO_MP2_H_
