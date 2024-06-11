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

#ifndef libfmm_fmm_tree_H
#define libfmm_fmm_tree_H

#include "psi4/pragma.h"

#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_shell_pair.h"
#include "psi4/libfmm/fmm_box.h"

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Extent normalization constant (to calculate extents of shell pairs)
#define ENC (1.306076308436251)

namespace psi {

class Options;
class CFMMShellPair;
class CFMMBox;

class PSI_API CFMMTree {
    protected:
      // The molecule that this tree structure references
      std::shared_ptr<Molecule> molecule_;
      // The basis set that the molecule uses
      std::shared_ptr<BasisSet> primary_;
      // List of all the significant shell-pairs in the molecule
      std::vector<std::shared_ptr<CFMMShellPair>> primary_shell_pairs_;

      // Number of Levels in the CFMM Tree
      int nlevels_;
      // maximum allowable numbr of levels in the CFMM Tree
      static constexpr size_t max_nlevels_ = 6;
      // Maximum Multipole Angular Momentum
      int lmax_;

      // Number of target distributions per occupied leaf box for adaptive CFMM
      // see White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
      int M_target_;
      // Number of leaf boxes for adaptive CFMM
      // see White 1996 (https://doi.org/10.1016/0009-2614(96)00574-X)
      int N_target_;
      // Final factor by which to scale the root CFMM box
      double f_;
      // Scaling factor connecting number of target distributions to number of target boxes
      double g_; 
      // static constexpr double g_ = 0.523598; // pi/6, ratio of cube to sphere 
      // Dimensionality of system modeled by CFMM Tree
      // Always 3 for molecular systems
      static constexpr int dimensionality_ = 3;
 
      // The tree structure (implemented as vector for random access)
      std::vector<std::shared_ptr<CFMMBox>> tree_;
      // number of total boxes in tree
      size_t num_boxes_;
      // List of all the leaf boxes (sorted by number of shell pairs for parallel efficiency)
      std::vector<std::shared_ptr<CFMMBox>> sorted_leaf_boxes_;
      // Harmonic Coefficients used to calculate multipoles
      std::shared_ptr<HarmonicCoefficients> mpole_coefs_;
      // Numerical cutoff for ERI screening
      double cutoff_;

      // Options object
      Options& options_;
      // Number of threads
      int nthread_;
      // Print flag, defaults to 1
      int print_;
      // Bench flag, defaults to 0
      int bench_;

      // The integral objects used to compute the integrals
      std::vector<std::shared_ptr<TwoBodyAOInt>> ints_;

      // Number of shell pairs contained in boxes
      size_t nshp_;
      // Index from the shell-pair index to shell pair info (shellpair, box, nf_boxes)
      std::vector<std::vector<
        std::tuple<std::shared_ptr<CFMMShellPair>, std::shared_ptr<CFMMBox> 
                  >
      >> primary_shellpair_list_;
      // local far-field box pairs at a given level of the tree
      std::vector<std::vector<std::pair<std::shared_ptr<CFMMBox>, std::shared_ptr<CFMMBox>>>> lff_task_pairs_per_level_;
      // Number of ERI shell quartets computed, i.e., not screened out
      size_t num_computed_shells_;

      // Use density-based integral screening?
      bool density_screening_;
      // ERI Screening Tolerance
      double ints_tolerance_;

      // number of boxes per CFMM tree level
      std::array<size_t, max_nlevels_> level_to_box_count_;
      // number of shell pairs per CFMM tree level
      std::array<size_t, max_nlevels_> level_to_shell_count_;

      // => Functions related to CFMM Tree construction <= // 

      // Generate the whole tree
      void make_tree(int nshell_pairs = -1);
      // Make the root node of the CFMMTree
      void make_root_node();
      // Regenerate the root node of the CFMMTree for iterative tree construction
      std::tuple<bool, bool> regenerate_root_node();
      // Create children
      void make_children();
      // Sort the leaf nodes by number of shell-pairs
      void sort_leaf_boxes();
      // Set up near field and far field information for each box in the tree
      void setup_regions();
      // Setup shell-pair information and calculate multipoles for each shell-pair
      virtual void setup_shellpair_info();
      // Set up information on local far field task pairs per level
      void setup_local_far_field_task_pairs();
      // Calculate ALL the shell-pair multipoles at each leaf box
      virtual void calculate_shellpair_multipoles(bool is_primary = true) = 0;

      // => Functions called ONCE per iteration <= //

      // Calculate multipoles
      virtual void calculate_multipoles(const std::vector<SharedMatrix>& D);
      // Helper method to compute far field
      void compute_far_field();
      // Build near-field J (Direct SCF)
      virtual void build_nf_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
                      const std::vector<double>& Jmet_max) = 0;
      // Build far-field J (long-range multipole interactions)
      virtual void build_ff_J(std::vector<SharedMatrix>& J) = 0;
      // Kernel for building overall J matrix
      void J_build_kernel(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                     const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
		     bool do_incfock_iter, const std::vector<double>& Jmet_max);

      // => ERI Screening <= //
      bool shell_significant(int P, int Q, int R, int S, std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                             const std::vector<SharedMatrix>& D);

      // => Other functions <= //
      // calculate number of boxes and shell pairs at each level of the CFMM tree
      void generate_per_level_info();
      // change tree depth and sanity-check
      void set_nlevels(int nlevels);
 
    public:
      // Constructor (automatically sets up the tree)
      CFMMTree(std::shared_ptr<BasisSet> primary, Options& options);
      //CFMMTree(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

      // Build the J matrix of CFMMTree
      virtual void build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                    const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, 
                    bool do_incfock_iter = false, const std::vector<double>& Jmet_max = {}) = 0;
      // Returns the max tree depth
      int nlevels() { return nlevels_; }
      // Returns the max multipole AM
      int lmax() { return lmax_; }
      // return average number of distributions per occupied lowest-level box
      int distributions() { return level_to_shell_count_[nlevels_ - 1] / level_to_box_count_[nlevels_ - 1]; }
      // Return number of shell quartets actually computed
      size_t num_computed_shells() { return num_computed_shells_; };
      // Print the CFMM Tree out
      void print_out();

}; // End class CFMMTree

class PSI_API DirectCFMMTree : public CFMMTree {
    protected:
      void calculate_shellpair_multipoles(bool is_primary = true) override;
      
      // Build near-field J (Direct SCF)
      void build_nf_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
                      const std::vector<double>& Jmet_max) override;
      // Build far-field J (long-range multipole interactions)
      void build_ff_J(std::vector<SharedMatrix>& J) override;

    public:
      // Constructor (automatically sets up the tree)
      DirectCFMMTree(std::shared_ptr<BasisSet> primary, Options& options);
      //DirectCFMMTree(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

      // Build the J matrix of CFMMTree
      void build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                    const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J, 
                    bool do_incfock_iter = false, const std::vector<double>& Jmet_max = {}) override;
}; // End class DirectCFMMTree

class PSI_API DFCFMMTree : public CFMMTree{
    protected:
      std::shared_ptr<BasisSet> auxiliary_;
      // List of all the significant primary shell-pairs in the molecule (U_SHELL, V_SHELL), U >= V
      //std::vector<std::shared_ptr<ShellPair>> primary_shell_pairs_;
      // List of all the significant auxiliary shell-pairs in the molecule (SHELL, 0)
      std::vector<std::shared_ptr<CFMMShellPair>> auxiliary_shell_pairs_;
      // What type of contraction is being performed? (Inferred by input parameters)
      ContractionType contraction_type_;

      // Index from the shell-pair index to shell pair info (shellpair, box, nf_boxes)
      std::vector<std::vector<
        std::tuple<std::shared_ptr<CFMMShellPair>, std::shared_ptr<CFMMBox> 
                  >
      >> auxiliary_shellpair_list_;
      // List of all the near field boxes that belong to a given primary shell-pair
      std::vector<std::vector<std::shared_ptr<CFMMBox>>> primary_shellpair_to_nf_boxes_;
      // List of all the near field boxes that belong to a given auxiliary shell-pair
      std::vector<std::vector<std::shared_ptr<CFMMBox>>> auxiliary_shellpair_to_nf_boxes_;

      // Calculate the shell-pair multipoles at each leaf box (primary or auxiliary)
      void calculate_shellpair_multipoles(bool is_primary = true) override;

      // Setup shell-pair information and calculate multipoles for each shell-pair
      virtual void setup_shellpair_info();

      // => Functions called ONCE per iteration <= //

      // Calculate multipoles
      void calculate_multipoles(const std::vector<SharedMatrix>& D) override;
      // Build near-field J (Gateway function, links to specific J builds based on contraction 
      void build_nf_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
		      const std::vector<double>& Jmet_max) override;
      // Build gammaP's near field (gammaP = (P|uv)Duv)
      void build_nf_gamma_P(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
                      const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
                      const std::vector<double>& Jmet_max);
      // Build density-fitted J's near field (Jpq = (pq|Q)*gammaQ)
      //void build_nf_df_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
      //                const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
	//	      const std::vector<double>& Jmet_max);
      // Builds the near field interactions of the Coulomb metric with an auxiliary density
      //void build_nf_metric(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints,
      //                const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J);

      // Build far-field J (long-range multipole interactions)
      void build_ff_J(std::vector<SharedMatrix>& J) override;


    public:
      /// Constructor (automatically sets up the tree)
      /// Pass in null pointer if no primary or auxiliary
      //DFCFMMTree(std::shared_ptr<BasisSet> primary, Options& options);
      DFCFMMTree(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options);

      // Build the J matrix of DFCFMMTree
      void build_J(std::vector<std::shared_ptr<TwoBodyAOInt>>& ints, 
                    const std::vector<SharedMatrix>& D, std::vector<SharedMatrix>& J,
		    bool do_incfock_iter = false, const std::vector<double>& Jmet_max = {}) override;

      // Flip the contraction type (for DF integrals)
      void set_contraction(ContractionType contraction_type);
}; // End class DFCFMMTree

} // namespace psi

#endif
