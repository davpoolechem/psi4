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

#ifndef libfmm_fmm_box_H
#define libfmm_fmm_box_H

#include "psi4/pragma.h"

#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libfmm/multipoles_helper.h"
#include "psi4/libfmm/fmm_shell_pair.h"

#include <optional>
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

class PSI_API CFMMBox : public std::enable_shared_from_this<CFMMBox> {
    protected:
      // Parent of the CFMMBox
      std::weak_ptr<CFMMBox> parent_;
      // Children of the CFMMBox
      std::vector<std::shared_ptr<CFMMBox>> children_;

      // The primary shell pairs belonging to this box (empty if none)
      std::vector<std::shared_ptr<CFMMShellPair>> primary_shell_pairs_;
      // The auxiliary shell pairs belonging to this box (empty if none)
      std::vector<std::shared_ptr<CFMMShellPair>> auxiliary_shell_pairs_;

      // The box's origin (lower-left-front corner)
      Vector3 origin_;
      // Center of the box
      Vector3 center_;
      // Length of the box
      double length_;
      // Level the box is at (0 = root)
      int level_;
      // Maximum Multipole Angular Momentum
      int lmax_;
      // Well-separatedness criterion for this box
      int ws_;
      // Maximum well-separatedness for any given shell in the box 
      // (same as ws_ except for the most diffuse boxes in the level)
      int ws_max_;

      // Number of threads the calculation is running on
      int nthread_;

      // Multipoles of the box (Density-Matrix contracted), one for each density matrix
      std::vector<std::shared_ptr<RealSolidHarmonics>> mpoles_;
      // Far field vector of the box, one for each density matrix
      std::vector<std::shared_ptr<RealSolidHarmonics>> Vff_;

      // A list of all the near-field boxes to this box
      std::vector<std::shared_ptr<CFMMBox>> near_field_;
      // A list of all of the local-far-field boxes to this box
      std::vector<std::shared_ptr<CFMMBox>> local_far_field_;

      // Returns a shared pointer to the CFMMBox object
      std::shared_ptr<CFMMBox> get() { return shared_from_this(); }
      
    public:
      // Generic Constructor
      CFMMBox(std::shared_ptr<CFMMBox> parent, 
              std::vector<std::shared_ptr<CFMMShellPair>> primary_shell_pairs, 
              std::vector<std::shared_ptr<CFMMShellPair>> auxiliary_shell_pairs,
              Vector3 origin, double length, int level, int lmax, int ws);
      CFMMBox(std::shared_ptr<CFMMBox> parent, 
              std::vector<std::shared_ptr<CFMMShellPair>> primary_shell_pairs, 
              Vector3 origin, double length, int level, int lmax, int ws);

      // Make children for this multipole box
      void make_children();
      // Sets the near field and local far field regions of the box
      void set_regions();

     // Compute multipoles for the box are contracted with the density matrix (depending on contraction type)
     void compute_multipoles(const std::vector<SharedMatrix>& D, std::optional<ContractionType> contraction_type);

      // Compute multipoles from children
      void compute_mpoles_from_children();
      // Computes the far field contribution from a far away sibling
      void compute_far_field_contribution(std::shared_ptr<CFMMBox> lff_box);
      // Compute the far field contibution from the parents
      void add_parent_far_field_contribution();

      // => USEFUL SETTER METHODS <= //

      // Set the maximum ws of the box
      void set_ws_max(int ws_max) { ws_max_ = ws_max; }

      // => USEFUL GETTER METHODS <= //

      // Get the multipole level the box is on
      int get_level() { return level_; }
      // Get the ws criterion of the box
      int get_ws() { return ws_; }
      // Get the value of a particular multipole (for the Nth density matrix)
      double get_mpole_val(int N, int l, int mu) { return mpoles_[N]->get_multipoles()[l][mu]; }
      // Get the far field value of a multipole (for the Nth density matrix)
      double get_Vff_val(int N, int l, int mu) { return Vff_[N]->get_multipoles()[l][mu]; }
      // Get the children of the box
      std::vector<std::shared_ptr<CFMMBox>>& get_children() { return children_; }
      // Get the bra shell pairs of the box
      std::vector<std::shared_ptr<CFMMShellPair>>& get_primary_shell_pairs() { return primary_shell_pairs_; }
      // Get the ket shell pairs of the box
      std::vector<std::shared_ptr<CFMMShellPair>>& get_auxiliary_shell_pairs() { return auxiliary_shell_pairs_; }
      // Gets the number of shell pairs in the box
      int nshell_pair() { return primary_shell_pairs_.size() + auxiliary_shell_pairs_.size(); }
      // Get the origin of this box
      Vector3 origin() { return origin_; }
      // Get the center of this box
      Vector3 center() { return center_; }
      // Get the length of one dimension of this box
      double length() { return length_; }
      // Gets the near_field_boxes of the box
      std::vector<std::shared_ptr<CFMMBox>>& near_field_boxes() { return near_field_; }
      // Gets the local far field boxes of the box
      std::vector<std::shared_ptr<CFMMBox>>& local_far_field_boxes() { return local_far_field_; }
      // Gets the far field vector
      std::vector<std::shared_ptr<RealSolidHarmonics>>& far_field_vector() { return Vff_; }
      
      // Print the CFMM Box out
      void print_out();

}; // End class CFMMBox

} // namespace psi

#endif
