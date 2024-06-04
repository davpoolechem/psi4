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

#ifndef libfmm_fmm_shell_pair_H
#define libfmm_fmm_shell_pair_H

#include "psi4/pragma.h"

#include "psi4/libmints/vector3.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libfmm/multipoles_helper.h"

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

// used by CFMMBox and CFMMTree
enum class ContractionType {
    DIRECT,
    DF_AUX_PRI,
    DF_PRI_AUX,
    METRIC
};

class PSI_API CFMMShellPair {
    protected:
      // The basis set of the first shell-pair index
      std::shared_ptr<BasisSet> bs1_;
      // The basis set of the second shell-pair index
      std::shared_ptr<BasisSet> bs2_;
      // The index of the shell-pair
      std::pair<int, int> pair_index_;
      // Exponent of most diffuse basis function in shell pair
      double exp_;
      // Center of shell pair (As defined in bagel FMM as the average)
      Vector3 center_;
      // Radial extent of shell pair
      double extent_;
      // The multipole moments (per basis pair (pq) the shell pair (PQ)), centered at the lowest level box the shell belongs to
      std::vector<std::shared_ptr<RealSolidHarmonics>> mpoles_;
      // Multipole coefficients of shellpair
      std::shared_ptr<HarmonicCoefficients> mpole_coefs_;

    public:
      CFMMShellPair(std::shared_ptr<BasisSet>& bs1, std::shared_ptr<BasisSet>& bs2, 
                std::pair<int, int> pair_index, 
                std::shared_ptr<HarmonicCoefficients>& mpole_coefs, double cfmm_extent_tol);

      // Calculate the multipole moments of the Shell-Pair about a center
      void calculate_mpoles(Vector3 box_center, std::shared_ptr<OneBodyAOInt>& s_ints,
                            std::shared_ptr<OneBodyAOInt>& mpole_ints, int lmax);

      // Returns the shell pair index
      std::pair<int, int> get_shell_pair_index() { return pair_index_; }
      // Returns the center of the shell pair
      Vector3 get_center() { return center_; }
      // Returns the radial extent of the shell pair
      double get_extent() { return extent_; }
      // Returns the multipole moments of the shell pairs about a center
      std::vector<std::shared_ptr<RealSolidHarmonics>>& get_mpoles() { return mpoles_; }
      // Returns bs1 of shell pair
      std::shared_ptr<BasisSet> bs1() { return bs1_; }
      // Returns bs2 of shell pair
      std::shared_ptr<BasisSet> bs2() { return bs2_; }
};

} // namespace psi

#endif
