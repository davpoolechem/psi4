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

double sign(double x) {
    if (std::abs(x) < 1.0e-8) {
        return 0.0;
    } else if (x < 0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

CFMMBox::CFMMBox(std::shared_ptr<CFMMBox> parent, 
              std::vector<std::shared_ptr<CFMMShellPair>> primary_shell_pairs, 
              std::vector<std::shared_ptr<CFMMShellPair>> auxiliary_shell_pairs, 
              Vector3 origin, double length, int level, int lmax, int ws) {
                  
    parent_ = parent;
    primary_shell_pairs_ = primary_shell_pairs;
    auxiliary_shell_pairs_ = auxiliary_shell_pairs;
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

CFMMBox::CFMMBox(std::shared_ptr<CFMMBox> parent, 
              std::vector<std::shared_ptr<CFMMShellPair>> primary_shell_pairs, 
              Vector3 origin, double length, int level, int lmax, int ws) : 
    CFMMBox(parent, primary_shell_pairs, {}, origin, length, level, lmax, ws) { } 

void CFMMBox::make_children() {

    int nchild = (level_ > 0) ? 16 : 8;
    std::vector<std::vector<std::shared_ptr<CFMMShellPair>>> child_shell_pair_primary_buffer(nchild);
    std::vector<std::vector<std::shared_ptr<CFMMShellPair>>> child_shell_pair_auxiliary_buffer(nchild);

    // Max WS at the child's level
    int child_level_max_ws = std::max(2, (int) std::pow(2, level_+1));
    int diffuse_child_max_ws = child_level_max_ws;

    // Fill order (ws,z,y,x) (0)000 (0)001 (0)010 (0)011 (0)100 (0)101 (0)110 (0)111
    // (1)000 (1)001 (1)010 (1)011 (1)100 (1)101 (1)110 (1)111
    for (std::shared_ptr<CFMMShellPair> shell_pair : primary_shell_pairs_) {
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
        child_shell_pair_primary_buffer[boxind].push_back(shell_pair);

        int child_ws = std::max(2, (int)std::ceil(2.0 * extent / length_));
        if (child_ws > diffuse_child_max_ws) diffuse_child_max_ws = child_ws;
    }

    for (std::shared_ptr<CFMMShellPair> shell_pair : auxiliary_shell_pairs_) {
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
        child_shell_pair_auxiliary_buffer[boxind].push_back(shell_pair);

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
        children_.push_back(std::make_shared<CFMMBox>(this->get(), child_shell_pair_primary_buffer[boxind], 
                                                      child_shell_pair_auxiliary_buffer[boxind],   
                                                      new_origin, 0.5 * length_, level_ + 1, lmax_, child_ws));
        if (child_ws == child_level_max_ws) children_[boxind]->set_ws_max(diffuse_child_max_ws);
    }
}

void CFMMBox::set_regions() {

    // Creates a temporary parent shared pointer
    std::shared_ptr<CFMMBox> parent = parent_.lock();
    // Maximum possible WS for a box at this level
    //int max_ws = std::max(2, (int)std::pow(2, level_));

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

// TODO: THIS
void CFMMBox::compute_multipoles(const std::vector<SharedMatrix>& D, std::optional<ContractionType> contraction_type) {
    if (mpoles_.size() == 0) {
        mpoles_.resize(D.size());
        Vff_.resize(D.size());
    }

    // Create multipoles and far field vectors for each density matrix
    for (int N = 0; N < D.size(); N++) {
        mpoles_[N] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Regular);
        Vff_[N] = std::make_shared<RealSolidHarmonics>(lmax_, center_, Irregular);
    }

    bool is_primary;
    if (!contraction_type.has_value()) { // default/null contraction type uses primary basis
        //is_primary = (contraction_type == ContractionType::DF_AUX_PRI || contraction_type == ContractionType::DIRECT);
        is_primary = (contraction_type == ContractionType::DF_AUX_PRI);
    } else {
        is_primary = true;
    }

    std::vector<std::shared_ptr<CFMMShellPair>>& ref_shell_pairs = (is_primary) ? primary_shell_pairs_ : auxiliary_shell_pairs_;
    if (ref_shell_pairs.empty()) return;

    std::shared_ptr<BasisSet> bs1 = ref_shell_pairs[0]->bs1();
    std::shared_ptr<BasisSet> bs2 = ref_shell_pairs[0]->bs2();
   
    int nbf = (is_primary) ? bs1->nbf() : 1;

    // Compute and contract the ket multipoles with the density matrix to get box multipoles
    for (const auto& sp : ref_shell_pairs) {

        std::vector<std::shared_ptr<RealSolidHarmonics>>& sp_mpoles = sp->get_mpoles();
        
        auto [P, Q] = sp->get_shell_pair_index();

        double prefactor = (P == Q) ? 1.0 : 2.0;

        const GaussianShell& Pshell = bs1->shell(P);
        const GaussianShell& Qshell = bs2->shell(Q);

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

void CFMMBox::print_out() {
    auto sp = this->get_primary_shell_pairs();
    //auto sp = this->get_auxiliary_shell_pairs();
    int nshells = sp.size();
    int level = this->get_level();
    int ws = this->get_ws();
    auto center = this->center() * bohr2ang; 
    auto length = this->length() * bohr2ang;
    if (nshells >= 0) {
        for (int ilevel = 0; ilevel != level; ++ilevel) {
            outfile->Printf("  ");
        }
        outfile->Printf("    Tree: %d, WS: %d, Center: (%f, %f, %f), Length: %f, Num. Shell Pairs: %d \n", level, ws, center[0], center[1], center[2], length, nshells);
    }

    for (auto& child : this->get_children()) {
       child->print_out();
    }
} 

} // end namespace psi
