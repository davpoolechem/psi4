#include "psi4/libfock/jk.h"
#include "psi4/libfock/SplitJK.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libqt/qt.h"
#include "psi4/libfmm/fmm_tree.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#include "psi4/libpsi4util/process.h"
#endif

using namespace psi;

namespace psi {

DFCFMM::DFCFMM(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options) : SplitJK(primary, options), auxiliary_(auxiliary) {
    timer_on("DFCFMM: Setup");

    // => General Setup <= //

    // thread count
    nthreads_ = 1;
#ifdef _OPENMP
    nthreads_ = Process::environment.get_n_threads();
#endif

    // pre-compute coulomb fitting metric
    timer_on("DFCFMM: Coulomb Metric");

    FittingMetric J_metric_obj(auxiliary_, true);
    J_metric_obj.form_fitting_metric();
    J_metric_ = J_metric_obj.get_metric();

    // build Coulomb metric maximums vector
    Jmet_max_= std::vector<double>(auxiliary_->nshell(), 0.0);
    double **Jmetp = J_metric_->pointer();

    if (density_screening_) {
#pragma omp parallel for
        for (size_t P = 0; P < auxiliary_->nshell(); P++) {
            int p_start = auxiliary_->shell_to_basis_function(P);
            int num_p = auxiliary_->shell(P).nfunction();
            for (size_t p = p_start; p < p_start + num_p; p++) {
                Jmet_max_[P] = std::max(Jmet_max_[P], Jmetp[p][p]);
            }
        }
    }
    
    timer_off("DFCFMM: Coulomb Metric");

    cfmmtree_ = std::make_shared<DFCFMMTree>(primary_, auxiliary_, options_);
    
    timer_off("DFCFMM: Setup");
}

DFCFMM::~DFCFMM() {}

size_t DFCFMM::num_computed_shells() {
    return cfmmtree_->num_computed_shells();
}

void DFCFMM::print_header() const {
    if (print_) {
        outfile->Printf("\n");
        outfile->Printf("  ==> Density-Fitted Continuous Fast Multipole Method (DF-CFMM) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", cfmmtree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", cfmmtree_->nlevels());
        outfile->Printf("    Shell Pairs/Box: %11d\n", cfmmtree_->distributions());
    }
}

void DFCFMM::build_G_component(std::vector<std::shared_ptr<Matrix> >& D,
     std::vector<std::shared_ptr<Matrix> >& G_comp,
     std::vector<std::shared_ptr<TwoBodyAOInt> >& eri_computers) { 
  
    // setup general varibales
    int naux = auxiliary_->nbf();
  
    std::vector<SharedMatrix> gamma(D.size());
    for (int i = 0; i < D.size(); i++) {
        gamma[i] = std::make_shared<Matrix>(naux, 1);
    }

    // Build gammaP = (P|uv)Duv
    cfmmtree_->set_contraction(ContractionType::DF_AUX_PRI);
    cfmmtree_->build_J(eri_computers, D, gamma, incfock_iter_, Jmet_max_);

/*
    // Solve for gammaQ => (P|Q)*gammaQ = gammaP
    for (int i = 0; i < D.size(); i++) {
        SharedMatrix Jmet_copy = Jmet_->clone();
        std::vector<int> ipiv(naux);

        C_DGESV(naux, 1, Jmet_copy->pointer()[0], naux, ipiv.data(), gamma[i]->pointer()[0], naux);
    }

    // Build Juv = (uv|Q) * gammaQ
    cfmmtree_->df_set_contraction(ContractionType::DF_PRI_AUX);
    cfmmtree_->build_J(eri_computers, gamma, J, incfock_iter_, Jmet_max_);
*/
}

} // end namespace psi