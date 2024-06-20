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

    density_screening_ = options_.get_str("SCREENING") == "DENSITY";
    
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
        gamma[i] = std::make_shared<Matrix>("gammaP_DFCFMM", naux, 1);
    }

    // Build gammaP = (P|uv)Duv
    cfmmtree_->set_contraction(ContractionType::DF_AUX_PRI);
    cfmmtree_->build_J(eri_computers, D, gamma, incfock_iter_, Jmet_max_);

    // Solve for gammaQ => (P|Q)*gammaQ = gammaP
    outfile->Printf("#==================# \n");
    outfile->Printf("#== Start GammaQ ==# \n");
    outfile->Printf("#==================# \n\n");
    for (int i = 0; i < D.size(); i++) {
        std::vector<int> ipiv(naux);

        outfile->Printf("  Ind = %d \n", i);
        outfile->Printf("  -------- \n");

        gamma[i]->save(gamma[i]->name() + ".dat", false, false, true);

        C_DGESV(naux, 1, J_metric_->clone()->pointer()[0], naux, ipiv.data(), gamma[i]->pointer()[0], naux);

        gamma[i]->set_name("gammaQ_DFCFMM");
        gamma[i]->print_out();
        gamma[i]->save(gamma[i]->name() + ".dat", false, false, true);
        outfile->Printf("  H[%i] Absmax: %f\n\n", i, gamma[i]->absmax());
    }
    outfile->Printf("#==================# \n");
    outfile->Printf("#==  End GammaQ  ==# \n");
    outfile->Printf("#==================# \n\n");
 
    // Build Juv = (uv|Q) * gammaQ
    cfmmtree_->set_contraction(ContractionType::DF_PRI_AUX);
    cfmmtree_->build_J(eri_computers, gamma, G_comp, incfock_iter_, Jmet_max_);

    for (int i = 0; i < D.size(); i++) {
        G_comp[i]->save(G_comp[i]->name() + "_DFCFMM.dat", false, false, true);
    }
}

} // end namespace psi
