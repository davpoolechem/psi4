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

CFMM::CFMM(std::shared_ptr<BasisSet> primary, Options& options) : SplitJK(primary, options) {
    outfile->Printf("BEGIN MAKE CFMM TREE \n");
    cfmmtree_ = std::make_shared<CFMMTree>(primary_, options_);
    outfile->Printf("END MAKE CFMM TREE \n");
}

CFMM::~CFMM() {}

size_t CFMM::num_computed_shells() {
    return cfmmtree_->num_computed_shells();
}

void CFMM::print_header() const {
    if (print_) {
        outfile->Printf("\n");
        outfile->Printf("  ==> Continuous Fast Multipole Method (CFMM) <==\n\n");
        outfile->Printf("    Primary Basis: %11s\n", primary_->name().c_str());
        outfile->Printf("    Max Multipole Order: %11d\n", cfmmtree_->lmax());
        outfile->Printf("    Max Tree Depth: %11d\n", cfmmtree_->nlevels());
    }
}

void CFMM::build_G_component(std::vector<std::shared_ptr<Matrix> >& D,
     std::vector<std::shared_ptr<Matrix> >& G_comp,
     std::vector<std::shared_ptr<TwoBodyAOInt> >& eri_computers) { 
    //timer_on("CFMM: J");

    //outfile->Printf("BEGIN BUILD J \n");
    cfmmtree_->build_J(eri_computers, D, G_comp);
    //outfile->Printf("END BUILD J");

    //timer_off("CFMM: J");
}

} // end namespace psi
