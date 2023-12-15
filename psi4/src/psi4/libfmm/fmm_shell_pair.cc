#include "psi4/pragma.h"

#include "psi4/libfmm/multipoles_helper.h"
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

CFMMShellPair::CFMMShellPair(std::shared_ptr<BasisSet>& basisset, std::pair<int, int> pair_index, 
                     std::shared_ptr<HarmonicCoefficients>& mpole_coefs, double cfmm_extent_tol) {
    basisset_ = basisset;
    pair_index_ = pair_index;

    const GaussianShell& Pshell = basisset_->shell(pair_index.first);
    const GaussianShell& Qshell = basisset_->shell(pair_index.second);

    Vector3 pcenter = Pshell.center();
    Vector3 qcenter = Qshell.center();

    center_ = Vector3(0.0, 0.0, 0.0);
    exp_ = INFINITY;

    int nprim_p = Pshell.nprimitive();
    int nprim_q = Qshell.nprimitive();
    for (int pp = 0; pp < nprim_p; pp++) {
        double pcoef = Pshell.coef(pp);
        double pexp = Pshell.exp(pp);
        for (int qp = 0; qp < nprim_q; qp++) {
            double qcoef = Qshell.coef(qp);
            double qexp = Qshell.exp(qp);

            const double pq_exp = std::abs(pexp + qexp);
            Vector3 pq_center = (pexp * pcenter + qexp * qcenter) / pq_exp;

            center_ += pq_center;
            exp_ = std::min(exp_, pq_exp);
        }
    }
    center_ /= (nprim_p * nprim_q);
    extent_ = std::sqrt(-2.0 * std::log(cfmm_extent_tol) / exp_);

    mpole_coefs_ = mpole_coefs;
}

void CFMMShellPair::calculate_mpoles(Vector3 box_center, std::shared_ptr<OneBodyAOInt>& s_ints, 
                                 std::shared_ptr<OneBodyAOInt>& mpole_ints, int lmax) {
    
    // number of total multipoles to compute, -1 since the overlap is not computed
    int nmpoles = (lmax + 1) * (lmax + 2) * (lmax + 3) / 6 - 1;

    int P = pair_index_.first;
    int Q = pair_index_.second;

    // Calculate the overlap integrals (Order 0 multipole integrals)
    s_ints->compute_shell(P, Q);
    const double* sbuffer = s_ints->buffers()[0];

    // Calculate the multipole integrals
    mpole_ints->compute_shell(P, Q);
    const double* mbuffer = mpole_ints->buffers()[0];

    const GaussianShell& Pshell = basisset_->shell(P);
    const GaussianShell& Qshell = basisset_->shell(Q);

    int p_start = Pshell.start();
    int num_p = Pshell.nfunction();

    int q_start = Qshell.start();
    int num_q = Qshell.nfunction();

    for (int p = p_start; p < p_start + num_p; p++) {
        int dp = p - p_start;
        for (int q = q_start; q < q_start + num_q; q++) {
            int dq = q - q_start;

            std::shared_ptr<RealSolidHarmonics> pq_mpoles = std::make_shared<RealSolidHarmonics>(lmax, box_center, Regular);

            pq_mpoles->add(0, 0, sbuffer[dp * num_q + dq]);

            int running_index = 0;
            for (int l = 1; l <= lmax; l++) {
                int l_ncart = ncart(l);
                for (int m = -l; m <= l; m++) {
                    int mu = m_addr(m);
                    std::unordered_map<int, double>& mpole_terms = mpole_coefs_->get_terms(l, mu);

                    int powdex = 0;
                    for (int ii = 0; ii <= l; ii++) {
                        int a = l - ii;
                        for (int jj = 0; jj <= ii; jj++) {
                            int b = ii - jj;
                            int c = jj;
                            int ind = a * l_ncart * l_ncart + b * l_ncart + c;

                            if (mpole_terms.count(ind)) {
                                double coef = mpole_terms[ind];
                                int abcindex = powdex + running_index;
                                pq_mpoles->add(l, mu, pow(-1.0, (double) l+1) * coef * mbuffer[abcindex * num_p * num_q + dp * num_q + dq]);
                            }
                            powdex += 1;
                        } // end jj
                    } // end ii
                } // end m loop
                running_index += l_ncart;
            } // end l
            mpoles_.push_back(pq_mpoles);
        } // end q
    } // end p
}

} // end namespace psi
