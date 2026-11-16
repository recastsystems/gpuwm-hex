// pmom_reverse() (join.hpp, -DJOIN_PMOM) against the forward statements it transposes, on the CPU.
// The forward here is written from the native statements (gen_whole.py --plume-live mom keeps exactly these live, coefficients
// held); the check is the dot identity <L du, s> = <du, L^T s> for random coefficients, flags, du and s, plus a planted error.
//   g++ -std=c++17 -O1 -DJOIN_PMOM -I<blocks> pmom_test.cpp -o pmom_test && ./pmom_test
#include <cstdio>
#include <cmath>
#include <random>
#include "join.hpp"
using namespace direct; using namespace direct::join;
typedef float P;
static void forward(const PlumeMom<P>& M, int n, const double* du, double* fu, double* sub, double* det) {
  const double psig = double(M.psig), adj = M.adj_set ? double(M.adj) : 1.0, cdet = double(P(1.f) / P(45.f));
  double envm[MAX + 2]{}, saw[MAX + 2]{};
  for (int k = 1; k <= n; ++k) envm[k] = du[k];
  for (int ip = 0; ip < NUPM; ++ip) {
    double upu[MAX + 2]{};
    if ((M.started >> ip) & 1) upu[2] = (du[1] * double(M.dz2) + du[2] * double(M.dz1)) / double(r6_add(M.dz1, M.dz2));
    for (int k = 2; k <= n - 1; ++k) {
      const double un = upu[k] * double(M.om[ip * MAX + k]) + du[k] * double(M.en[ip * MAX + k]);
      envm[k] += (0.5 * (un + upu[k]) - du[k]) * M.dc[ip * MAX + k];
      if ((M.pass[k] >> ip) & 1) upu[k + 1] = un;
    }
    for (int k = 1; k <= n; ++k) saw[k] += (double(M.c1[ip * MAX + k]) * upu[k]) * psig;
  }
  for (int k = 1; k <= n; ++k) fu[k] = saw[k] * adj;
  for (int k = 1; k < n; ++k) {
    sub[k] = (M.dzi[k] != P(0) && M.rhoi[k] != P(0)) ? ((double(M.cs[k]) * (double(M.rho[k + 1]) * du[k + 1] - double(M.rho[k]) * du[k])) / double(M.dzi[k])) / double(M.rhoi[k]) : 0.0;
    det[k] = ((cdet * (envm[k] - du[k])) * double(M.ea[k])) * psig;
  }
}
int main() {
  std::mt19937 g(7); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Nn(0.0, 1.0);
  double worst = 0, plant_least = 1e300; int fails = 0, plants = 0, plant_seen = 0;
  for (int t = 0; t < 2000; ++t) {
    const int n = 4 + int(U(g) * 56); PlumeMom<P> M{}; M.on = 1;
    M.dz1 = P(20 + 30 * U(g)); M.dz2 = P(25 + 40 * U(g)); M.psig = P(U(g)); M.adj = P(0.01 + U(g)); M.adj_set = U(g) < 0.7;
    for (int ip = 0; ip < NUPM; ++ip) {
      if (U(g) < 0.8) M.started |= 1 << ip;
      const int top = 2 + int(U(g) * (n - 2));        // the plume runs k = 2..top and passes its wind up while k < top
      for (int k = 2; k <= n - 2 && k <= top; ++k) {
        const P e = P(0.4 * U(g)); M.en[ip * MAX + k] = e; M.om[ip * MAX + k] = P(1) - e; M.dc[ip * MAX + k] = 1e-2 * U(g);
        if (k < top) M.pass[k] |= 1 << ip;
      }
      for (int k = 1; k <= n - 1; ++k) M.c1[ip * MAX + k] = (k <= top + 1) ? P(0.05 * U(g)) : P(0);
    }
    for (int k = 1; k < n; ++k) { M.cs[k] = P(0.02 * U(g)); M.dzi[k] = P(30 + 50 * U(g)); M.rhoi[k] = P(0.6 + 0.6 * U(g)); M.ea[k] = P(U(g)); }
    for (int k = 1; k <= n; ++k) M.rho[k] = P(0.6 + 0.6 * U(g));
    double du[MAX + 2]{}, fu[MAX + 2]{}, sub[MAX + 2]{}, det[MAX + 2]{}, sf[MAX + 2]{}, ss[MAX + 2]{}, sd[MAX + 2]{}, ub[MAX + 2]{}, ubp[MAX + 2]{};
    for (int k = 1; k <= n; ++k) { du[k] = Nn(g); sf[k] = Nn(g); }
    for (int k = 1; k < n; ++k) { ss[k] = Nn(g); sd[k] = Nn(g); }
    forward(M, n, du, fu, sub, det);
    pmom_reverse(M, n, sf, ss, sd, ub, false); pmom_reverse(M, n, sf, ss, sd, ubp, true);
    double lhs = 0, rhs = 0, rhsp = 0, mass = 0;
    for (int k = 1; k <= n; ++k) { lhs += fu[k] * sf[k]; mass += std::fabs(fu[k] * sf[k]); rhs += du[k] * ub[k]; rhsp += du[k] * ubp[k]; }
    for (int k = 1; k < n; ++k) { lhs += sub[k] * ss[k] + det[k] * sd[k]; mass += std::fabs(sub[k] * ss[k]) + std::fabs(det[k] * sd[k]); }
    const double r = std::fabs(lhs - rhs) / (mass > 0 ? mass : 1), rp = std::fabs(lhs - rhsp) / (mass > 0 ? mass : 1);
    if (r > worst) worst = r; if (r > 1e-13) ++fails;
    { const double want = 0.001 * std::fabs(sub[2] * ss[2]) / (mass > 0 ? mass : 1);     // the planted coefficient moves exactly this much
      if (want > 1e-10) { ++plants; if (std::fabs(rp - want) <= 0.01 * want + 1e-13) ++plant_seen; if (rp / (r > 1e-17 ? r : 1e-17) < plant_least) plant_least = rp / (r > 1e-17 ? r : 1e-17); } }
  }
  printf("%s pmom-transpose: 2000 random plume-momentum blocks, worst |<L du,s> - <du,L^T s>| / sum of |terms| = %.3e (bar 1e-13), %d over the bar; planted x1.001 coefficient: seen at its exact size in %d of %d blocks where it is over 1e-10, there at least %.0f times the clean reading\n",
         fails == 0 && plants > 100 && plant_seen == plants ? "PASS" : "FAIL", worst, fails, plant_seen, plants, plant_least);
  return !(fails == 0 && plant_seen == plants);
}
