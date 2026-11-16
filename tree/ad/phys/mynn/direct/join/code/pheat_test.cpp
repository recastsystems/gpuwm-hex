// pheat_reverse() and pmoist_reverse() (join.hpp, -DJOIN_PMOM -DJOIN_PHEAT) against the forward statements they
// transpose, on the CPU. The forward here is written from the native statements (gen_whole.py --plume-live momheat keeps
// exactly these live, coefficients held); the check is the dot identity <L dx, s> = <dx, L^T s> for random coefficients,
// flags, dx and s, plus a planted error in each.
//   g++ -std=c++17 -O1 -DJOIN_PMOM -DJOIN_PHEAT -I<blocks> pheat_test.cpp -o pheat_test && ./pheat_test
#include <cstdio>
#include <cmath>
#include <random>
#include "join.hpp"
using namespace direct; using namespace direct::join;
typedef float P;
// heat: dt = d thl
static void fwd_heat(const PlumeMom<P>& M, const PlumeHeat<P>& H, int n, const double* dt, double* fl, double* sub, double* det) {
  const double psig = double(M.psig), adj = M.adj_set ? double(M.adj) : 1.0, cdet = double(P(1.f) / P(45.f));
  double envm[MAX + 2]{}, saw[MAX + 2]{};
  for (int k = 1; k <= n; ++k) envm[k] = dt[k];
  for (int ip = 0; ip < NUPM; ++ip) {
    double up[MAX + 2]{};
    if ((M.started >> ip) & 1) up[2] = (dt[1] * double(M.dz2) + dt[2] * double(M.dz1)) / double(r6_add(M.dz1, M.dz2));
    for (int k = 2; k <= n - 1; ++k) {
      const double tn = up[k] * double(H.om[ip * MAX + k]) + dt[k] * double(H.en[ip * MAX + k]);
      envm[k] += (0.5 * (tn + up[k]) - dt[k]) * H.dc[ip * MAX + k];
      if ((M.pass[k] >> ip) & 1) up[k + 1] = tn;
    }
    for (int k = 1; k <= n; ++k) saw[k] += (double(M.c1[ip * MAX + k]) * up[k]) * psig;
  }
  for (int k = 1; k <= n; ++k) fl[k] = saw[k] * adj;
  for (int k = 1; k < n; ++k) {
    sub[k] = (M.dzi[k] != P(0) && M.rhoi[k] != P(0)) ? ((double(M.cs[k]) * (double(M.rho[k + 1]) * dt[k + 1] - double(M.rho[k]) * dt[k])) / double(M.dzi[k])) / double(M.rhoi[k]) : 0.0;
    det[k] = ((cdet * (envm[k] - dt[k])) * double(M.ea[k])) * psig;
  }
}
// moisture: dw = d qt (total water), dv = d qv (vapour); fq is d s_awqt, which is also d s_awqv
static void fwd_moist(const PlumeMom<P>& M, const PlumeHeat<P>& H, int n, const double* dw, const double* dv, double* fq, double* sub, double* det) {
  const double psig = double(M.psig), adj = M.adj_set ? double(M.adj) : 1.0, cdet = double(P(1.f) / P(45.f));
  double envm[MAX + 2]{}, saw[MAX + 2]{};
  for (int k = 1; k <= n; ++k) envm[k] = dv[k];
  for (int ip = 0; ip < NUPM; ++ip) {
    double up[MAX + 2]{};
    if ((M.started >> ip) & 1) up[2] = (dw[1] * double(M.dz2) + dw[2] * double(M.dz1)) / double(r6_add(M.dz1, M.dz2));
    for (int k = 2; k <= n - 1; ++k) {
      const double qn = up[k] * double(H.om[ip * MAX + k]) + dw[k] * double(H.en[ip * MAX + k]);
      const double qv_ent = 0.5 * ((((H.g1[k] >> ip) & 1) ? qn : 0.0) + (((H.g2[k] >> ip) & 1) ? up[k] : 0.0));
      envm[k] += (qv_ent - dv[k]) * H.dc[ip * MAX + k];
      if ((M.pass[k] >> ip) & 1) up[k + 1] = qn;
    }
    for (int k = 1; k <= n; ++k) saw[k] += (double(M.c1[ip * MAX + k]) * up[k]) * psig;
  }
  for (int k = 1; k <= n; ++k) fq[k] = saw[k] * adj;
  for (int k = 1; k < n; ++k) {
    sub[k] = (M.dzi[k] != P(0) && M.rhoi[k] != P(0)) ? ((double(M.cs[k]) * (double(M.rho[k + 1]) * dv[k + 1] - double(M.rho[k]) * dv[k])) / double(M.dzi[k])) / double(M.rhoi[k]) : 0.0;
    det[k] = ((cdet * (envm[k] - dv[k])) * double(M.ea[k])) * psig;
  }
}
int main() {
  std::mt19937 g(11); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Nn(0.0, 1.0);
  double worst[2] = {0, 0}; int fails[2] = {0, 0}, plants[2] = {0, 0}, seen[2] = {0, 0};
  for (int t = 0; t < 2000; ++t) {
    const int n = 4 + int(U(g) * 56); PlumeMom<P> M{}; PlumeHeat<P> H{}; M.on = 1; H.on = 1;
    M.dz1 = P(20 + 30 * U(g)); M.dz2 = P(25 + 40 * U(g)); M.psig = P(U(g)); M.adj = P(0.01 + U(g)); M.adj_set = U(g) < 0.7;
    for (int ip = 0; ip < NUPM; ++ip) {
      if (U(g) < 0.8) M.started |= 1 << ip;
      const int top = 2 + int(U(g) * (n - 2));
      for (int k = 2; k <= n - 2 && k <= top; ++k) {
        const P e = P(0.9 * U(g)); H.en[ip * MAX + k] = e; H.om[ip * MAX + k] = P(1) - e; H.dc[ip * MAX + k] = 1e-2 * U(g);
        M.en[ip * MAX + k] = P(7); M.om[ip * MAX + k] = P(-3); M.dc[ip * MAX + k] = 5.0;      // the momentum's own coefficients must not be read
        if (k < top) M.pass[k] |= 1 << ip;
        if (U(g) < 0.8) H.g1[k] |= 1 << ip;
        if (U(g) < 0.8) H.g2[k] |= 1 << ip;
      }
      for (int k = 1; k <= n - 1; ++k) M.c1[ip * MAX + k] = (k <= top + 1) ? P(0.05 * U(g)) : P(0);
    }
    for (int k = 1; k < n; ++k) { M.cs[k] = P(0.02 * U(g)); M.dzi[k] = P(30 + 50 * U(g)); M.rhoi[k] = P(0.6 + 0.6 * U(g)); M.ea[k] = P(U(g)); }
    for (int k = 1; k <= n; ++k) M.rho[k] = P(0.6 + 0.6 * U(g));
    double dt[MAX + 2]{}, dw[MAX + 2]{}, dv[MAX + 2]{}, sf[MAX + 2]{}, ss[MAX + 2]{}, sd[MAX + 2]{};
    for (int k = 1; k <= n; ++k) { dt[k] = Nn(g); dw[k] = Nn(g); dv[k] = Nn(g); sf[k] = Nn(g); }
    for (int k = 1; k < n; ++k) { ss[k] = Nn(g); sd[k] = Nn(g); }
    for (int c = 0; c < 2; ++c) {
      double f[MAX + 2]{}, sub[MAX + 2]{}, det[MAX + 2]{}, b1[MAX + 2]{}, b2[MAX + 2]{}, p1[MAX + 2]{}, p2[MAX + 2]{};
      if (c == 0) { fwd_heat(M, H, n, dt, f, sub, det); pheat_reverse(M, H, n, sf, ss, sd, b1, false); pheat_reverse(M, H, n, sf, ss, sd, p1, true); }
      else { fwd_moist(M, H, n, dw, dv, f, sub, det); pmoist_reverse(M, H, n, sf, ss, sd, b1, b2, false); pmoist_reverse(M, H, n, sf, ss, sd, p1, p2, true); }
      double lhs = 0, rhs = 0, rhsp = 0, mass = 0;
      for (int k = 1; k <= n; ++k) { lhs += f[k] * sf[k]; mass += std::fabs(f[k] * sf[k]);
        if (c == 0) { rhs += dt[k] * b1[k]; rhsp += dt[k] * p1[k]; } else { rhs += dw[k] * b1[k] + dv[k] * b2[k]; rhsp += dw[k] * p1[k] + dv[k] * p2[k]; } }
      for (int k = 1; k < n; ++k) { lhs += sub[k] * ss[k] + det[k] * sd[k]; mass += std::fabs(sub[k] * ss[k]) + std::fabs(det[k] * sd[k]); }
      const double r = std::fabs(lhs - rhs) / (mass > 0 ? mass : 1), rp = std::fabs(lhs - rhsp) / (mass > 0 ? mass : 1);
      if (r > worst[c]) worst[c] = r; if (r > 1e-13) ++fails[c];
      if (rp > 1e-9) { ++plants[c]; if (rp > 100 * (r > 1e-17 ? r : 1e-17)) ++seen[c]; }
    }
  }
  const bool ok = fails[0] == 0 && fails[1] == 0 && plants[0] > 100 && plants[1] > 100 && seen[0] == plants[0] && seen[1] == plants[1];
  printf("%s plume heat / moisture transpose: 2000 random blocks each, worst |<L dx,s> - <dx,L^T s>| / sum of |terms| = %.3e / %.3e (bar 1e-13), %d / %d over the bar; planted x1.001 coefficient seen in %d of %d / %d of %d blocks where it is over 1e-9\n",
         ok ? "PASS" : "FAIL", worst[0], worst[1], fails[0], fails[1], seen[0], plants[0], seen[1], plants[1]);
  return !ok;
}
