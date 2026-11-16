// CPU test of the tape-free mixing transpose (join.hpp JOIN_MIX, blocks/mixjac.hpp).
// A. The stability Jacobian of mixjac::stab (float values, double partials, the generated tangent's rules) against centred
//    differences of the same statements evaluated in double (mixjacd.hpp: mixjac.hpp with every float made double), on
//    random columns that reach both arms of the level-2.5 block, every Prandtl-limit regime and the cloud floor; points
//    where the difference straddles a branch (two step sizes disagree) are skipped and counted.
// B. mix_reverse against an independent tangent of the diffusivities written here from the native statements
//    (dfm = el*qtw*ksm/dzk, dfh = el*qkw*ksh/dzk, J applied to the ten words): <bars, M dx> = <M^T bars, dx> on random
//    frames, seeds and directions, the weak-shear rule on (J zero where gm < 1e-4); then with the test plant
//    (F.plant = 60: x1.001 on one level's stability bar) the same check must fail.
// C. mix_frame_ok: a frame whose solve diffusivities are the re-formed ones passes; one word off by one ulp is refused.
// build: g++ -std=c++17 -O1 -DJOIN_MIX -Iblocks -I<dir of mixjacd.hpp> code/mix_test.cpp -o t && ./t
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include <algorithm>
#include "join.hpp"
#include "mixjac.hpp"
#include "mixjacd.hpp"
using namespace direct; using namespace direct::join;

static const mixjac::Con CON = {24.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static mixjac::Con con() {   // the forward's constants (module_bl_mynnedmf.F90), evaluated as the forward's constexpr floats
  const float b1 = 24.0f, b2 = 15.0f, c2 = 0.729f, c3 = 0.340f, c5 = 0.2f, pr = 0.74f, g1 = 0.235f;
  const float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
  const float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
  const float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
  const float cc2 = ((1.0f) - (c2)), cc3 = ((1.0f) - (c3));
  const float e1c = ((((((3.0f) * (a2))) * (b2))) * (cc3)), e2c = ((((((9.0f) * (a1))) * (a2))) * (cc2));
  const float e3c = ((((((((9.0f) * (a2))) * (a2))) * (cc2))) * (((1.0f) - (c5)))), e4c = ((((((12.0f) * (a1))) * (a2))) * (cc2));
  const float e5c = ((((6.0f) * (a1))) * (a1));
  (void)CON; return mixjac::Con{b1, a1, a2, c1, e1c, e2c, e3c, e4c, e5c};
}
static mixjacd::Con cond(const mixjac::Con& c) { return mixjacd::Con{c.b1, c.a1, c.a2, c.c1, c.e1c, c.e2c, c.e3c, c.e4c, c.e5c}; }

struct Rng { std::mt19937_64 g; explicit Rng(unsigned s) : g(s) {} double u(double a, double b) { return std::uniform_real_distribution<double>(a, b)(g); }
             double lg(double a, double b) { return std::pow(10.0, u(a, b)); } int i(int a, int b) { return std::uniform_int_distribution<int>(a, b)(g); } };

static mixjac::In draw(Rng& r) {
  mixjac::In x{};
  x.qkw = float(r.lg(-1.6, 0.4)); x.gm = float(r.lg(-6.5, -1.5)); x.gh = float((r.u(0, 1) < 0.6 ? -1 : 1) * r.lg(-6.5, -2.0));
  x.uk = float(r.u(-15, 15)); x.ukm1 = float(x.uk + r.u(-3, 3)); x.vk = float(r.u(-15, 15)); x.vkm1 = float(x.vk + r.u(-3, 3));
  x.dzk = float(r.u(20, 300)); x.dzkm1 = float(x.dzk * r.u(0.8, 1.0)); x.zwk = float(r.u(20, 3000));
  x.cfk = r.u(0, 1) < 0.5 ? 0.0f : float(r.u(0, 1)); x.cfkm1 = r.u(0, 1) < 0.5 ? 0.0f : float(r.u(0, 1));
  x.el = float(r.u(5, 150)); x.sm2 = float(r.u(0.02, 1.2)); x.sh2 = float(r.u(0.02, 1.4)); x.pblh = float(r.u(100, 2500));
  x.eak = x.eakm1 = x.ewk = x.ewkm1 = x.eaddk = x.ewddk = 0.0f; if (r.u(0, 1) < 0.1) { x.eak = 0.01f; x.ewk = 0.5f; }
  x.tkdn = 0.0f; x.closure = 2.5999999046325684f; x.tke_budget = 0;
  return x;
}
static mixjacd::In todd(const mixjac::In& x) {
  return mixjacd::In{x.qkw, x.gm, x.gh, x.uk, x.ukm1, x.vk, x.vkm1, x.dzk, x.dzkm1, x.zwk, x.cfk, x.cfkm1, x.el, x.sm2, x.sh2,
                     x.pblh, x.eak, x.eakm1, x.ewk, x.ewkm1, x.eaddk, x.ewddk, x.tkdn, x.closure, x.tke_budget};
}
// the word w of a double input moved by h (du, dv move u(k), v(k))
static void bump(mixjacd::In& x, int w, double h) {
  switch (w) { case 0: x.qkw += h; break; case 1: x.gm += h; break; case 2: x.gh += h; break; case 3: x.uk += h; break; case 4: x.vk += h; break;
               case 5: x.dzk += h; break; case 6: x.dzkm1 += h; break; case 7: x.zwk += h; break; case 8: x.cfk += h; break; default: x.cfkm1 += h; }
}
static double word(const mixjacd::In& x, int w) {
  const double v[10] = {x.qkw, x.gm, x.gh, x.uk, x.vk, x.dzk, x.dzkm1, x.zwk, x.cfk, x.cfkm1}; return v[w];
}

int main() {
  const mixjac::Con c = con(); const mixjacd::Con cdd = cond(c);
  Rng r(20261114);
  // ---- A
  int na = 0, skipped = 0, arms[2] = {0, 0}; double worst = 0; int wi = -1; std::vector<double> errs; mixjac::In wx{};
  for (int t = 0; t < 20000; ++t) {
    const mixjac::In x = draw(r); mixjac::F sm, sh;
    if (!mixjac::stab(x, c, sm, sh)) { std::printf("A: stab refused a closure-2.6 point\n"); return 1; }
    const double q2 = 24.0 * double(x.el) * double(x.el) * (double(x.sm2) * x.gm + double(x.sh2) * x.gh); arms[double(x.qkw) * x.qkw < q2 ? 0 : 1]++;
    // each partial times its word's size, judged against the row's largest such term: the float values the partials
    // are evaluated at move a small partial that is a cancellation of large terms by about 1e-7 of those terms, which a
    // wrong derivative (an O(1) error) is not
    double fdv[2][10], jv[2][10], sz[10]; bool use[10];
    for (int w = 0; w < 10; ++w) {
      use[w] = !((w == 8 || w == 9) && (w == 8 ? x.cfk : x.cfkm1) == 0.0f);   // one-sided at a zero cloud fraction
      const double s = std::fabs(word(todd(x), w)) + 1e-3; double fd[2][2]; sz[w] = s;
      for (int hh = 0; hh < 2; ++hh) {
        const double h = s * (hh ? 2e-6 : 1e-6); mixjacd::In p = todd(x), m = todd(x); bump(p, w, h); bump(m, w, -h);
        mixjacd::F a, b, a2, b2; mixjacd::stab(p, cdd, a, b); mixjacd::stab(m, cdd, a2, b2);
        fd[hh][0] = (a.v - a2.v) / (2 * h); fd[hh][1] = (b.v - b2.v) / (2 * h);
      }
      for (int o = 0; o < 2; ++o) {
        fdv[o][w] = fd[0][o]; jv[o][w] = o ? sh.d[w] : sm.d[w];
        if (std::fabs(fd[0][o] - fd[1][o]) > 1e-6 * (std::fabs(fd[0][o]) + 1e-9)) use[w] = false;   // a branch inside the step
      }
      if (!use[w]) ++skipped;
    }
    for (int o = 0; o < 2; ++o) {
      double row = 1e-300; for (int w = 0; w < 10; ++w) if (use[w]) row = std::max(row, std::max(std::fabs(fdv[o][w]), std::fabs(jv[o][w])) * sz[w]);
      for (int w = 0; w < 10; ++w) if (use[w]) {
        const double e = std::fabs(jv[o][w] - fdv[o][w]) * sz[w] / row; ++na; errs.push_back(e); if (e > worst) { worst = e; wi = w; wx = x; }
      }
    }
  }
  std::printf("A: %d partials against centred differences, worst relative %.2e (word %d); %d skipped (a branch inside the step); arms qdiv %d, level 2.5 %d\n",
              na, worst, wi, skipped, arms[0], arms[1]);
  std::sort(errs.begin(), errs.end());
  auto q = [&](double f) { return errs[size_t(f * (errs.size() - 1))]; };
  int over[3] = {0, 0, 0}; for (double e : errs) { over[0] += e > 1e-6; over[1] += e > 1e-4; over[2] += e > 1e-2; }
  std::printf("A: median %.1e, 99th %.1e, 99.9th %.1e; over 1e-6: %d, over 1e-4: %d, over 1e-2: %d\n", q(0.5), q(0.99), q(0.999), over[0], over[1], over[2]);
  std::printf("A: worst point qkw %g gm %g gh %g du %g dv %g dz %g %g el %g sm2 %g sh2 %g\n", wx.qkw, wx.gm, wx.gh, wx.uk - wx.ukm1, wx.vk - wx.vkm1, wx.dzk, wx.dzkm1, wx.el, wx.sm2, wx.sh2);
  const bool okA = over[1] == 0 && q(0.999) < 1e-5;   // float-value rounding only: no partial off by 1e-4 of its row
  // ---- B and C
  auto F = std::make_unique<ColumnFrame<float>>();
  double worstB = 0, plantB = 1e300; int holds = 0, levels = 0;
  for (int t = 0; t < 300; ++t) {
    std::memset(F.get(), 0, sizeof(ColumnFrame<float>)); const int n = r.i(10, 59); F->n = n; F->mx.on = 1; F->mx.t = 1e-4;
    float u[MAX]{}, v[MAX]{}, dz[MAX]{}, zw[MAX + 1]{}, cf[MAX]{};
    for (int k = 1; k <= n; ++k) { u[k] = float(r.u(-15, 15)); v[k] = float(r.u(-15, 15)); dz[k] = float(r.u(20, 300)); cf[k] = r.u(0, 1) < 0.5 ? 0.0f : float(r.u(0, 1)); }
    for (int k = 1; k <= n; ++k) zw[k + 1] = zw[k] + dz[k];
    float* P = F->prod; P[5 * MAX + 1] = dz[1];
    for (int k = 2; k <= n; ++k) {
      mixjac::In x = draw(r); x.uk = u[k]; x.ukm1 = u[k - 1]; x.vk = v[k]; x.vkm1 = v[k - 1]; x.dzk = dz[k]; x.dzkm1 = dz[k - 1]; x.zwk = zw[k];
      x.cfk = cf[k]; x.cfkm1 = cf[k - 1];
      mixjac::F sm, sh; mixjac::stab(x, c, sm, sh); const bool hold = double(x.gm) < 1e-4; holds += hold; ++levels;
      P[k] = x.qkw; P[MAX + k] = x.gm; P[2 * MAX + k] = x.gh; P[5 * MAX + k] = dz[k]; P[6 * MAX + k] = x.el; P[7 * MAX + k] = sm.v; P[8 * MAX + k] = sh.v;
      F->mx.qtw[k] = x.qkw;
      for (int i = 0; i < 10; ++i) { F->mx.jac[20 * k + i] = hold ? 0.0 : sm.d[i]; F->mx.jac[20 * k + 10 + i] = hold ? 0.0 : sh.d[i]; }
      float dfm, dfh, a, b, d; mix_dfs(*F, k, dfm, dfh, a, b, d);
      F->solve[0][solves::DF * MAX + k] = dfm; F->solve[1][solves::DF * MAX + k] = dfm; for (int j = 2; j < 6; ++j) F->solve[j][solves::DF * MAX + k] = dfh;
      F->heat[DFH * MAX + k] = dfh; F->pred[3 * MAX + k] = dfm;
    }
    // C
    if (!mix_frame_ok(*F)) { std::printf("C: a consistent frame was refused\n"); return 1; }
    if (t < 20) { const int k = r.i(2, n); float& w = F->solve[2 + t % 4][solves::DF * MAX + k]; /* each of the four solves that read dfh (NC, NI, NWFA, NIFA) in turn */ const float keep = w; w = std::nextafter(w, 1e30f);
                  if (mix_frame_ok(*F)) { std::printf("C: a one-ulp diffusivity was not refused\n"); return 1; } w = keep; }
    // B
    double gbm[MAX]{}, gbh[MAX]{}; for (int k = 2; k <= n; ++k) { gbm[k] = r.u(-1, 1); gbh[k] = r.u(-1, 1); }
    double dqkw[MAX]{}, dgm[MAX]{}, dgh[MAX]{}, du[MAX]{}, dv[MAX]{}, ddz[MAX]{}, dzw[MAX + 1]{}, dcf[MAX]{};
    for (int k = 1; k <= n; ++k) { dqkw[k] = r.u(-1, 1); dgm[k] = r.u(-1, 1) * 1e-3; dgh[k] = r.u(-1, 1) * 1e-3; du[k] = r.u(-1, 1); dv[k] = r.u(-1, 1);
                                   ddz[k] = r.u(-1, 1); dzw[k] = r.u(-1, 1); dcf[k] = r.u(-1, 1) * 0.1; }
    double lhs = 0;
    for (int k = 2; k <= n; ++k) {
      float dfm, dfh, elqt, elq, dzk; mix_dfs(*F, k, dfm, dfh, elqt, elq, dzk);
      const double el = P[6 * MAX + k], ksm = P[7 * MAX + k], ksh = P[8 * MAX + k], ddzk = 0.5 * (ddz[k] + ddz[k - 1]);
      const double w[10] = {dqkw[k], dgm[k], dgh[k], du[k] - du[k - 1], dv[k] - dv[k - 1], ddz[k], ddz[k - 1], dzw[k], dcf[k], dcf[k - 1]};
      double dsm = 0, dsh = 0; for (int i = 0; i < 10; ++i) { dsm += F->mx.jac[20 * k + i] * w[i]; dsh += F->mx.jac[20 * k + 10 + i] * w[i]; }
      // the generated tangent rule of a float quotient: the quotient in double from the float operands (mw_ad_rules.inc r6_div)
      const double qm = double(float(elqt * P[7 * MAX + k])) / double(dzk), qh = double(float(elq * P[8 * MAX + k])) / double(dzk);
      const double tm = (el * dqkw[k] * ksm + double(elqt) * dsm) / double(dzk) - qm * ddzk / double(dzk);
      const double th = (el * dqkw[k] * ksh + double(elq) * dsh) / double(dzk) - qh * ddzk / double(dzk);
      lhs += gbm[k] * tm + gbh[k] * th;
    }
    for (int pl = 0; pl < 2; ++pl) {
      F->plant = pl ? 60 : 0; auto Bp = std::make_unique<ColumnBars>(); ColumnBars& B = *Bp; std::memset(&B, 0, sizeof B);
      mix_reverse(*F, gbm, gbh, B);
      double rhs = 0;
      for (int k = 1; k <= n; ++k) rhs += B.qkw[k] * dqkw[k] + B.gm[k] * dgm[k] + B.gh[k] * dgh[k] + B.u[k] * du[k] + B.v[k] * dv[k] + B.dz[k] * ddz[k] + B.zw[k] * dzw[k] + B.cf_bl[k] * dcf[k];
      const double e = std::fabs(lhs - rhs) / (std::fabs(lhs) + std::fabs(rhs) + 1e-300);
      if (pl) plantB = std::min(plantB, e); else worstB = std::max(worstB, e);
    }
  }
  std::printf("B: 300 random frames (%d levels, %d held by the rule): dot test worst %.2e; with the plant at one level the smallest gap %.2e\n",
              levels, holds, worstB, plantB);
  std::printf("C: consistent frames pass; 20 one-ulp diffusivities refused\n");
  const bool okB = worstB < 1e-13 && plantB > 1000.0 * worstB;   // the plant is seen in every frame, a thousand times over the clean noise
  std::printf("%s\n", okA && okB ? "PASS" : "FAIL");
  return okA && okB ? 0 : 1;
}
