// CPU test of the closure-live step of the tape-free boundary-layer backward
// (join.hpp JOIN_MIX + JOIN_CLOSURE, blocks/mixjac.hpp level2 / stab3, blocks/turbprod.hpp bars of sm, sh, qdiv).
// A. The 3 x 10 Jacobian of (sm, sh, qdiv) with the level-2 functions folded in (mixjac::level2 then stab3: float values,
//    double partials, the generated rules with the three closed forms) against centred differences of the same statements
//    in double (mixjacd.hpp), on random levels of both arms; a point where two step sizes disagree (a branch inside the
//    step) is skipped and counted. Also level2 alone (sm2, sh2 against gm, gh).
// B. The turbulence-energy stage: turbprod_reverse with the new bars, then closure_prod_reverse, against a tangent of
//    pdk, pdt, pdq, pdc written here from the native statements (el held, sm, sh, qdiv through the Jacobian, the weak-shear
//    rule on): <bars, M dx> = <M^T bars, dx> on random frames; with plant 61 (x1.001 on one level's bars of sm, sh, qdiv) and with the
//    Jacobian's level-2 part dropped the same check must fail.
// build: g++ -std=c++17 -O1 -DJOIN_MIX -DJOIN_CLOSURE -Iblocks -Icode code/clos_test.cpp -o t && ./t
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

struct Cons { mixjac::Con c; mixjac::L2Con l; };
static Cons con() {   // the forward's constants (module_bl_mynnedmf.F90), evaluated as the forward's constexpr floats
  const float b1 = 24.0f, b2 = 15.0f, c2 = 0.729f, c3 = 0.340f, c5 = 0.2f, pr = 0.74f, g1 = 0.235f;
  const float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
  const float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
  const float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
  const float g2 = ((((((b2) / (b1))) * (((1.0f) - (c3))))) + (((((((2.0f) * (a1))) / (b1))) * (((3.0f) - (((2.0f) * (c2))))))));
  const float cc2 = ((1.0f) - (c2)), cc3 = ((1.0f) - (c3));
  const float e1c = ((((((3.0f) * (a2))) * (b2))) * (cc3)), e2c = ((((((9.0f) * (a1))) * (a2))) * (cc2));
  const float e3c = ((((((((9.0f) * (a2))) * (a2))) * (cc2))) * (((1.0f) - (c5)))), e4c = ((((((12.0f) * (a1))) * (a2))) * (cc2));
  const float e5c = ((((6.0f) * (a1))) * (a1));
  return Cons{mixjac::Con{b1, a1, a2, c1, e1c, e2c, e3c, e4c, e5c}, mixjac::L2Con{g1, g2, b1, a1, a2, c1, c2, c5, 1}};
}
struct Rng { std::mt19937_64 g; explicit Rng(unsigned s) : g(s) {} double u(double a, double b) { return std::uniform_real_distribution<double>(a, b)(g); }
             double lg(double a, double b) { return std::pow(10.0, u(a, b)); } int i(int a, int b) { return std::uniform_int_distribution<int>(a, b)(g); } };
static mixjac::In draw(Rng& r) {
  mixjac::In x{};
  x.qkw = float(r.lg(-1.6, 0.4)); x.gm = float(r.lg(-5.0, -1.5)); x.gh = float((r.u(0, 1) < 0.6 ? -1 : 1) * r.lg(-6.5, -2.0));
  x.uk = float(r.u(-15, 15)); x.ukm1 = float(x.uk + r.u(-3, 3)); x.vk = float(r.u(-15, 15)); x.vkm1 = float(x.vk + r.u(-3, 3));
  x.dzk = float(r.u(20, 300)); x.dzkm1 = float(x.dzk * r.u(0.8, 1.0)); x.zwk = float(r.u(20, 3000));
  x.cfk = r.u(0, 1) < 0.5 ? 0.0f : float(r.u(0, 1)); x.cfkm1 = r.u(0, 1) < 0.5 ? 0.0f : float(r.u(0, 1));
  x.el = float(r.u(5, 150)); x.sm2 = 0.0f; x.sh2 = 0.0f; x.pblh = float(r.u(100, 2500));
  x.eak = x.eakm1 = x.ewk = x.ewkm1 = x.eaddk = x.ewddk = 0.0f; if (r.u(0, 1) < 0.1) { x.eak = 0.01f; x.ewk = 0.5f; }
  x.tkdn = 0.0f; x.closure = 2.5999999046325684f; x.tke_budget = 0;
  return x;
}
static mixjacd::In todd(const mixjac::In& x) {
  return mixjacd::In{x.qkw, x.gm, x.gh, x.uk, x.ukm1, x.vk, x.vkm1, x.dzk, x.dzkm1, x.zwk, x.cfk, x.cfkm1, x.el, x.sm2, x.sh2,
                     x.pblh, x.eak, x.eakm1, x.ewk, x.ewkm1, x.eaddk, x.ewddk, x.tkdn, x.closure, x.tke_budget};
}
static void bump(mixjacd::In& x, int w, double h) {
  switch (w) { case 0: x.qkw += h; break; case 1: x.gm += h; break; case 2: x.gh += h; break; case 3: x.uk += h; break; case 4: x.vk += h; break;
               case 5: x.dzk += h; break; case 6: x.dzkm1 += h; break; case 7: x.zwk += h; break; case 8: x.cfk += h; break; default: x.cfkm1 += h; }
}
static double word(const mixjacd::In& x, int w) {
  const double v[10] = {x.qkw, x.gm, x.gh, x.uk, x.vk, x.dzk, x.dzkm1, x.zwk, x.cfk, x.cfkm1}; return v[w];
}
// the three outputs in double, level-2 recomputed from gm, gh
static void evald(const mixjacd::In& x, const Cons& K, double o[5]) {
  const mixjacd::Con cd{K.c.b1, K.c.a1, K.c.a2, K.c.c1, K.c.e1c, K.c.e2c, K.c.e3c, K.c.e4c, K.c.e5c};
  const mixjacd::L2Con ld{K.l.g1, K.l.g2, K.l.b1, K.l.a1, K.l.a2, K.l.c1, K.l.c2, K.l.c5, 1};
  mixjacd::F m2, h2, sm, sh; mixjacd::D q; mixjacd::level2(x.gm, x.gh, ld, m2, h2); mixjacd::stab3(x, cd, m2, h2, sm, sh, q);
  o[0] = sm.v; o[1] = sh.v; o[2] = q.v; o[3] = m2.v; o[4] = h2.v;
}

int main() {
  const Cons K = con(); Rng r(20261115);
  // ---- A
  int na = 0, skipped = 0, arms[2] = {0, 0}, ov[5] = {0, 0, 0, 0, 0}, ovlive = 0; std::vector<double> errs; double worst = 0; int wi = -1, wo = -1; mixjac::In wx{};
  for (int t = 0; t < 20000; ++t) {
    const mixjac::In x = draw(r); mixjac::F m2, h2, sm, sh; mixjac::D qd;
    mixjac::level2(x.gm, x.gh, K.l, m2, h2);
    if (!mixjac::stab3(x, K.c, m2, h2, sm, sh, qd)) { std::printf("A: stab3 refused a closure-2.6 point\n"); return 1; }
    arms[qd.v != 1.0 ? 0 : 1]++;
    double fdv[5][10], jv[5][10], sz[10]; bool use[10];
    for (int w = 0; w < 10; ++w) {
      use[w] = !((w == 8 || w == 9) && (w == 8 ? x.cfk : x.cfkm1) == 0.0f);
      const double s = std::fabs(word(todd(x), w)) + (w == 1 || w == 2 ? 0.0 : 1e-3); double fd[2][5]; sz[w] = s;
      for (int hh = 0; hh < 2; ++hh) {
        const double h = s * (hh ? 2e-6 : 1e-6); mixjacd::In p = todd(x), m = todd(x); bump(p, w, h); bump(m, w, -h);
        double a[5], b[5]; evald(p, K, a); evald(m, K, b); for (int o = 0; o < 5; ++o) fd[hh][o] = (a[o] - b[o]) / (2 * h);
      }
      for (int o = 0; o < 5; ++o) {
        fdv[o][w] = fd[0][o]; jv[o][w] = o == 0 ? sm.d[w] : o == 1 ? sh.d[w] : o == 2 ? qd.d[w] : o == 3 ? m2.d[w] : h2.d[w];
        if (std::fabs(fd[0][o] - fd[1][o]) > 1e-6 * (std::fabs(fd[0][o]) + 1e-9)) use[w] = false;
      }
      if (!use[w]) ++skipped;
    }
    for (int o = 0; o < 5; ++o) {
      double row = 1e-300; for (int w = 0; w < 10; ++w) if (use[w]) row = std::max(row, std::max(std::fabs(fdv[o][w]), std::fabs(jv[o][w])) * sz[w]);
      for (int w = 0; w < 10; ++w) if (use[w]) {
        const double e = std::fabs(jv[o][w] - fdv[o][w]) * sz[w] / row; ++na; errs.push_back(e); if (e > 1e-4) { ++ov[o]; if (double(x.gm) >= 1e-4) ++ovlive; } if (e > worst) { worst = e; wi = w; wo = o; wx = x; }
      }
    }
  }
  std::sort(errs.begin(), errs.end());
  auto q = [&](double f) { return errs[size_t(f * (errs.size() - 1))]; };
  int over[3] = {0, 0, 0}; for (double e : errs) { over[0] += e > 1e-6; over[1] += e > 1e-4; over[2] += e > 1e-2; }
  std::printf("A: %d partials of sm, sh, qdiv, sm2, sh2 against centred differences, worst relative %.2e (output %d word %d); %d skipped; arms qdiv %d, level 2.5 %d\n",
              na, worst, wo, wi, skipped, arms[0], arms[1]);
  std::printf("A: median %.1e, 99th %.1e, 99.9th %.1e; over 1e-6: %d, over 1e-4: %d, over 1e-2: %d\n", q(0.5), q(0.99), q(0.999), over[0], over[1], over[2]);
  std::printf("A: worst point qkw %g gm %g gh %g el %g ri %g\n", wx.qkw, wx.gm, wx.gh, wx.el, -wx.gh / wx.gm);
  std::printf("A: over 1e-4 by output (sm, sh, qdiv, sm2, sh2): %d %d %d %d %d; of them at levels the rule leaves live (gm >= 1e-4): %d\n", ov[0], ov[1], ov[2], ov[3], ov[4], ovlive);
  // The partials are evaluated at float values (as those of the generated tangent are); near rf -> rfc and at large |ri| a
  // partial that is a cancellation of large terms moves by more than 1e-6 of its row. A wrong formula is an O(1) error.
  const bool okA = over[2] == 0 && q(0.999) < 1e-4;
  // ---- B
  auto F = std::make_unique<ColumnFrame<float>>();
  double worstB = 0, plantB = 1e300, dropB = 1e300; int holds = 0, levels = 0, planted = 0;
  for (int t = 0; t < 400; ++t) {
    std::memset(F.get(), 0, sizeof(ColumnFrame<float>)); const int n = r.i(10, 59); F->n = n; F->mx.on = 1; F->mx.t = 1e-4;
    float* P = F->prod; float dz[MAX]{}; for (int k = 1; k <= n; ++k) { dz[k] = float(r.u(20, 300)); P[5 * MAX + k] = dz[k]; }
    std::vector<double> J0(MAX * 30, 0.0);   // the Jacobian with level-2 held (the mixing form's), for the drop control
    for (int k = 2; k <= n; ++k) {
      mixjac::In x = draw(r); x.dzk = dz[k]; x.dzkm1 = dz[k - 1];
      mixjac::F m2, h2, sm, sh; mixjac::D qd; mixjac::level2(x.gm, x.gh, K.l, m2, h2); mixjac::stab3(x, K.c, m2, h2, sm, sh, qd);
      mixjac::In xh = x; xh.sm2 = m2.v; xh.sh2 = h2.v; mixjac::F sm0, sh0; mixjac::D q0; mixjac::stab3(xh, K.c, mixjac::cf(m2.v), mixjac::cf(h2.v), sm0, sh0, q0);
      const bool hold = double(x.gm) < 1e-4; holds += hold; ++levels;
      P[k] = x.qkw; P[MAX + k] = x.gm; P[2 * MAX + k] = x.gh; P[3 * MAX + k] = float(r.u(-0.02, 0.02)); P[4 * MAX + k] = float(r.u(-2e-5, 2e-5));
      P[6 * MAX + k] = x.el; P[7 * MAX + k] = sm.v; P[8 * MAX + k] = sh.v; F->qdiv[k] = qd.v; F->mx.qtw[k] = x.qkw;
      for (int i = 0; i < 10; ++i) { F->mx.jac[20 * k + i] = hold ? 0.0 : sm.d[i]; F->mx.jac[20 * k + 10 + i] = hold ? 0.0 : sh.d[i]; F->mx.jq[10 * k + i] = hold ? 0.0 : qd.d[i];
                                     J0[30 * k + i] = hold ? 0.0 : sm0.d[i]; J0[30 * k + 10 + i] = hold ? 0.0 : sh0.d[i]; J0[30 * k + 20 + i] = hold ? 0.0 : q0.d[i]; }
    }
    double seed[TPNO]{}; for (int k = 2; k <= n; ++k) for (int j = 0; j < 4; ++j) seed[j * MAX + k] = r.u(-1, 1);
    double dqkw[MAX]{}, dgm[MAX]{}, dgh[MAX]{}, ddtl[MAX]{}, ddqw[MAX]{}, du[MAX]{}, dv[MAX]{}, ddz[MAX]{}, dzw[MAX + 1]{}, dcf[MAX]{};
    for (int k = 1; k <= n; ++k) { dqkw[k] = r.u(-1, 1); dgm[k] = r.u(-1, 1) * 1e-3; dgh[k] = r.u(-1, 1) * 1e-3; ddtl[k] = r.u(-1, 1) * 1e-2; ddqw[k] = r.u(-1, 1) * 1e-5;
                                   du[k] = r.u(-1, 1); dv[k] = r.u(-1, 1); ddz[k] = r.u(-1, 1); dzw[k] = r.u(-1, 1); dcf[k] = r.u(-1, 1) * 0.1; }
    double lhs[2] = {0, 0};
    for (int v = 0; v < 2; ++v) for (int k = 2; k <= n; ++k) {
      const double w[10] = {dqkw[k], dgm[k], dgh[k], du[k] - du[k - 1], dv[k] - dv[k - 1], ddz[k], ddz[k - 1], dzw[k], dcf[k], dcf[k - 1]};
      double dsm = 0, dsh = 0, dqd = 0;
      for (int i = 0; i < 10; ++i) { dsm += (v ? J0[30 * k + i] : F->mx.jac[20 * k + i]) * w[i]; dsh += (v ? J0[30 * k + 10 + i] : F->mx.jac[20 * k + 10 + i]) * w[i];
                                     dqd += (v ? J0[30 * k + 20 + i] : F->mx.jq[10 * k + i]) * w[i]; }
      const float el = P[6 * MAX + k], qkw = P[k], sm = P[7 * MAX + k], sh = P[8 * MAX + k], gm = P[MAX + k], gh = P[2 * MAX + k], dtl = P[3 * MAX + k], dqw = P[4 * MAX + k];
      const double qdiv = F->qdiv[k];
      const float elq = el * qkw, elh = float(double(elq) * qdiv), S = sm * gm + sh * gh, A = sh * dtl, Bq = sh * dqw, hA = elh * A, hB = elh * Bq;
      const double delq = double(el) * dqkw[k], dS = dsm * gm + double(sm) * dgm[k] + dsh * gh + double(sh) * dgh[k];
      const double delh = delq * qdiv + double(elq) * dqd, dA = dsh * dtl + double(sh) * ddtl[k], dB = dsh * dqw + double(sh) * ddqw[k];
      const double dhA = delh * A + double(elh) * dA, dhB = delh * Bq + double(elh) * dB;
      const double tk = delq * S + double(elq) * dS, tt = dhA * dtl + double(hA) * ddtl[k], tq = dhB * dqw + double(hB) * ddqw[k];
      const double tc = 0.5 * (dhA * dqw + double(hA) * ddqw[k]) + 0.5 * (dhB * dtl + double(hB) * ddtl[k]);
      lhs[v] += seed[k] * tk + seed[MAX + k] * tt + seed[2 * MAX + k] * tq + seed[3 * MAX + k] * tc;
    }
    for (int pl = 0; pl < 2; ++pl) {
      F->plant = pl ? 61 : 0; auto Bp = std::make_unique<ColumnBars>(); ColumnBars& B = *Bp; std::memset(&B, 0, sizeof B);
      double b[TPNB]{}, kb[3 * MAX]{};
      turbprod_reverse(n, 2.6f, 0, F->prod, F->qdiv, seed, b, false, kb); closure_prod_reverse(*F, kb, B);
      double rhs = 0;
      for (int k = 1; k <= n; ++k) rhs += (B.qkw[k] + b[k]) * dqkw[k] + (B.gm[k] + b[MAX + k]) * dgm[k] + (B.gh[k] + b[2 * MAX + k]) * dgh[k] + b[3 * MAX + k] * ddtl[k] + b[4 * MAX + k] * ddqw[k]
                                          + B.u[k] * du[k] + B.v[k] * dv[k] + (B.dz[k] + b[5 * MAX + k]) * ddz[k] + B.zw[k] * dzw[k] + B.cf_bl[k] * dcf[k];
      const double e = std::fabs(lhs[0] - rhs) / (std::fabs(lhs[0]) + std::fabs(rhs) + 1e-300);
      if (pl) { bool live = false; for (int i = 0; i < 10; ++i) live = live || F->mx.jac[20 * (n / 2) + i] != 0.0; if (live) { ++planted; plantB = std::min(plantB, e); } } else { worstB = std::max(worstB, e); dropB = std::min(dropB, std::fabs(lhs[1] - rhs) / (std::fabs(lhs[1]) + std::fabs(rhs) + 1e-300)); }
    }
  }
  std::printf("B: 400 random frames (%d levels, %d held by the rule): dot test worst %.2e; plant at one level (%d frames where the rule leaves that level live), smallest gap %.2e; level-2 dropped from the tangent, smallest gap %.2e\n",
              levels, holds, worstB, planted, plantB, dropB);
  const bool okB = worstB < 1e-13 && planted > 50 && plantB > 100.0 * worstB && dropB > 100.0 * worstB;
  std::printf("%s\n", okA && okB ? "PASS" : "FAIL");
  return okA && okB ? 0 : 1;
}
