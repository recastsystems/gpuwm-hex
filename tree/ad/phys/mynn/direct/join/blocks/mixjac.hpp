#pragma once
// the forward-mode copy of the level-2.5 stability block of f_mym_turbulence, for the
// tape-free backward of the live mixing (gen_whole.py --freeze closure --mix-live q:1,s:1 [--weakshear T]).
// In that form the diffusivities keep their derivative through the final stability functions ksm = sm(k), ksh = sh(k) of
// the turbulence loop (F90:2757-3082, closure 2.6: no level-3 block). At level k (2..n) they depend on ten live words:
//   0 qkw(k)  1 gm(k)  2 gh(k)  3 du = u(k)-u(k-1)  4 dv = v(k)-v(k-1)  5 dz(k)  6 dz(k-1)  7 zw(k)  8 cldfra(k)  9 cldfra(k-1)
// and on held words: el(k), the level-2 sm/sh, pblh, the plume's edmf_a/edmf_w (and the down-draft's), tkeprod_dn.
// stab() recomputes ksm, ksh from those words with the native statements in the native order (values) and carries the ten
// partials in double with the rules of the generated tangent (mw_ad_rules.inc: product, quotient, square, square root;
// max/min take the native operand, ties to the second; a constant has none). The caller compares the recomputed values
// with the forward's own sm(k), sh(k) word for word and refuses the column if they differ (then the branches were not the
// forward's). Value operations are macros: the capture build maps them to the forward's own r6_* / mynn_d* functions,
// the CPU test to IEEE operations.
#ifndef MJ_HD
#define MJ_HD inline
#endif
#ifndef MJ_FADD
#define MJ_FADD(a, b) ((float)((float)(a) + (float)(b)))
#define MJ_FSUB(a, b) ((float)((float)(a) - (float)(b)))
#define MJ_FMUL(a, b) ((float)((float)(a) * (float)(b)))
#define MJ_FDIV(a, b) ((float)((float)(a) / (float)(b)))
#define MJ_DADD(a, b) ((double)(a) + (double)(b))
#define MJ_DSUB(a, b) ((double)(a) - (double)(b))
#define MJ_DMUL(a, b) ((double)(a) * (double)(b))
#define MJ_DDIV(a, b) ((double)(a) / (double)(b))
#define MJ_DSQRT(a) mj_host_sqrt(a)
#define MJ_FSQRT(a) mj_host_sqrtf(a)
#include <cmath>
inline double mj_host_sqrt(double a) { return std::sqrt(a); }
inline float mj_host_sqrtf(float a) { return std::sqrt(a); }
#endif
namespace mixjac {
constexpr int NJ = 10;
enum Word : int { QKW = 0, GM, GH, DU, DV, DZK, DZKM1, ZWK, CFK, CFKM1 };
struct F { float v; double d[NJ]; };
struct D { double v; double d[NJ]; };
MJ_HD F cf(float v) { F r; r.v = v; for (int i = 0; i < NJ; ++i) r.d[i] = 0.0; return r; }
MJ_HD D cd(double v) { D r; r.v = v; for (int i = 0; i < NJ; ++i) r.d[i] = 0.0; return r; }
MJ_HD F var(float v, int w) { F r = cf(v); r.d[w] = 1.0; return r; }
MJ_HD D toD(const F& a) { D r; r.v = (double)a.v; for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i]; return r; }
MJ_HD F toF(const D& a) { F r; r.v = (float)a.v; for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i]; return r; }
MJ_HD F add(const F& a, const F& b) { F r; r.v = MJ_FADD(a.v, b.v); for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
MJ_HD F sub(const F& a, const F& b) { F r; r.v = MJ_FSUB(a.v, b.v); for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] - b.d[i]; return r; }
MJ_HD F mul(const F& a, const F& b) { F r; r.v = MJ_FMUL(a.v, b.v); const double x = a.v, y = b.v; for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] * y + x * b.d[i]; return r; }
MJ_HD F dvd(const F& a, const F& b) { F r; r.v = MJ_FDIV(a.v, b.v); const double q = (double)a.v / (double)b.v, y = b.v; for (int i = 0; i < NJ; ++i) r.d[i] = (a.d[i] - q * b.d[i]) / y; return r; }
MJ_HD F neg(const F& a) { F r; r.v = -a.v; for (int i = 0; i < NJ; ++i) r.d[i] = -a.d[i]; return r; }
MJ_HD F sq(const F& a) { F r; r.v = MJ_FMUL(a.v, a.v); const double x = a.v; for (int i = 0; i < NJ; ++i) r.d[i] = 2.0 * x * a.d[i]; return r; }
MJ_HD F mx(const F& a, const F& b) { return a.v > b.v ? a : b; }
MJ_HD F mn(const F& a, const F& b) { return a.v < b.v ? a : b; }
MJ_HD D dadd(const D& a, const D& b) { D r; r.v = MJ_DADD(a.v, b.v); for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
MJ_HD D dsub(const D& a, const D& b) { D r; r.v = MJ_DSUB(a.v, b.v); for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] - b.d[i]; return r; }
MJ_HD D dmul(const D& a, const D& b) { D r; r.v = MJ_DMUL(a.v, b.v); for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i]; return r; }
MJ_HD D ddv(const D& a, const D& b) { D r; r.v = MJ_DDIV(a.v, b.v); for (int i = 0; i < NJ; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) / b.v; return r; }
MJ_HD D dsqrt(const D& a) { D r; r.v = MJ_DSQRT(a.v); for (int i = 0; i < NJ; ++i) r.d[i] = a.d[i] / (2.0 * r.v); return r; }
MJ_HD D dneg(const D& a) { D r; r.v = -a.v; for (int i = 0; i < NJ; ++i) r.d[i] = -a.d[i]; return r; }
MJ_HD D dmx(const D& a, const D& b) { return a.v > b.v ? a : b; }
MJ_HD float fmx(float a, float b) { return a > b ? a : b; }
MJ_HD float fmn(float a, float b) { return a < b ? a : b; }

// the closure constants of the forward (module_bl_mynnedmf.F90 constants, read from the forward's own globals)
struct Con { float b1, a1, a2, c1, e1c, e2c, e3c, e4c, e5c; };
// the level's words: live (seeded) then held
struct In {
  float qkw, gm, gh, uk, ukm1, vk, vkm1, dzk, dzkm1, zwk, cfk, cfkm1;
  float el, sm2, sh2, pblh, eak, eakm1, ewk, ewkm1, eaddk, ewddk, tkdn, closure;
  int tke_budget;
};

// ---- the level-2 functions of f_mym_level2's last loop, live (generator freeze group 2 off).
// sm2(k), sh2(k) are functions of gm(k), gh(k) alone through ri = -gh / max(gm, 1e-10); with ckmod = 1 every coefficient
// (a2fac, f1, rf1, smc, shc, ri1..ri4) answers ri too. Values: the native statements in the native order. Partials: the
// generated rules, with the three closed forms of cuda/mw_ad_rules.inc (mw_l2_rf, mw_l2_sh, mw_l2_sm) in place of the
// generic quotient rule, term for term in their order.
struct L2Con { float g1, g2, b1, a1, a2, c1, c2, c5; int ckmod; };
MJ_HD F l2_rf(const F& ri1, const F& ri, const F& ri2, const F& ri3, const F& ri4) {
  F r; const float arg = MJ_FADD(MJ_FSUB(MJ_FMUL(ri.v, ri.v), MJ_FMUL(ri3.v, ri.v)), ri4.v);
  r.v = MJ_FMUL(ri1.v, MJ_FSUB(MJ_FADD(ri.v, ri2.v), MJ_FSQRT(arg)));
  const double x = ri.v, a1 = ri1.v, a2 = ri2.v, a3 = ri3.v, a4 = ri4.v;
  const double Dq = x * x - a3 * x + a4, s = MJ_DSQRT(Dq > 0.0 ? Dq : 0.0), t = 2.0 * x - a3;
  const double c = t > 0.0 ? (4.0 * a4 - a3 * a3) / (2.0 * s * (2.0 * s + t)) : 1.0 - t / (2.0 * s);
  const double w = t > 0.0 ? (x * (2.0 * a2 + a3) + (a2 * a2 - a4)) / (x + a2 + s) : x + a2 - s;
  for (int i = 0; i < NJ; ++i) {
    double d = 0.0;
    if (ri1.d[i] != 0.0) d = d + ri1.d[i] * w;
    if (ri.d[i] != 0.0) d = d + ri.d[i] * (a1 * c);
    if (ri2.d[i] != 0.0) d = d + ri2.d[i] * a1;
    if (ri3.d[i] != 0.0) d = d + ri3.d[i] * (a1 * x / (2.0 * s));
    if (ri4.d[i] != 0.0) d = d + ri4.d[i] * (-a1 / (2.0 * s));
    r.d[i] = d;
  }
  return r;
}
MJ_HD F l2_sh(const F& shc, const F& rfc, const F& rf) {
  F r; const float den = MJ_FSUB(1.0f, rf.v); r.v = MJ_FDIV(MJ_FMUL(shc.v, MJ_FSUB(rfc.v, rf.v)), den);
  const double b = 1.0 - (double)rf.v, g = ((double)rfc.v - (double)rf.v) / b, wr = (double)shc.v / b, wf = (double)shc.v * ((double)rfc.v - 1.0) / (b * b);
  for (int i = 0; i < NJ; ++i) {
    double d = 0.0;
    if (shc.d[i] != 0.0) d = d + shc.d[i] * g;
    if (rfc.d[i] != 0.0) d = d + rfc.d[i] * wr;
    if (rf.d[i] != 0.0) d = d + rf.d[i] * wf;
    r.d[i] = d;
  }
  return r;
}
MJ_HD F l2_sm(const F& smc, const F& rf1, const F& rf2, const F& rf, const F& sh) {
  F r; const float den = MJ_FSUB(rf2.v, rf.v); r.v = MJ_FMUL(MJ_FDIV(MJ_FMUL(smc.v, MJ_FSUB(rf1.v, rf.v)), den), sh.v);
  const double Dq = (double)rf2.v - (double)rf.v, h = ((double)rf1.v - (double)rf.v) / Dq, s = (double)smc.v * (double)sh.v;
  for (int i = 0; i < NJ; ++i) {
    double d = 0.0;
    if (smc.d[i] != 0.0) d = d + smc.d[i] * (h * (double)sh.v);
    if (rf1.d[i] != 0.0) d = d + rf1.d[i] * (s / Dq);
    if (rf2.d[i] != 0.0) d = d + rf2.d[i] * (-s * h / Dq);
    if (rf.d[i] != 0.0) d = d + rf.d[i] * (s * ((double)rf1.v - (double)rf2.v) / (Dq * Dq));
    if (sh.d[i] != 0.0) d = d + sh.d[i] * ((double)smc.v * h);
    r.d[i] = d;
  }
  return r;
}
MJ_HD void level2(float gmv, float ghv, const L2Con& c, F& sm, F& sh) {
  const F one = cf(1.0f), zero = cf(0.0f);
  const F gm = var(gmv, GM), gh = var(ghv, GH);
  const F ri = neg(dvd(gh, mx(gm, cf(1.0e-10f))));                                 // ri(k) = -gh(k) / max(duz, 1e-10), gm(k) = duz
  const F a2fac = c.ckmod == 1 ? dvd(one, add(one, mx(zero, ri))) : one;
  const float g12 = MJ_FADD(c.g1, c.g2), omc2 = MJ_FSUB(1.0f, c.c2), omc5 = MJ_FSUB(1.0f, c.c5), bg = MJ_FMUL(c.b1, MJ_FSUB(c.g1, c.c1));
  const F rfc = cf(MJ_FDIV(c.g1, g12));
  const F f1 = add(add(cf(bg), mul(mul(mul(cf(MJ_FMUL(3.0f, c.a2)), a2fac), cf(omc2)), cf(omc5))), cf(MJ_FMUL(MJ_FMUL(2.0f, c.a1), MJ_FSUB(3.0f, MJ_FMUL(2.0f, c.c2)))));
  const F f2 = cf(MJ_FSUB(MJ_FMUL(c.b1, g12), MJ_FMUL(MJ_FMUL(3.0f, c.a1), omc2)));
  const F rf1 = dvd(cf(bg), f1);
  const F rf2 = dvd(cf(MJ_FMUL(c.b1, c.g1)), f2);
  const F a2a = mul(cf(c.a2), a2fac);
  const F smc = dvd(mul(dvd(cf(c.a1), a2a), f1), f2);
  const F shc = mul(mul(cf(3.0f), a2a), cf(g12));
  const F ri1 = dvd(cf(0.5f), smc);
  const F ri2 = mul(rf1, smc);
  const F ri3 = sub(mul(mul(cf(4.0f), rf2), smc), mul(cf(2.0f), ri2));
  const F ri4 = sq(ri2);
  const F rf = mn(l2_rf(ri1, ri, ri2, ri3, ri4), rfc);
  sh = l2_sh(shc, rfc, rf); sm = l2_sm(smc, rf1, rf2, rf, sh);
}

// 1: computed on the path mirrored here; 0: outside it (closure 2.7 or above), the caller refuses the column.
// stab3: sm2, sh2 are the level-2 functions as carriers (cf(x.sm2), cf(x.sh2) where they are held; level2()'s where they
// are live) and qdv is the third output, qdiv (the constant one on the level-2.5 arm).
MJ_HD int stab3(const In& x, const Con& c, const F& sm2, const F& sh2, F& sm, F& sh, D& qdv) {
  if (!(x.closure < 2.7f)) return 0;
  const F one = cf(1.0f), zero = cf(0.0f);
  const F dz_k = var(x.dzk, DZK), dz_m = var(x.dzkm1, DZKM1);
  const F dzk = mul(cf(0.5f), add(dz_k, dz_m));                                   // dzk = p5*(dz(k)+dz(k-1))
  const double elsq = (double)MJ_FMUL(x.el, x.el);                               // elsq = el**2 (held)
  const F qkw = var(x.qkw, QKW), gm = var(x.gm, GM), gh = var(x.gh, GH);
  D q3sq = toD(sq(qkw));                                                          // q3sq = qkw**2
  const D q2sq = dmul(dmul(cd((double)c.b1), cd(elsq)), toD(add(mul(sm2, gm), mul(sh2, gh))));
  const F shf = mx(sh2, cf(1e-5f));                                               // sh(k) = max(sh(k), 1e-5)
  const F du = var(MJ_FSUB(x.uk, x.ukm1), DU), dv = var(MJ_FSUB(x.vk, x.vkm1), DV);
  F duz = add(sq(du), sq(dv));
  duz = dvd(duz, sq(dzk));
  const F ri = neg(dvd(gh, mx(duz, cf(1.0e-10f))));
  const F a2fac = dvd(one, add(one, mx(zero, ri)));                               // ckmod = 1
  F prlim;
  if (ri.v >= 1.0f) prlim = mn(mul(cf(7.0f), ri), cf(6.0f));
  else if (ri.v >= 0.01f && ri.v <= 1.0f) prlim = mn(add(mul(cf(6.873f), ri), dvd(one, mul(cf(6.873f), ri))), cf(6.0f));
  else prlim = cf(6.0f);
  const F wt = mn(dvd(var(x.zwk, ZWK), cf(fmx(x.pblh, 200.0f))), one);
  prlim = add(mul(wt, prlim), mul(sub(one, wt), mn(prlim, cf(4.0f))));
  const D gmel = dmul(toD(gm), cd(elsq)), ghel = dmul(toD(gh), cd(elsq));
  const double dlsq = elsq;
  if (MJ_DDIV(q3sq.v, dlsq) < (double)(-gh.v)) q3sq = dneg(dmul(cd(dlsq), toD(gh)));
  if (q3sq.v < q2sq.v) {
    const D qdiv = dsqrt(ddv(q3sq, q2sq));
    sh = toF(dmul(toD(shf), qdiv));
    sm = toF(dmul(toD(sm2), qdiv));
    qdv = qdiv;
  } else {
    qdv = cd(1.0);
    const D a2 = toD(a2fac);
    const D e1 = dsub(q3sq, dmul(dmul(cd((double)c.e1c), ghel), a2));
    const D e2 = dsub(q3sq, dmul(dmul(cd((double)c.e2c), ghel), a2));
    const D e3 = dadd(e1, dmul(dmul(cd((double)c.e3c), ghel), toD(sq(a2fac))));
    const D e4 = dsub(e1, dmul(dmul(cd((double)c.e4c), ghel), a2));
    D eden = dadd(dmul(e2, e4), dmul(dmul(e3, cd((double)c.e5c)), gmel));
    eden = dmx(eden, cd(1.0e-20));
    const float c13 = MJ_FMUL(3.0f, c.c1), c135 = MJ_FMUL(c13, c.e5c);
    sm = toF(ddv(dmul(dmul(q3sq, cd((double)c.a1)), dsub(e3, dmul(cd((double)c13), e4))), eden));
    const D A = dadd(dmul(cd((double)c135), gmel), e2), Bq = dmul(q3sq, toD(mul(cf(c.a2), a2fac)));
    sh = toF(ddv(x.tke_budget == 1 ? dmul(Bq, A) : dmul(A, Bq), eden));
  }
  if (sh.v > 4.0f) sh = cf(4.0f);
  if (sh.v < 0.0f) sh = cf(0.0f);
  const F shb = mx(cf(0.004f), sh);
  sm = mn(sm, mul(prlim, shb));
  const F cldavg = mul(cf(0.5f), add(var(x.cfkm1, CFKM1), var(x.cfk, CFK)));
  const float am = MJ_FMUL(MJ_FMUL(MJ_FMUL(0.5f, MJ_FADD(x.eakm1, x.eak)), 0.5f), MJ_FADD(x.ewkm1, x.ewk));
  const float ad = MJ_FMUL(x.eaddk, x.ewddk);
  const float mfmax = fmx(am, ad < 0.0f ? -ad : ad);
  const F mfl = cf(MJ_FMUL(0.04f, fmn(MJ_FMUL(10.0f, mfmax), 1.0f)));
  sm = mx(sm, mfl); sh = mx(mfl, sh);
  const F cl = mul(cf(0.04f), mn(cldavg, cf(0.5f)));
  sm = mx(sm, cl); sh = mx(cl, sh);
  if (x.tkdn > 1e-6f) { sm = mx(sm, cf(0.01f)); sh = mx(sh, cf(0.01f)); }
  return 1;
}
// the mixing form: the level-2 functions held, two outputs
MJ_HD int stab(const In& x, const Con& c, F& sm, F& sh) { D q; return stab3(x, c, cf(x.sm2), cf(x.sh2), sm, sh, q); }
}
