#pragma once
// the forward-mode mirror of f_get_pblh (module_bl_mynnedmf.F90 GET_PBLH with
// stable_method = 1, the generated source's constant), for a tape-free backward of the boundary-layer height (generator
// freeze group 1 off). NOT wired into the join yet: this header and code/pblh_test.cpp only.
// The height answers six live words: the virtual potential temperature at the crossing level kc and at kc-1, the
// near-ground minimum (level kmin, searched below 200 m from level kts+1), zw(kz), dz(kc-1) and the friction velocity.
// Where no level crosses before kte-1 the height is zw(kts+1): kz = kts+1, word ZWK has partial 1 and the other
// profile words none. kc-1 and kmin may be the same level: the two partials are kept apart and the caller adds both
// to that level's bar. Values: the native statements in the native order. Partials: the generated rules
// (mw_ad_rules.inc: quotient from the float operands in double; min/max take the native operand, ties to the second;
// tanh by its float value). The caller compares the value with the forward's own pblh bit for bit and refuses the
// column if they differ (then the branches were not the forward's).
// Value operations are the MJ_F* macros of mixjac.hpp (the capture build maps them to the forward's own functions);
// MJ_FTANH is the forward's tanh there and the host's here.
#ifndef MJ_HD
#define MJ_HD inline
#endif
#ifndef MJ_FADD
#define MJ_FADD(a, b) ((float)((float)(a) + (float)(b)))
#define MJ_FSUB(a, b) ((float)((float)(a) - (float)(b)))
#define MJ_FMUL(a, b) ((float)((float)(a) * (float)(b)))
#define MJ_FDIV(a, b) ((float)((float)(a) / (float)(b)))
#endif
#ifndef MJ_FTANH
#include <cmath>
inline float pj_host_tanhf(float a) { return std::tanh(a); }
#define MJ_FTANH(a) pj_host_tanhf(a)
#endif
namespace pblhjac {
constexpr int NW = 6;
enum Word : int { THK = 0, THKM1, THMIN, ZWK, DZKM1, UST };
struct F { float v; double d[NW]; };
MJ_HD F cf(float v) { F r; r.v = v; for (int i = 0; i < NW; ++i) r.d[i] = 0.0; return r; }
MJ_HD F var(float v, int w) { F r = cf(v); r.d[w] = 1.0; return r; }
MJ_HD F add(const F& a, const F& b) { F r; r.v = MJ_FADD(a.v, b.v); for (int i = 0; i < NW; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
MJ_HD F sub(const F& a, const F& b) { F r; r.v = MJ_FSUB(a.v, b.v); for (int i = 0; i < NW; ++i) r.d[i] = a.d[i] - b.d[i]; return r; }
MJ_HD F mul(const F& a, const F& b) { F r; r.v = MJ_FMUL(a.v, b.v); const double x = a.v, y = b.v; for (int i = 0; i < NW; ++i) r.d[i] = a.d[i] * y + x * b.d[i]; return r; }
MJ_HD F dvd(const F& a, const F& b) { F r; r.v = MJ_FDIV(a.v, b.v); const double q = (double)a.v / (double)b.v, y = b.v; for (int i = 0; i < NW; ++i) r.d[i] = (a.d[i] - q * b.d[i]) / y; return r; }
MJ_HD F tnh(const F& a) { F r; r.v = MJ_FTANH(a.v); const double t = r.v, s = 1.0 - t * t; for (int i = 0; i < NW; ++i) r.d[i] = a.d[i] * s; return r; }
MJ_HD F mx(const F& a, const F& b) { return a.v > b.v ? a : b; }
MJ_HD F mn(const F& a, const F& b) { return a.v < b.v ? a : b; }

// kc: the crossing level (THK at kc, THKM1 and DZKM1 at kc-1); kmin: the level of THMIN (0: none found, no partial);
// kz: the level of ZWK. d: the six partials of the final pblh.
struct Out { float pblh; int kc, kmin, kz; double d[NW]; };

// thv, zw, dz: callables k -> float, native indices kts..kte (zw(k) the height of the lower face of level k).
template<class TH, class ZW, class DZ>
MJ_HD Out get_pblh(int kts, int kte, const TH& thv, const ZW& zw, const DZ& dz, float ust, float landsea) {
  Out o; o.kc = kts + 1; o.kmin = 0; o.kz = kts + 1;
  int k = kts + 1, kthv = 1; float minthv = 9.e9f;
  while (k <= kte && zw(k) <= 200.f) {                       // the native loop has no upper bound; zw passes 200 m long before kte
    if (minthv > thv(k)) { minthv = thv(k); kthv = k; o.kmin = k; }
    k = k + 1;
  }
  (void)kthv;
  const float delt = (MJ_FSUB(landsea, 1.5f) >= 0.0f) ? 0.75f : 1.25f;
  const F tmin = o.kmin ? var(minthv, THMIN) : cf(minthv);
  F pblh = cf(0.0f);
  for (k = kts + 1; k <= kte - 1; ++k) {
    if (thv(k) >= MJ_FADD(minthv, delt)) {
      const F tk = var(thv(k), THK), tkm = var(thv(k - 1), THKM1);
      const F fr = mn(dvd(sub(tk, add(tmin, cf(delt))), mx(sub(tk, tkm), cf(1e-6f))), cf(1.0f));
      pblh = sub(var(zw(k), ZWK), mul(var(dz(k - 1), DZKM1), fr)); o.kc = k; o.kz = k;
    }
    if (k == kte - 1) { pblh = var(zw(kts + 1), ZWK); o.kc = kts + 1; o.kz = kts + 1; }
    if (pblh.v > 0.0f) break;
  }
  const F one = cf(1.0f), p5 = cf(0.5f);
  F wt = add(mul(p5, tnh(dvd(sub(pblh, cf(100.0f)), cf(100.0f)))), p5);
  const F cp = add(mul(cf(400.0f), sub(one, wt)), mul(cf(700.0f), wt));
  F ps = mx(mul(var(ust, UST), cp), cf(10.0f));
  ps = mx(ps, sub(pblh, cf(200.0f)));
  ps = mn(ps, add(pblh, cf(200.0f)));
  wt = add(mul(p5, tnh(dvd(sub(pblh, cf(200.0f)), cf(170.0f)))), p5);
  pblh = add(mul(ps, sub(one, wt)), mul(pblh, wt));
  o.pblh = pblh.v; for (int i = 0; i < NW; ++i) o.d[i] = pblh.d[i];
  return o;
}

// the transpose: pblh's bar to the bars of the profile (arrays indexed by native level) and of the friction velocity
MJ_HD void reverse(const Out& o, double pblh_bar, double* thv_bar, double* zw_bar, double* dz_bar, double& ust_bar) {
  if (pblh_bar == 0.0) return;
  thv_bar[o.kc] += pblh_bar * o.d[THK]; thv_bar[o.kc - 1] += pblh_bar * o.d[THKM1];
  if (o.kmin) thv_bar[o.kmin] += pblh_bar * o.d[THMIN];
  zw_bar[o.kz] += pblh_bar * o.d[ZWK]; dz_bar[o.kc - 1] += pblh_bar * o.d[DZKM1];
  ust_bar += pblh_bar * o.d[UST];
}
}
