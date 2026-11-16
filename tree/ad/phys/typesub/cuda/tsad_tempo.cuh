// SPDX-License-Identifier: Apache-2.0
// tsad_tempo.cuh: TEMPO-style rules for the type-substitution generator (P2).
// Included after the verbatim captured TEMPO source and tsad.cuh, before the generated copies.
//
// 1. Leaf rules for the source's own maths wrappers (tempo_exp/log/log10/pow): the primal calls
//    the identical wrapper on the primal operands, the derivative is analytic in double.
// 2. tmax/tmin overloads for active operands, with the same promoted comparison and result
//    type as the source's `a>b?a:b` templates (a ternary across VF/double cannot compile).
// 3. The serial column shape for the generated copies.  The captured forward runs one warp per
//    column (TEMPO_WARP 1); the source states both shapes evaluate the same statements on the
//    same operands, so the generated derivative runs the serial transcription, one thread per
//    column, and keeps a single per-thread tape.  The originals above are untouched.
#pragma once

namespace tsad {

#define TSAD_LEAF1(NAME, DEXPR) \
template<class K> __device__ __forceinline__ VF<K> NAME(const VF<K>& a) { \
  const float rf = ::NAME(a.v); \
  if (!K::on(a.d)) return VF<K>(rf); \
  const double x = (double)a.v, r = (double)rf; (void)x; (void)r; \
  return VF<K>(rf, K::lin1(a.d, (DEXPR)), 0); } \
template<class K> __device__ __forceinline__ VD<K> NAME(const VD<K>& a) { \
  const double rf = ::NAME(a.v); \
  if (!K::on(a.d)) return VD<K>(rf); \
  const double x = a.v, r = rf; (void)x; (void)r; \
  return VD<K>(rf, K::lin1(a.d, (DEXPR)), 0); }
// Local partials are the real function's derivative at the executed operands, evaluated in
// double (exp re-evaluated in double, not the rounded float result).
#ifdef TSAD_HAND_PARTIALS
TSAD_LEAF1(tempo_exp, r)
#else
TSAD_LEAF1(tempo_exp, ::exp(x))
#endif
TSAD_LEAF1(tempo_log, 1.0 / x)
TSAD_LEAF1(tempo_log10, 1.0 / (x * 2.302585092994045684))
#undef TSAD_LEAF1

// d(x^y)/dx = y x^(y-1), d/dy = x^y ln x, with x^y re-evaluated in double at the executed
// operands (x = 0: 0 unless y == 1; x < 0: 0, a domain the TEMPO sources never reach).
template<class VT, class PT>
__device__ __forceinline__ VT tsad_pow_rule(const VT& x, const VT& y, PT rf) {
  typedef typename VT::kind K;
  const double xv = (double)x.v, yv = (double)y.v;
#ifdef TSAD_HAND_PARTIALS  // diagnostic only: the 11-02 hand pairs' partials in the operand type
  typedef decltype(x.v) P;
  if (x.v > (P)0) {
    const P da = y.v * rf / x.v, db = rf * ::tempo_log(x.v);
    return VT(rf, K::lin2(x.d, (double)da, y.d, (double)db), 0);
  }
  return VT(rf, K::lin2(x.d, 0.0, y.d, 0.0), 0);
#endif
  const double r = xv > 0.0 ? ::pow(xv, yv) : (double)rf;
  double wx = 0.0, wy = 0.0;
  if (K::on(x.d)) wx = xv != 0.0 ? yv * (r / xv) : (yv == 1.0 ? 1.0 : 0.0);
  if (K::on(y.d)) wy = (xv > 0.0) ? r * ::log(xv) : 0.0;
  return VT(rf, K::lin2(x.d, wx, y.d, wy), 0);
}
template<class K> __device__ __forceinline__ VF<K> tempo_pow(const VF<K>& x, const VF<K>& y) {
  return tsad_pow_rule(x, y, ::tempo_pow(x.v, y.v));
}
template<class K> __device__ __forceinline__ VD<K> tempo_pow(const VD<K>& x, const VD<K>& y) {
  return tsad_pow_rule(x, y, ::tempo_pow(x.v, y.v));
}
template<class K> __device__ __forceinline__ VD<K> copysign(const VD<K>& a, const VD<K>& b) {
  const double r = ::copysign(a.v, b.v);
  return VD<K>(r, (::signbit(r) == ::signbit(a.v)) ? K::same(a.d) : K::neg(a.d), 0);
}
template<class K> __device__ __forceinline__ VF<K> copysign(const VF<K>& a, const VF<K>& b) {
  const float r = ::copysignf(a.v, b.v);
  return VF<K>(r, (::signbit(r) == ::signbit(a.v)) ? K::same(a.d) : K::neg(a.d), 0);
}

// tmax/tmin: more specialised than the source's templates, so chosen for active operands.
#define TSAD_TMINMAX(A, B) \
template<class K> __device__ __forceinline__ typename bin<A, B>::R tmax(const A& a, const B& b) { \
  typedef typename bin<A, B>::R R; const R x = R(a), y = R(b); return x > y ? x : y; } \
template<class K> __device__ __forceinline__ typename bin<A, B>::R tmin(const A& a, const B& b) { \
  typedef typename bin<A, B>::R R; const R x = R(a), y = R(b); return x < y ? x : y; }
TSAD_TMINMAX(VF<K>, VF<K>) TSAD_TMINMAX(VD<K>, VD<K>)
TSAD_TMINMAX(VF<K>, VD<K>) TSAD_TMINMAX(VD<K>, VF<K>)
TSAD_TMINMAX(VF<K>, float) TSAD_TMINMAX(float, VF<K>)
TSAD_TMINMAX(VF<K>, int) TSAD_TMINMAX(int, VF<K>)
TSAD_TMINMAX(VF<K>, double) TSAD_TMINMAX(double, VF<K>)
TSAD_TMINMAX(VD<K>, double) TSAD_TMINMAX(double, VD<K>)
TSAD_TMINMAX(VD<K>, float) TSAD_TMINMAX(float, VD<K>)
TSAD_TMINMAX(VD<K>, int) TSAD_TMINMAX(int, VD<K>)
#undef TSAD_TMINMAX

}  // namespace tsad

// Serial column shape for the generated copies (see 3. above).
#undef TEMPO_WARP
#define TEMPO_WARP 0
#undef TEMPO_LANE
#undef TEMPO_STRIDE
#undef TEMPO_SYNC
#undef TEMPO_FULL
#undef TEMPO_LEVELS
#undef TEMPO_LEVELS_DOWN
#undef TEMPO_SLOTS
#undef TEMPO_LEAD
#define TEMPO_LANE 0
#define TEMPO_STRIDE 1
#define TEMPO_SYNC() ((void)0)
#define TEMPO_LEAD (TEMPO_LANE == 0)
#define TEMPO_LEVELS(k, n) for (int k = 1 + TEMPO_LANE; k <= (n); k += TEMPO_STRIDE)
#define TEMPO_LEVELS_DOWN(k, n) for (int k = (n); k >= 1; --k)
#define TEMPO_SLOTS(k, lo, hi) for (int k = (lo) + TEMPO_LANE; k < (hi); k += TEMPO_STRIDE)
