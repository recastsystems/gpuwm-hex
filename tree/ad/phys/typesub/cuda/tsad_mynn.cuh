// SPDX-License-Identifier: Apache-2.0
// tsad_mynn.cuh: MYNN-EDMF (c6d08709 capture) rules for the type-substitution generator (P2).
// The captured MYNN source pins every operation through named routines (r6_add, mynn_dmul, ...);
// each rule's primal calls that same routine on the primal operands, and its local partial is the
// real operation's derivative at those operands, in double.  Ported from the 11-02 lane's
// mynn_ad_rules.inc (same partials) onto the tsad value types.
#pragma once

namespace tsad {

#define TSAD_M2(NAME, VT, PT, WA, WB) \
template<class K> __device__ __forceinline__ VT<K> NAME(const VT<K>& a, const VT<K>& b) { \
  const PT r = ::NAME(a.v, b.v); const double x = (double)a.v, y = (double)b.v; (void)x; (void)y; \
  return VT<K>(r, K::lin2(a.d, (WA), b.d, (WB)), 0); } \
template<class K> __device__ __forceinline__ VT<K> NAME(const VT<K>& a, PT b) { return NAME(a, VT<K>(b)); } \
template<class K> __device__ __forceinline__ VT<K> NAME(PT a, const VT<K>& b) { return NAME(VT<K>(a), b); }
TSAD_M2(r6_add, VF, float, 1.0, 1.0)
TSAD_M2(r6_sub, VF, float, 1.0, -1.0)
TSAD_M2(r6_mul, VF, float, y, x)
TSAD_M2(r6_div, VF, float, 1.0 / y, -(x / y) / y)
TSAD_M2(mynn_dadd, VD, double, 1.0, 1.0)
TSAD_M2(mynn_dsub, VD, double, 1.0, -1.0)
TSAD_M2(mynn_dmul, VD, double, y, x)
TSAD_M2(mynn_ddiv, VD, double, 1.0 / y, -(x / y) / y)
#undef TSAD_M2
template<class K> __device__ __forceinline__ VF<K> r6_neg(const VF<K>& a) { return VF<K>(::r6_neg(a.v), K::neg(a.d), 0); }
template<class K> __device__ __forceinline__ VD<K> r6_neg(const VD<K>& a) { return VD<K>(::r6_neg(a.v), K::neg(a.d), 0); }

#define TSAD_M1(NAME, DEXPR) \
template<class K> __device__ __forceinline__ VF<K> NAME(const VF<K>& a) { \
  const float rf = ::NAME(a.v); if (!K::on(a.d)) return VF<K>(rf); \
  const double x = (double)a.v; (void)x; return VF<K>(rf, K::lin1(a.d, (DEXPR)), 0); }
TSAD_M1(mynn_exp, ::exp(x))
TSAD_M1(mynn_log, 1.0 / x)
TSAD_M1(mynn_log10, 1.0 / (x * 2.302585092994045684))
TSAD_M1(mynn_tanh, (1.0 - ::tanh(x) * ::tanh(x)))
TSAD_M1(mynn_erf, 1.1283791670955125739 * ::exp(-x * x))
TSAD_M1(mynn_atan, 1.0 / (1.0 + x * x))
TSAD_M1(mynn_sqrt, (x > 0.0 ? 0.5 / ::sqrt(x) : 0.0))
TSAD_M1(mynn_abs, (::signbit(x) ? -1.0 : 1.0))
#undef TSAD_M1
template<class K> __device__ __forceinline__ VD<K> mynn_sqrt(const VD<K>& a) {
  const double r = ::mynn_sqrt(a.v); return VD<K>(r, K::lin1(a.d, a.v > 0.0 ? 0.5 / ::sqrt(a.v) : 0.0), 0);
}
template<class K> __device__ __forceinline__ VD<K> mynn_abs(const VD<K>& a) {
  return VD<K>(::mynn_abs(a.v), ::signbit(a.v) ? K::neg(a.d) : K::same(a.d), 0);
}
// sign(a, b) = |a| with b's sign: derivative of a times the executed sign change.
template<class K> __device__ __forceinline__ VF<K> mynn_sign(const VF<K>& a, const VF<K>& b) {
  const float r = ::mynn_sign(a.v, b.v); return VF<K>(r, (::signbit(r) == ::signbit(a.v)) ? K::same(a.d) : K::neg(a.d), 0);
}
template<class K> __device__ __forceinline__ VD<K> mynn_sign(const VD<K>& a, const VD<K>& b) {
  const double r = ::mynn_sign(a.v, b.v); return VD<K>(r, (::signbit(r) == ::signbit(a.v)) ? K::same(a.d) : K::neg(a.d), 0);
}
template<class K> __device__ __forceinline__ VF<K> mynn_sign(const VF<K>& a, float b) { return mynn_sign(a, VF<K>(b)); }
template<class K> __device__ __forceinline__ VF<K> mynn_sign(float a, const VF<K>& b) { return mynn_sign(VF<K>(a), b); }
template<class K> __device__ __forceinline__ VF<K> mynn_pow(const VF<K>& x, const VF<K>& y) {
  const float rf = ::mynn_pow(x.v, y.v);
  const double xv = x.v, yv = y.v, r = xv > 0.0 ? ::pow(xv, yv) : (double)rf;
  double wx = 0.0, wy = 0.0;
  if (K::on(x.d)) wx = xv != 0.0 ? yv * (r / xv) : (yv == 1.0 ? 1.0 : 0.0);
  if (K::on(y.d)) wy = xv > 0.0 ? r * ::log(xv) : 0.0;
  return VF<K>(rf, K::lin2(x.d, wx, y.d, wy), 0);
}
template<class K> __device__ __forceinline__ VF<K> mynn_pow(const VF<K>& x, float y) { return mynn_pow(x, VF<K>(y)); }
template<class K> __device__ __forceinline__ VF<K> mynn_pow(float x, const VF<K>& y) { return mynn_pow(VF<K>(x), y); }
template<int N, class K> __device__ __forceinline__ VF<K> mynn_ipow(const VF<K>& a) {
  const float r = ::mynn_ipow<N>(a.v); const double x = a.v;
  return VF<K>(r, K::lin1(a.d, N == 2 ? 2.0 * x : 3.0 * x * x), 0);
}
template<int N, class K> __device__ __forceinline__ VD<K> mynn_ipow(const VD<K>& a) {
  const double r = ::mynn_ipow<N>(a.v); const double x = a.v;
  return VD<K>(r, K::lin1(a.d, N == 2 ? 2.0 * x : 3.0 * x * x), 0);
}
// Integer-valued routines: zero derivative.
template<class K> __device__ __forceinline__ int mynn_nint(const VF<K>& a) { return ::mynn_nint(a.v); }
template<class K> __device__ __forceinline__ int mynn_floor(const VF<K>& a) { return ::mynn_floor(a.v); }
template<class K> __device__ __forceinline__ int mynn_ceiling(const VF<K>& a) { return ::mynn_ceiling(a.v); }
// Lane broadcast of an active value: primal and derivative travel together.  The generated
// copies run lane width 1 (one thread per column), where the source lane is the thread itself.
template<class K> __device__ __forceinline__ VF<K> __shfl_sync(unsigned m, const VF<K>& v, int src, int w) {
  return VF<K>(::__shfl_sync(m, v.v, src, w), ::__shfl_sync(m, v.d, src, w), 0);
}
template<class K> __device__ __forceinline__ VD<K> __shfl_sync(unsigned m, const VD<K>& v, int src, int w) {
  return VD<K>(::__shfl_sync(m, v.v, src, w), ::__shfl_sync(m, v.d, src, w), 0);
}

}  // namespace tsad
