// SPDX-License-Identifier: Apache-2.0
// ======================================================================
// adreal.cuh: executed-branch tape types for the generated TEMPO tangent
// and adjoint (autodiff physics rung, unit P5). Private program code.
//
// AR<T> carries the primal value v (exactly the float or double the
// forward computes, same operation, same operands, same order) and the
// tape index i of the node that holds its derivative (0 = constant).
// Every arithmetic operation on an active value appends one node with up
// to two parents and their partial derivatives, evaluated in double from
// the float/double primal operands. Comparisons return the primal result
// (the executed branch) and append one bit to the column's branch log;
// conversions to integers return the primal integer and append it to the
// column's integer log (table cells, counts). The tangent is the forward
// sweep of the tape, the adjoint its reverse sweep: both are the exact
// derivative of the branch the forecast took.
//
// One thread runs one column (TEMPO_WARP 0, the serial shape the forward
// source documents as evaluating the same statements on the same
// operands). Tapes are interleaved [node][thread] in global memory; the
// per-thread counters live in shared memory.
// ======================================================================
#ifndef AD_BLOCK
#define AD_BLOCK 64
#endif

struct AdTape {
  int* a;            // parent 1 (-1 marks an input leaf)
  int* b;            // parent 2 (0 = none)
  double* pa;        // partial wrt parent 1
  double* pb;        // partial wrt parent 2
  unsigned* br;      // branch bits [word][thread]
  int* il;           // integer log [entry][thread]
  int* count;        // per thread: nodes, branch bits, int entries, overflow
  const unsigned* rbr; // replay: branch bits to impose (null: free run)
  const int* ril;      // replay: integer log to impose
  long long T;       // threads in the tile (interleave stride)
  long long cap;     // node capacity per thread (0: count only)
  long long bcap;    // branch-bit capacity per thread (bits)
  long long icap;    // integer-log capacity per thread
};
__device__ AdTape g_ad;
__shared__ int ad_n_s[AD_BLOCK];
__shared__ int ad_b_s[AD_BLOCK];
__shared__ int ad_i_s[AD_BLOCK];
__shared__ int ad_o_s[AD_BLOCK];

__device__ __forceinline__ long long ad_tid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }
__device__ __forceinline__ void ad_begin() {
  ad_n_s[threadIdx.x] = 1; ad_b_s[threadIdx.x] = 0; ad_i_s[threadIdx.x] = 0; ad_o_s[threadIdx.x] = 0;
}
__device__ __forceinline__ void ad_end() {
  const long long t = ad_tid();
  g_ad.count[4 * t + 0] = ad_n_s[threadIdx.x];
  g_ad.count[4 * t + 1] = ad_b_s[threadIdx.x];
  g_ad.count[4 * t + 2] = ad_i_s[threadIdx.x];
  g_ad.count[4 * t + 3] = ad_o_s[threadIdx.x];
}
__device__ __noinline__ int ad_push(int a, double pa, int b, double pb) {
  const int n = ad_n_s[threadIdx.x]++;
  if (n < g_ad.cap) {
    const long long o = (long long)n * g_ad.T + ad_tid();
    g_ad.a[o] = a; g_ad.b[o] = b; g_ad.pa[o] = pa; g_ad.pb[o] = pb;
  } else if (g_ad.cap > 0) {
    ad_o_s[threadIdx.x] |= 1;
  }
  return n;
}
__device__ __forceinline__ int ad_node1(int a, double pa) {
  return a ? ad_push(a, pa, 0, 0.0) : 0;
}
__device__ __forceinline__ int ad_node2(int a, double pa, int b, double pb) {
  if (!a) return b ? ad_push(b, pb, 0, 0.0) : 0;
  if (!b) return ad_push(a, pa, 0, 0.0);
  return ad_push(a, pa, b, pb);
}
__device__ __forceinline__ int ad_leaf() { return ad_push(-1, 0.0, 0, 0.0); }
// Branch and integer logs. In replay mode (proof only: the branch-held
// finite difference) the logged decision of the base run is imposed instead.
__device__ __noinline__ bool ad_bit(bool r) {
  const int n = ad_b_s[threadIdx.x]++;
  if (n < g_ad.bcap) {
    const long long o = (long long)(n >> 5) * g_ad.T + ad_tid();
    if (g_ad.rbr) r = (g_ad.rbr[o] >> (n & 31)) & 1u;
    if (r) g_ad.br[o] |= 1u << (n & 31);
  } else if (g_ad.bcap > 0) {
    ad_o_s[threadIdx.x] |= 2;
  }
  return r;
}
__device__ __noinline__ long long ad_int(long long v) {
  const int n = ad_i_s[threadIdx.x]++;
  if (n < g_ad.icap) {
    const long long o = (long long)n * g_ad.T + ad_tid();
    if (g_ad.ril) v = g_ad.ril[o];
    g_ad.il[o] = (int)v;
  } else if (g_ad.icap > 0) ad_o_s[threadIdx.x] |= 4;
  return v;
}

// ---------------------------------------------------------------- traits
// AR and every overload live in namespace adr: argument-dependent lookup then
// finds them from the active column code, and never the forward's own global
// templates (tmax, tempo_pow), which would tie with the active copies.
namespace adr {
template<class T> struct ad_arith { static const bool value = false; };
template<> struct ad_arith<float> { static const bool value = true; };
template<> struct ad_arith<double> { static const bool value = true; };
template<> struct ad_arith<int> { static const bool value = true; };
template<> struct ad_arith<unsigned> { static const bool value = true; };
template<> struct ad_arith<long> { static const bool value = true; };
template<> struct ad_arith<unsigned long> { static const bool value = true; };
template<> struct ad_arith<long long> { static const bool value = true; };
template<> struct ad_arith<unsigned long long> { static const bool value = true; };
template<> struct ad_arith<short> { static const bool value = true; };
template<bool B, class T = void> struct ad_if {};
template<class T> struct ad_if<true, T> { typedef T type; };
template<class T> T ad_val();

template<class T> struct AR {
  T v; int i;
  AR() = default;
  __device__ __forceinline__ AR(T x) : v(x), i(0) {}
  template<class S, class = typename ad_if<ad_arith<S>::value>::type>
  __device__ __forceinline__ AR(S x) : v((T)x), i(0) {}
  template<class U> __device__ __forceinline__ AR(const AR<U>& o) : v((T)o.v), i(o.i) {}
  __device__ __forceinline__ explicit operator int() const { return (int)ad_int((int)v); }
  __device__ __forceinline__ explicit operator long long() const { return (long long)ad_int((long long)v); }
  __device__ __forceinline__ explicit operator long() const { return (long)ad_int((long)v); }
  __device__ __forceinline__ explicit operator unsigned() const { return (unsigned)ad_int((unsigned)v); }
  __device__ __forceinline__ explicit operator bool() const { return ad_bit((bool)v); }
};
template<class T> struct ad_is_ar { static const bool value = false; };
template<class T> struct ad_is_ar<AR<T> > { static const bool value = true; };
template<class T> __device__ __forceinline__ AR<T> ad_mk(T v, int i) { AR<T> z; z.v = v; z.i = i; return z; }

// ---------------------------------------------------------------- arithmetic
#define AD_BIN(OP, PA, PB)                                                                         \
template<class A, class B> __device__ __forceinline__ AR<decltype(ad_val<A>() OP ad_val<B>())>    \
operator OP(const AR<A>& x, const AR<B>& y) {                                                      \
  const double xa = (double)x.v, yb = (double)y.v; (void)xa; (void)yb;                             \
  return ad_mk(x.v OP y.v, ad_node2(x.i, (PA), y.i, (PB))); }                                      \
template<class A, class S, class = typename ad_if<ad_arith<S>::value>::type>                       \
__device__ __forceinline__ AR<decltype(ad_val<A>() OP ad_val<S>())> operator OP(const AR<A>& x, S y) { \
  const double xa = (double)x.v, yb = (double)y; (void)xa; (void)yb;                               \
  return ad_mk(x.v OP y, ad_node1(x.i, (PA))); }                                                   \
template<class S, class B, class = typename ad_if<ad_arith<S>::value>::type>                       \
__device__ __forceinline__ AR<decltype(ad_val<S>() OP ad_val<B>())> operator OP(S x, const AR<B>& y) { \
  const double xa = (double)x, yb = (double)y.v; (void)xa; (void)yb;                               \
  return ad_mk(x OP y.v, ad_node1(y.i, (PB))); }
AD_BIN(+, 1.0, 1.0)
AD_BIN(-, 1.0, -1.0)
AD_BIN(*, yb, xa)
AD_BIN(/, 1.0 / yb, -xa / (yb * yb))
#undef AD_BIN

template<class A> __device__ __forceinline__ AR<A> operator-(const AR<A>& x) { return ad_mk((A)(-x.v), ad_node1(x.i, -1.0)); }
template<class A> __device__ __forceinline__ AR<A> operator+(const AR<A>& x) { return x; }

#define AD_CMP(OP)                                                                                 \
template<class A, class B> __device__ __forceinline__ bool operator OP(const AR<A>& x, const AR<B>& y) { \
  return ad_bit(x.v OP y.v); }                                                \
template<class A, class S, class = typename ad_if<ad_arith<S>::value>::type>                       \
__device__ __forceinline__ bool operator OP(const AR<A>& x, S y) { return ad_bit(x.v OP y); } \
template<class S, class B, class = typename ad_if<ad_arith<S>::value>::type>                       \
__device__ __forceinline__ bool operator OP(S x, const AR<B>& y) { return ad_bit(x OP y.v); }
AD_CMP(<) AD_CMP(>) AD_CMP(<=) AD_CMP(>=) AD_CMP(==) AD_CMP(!=)
#undef AD_CMP

// Compound assignment: the C++ rule (compute in the promoted type, convert back).
#define AD_CAS(OP)                                                                                 \
template<class A, class Y> __device__ __forceinline__ AR<A>& operator OP##=(AR<A>& x, const Y& y) { \
  x = AR<A>(x OP y); return x; }
AD_CAS(+) AD_CAS(-) AD_CAS(*) AD_CAS(/)
#undef AD_CAS

// ---------------------------------------------------------------- rounding intrinsics
#define AD_RN2(NAME, T, OP)                                                                        \
__device__ __forceinline__ AR<T> NAME(const AR<T>& x, const AR<T>& y) { return AR<T>(x OP y); }  \
__device__ __forceinline__ AR<T> NAME(const AR<T>& x, T y) { return AR<T>(x OP y); }              \
__device__ __forceinline__ AR<T> NAME(T x, const AR<T>& y) { return AR<T>(x OP y); }
AD_RN2(__fadd_rn, float, +) AD_RN2(__fsub_rn, float, -) AD_RN2(__fmul_rn, float, *) AD_RN2(__fdiv_rn, float, /)
AD_RN2(__dadd_rn, double, +) AD_RN2(__dsub_rn, double, -) AD_RN2(__dmul_rn, double, *) AD_RN2(__ddiv_rn, double, /)
#undef AD_RN2
template<class T> __device__ __forceinline__ AR<T> ad_fma3(T z, const AR<T>& a, const AR<T>& b, const AR<T>& c) {
  const int m = ad_node2(a.i, (double)b.v, b.i, (double)a.v);
  return ad_mk(z, ad_node2(m, 1.0, c.i, 1.0));
}
__device__ __forceinline__ AR<float> __fmaf_rn(const AR<float>& a, const AR<float>& b, const AR<float>& c) {
  return ad_fma3(::__fmaf_rn(a.v, b.v, c.v), a, b, c); }
__device__ __forceinline__ AR<double> __fma_rn(const AR<double>& a, const AR<double>& b, const AR<double>& c) {
  return ad_fma3(::__fma_rn(a.v, b.v, c.v), a, b, c); }

// ---------------------------------------------------------------- unary maths
// Each primal is the forward's own function on the primal operand; the
// partial is the analytic derivative evaluated in double.
#define AD_UN(NAME, T, PRIMAL, DERIV)                                                              \
__device__ __forceinline__ AR<T> NAME(const AR<T>& x) {                                            \
  const T z = ::PRIMAL(x.v); const double xd = (double)x.v, zd = (double)z; (void)xd; (void)zd;      \
  return ad_mk(z, x.i ? ad_node1(x.i, (DERIV)) : 0); }

__device__ __forceinline__ double ad_dsqrt(double x, double z) { return x > 0.0 ? 0.5 / z : 0.0; }
__device__ double ad_digamma(double x) {
  double r = 0.0;
  if (x <= 0.0 && x == ::floor(x)) return 0.0;
  if (x < 0.0) { r = -3.14159265358979323846 / ::tan(3.14159265358979323846 * x); x = 1.0 - x; }
  while (x < 6.0) { r -= 1.0 / x; x += 1.0; }
  const double f = 1.0 / (x * x);
  return r + ::log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f * (1.0 / 132)))));
}
AD_UN(sqrtf, float, sqrtf, ad_dsqrt(xd, ::sqrt(xd)))
AD_UN(sqrt, float, sqrtf, ad_dsqrt(xd, ::sqrt(xd)))
AD_UN(sqrt, double, sqrt, ad_dsqrt(xd, ::sqrt(xd)))
AD_UN(__fsqrt_rn, float, __fsqrt_rn, ad_dsqrt(xd, ::sqrt(xd)))
AD_UN(__dsqrt_rn, double, __dsqrt_rn, ad_dsqrt(xd, ::sqrt(xd)))
AD_UN(thompson_aa_expf, float, thompson_aa_expf, ::exp(xd))
AD_UN(thompson_aa_exp, double, thompson_aa_exp, ::exp(xd))
AD_UN(thompson_aa_logf, float, thompson_aa_logf, 1.0 / xd)
AD_UN(thompson_aa_log, double, thompson_aa_log, 1.0 / xd)
AD_UN(thompson_aa_log10f, float, thompson_aa_log10f, 1.0 / (xd * 2.302585092994045684))
AD_UN(thompson_aa_log10, double, thompson_aa_log10, 1.0 / (xd * 2.302585092994045684))
AD_UN(ubm_log10f, float, ubm_log10f, 1.0 / (xd * 2.302585092994045684))
AD_UN(gfk_exp, float, gfk_exp, ::exp(xd))
AD_UN(gfk_log, float, gfk_log, 1.0 / xd)
AD_UN(exp, double, exp, ::exp(xd))
AD_UN(expf, float, expf, ::exp(xd))
AD_UN(log, double, log, 1.0 / xd)
AD_UN(logf, float, logf, 1.0 / xd)
AD_UN(log10, double, log10, 1.0 / (xd * 2.302585092994045684))
AD_UN(sin, double, sin, ::cos(xd))
AD_UN(cos, double, cos, -::sin(xd))
AD_UN(glibc_cosf, float, glibc_cosf, -::sin(xd))
AD_UN(glibc_sinf, float, glibc_sinf, ::cos(xd))
AD_UN(atan, double, atan, 1.0 / (1.0 + xd * xd))
AD_UN(atanf, float, atanf, 1.0 / (1.0 + xd * xd))
AD_UN(tgammaf, float, tgammaf, ::tgamma(xd) * ad_digamma(xd))
AD_UN(tgamma, double, tgamma, ::tgamma(xd) * ad_digamma(xd))
AD_UN(gfk_tgamma, float, gfk_tgamma, ::tgamma(xd) * ad_digamma(xd))
AD_UN(fabs, double, fabs, ::signbit(xd) ? -1.0 : 1.0)
AD_UN(fabs, float, fabsf, ::signbit(xd) ? -1.0 : 1.0)
AD_UN(fabsf, float, fabsf, ::signbit(xd) ? -1.0 : 1.0)
AD_UN(floor, double, floor, 0.0)
AD_UN(floorf, float, floorf, 0.0)
AD_UN(floor, float, floorf, 0.0)
AD_UN(ceil, double, ceil, 0.0)
AD_UN(ceilf, float, ceilf, 0.0)
#undef AD_UN

__device__ __forceinline__ long lroundf(const AR<float>& x) { return (long)ad_int(::lroundf(x.v)); }
__device__ __forceinline__ long lround(const AR<double>& x) { return (long)ad_int(::lround(x.v)); }
__device__ __forceinline__ bool isnan(const AR<float>& x) { return ad_bit(::isnan(x.v)); }
__device__ __forceinline__ bool isnan(const AR<double>& x) { return ad_bit(::isnan(x.v)); }
__device__ __forceinline__ bool isinf(const AR<float>& x) { return ad_bit(::isinf(x.v)); }
__device__ __forceinline__ bool isinf(const AR<double>& x) { return ad_bit(::isinf(x.v)); }
__device__ __forceinline__ bool signbit(const AR<float>& x) { return ad_bit(::signbit(x.v)); }
__device__ __forceinline__ bool signbit(const AR<double>& x) { return ad_bit(::signbit(x.v)); }

// copysign(a, b) = |a| with the sign of b: d/da = sign(a) sign(b), d/db = 0.
template<class T> __device__ __forceinline__ AR<T> ad_copysign(const AR<T>& a, const AR<T>& b) {
  const T z = ::copysign(a.v, b.v);
  const double s = (::signbit(a.v) != ::signbit(b.v)) ? -1.0 : 1.0;
  return ad_mk(z, ad_node1(a.i, s));
}
__device__ __forceinline__ AR<double> copysign(const AR<double>& a, const AR<double>& b) { return ad_copysign(a, b); }
__device__ __forceinline__ AR<float> copysign(const AR<float>& a, const AR<float>& b) { return ad_copysign(a, b); }
__device__ __forceinline__ AR<float> copysignf(const AR<float>& a, const AR<float>& b) { return ad_copysign(a, b); }

// ---------------------------------------------------------------- binary maths
// pow: d/dx = y x^(y-1), d/dy = x^y log x (x > 0; 0 otherwise, the
// executed arm of the forward's own special cases).
__device__ __forceinline__ void ad_pow_partials(double x, double y, double z, double* px, double* py) {
  *px = (y == 0.0) ? 0.0 : (x != 0.0 ? y * ::pow(x, y - 1.0) : (y == 1.0 ? 1.0 : 0.0));
  *py = (x > 0.0) ? z * ::log(x) : 0.0;
}
template<class T> __device__ __forceinline__ AR<T> ad_pow_out(T z, const AR<T>& x, const AR<T>& y) {
  double px, py; ad_pow_partials((double)x.v, (double)y.v, (double)z, &px, &py);
  return ad_mk(z, ad_node2(x.i, px, y.i, py));
}
#define AD_POW(NAME, T)                                                                            \
__device__ __forceinline__ AR<T> NAME(const AR<T>& x, const AR<T>& y) { return ad_pow_out(::NAME(x.v, y.v), x, y); } \
__device__ __forceinline__ AR<T> NAME(const AR<T>& x, T y) { return ad_pow_out(::NAME(x.v, y), x, AR<T>(y)); } \
__device__ __forceinline__ AR<T> NAME(T x, const AR<T>& y) { return ad_pow_out(::NAME(x, y.v), AR<T>(x), y); }
AD_POW(thompson_aa_powf, float)
AD_POW(thompson_aa_pow, double)
AD_POW(gfk_pow, float)
AD_POW(pow, double)
AD_POW(powf, float)
#undef AD_POW

template<class T> __device__ __forceinline__ AR<T> ad_hypot_out(T z, const AR<T>& x, const AR<T>& y) {
  const double h = (double)z;
  const double px = h > 0.0 ? (double)x.v / h : 0.0, py = h > 0.0 ? (double)y.v / h : 0.0;
  return ad_mk(z, ad_node2(x.i, px, y.i, py));
}
__device__ __forceinline__ AR<double> thompson_aa_hypot(const AR<double>& x, const AR<double>& y) {
  return ad_hypot_out(::thompson_aa_hypot(x.v, y.v), x, y); }
template<class T> __device__ __forceinline__ AR<T> ad_atan2_out(T z, const AR<T>& y, const AR<T>& x) {
  const double xd = (double)x.v, yd = (double)y.v, r = xd * xd + yd * yd;
  return ad_mk(z, ad_node2(y.i, r > 0.0 ? xd / r : 0.0, x.i, r > 0.0 ? -yd / r : 0.0));
}
__device__ __forceinline__ AR<double> atan2(const AR<double>& y, const AR<double>& x) { return ad_atan2_out(::atan2(y.v, x.v), y, x); }
__device__ __forceinline__ AR<double> thompson_aa_atan2(const AR<double>& y, const AR<double>& x) {
  return ad_atan2_out(::thompson_aa_atan2(y.v, x.v), y, x); }

// fmin/fmax: the executed arm (the forward's own selection).
#define AD_MM(NAME, T, PRIMAL, CMP)                                                                \
__device__ __forceinline__ AR<T> NAME(const AR<T>& x, const AR<T>& y) {                            \
  T z = ::PRIMAL(x.v, y.v); const bool r0 = (x.v CMP y.v) || ::isnan(y.v); const bool r = ad_bit(r0); \
  if (r != r0) z = r ? x.v : y.v;                                                                  \
  return ad_mk(z, r ? x.i : y.i); }
AD_MM(fminf, float, fminf, <=) AD_MM(fmaxf, float, fmaxf, >=)
AD_MM(fmin, double, fmin, <=) AD_MM(fmax, double, fmax, >=)
#undef AD_MM

#ifdef AD_PROMOTE
// Float64-promoted grade only (never a production entry): every float of the
// column code is AR<double>, and the float-only maths it calls resolves to the
// binary64 function of the same family, with the same partials.
#define AD_PU(NAME, PRIMAL, DERIV)                                                                 \
__device__ __forceinline__ AR<double> NAME(const AR<double>& x) {                                  \
  const double z = ::PRIMAL(x.v); const double xd = x.v, zd = z; (void)xd; (void)zd;               \
  return ad_mk(z, x.i ? ad_node1(x.i, (DERIV)) : 0); }
AD_PU(thompson_aa_expf, thompson_aa_exp, ::exp(xd))
AD_PU(thompson_aa_logf, thompson_aa_log, 1.0 / xd)
AD_PU(thompson_aa_log10f, thompson_aa_log10, 1.0 / (xd * 2.302585092994045684))
AD_PU(ubm_log10f, thompson_aa_log10, 1.0 / (xd * 2.302585092994045684))
AD_PU(gfk_exp, exp, ::exp(xd))
AD_PU(gfk_log, log, 1.0 / xd)
AD_PU(expf, exp, ::exp(xd))
AD_PU(logf, log, 1.0 / xd)
AD_PU(sqrtf, sqrt, ad_dsqrt(xd, ::sqrt(xd)))
AD_PU(__fsqrt_rn, sqrt, ad_dsqrt(xd, ::sqrt(xd)))
AD_PU(glibc_cosf, cos, -::sin(xd))
AD_PU(glibc_sinf, sin, ::cos(xd))
AD_PU(atanf, atan, 1.0 / (1.0 + xd * xd))
AD_PU(tgammaf, tgamma, ::tgamma(xd) * ad_digamma(xd))
AD_PU(gfk_tgamma, tgamma, ::tgamma(xd) * ad_digamma(xd))
AD_PU(fabsf, fabs, ::signbit(xd) ? -1.0 : 1.0)
AD_PU(floorf, floor, 0.0)
AD_PU(ceilf, ceil, 0.0)
#undef AD_PU
__device__ __forceinline__ long lroundf(const AR<double>& x) { return (long)ad_int(::lround(x.v)); }
#define AD_PRN(NAME, OP)                                                                           \
__device__ __forceinline__ AR<double> NAME(const AR<double>& x, const AR<double>& y) { return x OP y; } \
__device__ __forceinline__ AR<double> NAME(const AR<double>& x, double y) { return x OP y; }      \
__device__ __forceinline__ AR<double> NAME(double x, const AR<double>& y) { return x OP y; }
AD_PRN(__fadd_rn, +) AD_PRN(__fsub_rn, -) AD_PRN(__fmul_rn, *) AD_PRN(__fdiv_rn, /)
#undef AD_PRN
__device__ __forceinline__ AR<double> __fmaf_rn(const AR<double>& a, const AR<double>& b, const AR<double>& c) {
  return ad_fma3(::__fma_rn(a.v, b.v, c.v), a, b, c); }
#define AD_PPOW(NAME, PRIMAL)                                                                      \
__device__ __forceinline__ AR<double> NAME(const AR<double>& x, const AR<double>& y) { return ad_pow_out(::PRIMAL(x.v, y.v), x, y); } \
__device__ __forceinline__ AR<double> NAME(const AR<double>& x, double y) { return ad_pow_out(::PRIMAL(x.v, y), x, AR<double>(y)); } \
__device__ __forceinline__ AR<double> NAME(double x, const AR<double>& y) { return ad_pow_out(::PRIMAL(x, y.v), AR<double>(x), y); }
AD_PPOW(thompson_aa_powf, thompson_aa_pow) AD_PPOW(gfk_pow, pow) AD_PPOW(powf, pow)
#undef AD_PPOW
__device__ __forceinline__ AR<double> copysignf(const AR<double>& a, const AR<double>& b) { return ad_copysign(a, b); }
__device__ __forceinline__ AR<double> fminf(const AR<double>& x, const AR<double>& y) { return fmin(x, y); }
__device__ __forceinline__ AR<double> fmaxf(const AR<double>& x, const AR<double>& y) { return fmax(x, y); }
#endif
} // namespace adr
using adr::AR;
using adr::ad_mk;
