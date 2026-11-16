// SPDX-License-Identifier: Apache-2.0
// tsad.cuh: value types for the type-substitution derivative generator (P2, private autodiff program).
//
// The generator (tsgen) copies a captured physics source verbatim, then re-emits the selected
// functions with `float` -> TF and `double` -> TD.  TF/TD are the types below:
//   VF<K> carries a float primal, VD<K> a double primal, plus a derivative of kind K:
//     K = Tan : forward derivative (tangent), a double per value;
//     K = Tap : reverse derivative, the index of a node on this thread's linear tape.
// The primal of every operation is the identical float/double operation on the same operands
// (C++ usual arithmetic conversions are reproduced exactly), so with the forward's own compile
// flags the primal is the forward, bit for bit.  The derivative is the linearisation of the
// branch the primal took: comparisons read primal values only.
//
// Conversions: VD(VF) and VF(VD) are implicit (as float <-> double are in C++); VF(double)
// from a plain double is explicit.  A float/double mix the original resolved to double therefore
// cannot silently become float here: it fails to compile instead.  There is no conversion to
// float/double at all, so an AD value can never drop its derivative unseen.
#pragma once

namespace tsad {

template<bool B, class T = void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };
template<bool B, class T, class F> struct conditional { typedef T type; };
template<class T, class F> struct conditional<false, T, F> { typedef F type; };
template<class A, class B> struct is_same { static constexpr bool value = false; };
template<class A> struct is_same<A, A> { static constexpr bool value = true; };

// ---------------------------------------------------------------------------------------------
// Tape arena.  One slice per thread of the launch (slot = global thread index).  The host sets
// g_tsad_arena before each tile; nodes == nullptr is the count-only sizing pass.
// ---------------------------------------------------------------------------------------------
struct Node { int a, b; double wa, wb; };
struct Arena {
  Node* nodes;      // [slots * capacity] or nullptr (count only)
  double* bars;     // [slots * capacity]
  int* used;        // [slots]
  int* overflow;    // [slots]
  int capacity;
};
__device__ Arena g_tsad_arena;

__device__ __forceinline__ int slot_id() { return (int)(blockIdx.x * blockDim.x + threadIdx.x); }

__device__ __noinline__ int push_node(int a, double wa, int b, double wb) {
  Arena& T = g_tsad_arena;
  const int s = slot_id();
  const int id = T.used[s];
  if (id >= T.capacity && T.nodes) { T.overflow[s] = 1; return -1; }
  T.used[s] = id + 1;
  if (T.nodes) {
    const long long o = (long long)s * T.capacity + id;
    Node n; n.a = a; n.b = b; n.wa = wa; n.wb = wb;
    T.nodes[o] = n; T.bars[o] = 0.0;
  }
  return id;
}
__device__ __forceinline__ int leaf_node() { return push_node(-1, 0.0, -1, 0.0); }

// Reverse sweep of this thread's slice: bars of every node it reaches.
__device__ __noinline__ void reverse_slice() {
  Arena& T = g_tsad_arena;
  const int s = slot_id();
  const long long base = (long long)s * T.capacity;
  for (int i = T.used[s] - 1; i >= 0; --i) {
    const double b = T.bars[base + i];
    if (b == 0.0) continue;
    const Node n = T.nodes[base + i];
    if (n.a >= 0) T.bars[base + n.a] += n.wa * b;
    if (n.b >= 0) T.bars[base + n.b] += n.wb * b;
  }
}
__device__ __forceinline__ void seed_bar(int id, double b) {
  if (id < 0 || b == 0.0) return;
  Arena& T = g_tsad_arena;
  T.bars[(long long)slot_id() * T.capacity + id] += b;
}
__device__ __forceinline__ double read_bar(int id) {
  if (id < 0) return 0.0;
  Arena& T = g_tsad_arena;
  return T.bars[(long long)slot_id() * T.capacity + id];
}

// ---------------------------------------------------------------------------------------------
// Derivative kinds.  A term whose derivative is exactly zero (tangent) or absent (tape) is
// skipped, so a zero tangent times an infinite local weight never manufactures a NaN, and the
// tape and tangent skip the same terms.
// ---------------------------------------------------------------------------------------------
struct Tan {
  typedef double D;
  static __device__ __forceinline__ D zero() { return 0.0; }
  static __device__ __forceinline__ bool on(D a) { return a != 0.0; }
  static __device__ __forceinline__ D lin1(D a, double wa) { return a != 0.0 ? wa * a : 0.0; }
  static __device__ __forceinline__ D lin2(D a, double wa, D b, double wb) {
    double r = 0.0;
    if (a != 0.0) r += wa * a;
    if (b != 0.0) r += wb * b;
    return r;
  }
  static __device__ __forceinline__ D same(D a) { return a; }
  static __device__ __forceinline__ D neg(D a) { return -a; }
};
struct Tap {
  typedef int D;
  static __device__ __forceinline__ D zero() { return -1; }
  static __device__ __forceinline__ bool on(D a) { return a >= 0; }
  static __device__ __forceinline__ D lin1(D a, double wa) { return a >= 0 ? push_node(a, wa, -1, 0.0) : -1; }
  static __device__ __forceinline__ D lin2(D a, double wa, D b, double wb) {
    if (a < 0 && b < 0) return -1;
    if (a < 0) return push_node(b, wb, -1, 0.0);
    if (b < 0) return push_node(a, wa, -1, 0.0);
    return push_node(a, wa, b, wb);
  }
  static __device__ __forceinline__ D same(D a) { return a; }
  static __device__ __forceinline__ D neg(D a) { return a >= 0 ? push_node(a, -1.0, -1, 0.0) : -1; }
};

template<class K> struct VD;

// Scalar traits: rank 0 integer, 1 float, 2 double.
template<class T> struct st { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = false; };
template<> struct st<bool> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<char> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<signed char> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<unsigned char> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<short> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<unsigned short> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<int> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<unsigned> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<long> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<unsigned long> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<long long> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<unsigned long long> { static constexpr bool ad = false; static constexpr int rank = 0; static constexpr bool ok = true; };
template<> struct st<float> { static constexpr bool ad = false; static constexpr int rank = 1; static constexpr bool ok = true; };
template<> struct st<double> { static constexpr bool ad = false; static constexpr int rank = 2; static constexpr bool ok = true; };

// ---------------------------------------------------------------------------------------------
// VF: float primal.
// ---------------------------------------------------------------------------------------------
template<class K> struct VF {
  typedef K kind; typedef float P;
  float v; typename K::D d;
  __device__ __forceinline__ VF() : v(0.0f), d(K::zero()) {}
  __device__ __forceinline__ VF(float x) : v(x), d(K::zero()) {}
  // integers convert implicitly, exactly as `float f = i;`
  template<class T, class = typename enable_if<st<T>::ok && st<T>::rank == 0>::type>
  __device__ __forceinline__ VF(T x) : v((float)x), d(K::zero()) {}
  // double -> float narrowing is explicit only (see the header note)
  __device__ __forceinline__ explicit VF(double x) : v((float)x), d(K::zero()) {}
  // VD -> VF narrows implicitly, as `float f = d;` does; a ternary mixing VF and VD is then
  // ambiguous and fails to compile (visible), never silently resolved.
  __device__ __forceinline__ VF(const VD<K>& q);
  __device__ __forceinline__ VF(float x, typename K::D dd, int) : v(x), d(dd) {}
  __device__ __forceinline__ VF& operator=(const VD<K>& q);
  __device__ __forceinline__ VF& operator=(double x) { v = (float)x; d = K::zero(); return *this; }
  __device__ __forceinline__ explicit operator bool() const { return v != 0.0f; }
  __device__ __forceinline__ explicit operator int() const { return (int)v; }
  __device__ __forceinline__ explicit operator unsigned() const { return (unsigned)v; }
  __device__ __forceinline__ explicit operator long() const { return (long)v; }
  __device__ __forceinline__ explicit operator long long() const { return (long long)v; }
};
template<class K> struct VD {
  typedef K kind; typedef double P;
  double v; typename K::D d;
  __device__ __forceinline__ VD() : v(0.0), d(K::zero()) {}
  template<class T, class = typename enable_if<st<T>::ok>::type>
  __device__ __forceinline__ VD(T x) : v((double)x), d(K::zero()) {}
  __device__ __forceinline__ VD(const VF<K>& q) : v((double)q.v), d(K::same(q.d)) {}
  __device__ __forceinline__ VD(double x, typename K::D dd, int) : v(x), d(dd) {}
  __device__ __forceinline__ explicit operator bool() const { return v != 0.0; }
  __device__ __forceinline__ explicit operator int() const { return (int)v; }
  __device__ __forceinline__ explicit operator unsigned() const { return (unsigned)v; }
  __device__ __forceinline__ explicit operator long() const { return (long)v; }
  __device__ __forceinline__ explicit operator long long() const { return (long long)v; }
};
// Rounding a double to float has derivative 1: the node (or tangent) is passed through.
template<class K> __device__ __forceinline__ VF<K>::VF(const VD<K>& q) : v((float)q.v), d(K::same(q.d)) {}
template<class K> __device__ __forceinline__ VF<K>& VF<K>::operator=(const VD<K>& q) { v = (float)q.v; d = K::same(q.d); return *this; }

template<class K> struct st<VF<K> > { static constexpr bool ad = true; static constexpr int rank = 1; static constexpr bool ok = true; typedef K kind; };
template<class K> struct st<VD<K> > { static constexpr bool ad = true; static constexpr int rank = 2; static constexpr bool ok = true; typedef K kind; };

template<class A, class B, bool AA = st<A>::ad> struct kind_of { typedef typename st<A>::kind type; };
template<class A, class B> struct kind_of<A, B, false> { typedef typename st<B>::kind type; };

// Result of a binary arithmetic op, by C++ usual arithmetic conversions.
template<class A, class B> struct bin {
  typedef typename kind_of<A, B>::type K;
  static constexpr int r = st<A>::rank > st<B>::rank ? st<A>::rank : st<B>::rank;
  typedef typename conditional<(r == 2), VD<K>, VF<K> >::type R;
  typedef typename conditional<(r == 2), double, float>::type P;
};
template<class A, class B> struct either_ad {
  static constexpr bool value = (st<A>::ad || st<B>::ad) && st<A>::ok && st<B>::ok;
};

template<class T> __device__ __forceinline__ T val(T x) { return x; }
template<class K> __device__ __forceinline__ float val(const VF<K>& x) { return x.v; }
template<class K> __device__ __forceinline__ double val(const VD<K>& x) { return x.v; }
template<class K, class T> struct der_t { static __device__ __forceinline__ typename K::D get(const T&) { return K::zero(); } };
template<class K> struct der_t<K, VF<K> > { static __device__ __forceinline__ typename K::D get(const VF<K>& x) { return x.d; } };
template<class K> struct der_t<K, VD<K> > { static __device__ __forceinline__ typename K::D get(const VD<K>& x) { return x.d; } };
template<class K, class T> __device__ __forceinline__ typename K::D der(const T& x) { return der_t<K, T>::get(x); }

#define TSAD_ENABLE(A, B) class = typename enable_if<either_ad<A, B>::value>::type

template<class A, class B, TSAD_ENABLE(A, B)>
__device__ __forceinline__ typename bin<A, B>::R operator+(const A& a, const B& b) {
  typedef bin<A, B> T; typedef typename T::K K; typedef typename T::P P;
  const P x = (P)val(a), y = (P)val(b);
  return typename T::R(x + y, K::lin2(der<K>(a), 1.0, der<K>(b), 1.0), 0);
}
template<class A, class B, TSAD_ENABLE(A, B)>
__device__ __forceinline__ typename bin<A, B>::R operator-(const A& a, const B& b) {
  typedef bin<A, B> T; typedef typename T::K K; typedef typename T::P P;
  const P x = (P)val(a), y = (P)val(b);
  return typename T::R(x - y, K::lin2(der<K>(a), 1.0, der<K>(b), -1.0), 0);
}
template<class A, class B, TSAD_ENABLE(A, B)>
__device__ __forceinline__ typename bin<A, B>::R operator*(const A& a, const B& b) {
  typedef bin<A, B> T; typedef typename T::K K; typedef typename T::P P;
  const P x = (P)val(a), y = (P)val(b);
  return typename T::R(x * y, K::lin2(der<K>(a), (double)y, der<K>(b), (double)x), 0);
}
template<class A, class B, TSAD_ENABLE(A, B)>
__device__ __forceinline__ typename bin<A, B>::R operator/(const A& a, const B& b) {
  typedef bin<A, B> T; typedef typename T::K K; typedef typename T::P P;
  const P x = (P)val(a), y = (P)val(b);
  const P q = x / y;
#ifdef TSAD_HAND_PARTIALS  // diagnostic only: the 11-02 hand pairs' partials, rounded in the operand type
  const double wa = (double)((P)1 / y), wb = (double)(-q / y);
#else
  const double yd = (double)y, wa = 1.0 / yd, wb = -((double)x / yd) / yd;
#endif
  return typename T::R(q, K::lin2(der<K>(a), wa, der<K>(b), wb), 0);
}
template<class K> __device__ __forceinline__ VF<K> operator-(const VF<K>& a) { return VF<K>(-a.v, K::neg(a.d), 0); }
template<class K> __device__ __forceinline__ VD<K> operator-(const VD<K>& a) { return VD<K>(-a.v, K::neg(a.d), 0); }
template<class K> __device__ __forceinline__ VF<K> operator+(const VF<K>& a) { return a; }
template<class K> __device__ __forceinline__ VD<K> operator+(const VD<K>& a) { return a; }

#define TSAD_CMP(OP) \
template<class A, class B, TSAD_ENABLE(A, B)> \
__device__ __forceinline__ bool operator OP(const A& a, const B& b) { return val(a) OP val(b); }
TSAD_CMP(<) TSAD_CMP(>) TSAD_CMP(<=) TSAD_CMP(>=) TSAD_CMP(==) TSAD_CMP(!=)
#undef TSAD_CMP

// Compound assignment: `float x; x op= e;` is x = (float)(x op e) in the promoted type.
#define TSAD_COMPOUND(OP, BOP) \
template<class K, class B, class = typename enable_if<st<B>::ok>::type> \
__device__ __forceinline__ VF<K>& operator OP(VF<K>& a, const B& b) { a = a BOP b; return a; } \
template<class K, class B, class = typename enable_if<st<B>::ok>::type> \
__device__ __forceinline__ VD<K>& operator OP(VD<K>& a, const B& b) { a = a BOP b; return a; }
TSAD_COMPOUND(+=, +) TSAD_COMPOUND(-=, -) TSAD_COMPOUND(*=, *) TSAD_COMPOUND(/=, /)
#undef TSAD_COMPOUND

// ---------------------------------------------------------------------------------------------
// Rounding intrinsics (TEMPO and the shared maths use them for pinned operation order).
// ---------------------------------------------------------------------------------------------
#define TSAD_RN(NAME, PT, VT, OPW) \
template<class K> __device__ __forceinline__ VT<K> NAME(const VT<K>& a, const VT<K>& b) { \
  const PT x = a.v, y = b.v; OPW } \
template<class K> __device__ __forceinline__ VT<K> NAME(const VT<K>& a, PT b) { return NAME(a, VT<K>(b)); } \
template<class K> __device__ __forceinline__ VT<K> NAME(PT a, const VT<K>& b) { return NAME(VT<K>(a), b); }
TSAD_RN(__fadd_rn, float, VF, return VF<K>(::__fadd_rn(x, y), K::lin2(a.d, 1.0, b.d, 1.0), 0);)
TSAD_RN(__fsub_rn, float, VF, return VF<K>(::__fsub_rn(x, y), K::lin2(a.d, 1.0, b.d, -1.0), 0);)
TSAD_RN(__fmul_rn, float, VF, return VF<K>(::__fmul_rn(x, y), K::lin2(a.d, (double)y, b.d, (double)x), 0);)
TSAD_RN(__fdiv_rn, float, VF, return VF<K>(::__fdiv_rn(x, y), K::lin2(a.d, 1.0 / (double)y, b.d, -((double)x / (double)y) / (double)y), 0);)
TSAD_RN(__dadd_rn, double, VD, return VD<K>(::__dadd_rn(x, y), K::lin2(a.d, 1.0, b.d, 1.0), 0);)
TSAD_RN(__dsub_rn, double, VD, return VD<K>(::__dsub_rn(x, y), K::lin2(a.d, 1.0, b.d, -1.0), 0);)
TSAD_RN(__dmul_rn, double, VD, return VD<K>(::__dmul_rn(x, y), K::lin2(a.d, y, b.d, x), 0);)
TSAD_RN(__ddiv_rn, double, VD, return VD<K>(::__ddiv_rn(x, y), K::lin2(a.d, 1.0 / y, b.d, -(x / y) / y), 0);)
#undef TSAD_RN
template<class K> __device__ __forceinline__ VF<K> __fmaf_rn(const VF<K>& a, const VF<K>& b, const VF<K>& c) {
  const VF<K> ab(0.0f, K::lin2(a.d, (double)b.v, b.d, (double)a.v), 0);
  return VF<K>(::__fmaf_rn(a.v, b.v, c.v), K::lin2(ab.d, 1.0, c.d, 1.0), 0);
}
template<class K> __device__ __forceinline__ VD<K> __fma_rn(const VD<K>& a, const VD<K>& b, const VD<K>& c) {
  const VD<K> ab(0.0, K::lin2(a.d, b.v, b.d, a.v), 0);
  return VD<K>(::__fma_rn(a.v, b.v, c.v), K::lin2(ab.d, 1.0, c.d, 1.0), 0);
}
template<class K> __device__ __forceinline__ float __double2float_rn(const VD<K>&);  // deliberately undefined: use VF(VD)

// ---------------------------------------------------------------------------------------------
// Generic one-argument rule: primal is the source's own function on the primal; the local
// derivative is written in double from x and the primal result r.
// ---------------------------------------------------------------------------------------------
#define TSAD_UNARY_F(NAME, CALL, DEXPR) \
template<class K> __device__ __forceinline__ VF<K> NAME(const VF<K>& a) { \
  const float rf = CALL(a.v); \
  if (!K::on(a.d)) return VF<K>(rf); \
  const double x = (double)a.v, r = (double)rf; (void)x; (void)r; \
  return VF<K>(rf, K::lin1(a.d, (DEXPR)), 0); }
#define TSAD_UNARY_D(NAME, CALL, DEXPR) \
template<class K> __device__ __forceinline__ VD<K> NAME(const VD<K>& a) { \
  const double rf = CALL(a.v); \
  if (!K::on(a.d)) return VD<K>(rf); \
  const double x = a.v, r = rf; (void)x; (void)r; \
  return VD<K>(rf, K::lin1(a.d, (DEXPR)), 0); }

// Standard library forms the TEMPO/MYNN sources call directly.
TSAD_UNARY_F(sqrt, ::sqrtf, (x > 0.0 ? 0.5 / ::sqrt(x) : 0.0))
TSAD_UNARY_F(sqrtf, ::sqrtf, (x > 0.0 ? 0.5 / ::sqrt(x) : 0.0))
TSAD_UNARY_D(sqrt, ::sqrt, (r > 0.0 ? 0.5 / r : 0.0))
TSAD_UNARY_F(fabs, ::fabsf, (::signbit(x) ? -1.0 : 1.0))
TSAD_UNARY_F(fabsf, ::fabsf, (::signbit(x) ? -1.0 : 1.0))
TSAD_UNARY_D(fabs, ::fabs, (::signbit(x) ? -1.0 : 1.0))
TSAD_UNARY_F(abs, ::fabsf, (::signbit(x) ? -1.0 : 1.0))
TSAD_UNARY_D(abs, ::fabs, (::signbit(x) ? -1.0 : 1.0))
// Integer-valued: zero derivative, primal unchanged.
TSAD_UNARY_F(floorf, ::floorf, 0.0) TSAD_UNARY_F(floor, ::floorf, 0.0) TSAD_UNARY_D(floor, ::floor, 0.0)
TSAD_UNARY_F(ceilf, ::ceilf, 0.0) TSAD_UNARY_F(ceil, ::ceilf, 0.0) TSAD_UNARY_D(ceil, ::ceil, 0.0)
TSAD_UNARY_F(truncf, ::truncf, 0.0) TSAD_UNARY_F(trunc, ::truncf, 0.0) TSAD_UNARY_D(trunc, ::trunc, 0.0)
TSAD_UNARY_F(rintf, ::rintf, 0.0) TSAD_UNARY_F(roundf, ::roundf, 0.0) TSAD_UNARY_D(rint, ::rint, 0.0)
template<class K> __device__ __forceinline__ long lroundf(const VF<K>& a) { return ::lroundf(a.v); }
template<class K> __device__ __forceinline__ long lround(const VD<K>& a) { return ::lround(a.v); }
template<class K> __device__ __forceinline__ int __float2int_rn(const VF<K>& a) { return ::__float2int_rn(a.v); }
template<class K> __device__ __forceinline__ int __float2int_rz(const VF<K>& a) { return ::__float2int_rz(a.v); }
template<class K> __device__ __forceinline__ int __double2int_rn(const VD<K>& a) { return ::__double2int_rn(a.v); }
template<class K> __device__ __forceinline__ bool isfinite(const VF<K>& a) { return ::isfinite(a.v); }
template<class K> __device__ __forceinline__ bool isfinite(const VD<K>& a) { return ::isfinite(a.v); }
template<class K> __device__ __forceinline__ bool isnan(const VF<K>& a) { return ::isnan(a.v); }
template<class K> __device__ __forceinline__ bool isnan(const VD<K>& a) { return ::isnan(a.v); }
template<class K> __device__ __forceinline__ bool signbit(const VF<K>& a) { return ::signbit(a.v); }
template<class K> __device__ __forceinline__ bool signbit(const VD<K>& a) { return ::signbit(a.v); }

// copysign(a, b): derivative of a times the executed sign change.
template<class K> __device__ __forceinline__ VF<K> copysignf(const VF<K>& a, const VF<K>& b) {
  const float r = ::copysignf(a.v, b.v);
  return VF<K>(r, (::signbit(r) == ::signbit(a.v)) ? K::same(a.d) : K::neg(a.d), 0);
}

// fmax/fmin: the executed winner (ties and NaN follow the library's own choice: compare bits).
#define TSAD_MINMAX(NAME, CALL, PT, VT) \
template<class K> __device__ __forceinline__ VT<K> NAME(const VT<K>& a, const VT<K>& b) { \
  const PT r = CALL(a.v, b.v); \
  const bool left = (r == a.v) && !(r == b.v && !(r == a.v)); \
  return VT<K>(r, left ? K::same(a.d) : K::same(b.d), 0); } \
template<class K> __device__ __forceinline__ VT<K> NAME(const VT<K>& a, PT b) { return NAME(a, VT<K>(b)); } \
template<class K> __device__ __forceinline__ VT<K> NAME(PT a, const VT<K>& b) { return NAME(VT<K>(a), b); }
TSAD_MINMAX(fmaxf, ::fmaxf, float, VF) TSAD_MINMAX(fminf, ::fminf, float, VF)
TSAD_MINMAX(fmax, ::fmax, double, VD) TSAD_MINMAX(fmin, ::fmin, double, VD)
#undef TSAD_MINMAX

}  // namespace tsad
