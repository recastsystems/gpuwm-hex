// P3 copy of the 11-02 ad2-mynn-typesub-device rules (oracle-checked there); every primal comparison and
// sign choice is now recorded by mw_rec (branch record for flip counts). Arithmetic rules unchanged.
// Included twice by adf.h: (ADF, ADD, DT) = (AdF, AdD, double) for the tangent and (AdT, AdTD, Lin) for the
// tape. Primal: the captured float/double arithmetic of the generated source (bitwise by construction).
// Derivative: linear rule from executed values, in DT (double tangent, or a recorded linear tape node).
struct ADD;
// The float carrier holds only its value and derivative. (Until 2026-11-04 it also held a write-tracking pointer
// that nothing ever set; operator= dereferenced it, and arena elements are reused raw memory, so the pointer read
// back stale bytes of earlier arrays: one-byte writes to stray addresses, and the illegal address in
// f_topdown_cloudrad's arrays on the downdraft/top-down captures.)
struct ADF {
    float v; DT d;
    __device__ ADF() : v(0.f), d(0.0) {}
    __device__ ADF(float x) : v(x), d(0.0) {}
    __device__ ADF(double x) : v((float)x), d(0.0) {}
    __device__ ADF(int x) : v((float)x), d(0.0) {}
    __device__ ADF(float x, DT dx) : v(x), d(dx) {}
    ADF(const ADF&) = default;
    ADF& operator=(const ADF&) = default;
    __device__ ADF& operator=(const ADD& q);
    __device__ ADF& operator=(float x) { v = x; d = DT(0.0); return *this; }
    __device__ ADF& operator=(double x) { v = (float)x; d = DT(0.0); return *this; }
    __device__ ADF& operator=(int x) { v = (float)x; d = DT(0.0); return *this; }
};
__device__ static __noinline__ ADF r6_add(ADF a, ADF b) { return ADF(r6_add(a.v, b.v), a.d + b.d); }
__device__ static __noinline__ ADF r6_sub(ADF a, ADF b) { return ADF(r6_sub(a.v, b.v), a.d - b.d); }
__device__ static __noinline__ ADF r6_mul(ADF a, ADF b) { return ADF(r6_mul(a.v, b.v), mynn_ad_multiply(a.d,(double)a.v,b.d,(double)b.v)); }
__device__ static __noinline__ ADF r6_div(ADF a, ADF b) { float q = r6_div(a.v, b.v); if (b.v == 0.0f) { mw_domain(1, a.d); mw_domain(1, b.d); } return ADF(q, mw_quot(a.d, b.d, ((double)a.v / (double)b.v), (double)b.v)); }
__device__ static __noinline__ ADF r6_neg(ADF a) { return ADF(r6_neg(a.v), -a.d); }
#define ADF_CMP(OP) \
__device__ static inline bool operator OP(ADF a, ADF b) { return mw_rec(a.v OP b.v); } \
__device__ static inline bool operator OP(ADF a, float b) { return mw_rec(a.v OP b); } \
__device__ static inline bool operator OP(float a, ADF b) { return mw_rec(a OP b.v); } \
__device__ static inline bool operator OP(ADF a, double b) { return mw_rec(a.v OP b); } \
__device__ static inline bool operator OP(double a, ADF b) { return mw_rec(a OP b.v); } \
__device__ static inline bool operator OP(ADF a, int b) { return mw_rec(a.v OP b); } \
__device__ static inline bool operator OP(int a, ADF b) { return mw_rec(a OP b.v); }
ADF_CMP(<) ADF_CMP(>) ADF_CMP(<=) ADF_CMP(>=) ADF_CMP(==) ADF_CMP(!=)
#undef ADF_CMP
__device__ static inline ADF mynn_sqrt(ADF a) { float s = mynn_sqrt(a.v); if (a.v == 0.0f && !a.d.depends) return ADF(s, mynn_ad_zero(a.d,a.d)); if (a.v <= 0.0f) mw_domain(2, a.d); return ADF(s, mynn_ad_depend(a.d / (2.0 * (double)s),a.d.depends)); }
__device__ static inline ADF mynn_exp(ADF a) { float e = mynn_exp(a.v); return ADF(e, mynn_ad_depend(a.d * (double)e,a.d.depends)); }
__device__ static inline ADF mynn_tanh(ADF a) { float t = mynn_tanh(a.v); return ADF(t, mynn_ad_depend(a.d * (1.0 - (double)t * (double)t),a.d.depends)); }
__device__ static inline ADF mynn_pow(ADF x, ADF y) {
    float p = mynn_pow(x.v, y.v);
    DT d = mynn_ad_zero(x.d, y.d);
    if (x.v <= 0.0f) { mw_domain(3, x.d); mw_domain(3, y.d); }
    if (!dt_zero(x.d)) d = d + x.d * ((double)y.v * ((double)p / (double)x.v));
    if (!dt_zero(y.d)) d = d + y.d * ((double)p * log((double)x.v));
    return ADF(p, mynn_ad_depend(d,x.d.depends || y.d.depends));
}
__device__ static inline ADF mynn_pow(ADF x, float y) { return mynn_pow(x, ADF(y)); }
template<int N> __device__ static inline ADF mynn_ipow(ADF x) {
    float p = mynn_ipow<N>(x.v); double v = (double)x.v;
    double dp = N == 2 ? 2.0*v : 2.0*v*v + (double)r6_mul(x.v, x.v);
    return ADF(p, mynn_ad_depend(x.d * dp,x.d.depends));
}
struct ADD {
    double v; DT d;
    __device__ ADD() : v(0.0), d(0.0) {}
    __device__ ADD(double x) : v(x), d(0.0) {}
    __device__ ADD(float x) : v((double)x), d(0.0) {}
    __device__ ADD(int x) : v((double)x), d(0.0) {}
    __device__ ADD(ADF a) : v((double)a.v), d(a.d) {}
    __device__ ADD(double x, DT dx) : v(x), d(dx) {}
    __device__ explicit operator ADF() const { return ADF((float)v, d); }
};
__device__ inline ADF& ADF::operator=(const ADD& q) { v = (float)q.v; d = q.d; return *this; }
__device__ static __noinline__ ADD mynn_dadd(ADD a, ADD b) { return ADD(mynn_dadd(a.v, b.v), a.d + b.d); }
__device__ static __noinline__ ADD mynn_dsub(ADD a, ADD b) { return ADD(mynn_dsub(a.v, b.v), a.d - b.d); }
__device__ static __noinline__ ADD mynn_dmul(ADD a, ADD b) { return ADD(mynn_dmul(a.v, b.v), mynn_ad_multiply(a.d,a.v,b.d,b.v)); }
__device__ static __noinline__ ADD mynn_ddiv(ADD a, ADD b) { double q = mynn_ddiv(a.v, b.v); if (b.v == 0.0) { mw_domain(4, a.d); mw_domain(4, b.d); } return ADD(q, mw_quot(a.d, b.d, q, b.v)); }
__device__ static __noinline__ ADD r6_add(ADD a, ADD b) { return ADD(r6_add((float)a.v, (float)b.v), a.d + b.d); }
__device__ static __noinline__ ADD r6_sub(ADD a, ADD b) { return ADD(r6_sub((float)a.v, (float)b.v), a.d - b.d); }
__device__ static __noinline__ ADD r6_mul(ADD a, ADD b) { return ADD(r6_mul((float)a.v, (float)b.v), mynn_ad_multiply(a.d,(double)(float)a.v,b.d,(double)(float)b.v)); }
__device__ static __noinline__ ADD r6_div(ADD a, ADD b) { double q = r6_div((float)a.v, (float)b.v); if (b.v == 0.0) { mw_domain(5, a.d); mw_domain(5, b.d); } return ADD(q, mw_quot(a.d, b.d, q, (double)(float)b.v)); }
__device__ static __noinline__ ADD r6_neg(ADD a) { return ADD(r6_neg(a.v), -a.d); }
#define ADF_FLOAT_MIX(F) \
__device__ static inline ADF F(ADF a, float b) { return F(a, ADF(b)); } \
__device__ static inline ADF F(float a, ADF b) { return F(ADF(a), b); } \
template<class A, class I> __device__ static inline typename mw_en<mw_same<A,ADF>::v && mw_is_int<I>::v, ADF>::type F(A a, I b) { return F(a, ADF(b)); } \
template<class I, class A> __device__ static inline typename mw_en<mw_same<A,ADF>::v && mw_is_int<I>::v, ADF>::type F(I a, A b) { return F(ADF(a), b); }
ADF_FLOAT_MIX(r6_add) ADF_FLOAT_MIX(r6_sub) ADF_FLOAT_MIX(r6_mul) ADF_FLOAT_MIX(r6_div)
#undef ADF_FLOAT_MIX
#define ADF_MIX(F) \
__device__ static inline ADD F(ADF a, ADD b) { return F(ADD(a), b); } \
__device__ static inline ADD F(ADD a, ADF b) { return F(a, ADD(b)); } \
__device__ static inline ADD F(ADD a, float b) { return F(a, ADD(b)); } \
__device__ static inline ADD F(float a, ADD b) { return F(ADD(a), b); } \
__device__ static inline ADD F(ADD a, double b) { return F(a, ADD(b)); } \
__device__ static inline ADD F(double a, ADD b) { return F(ADD(a), b); } \
__device__ static inline ADD F(ADF a, double b) { return F(ADD(a), ADD(b)); } \
__device__ static inline ADD F(double a, ADF b) { return F(ADD(a), ADD(b)); }
ADF_MIX(r6_add) ADF_MIX(r6_sub) ADF_MIX(r6_mul) ADF_MIX(r6_div)
#undef ADF_MIX
#define ADD_CMP(OP) \
__device__ static inline bool operator OP(ADD a, ADD b) { return mw_rec(a.v OP b.v); } \
__device__ static inline bool operator OP(ADD a, double b) { return mw_rec(a.v OP b); } \
__device__ static inline bool operator OP(double a, ADD b) { return mw_rec(a OP b.v); } \
__device__ static inline bool operator OP(ADD a, float b) { return mw_rec(a.v OP b); } \
__device__ static inline bool operator OP(float a, ADD b) { return mw_rec(a OP b.v); } \
__device__ static inline bool operator OP(ADD a, ADF b) { return mw_rec(a.v OP (double)b.v); } \
__device__ static inline bool operator OP(ADF a, ADD b) { return mw_rec((double)a.v OP b.v); } \
__device__ static inline bool operator OP(ADD a, int b) { return mw_rec(a.v OP b); }
ADD_CMP(<) ADD_CMP(>) ADD_CMP(<=) ADD_CMP(>=) ADD_CMP(==) ADD_CMP(!=)
#undef ADD_CMP
__device__ static inline ADD mynn_sqrt(ADD a) { double s = mynn_sqrt(a.v); if (a.v == 0.0 && !a.d.depends) return ADD(s, mynn_ad_zero(a.d,a.d)); if (a.v <= 0.0) mw_domain(6, a.d); return ADD(s, mynn_ad_depend(a.d / (2.0 * s),a.d.depends)); }
template<int N> __device__ static inline ADD mynn_ipow(ADD x) {
    double p = mynn_ipow<N>(x.v);
    double dp = N == 2 ? 2.0*x.v : 2.0*x.v*x.v + mynn_dmul(x.v, x.v);
    return ADD(p, mynn_ad_depend(x.d * dp,x.d.depends));
}
__device__ static inline ADF mynn_abs(ADF a) { float r = mynn_abs(a.v); return ADF(r, mw_rec(a.v < 0.f || (a.v == 0.f && signbit(a.v))) ? -a.d : a.d); }
__device__ static inline ADD mynn_abs(ADD a) { double r = mynn_abs(a.v); return ADD(r, mw_rec(a.v < 0.0 || (a.v == 0.0 && signbit(a.v))) ? -a.d : a.d); }
__device__ static inline ADF mynn_sign(ADF a, ADF b) { float r = mynn_sign(a.v, b.v); bool same = mw_rec(signbit(r) == signbit(a.v)); return ADF(r, same ? a.d : -a.d); }
__device__ static inline ADD mynn_sign(ADD a, ADD b) { double r = mynn_sign(a.v, b.v); bool same = mw_rec(signbit(r) == signbit(a.v)); return ADD(r, same ? a.d : -a.d); }
// The generated source calls the float library on a real*8 value (implicit narrowing); same here.
__device__ static inline ADF mynn_exp(ADD a) { return mynn_exp((ADF)a); }
__device__ static inline ADF mynn_tanh(ADD a) { return mynn_tanh((ADF)a); }
__device__ static inline ADF mynn_pow(ADD x, float y) { return mynn_pow((ADF)x, ADF(y)); }
__device__ static inline ADF mynn_log(ADF a) { float r = mynn_log(a.v); if (a.v <= 0.0f) mw_domain(7, a.d); return ADF(r, a.d / (double)a.v); }
__device__ static inline ADF mynn_log10(ADF a) { float r = mynn_log10(a.v); if (a.v <= 0.0f) mw_domain(8, a.d); return ADF(r, a.d / ((double)a.v * 2.302585092994045684)); }
// bhunt logsec (2026-11-11; labelled, OFF by default; derivative only). MW_ICE_LOG10(X, QS) is the log10 of the ice
// block above the PBL in the cover's humidity replacement (module_bl_mynnedmf.F90:3987; written there by
// tape_tools/patch_logsec.py). The primal is the same library call in every mode, so the values are bitwise the same.
// Off: mynn_log10(X), the same tokens as before. MW_LOG10_SECANT: the slope 1/(x ln10) becomes the upward one-sided
// secant over the state-only spread dl = 0.01 qsat, log10(1 + dl/x)/dl (it tends to the exact slope as dl/x -> 0).
// MW_LOG10_HOLD: that log10 carries no derivative (the control arm). Only the tangent image is built for the test.
#ifndef MW_LOGSEC_ONCE
#define MW_LOGSEC_ONCE
#if defined(MW_LOG10_SECANT)
#define MW_ICE_LOG10(X, QS) mynn_log10_sec(X, 0.01f * mw_pv(QS))
#elif defined(MW_LOG10_HOLD)
#define MW_ICE_LOG10(X, QS) mw_frz(mynn_log10(X))
#else
#define MW_ICE_LOG10(X, QS) mynn_log10(X)
#endif
#if defined(MW_LOG10_SECANT)
// x / slope: the slope is 1 / mw_log10_secden(x, dl); log1p keeps it exact where dl/x is small.
__device__ static inline double mw_log10_secden(double x, double dl) {
    return (dl > 0.0 && x > 0.0) ? dl * 2.302585092994045684 / log1p(dl / x) : x * 2.302585092994045684;
}
__device__ static inline float mw_pv(float x) { return x; }
__device__ static inline float mynn_log10_sec(float a, float dl) { return mynn_log10(a); }
#endif
#endif
#if defined(MW_LOG10_SECANT)
__device__ static inline float mw_pv(ADF x) { return x.v; }
__device__ static inline ADF mynn_log10_sec(ADF a, float dl) { float r = mynn_log10(a.v); if (a.v <= 0.0f) mw_domain(8, a.d); return ADF(r, a.d / mw_log10_secden((double)a.v, (double)dl)); }
#endif
__device__ static inline ADF mynn_atan(ADF a) { float r = mynn_atan(a.v); return ADF(r, a.d / (1.0 + (double)a.v * (double)a.v)); }
__device__ static inline ADF mynn_erf(ADF a) { float r = mynn_erf(a.v); return ADF(r, mynn_ad_depend(a.d * (1.1283791670955125739 * exp(-(double)a.v * (double)a.v)),a.d.depends)); }
__device__ static inline ADF mynn_log(ADD a) { return mynn_log((ADF)a); }
__device__ static inline ADF mynn_atan(ADD a) { return mynn_atan((ADF)a); }
__device__ static inline ADF mynn_erf(ADD a) { return mynn_erf((ADF)a); }

// Trajectory freeze: value kept bitwise, no derivative (tangent 0; no tape node, so no adjoint flows back).
__device__ static inline ADF mw_frz(ADF a) { return ADF(a.v); }
__device__ static inline ADD mw_frz(ADD a) { return ADD(a.v); }

// Integer-valued intrinsics: piecewise constant, zero derivative; the primal is the captured routine.
__device__ static inline int mynn_ceiling(ADF a) { return mynn_ceiling(a.v); }
__device__ static inline int mynn_floor(ADF a) { return mynn_floor(a.v); }
__device__ static inline int mynn_nint(ADF a) { return mynn_nint(a.v); }

// domain codes: 1 r6_div_ADF, 2 mynn_sqrt_ADF, 3 other, 4 mynn_ddiv_ADD, 5 r6_div_ADD, 6 mynn_sqrt_ADD, 7 mynn_log_ADF, 8 mynn_log10_ADF

// a12 truephys-term (2026-11-13; gen_whole.py --l2-quot exact): the level-2 stability quotients of f_mym_level2
// (module_bl_mynnedmf.F90:1791-1792) with closed-form partials. Value: the generated statement's own float32
// operations in the same order, so bitwise the forward's. Derivative: evaluated in double from the float32 words with no
// subtraction of nearly equal terms (the generic quotient rule left |drf| / rf^2 as the difference of two terms of size
// |drf|; see gen_whole.py --l2-quot). Domain: a zero denominator is reported as r6_div reports it.
__device__ static inline ADF mw_l2_sh(ADF shc, ADF rfc, ADF rf, float one) {
    const float den = r6_sub(one, rf.v);
    const float v = r6_div(r6_mul(shc.v, r6_sub(rfc.v, rf.v)), den);
    if (den == 0.0f) { mw_domain(1, shc.d); mw_domain(1, rfc.d); mw_domain(1, rf.d); }
    const double b = (double)one - (double)rf.v;                       // 1 - rf
    const double g = ((double)rfc.v - (double)rf.v) / b;               // dsh/dshc = (rfc - rf) / (1 - rf)
    const double wr = (double)shc.v / b;                               // dsh/drfc
    const double wf = (double)shc.v * ((double)rfc.v - (double)one) / (b * b);   // dsh/drf
    DT d = mynn_ad_zero(shc.d, rf.d);
    if (!dt_zero(shc.d)) d = d + shc.d * g;
    if (!dt_zero(rfc.d)) d = d + rfc.d * wr;
    if (!dt_zero(rf.d)) d = d + rf.d * wf;
    return ADF(v, mynn_ad_depend(d, shc.d.depends || rfc.d.depends || rf.d.depends));
}
__device__ static inline ADF mw_l2_sm(ADF smc, ADF rf1, ADF rf2, ADF rf, ADF sh) {
    const float den = r6_sub(rf2.v, rf.v);
    const float v = r6_mul(r6_div(r6_mul(smc.v, r6_sub(rf1.v, rf.v)), den), sh.v);
    if (den == 0.0f) { mw_domain(1, smc.d); mw_domain(1, rf1.d); mw_domain(1, rf2.d); mw_domain(1, rf.d); }
    const double D = (double)rf2.v - (double)rf.v;                     // rf2 - rf
    const double h = ((double)rf1.v - (double)rf.v) / D;               // (rf1 - rf) / (rf2 - rf)
    const double s = (double)smc.v * (double)sh.v;
    DT d = mynn_ad_zero(smc.d, rf.d);
    if (!dt_zero(smc.d)) d = d + smc.d * (h * (double)sh.v);
    if (!dt_zero(rf1.d)) d = d + rf1.d * (s / D);
    if (!dt_zero(rf2.d)) d = d + rf2.d * (-s * h / D);
    if (!dt_zero(rf.d)) d = d + rf.d * (s * ((double)rf1.v - (double)rf2.v) / (D * D));
    if (!dt_zero(sh.d)) d = d + sh.d * ((double)smc.v * h);
    return ADF(v, mynn_ad_depend(d, smc.d.depends || rf1.d.depends || rf2.d.depends || rf.d.depends || sh.d.depends));
}
// a12 truephys-term (--l2-quot exact): the inner level-2 flux Richardson number of f_mym_level2 (F90:1789; the
// generated statement keeps its mynn_min against rfc outside this call), rf = ri1 (ri + ri2 - sqrt(D)),
// D = ri^2 - ri3 ri + ri4. Value: the statement's own float32 operations. Derivative: for 2 ri - ri3 > 0 (stable side)
// d rf / d ri = ri1 (4 ri4 - ri3^2) / (2 sqrt(D) (2 sqrt(D) + 2 ri - ri3)) and
// d rf / d ri1 = (ri (2 ri2 + ri3) + ri2^2 - ri4) / (ri + ri2 + sqrt(D)), the same numbers without the difference of
// nearly equal terms that 1 - (2 ri - ri3) / (2 sqrt(D)) and ri + ri2 - sqrt(D) take there; the plain forms elsewhere.
__device__ static inline ADF mw_l2_rf(ADF ri1, ADF ri, ADF ri2, ADF ri3, ADF ri4) {
    const float arg = r6_add(r6_sub(mynn_ipow<2>(ri.v), r6_mul(ri3.v, ri.v)), ri4.v);
    const float v = r6_mul(ri1.v, r6_sub(r6_add(ri.v, ri2.v), mynn_sqrt(arg)));
    if (arg <= 0.0f) { mw_domain(2, ri.d); mw_domain(2, ri3.d); mw_domain(2, ri4.d); }
    const double x = ri.v, a1 = ri1.v, a2 = ri2.v, a3 = ri3.v, a4 = ri4.v;
    const double D = x * x - a3 * x + a4, s = sqrt(D > 0.0 ? D : 0.0), t = 2.0 * x - a3;
    const double c = t > 0.0 ? (4.0 * a4 - a3 * a3) / (2.0 * s * (2.0 * s + t)) : 1.0 - t / (2.0 * s);
    const double w = t > 0.0 ? (x * (2.0 * a2 + a3) + (a2 * a2 - a4)) / (x + a2 + s) : x + a2 - s;
    DT d = mynn_ad_zero(ri.d, ri1.d);
    if (!dt_zero(ri1.d)) d = d + ri1.d * w;
    if (!dt_zero(ri.d)) d = d + ri.d * (a1 * c);
    if (!dt_zero(ri2.d)) d = d + ri2.d * a1;
    if (!dt_zero(ri3.d)) d = d + ri3.d * (a1 * x / (2.0 * s));
    if (!dt_zero(ri4.d)) d = d + ri4.d * (-a1 / (2.0 * s));
    return ADF(v, mynn_ad_depend(d, ri1.d.depends || ri.d.depends || ri2.d.depends || ri3.d.depends || ri4.d.depends));
}

// fullphys mynn-exact (2026-11-13; gen_whole.py --l25-quot exact): the level-2.5 stability quotients of f_mym_turbulence
// (module_bl_mynnedmf.F90:2874 sm, 2878 sh; the branch q3sq >= q2sq, where qdiv = 1) as one node each, with closed-form
// partials in the inputs q = q3sq, h = ghel, m = gmel, a = a2fac (and, for sh, the factor y = a2 a2fac).
// Value: the generated statement's own double operations in the same order on the same words, so bitwise the forward's.
// Derivative: with e1 = q - e1c h a, e2 = q - e2c h a, e3 = e1 + e3c h a^2, e4 = e1 - e4c h a,
// D = e2 e4 + e5c e3 m (eden, floored at 1e-20 by the statement before), P = e3 - 3 c1 e4:
//   sm = a1 q P / D, sh = (3 c1 e5c m + e2) q y / D.
// Both are homogeneous of degree 0 in (q, h, m), so q d/dq = -(h d/dh + m d/dm). The generic quotient rule formed the
// q-derivative as the difference of two terms of size 2 s / q that cancel to the true value, of relative size h / q:
// where the gradients are small against the turbulence (h, m << q: a well-mixed layer) the tangent and the adjoint
// each kept the cancellation error of those large terms, in different orders (the statement harness: up to 4.1e-09 of
// the statement's own terms). Here the q-partial is taken from whichever of the two exact forms has the smaller
// rounding bound, and the statement is one node, so the tangent and the adjoint use the same partials.
// Where eden was floored (D <= 1e-20) the floor is a constant: only the numerator moves.
__device__ static inline ADD mw_l25_sm(ADD q3, ADD h, ADD m, ADF a, ADD e2, ADD e3, ADD e4, ADD eden, float a1, float c13,
                                       float e1c, float e2c, float e3c, float e4c, float e5c) {
    const double v = mynn_ddiv(mynn_dmul(mynn_dmul(q3.v, (double)a1), mynn_dsub(e3.v, mynn_dmul((double)c13, e4.v))), eden.v);
    const double q = q3.v, hv = h.v, mv = m.v, A = (double)a.v, D = eden.v, A1 = a1, C = c13;
    const double E1 = e1c, E2 = e2c, E3 = e3c, E4 = e4c, E5 = e5c;
    const bool live = D > 1.0e-20;
    const double P = e3.v - C * e4.v;
    const double e3h = -E1 * A + E3 * A * A, e4h = -(E1 + E4) * A, e2h = -E2 * A;
    const double e3a = -E1 * hv + 2.0 * E3 * hv * A, e4a = -(E1 + E4) * hv, e2a = -E2 * hv;
    const double Dh = live ? e2h * e4.v + e2.v * e4h + E5 * mv * e3h : 0.0;
    const double Dm = live ? E5 * e3.v : 0.0;
    const double Da = live ? e2a * e4.v + e2.v * e4a + E5 * mv * e3a : 0.0;
    const double Dq = live ? e4.v + e2.v + E5 * mv : 0.0;
    const double sh_ = (A1 * q * (e3h - C * e4h) - v * Dh) / D;
    const double sm_ = -v * Dm / D;
    const double sa_ = (A1 * q * (e3a - C * e4a) - v * Da) / D;
    double sq_ = (A1 * (P + q * (1.0 - C)) - v * Dq) / D;
    if (live && q != 0.0) {
        const double bh = (fabs(hv * sh_) + fabs(mv * sm_)) / fabs(q);            // rounding bound of the homogeneous form
        const double bd = (fabs(A1 * (P + q * (1.0 - C))) + fabs(v * Dq)) / D;     // rounding bound of the direct form
        if (bh < bd) sq_ = -(hv * sh_ + mv * sm_) / q;
    }
    DT d = mynn_ad_zero(q3.d, h.d);
    if (!dt_zero(q3.d)) d = d + q3.d * sq_;
    if (!dt_zero(h.d)) d = d + h.d * sh_;
    if (!dt_zero(m.d)) d = d + m.d * sm_;
    if (!dt_zero(a.d)) d = d + a.d * sa_;
    return ADD(v, mynn_ad_depend(d, q3.d.depends || h.d.depends || m.d.depends || a.d.depends));
}
__device__ static inline ADD mw_l25_sh(int tke_budget, ADD q3, ADD h, ADD m, ADF a, ADD e2, ADD e3, ADD e4, ADD eden,
                                       float c13e5c, ADF y, float e1c, float e2c, float e3c, float e4c, float e5c) {
    const double X = mynn_dadd(mynn_dmul((double)c13e5c, m.v), e2.v), Y = mynn_dmul(q3.v, (double)y.v);
    const double v = mynn_ddiv(tke_budget == 1 ? mynn_dmul(Y, X) : mynn_dmul(X, Y), eden.v);
    const double q = q3.v, hv = h.v, mv = m.v, A = (double)a.v, D = eden.v, yv = (double)y.v;
    const double E1 = e1c, E2 = e2c, E3 = e3c, E4 = e4c, E5 = e5c, K = c13e5c;
    const bool live = D > 1.0e-20;
    const double e3h = -E1 * A + E3 * A * A, e4h = -(E1 + E4) * A, e2h = -E2 * A;
    const double e3a = -E1 * hv + 2.0 * E3 * hv * A, e4a = -(E1 + E4) * hv, e2a = -E2 * hv;
    const double Dh = live ? e2h * e4.v + e2.v * e4h + E5 * mv * e3h : 0.0;
    const double Dm = live ? E5 * e3.v : 0.0;
    const double Da = live ? e2a * e4.v + e2.v * e4a + E5 * mv * e3a : 0.0;
    const double Dq = live ? e4.v + e2.v + E5 * mv : 0.0;
    const double sh_ = (e2h * q * yv - v * Dh) / D;
    const double sm_ = (K * q * yv - v * Dm) / D;
    const double sa_ = (e2a * q * yv - v * Da) / D;
    const double sy_ = X * q / D;
    double sq_ = ((q * yv + X * yv) - v * Dq) / D;
    if (live && q != 0.0) {
        const double bh = (fabs(hv * sh_) + fabs(mv * sm_)) / fabs(q);
        const double bd = (fabs(q * yv) + fabs(X * yv) + fabs(v * Dq)) / D;
        if (bh < bd) sq_ = -(hv * sh_ + mv * sm_) / q;
    }
    DT d = mynn_ad_zero(q3.d, h.d);
    if (!dt_zero(q3.d)) d = d + q3.d * sq_;
    if (!dt_zero(h.d)) d = d + h.d * sh_;
    if (!dt_zero(m.d)) d = d + m.d * sm_;
    if (!dt_zero(a.d)) d = d + a.d * sa_;
    if (!dt_zero(y.d)) d = d + y.d * sy_;
    return ADD(v, mynn_ad_depend(d, q3.d.depends || h.d.depends || m.d.depends || a.d.depends || y.d.depends));
}
