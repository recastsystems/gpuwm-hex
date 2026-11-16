// ---- p7 SW derivative support for rsw_taumol_b (generated section; Apache-2.0; ad/held-rsw_taumol_b) ----
// Value part: the forward's own helpers (rsw_rn_add / rsw_rn_sub / rsw_rn_mul / mpas_sw_div), so every float the
// derivative computes is the forward's float bit for bit. Derivative: the exact derivative of the executed operation,
// in double.
#undef AD
#undef SU
#undef MU
#undef DV
#define AD(a, b) p7_add((a), (b))
#define SU(a, b) p7_sub((a), (b))
#define MU(a, b) p7_mul((a), (b))
#define DV(a, b) p7_div((a), (b))
// Tape entries per (column, layer, g-point). The count depends only on the executed band/branch path (constants make no
// entries), never on values; every launch covers all 28 band paths (14 bands x lower/upper), and the measured high
// water is 60 on every captured launch (steps 1, 3, 6, 12). 96 keeps a 1.6x margin; an overflow refuses by name.
#ifndef P7SW_TMAX
#define P7SW_TMAX 96
#endif
#define P7SW_NIN 14

__device__ __forceinline__ float p7_add(float a, float b) { return rsw_rn_add(a, b); }
__device__ __forceinline__ float p7_sub(float a, float b) { return rsw_rn_sub(a, b); }
__device__ __forceinline__ float p7_mul(float a, float b) { return rsw_rn_mul(a, b); }
__device__ __forceinline__ float p7_div(float a, float b) { return mpas_sw_div(a, b); }

// ---------------- tangent type ----------------
struct adf {
    float v; double d;
    __device__ adf() : v(0.0f), d(0.0) {}
    __device__ adf(float x) : v(x), d(0.0) {}
    __device__ adf(float x, double y) : v(x), d(y) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ adf p7_add(adf a, adf b) { return adf(rsw_rn_add(a.v, b.v), a.d + b.d); }
__device__ __forceinline__ adf p7_sub(adf a, adf b) { return adf(rsw_rn_sub(a.v, b.v), a.d - b.d); }
__device__ __forceinline__ adf p7_mul(adf a, adf b) {
    return adf(rsw_rn_mul(a.v, b.v), a.d * (double)b.v + (double)a.v * b.d); }
__device__ __forceinline__ adf p7_div(adf a, adf b) {
    double bv = (double)b.v;
    return adf(mpas_sw_div(a.v, b.v), (a.d * bv - (double)a.v * b.d) / (bv * bv)); }
// truncf is piecewise constant: zero derivative. The eta bin js = 1 + (int)specmult is the trajectory's; the fraction
// fs = specmult - truncf(specmult) keeps d specmult.
__device__ __forceinline__ adf truncf(adf a) { return adf(truncf(a.v)); }

// ---------------- adjoint type: per-thread tape ----------------
struct P7SwTape {
    int n, fail;
    short a[P7SW_TMAX], b[P7SW_TMAX], leaf[P7SW_TMAX];
    double pa[P7SW_TMAX], pb[P7SW_TMAX];
};
struct tv {
    float v; int i; P7SwTape* t;
    __device__ tv() : v(0.0f), i(-1), t(nullptr) {}
    __device__ tv(float x) : v(x), i(-1), t(nullptr) {}
    __device__ tv(float x, int k, P7SwTape* tape) : v(x), i(k), t(tape) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ tv p7_node(float v, const tv& x, double px, const tv& y, double py) {
    P7SwTape* t = x.t ? x.t : y.t;
    if (!t || (x.i < 0 && y.i < 0)) return tv(v);
    if (t->n >= P7SW_TMAX) { t->fail = 1; return tv(v); }
    int k = t->n++;
    t->a[k] = (short)x.i; t->pa[k] = px; t->b[k] = (short)y.i; t->pb[k] = py; t->leaf[k] = -1;
    return tv(v, k, t);
}
__device__ __forceinline__ tv p7_add(tv a, tv b) { return p7_node(rsw_rn_add(a.v, b.v), a, 1.0, b, 1.0); }
__device__ __forceinline__ tv p7_sub(tv a, tv b) { return p7_node(rsw_rn_sub(a.v, b.v), a, 1.0, b, -1.0); }
__device__ __forceinline__ tv p7_mul(tv a, tv b) { return p7_node(rsw_rn_mul(a.v, b.v), a, (double)b.v, b, (double)a.v); }
__device__ __forceinline__ tv p7_div(tv a, tv b) {
    double bv = (double)b.v;
    return p7_node(mpas_sw_div(a.v, b.v), a, 1.0 / bv, b, -(double)a.v / (bv * bv)); }
__device__ __forceinline__ tv truncf(tv a) { return tv(truncf(a.v)); }

// mixed-operand forms and comparisons for both types (the copied text uses only the macros and comparisons)
#define P7SW_MIXED(T) \
__device__ __forceinline__ T p7_add(T a, float b) { return p7_add(a, T(b)); } \
__device__ __forceinline__ T p7_add(float a, T b) { return p7_add(T(a), b); } \
__device__ __forceinline__ T p7_sub(T a, float b) { return p7_sub(a, T(b)); } \
__device__ __forceinline__ T p7_sub(float a, T b) { return p7_sub(T(a), b); } \
__device__ __forceinline__ T p7_mul(T a, float b) { return p7_mul(a, T(b)); } \
__device__ __forceinline__ T p7_mul(float a, T b) { return p7_mul(T(a), b); } \
__device__ __forceinline__ T p7_div(T a, float b) { return p7_div(a, T(b)); } \
__device__ __forceinline__ T p7_div(float a, T b) { return p7_div(T(a), b); } \
__device__ __forceinline__ bool operator<(T a, float b) { return a.v < b; } \
__device__ __forceinline__ bool operator>(T a, float b) { return a.v > b; } \
__device__ __forceinline__ bool operator<=(T a, float b) { return a.v <= b; } \
__device__ __forceinline__ bool operator>=(T a, float b) { return a.v >= b; }
P7SW_MIXED(adf)
P7SW_MIXED(tv)

// ---------------- argument readers / writers ----------------
// Tangent: an active input is (value, tangent) at the same index; an output stores the value and the tangent. Adding
// +0.0 turns a -0.0 tangent word into +0.0, so the zero direction is +0.0 by bit pattern.
struct TlIn {
    const float* v; const double* d;
    __device__ adf operator[](int i) const { return adf(v[i], d[i]); }
};
struct TlOut {
    float* v; double* d;
    struct P { float* v; double* d; __device__ P& operator=(adf x) { *v = x.v; *d = x.d + 0.0; return *this; } };
    __device__ P operator[](size_t i) const { return P{v + i, d + i}; }
};
// Adjoint: every read of an active input is a tape leaf of its slot. The thread owns layer L of one column, so every
// read must be at index L (fail code 4 otherwise: not a pattern of the pinned body). Each output (taug, taur) is
// written once per g-point (fail code 5 otherwise).
struct AdCtx {
    P7SwTape* t; int L;
    int oe[2]; long long oi[2]; int nout;
};
struct AdIn {
    const float* v; int slot; AdCtx* c;
    __device__ tv operator[](int i) const {
        float x = v[i];
        if (i != c->L) { c->t->fail = 4; return tv(x); }
        P7SwTape* t = c->t;
        if (t->n >= P7SW_TMAX) { t->fail = 1; return tv(x); }
        int k = t->n++;
        t->a[k] = -1; t->b[k] = -1; t->pa[k] = 0.0; t->pb[k] = 0.0; t->leaf[k] = (short)slot;
        return tv(x, k, t);
    }
};
struct AdOut {
    int k; AdCtx* c;
    struct P {
        int k; AdCtx* c; long long i;
        __device__ P& operator=(tv x) {
            if (c->nout & (1 << k)) { c->t->fail = 5; return *this; }
            c->nout |= 1 << k; c->oe[k] = x.i; c->oi[k] = i;
            return *this;
        }
    };
    __device__ P operator[](size_t i) const { return P{k, c, (long long)i}; }
};

// ---------------- adjoint type, compact tape (ad/last-rad) ----------------
// The same tape and the same reverse arithmetic as tv/P7SwTape, in 12 bytes an entry instead of 22: one code word
// (op | (a+1) << 8 | (b+1) << 16; op 0 = leaf of input slot a) and the two float operand values. The partials are
// computed in the sweep by the very expressions p7_node was given (add 1, 1; sub 1, -1; mul (double)b, (double)a; div
// 1/(double)b, -(double)a/((double)b*(double)b)), so every double the sweep multiplies is the same bit for bit (the image
// is built with --fmad=false). THE BREAKAGE THIS PREVENTS: none; it is speed only (the 22-byte tape per thread did not
// stay in cache on the radiation step).
#ifndef P7SWC_TMAX
#define P7SWC_TMAX 64
#endif
struct P7SwTapeC {
    int n, fail;
    int code[P7SWC_TMAX];
    float va[P7SWC_TMAX], vb[P7SWC_TMAX];
};
struct tvc {
    float v; int i; P7SwTapeC* t;
    __device__ tvc() : v(0.0f), i(-1), t(nullptr) {}
    __device__ tvc(float x) : v(x), i(-1), t(nullptr) {}
    __device__ tvc(float x, int k, P7SwTapeC* tape) : v(x), i(k), t(tape) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ tvc p7c_node(float v, int op, const tvc& x, const tvc& y) {
    P7SwTapeC* t = x.t ? x.t : y.t;
    if (!t || (x.i < 0 && y.i < 0)) return tvc(v);
    if (t->n >= P7SWC_TMAX) { t->fail = 1; return tvc(v); }
    int k = t->n++;
    t->code[k] = op | ((x.i + 1) << 8) | ((y.i + 1) << 16); t->va[k] = x.v; t->vb[k] = y.v;
    return tvc(v, k, t);
}
__device__ __forceinline__ tvc p7_add(tvc a, tvc b) { return p7c_node(rsw_rn_add(a.v, b.v), 1, a, b); }
__device__ __forceinline__ tvc p7_sub(tvc a, tvc b) { return p7c_node(rsw_rn_sub(a.v, b.v), 2, a, b); }
__device__ __forceinline__ tvc p7_mul(tvc a, tvc b) { return p7c_node(rsw_rn_mul(a.v, b.v), 3, a, b); }
__device__ __forceinline__ tvc p7_div(tvc a, tvc b) { return p7c_node(mpas_sw_div(a.v, b.v), 4, a, b); }
__device__ __forceinline__ tvc truncf(tvc a) { return tvc(truncf(a.v)); }
P7SW_MIXED(tvc)
struct AdCtxC {
    P7SwTapeC* t; int L;
    int oe[2]; long long oi[2]; int nout;
};
struct AdInC {
    const float* v; int slot; AdCtxC* c;
    __device__ tvc operator[](int i) const {
        float x = v[i];
        if (i != c->L) { c->t->fail = 4; return tvc(x); }
        P7SwTapeC* t = c->t;
        if (t->n >= P7SWC_TMAX) { t->fail = 1; return tvc(x); }
        int k = t->n++;
        t->code[k] = slot << 8;
        return tvc(x, k, t);
    }
};
struct AdOutC {
    int k; AdCtxC* c;
    struct P {
        int k; AdCtxC* c; long long i;
        __device__ P& operator=(tvc x) {
            if (c->nout & (1 << k)) { c->t->fail = 5; return *this; }
            c->nout |= 1 << k; c->oe[k] = x.i; c->oi[k] = i;
            return *this;
        }
    };
    __device__ P operator[](size_t i) const { return P{k, c, (long long)i}; }
};
