// ---- gen_kernel.py derivative support (generated section; Apache-2.0) ----
// Value parts keep the forward's exact IEEE float operations; derivatives are the
// exact derivative of the executed operation, carried in double.
#undef RLW_AD
#undef RLW_SU
#undef RLW_MU
#undef RLW_DV
#define RLW_AD(a, b) p7_add((a), (b))
#define RLW_SU(a, b) p7_sub((a), (b))
#define RLW_MU(a, b) p7_mul((a), (b))
#define RLW_DV(a, b) p7_div((a), (b))
#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#undef FSQRT
#define FADD(a, b) p7_add((a), (b))
#define FSUB(a, b) p7_sub((a), (b))
#define FMUL(a, b) p7_mul((a), (b))
#define FDIV(a, b) p7_div((a), (b))
#define FSQRT(a) p7_sqrt((a))
#define G7_MAXARR 64

__device__ __forceinline__ float p7_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float p7_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float p7_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float p7_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float p7_sqrt(float a) { return __fsqrt_rn(a); }

// ---------------- tangent ----------------
struct adf {
    float v; double d;
    __device__ adf() : v(0.0f), d(0.0) {}
    __device__ adf(float x) : v(x), d(0.0) {}
    __device__ adf(float x, double y) : v(x), d(y) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ adf p7_add(adf a, adf b) { return adf(__fadd_rn(a.v, b.v), a.d + b.d); }
__device__ __forceinline__ adf p7_sub(adf a, adf b) { return adf(__fsub_rn(a.v, b.v), a.d - b.d); }
__device__ __forceinline__ adf p7_mul(adf a, adf b) { return adf(__fmul_rn(a.v, b.v), a.d * (double)b.v + (double)a.v * b.d); }
__device__ __forceinline__ adf p7_div(adf a, adf b) {
    double bv = (double)b.v; return adf(__fdiv_rn(a.v, b.v), (a.d * bv - (double)a.v * b.d) / (bv * bv)); }
__device__ __forceinline__ adf p7_sqrt(adf a) { float r = __fsqrt_rn(a.v); return adf(r, a.d / (2.0 * sqrt((double)a.v))); }
__device__ __forceinline__ adf operator-(adf a) { return adf(-a.v, -a.d); }
__device__ __forceinline__ adf p7_unary(adf a, float r, double p) { return adf(r, p * a.d); }

// ---------------- adjoint: per-thread tape in global scratch ----------------
struct G7Scratch {
    int cap, wcap, nin, pad;
    int *a, *b; double *pa, *pb, *adj; short* leaf; long long* li;
    short* warr; long long* widx; int* wt;
    int* ti[G7_MAXARR]; float* sv[G7_MAXARR];
};
struct G7Tape {
    int n, nw, fail; G7Scratch* s; long long base, wbase;
    __device__ void init(G7Scratch* sc, long long tid) {
        s = sc; n = 0; nw = 0; fail = 0; base = tid * (long long)sc->cap; wbase = tid * (long long)sc->wcap; }
    __device__ int push(int x, double px, int y, double py) {
        if (n >= s->cap) { fail |= 1; return -1; }
        long long k = base + n;
        s->a[k] = x; s->pa[k] = px; s->b[k] = y; s->pb[k] = py; s->leaf[k] = -1; s->li[k] = 0;
        return n++;
    }
    __device__ int leafe(int arr, long long idx) {
        if (n >= s->cap) { fail |= 1; return -1; }
        long long k = base + n;
        s->a[k] = -1; s->b[k] = -1; s->pa[k] = 0.0; s->pb[k] = 0.0; s->leaf[k] = (short)arr; s->li[k] = idx;
        return n++;
    }
    __device__ void wrote(int arr, long long idx, int t) {
        if (nw >= s->wcap) { fail |= 2; return; }
        long long k = wbase + nw++;
        s->warr[k] = (short)arr; s->widx[k] = idx; s->wt[k] = t;
        s->ti[arr - s->nin][idx] = t;
    }
    __device__ void finish(double** bp, int nin, int* status) {
        for (int k = 0; k < n; ++k) s->adj[base + k] = 0.0;
        // seed from the last write of each output word; consume its cotangent
        for (int w = 0; w < nw; ++w) {
            long long kw = wbase + w; int arr = s->warr[kw]; long long idx = s->widx[kw]; int t = s->wt[kw];
            int* ti = s->ti[arr - nin];
            if (ti[idx] != t) continue;
            double g = bp[arr][idx]; bp[arr][idx] = 0.0; ti[idx] = -3;
            if (t >= 0) s->adj[base + t] += g;
        }
        for (int k = n - 1; k >= 0; --k) {
            long long kk = base + k; double g = s->adj[kk];
            if (g == 0.0) continue;
            if (s->leaf[kk] >= 0) { bp[s->leaf[kk]][s->li[kk]] += g; continue; }
            if (s->a[kk] >= 0) s->adj[base + s->a[kk]] += s->pa[kk] * g;
            if (s->b[kk] >= 0) s->adj[base + s->b[kk]] += s->pb[kk] * g;
        }
        if (fail) atomicMax(&status[0], fail);
        atomicMax(&status[1], n); atomicMax(&status[2], nw);
    }
};
struct gtv {
    float v; int i; G7Tape* t;
    __device__ gtv() : v(0.0f), i(-1), t(nullptr) {}
    __device__ gtv(float x) : v(x), i(-1), t(nullptr) {}
    __device__ gtv(float x, int k, G7Tape* tape) : v(x), i(k), t(tape) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ gtv p7_node(float v, const gtv& x, double px, const gtv& y, double py) {
    G7Tape* t = x.t ? x.t : y.t;
    if (!t || (x.i < 0 && y.i < 0)) return gtv(v);
    int k = t->push(x.i, px, y.i, py);
    return k < 0 ? gtv(v) : gtv(v, k, t);
}
__device__ __forceinline__ gtv p7_add(gtv a, gtv b) { return p7_node(__fadd_rn(a.v, b.v), a, 1.0, b, 1.0); }
__device__ __forceinline__ gtv p7_sub(gtv a, gtv b) { return p7_node(__fsub_rn(a.v, b.v), a, 1.0, b, -1.0); }
__device__ __forceinline__ gtv p7_mul(gtv a, gtv b) { return p7_node(__fmul_rn(a.v, b.v), a, (double)b.v, b, (double)a.v); }
__device__ __forceinline__ gtv p7_div(gtv a, gtv b) {
    double bv = (double)b.v; return p7_node(__fdiv_rn(a.v, b.v), a, 1.0 / bv, b, -(double)a.v / (bv * bv)); }
__device__ __forceinline__ gtv p7_sqrt(gtv a) { float r = __fsqrt_rn(a.v); return p7_node(r, a, 1.0 / (2.0 * sqrt((double)a.v)), gtv(), 0.0); }
__device__ __forceinline__ gtv operator-(gtv a) { return p7_node(-a.v, a, -1.0, gtv(), 0.0); }
__device__ __forceinline__ gtv p7_unary(gtv a, float r, double p) { return p7_node(r, a, p, gtv(), 0.0); }

#define G7_MIXED(T) \
__device__ __forceinline__ T p7_add(T a, float b) { return p7_add(a, T(b)); } \
__device__ __forceinline__ T p7_add(float a, T b) { return p7_add(T(a), b); } \
__device__ __forceinline__ T p7_sub(T a, float b) { return p7_sub(a, T(b)); } \
__device__ __forceinline__ T p7_sub(float a, T b) { return p7_sub(T(a), b); } \
__device__ __forceinline__ T p7_mul(T a, float b) { return p7_mul(a, T(b)); } \
__device__ __forceinline__ T p7_mul(float a, T b) { return p7_mul(T(a), b); } \
__device__ __forceinline__ T p7_div(T a, float b) { return p7_div(a, T(b)); } \
__device__ __forceinline__ T p7_div(float a, T b) { return p7_div(T(a), b); } \
__device__ __forceinline__ T operator+(T a, T b) { return p7_add(a, b); } \
__device__ __forceinline__ T operator-(T a, T b) { return p7_sub(a, b); } \
__device__ __forceinline__ T operator*(T a, T b) { return p7_mul(a, b); } \
__device__ __forceinline__ T operator/(T a, T b) { return p7_div(a, b); } \
__device__ __forceinline__ T operator+(T a, float b) { return p7_add(a, b); } \
__device__ __forceinline__ T operator-(T a, float b) { return p7_sub(a, b); } \
__device__ __forceinline__ T operator*(T a, float b) { return p7_mul(a, b); } \
__device__ __forceinline__ T operator/(T a, float b) { return p7_div(a, b); } \
__device__ __forceinline__ T operator+(float a, T b) { return p7_add(a, b); } \
__device__ __forceinline__ T operator-(float a, T b) { return p7_sub(a, b); } \
__device__ __forceinline__ T operator*(float a, T b) { return p7_mul(a, b); } \
__device__ __forceinline__ T operator/(float a, T b) { return p7_div(a, b); } \
__device__ __forceinline__ T& operator+=(T& a, T b) { a = p7_add(a, b); return a; } \
__device__ __forceinline__ T& operator-=(T& a, T b) { a = p7_sub(a, b); return a; } \
__device__ __forceinline__ T& operator*=(T& a, T b) { a = p7_mul(a, b); return a; } \
__device__ __forceinline__ bool operator<(T a, T b) { return a.v < b.v; } \
__device__ __forceinline__ bool operator>(T a, T b) { return a.v > b.v; } \
__device__ __forceinline__ bool operator<=(T a, T b) { return a.v <= b.v; } \
__device__ __forceinline__ bool operator>=(T a, T b) { return a.v >= b.v; } \
__device__ __forceinline__ bool operator==(T a, T b) { return a.v == b.v; } \
__device__ __forceinline__ bool operator!=(T a, T b) { return a.v != b.v; } \
__device__ __forceinline__ bool operator<(T a, float b) { return a.v < b; } \
__device__ __forceinline__ bool operator>(T a, float b) { return a.v > b; } \
__device__ __forceinline__ bool operator<=(T a, float b) { return a.v <= b; } \
__device__ __forceinline__ bool operator>=(T a, float b) { return a.v >= b; } \
__device__ __forceinline__ bool operator==(T a, float b) { return a.v == b; } \
__device__ __forceinline__ bool operator!=(T a, float b) { return a.v != b; } \
__device__ __forceinline__ bool operator<(float a, T b) { return a < b.v; } \
__device__ __forceinline__ bool operator>(float a, T b) { return a > b.v; } \
__device__ __forceinline__ bool operator<=(float a, T b) { return a <= b.v; } \
__device__ __forceinline__ bool operator>=(float a, T b) { return a >= b.v; } \
__device__ __forceinline__ T fmodf(T a, float c) { return p7_unary(a, fmodf(a.v, c), 1.0); } \
__device__ __forceinline__ T fabsf(T a) { return p7_unary(a, fabsf(a.v), a.v < 0.0f ? -1.0 : 1.0); } \
__device__ __forceinline__ T fminf(T a, T b) { return fminf(a.v, b.v) == a.v ? a : b; } \
__device__ __forceinline__ T fmaxf(T a, T b) { return fmaxf(a.v, b.v) == a.v ? a : b; } \
__device__ __forceinline__ T fminf(T a, float b) { return fminf(a, T(b)); } \
__device__ __forceinline__ T fmaxf(T a, float b) { return fmaxf(a, T(b)); } \
__device__ __forceinline__ T fminf(float a, T b) { return fminf(T(a), b); } \
__device__ __forceinline__ T fmaxf(float a, T b) { return fmaxf(T(a), b); } \
__device__ __forceinline__ T rlw_log(T a) { return p7_unary(a, rlw_log(a.v), 1.0 / (double)a.v); } \
__device__ __forceinline__ T rlw_exp(T a) { return p7_unary(a, rlw_exp(a.v), exp((double)a.v)); } \
__device__ __forceinline__ T rlw_pow(T x, T y) { \
    float r = rlw_pow(x.v, y.v); double xv = x.v, yv = y.v; \
    T rx = p7_unary(x, r, yv * pow(xv, yv - 1.0)); \
    return p7_add(p7_sub(rx, T(r)), p7_unary(y, r, xv > 0.0 ? (double)r * log(xv) : 0.0)); } \
__device__ __forceinline__ T rlw_pow(T x, float y) { return rlw_pow(x, T(y)); } \
__device__ __forceinline__ T rlw_pow(float x, T y) { return rlw_pow(T(x), y); }
G7_MIXED(adf)
G7_MIXED(gtv)

namespace g7base_g7fwd {
typedef float real;
typedef const float* AIn;
typedef float* AOut;
}
namespace g7base_g7tl {
typedef adf real;
struct Ref {
    float* v; double* d;
    __device__ Ref& operator=(adf x) { *v = x.v; *d = x.d; return *this; }
    __device__ Ref& operator=(float x) { *v = x; *d = 0.0; return *this; }
    __device__ Ref& operator=(const Ref& r) { return *this = (adf)r; }
    __device__ operator adf() const { return adf(*v, *d); }
};
struct AIn {
    const float* v; const double* d;
    __device__ adf operator[](long long i) const { return adf(v[i], d[i]); }
    __device__ AIn operator+(long long o) const { return AIn{v + o, d + o}; }
};
struct AOut {
    float* v; double* d;
    __device__ Ref operator[](long long i) const { return Ref{v + i, d + i}; }
    __device__ AOut operator+(long long o) const { return AOut{v + o, d + o}; }
};
}
namespace g7base_g7ad {
typedef gtv real;
struct AIn {
    const float* v; int id; G7Tape* t; long long off;
    __device__ AIn(const float* p, int i, G7Tape* tp, long long o = 0) : v(p), id(i), t(tp), off(o) {}
    __device__ gtv operator[](long long i) const { long long k = off + i; return gtv(v[k], t->leafe(id, k), t); }
    __device__ AIn operator+(long long o) const { return AIn(v, id, t, off + o); }
};
struct Ref {
    const float* v; int id; G7Tape* t; long long k;
    __device__ Ref& operator=(gtv x) {
        t->s->sv[id - t->s->nin][k] = x.v; t->wrote(id, k, x.i >= 0 ? x.i : -2); return *this; }
    __device__ Ref& operator=(float x) { return *this = gtv(x); }
    __device__ Ref& operator=(const Ref& r) { return *this = (gtv)r; }
    __device__ operator gtv() const {
        int ti = t->s->ti[id - t->s->nin][k];
        if (ti == -1) return gtv(v[k], t->leafe(id, k), t);     // read before any write: pre-launch value
        float x = t->s->sv[id - t->s->nin][k];
        return ti >= 0 ? gtv(x, ti, t) : gtv(x);
    }
};
struct AOut {
    const float* v; int id; G7Tape* t; long long off;
    __device__ AOut(const float* p, int i, G7Tape* tp, long long o = 0) : v(p), id(i), t(tp), off(o) {}
    __device__ Ref operator[](long long i) const { return Ref{v, id, t, off + i}; }
    __device__ AOut operator+(long long o) const { return AOut(v, id, t, off + o); }
};
}
