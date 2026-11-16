// ---- p7 derivative support (generated section; Apache-2.0) ----
// Arithmetic keeps the forward's exact IEEE float rounding for the value part;
// derivatives are the exact derivative of the executed operation, in double.
#undef RLW_AD
#undef RLW_SU
#undef RLW_MU
#undef RLW_DV
#define RLW_AD(a, b) p7_add((a), (b))
#define RLW_SU(a, b) p7_sub((a), (b))
#define RLW_MU(a, b) p7_mul((a), (b))
#define RLW_DV(a, b) p7_div((a), (b))
#undef FS
#undef WXS
#undef TAUG
#undef FRACS
#undef TAUGB_GPOINT_PROLOGUE
#define TAUGB_GPOINT_PROLOGUE
#define FS(slot) ctx.ld(((long long)(slot) * ncol + col) * nl + ((lay) - 1), (slot), 0)
#define WXS(ix) ctx.ld(((long long)col * MAXXSEC + ((ix) - 1)) * nl + ((lay) - 1), NFSLOT + (ix) - 1, 1)
#define TAUG(ig1) ctx.out(0, ((long long)col * nl + ((lay) - 1)) * NGPTLW + ((ig1) - 1))
#define FRACS(ig1) ctx.out(1, ((long long)col * nl + ((lay) - 1)) * NGPTLW + ((ig1) - 1))
#define P7_NLEAF (NFSLOT + MAXXSEC)
#ifndef P7_TMAX
#define P7_TMAX 512
#endif

__device__ __forceinline__ float p7_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float p7_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float p7_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float p7_div(float a, float b) { return __fdiv_rn(a, b); }

// ---------------- tangent type ----------------
struct adf {
    float v; double d;
    __device__ adf() : v(0.0f), d(0.0) {}
    __device__ adf(float x) : v(x), d(0.0) {}
    __device__ adf(float x, double y) : v(x), d(y) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ adf p7_add(adf a, adf b) { return adf(__fadd_rn(a.v, b.v), a.d + b.d); }
__device__ __forceinline__ adf p7_sub(adf a, adf b) { return adf(__fsub_rn(a.v, b.v), a.d - b.d); }
__device__ __forceinline__ adf p7_mul(adf a, adf b) {
    return adf(__fmul_rn(a.v, b.v), a.d * (double)b.v + (double)a.v * b.d); }
__device__ __forceinline__ adf p7_div(adf a, adf b) {
    double bv = (double)b.v;
    return adf(__fdiv_rn(a.v, b.v), (a.d * bv - (double)a.v * b.d) / (bv * bv)); }
__device__ __forceinline__ adf operator-(adf a) { return adf(-a.v, -a.d); }
__device__ __forceinline__ adf p7_fmod(adf a, float c) { return adf(fmodf(a.v, c), a.d); }

// ---------------- adjoint type: per-thread tape ----------------
// G4 (2026-11-06): the tape's entries live in the launch's frame arena (global memory, set by the host through
// p7_frame_base), laid out entry-major across the launch's threads (entry k of thread t at p[k * nthreads + t]) so a
// warp's accesses coalesce. On the stack they were a 15,968-byte frame per thread that the driver reserved for every
// resident thread of the card (4.29 GiB on an RTX PRO 6000). Storage only: every value and slope is the same.
template <class T> struct P7Col {
    T* p; long long s;
    __device__ __forceinline__ T& operator[](int k) const { return p[(long long)k * s]; }
};
struct P7Tape {
    int n, fail;
    P7Col<short> a, b, leaf;
    P7Col<double> pa, pb;
};
struct tv {
    float v; int i; P7Tape* t;
    __device__ tv() : v(0.0f), i(-1), t(nullptr) {}
    __device__ tv(float x) : v(x), i(-1), t(nullptr) {}
    __device__ tv(float x, int k, P7Tape* tape) : v(x), i(k), t(tape) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ tv p7_node(float v, const tv& x, double px, const tv& y, double py) {
    P7Tape* t = x.t ? x.t : y.t;
    if (!t || (x.i < 0 && y.i < 0)) return tv(v);
    if (t->n >= P7_TMAX) { t->fail = 1; return tv(v); }
    int k = t->n++;
    t->a[k] = (short)x.i; t->pa[k] = px; t->b[k] = (short)y.i; t->pb[k] = py; t->leaf[k] = -1;
    return tv(v, k, t);
}
__device__ __forceinline__ tv p7_add(tv a, tv b) { return p7_node(__fadd_rn(a.v, b.v), a, 1.0, b, 1.0); }
__device__ __forceinline__ tv p7_sub(tv a, tv b) { return p7_node(__fsub_rn(a.v, b.v), a, 1.0, b, -1.0); }
__device__ __forceinline__ tv p7_mul(tv a, tv b) { return p7_node(__fmul_rn(a.v, b.v), a, (double)b.v, b, (double)a.v); }
__device__ __forceinline__ tv p7_div(tv a, tv b) {
    double bv = (double)b.v;
    return p7_node(__fdiv_rn(a.v, b.v), a, 1.0 / bv, b, -(double)a.v / (bv * bv)); }
__device__ __forceinline__ tv operator-(tv a) { return p7_node(-a.v, a, -1.0, tv(), 0.0); }
__device__ __forceinline__ tv p7_fmod(tv a, float c) { return p7_node(fmodf(a.v, c), a, 1.0, tv(), 0.0); }

// mixed-operand forms, comparisons and plain operators for both types
#define P7_MIXED(T) \
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
__device__ __forceinline__ bool operator<(T a, T b) { return a.v < b.v; } \
__device__ __forceinline__ bool operator>(T a, T b) { return a.v > b.v; } \
__device__ __forceinline__ bool operator<=(T a, T b) { return a.v <= b.v; } \
__device__ __forceinline__ bool operator>=(T a, T b) { return a.v >= b.v; } \
__device__ __forceinline__ bool operator<(T a, float b) { return a.v < b; } \
__device__ __forceinline__ bool operator>(T a, float b) { return a.v > b; } \
__device__ __forceinline__ bool operator<=(T a, float b) { return a.v <= b; } \
__device__ __forceinline__ bool operator>=(T a, float b) { return a.v >= b; } \
__device__ __forceinline__ bool operator==(T a, float b) { return a.v == b; } \
__device__ __forceinline__ bool operator!=(T a, float b) { return a.v != b; } \
__device__ __forceinline__ bool operator<(float a, T b) { return a < b.v; } \
__device__ __forceinline__ bool operator>(float a, T b) { return a > b.v; } \
__device__ __forceinline__ T fmodf(T a, float c) { return p7_fmod(a, c); }
P7_MIXED(adf)
P7_MIXED(tv)

// rlw_pow: value from the pinned glibc transcription, derivative of x**y.
__device__ __forceinline__ adf rlw_pow(adf x, adf y) {
    float r = rlw_pow(x.v, y.v); double xv = x.v, yv = y.v;
    double px = yv * pow(xv, yv - 1.0), py = xv > 0.0 ? (double)r * log(xv) : 0.0;
    return adf(r, px * x.d + py * y.d); }
__device__ __forceinline__ tv rlw_pow(tv x, tv y) {
    float r = rlw_pow(x.v, y.v); double xv = x.v, yv = y.v;
    double px = yv * pow(xv, yv - 1.0), py = xv > 0.0 ? (double)r * log(xv) : 0.0;
    return p7_node(r, x, px, y, py); }
__device__ __forceinline__ adf rlw_pow(adf x, float y) { return rlw_pow(x, adf(y)); }
__device__ __forceinline__ adf rlw_pow(float x, adf y) { return rlw_pow(adf(x), y); }
__device__ __forceinline__ tv rlw_pow(tv x, float y) { return rlw_pow(x, tv(y)); }
__device__ __forceinline__ tv rlw_pow(float x, tv y) { return rlw_pow(tv(x), y); }

namespace p7base_p7fwd {
typedef float real;
struct Ctx {
    const float* fs; const float* wx; float* taug; float* fracs;
    __device__ float ld(long long i, int, int kind) const { return kind ? wx[i] : fs[i]; }
    __device__ float& out(int k, long long i) { return k ? fracs[i] : taug[i]; }
};
}
namespace p7base_p7tl {
typedef adf real;
struct Out {
    float* v; double* d;
    __device__ Out& operator=(adf x) { *v = x.v; *d = x.d; return *this; }
    __device__ Out& operator=(float x) { *v = x; *d = 0.0; return *this; }
    // read-modify-write of an output this thread already wrote (g-point corrections)
    __device__ operator adf() const { return adf(*v, *d); }
};
struct Ctx {
    const float* fs; const double* dfs; const float* wx; const double* dwx;
    float* taug; float* fracs; double* dtaug; double* dfracs;
    __device__ adf ld(long long i, int, int kind) const { return kind ? adf(wx[i], dwx[i]) : adf(fs[i], dfs[i]); }
    __device__ Out out(int k, long long i) { return k ? Out{fracs + i, dfracs + i} : Out{taug + i, dtaug + i}; }
};
}
namespace p7base_p7ad {
typedef tv real;
struct Ctx;
struct Out {
    Ctx* c; int k; long long i;
    __device__ Out& operator=(tv x);
    __device__ Out& operator=(float x) { return *this = tv(x); }
    __device__ operator tv() const;
};
struct Ctx {
    const float* fs; const float* wx; P7Tape* t; float* vt; float* vf;
    int nout; int ok[8]; long long oi[8]; int oe[8]; float ov[8];
    __device__ tv ld(long long i, int slot, int kind) {
        float v = kind ? wx[i] : fs[i];
        if (t->n >= P7_TMAX) { t->fail = 1; return tv(v); }
        int k = t->n++;
        t->a[k] = -1; t->b[k] = -1; t->pa[k] = 0.0; t->pb[k] = 0.0; t->leaf[k] = (short)slot;
        return tv(v, k, t);
    }
    __device__ Out out(int k, long long i) { return Out{this, k, i}; }
};
__device__ __forceinline__ Out& Out::operator=(tv x) {
    int j = 0;
    for (; j < c->nout; ++j) if (c->ok[j] == k && c->oi[j] == i) break;
    if (j == c->nout) { if (c->nout >= 8) { c->t->fail = 2; return *this; } c->nout++; }
    c->ok[j] = k; c->oi[j] = i; c->oe[j] = x.i; c->ov[j] = x.v;
    if (c->vt) { if (k) c->vf[i] = x.v; else c->vt[i] = x.v; }
    return *this;
}
__device__ __forceinline__ Out::operator tv() const {
    for (int j = 0; j < c->nout; ++j)
        if (c->ok[j] == k && c->oi[j] == i) return c->oe[j] >= 0 ? tv(c->ov[j], c->oe[j], c->t) : tv(c->ov[j]);
    c->t->fail = 3;  // read of an output not written by this thread: not a pattern in the pinned source
    return tv(0.0f);
}
}

// ---------------- adjoint type, compact tape (ad/last-rad) ----------------
// The same tape and the same reverse arithmetic as tv/P7Tape, in 20 bytes an entry (code, two float operands, the
// double adjoint) instead of 30: code = op | (a+1) << 10 | (b+1) << 20 (op 0 = leaf of slot a). The partials are
// computed in the sweep by the very expressions p7_node was given (add 1, 1; sub 1, -1; mul (double)b, (double)a;
// div 1/(double)b, -(double)a/((double)b*(double)b); neg -1; fmod 1; pow from x, y and r = rlw_pow(x, y) as above), so
// every double the sweep multiplies is the same bit for bit (--fmad=false). Speed only; no breakage prevented.
struct P7TapeC {
    int n, fail;
    P7Col<int> code; P7Col<float> va, vb;
};
struct tvc {
    float v; int i; P7TapeC* t;
    __device__ tvc() : v(0.0f), i(-1), t(nullptr) {}
    __device__ tvc(float x) : v(x), i(-1), t(nullptr) {}
    __device__ tvc(float x, int k, P7TapeC* tape) : v(x), i(k), t(tape) {}
    __device__ explicit operator int() const { return (int)v; }
};
__device__ __forceinline__ tvc p7c_node(float v, int op, const tvc& x, const tvc& y) {
    P7TapeC* t = x.t ? x.t : y.t;
    if (!t || (x.i < 0 && y.i < 0)) return tvc(v);
    if (t->n >= P7_TMAX) { t->fail = 1; return tvc(v); }
    int k = t->n++;
    t->code[k] = op | ((x.i + 1) << 10) | ((y.i + 1) << 20); t->va[k] = x.v; t->vb[k] = y.v;
    return tvc(v, k, t);
}
__device__ __forceinline__ tvc p7_add(tvc a, tvc b) { return p7c_node(__fadd_rn(a.v, b.v), 1, a, b); }
__device__ __forceinline__ tvc p7_sub(tvc a, tvc b) { return p7c_node(__fsub_rn(a.v, b.v), 2, a, b); }
__device__ __forceinline__ tvc p7_mul(tvc a, tvc b) { return p7c_node(__fmul_rn(a.v, b.v), 3, a, b); }
__device__ __forceinline__ tvc p7_div(tvc a, tvc b) { return p7c_node(__fdiv_rn(a.v, b.v), 4, a, b); }
__device__ __forceinline__ tvc operator-(tvc a) { return p7c_node(-a.v, 5, a, tvc()); }
__device__ __forceinline__ tvc p7_fmod(tvc a, float c) { return p7c_node(fmodf(a.v, c), 6, a, tvc()); }
P7_MIXED(tvc)
__device__ __forceinline__ tvc rlw_pow(tvc x, tvc y) { return p7c_node(rlw_pow(x.v, y.v), 7, x, y); }
__device__ __forceinline__ tvc rlw_pow(tvc x, float y) { return rlw_pow(x, tvc(y)); }
__device__ __forceinline__ tvc rlw_pow(float x, tvc y) { return rlw_pow(tvc(x), y); }
// the partials of entry k, exactly as the first form computed them at record time
__device__ __forceinline__ void p7c_partials(int op, float xa, float xb, double& pa, double& pb) {
    switch (op) {
    case 1: pa = 1.0; pb = 1.0; break;
    case 2: pa = 1.0; pb = -1.0; break;
    case 3: pa = (double)xb; pb = (double)xa; break;
    case 4: { double bv = (double)xb; pa = 1.0 / bv; pb = -(double)xa / (bv * bv); } break;
    case 5: pa = -1.0; pb = 0.0; break;
    case 6: pa = 1.0; pb = 0.0; break;
    default: { float r = rlw_pow(xa, xb); double xv = xa, yv = xb;
               pa = yv * pow(xv, yv - 1.0); pb = xv > 0.0 ? (double)r * log(xv) : 0.0; } break;
    }
}
namespace p7base_p7ac {
typedef tvc real;
struct Ctx;
struct Out {
    Ctx* c; int k; long long i;
    __device__ Out& operator=(tvc x);
    __device__ Out& operator=(float x) { return *this = tvc(x); }
    __device__ operator tvc() const;
};
struct Ctx {
    const float* fs; const float* wx; P7TapeC* t; float* vt; float* vf;
    int nout; int ok[8]; long long oi[8]; int oe[8]; float ov[8];
    __device__ tvc ld(long long i, int slot, int kind) {
        float v = kind ? wx[i] : fs[i];
        if (t->n >= P7_TMAX) { t->fail = 1; return tvc(v); }
        int k = t->n++;
        t->code[k] = slot << 10;
        return tvc(v, k, t);
    }
    __device__ Out out(int k, long long i) { return Out{this, k, i}; }
};
__device__ __forceinline__ Out& Out::operator=(tvc x) {
    int j = 0;
    for (; j < c->nout; ++j) if (c->ok[j] == k && c->oi[j] == i) break;
    if (j == c->nout) { if (c->nout >= 8) { c->t->fail = 2; return *this; } c->nout++; }
    c->ok[j] = k; c->oi[j] = i; c->oe[j] = x.i; c->ov[j] = x.v;
    if (c->vt) { if (k) c->vf[i] = x.v; else c->vt[i] = x.v; }
    return *this;
}
__device__ __forceinline__ Out::operator tvc() const {
    for (int j = 0; j < c->nout; ++j)
        if (c->ok[j] == k && c->oi[j] == i) return c->oe[j] >= 0 ? tvc(c->ov[j], c->oe[j], c->t) : tvc(c->ov[j]);
    c->t->fail = 3;
    return tvc(0.0f);
}
}
