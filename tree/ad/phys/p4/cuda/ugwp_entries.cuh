// SPDX-License-Identifier: Apache-2.0
// ad/phys P4: GSL orographic drag (bl_ugwp, hrrrv5 MPAS physics route) whole-launch derivative entries.
// Forward: hex tree/src/hexcore/mpasphys/ugwp_oro_v842.py (pin 16251589), captured source e1af1284
// (MPAS physics_noaa/UGWP/bl_ugwp.F, NOAA GSL, US government code). Same ABI as mynnsfc_entries.cuh.
// Terrain statistics (surface rows 6-25: var, oc1, oa, ol, varss ...) and geometry are read as
// inputs like every other row: they are active, and simply carry a zero seed when held fixed.

__device__ __forceinline__ ADF gfk_pow(const ADF &x, const ADF &y) { return p4_pow_of(gfk_pow(x.v, y.v), x, y); }
__device__ __forceinline__ ADF gfk_exp(const ADF &x) { return p4_exp_of(gfk_exp(x.v), x); }
__device__ __forceinline__ ADF glibc_sinf(const ADF &x) { return p4_sin_of(glibc_sinf(x.v), x); }

__device__ __forceinline__ void p4_ctx_reset(long long gslot) {
    P4Ctx &x = P4C();
    x.br = nullptr; x.br_words = 0; x.br_n = 0; x.br_stride = 0; x.gslot = gslot;
    x.ab = nullptr; x.w = nullptr; x.cnt = 0; x.cap = 0; x.stride = 0; x.overflow = 0; x.singular = 0;
}

// Input accessor: value from the forward array; TL tangent from d_ (float or double), REV leaf node.
struct P4In {
    const float *v; const float *t32; const double *t64; int id;
    __device__ __forceinline__ ADF operator[](size_t g) const {
#if defined(P4_MODE_TL)
        return ADF(v[g], t64 ? t64[g] : (t32 ? (double)t32[g] : 0.0));
#else
        return p4_leaf(v[g], id, (long long)g);
#endif
    }
};

// G4 frame arena: base set by the host per launch (module global); element k of a region for launch thread t at
// region + (k * nthreads + t) * sizeof(T), regions ordered 8-byte types first so every region stays aligned.
__device__ unsigned char *p4_frame_base;
template <class T> struct P4Col {
    T *p; long long s;
    __device__ __forceinline__ T &operator[](long long k) const { return p[k * s]; }
};
template <class T> __device__ __forceinline__ P4Col<T> p4_frame_col(long long off) {
    const long long nt = (long long)gridDim.x * blockDim.x, t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    return P4Col<T>{(T *)(p4_frame_base + off * nt) + t, nt};
}
#if defined(P4_MODE_TL)
typedef double P4TanT;
#else
typedef int P4TanT;
#endif
#define P4F_NARR 16
#define P4F_ADF 0LL
#define P4F_OTAN (P4F_ADF + (long long)P4F_NARR * (UGWP_LEVELS + 2) * (long long)sizeof(ADF))
#define P4F_STAN (P4F_OTAN + 12LL * UGWP_LEVELS * (long long)sizeof(P4TanT))
#define P4F_OVAL (P4F_STAN + 10LL * (long long)sizeof(P4TanT))
#define P4F_SVAL (P4F_OVAL + 12LL * UGWP_LEVELS * 4)
#define P4F_OWR (P4F_SVAL + 10LL * 4)
#define P4F_SWR (P4F_OWR + 12LL * UGWP_LEVELS)
#define P4F_BYTES (P4F_SWR + 10LL)
__device__ const unsigned long long p4_frame_bytes = (unsigned long long)((P4F_BYTES + 15) / 16 * 16);
// Output accessor: the column's slots live in registers/local memory; the forward array itself is
// written only by the augmented entry. slot = (g / plane) * L + (g % plane) / nc.
template <int CAP>
struct P4Out {
    float *fw;             // forward array (aug writes it; others read it only for read-before-write)
    size_t plane; int nc; int L;
    // G4: the slots live in the frame arena (P4Col views, entry-major across the launch threads), not on the stack.
    P4Col<float> val; P4Col<bool> wr;
#if defined(P4_MODE_TL)
    P4Col<double> tan;
#else
    P4Col<int> node;
#endif
    int rbw;               // reads of a slot this launch had not written yet (expected 0)
    __device__ __forceinline__ void init(float *fw_, size_t plane_, int nc_, int L_, long long tan_off, long long val_off, long long wr_off) {
#if defined(P4_MODE_TL)
        tan = p4_frame_col<double>(tan_off);
#else
        node = p4_frame_col<int>(tan_off);
#endif
        val = p4_frame_col<float>(val_off); wr = p4_frame_col<bool>(wr_off);
        fw = fw_; plane = plane_; nc = nc_; L = L_; rbw = 0;
        for (int s = 0; s < CAP; ++s) wr[s] = false;
    }
    __device__ __forceinline__ int slot(size_t g) const { return (int)(g / plane) * L + (int)((g % plane) / (size_t)nc); }
    struct Ref {
        P4Out *o; int s; size_t g;
        __device__ __forceinline__ operator ADF() const {
            if (!o->wr[s]) { o->rbw++; return ADF(o->fw[g]); }
#if defined(P4_MODE_TL)
            return ADF(o->val[s], o->tan[s]);
#else
            return ADF(o->val[s], o->node[s]);
#endif
        }
        __device__ __forceinline__ Ref &operator=(const ADF &a) {
            o->val[s] = a.v; o->wr[s] = true;
#if defined(P4_MODE_TL)
            o->tan[s] = a.t;
#else
            o->node[s] = a.i;
#endif
            return *this;
        }
        __device__ __forceinline__ Ref &operator=(const Ref &r) { return *this = (ADF)r; }
    };
    __device__ __forceinline__ Ref operator[](size_t g) { Ref r; r.o = this; r.s = slot(g); r.g = g; return r; }
};
typedef P4Out<12 * UGWP_LEVELS> P4OutO;
typedef P4Out<10> P4OutS;
//@@P4_SPLIT@@
// global index of output slot s of column i
__device__ __forceinline__ size_t p4_gidx(int s, int L, size_t plane, int nc, int i) {
    return (size_t)(s / L) * plane + (size_t)(s % L) * nc + i;
}

#if defined(P4_MODE_TL)
template <typename TT>
__device__ __forceinline__ void p4_ugwp_tl(const float *layers, const float *prsi0, const float *surface,
    const int *kpbl0, float *out, float *sout, int *status, float g, float cp, float rd, float rv, float fv,
    float pi_c, float dt, int nz, int nc, int diags, const TT *d_layers, const TT *d_prsi0, const TT *d_surface,
    TT *d_out, TT *d_sout, bool aug, unsigned *br, int br_words, int *br_count, int *singular, int *rbw) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    p4_ctx_reset(t);
    if (t >= nc) return;
    P4Ctx &x = P4C();
    if (br) { x.br = br; x.br_words = br_words; x.br_stride = nc; for (int k = 0; k < br_words; ++k) br[(long long)k * nc + t] = 0u; }
    P4In L_{layers, nullptr, nullptr, 0}, P_{prsi0, nullptr, nullptr, 1}, S_{surface, nullptr, nullptr, 2};
    if (sizeof(TT) == 8) { L_.t64 = (const double *)d_layers; P_.t64 = (const double *)d_prsi0; S_.t64 = (const double *)d_surface; }
    else { L_.t32 = (const float *)d_layers; P_.t32 = (const float *)d_prsi0; S_.t32 = (const float *)d_surface; }
    size_t n3 = (size_t)nz * nc;
    P4OutO O_; O_.init(out, n3, nc, nz, P4F_OTAN, P4F_OVAL, P4F_OWR);
    P4OutS S2; S2.init(sout, (size_t)nc, nc, 1, P4F_STAN, P4F_SVAL, P4F_SWR);
    adx_ugwp_oro_run_body(L_, P_, S_, kpbl0, O_, S2, status, g, cp, rd, rv, fv, pi_c, dt, nz, nc, diags, 0, nc);
    for (int s = 0; s < 12 * nz; ++s) if (O_.wr[s]) {
        size_t gi = p4_gidx(s, nz, n3, nc, t);
        if (aug) out[gi] = O_.val[s]; else d_out[gi] = (TT)O_.tan[s];
    }
    for (int s = 0; s < 10; ++s) if (S2.wr[s]) {
        size_t gi = (size_t)s * nc + t;
        if (aug) sout[gi] = S2.val[s]; else d_sout[gi] = (TT)S2.tan[s];
    }
    if (br_count) br_count[t] = x.br_n;
    if (singular && x.singular) atomicAdd(singular, x.singular);
    if (rbw && (O_.rbw || S2.rbw)) atomicAdd(rbw, O_.rbw + S2.rbw);
}
#define P4_UGWP_TL_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *layers, const float *prsi0, const float *surface, const int *kpbl0, \
    float *out, float *sout, int *status, float g, float cp, float rd, float rv, float fv, float pi_c, float dt, \
    int nz, int nc, int diags, const TT *d_layers, const TT *d_prsi0, const TT *d_surface, TT *d_out, TT *d_sout, \
    int *scratch_status, int *singular, int *rbw) { \
    p4_ugwp_tl<TT>(layers, prsi0, surface, kpbl0, out, sout, scratch_status, g, cp, rd, rv, fv, pi_c, dt, nz, nc, diags, \
                   d_layers, d_prsi0, d_surface, d_out, d_sout, false, nullptr, 0, nullptr, singular, rbw); }
P4_UGWP_TL_ENTRY(ugwp_oro_run__tl, float)
P4_UGWP_TL_ENTRY(ugwp_oro_run__tl64, double)
extern "C" __global__ void ugwp_oro_run__aug(const float *layers, const float *prsi0, const float *surface,
    const int *kpbl0, float *out, float *sout, int *status, float g, float cp, float rd, float rv, float fv,
    float pi_c, float dt, int nz, int nc, int diags, unsigned *br, int br_words, int *br_count, int *rbw) {
    p4_ugwp_tl<float>(layers, prsi0, surface, kpbl0, out, sout, status, g, cp, rd, rv, fv, pi_c, dt, nz, nc, diags,
                      nullptr, nullptr, nullptr, nullptr, nullptr, true, br, br_words, br_count, nullptr, rbw);
}
// ugwp_oro_dx: dx = length / density^0.25, density is mesh geometry: tangent 0.
template <typename TT> __device__ __forceinline__ void p4_dx_tl(TT *d_dx, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) d_dx[i] = (TT)0; }
extern "C" __global__ void ugwp_oro_dx__tl(const float *density, float *dx, float length, int n, float *d_dx) { p4_dx_tl<float>(d_dx, n); }
extern "C" __global__ void ugwp_oro_dx__tl64(const float *density, float *dx, float length, int n, double *d_dx) { p4_dx_tl<double>(d_dx, n); }
// ugwp_oro_diff: diff = after - before (rounded float subtraction; tangent exact difference).
template <typename TT> __device__ __forceinline__ void p4_diff_tl(const TT *d_b, const TT *d_a, TT *d_d, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) d_d[i] = (TT)((double)d_a[i] - (double)d_b[i]); }
extern "C" __global__ void ugwp_oro_diff__tl(const float *before, const float *after, float *diff, int n,
    const float *d_before, const float *d_after, float *d_diff) { p4_diff_tl<float>(d_before, d_after, d_diff, n); }
extern "C" __global__ void ugwp_oro_diff__tl64(const float *before, const float *after, float *diff, int n,
    const double *d_before, const double *d_after, double *d_diff) { p4_diff_tl<double>(d_before, d_after, d_diff, n); }
#endif  // P4_MODE_TL

#if defined(P4_MODE_REV)
template <typename TT>
__device__ __forceinline__ void p4_ugwp_ad(const float *layers, const float *prsi0, const float *surface,
    const int *kpbl0, float *out, float *sout, int *status, float g, float cp, float rd, float rv, float fv,
    float pi_c, float dt, int nz, int nc, int diags, TT *b_layers, TT *b_prsi0, TT *b_surface, TT *b_out, TT *b_sout,
    int2 *ab, double2 *w, double *adj, int cap, long long stride, int col0, int ncols,
    int *counts, int *overflow, int *singular) {
    int t = blockDim.x * blockIdx.x + threadIdx.x, i = col0 + t;
    p4_ctx_reset(t);
    if (t >= ncols || i >= nc) return;
    P4Ctx &x = P4C();
    x.ab = ab; x.w = w; x.cap = cap; x.stride = stride;
    P4In L_{layers, nullptr, nullptr, 0}, P_{prsi0, nullptr, nullptr, 1}, S_{surface, nullptr, nullptr, 2};
    size_t n3 = (size_t)nz * nc;
    P4OutO O_; O_.init(out, n3, nc, nz, P4F_OTAN, P4F_OVAL, P4F_OWR);
    P4OutS S2; S2.init(sout, (size_t)nc, nc, 1, P4F_STAN, P4F_SVAL, P4F_SWR);
    adx_ugwp_oro_run_body(L_, P_, S_, kpbl0, O_, S2, status, g, cp, rd, rv, fv, pi_c, dt, nz, nc, diags, col0, ncols);
    if (counts) counts[i] = x.cnt;
    if (singular && x.singular) atomicAdd(singular, x.singular);
    if (!ab) return;
    if (x.overflow) { atomicExch(overflow, 1); return; }
    int cnt = x.cnt;
    for (int k = 0; k < cnt; ++k) adj[(long long)k * stride + t] = 0.0;
    for (int s = 0; s < 12 * nz; ++s) if (O_.wr[s]) {
        size_t gi = p4_gidx(s, nz, n3, nc, i);
        if (O_.node[s] >= 0) adj[(long long)O_.node[s] * stride + t] += (double)b_out[gi];
        b_out[gi] = (TT)0;
    }
    for (int s = 0; s < 10; ++s) if (S2.wr[s]) {
        size_t gi = (size_t)s * nc + i;
        if (S2.node[s] >= 0) adj[(long long)S2.node[s] * stride + t] += (double)b_sout[gi];
        b_sout[gi] = (TT)0;
    }
    TT *bin[3] = {b_layers, b_prsi0, b_surface};
    for (int k = cnt - 1; k >= 0; --k) {
        long long o = (long long)k * stride + t;
        double a = adj[o];
        if (a == 0.0) continue;
        int2 p = ab[o]; double2 ww = w[o];
        if (p.x <= -2) { TT *b = bin[-2 - p.x]; size_t gi = (size_t)ww.x; b[gi] = (TT)((double)b[gi] + a); continue; }
        adj[(long long)p.x * stride + t] += ww.x * a;
        if (p.y >= 0) adj[(long long)p.y * stride + t] += ww.y * a;
    }
}
#define P4_UGWP_AD_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *layers, const float *prsi0, const float *surface, const int *kpbl0, \
    float *out, float *sout, int *status, float g, float cp, float rd, float rv, float fv, float pi_c, float dt, \
    int nz, int nc, int diags, TT *b_layers, TT *b_prsi0, TT *b_surface, TT *b_out, TT *b_sout, \
    int2 *ab, double2 *w, double *adj, int cap, long long stride, int col0, int ncols, int *counts, int *overflow, \
    int *singular, int *scratch_status) { \
    p4_ugwp_ad<TT>(layers, prsi0, surface, kpbl0, out, sout, scratch_status, g, cp, rd, rv, fv, pi_c, dt, nz, nc, diags, \
                   b_layers, b_prsi0, b_surface, b_out, b_sout, ab, w, adj, cap, stride, col0, ncols, counts, overflow, singular); }
P4_UGWP_AD_ENTRY(ugwp_oro_run__ad, float)
P4_UGWP_AD_ENTRY(ugwp_oro_run__ad64, double)
template <typename TT> __device__ __forceinline__ void p4_dx_ad(TT *b_dx, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) b_dx[i] = (TT)0; }
extern "C" __global__ void ugwp_oro_dx__ad(const float *density, float *dx, float length, int n, float *b_dx) { p4_dx_ad<float>(b_dx, n); }
extern "C" __global__ void ugwp_oro_dx__ad64(const float *density, float *dx, float length, int n, double *b_dx) { p4_dx_ad<double>(b_dx, n); }
template <typename TT> __device__ __forceinline__ void p4_diff_ad(TT *b_b, TT *b_a, TT *b_d, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) { double d = (double)b_d[i]; b_a[i] = (TT)((double)b_a[i] + d); b_b[i] = (TT)((double)b_b[i] - d); b_d[i] = (TT)0; } }
extern "C" __global__ void ugwp_oro_diff__ad(const float *before, const float *after, float *diff, int n,
    float *b_before, float *b_after, float *b_diff) { p4_diff_ad<float>(b_before, b_after, b_diff, n); }
extern "C" __global__ void ugwp_oro_diff__ad64(const float *before, const float *after, float *diff, int n,
    double *b_before, double *b_after, double *b_diff) { p4_diff_ad<double>(b_before, b_after, b_diff, n); }
#endif  // P4_MODE_REV
