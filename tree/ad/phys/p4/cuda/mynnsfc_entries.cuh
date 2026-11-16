// SPDX-License-Identifier: Apache-2.0
// ad/phys P4: MYNN surface-layer whole-launch derivative entries (hrrrv5, MPAS physics route).
// Forward: hex tree/src/hexcore/mpasphys/rrfs/sf_mynnsfc_v600.py (pin 16251589), captured
// source 9c87dce8 (RRFS MYNN-SFC @1d6e370e, NOAA GSL, US government code).
// Field layout, IDX_* and BLEND_FIELD come from the verbatim forward source above.
//
// Entry ABI (FRAMEWORK naming; TT = float for the plain entry, double for the *64 entry):
//   K__tl : forward inputs + d_<arg> (inout arrays: d_ holds pre tangent, receives post tangent).
//           Never writes forward outputs.
//   K__aug: the forward through the derivative types; writes the forward outputs (identity proof).
//   K__ad : forward inputs (pre-state) + b_<arg>; b_in += J^T b_out, overwritten outputs zeroed;
//           inout arrays: b_ holds the post cotangent and receives the pre cotangent.
//           The column is recomputed from the pre-state into a per-call workspace tape.

// ---- libm of the forward, by value; partial from the true function ----
__device__ __forceinline__ ADF mynn_glibc_logf(const ADF &x) { return p4_log_of(mynn_glibc_logf(x.v), x); }
__device__ __forceinline__ ADF mynn_glibc_expf(const ADF &x) { return p4_exp_of(mynn_glibc_expf(x.v), x); }
__device__ __forceinline__ ADF mynn_glibc_powf(const ADF &x, const ADF &y) { return p4_pow_of(mynn_glibc_powf(x.v, y.v), x, y); }
__device__ __forceinline__ ADF glibc_atanf(const ADF &x) { return p4_atan_of(glibc_atanf(x.v), x); }

__device__ __forceinline__ void p4_ctx_reset(long long gslot) {
    P4Ctx &x = P4C();
    x.br = nullptr; x.br_words = 0; x.br_n = 0; x.br_stride = 0; x.gslot = gslot;
    x.ab = nullptr; x.w = nullptr; x.cnt = 0; x.cap = 0; x.stride = 0; x.overflow = 0; x.singular = 0;
}
//@@P4_SPLIT@@

#if defined(P4_MODE_TL)
// TL and augmented forward of mynnsfc_columns.
template <typename TT>
__device__ __forceinline__ void p4_sfc_tl(const float *f, TT *d_f, float *f_out, int *errors_out,
    const int *options, const int *vegtype, float xice_threshold, unsigned long long present, int n,
    int step_override, int init_override, unsigned *br, int br_words, int *br_count, int *singular) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    p4_ctx_reset(c);
    if (c >= n) return;
    P4Ctx &x = P4C();
    if (br) { x.br = br; x.br_words = br_words; x.br_stride = n; for (int k = 0; k < br_words; ++k) br[(long long)k * n + c] = 0u; }
    ADF fl[FIELD_COUNT];
    for (int j = 0; j < FIELD_COUNT; ++j) fl[j] = ADF(f[j * n + c], d_f ? (double)d_f[j * n + c] : 0.0);
    int error = 0;
    p4_mynnsfc_column(fl, options, vegtype, error, xice_threshold, present, n, c, step_override, init_override);
    for (int j = 0; j < FIELD_COUNT; ++j) {
        if (d_f) d_f[j * n + c] = (TT)fl[j].t;
        if (f_out) f_out[j * n + c] = fl[j].v;
    }
    if (errors_out) errors_out[c] = error;
    if (br_count) br_count[c] = x.br_n;
    if (singular && x.singular) atomicAdd(singular, x.singular);
}
#define P4_SFC_TL_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *f, const int *options, const int *vegtype, float xice_threshold, \
    unsigned long long present, int n, int step_override, int init_override, TT *d_f, int *singular) { \
    p4_sfc_tl<TT>(f, d_f, nullptr, nullptr, options, vegtype, xice_threshold, present, n, step_override, init_override, \
                  nullptr, 0, nullptr, singular); }
P4_SFC_TL_ENTRY(mynnsfc_columns__tl, float)
P4_SFC_TL_ENTRY(mynnsfc_columns__tl64, double)
// augmented forward: writes f_out (may alias f) and errors, plus an optional branch record
extern "C" __global__ void mynnsfc_columns__aug(const float *f, float *f_out, int *errors, const int *options,
    const int *vegtype, float xice_threshold, unsigned long long present, int n, int step_override, int init_override,
    unsigned *br, int br_words, int *br_count) {
    p4_sfc_tl<float>(f, nullptr, f_out, errors, options, vegtype, xice_threshold, present, n, step_override,
                     init_override, br, br_words, br_count, nullptr);
}

// mynnsfc_from_mpas: dx is geometry (meshDensity), so its tangent is 0; sea = copy of f, then the
// fractional-ice overrides (constants, and tsk = fmaxf(sst, 271.4): executed arm, tie -> sst).
template <typename TT>
__device__ __forceinline__ void p4_from_tl(TT *d_f, TT *d_sea, const float *sst, const TT *d_sst, const float *ice,
                                           float threshold, int fraction, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_f[IDX_dx * n + i] = (TT)0;
    if (fraction) {
        for (int j = 0; j < FIELD_COUNT; ++j) d_sea[j * n + i] = d_f[j * n + i];
        d_sea[IDX_qsfc * n + i] = (TT)0;
        if (ice[i] >= threshold && ice[i] < 1.0f) {
            d_sea[IDX_xland * n + i] = (TT)0; d_f[IDX_xland * n + i] = (TT)0;
            d_sea[IDX_mavail * n + i] = (TT)0; d_sea[IDX_znt * n + i] = (TT)0;
            bool left = !isnan(sst[i]) && sst[i] >= 271.4f;
            d_sea[IDX_tsk * n + i] = left ? d_sst[i] : (TT)0;
        }
    }
}
#define P4_FROM_TL_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *f, const float *sea, const float *density, const float *sst, const float *ice, \
    float len_disp, float threshold, int fraction, int n, TT *d_f, TT *d_sea, const TT *d_sst) { \
    p4_from_tl<TT>(d_f, d_sea, sst, d_sst, ice, threshold, fraction, n); }
P4_FROM_TL_ENTRY(mynnsfc_from_mpas__tl, float)
P4_FROM_TL_ENTRY(mynnsfc_from_mpas__tl64, double)

// mynnsfc_to_mpas: f = f*x + (1-x)*sea on BLEND fields where threshold <= ice < 1; x = ice is active.
template <typename TT>
__device__ __forceinline__ void p4_to_tl(const float *f, const float *sea, const float *ice, float threshold, int n,
                                         TT *d_f, const TT *d_sea, const TT *d_ice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ice[i] < threshold || ice[i] >= 1.0f) return;
    double x = ice[i], y = (double)__fsub_rn(1.0f, ice[i]), dx = d_ice ? (double)d_ice[i] : 0.0;
    for (int j = 0; j < FIELD_COUNT; ++j)
        if (BLEND_FIELD(j)) {
            double df = (double)d_f[j * n + i], ds = (double)d_sea[j * n + i];
            d_f[j * n + i] = (TT)(df * x + (double)f[j * n + i] * dx - dx * (double)sea[j * n + i] + y * ds);
        }
}
#define P4_TO_TL_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *f, const float *sea, const float *ice, float threshold, int n, \
    TT *d_f, const TT *d_sea, const TT *d_ice) { p4_to_tl<TT>(f, sea, ice, threshold, n, d_f, d_sea, d_ice); }
P4_TO_TL_ENTRY(mynnsfc_to_mpas__tl, float)
P4_TO_TL_ENTRY(mynnsfc_to_mpas__tl64, double)
#endif  // P4_MODE_TL

#if defined(P4_MODE_REV)
// Reverse sweep over this thread's tape. Leaves (a <= -2) are the 68 loaded fields (nodes 0..67).
__device__ __forceinline__ void p4_sweep(const int2 *ab, const double2 *w, double *adj, int cnt, long long stride, int t) {
    for (int k = cnt - 1; k >= 0; --k) {
        long long o = (long long)k * stride + t;
        double a = adj[o];
        if (a == 0.0) continue;
        int2 p = ab[o];
        if (p.x <= -2) continue;
        double2 ww = w[o];
        adj[(long long)p.x * stride + t] += ww.x * a;
        if (p.y >= 0) adj[(long long)p.y * stride + t] += ww.y * a;
    }
}
template <typename TT>
__device__ __forceinline__ void p4_sfc_ad(const float *f, TT *b_f, const int *options, const int *vegtype,
    float xice_threshold, unsigned long long present, int n, int step_override, int init_override,
    int2 *ab, double2 *w, double *adj, int cap, long long stride, int col0, int ncols,
    int *counts, int *overflow, int *singular) {
    int t = blockIdx.x * blockDim.x + threadIdx.x, c = col0 + t;
    p4_ctx_reset(t);
    if (t >= ncols || c >= n) return;
    P4Ctx &x = P4C();
    x.ab = ab; x.w = w; x.cap = cap; x.stride = stride;
    ADF fl[FIELD_COUNT];
    for (int j = 0; j < FIELD_COUNT; ++j) fl[j] = p4_leaf(f[j * n + c], 0, j);
    int error = 0;
    p4_mynnsfc_column(fl, options, vegtype, error, xice_threshold, present, n, c, step_override, init_override);
    if (counts) counts[c] = x.cnt;
    if (singular && x.singular) atomicAdd(singular, x.singular);
    if (!ab) return;
    if (x.overflow) { atomicExch(overflow, 1); return; }
    int cnt = x.cnt;
    for (int k = 0; k < cnt; ++k) adj[(long long)k * stride + t] = 0.0;
    for (int j = 0; j < FIELD_COUNT; ++j)
        if (fl[j].i >= 0) adj[(long long)fl[j].i * stride + t] += (double)b_f[j * n + c];
    p4_sweep(ab, w, adj, cnt, stride, t);
    for (int j = 0; j < FIELD_COUNT; ++j) b_f[j * n + c] = (TT)adj[(long long)j * stride + t];
}
#define P4_SFC_AD_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *f, const int *options, const int *vegtype, float xice_threshold, \
    unsigned long long present, int n, int step_override, int init_override, TT *b_f, \
    int2 *ab, double2 *w, double *adj, int cap, long long stride, int col0, int ncols, int *counts, int *overflow, int *singular) { \
    p4_sfc_ad<TT>(f, b_f, options, vegtype, xice_threshold, present, n, step_override, init_override, \
                  ab, w, adj, cap, stride, col0, ncols, counts, overflow, singular); }
P4_SFC_AD_ENTRY(mynnsfc_columns__ad, float)
P4_SFC_AD_ENTRY(mynnsfc_columns__ad64, double)

template <typename TT>
__device__ __forceinline__ void p4_from_ad(TT *b_f, TT *b_sea, const float *sst, TT *b_sst, const float *ice,
                                           float threshold, int fraction, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (fraction) {
        if (ice[i] >= threshold && ice[i] < 1.0f) {
            bool left = !isnan(sst[i]) && sst[i] >= 271.4f;
            if (left) b_sst[i] = (TT)((double)b_sst[i] + (double)b_sea[IDX_tsk * n + i]);
            b_sea[IDX_tsk * n + i] = (TT)0; b_sea[IDX_znt * n + i] = (TT)0; b_sea[IDX_mavail * n + i] = (TT)0;
            b_f[IDX_xland * n + i] = (TT)0; b_sea[IDX_xland * n + i] = (TT)0;
        }
        b_sea[IDX_qsfc * n + i] = (TT)0;
        for (int j = 0; j < FIELD_COUNT; ++j) {
            b_f[j * n + i] = (TT)((double)b_f[j * n + i] + (double)b_sea[j * n + i]);
            b_sea[j * n + i] = (TT)0;
        }
    }
    b_f[IDX_dx * n + i] = (TT)0;
}
#define P4_FROM_AD_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *f, const float *sea, const float *density, const float *sst, const float *ice, \
    float len_disp, float threshold, int fraction, int n, TT *b_f, TT *b_sea, TT *b_sst) { \
    p4_from_ad<TT>(b_f, b_sea, sst, b_sst, ice, threshold, fraction, n); }
P4_FROM_AD_ENTRY(mynnsfc_from_mpas__ad, float)
P4_FROM_AD_ENTRY(mynnsfc_from_mpas__ad64, double)

template <typename TT>
__device__ __forceinline__ void p4_to_ad(const float *f, const float *sea, const float *ice, float threshold, int n,
                                         TT *b_f, TT *b_sea, TT *b_ice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ice[i] < threshold || ice[i] >= 1.0f) return;
    double x = ice[i], y = (double)__fsub_rn(1.0f, ice[i]), bi = 0.0;
    for (int j = 0; j < FIELD_COUNT; ++j)
        if (BLEND_FIELD(j)) {
            double bf = (double)b_f[j * n + i];
            bi += bf * ((double)f[j * n + i] - (double)sea[j * n + i]);
            b_sea[j * n + i] = (TT)((double)b_sea[j * n + i] + y * bf);
            b_f[j * n + i] = (TT)(x * bf);
        }
    if (b_ice) b_ice[i] = (TT)((double)b_ice[i] + bi);
}
#define P4_TO_AD_ENTRY(NAME, TT) \
extern "C" __global__ void NAME(const float *f, const float *sea, const float *ice, float threshold, int n, \
    TT *b_f, TT *b_sea, TT *b_ice) { p4_to_ad<TT>(f, sea, ice, threshold, n, b_f, b_sea, b_ice); }
P4_TO_AD_ENTRY(mynnsfc_to_mpas__ad, float)
P4_TO_AD_ENTRY(mynnsfc_to_mpas__ad64, double)
#endif  // P4_MODE_REV
