// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of mpas_cmbgb, the RRTMG LW g-point reduction of the k-coefficient tables
// (MPAS module_ra_rrtmg_lw.F:8331-9982, all 16 cmbgb routines, called once from rrtmg_lw_ini).
// Forward: captured source 683d7a07 lines 10564-10581 (image 34da9e09), launched from
// hexcore/mpasphys/rrtmg_lw_v842.py:2413-2415 (the table build, once per module; step 0 only).
// For band b (0-based), group g < ngc[b] and point p < points, with start = sum_{j<g} ngn[off+g'] and
// off = sum_{b'<b} ngc[b']:
//   out[o(g,p)] = sum_{j < ngn[off+g]} term_j,   term_j = raw[r(start+j,p)] * (weighted ? rwgt[16 b + start+j] : 1)
//   r(old,p) = first ? old + 16 p : p + points*old;   o(g,p) = first ? g + groups p : p + points g
// summed in the forward's order (f32, __fadd_rn / __fmul_rn).
// Active: raw (in), rwgt (in), out (out, full overwrite of points*ngc[b] words). band, points, first, weighted are
// launch configuration (inactive). raw is the RRTMG kg data file's coefficients and rwgt the mpas_init_rwgt weights:
// neither depends on model state, so in a forecast window both tangents are zero and so is out's. The entry is the
// exact derivative of the launch anyway (linear in raw, bilinear in raw and rwgt), so it stays exact if a table is
// ever made a control.
// No limiter, clamp, table index on state or state-dependent branch: no frozen-coefficient treatment (none applied).
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__f*_rn, built
// --fmad=false --ftz=false) and is bitwise to the forward image. __tl forms the partial sums in double and rounds
// once to f32 per written word. __ad: thread g (< groups) owns group g of the band for every point, so it is the only
// writer of every raw cotangent word of its olds, of rwgt[16 b + old] for its olds, and of out's group-g words; it
// walks the points in order (no atomics on data words, deterministic), accumulates in double, rounds once per word.
// failure[0] |= 1 on a non-finite partial or written word, |= 2 on a band outside 0..15.
#define CMB_AD(a, b) __fadd_rn((a), (b))
#define CMB_MU(a, b) __fmul_rn((a), (b))

// The forward's static tables (captured source 683d7a07 lines 10461-10484, MPAS rrlw_wvn ngc/ngn).
__device__ const int cmb_ngc[16] = {10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2};
__device__ const int cmb_ngn[140] = {
    1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 2, 2, 2, 2,
    3, 3, 1, 1, 1, 1, 2, 2, 4, 4, 3, 3, 4, 6, 8, 8, 8, 8, 4, 12};

struct CmbGroup { int groups, start, count, offset; };

__device__ __forceinline__ CmbGroup cmb_group(int band, int g) {
    CmbGroup c; c.groups = cmb_ngc[band]; c.offset = 0; c.start = 0;
    for (int b = 0; b < band; ++b) c.offset += cmb_ngc[b];
    for (int j = 0; j < g; ++j) c.start += cmb_ngn[c.offset + j];
    c.count = cmb_ngn[c.offset + g];
    return c;
}
__device__ __forceinline__ long long cmb_r(int first, int points, int old, int p) { return first ? (long long)old + 16LL * p : (long long)p + (long long)points * old; }
__device__ __forceinline__ long long cmb_o(int first, int points, int groups, int g, int p) { return first ? (long long)g + (long long)groups * p : (long long)p + (long long)points * g; }
__device__ __forceinline__ void cmb_flag(int* failure, double v) { if (!isfinite(v)) atomicOr(failure, 1); }

extern "C" __global__ void mpas_cmbgb__aug(int band, int points, int first, int weighted,
    const float* __restrict__ raw, const float* __restrict__ rwgt, float* __restrict__ out, int* __restrict__ failure)
{
    if (band < 0 || band > 15) { if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(failure, 2); return; }
    int idx = blockIdx.x * blockDim.x + threadIdx.x, groups = cmb_ngc[band];
    if (idx >= points * groups) return;
    int g = idx / points, p = idx % points;
    CmbGroup c = cmb_group(band, g);
    float sum = 0.0f;
    for (int j = 0; j < c.count; ++j) {
        int old = c.start + j;
        float term = raw[cmb_r(first, points, old, p)];
        if (weighted) term = CMB_MU(term, rwgt[band * 16 + old]);
        sum = CMB_AD(sum, term);
    }
    out[cmb_o(first, points, groups, g, p)] = sum;
    if (!isfinite(sum)) atomicOr(failure, 1);
}

extern "C" __global__ void mpas_cmbgb__tl(int band, int points, int first, int weighted,
    const float* __restrict__ raw, const float* __restrict__ rwgt,
    const float* __restrict__ d_raw, const float* __restrict__ d_rwgt, float* __restrict__ o_out, int* __restrict__ failure)
{
    if (band < 0 || band > 15) { if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(failure, 2); return; }
    int idx = blockIdx.x * blockDim.x + threadIdx.x, groups = cmb_ngc[band];
    if (idx >= points * groups) return;
    int g = idx / points, p = idx % points;
    CmbGroup c = cmb_group(band, g);
    double t = 0.0;
    for (int j = 0; j < c.count; ++j) {
        int old = c.start + j;
        long long r = cmb_r(first, points, old, p);
        if (weighted) t += (double)d_raw[r] * (double)rwgt[band * 16 + old] + (double)raw[r] * (double)d_rwgt[band * 16 + old];
        else t += (double)d_raw[r];
    }
    cmb_flag(failure, t);
    o_out[cmb_o(first, points, groups, g, p)] = (float)t;
}

// Reverse. out's seed is read and consumed in place (AD inout: full overwrite, no pre-launch cotangent). raw and
// rwgt accumulate: b_raw[r] += s * w, b_rwgt[16 b + old] += sum_p s(g,p) * raw[r(old,p)] (weighted launches only).
extern "C" __global__ void mpas_cmbgb__ad(int band, int points, int first, int weighted,
    const float* __restrict__ raw, const float* __restrict__ rwgt,
    float* __restrict__ b_raw, float* __restrict__ b_rwgt, float* __restrict__ b_out, int* __restrict__ failure)
{
    if (band < 0 || band > 15) { if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(failure, 2); return; }
    int g = blockIdx.x * blockDim.x + threadIdx.x, groups = cmb_ngc[band];
    if (g >= groups || points <= 0) return;
    CmbGroup c = cmb_group(band, g);
    double acc[16];
    for (int j = 0; j < c.count; ++j) acc[j] = 0.0;
    for (int p = 0; p < points; ++p) {
        long long o = cmb_o(first, points, groups, g, p);
        const double s = (double)b_out[o];
        for (int j = 0; j < c.count; ++j) {
            int old = c.start + j;
            long long r = cmb_r(first, points, old, p);
            const double w = weighted ? (double)rwgt[band * 16 + old] : 1.0;
            const double v = (double)b_raw[r] + s * w;
            cmb_flag(failure, v);
            b_raw[r] = (float)v;
            if (weighted) acc[j] += s * (double)raw[r];
        }
        b_out[o] = 0.0f;
    }
    if (weighted) for (int j = 0; j < c.count; ++j) {
        int w = band * 16 + c.start + j;
        const double v = (double)b_rwgt[w] + acc[j];
        cmb_flag(failure, v);
        b_rwgt[w] = (float)v;
    }
}
