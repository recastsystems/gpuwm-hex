// SPDX-License-Identifier: Apache-2.0
// L2 binding of the RUC land whole launch (oracle_land_ruc_land) as an ad.scheme.rows/v1 row (schemes.rs, kind
// ruc_land; lane ad/step6-rucland). Data movement only, no field arithmetic: the derivative is the unchanged entry of
// ad/held-oracle_land_ruc_land (oracle_land_ruc_land__tl / __ad), which works in place on float32 tangent arrays.
//
// One segment per float field: `sh` = its float64 window shadow (one element per forward float32 word), `t` = the entry's
// float32 tangent array, `n` = words. The shadow words are float32 words widened by the window (xfer), so the narrowing
// is exact, and so is the widening of the entry's float32 results. Grid: x strides the words, y is the segment.
struct RlbSeg { unsigned long long sh, t; long long n; };

extern "C" __global__ void rlb_narrow(const RlbSeg *s) {
    const RlbSeg g = s[blockIdx.y];
    const double *a = reinterpret_cast<const double *>(g.sh);
    float *b = reinterpret_cast<float *>(g.t);
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < g.n; i += (long long)gridDim.x * blockDim.x)
        b[i] = (float)a[i];
}

extern "C" __global__ void rlb_widen(const RlbSeg *s) {
    const RlbSeg g = s[blockIdx.y];
    double *a = reinterpret_cast<double *>(g.sh);
    const float *b = reinterpret_cast<const float *>(g.t);
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < g.n; i += (long long)gridDim.x * blockDim.x)
        a[i] = (double)b[i];
}

// Primal identity: `sh` = an entry field array after its call (AF: 16 bytes per word, AR: 8 bytes per word; the float32
// primal at byte 0, so `stride` = 4 or 2 words), `t` = the forward's live post-state words. Counts the words whose bits differ.
extern "C" __global__ void rlb_cmp(const RlbSeg *s, unsigned long long *diff, int stride) {
    const RlbSeg g = s[blockIdx.y];
    const unsigned int *a = reinterpret_cast<const unsigned int *>(g.sh);
    const unsigned int *b = reinterpret_cast<const unsigned int *>(g.t);
    unsigned long long d = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < g.n; i += (long long)gridDim.x * blockDim.x)
        d += a[(long long)stride * i] != b[i];
    if (d) atomicAdd(diff, d);
}

// Per-launch dot directions for the float32 entry: round each float64 word to its float32 value, in place.
extern "C" __global__ void rlb_round(double *a, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (double)(float)a[i];
}
