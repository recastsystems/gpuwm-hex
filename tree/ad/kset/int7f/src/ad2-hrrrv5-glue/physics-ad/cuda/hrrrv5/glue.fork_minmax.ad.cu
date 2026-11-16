// SPDX-License-Identifier: Apache-2.0
// surface_rrfs.py:49-54: first month wins a tie; comparisons execute in order.

__device__ __forceinline__ void glue_minmax_select(
    const float* f, int i, int* imin, int* imax, unsigned int* branches,
    int* failure) {
    const int base = 12 * i;
    float mn = f[base], mx = mn;
    int il = 0, ih = 0;
    if (!isfinite(mn)) atomicExch(failure, 1);
    for (int k = 1; k < 12; ++k) {
        const float v = f[base + k];
        if (!isfinite(v)) atomicExch(failure, 1);
        const bool lo = v < mn;
        atomicAdd(branches + 6 + (lo ? 1 : 0), 1u);
        if (lo) { mn = v; il = k; }
        const bool hi = v > mx;
        atomicAdd(branches + 10 + (hi ? 1 : 0), 1u);
        if (hi) { mx = v; ih = k; }
    }
    *imin = il;
    *imax = ih;
}

extern "C" __global__ void fork_minmax__aug(
    int n, const float* f, float* lo, float* hi, unsigned int* branches,
    int* failure) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int il, ih;
    glue_minmax_select(f, i, &il, &ih, branches, failure);
    lo[i] = f[12 * i + il];
    hi[i] = f[12 * i + ih];
}

extern "C" __global__ void fork_minmax__tl(
    int n, const float* f, const float* df, float* dlo, float* dhi,
    unsigned int* branches, int* failure) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int il, ih;
    glue_minmax_select(f, i, &il, &ih, branches, failure);
    const float a = df[12 * i + il], b = df[12 * i + ih];
    dlo[i] = a;
    dhi[i] = b;
    if (!isfinite(a) || !isfinite(b)) atomicExch(failure, 1);
}

extern "C" __global__ void fork_minmax__ad(
    int n, const float* f, float* bf, float* blo, float* bhi,
    unsigned int* branches, int* failure) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int il, ih;
    glue_minmax_select(f, i, &il, &ih, branches, failure);
    const int pl = 12 * i + il, ph = 12 * i + ih;
    const float l = blo[i], h = bhi[i];
    if (!isfinite(l) || !isfinite(h) || !isfinite(bf[pl]) || !isfinite(bf[ph]))
        atomicExch(failure, 1);
    if (pl == ph) bf[pl] = (float)((double)bf[pl] + (double)l + (double)h);
    else {
        bf[pl] = (float)((double)bf[pl] + (double)l);
        bf[ph] = (float)((double)bf[ph] + (double)h);
    }
    if (!isfinite(bf[pl]) || !isfinite(bf[ph])) atomicExch(failure, 1);
    blo[i] = 0.0f;
    bhi[i] = 0.0f;
}
