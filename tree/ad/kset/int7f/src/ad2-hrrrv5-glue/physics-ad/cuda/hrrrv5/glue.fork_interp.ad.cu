// SPDX-License-Identifier: Apache-2.0
// Derivative of surface_rrfs.py:36-40, with the calendar held discrete.
// One thread owns all twelve months of one cell, so the transpose has no
// conflicting writes, including when both selected month indices coincide.

__device__ __forceinline__ bool glue_interp_valid(int n, int m1, int m2, int span,
                                                   int* failure) {
    if (n >= 0 && m1 >= 0 && m1 < 12 && m2 >= 0 && m2 < 12 && span != 0) return true;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(failure + 0, 1);
    return false;
}

extern "C" __global__ void fork_interp__aug(
    int n, int m1, int m2, int w1, int w2, int span,
    const float* f, float* out, int* failure) {
    if (!glue_interp_valid(n, m1, m2, span, failure)) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = f[m2 + 12 * i], b = f[m1 + 12 * i];
    const float value = __fdiv_rn(__fadd_rn(__fmul_rn(a, (float)w2),
                                            __fmul_rn(b, (float)w1)), (float)span);
    out[i] = value;
    if (!isfinite(a) || !isfinite(b) || !isfinite(value)) atomicExch(failure + 1, 1);
}

extern "C" __global__ void fork_interp__tl(
    int n, int m1, int m2, int w1, int w2, int span,
    const float* f, const float* df, float* dout, int* failure) {
    if (!glue_interp_valid(n, m1, m2, span, failure)) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = m2 + 12 * i, q = m1 + 12 * i;
    const double fw1 = (double)(float)w1, fw2 = (double)(float)w2;
    const double fs = (double)(float)span;
    const float value = (float)((fw2 * (double)df[p] + fw1 * (double)df[q]) / fs);
    dout[i] = value;
    if (!isfinite(f[p]) || !isfinite(f[q]) || !isfinite(df[p]) ||
        !isfinite(df[q]) || !isfinite(value)) atomicExch(failure + 1, 1);
}

extern "C" __global__ void fork_interp__ad(
    int n, int m1, int m2, int w1, int w2, int span,
    const float* f, float* bf, float* bout, int* failure) {
    if (!glue_interp_valid(n, m1, m2, span, failure)) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = m2 + 12 * i, q = m1 + 12 * i;
    const double b = (double)bout[i];
    const double fw1 = (double)(float)w1, fw2 = (double)(float)w2;
    const double fs = (double)(float)span;
    if (!isfinite(f[p]) || !isfinite(f[q]) || !isfinite(bout[i]) ||
        !isfinite(bf[p]) || !isfinite(bf[q])) atomicExch(failure + 1, 1);
    if (p == q) {
        bf[p] = (float)((double)bf[p] + b * (fw2 + fw1) / fs);
        if (!isfinite(bf[p])) atomicExch(failure + 1, 1);
    } else {
        bf[p] = (float)((double)bf[p] + b * fw2 / fs);
        bf[q] = (float)((double)bf[q] + b * fw1 / fs);
        if (!isfinite(bf[p]) || !isfinite(bf[q])) atomicExch(failure + 1, 1);
    }
    bout[i] = 0.0f;
}
