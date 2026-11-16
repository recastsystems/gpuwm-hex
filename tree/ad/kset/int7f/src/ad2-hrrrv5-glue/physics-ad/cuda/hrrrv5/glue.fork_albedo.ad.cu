// SPDX-License-Identifier: Apache-2.0
// Derivative of fork_albedo, surface_rrfs.py:41-46 at pin 16251589 (update_surface.F:112-120):
//   v = (alb[m2+12i]*w2 + alb[m1+12i]*w1) / span;  bg[i] = land[i]==0 ? 0.08 : v/100
// Written for ad2-hrrrv5-glue (assignment 7), in the layout of glue.fork_interp.ad.cu.
// Calendar indices, weights, span and the integer land mask are fixed (schedule / static mask).
// Water cells write a constant: zero tangent, and the reverse kills bg's cotangent without contribution.
// One thread owns the twelve months of one cell, so the transpose has no conflicting writes, including m1 == m2.

__device__ __forceinline__ bool glue_albedo_valid(int n, int m1, int m2, int span, int* failure) {
    if (n >= 0 && m1 >= 0 && m1 < 12 && m2 >= 0 && m2 < 12 && span != 0) return true;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(failure + 0, 1);
    return false;
}

extern "C" __global__ void fork_albedo__aug(int n, int m1, int m2, int w1, int w2, int span,
                                            const float* alb, const int* land, float* bg, int* failure) {
    if (!glue_albedo_valid(n, m1, m2, span, failure)) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __fdiv_rn(__fadd_rn(__fmul_rn(alb[m2 + 12 * i], float(w2)),
                                  __fmul_rn(alb[m1 + 12 * i], float(w1))), float(span));
    bg[i] = land[i] == 0 ? 0.08f : __fdiv_rn(v, 100.f);
    if (!isfinite(bg[i])) atomicExch(failure + 1, 1);
}

extern "C" __global__ void fork_albedo__tl(int n, int m1, int m2, int w1, int w2, int span,
                                           const float* alb, const int* land, const float* dalb,
                                           float* dbg, int* failure) {
    if (!glue_albedo_valid(n, m1, m2, span, failure)) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (land[i] == 0) { dbg[i] = 0.0f; return; }
    const int p = m2 + 12 * i, q = m1 + 12 * i;
    float value = (float)((((double)w2 * (double)dalb[p] + (double)w1 * (double)dalb[q]) / (double)span) / 100.0);
    dbg[i] = value;
    if (!isfinite(alb[p]) || !isfinite(alb[q]) || !isfinite(value)) atomicExch(failure + 1, 1);
}

extern "C" __global__ void fork_albedo__ad(int n, int m1, int m2, int w1, int w2, int span,
                                           const float* alb, const int* land, float* balb,
                                           float* bbg, int* failure) {
    if (!glue_albedo_valid(n, m1, m2, span, failure)) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (land[i] != 0) {
        const int p = m2 + 12 * i, q = m1 + 12 * i;
        const double b = (double)bbg[i] / 100.0 / (double)span;
        if (!isfinite(alb[p]) || !isfinite(alb[q]) || !isfinite(bbg[i])) atomicExch(failure + 1, 1);
        if (p == q) {
            balb[p] = (float)((double)balb[p] + b * ((double)w2 + (double)w1));
            if (!isfinite(balb[p])) atomicExch(failure + 1, 1);
        } else {
            balb[p] = (float)((double)balb[p] + b * (double)w2);
            balb[q] = (float)((double)balb[q] + b * (double)w1);
            if (!isfinite(balb[p]) || !isfinite(balb[q])) atomicExch(failure + 1, 1);
        }
    }
    bbg[i] = 0.0f;
}
