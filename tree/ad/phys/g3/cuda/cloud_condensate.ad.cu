// SPDX-License-Identifier: Apache-2.0
// g3-radland: cloud_condensate (radiation_rrfs.py:76, captured .cu 4a88150e): if (qc < 1e-6 && cf > .001) qc += added_c,
// the same for qi with 1e-9. Frozen gate on the pre-launch qc/qi/cf words (cf enters only through the gate).
// Refusal slot 0: an added array aliasing qc or qi.
#include "cloud.cu"
#include "g3_common.cuh"
extern "C" __global__ void cloud_condensate__tl(int n, const float* __restrict__ cf, const float* __restrict__ qc,
    const float* __restrict__ qi, const float* d_ac, const float* d_ai, float* d_qc, float* d_qi, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (d_ac == d_qc || d_ai == d_qi || d_ac == d_qi || d_ai == d_qc) g3_fail(failure, 0);
    if (qc[i] < 1.e-6f && cf[i] > .001f) d_qc[i] = g3_st(G3D(d_qc[i]) + G3D(d_ac[i]));
    if (qi[i] < 1.e-9f && cf[i] > .001f) d_qi[i] = g3_st(G3D(d_qi[i]) + G3D(d_ai[i]));
}
extern "C" __global__ void cloud_condensate__ad(int n, const float* __restrict__ cf, const float* __restrict__ qc,
    const float* __restrict__ qi, float* b_ac, float* b_ai, float* b_qc, float* b_qi, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (b_ac == b_qc || b_ai == b_qi || b_ac == b_qi || b_ai == b_qc) { g3_fail(failure, 0); return; }
    if (qc[i] < 1.e-6f && cf[i] > .001f) g3_acc(b_ac + i, b_qc[i]);
    if (qi[i] < 1.e-9f && cf[i] > .001f) g3_acc(b_ai + i, b_qi[i]);
}
