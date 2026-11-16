// SPDX-License-Identifier: Apache-2.0
// g3-radland: mpas_sw_diffuse (rrtmg_sw_v842.py:3251, captured .cu 3c499c7b): out = total - direct. Linear.
// Refusal slot 0: out aliasing total or direct.
#include "sw.cu"
#include "g3_common.cuh"
extern "C" __global__ void mpas_sw_diffuse__tl(int n, const float* d_t, const float* d_d, float* d_o, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (d_o == d_t || d_o == d_d) g3_fail(failure, 0);
    d_o[i] = g3_st(G3D(d_t[i]) - G3D(d_d[i]));
}
extern "C" __global__ void mpas_sw_diffuse__ad(int n, float* b_t, float* b_d, float* b_o, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (b_o == b_t || b_o == b_d) { g3_fail(failure, 0); return; }
    double g = b_o[i];
    if (b_t == b_d) { g3_acc(b_t + i, 0.0); b_o[i] = 0.0f; return; }   // total == direct: out = 0 identically
    g3_acc(b_t + i, g); g3_acc(b_d + i, -g); b_o[i] = 0.0f;
}
