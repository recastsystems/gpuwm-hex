// SPDX-License-Identifier: Apache-2.0
// g3-radland: degrees (radiation_rrfs.py:29, captured .cu 4a88150e): out = radians / degrad. Linear.
#include "cloud.cu"
#include "g3_common.cuh"
extern "C" __global__ void degrees__tl(int n, float degrad, const float* __restrict__ d_rad, float* d_out, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    d_out[i] = g3_st(G3D(d_rad[i]) / G3D(degrad));            // in place (d_rad == d_out) reads before it writes
}
extern "C" __global__ void degrees__ad(int n, float degrad, float* b_rad, float* b_out, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    double g = G3D(b_out[i]) / G3D(degrad);
    if (b_rad == b_out) { b_out[i] = g3_st(g); return; }       // in place: b_old = b_new / degrad
    g3_acc(b_rad + i, g); b_out[i] = 0.0f;
}
