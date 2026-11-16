// SPDX-License-Identifier: Apache-2.0
// g3-radland: profile_from_mpas (radshared_v842.py:407, captured .cu 9b6ced0f): dst[i] = src[(i%nz)*ntotal + i/nz], the
// [level, cell] -> [cell, level] gather. A permutation onto dst (injective in src while nc <= ntotal): TL copy, AD
// b_src[j] += b_dst[i], b_dst[i] = 0. Refusal slot 0: nc > ntotal (two dst words read one src word) or src == dst.
#include "radshared.cu"
#include "g3_common.cuh"
extern "C" __global__ void profile_from_mpas__tl(int nc, int nz, int ntotal, const float* d_src, float* d_dst, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nc * nz) return;
    if (nc > ntotal || d_src == d_dst) g3_fail(failure, 0);
    d_dst[i] = d_src[(size_t)(i % nz) * ntotal + i / nz];
}
extern "C" __global__ void profile_from_mpas__ad(int nc, int nz, int ntotal, float* b_src, float* b_dst, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nc * nz) return;
    if (nc > ntotal || b_src == b_dst) { g3_fail(failure, 0); return; }
    g3_acc(b_src + (size_t)(i % nz) * ntotal + i / nz, b_dst[i]); b_dst[i] = 0.0f;
}
