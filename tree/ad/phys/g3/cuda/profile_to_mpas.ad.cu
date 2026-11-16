// SPDX-License-Identifier: Apache-2.0
// g3-radland: profile_to_mpas (radshared_v842.py:413, captured .cu 9b6ced0f): dst[(i%nz)*ntotal + i/nz] = src[i], the
// [cell, level] -> [level, cell] scatter (dst words of cells >= nc untouched). Injective while nc <= ntotal: TL copy,
// AD b_src[i] += b_dst[j], b_dst[j] = 0. Refusal slot 0: nc > ntotal or src == dst.
#include "radshared.cu"
#include "g3_common.cuh"
extern "C" __global__ void profile_to_mpas__tl(int nc, int nz, int ntotal, const float* d_src, float* d_dst, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nc * nz) return;
    if (nc > ntotal || d_src == d_dst) g3_fail(failure, 0);
    d_dst[(size_t)(i % nz) * ntotal + i / nz] = d_src[i];
}
extern "C" __global__ void profile_to_mpas__ad(int nc, int nz, int ntotal, float* b_src, float* b_dst, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nc * nz) return;
    if (nc > ntotal || b_src == b_dst) { g3_fail(failure, 0); return; }
    size_t j = (size_t)(i % nz) * ntotal + i / nz;
    g3_acc(b_src + i, b_dst[j]); b_dst[j] = 0.0f;
}
