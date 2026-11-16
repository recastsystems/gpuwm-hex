// SPDX-License-Identifier: Apache-2.0
// G8 (autodiff speed): fixed-order, two-level float64 reductions for the
// adjoint's scalar and per-level cotangents. Phase A: every block reduces a
// fixed chunk of G8_CHUNK words (each thread walks G8_PER_THREAD words in a
// fixed stride, then a fixed shared-memory tree) into one partial. Phase B:
// one block per row reduces that row's partials in a fixed tree and adds the
// total into the cotangent. No atomics, so two runs are bitwise equal; no
// thread walks more than max(G8_PER_THREAD, nb/G8_THREADS) words.
#ifndef G8_REDUCE_CUH
#define G8_REDUCE_CUH
#define G8_THREADS 256
#define G8_PER_THREAD 8
#define G8_CHUNK (G8_THREADS * G8_PER_THREAD)
// Every thread of the block must call it; blockDim.x must be G8_THREADS.
__device__ __forceinline__ double g8_block_sum(double v)
{
    __shared__ double g8_sh[G8_THREADS];
    g8_sh[threadIdx.x] = v;
    __syncthreads();
#pragma unroll
    for (int w = G8_THREADS / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w) g8_sh[threadIdx.x] = __dadd_rn(g8_sh[threadIdx.x], g8_sh[threadIdx.x + w]);
        __syncthreads();
    }
    const double r = g8_sh[0];
    __syncthreads();
    return r;
}
// Phase B helper: partials of one row, partial[row*nb + b], b < nb.
__device__ __forceinline__ double g8_partials_sum(const double *partial, long long row, int nb)
{
    double v = 0.0;
    for (int b = threadIdx.x; b < nb; b += G8_THREADS) v = __dadd_rn(v, partial[row * (long long)nb + b]);
    return g8_block_sum(v);
}
#endif
