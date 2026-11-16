// SPDX-License-Identifier: Apache-2.0
// L4 speed (ad/window-L4-speed): shared device pieces for owner-computes adjoints.
//
// Inverse topology, built on the device per reverse run, deterministic:
//   workspace words (int32): offsets[nkeys+1] | cursor[nkeys] | list[nitems]
//   1. count  (one thread per item, integer atomics: the count is order-free)
//   2. scan   (one block, fixed chunking: exclusive prefix of the counts)
//   3. fill   (one thread per item, integer atomics on the cursor: order not fixed)
//   4. sort   (one thread per key: insertion sort of its items ascending)
// After step 4 every key lists its items in ascending item index, which is the
// order the old (serial-search) adjoints visited them, so a gather over the list
// adds the same binary64 terms in the same order: bitwise the old result.
// No float atomics anywhere.
#pragma once
#ifndef L4_COMMON_CUH
#define L4_COMMON_CUH

__device__ __forceinline__ int *l4_offsets(int *ws) { return ws; }
__device__ __forceinline__ int *l4_cursor(int *ws, int nkeys) { return ws + nkeys + 1; }
__device__ __forceinline__ int *l4_list(int *ws, int nkeys) { return ws + 2 * nkeys + 1; }

// Exclusive scan of counts (stored in offsets[0..nkeys)) into offsets[0..nkeys],
// one block of 1024 threads, each owning a fixed contiguous chunk.
__device__ __forceinline__ void l4_scan_block(int *ws, int nkeys) {
    __shared__ int part[1024];
    const int t = threadIdx.x, nt = blockDim.x;
    const int chunk = (nkeys + nt - 1) / nt;
    const int lo = min(nkeys, t * chunk), hi = min(nkeys, lo + chunk);
    int s = 0;
    for (int i = lo; i < hi; ++i) s += ws[i];
    part[t] = s;
    __syncthreads();
    if (t == 0) { int run = 0; for (int i = 0; i < nt; ++i) { int v = part[i]; part[i] = run; run += v; } }
    __syncthreads();
    int run = part[t];
    for (int i = lo; i < hi; ++i) { int v = ws[i]; ws[i] = run; run += v; }
    if (hi == nkeys && lo < hi) ws[nkeys] = run;
    if (nkeys == 0 && t == 0) ws[0] = 0;
}

__device__ __forceinline__ void l4_sort_key(int *ws, int nkeys, int key) {
    const int *off = l4_offsets(ws);
    int *list = l4_list(ws, nkeys);
    const int lo = off[key], hi = off[key + 1];
    for (int i = lo + 1; i < hi; ++i) {
        int v = list[i], j = i - 1;
        while (j >= lo && list[j] > v) { list[j + 1] = list[j]; --j; }
        list[j + 1] = v;
    }
}

// Deterministic binary64 sum of partial[0..n) added to *target (float32), one block of 1024:
// thread t sums partial[t], partial[t+1024], ... in order; then a fixed pairwise tree.
__device__ __forceinline__ double l4_block_sum(const double *partial, long long n) {
    __shared__ double acc[1024];
    const int t = threadIdx.x, nt = blockDim.x;
    double s = 0.0;
    for (long long i = t; i < n; i += nt) s += partial[i];
    acc[t] = s;
    __syncthreads();
    for (int w = nt / 2; w > 0; w >>= 1) { if (t < w) acc[t] += acc[t + w]; __syncthreads(); }
    return acc[0];
}
#endif
