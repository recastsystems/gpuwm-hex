// SPDX-License-Identifier: Apache-2.0
// Derivative entries (TL and AD) of the TEMPO column-order scatter, written from the forward CUDA source:
// hexcore/mpasphys/rrfs/tempo_v303.py:4031-4036 (tempo_order_bucket) and 4043-4046 (pkg18 module 477158923b00908c),
// host driver tempo_v303.py:4662-4666 (_launch_warp: hist.fill(0), tempo_order_hist, tempo_order_scan, then this
// kernel over cells threads in blocks of 32, then tempo_warp_steps reading order).
//
// Forward:  order[atomicAdd(&start[tempo_order_bucket(cost[i])], 1u)] = i      for i < n
//   cost   const unsigned[n]  previous TEMPO call's per-column cycle counts >> 4 (clock64 in tempo_warp_steps, :4121)
//   start  unsigned[128]      exclusive scan of the bucket counts on entry; inclusive scan (start + count) on exit
//   order  int[n]             the column schedule (a permutation of 0..n-1, heaviest bucket first)
//
// Derivative policy (COMMON-AD 87.1, PHYS-MAP P5 "order kernels as fixed permutations"; P8 rules.rs class inactive):
//   * Every argument is integer. No float state word reaches the kernel and nothing it writes is float, so the
//     Jacobian of the launch with respect to the float state is exactly zero.
//   * tempo_order_bucket (__clz, shifts, the imin(key,127) clamp at :4035) is an integer index map: no coefficient to
//     freeze (inactive-index policy). The atomicAdd slot order within a bucket is nondeterministic; it is integer
//     index selection too and is not differentiated.
//   * start and order are integer storage. Their derivative words carry zero: the TL writes zero over the 128 start
//     words and the n order words; the AD consumes both cotangents (writes zero) and sends nothing to cost (integer
//     input). On the bound derivative memory the map is the orthogonal projection P that zeroes the start and order
//     spans: P^T = P, ||P x|| <= ||x||, gain exactly 0 into start and order.
//   * cost is read only as an integer key: its tangent and cotangent are left untouched.
// No float arithmetic: precision flags (--fmad=false --ftz=false, as the live schemes) are carried for uniformity.
// Grid-stride loops over 128 and n words, so any launch geometry (the forward's: ceil(n/32) blocks of 32) covers them.
#define TEMPO_ORDER_BUCKETS 128

extern "C" __global__ void tempo_order_scatter__tl(int n, const float* __restrict__ cost_t, float* __restrict__ start_t,
                                                   float* __restrict__ order_t) {
    (void)cost_t;
    const int stride = (int)(gridDim.x * blockDim.x);
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    for (int b = t; b < TEMPO_ORDER_BUCKETS; b += stride) start_t[b] = 0.0f;
    for (int i = t; i < n; i += stride) order_t[i] = 0.0f;
}

extern "C" __global__ void tempo_order_scatter__ad(int n, float* __restrict__ cost_a, float* __restrict__ start_a,
                                                   float* __restrict__ order_a) {
    (void)cost_a;  // cost is an integer input: it receives no cotangent
    const int stride = (int)(gridDim.x * blockDim.x);
    const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    for (int b = t; b < TEMPO_ORDER_BUCKETS; b += stride) start_a[b] = 0.0f;
    for (int i = t; i < n; i += stride) order_a[i] = 0.0f;
}
