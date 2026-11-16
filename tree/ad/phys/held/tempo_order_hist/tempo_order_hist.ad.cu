// SPDX-License-Identifier: Apache-2.0
// Derivative entries (TL and AD) of the TEMPO column-order histogram, written from the forward CUDA source:
// tree/src/hexcore/mpasphys/rrfs/tempo_v303.py:4031-4039 (pkg18 module 477158923b00908c), host driver
// tempo_v303.py:4662-4664 (_launch_warp: hist.fill(0), then tempo_order_hist over `cells` threads in blocks of 32).
//
// Forward:  hist[tempo_order_bucket(cost[i])] += 1   for i < n   (atomicAdd, unsigned)
//   cost   const unsigned[n]  previous TEMPO call's per-column cycle counts / 16 (clock64 in tempo_warp_steps)
//   hist   unsigned[128]      bucket counts, heaviest first (the plan clears it right before every launch)
//
// Derivative policy (COMMON-AD 87.1, PHYS-MAP P5 "order kernels as fixed permutations"; P8 rules.rs class
// "inactive": no derivative, outputs zero in both directions):
//   * Every argument is integer. cost is a cycle count, hist a count; neither is a function of any float state
//     word, so the Jacobian of the launch with respect to the float state is exactly zero.
//   * The bucket function (tempo_order_bucket: __clz, shifts, the imin(key,127) clamp) is an integer map. It is the
//     only limiter-like clamp in the kernel; it acts on an integer key, so there is no coefficient to freeze: it is
//     index selection, fixed by the forward (inactive-index policy).
//   * hist is integer storage. Its derivative words carry zero: the tangent entry writes zero over the 128 hist
//     words; the adjoint entry consumes the hist cotangent (writes zero) and sends nothing to cost (integer input).
//     The map on the bound derivative memory is the orthogonal projection P that zeroes the hist span: P^T = P,
//     ||P x|| <= ||x||, gain exactly 0 from any input into hist.
//   * cost is read by the forward only as an integer key. Its tangent and cotangent are left untouched.
//   * Equivalence with the composed window's current rule (wrules.rs INACTIVE list: Identity both ways): the plan
//     clears hist (a `clear` command, wrules R::Zero on the shadow) immediately before every tempo_order_hist
//     launch, so the hist shadow is already zero there and this entry gives the window bitwise the same shadows.
// No float arithmetic: precision flags (--fmad=false --ftz=false, as the live schemes) are carried for uniformity.
// Grid-stride over the 128 words, so any launch geometry (the forward's: ceil(n/32) blocks of 32) covers them.
#define TEMPO_ORDER_BUCKETS 128

extern "C" __global__ void tempo_order_hist__tl(int n, const float* __restrict__ cost_t, float* __restrict__ hist_t) {
    (void)n; (void)cost_t;
    const int stride = (int)(gridDim.x * blockDim.x);
    for (int b = (int)(blockIdx.x * blockDim.x + threadIdx.x); b < TEMPO_ORDER_BUCKETS; b += stride) hist_t[b] = 0.0f;
}

extern "C" __global__ void tempo_order_hist__ad(int n, float* __restrict__ cost_a, float* __restrict__ hist_a) {
    (void)n; (void)cost_a;  // cost is an integer input: it receives no cotangent
    const int stride = (int)(gridDim.x * blockDim.x);
    for (int b = (int)(blockIdx.x * blockDim.x + threadIdx.x); b < TEMPO_ORDER_BUCKETS; b += stride) hist_a[b] = 0.0f;
}
