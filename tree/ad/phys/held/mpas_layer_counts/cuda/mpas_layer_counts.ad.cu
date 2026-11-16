// SPDX-License-Identifier: Apache-2.0
// Derivative entries (TL and AD) of the RRTMG longwave wrapper layer count, written from the forward CUDA source:
// tree/src/hexcore/mpasphys/rrtmg_lw_v842.py:348-353 (capture source 683d7a07 lines 9393-9398, image 34da9e09),
// launched at rrtmg_lw_v842.py:1184 (one thread per column, ceil(nc/32) blocks of 32), MPAS module_ra_rrtmg_lw.F:11817-11821.
//
// Forward (per column c < nc):
//   top       = interfaces[c*2*(nm+1)+nm] / 100            (model-top interface pressure, Pa -> hPa, __fdiv_rn)
//   counts[c] = nm + max((int)(top/4 + 0.5), 1)            (NINT(top/deltap), deltap = 4 hPa; __fdiv_rn, __fadd_rn)
//   interfaces const float[nc, 2, nm+1]  row 0 pressure (Pa), row 1 temperature (K); only word [c][0][nm] is read
//   counts     int[nc]                   wrapper layer count of the column: nm model layers plus the extra layers
//                                        RRTMG adds between the model top and 0 hPa (launch metadata: _layer_groups
//                                        groups columns by it and reads its histogram back to the host)
//
// Derivative policy (COMMON-AD 87.1; P8 rules.rs class "inactive"; kernel set class inactive-entry):
//   * counts is integer. As a function of the one float word it reads it is piecewise constant: d counts / d top = 0
//     everywhere except on the tie set top = 4*(k + 1/2) hPa, where it jumps by 1. The linearisation is the frozen-
//     index one: the trajectory's counts (written by the forward itself) are kept, and the derivative words of counts
//     are zero in both directions. That is the documented frozen-coefficient treatment of the two limiter-like
//     elements here, the NINT truncation and the max(.,1) floor: both select an integer, neither has a coefficient.
//   * TL: writes +0 over counts' derivative words [0, nc); interfaces' tangent is read-only and untouched.
//   * AD: consumes counts' cotangent (writes +0 over [0, nc)); sends nothing to interfaces (accumulate, no write).
//   * On the bound derivative memory the map is the orthogonal projection P that zeroes the counts span: P^T = P,
//     ||P x|| <= ||x||, gain exactly 0 from any input into counts.
//   * Downstream use of counts (group widths, the extra-layer pressures in mpas_lw_prepare) stays the trajectory's:
//     a radiation-live window must take the groups from the forward replay, never re-derive them from a perturbed
//     state (layer_groups.rs rule: live histogram, never a recording).
// Refusal (failure slot, never silent): the forward's (int) conversion is implementation-defined for a non-finite or
// out-of-int32 value, nm + n can overflow int32, and the forward's int index c*2*(nm+1)+nm overflows past 2^31 words.
// The entries recompute the forward's own f32 value with the forward's __f*_rn order and set the slot on any of these.
// No float derivative arithmetic. Built with the live schemes' flags (-O3 --fmad=false --ftz=false, sm_120/sm_89
// SASS + compute_80 PTX) for uniformity.
#include <limits.h>

__device__ __forceinline__ bool mlc_bad(int c, int nm, const float* interfaces) {
    const long long idx = (long long)c * 2LL * ((long long)nm + 1LL) + (long long)nm;
    if (nm < 0 || idx > (long long)INT_MAX) return true;                    // forward's int index would overflow
    const float top = __fdiv_rn(interfaces[idx], 100.0f);                   // forward line 9396
    const float v = __fadd_rn(__fdiv_rn(top, 4.0f), 0.5f);                  // forward line 9397
    if (!isfinite(v) || v >= 2147483648.0f || v <= -2147483904.0f) return true;  // (int)v implementation-defined
    const long long n = (long long)max((int)v, 1);
    return (long long)nm + n > (long long)INT_MAX;                          // nm + n overflows int32
}

extern "C" __global__ void mpas_layer_counts__tl(int nc, int nm, const float* __restrict__ interfaces,
                                                 const float* __restrict__ d_interfaces, float* __restrict__ d_counts,
                                                 int* failure) {
    (void)d_interfaces;  // read-only tangent of an input whose only consumer is an integer: untouched
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    if (mlc_bad(c, nm, interfaces)) atomicExch(failure, 1);
    d_counts[c] = 0.0f;
}

extern "C" __global__ void mpas_layer_counts__ad(int nc, int nm, const float* __restrict__ interfaces,
                                                 float* __restrict__ b_interfaces, float* __restrict__ b_counts,
                                                 int* failure) {
    (void)b_interfaces;  // accumulate: d counts / d interfaces = 0, nothing is added
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    if (mlc_bad(c, nm, interfaces)) atomicExch(failure, 1);
    b_counts[c] = 0.0f;  // cotangent consumed
}
