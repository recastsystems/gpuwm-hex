// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of mpas_sw_driver_times, the RRTMG SW driver's two clock words
// (MPAS mpas_atmphys_driver_radiation_sw.F:874,899: xtime_m = xtime_s/60 and radt = dt_radtsw/60, in minutes).
// Forward: captured source 3c499c7b lines 3748-3751 (hexcore/mpasphys/rrtmg_sw_v842.py 3429-3432), image c22c485e,
// launched once per SW radiation call from rrtmg_sw_v842.py:350-351 (driver_controls), grid 1, block 64, into words[:2]
// of a 4-word buffer whose words[2:] mpas_sw_radconst writes next; the host reads all 4 words back at line 354.
//   out[0] = seconds/60   out[1] = interval/60      (thread (0,0) only; the other 63 threads return)
// Active: out (out, full overwrite of words 0 and 1). seconds (the model clock, call.time_seconds) and interval
// (dt_radtsw, a namelist constant) are f32 launch scalars set on the host, never a function of the model state, so
// they are inactive like every launch scalar in ad.entry/v1. The forward reads no device memory at all.
// So the exact derivative of out with respect to the state is identically zero: __tl writes +0 to both tangent words
// (full overwrite, so a stale tangent in the buffer can never leak through), and __ad reads and consumes both seeds
// (sets them +0) and accumulates into nothing. (With respect to the scalars the map is affine, slope 1/60; that is
// recorded in the spec and RESULT.md, and is not part of the state derivative.)
// Limiters, clamps, selects: none. The only branch is the thread guard, an inactive index predicate.
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__fdiv_rn, built
// --fmad=false --ftz=false --prec-div=true) and is bitwise to the forward image. __tl/__ad write exact zeros.
// Determinism: one writing thread, as the forward; no atomics except the failure flag. failure[0] |= 1 when a clock
// scalar is non-finite (the aug only; the forward would carry the non-finite word into the host controls).
extern "C" __global__ void mpas_sw_driver_times__aug(float seconds, float interval, float* __restrict__ out,
                                                      int* __restrict__ failure)
{
    if (blockIdx.x || threadIdx.x) return;
    out[0] = __fdiv_rn(seconds, 60.0f);
    out[1] = __fdiv_rn(interval, 60.0f);
    if (!isfinite(seconds) || !isfinite(interval)) atomicOr(failure, 1);
}

extern "C" __global__ void mpas_sw_driver_times__tl(float seconds, float interval, float* __restrict__ o_out,
                                                     int* __restrict__ failure)
{
    (void)seconds; (void)interval; (void)failure;
    if (blockIdx.x || threadIdx.x) return;
    o_out[0] = 0.0f;   // d(seconds/60)/d(state) = 0: the clock is not a state word
    o_out[1] = 0.0f;   // d(interval/60)/d(state) = 0: the radiation interval is a namelist constant
}

// Reverse. out is a full-overwrite output: its two seeds are read and consumed (set +0); no input accumulates,
// because the kernel has no active input. words[2:] of the same buffer belong to mpas_sw_radconst and are not touched.
extern "C" __global__ void mpas_sw_driver_times__ad(float seconds, float interval, float* __restrict__ b_out,
                                                     int* __restrict__ failure)
{
    (void)seconds; (void)interval; (void)failure;
    if (blockIdx.x || threadIdx.x) return;
    b_out[0] = 0.0f;
    b_out[1] = 0.0f;
}
