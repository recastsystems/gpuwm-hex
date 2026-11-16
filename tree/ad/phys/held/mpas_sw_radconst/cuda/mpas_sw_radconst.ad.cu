// SPDX-License-Identifier: Apache-2.0
// Derivative entries (TL and AD) of the RRTMG shortwave solar constants, written from the forward CUDA source:
// tree/src/hexcore/mpasphys/rrtmg_sw_v842.py:3434 (capture source 3c499c7b lines 3753-3763, image c22c485e),
// launched at rrtmg_sw_v842.py:352 (driver_controls, out = words[2:4]) and :325 (radconst), grid 1 x block 64 with no
// thread guard (every thread writes the same two words). MPAS mpas_atmphys_driver_radiation_sw.F:1009-1050 (radconst).
//
// Forward (lines 3754-3762; FMUL/FADD/FSUB/FDIV are __f*_rn, glibc_* are the glibc 2.39 float transcriptions):
//   sinob  = sinf(23.5*degrad)
//   sx     = dpd*(julian >= 80 ? julian-80 : julian+285)*degrad
//   out[0] = asinf(sinob*sinf(sx))                                   declination (rad)
//   rjul   = julian*360/365*degrad
//   out[1] = 1370*(1.000110 + .034221 cos rjul + .001280 sin rjul + .000719 cos 2rjul + .000077 sin 2rjul)   solcon (W m-2)
//   julian, degrad, dpd   by-value f32 scalars from the host calendar (curr_julday, pi/180, 360/365)
//   out                   f32[2], a view at word 2 of the 4-word `words` buffer (byte offset 8); the host reads it back
//
// Derivative policy (COMMON-AD 87.1; kernel set class inactive-entry):
//   * The kernel reads no device memory. Its only inputs are by-value calendar scalars, which are not state: the
//     window's direction x lives on device allocations, and a by-value scalar has no tangent there. So d out / d x = 0
//     exactly, for every bound allocation.
//   * out IS overwritten, so Identity would be wrong: it would carry the pre-launch tangent of words[2:4] through
//     (d out / d pre = 1, when the true value is 0). The exact rule is zero on the two written words:
//       TL: writes +0 over out's derivative words [0, 2); nothing else is touched.
//       AD: consumes out's cotangent (writes +0 over [0, 2)); nothing is accumulated anywhere.
//     On the bound derivative memory the map is the orthogonal projection P that zeroes those two words: P^T = P,
//     ||P x|| <= ||x||, gain exactly 0 into out.
//   * No limiter-like clamp exists in the forward. Its one selector, julian >= 80 (line 3755), chooses the day-of-year
//     origin on an inactive scalar; nothing to freeze. asinf's argument is bounded by |sin(23.5 deg)| = 0.399 < 1.
//   * Downstream, declin and solcon go back to the host (words.get()) and reach later SW launches as by-value scalars:
//     inactive there too, so a radiation-live window takes them from the trajectory, never from a perturbed state.
// Refusal (failure slot, never silent): a non-finite primal. The entries recompute the forward's own f32 arguments of
// sinf/cosf (same __f*_rn order) and refuse if any is non-finite; with them finite, every sinf/cosf is in [-1, 1],
// asinf's argument is in [-1, 1] and both outputs are finite, so this is exactly the non-finite-output set.
// No float derivative arithmetic. Built with the live schemes' flags (-O3 --fmad=false --ftz=false) for uniformity.

__device__ __forceinline__ bool srs_bad(float julian, float degrad, float dpd) {
    const float a = __fmul_rn(23.5f, degrad);                                                      // line 3754
    const float sel = julian >= 80.f ? __fsub_rn(julian, 80.f) : __fadd_rn(julian, 285.f);         // line 3755
    const float sx = __fmul_rn(__fmul_rn(dpd, sel), degrad);                                       // line 3755
    const float rjul = __fmul_rn(__fdiv_rn(__fmul_rn(julian, 360.f), 365.f), degrad);              // line 3757
    const float r2 = __fmul_rn(2.f, rjul);                                                         // line 3760
    return !(isfinite(julian) && isfinite(degrad) && isfinite(dpd) && isfinite(a) && isfinite(sx) &&
             isfinite(rjul) && isfinite(r2));
}

extern "C" __global__ void mpas_sw_radconst__tl(float julian, float degrad, float dpd, float* __restrict__ d_out,
                                                int* failure) {
    if (blockIdx.x | threadIdx.x) return;  // the forward's 64 threads write identical words; one writer suffices
    if (srs_bad(julian, degrad, dpd)) atomicExch(failure, 1);
    d_out[0] = 0.0f;  // declination: depends only on inactive scalars
    d_out[1] = 0.0f;  // solar constant: same
}

extern "C" __global__ void mpas_sw_radconst__ad(float julian, float degrad, float dpd, float* __restrict__ b_out,
                                                int* failure) {
    if (blockIdx.x | threadIdx.x) return;
    if (srs_bad(julian, degrad, dpd)) atomicExch(failure, 1);
    b_out[0] = 0.0f;  // cotangent consumed (out overwritten); nothing reaches any input
    b_out[1] = 0.0f;
}
