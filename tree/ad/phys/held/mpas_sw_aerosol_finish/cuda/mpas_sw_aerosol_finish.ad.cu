// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1) of mpas_sw_aerosol_finish, the third pass of MPAS's SW aerosol optics
// (module_ra_rrtmg_sw_aerosols.F:70-451, the column diagnostics angexp2d / aerssa2d / aerasy2d).
// Forward: captured source 3c499c7b lines 3903-3915 (hexcore/mpasphys/rrtmg_sw_v842.py 3584-3596), image c22c485e,
// launched from MPASShortwave._aerosol_passes (rrtmg_sw_v842.py:447-449), grid ceil(nc/64), block 64.
// Per column c (row strides: exponent/ang2/ssa2/asy2 1, slope nz when allocated, ssa/asy 14*nl):
//   ang2[c] = exponent[c]                                      if angopt == 1
//           = -(sum_k slope[c][k]) (f32 running FSUB)          if (mask&1) && angopt == 3 && has3d
//           = -slope[c][0]                                     if (mask&1) && angopt == 3 && !has3d
//           (word untouched otherwise)
//   ssa2[c] = ssaval                                           if ssaopt == 1
//           = (sum_{b=7..10} ssa[c][b][0]) / 4                 if (mask&2) && ssaopt == 3  (untouched otherwise)
//   asy2[c]: the same with asyopt, asyval and mask&4.
// Every branch is a launch-scalar select (has3d, angopt, ssaopt, asyopt, diagnostic_mask), the same for every
// column of a launch: there is no state-dependent branch, clamp or limiter, so nothing is frozen. The map is linear
// in exponent, slope, ssa and asy (constant in the ssaval / asyval arms), so TL and AD read no primal words.
// Unwritten outputs: where the forward does not write ang2 / ssa2 / asy2 (the hrrrv5 configuration: has3d 1, every
// option 3, diagnostic_mask 0, so the launch writes NOTHING), __tl does not write their tangent words (the incoming
// tangent passes through: identity) and __ad neither reads nor consumes their seeds (identity cotangent). Where the
// forward writes, __tl overwrites the tangent and __ad consumes the seed (read, set 0) and accumulates the inputs.
// Precision (the glue-row rule, as the live schemes): __aug repeats the forward word for word in f32 (__f*_rn, built
// --fmad=false --ftz=false). __tl / __ad sum in double and round once to f32 per written word (the 1/4 is exact).
// One thread per column, as the forward; columns own disjoint words (exponent[c], slope[c][*], ssa/asy[c][7..10][0],
// the three column outputs), so the only atomic is the failure OR and both entries are deterministic.
// failure[0] |= 1 on a non-finite written derivative word.
#define MSAF_ADD(a, b) __fadd_rn((a), (b))
#define MSAF_SUB(a, b) __fsub_rn((a), (b))
#define MSAF_DIV(a, b) __fdiv_rn((a), (b))
#define SL(arr, k) arr[(long long)c * nz + (k)]
#define BW(arr, b) arr[((long long)c * 14 + (b)) * nl]

__device__ __forceinline__ bool msaf_ang(int angopt, int mask) { return angopt == 1 || ((mask & 1) && angopt == 3); }
__device__ __forceinline__ bool msaf_ssa(int ssaopt, int mask) { return ssaopt == 1 || ((mask & 2) && ssaopt == 3); }
__device__ __forceinline__ bool msaf_asy(int asyopt, int mask) { return asyopt == 1 || ((mask & 4) && asyopt == 3); }
__device__ __forceinline__ void msaf_flag(int* failure, double v) { if (!isfinite(v)) atomicOr(failure, 1); }

extern "C" __global__ void mpas_sw_aerosol_finish__aug(
    int nc, int nz, int nl, int has3d, int angopt, int ssaopt, int asyopt, float ssaval, float asyval,
    int diagnostic_mask, const float* __restrict__ exponent, const float* __restrict__ slope,
    float* __restrict__ ang2, float* __restrict__ ssa2, float* __restrict__ asy2,
    const float* __restrict__ ssa, const float* __restrict__ asy, int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    float angstrom = 0.f;
    if ((diagnostic_mask & 1) && angopt == 3)
        for (int k = 0; k < nz; k++) { float s = SL(slope, k); if (has3d) angstrom = MSAF_SUB(angstrom, s); else if (k == 0) angstrom = -s; }
    if (angopt == 1) ang2[c] = exponent[c]; else if ((diagnostic_mask & 1) && angopt == 3) ang2[c] = angstrom;
    if (ssaopt == 1) ssa2[c] = ssaval;
    else if ((diagnostic_mask & 2) && ssaopt == 3) { float v = 0.f; for (int b = 7; b <= 10; b++) v = MSAF_ADD(v, BW(ssa, b)); ssa2[c] = MSAF_DIV(v, 4.f); }
    if (asyopt == 1) asy2[c] = asyval;
    else if ((diagnostic_mask & 4) && asyopt == 3) { float v = 0.f; for (int b = 7; b <= 10; b++) v = MSAF_ADD(v, BW(asy, b)); asy2[c] = MSAF_DIV(v, 4.f); }
}

extern "C" __global__ void mpas_sw_aerosol_finish__tl(
    int nc, int nz, int nl, int has3d, int angopt, int ssaopt, int asyopt, int diagnostic_mask,
    const float* __restrict__ d_exponent, const float* __restrict__ d_slope,
    const float* __restrict__ d_ssa, const float* __restrict__ d_asy,
    float* __restrict__ o_ang2, float* __restrict__ o_ssa2, float* __restrict__ o_asy2, int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    if (angopt == 1) { o_ang2[c] = d_exponent[c]; msaf_flag(failure, d_exponent[c]); }
    else if ((diagnostic_mask & 1) && angopt == 3) {
        double t = 0.0;
        if (has3d) { for (int k = 0; k < nz; k++) t -= (double)SL(d_slope, k); }
        else if (nz > 0) t = 0.0 - (double)SL(d_slope, 0);   // 0 - x, not -x: a zero tangent gives +0 bits
        msaf_flag(failure, t);
        o_ang2[c] = (float)t;
    }
    if (ssaopt == 1) o_ssa2[c] = 0.0f;   // ssaval is a namelist constant
    else if ((diagnostic_mask & 2) && ssaopt == 3) {
        double t = 0.0;
        for (int b = 7; b <= 10; b++) t += (double)BW(d_ssa, b);
        t *= 0.25; msaf_flag(failure, t); o_ssa2[c] = (float)t;
    }
    if (asyopt == 1) o_asy2[c] = 0.0f;   // asyval is a namelist constant
    else if ((diagnostic_mask & 4) && asyopt == 3) {
        double t = 0.0;
        for (int b = 7; b <= 10; b++) t += (double)BW(d_asy, b);
        t *= 0.25; msaf_flag(failure, t); o_asy2[c] = (float)t;
    }
}

// Reverse. A written output's seed is read and consumed (set 0) and the inputs accumulate. An unwritten output's
// seed is neither read nor consumed: the forward leaves that word, so its cotangent passes through unchanged.
extern "C" __global__ void mpas_sw_aerosol_finish__ad(
    int nc, int nz, int nl, int has3d, int angopt, int ssaopt, int asyopt, int diagnostic_mask,
    float* __restrict__ b_exponent, float* __restrict__ b_slope, float* __restrict__ b_ssa, float* __restrict__ b_asy,
    float* __restrict__ b_ang2, float* __restrict__ b_ssa2, float* __restrict__ b_asy2, int* __restrict__ failure)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    if (msaf_ang(angopt, diagnostic_mask)) {
        const double g = (double)b_ang2[c];
        b_ang2[c] = 0.0f;
        msaf_flag(failure, g);
        if (angopt == 1) b_exponent[c] = (float)((double)b_exponent[c] + g);
        else if (has3d) { for (int k = 0; k < nz; k++) SL(b_slope, k) = (float)((double)SL(b_slope, k) - g); }
        else if (nz > 0) SL(b_slope, 0) = (float)((double)SL(b_slope, 0) - g);
    }
    if (msaf_ssa(ssaopt, diagnostic_mask)) {
        const double g = (double)b_ssa2[c];
        b_ssa2[c] = 0.0f;
        msaf_flag(failure, g);
        if (ssaopt == 3) for (int b = 7; b <= 10; b++) BW(b_ssa, b) = (float)((double)BW(b_ssa, b) + 0.25 * g);
    }
    if (msaf_asy(asyopt, diagnostic_mask)) {
        const double g = (double)b_asy2[c];
        b_asy2[c] = 0.0f;
        msaf_flag(failure, g);
        if (asyopt == 3) for (int b = 7; b <= 10; b++) BW(b_asy, b) = (float)((double)BW(b_asy, b) + 0.25 * g);
    }
}
#undef SL
#undef BW
