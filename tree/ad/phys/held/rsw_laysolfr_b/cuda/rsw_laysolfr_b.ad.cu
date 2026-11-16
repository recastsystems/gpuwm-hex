// SPDX-License-Identifier: Apache-2.0
// Derivative entries (TL and AD) of the RRTMG shortwave per-band solar-source layer index (laysolfr), batched, written
// from the forward CUDA source: tree/src/hexcore/mpasphys/rrtmg_sw_v842.py:2991-3021 (pin sha256 44588ef0; capture
// source 3c499c7b lines 3310-3340, image c22c485e), launched by gpuwm/core/rrtmg_sw.py:3165-3194
// (_laysolfr_batch_device: ceil(nc*14/128) blocks of 128, one thread per (column, band)), MPAS
// module_ra_rrtmg_sw.F:3201-4666 (the laysolfr scans inside taumol16..29).
//
// Forward (per thread idx < ncol*14; col = idx/14, band = idx%14; row = jp + col*nlayers; trop = laytrop[col]):
//   lr = spec[band], upper = spec[14+band], has_lr = spec[28+band]       (layreffr, upper-loop band, band has layreffr)
//   !has_lr (band 26):  result = trop
//   upper (16,17,27,28,29): result = nlayers; for lay = trop+1..nlayers:
//                         if row[(lay-2+nlayers)%nlayers] < lr && row[lay-1] >= lr: result = lay      (last wins)
//   lower (the rest):   result = trop;    for lay = 1..trop:
//                         if row[lay-1] < lr && row[min(lay, nlayers-1)] >= lr: result = min(lay+1, trop)
//   jp      const int[ncol, nlayers]  pressure-table index per layer (rsw_setcoef_b: (int)(36 - 5(ln p + 0.04)), [1,58])
//   laytrop const int[ncol]           tropospheric layer count (host sum of setcoef's trop flags)
//   spec    const int[3, 14]          host table constants (gpuwm/core/rrtmg_sw.py:3144 _laysolfr_band_spec)
//   result  int[ncol, 14]             Fortran layer index of the band's solar source; rsw_sfluxzen_b reads it as the
//                                     layer whose colh2o/fac** pick sfluxzen (rrtmg_sw_v842.py:1784)
//
// Derivative policy (COMMON-AD 87.1; P8 rules.rs class "inactive"; kernel set class inactive-entry):
//   * Every argument is an integer. The map has no float input and no float output, so its derivative is identically
//     zero. The linearisation is the frozen-index one: the trajectory's laysolfr (written by the forward itself) is kept,
//     and its derivative words are zero in both directions. That is the documented frozen-coefficient treatment of the
//     limiter-like elements here, the min(lay+1, trop) clamp, the last-match-wins scan and the jp-vs-layreffr
//     comparisons: each selects an integer, none has a coefficient. Upstream, jp itself is a frozen integer
//     truncation of ln p (setcoef), so the whole chain p -> jp -> laysolfr is piecewise constant.
//   * TL: writes +0 over result's derivative words [0, ncol*14); the jp, laytrop and spec tangents are read-only and
//     untouched.
//   * AD: consumes result's cotangent (writes +0 over [0, ncol*14)); sends nothing to jp, laytrop or spec (accumulate,
//     no write).
//   * On the bound derivative memory the map is the orthogonal projection P that zeroes the result span: P^T = P,
//     ||P x|| <= ||x||, gain exactly 0 from any input into result.
//   * Downstream (rsw_sfluxzen_b) the index selects which layer's column amounts and interpolation factors feed the
//     band's solar source; a radiation-live window takes the index from the forward replay, never re-derives it from a
//     perturbed state.
// Refusal (failure slot, never silent): the forward reads out of bounds or divides by zero when nlayers < 1
// ((lay-2+nlayers)%nlayers), when laytrop < 0 (upper loop starts at lay <= 0: row[-1]), or when laytrop > nlayers
// (lower loop reads row[lay-1] past the column), and its int thread index overflows when ncol*14 > INT_MAX. The entries
// check exactly those preconditions per thread and set the slot on any of them.
// No float derivative arithmetic. Built with the live schemes' flags (-O3 --fmad=false --ftz=false, sm_120/sm_89
// SASS + compute_80 PTX) for uniformity.
#include <limits.h>
#define LSF_NBND 14

__device__ __forceinline__ bool lsf_bad(int ncol, int nlayers, int col, const int* laytrop) {
    if (nlayers < 1) return true;                                           // forward % nlayers by zero
    if ((long long)ncol * LSF_NBND > (long long)INT_MAX) return true;       // forward's int idx would overflow
    const int trop = laytrop[col];
    return trop < 0 || trop > nlayers;                                      // forward reads jp outside the column
}

extern "C" __global__ void rsw_laysolfr_b__tl(int ncol, int nlayers, const int* __restrict__ laytrop,
                                              const float* __restrict__ d_jp, const float* __restrict__ d_laytrop,
                                              const float* __restrict__ d_spec, float* __restrict__ d_result,
                                              int* failure) {
    (void)d_jp; (void)d_laytrop; (void)d_spec;  // read-only tangents of integer inputs: untouched
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * LSF_NBND) return;
    if (lsf_bad(ncol, nlayers, (int)(idx / LSF_NBND), laytrop)) atomicExch(failure, 1);
    d_result[idx] = 0.0f;
}

extern "C" __global__ void rsw_laysolfr_b__ad(int ncol, int nlayers, const int* __restrict__ laytrop,
                                              float* __restrict__ b_jp, float* __restrict__ b_laytrop,
                                              float* __restrict__ b_spec, float* __restrict__ b_result,
                                              int* failure) {
    (void)b_jp; (void)b_laytrop; (void)b_spec;  // accumulate: d result / d (jp, laytrop, spec) = 0, nothing is added
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * LSF_NBND) return;
    if (lsf_bad(ncol, nlayers, (int)(idx / LSF_NBND), laytrop)) atomicExch(failure, 1);
    b_result[idx] = 0.0f;  // cotangent consumed
}
