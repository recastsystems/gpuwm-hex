// SPDX-License-Identifier: Apache-2.0
// Derivative entries (TL and AD) of the RRTMG LW wrapper layer-count histogram, written from the forward CUDA source:
// tree/src/hexcore/mpasphys/rrtmg_lw_v842.py:560-565 (kernel 562-565; captured module source 683d7a07.cu:9607, image 34da9e09),
// host driver rrtmg_lw_v842.py:1002-1035 (_layer_groups: hist = cp.zeros(1025) at 1015, launch ceil(nc/256) x 256 at 1016, then
// sizes = hist.get() at 1019 selects the wrapper launch shapes; called from rrtmg_lwrad_device at 1190).
//
// Forward:  b = counts[c] - base;  hist[(b < 0 || b >= nbins) ? nbins : b] += 1   for c < nc   (atomicAdd, int)
//   counts  const int[nc]      wrapper layer count per column, nm + max(NINT(ptop/400 Pa),1)  (mpas_layer_counts)
//   hist    int[nbins+1]       column count per layer count; bin nbins is the overflow bin (host falls back)
//
// Derivative policy (P8 rules.rs class "inactive"; the same form as tempo_order_hist, COMMON-AD 87.1):
//   * Every argument is integer. counts is a layer count (an integer NINT of the top pressure, computed and rounded by
//     mpas_layer_counts), hist a column count. Neither is a float state word, and the Jacobian of this launch with
//     respect to the float state is exactly zero. The float -> integer step (NINT) lives in mpas_layer_counts, not here.
//   * The range test (b < 0 || b >= nbins) is the only limiter-like clamp. It acts on an integer bin index, so there is
//     no coefficient to freeze: it is index selection, fixed by the forward (inactive-index policy).
//   * hist is integer storage. Its derivative words carry zero: the tangent entry writes zero over the nbins+1 hist
//     words; the adjoint entry consumes the hist cotangent (writes zero) and sends nothing to counts (integer input).
//     The map on the bound derivative memory is the orthogonal projection P that zeroes the hist span: P^T = P,
//     ||P x|| <= ||x||, gain exactly 0 from any input into hist. counts' tangent and cotangent are left untouched.
//   * The host clears hist (cp.zeros) right before every launch, so the hist shadow is already zero there under the
//     window's clear rule, and this entry gives the window bitwise the same shadows as holding the launch.
// No float arithmetic: the precision flags of the live schemes (--fmad=false --ftz=false) are carried for uniformity.
// Grid-stride over the nbins+1 words, so any launch geometry (the forward's: ceil(nc/256) blocks of 256) covers them,
// including a batch with fewer than nbins+1 columns.

extern "C" __global__ void mpas_layer_histogram__tl(int nc, int base, int nbins, const float* __restrict__ counts_t,
                                                    float* __restrict__ hist_t) {
    (void)nc; (void)base; (void)counts_t;
    const long long n = (long long)nbins + 1, stride = (long long)gridDim.x * blockDim.x;
    for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < n; b += stride) hist_t[b] = 0.0f;
}

extern "C" __global__ void mpas_layer_histogram__ad(int nc, int base, int nbins, float* __restrict__ counts_a,
                                                    float* __restrict__ hist_a) {
    (void)nc; (void)base; (void)counts_a;  // counts is an integer input: it receives no cotangent
    const long long n = (long long)nbins + 1, stride = (long long)gridDim.x * blockDim.x;
    for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < n; b += stride) hist_a[b] = 0.0f;
}
