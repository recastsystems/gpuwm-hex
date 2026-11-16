// SPDX-License-Identifier: Apache-2.0
// Tangent and adjoint of the TEMPO surface aerosol emission launch
//   tree/src/hexcore/mpasphys/rrfs/tempo_v303.py:4417-4421 (pkg18 module 477158923b00908c)
//     nwfa[c] = __fadd_rn(nwfa[c], __fmul_rn(surface[c], dt))   for c < columns
//   (RRFS module_mp_tempo_driver.F90:439-454: nwfa(:,1,:) += nwfa2d * dt), launched at tempo_v303.py:4624.
// Precision: float32 in and out, the forward's own round-to-nearest products and sums (explicit __fmul_rn /
// __fadd_rn, built with --fmad=false), so the TL and the AD apply the same rounded linear map.
// Active: nwfa (inout, level-zero words only; the other levels are untouched words, identity in the dot) and
// surface (nwfa2d, in). dt (dt_microp, a recorded configuration constant of the plan) and columns are inactive.
// The map is affine in (nwfa, surface): no branch on state, no limiter, no clamp, nothing frozen.
// Refusal: one int32 slot, set when any participating primal, derivative or result is nonfinite.
__device__ __forceinline__ bool te_finite(float x) { return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u; }

extern "C" __global__ void tempo_emissions__tl(
    const float *nwfa, const float *surface, float dt, int columns,
    const float *d_nwfa, const float *d_surface, float *d_out, int *failures)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < columns; c += gridDim.x * blockDim.x) {
        const float p = __fadd_rn(nwfa[c], __fmul_rn(surface[c], dt));
        const float d = __fadd_rn(d_nwfa[c], __fmul_rn(d_surface[c], dt));
        if (!te_finite(nwfa[c]) || !te_finite(surface[c]) || !te_finite(dt) || !te_finite(p) ||
            !te_finite(d_nwfa[c]) || !te_finite(d_surface[c]) || !te_finite(d))
            atomicExch(failures, 1);
        d_out[c] = d;
    }
}

extern "C" __global__ void tempo_emissions__ad(
    const float *nwfa, const float *surface, float dt, int columns,
    float *b_out, float *b_surface, float *b_nwfa, int *failures)
{
    // When nwfa's pre- and post-state share one cotangent buffer, that buffer already holds the pass-through
    // contribution and stays live as the pre-state cotangent; otherwise b_out (the post-state seed) is consumed
    // and zeroed and b_nwfa accumulates it (the add_inplace_f32__ad contract).
    const bool shared = b_nwfa == b_out;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < columns; c += gridDim.x * blockDim.x) {
        const float b = b_out[c];
        const float p = __fadd_rn(nwfa[c], __fmul_rn(surface[c], dt));
        const float bs = __fadd_rn(b_surface[c], __fmul_rn(b, dt));
        const float bn = shared ? b : __fadd_rn(b_nwfa[c], b);
        if (!te_finite(nwfa[c]) || !te_finite(surface[c]) || !te_finite(dt) || !te_finite(p) ||
            !te_finite(b) || !te_finite(bs) || !te_finite(bn))
            atomicExch(failures, 1);
        b_surface[c] = bs;
        b_nwfa[c] = bn;
        if (!shared) b_out[c] = 0.0f;
    }
}
