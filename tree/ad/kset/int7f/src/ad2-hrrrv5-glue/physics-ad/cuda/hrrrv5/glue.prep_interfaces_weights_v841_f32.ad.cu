// SPDX-License-Identifier: Apache-2.0
// This capture variant is a separate CUDA module for the native kernel name.
#define prep_interfaces_weights_v841_f32__tl prep_interfaces_v841_f32__tl
#define prep_interfaces_weights_v841_f32__ad prep_interfaces_v841_f32__ad
#define AD_INTERFACE_CAPTURE_WEIGHTS 1
#include "glue.prep_interfaces_shared.cuh"

// The captured allocations can include levels the forward never writes.  The
// interpolation-weight endpoints are also untouched.  Their output seeds
// must reach separate pre-state cotangent buffers, not the consumed seeds.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void prep_interfaces_v841_f32__ad_passthrough(
    int nlev, int ncells, int hyd_levels, int hydd_levels,
    int znu_levels, int znu_hyd_levels,
    const int *mass_levels, const int *interior_levels,
    const float *s_hyd, const float *s_hydd, const float *s_znu,
    const float *s_znu_hyd, const float *s_minus, const float *s_plus,
    float *b_hyd, float *b_hydd, float *b_znu, float *b_znu_hyd,
    float *b_minus, float *b_plus)
{
    (void)mass_levels; (void)interior_levels;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    for (int k = nlev; k < hyd_levels; ++k) {
        const int i = k * ncells + c; b_hyd[i] += s_hyd[i];
    }
    for (int k = nlev; k < hydd_levels; ++k) {
        const int i = k * ncells + c; b_hydd[i] += s_hydd[i];
    }
    for (int k = nlev; k < znu_levels; ++k) {
        const int i = k * ncells + c; b_znu[i] += s_znu[i];
    }
    for (int k = nlev; k < znu_hyd_levels; ++k) {
        const int i = k * ncells + c; b_znu_hyd[i] += s_znu_hyd[i];
    }
    const int top = nlev * ncells + c;
    b_minus[c] += s_minus[c]; b_minus[top] += s_minus[top];
    b_plus[c] += s_plus[c]; b_plus[top] += s_plus[top];
}
#endif
