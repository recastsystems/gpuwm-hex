// SPDX-License-Identifier: Apache-2.0
// Derivative of the in-place endpoint overwrite at the pinned hex revision.
// The caller supplies two pre-cleared, source-attributed failure slots.
using AdFailureRecord = unsigned int;

__device__ __forceinline__ bool endpoints_valid(
    int nlev, int ncells, int cell, const float *rw, const float *seed,
    AdFailureRecord *failures)
{
    for (int k = 0; k <= nlev; ++k) {
        const unsigned long long i = (unsigned long long)k * ncells + cell;
        const bool bad_rw = !isfinite(rw[i]);
        const bool bad_seed = !isfinite(seed[i]);
        if (bad_rw) atomicExch(&failures[0], 1u);
        if (bad_seed) atomicExch(&failures[1], 1u);
        if (bad_rw || bad_seed) return false;
    }
    return true;
}

#if !defined(AD_ONLY_AD)
extern "C" __global__ void enforce_rw_endpoints_v841_f32__tl(
    const int nlev, const int ncells, const float *rw, float *d_rw,
    AdFailureRecord *failures)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;
    if (!endpoints_valid(nlev, ncells, cell, rw, d_rw, failures)) return;
    d_rw[cell] = 0.0f;
    d_rw[(unsigned long long)nlev * ncells + cell] = 0.0f;
}

#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void enforce_rw_endpoints_v841_f32__ad(
    const int nlev, const int ncells, const float *rw, float *b_rw_out,
    float *b_rw_in, AdFailureRecord *failures)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;
    if (!endpoints_valid(nlev, ncells, cell, rw, b_rw_out, failures)) return;
    // The separate input cotangent accumulates the untouched interior.
    for (int k = 1; k < nlev; ++k) {
        const unsigned long long i = (unsigned long long)k * ncells + cell;
        b_rw_in[i] += b_rw_out[i];
    }
    // The original forward overwrites only these output cotangent slots.
    b_rw_out[cell] = 0.0f;
    b_rw_out[(unsigned long long)nlev * ncells + cell] = 0.0f;
}

#endif
