// SPDX-License-Identifier: Apache-2.0
#ifndef HEX_AD_DYCORE_COMMON_CUH
#define HEX_AD_DYCORE_COMMON_CUH
#include "mpas_fp32.cuh"
// The device contract uses one 32-bit refusal slot per declared owner.
// Host code expands a nonzero slot into the FRAMEWORK-v1 failure record.
__device__ __forceinline__ void ad_bad(int *slots, unsigned long long slot,
        unsigned long long field, unsigned long long element, unsigned long long source) {
    (void)slot; (void)field; (void)element;
    atomicExch(&slots[source], 1);
}
__device__ __forceinline__ bool ad_finite(float x) { return isfinite(x); }
// A double accumulator is destined for an f32 output: reject an overflow before
// its final conversion as well as an already non-finite partial result.
__device__ __forceinline__ bool ad_finite(double x) {
    return isfinite(x) && fabs(x) <= 3.4028234663852886e38;
}
#endif
