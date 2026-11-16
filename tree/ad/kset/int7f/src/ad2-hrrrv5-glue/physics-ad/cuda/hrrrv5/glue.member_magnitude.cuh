// SPDX-License-Identifier: Apache-2.0
#pragma once
// Exact common pure bit operation from the g26, g27 and g28 member helpers.
__device__ __forceinline__ unsigned int glue_f32_magnitude_bits(float x) {
    return __float_as_uint(x) & 0x7fffffffu;
}
