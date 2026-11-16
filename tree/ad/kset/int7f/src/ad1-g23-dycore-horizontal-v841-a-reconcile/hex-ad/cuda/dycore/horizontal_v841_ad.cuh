// SPDX-License-Identifier: Apache-2.0
// Small arithmetic and failure-record helpers shared by these derivatives.
#ifndef HEX_AD_HORIZONTAL_V841_CUH
#define HEX_AD_HORIZONTAL_V841_CUH
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
// Device-side refusal slots: the source line is the slot index. The host
// creates any richer AdFailureRecord after reading the nonzero slots.
typedef int AdFailureRecord;
__device__ __forceinline__ bool ad_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
// Integer atomics are deterministic: a slot is either zero or one.
__device__ __forceinline__ void ad_fail(AdFailureRecord *f, int slot,
                                        uint64_t field, uint64_t index,
                                        uint64_t source) {
    (void)slot; (void)field; (void)index;
    atomicExch(f + (source < 512 ? (unsigned)source : 0), 1);
}
#define AD_TEST(value, field, index, source) \
    do { if (!ad_finite(value)) ad_fail(failure, ad_thread, field, index, source); } while (0)
#define AD_LOOP(total) \
    for (int ad_i = ad_thread; ad_i < (total); ad_i += blockDim.x * gridDim.x)
#endif
