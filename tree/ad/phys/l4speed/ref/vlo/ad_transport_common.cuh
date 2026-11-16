// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(__CUDACC_RTC__)
// NVRTC (the passed runner's compiler) has no host headers; same widths as <stdint.h> on every CUDA target.
typedef int int32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#include <math.h>
#endif

// Device failure slots are int32 flags, one per source id (COMMON-AD 35.4).
// Many threads may refuse for the same source, so the update is atomic.
typedef int32_t AdTransportFailure;
__device__ __forceinline__ void ad_transport_fail(
    AdTransportFailure *slots, uint64_t thread_slot, uint64_t field,
    uint64_t element, uint64_t source)
{
    (void)thread_slot; (void)field; (void)element;
    atomicExch(&slots[source - 1], 1);
}

// Deterministic f32 compensated sum, used only by a uniquely owning thread.
__device__ __forceinline__ void ad_transport_sum(float value, float &sum, float &error)
{
    const float y = value - error;
    const float t = sum + y;
    error = (t - sum) - y;
    sum = t;
}
