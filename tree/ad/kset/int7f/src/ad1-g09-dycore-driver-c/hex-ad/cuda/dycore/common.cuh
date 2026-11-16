// SPDX-License-Identifier: MIT
// Shared failure slots: one int32 device slot per source guard. Integer
// atomics are deterministic; any nonzero slot refuses the gradient.
#pragma once
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
__device__ __forceinline__ void ad_refuse(int *failure, int index,
        uint64_t field, int source_slot) {
    (void)index; (void)field;
    atomicExch(failure + source_slot, 1);
}

// Every block first finishes its contributions and all reads of output seeds.
// The final arriving block can then consume those seeds without racing a
// gather or a scalar reduction in another block. The caller supplies a
// zero-initialized, per-launch workspace word. No block waits for another.
// The device-scoped acquire/release RMW chain orders each completed block's
// seed reads before the final block begins clearing; the block barriers carry
// that ordering between each block's reader threads and its counter thread.
__device__ __forceinline__ bool ad_last_block(unsigned int *completed) {
    __shared__ unsigned int last;
    __syncthreads();
    if (threadIdx.x == 0)
        last = __nv_atomic_fetch_add(completed, 1u, __NV_ATOMIC_ACQ_REL,
                                     __NV_THREAD_SCOPE_DEVICE) == gridDim.x - 1u;
    __syncthreads();
    return last != 0u;
}
