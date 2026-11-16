// SPDX-License-Identifier: Apache-2.0
#pragma once
// One nonzero int32 slot per exact derivative validity predicate.
__device__ __forceinline__ bool ad_validity(int* failure, int slot, bool valid) {
    if (!valid) atomicExch(failure + slot, 1);
    return valid;
}
