// SPDX-License-Identifier: Apache-2.0
// Shared support for the RRFS horizontal derivative kernels.
#pragma once

// The device contract is one zeroed int32 slot per statically distinct
// refusal site. Atomic writes are deterministic: every writer stores 1.
typedef int AdFailureRecord;
__device__ __forceinline__ void ad_finite(
    AdFailureRecord *fail, unsigned worker, float x,
    unsigned long long field, unsigned long long index,
    unsigned long long source_slot)
{
    (void)worker; (void)field; (void)index;
    if (!isfinite(x)) atomicExch(&fail[source_slot], 1);
}

__device__ __forceinline__ int ad_index(int k, int i, int count) {
    return k * count + i;
}
__device__ __forceinline__ float ad_vertex_sign(
    int vertex, int edge, const int *vertices_on_edge) {
    return vertices_on_edge[2 * edge] == vertex ? -1.0f : 1.0f;
}
__device__ __forceinline__ float ad_cell_sign(
    int cell, int edge, const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}

