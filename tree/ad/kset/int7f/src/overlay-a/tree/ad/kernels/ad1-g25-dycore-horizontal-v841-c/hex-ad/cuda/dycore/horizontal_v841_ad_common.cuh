// SPDX-License-Identifier: MIT
// Shared declarations for the horizontal-v841 derivative kernels.
#pragma once
// Section 33/35: zero-initialised int32 device slots are refusal flags, one per source predicate.
// The host constructs the detailed failure record from the launch and slot.
using HorizontalAdFailure = int;
__device__ __forceinline__ void horizontal_ad_fail(
    HorizontalAdFailure *f, unsigned long long slot,
    unsigned long long field, unsigned long long element,
    unsigned long long source)
{
    (void)field; (void)element; (void)source;
    atomicExch(f + slot, 1);
}
__device__ __forceinline__ float horizontal_sign(int cell, int edge, const int *cells_on_edge)
{
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}
__device__ __forceinline__ int horizontal_kite(
    int vertex, int cell, const int *cells_on_vertex, int degree)
{
    for (int t = 0; t < degree; ++t)
        if (cells_on_vertex[vertex * degree + t] == cell) return t;
    return -1;
}
