// SPDX-License-Identifier: Apache-2.0
// Helpers shared by the RRFS horizontal derivative entry points.
#pragma once

// Section 35: one int32 device refusal slot per source ID; the runtime
// builds the host AdFailureRecord from the slot and launch metadata.
using HorizontalFailure = int;
__device__ __forceinline__ void horizontal_fail(
    HorizontalFailure *f, long long slot, unsigned long long field,
    unsigned long long element, unsigned long long source)
{
    (void)slot; (void)field; (void)element;
    atomicExch(&f[source - 1], 1);  // deterministic refusal flag, never a field accumulation
}
__device__ __forceinline__ float h_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float h_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float h_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float h_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ long long h_tid() {
    return (long long)blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ long long h_stride() {
    return (long long)gridDim.x * blockDim.x;
}
__device__ __forceinline__ float h_sign(int c, int e, const int *coe) {
    return coe[2 * e] == c ? 1.0f : -1.0f;
}
__device__ __forceinline__ int h_kite(int v, int c, const int *cov, int degree) {
    for (int s = 0; s < degree; ++s) if (cov[v * degree + s] == c) return s;
    return -1;
}
__device__ __forceinline__ bool h_finite(float x, HorizontalFailure *f,
    long long slot, unsigned long long id, long long index, unsigned long long source) {
    if (!isfinite(x)) { horizontal_fail(f,slot,id,index,source); return false; }
    return true;
}
