// SPDX-License-Identifier: Apache-2.0
// g3-radland: the float CuPy elementwise launches of the RRTMG radiation slot that no P8 rule covers, as hand
// ad.entry/v1 rows. CuPy passes each array as a CArray<T, ndim> by value {data, size, shape[ndim], strides[ndim]}; the
// runner's `scalar` binding of such an argument hands the entry the forward's own struct (deriv.rs, g3-radland), so the
// entry reads the view's size and byte strides. These entries handle 1-D views only (every captured launch: 18 of 18
// each); a 2-D view read through this 1-D struct shows size != shape[0] or a byte stride that is not the element size,
// and is refused (slot 0), never run on a misread layout.
// Compiled once per kernel with -DG3_KERNEL=<n>.
#include "g3_common.cuh"
struct G3Arr1 { const void* data; long long size; long long shape[1]; long long stride[1]; };
__device__ __forceinline__ bool g3_ok1(const G3Arr1& a, long long n, long long item) {
    return a.size == n && a.shape[0] == n && a.stride[0] == item;
}
#if G3_KERNEL == 11
// cupy_multiply__float32_float32_float32: out = in0 * in1 (both float32 arrays). TL d_out = in1 d_in0 + in0 d_in1 from the
// pre-launch primal words; AD reads b_out, consumes it, then adds in1 g and in0 g, so out aliasing an input (a *= b) and
// in0 == in1 (x * x) are both the exact transpose.
extern "C" __global__ void cupy_multiply__float32_float32_float32__tl(G3Arr1 A, G3Arr1 B, G3Arr1 O, const float* __restrict__ a,
    const float* __restrict__ b, const float* d_a, const float* d_b, float* d_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long n = O.size; if (i >= n) return;
    if (!g3_ok1(A, n, 4) || !g3_ok1(B, n, 4) || !g3_ok1(O, n, 4)) { g3_fail(failure, 0); return; }
    d_o[i] = g3_st(G3D(b[i]) * G3D(d_a[i]) + G3D(a[i]) * G3D(d_b[i]));
}
extern "C" __global__ void cupy_multiply__float32_float32_float32__ad(G3Arr1 A, G3Arr1 B, G3Arr1 O, const float* __restrict__ a,
    const float* __restrict__ b, float* b_a, float* b_b, float* b_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long n = O.size; if (i >= n) return;
    if (!g3_ok1(A, n, 4) || !g3_ok1(B, n, 4) || !g3_ok1(O, n, 4)) { g3_fail(failure, 0); return; }
    double g = b_o[i]; b_o[i] = 0.0f;
    g3_acc(b_a + i, G3D(b[i]) * g); g3_acc(b_b + i, G3D(a[i]) * g);
}
#elif G3_KERNEL == 12
// cupy_where__bool_float_float32_float32: out = mask ? c : x (c a host scalar). Executed arm of the pre-launch mask:
// TL d_out = mask ? 0 : d_x; AD g = b_out, b_out consumed, b_x += g where !mask (out == x in place included).
extern "C" __global__ void cupy_where__bool_float_float32_float32__tl(G3Arr1 M, G3Arr1 X, G3Arr1 O, const unsigned char* __restrict__ mask,
    const float* d_x, float* d_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long n = O.size; if (i >= n) return;
    if (!g3_ok1(M, n, 1) || !g3_ok1(X, n, 4) || !g3_ok1(O, n, 4)) { g3_fail(failure, 0); return; }
    d_o[i] = mask[i] ? 0.0f : d_x[i];
}
extern "C" __global__ void cupy_where__bool_float_float32_float32__ad(G3Arr1 M, G3Arr1 X, G3Arr1 O, const unsigned char* __restrict__ mask,
    float* b_x, float* b_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long n = O.size; if (i >= n) return;
    if (!g3_ok1(M, n, 1) || !g3_ok1(X, n, 4) || !g3_ok1(O, n, 4)) { g3_fail(failure, 0); return; }
    double g = b_o[i]; b_o[i] = 0.0f;
    if (!mask[i]) g3_acc(b_x + i, g);
}
#elif G3_KERNEL == 13 || G3_KERNEL == 14
// cupy_copy__float_float32: `a[...] = x`, x a host scalar (CuPy types it `float`; the captured scalar is float64 in 72
// launches and float32 in 18, so it is not bound): every written word is state-independent. TL +0, AD consumes.
#if G3_KERNEL == 13
#define G3_FILL(s) cupy_copy__float_float32##s
#else
// cupy_copy__int_float32: the same fill with a host integer scalar (54 captured launches, one float32 view each).
#define G3_FILL(s) cupy_copy__int_float32##s
#endif
extern "C" __global__ void G3_FILL(__tl)(G3Arr1 O, float* d_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long n = O.size; if (i >= n) return;
    if (!g3_ok1(O, n, 4)) { g3_fail(failure, 0); return; }
    d_o[i] = 0.0f;
}
extern "C" __global__ void G3_FILL(__ad)(G3Arr1 O, float* b_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long n = O.size; if (i >= n) return;
    if (!g3_ok1(O, n, 4)) { g3_fail(failure, 0); return; }
    b_o[i] = 0.0f;
}
#endif
