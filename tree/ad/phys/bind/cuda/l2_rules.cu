// SPDX-License-Identifier: Apache-2.0
// L2 bindings (ad/window-L2-bindings, private): the two P8-table rules the hrrrv5 plan needs that cuda/p8.cu does
// not carry, in P8's shadow-arena convention (one shadow element per forward 4-byte word; every index is an
// element index into one shadow arena; the view is P8View, element strides).
//   * cupy_multiply__float32_float_float32: out = x * c with c a host constant (the output-step unit change,
//     c = 100 in pkg18). Tangent d_out = c d_x; adjoint b_x += c b_out, b_out = 0 (in place: b_x = c b_x).
//   * inactive float outputs of P8's table (cupy_copy from an integer source into float64, 39 launches in pkg18):
//     "output 0, nothing propagates": the written words' shadow is zero in both directions.
// Strict recipe as p8.cu: --fmad=false --prec-div=true --prec-sqrt=true -O3, sm_120 (and sm_89).
struct P8View { long long base, size; int ndim, pad; long long shape[4], stride[4]; };
__device__ __forceinline__ long long l2_at(const P8View &v, long long i) {
    long long off = v.base;
    for (int k = v.ndim - 1; k >= 0; --k) { const long long n = v.shape[k]; off += (i % n) * v.stride[k]; i /= n; }
    return off;
}
#define L2_RULES(T) \
extern "C" __global__ void l2_scale_tl_##T(T *a, P8View out, P8View x, double c) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= out.size) return; \
    a[l2_at(out, i)] = (T)(c * (double)a[l2_at(x, i)]); \
} \
extern "C" __global__ void l2_scale_ad_##T(T *a, P8View out, P8View x, double c) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= out.size) return; \
    const long long o = l2_at(out, i), s = l2_at(x, i); \
    const double b = (double)a[o]; a[o] = (T)0; a[s] = (T)((double)a[s] + c * b); \
} \
extern "C" __global__ void l2_zero_view_##T(T *a, P8View out) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < out.size) a[l2_at(out, i)] = (T)0; \
}
L2_RULES(float)
L2_RULES(double)
