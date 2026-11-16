// SPDX-License-Identifier: Apache-2.0
// g3-radland: CuPy take (gather) and scatter_update (a[idx] = v) of the RRTMG radiation slot (the sunlit-column
// selection and its write-back), as hand ad.entry/v1 rows. Every array arrives as the fixed-rank CArray the runner's
// `carray` scalar binding builds (deriv.rs, g3-radland): {data, size, nd, shape[4], strides[4]} with byte strides.
// CuPy semantics (cupy/_core/_routines_indexing.pyx):
//   take:    out[i] = a[(li*range + wrap(idx[i]))*rdim + ri], li = i/(rdim*cdim), ri = i%rdim, wrap = mod range;
//   scatter: a[(li*adim + wrap(idx[i]))*rdim + ri] = v[i], li = i/(rdim*cdim), ri = i%rdim, wrap = mod adim;
// `a` is a raw array (flat index unravelled over its shape, row-major, with its strides); idx and out/v are elementwise
// operands (the element index i unravelled over the output's / v's shape, each with its own strides).
// Index width: CuPy names both int32 and int64 index launches the same; strides are multiples of the item size, so a
// smallest moving byte stride of 4 means int32. Otherwise the index is read as int64 and must lie in [0, 2^31) (an
// int32 array read 8 bytes wide shows its neighbour in the high word and fails that); else refusal slot 1.
// Derivative: copies (TL d_out = d_a[k]; AD b_a[k] += b_out, b_out consumed / scatter: TL d_a[k] = d_v; AD b_v += b_a[k],
// b_a[k] consumed). Exactness needs k(i) injective (a repeated index makes the take adjoint a fan-in and the forward
// scatter order-dependent): a claims workspace counts each k first; a launch with a repeated k is refused (slot 0).
// Launch order: zero the claims, count, then the copy (TL and AD alike).
#include "g3_common.cuh"
struct G3CA { const char* data; long long size; long long nd; long long shape[4]; long long strides[4]; };
// byte offset of flat element index e over shape (row-major), with the array's own strides
__device__ __forceinline__ long long g3_off(const G3CA& A, long long e) {
    long long off = 0;
    for (int d = (int)A.nd - 1; d >= 0; --d) { long long n = A.shape[d] > 0 ? A.shape[d] : 1; off += (e % n) * A.strides[d]; e /= n; }
    return off;
}
// byte offset in operand X of the element whose multi-index is element e of shape S (the elementwise broadcast)
__device__ __forceinline__ long long g3_boff(const G3CA& S, const G3CA& X, long long e) {
    long long off = 0;
    for (int d = (int)S.nd - 1; d >= 0; --d) { long long n = S.shape[d] > 0 ? S.shape[d] : 1; long long id = e % n; e /= n;
        int dx = d - (int)(S.nd - X.nd); if (dx >= 0) off += (X.shape[dx] == 1 ? 0 : id) * X.strides[dx]; }
    return off;
}
__device__ __forceinline__ int g3_iw(const G3CA& I) {
    long long s = 0;
    for (int d = 0; d < I.nd; ++d) if (I.shape[d] > 1 && I.strides[d] != 0) { long long a = I.strides[d] < 0 ? -I.strides[d] : I.strides[d]; if (s == 0 || a < s) s = a; }
    return s == 4 ? 4 : 8;
}
// the wrapped index of element e, or -1 (refuse) if it cannot be read safely
__device__ __forceinline__ long long g3_idx(const G3CA& S, const G3CA& I, const char* ip, long long e, long long range) {
    const char* p = ip + g3_boff(S, I, e);   // ip: the pre-launch primal copy of the index array (its origin)
    long long v;
    if (g3_iw(I) == 4) v = *(const int*)p;
    else { v = *(const long long*)p; if (v < 0 || v >= (1ll << 31)) return -1; }
    if (range <= 0) return -1;
    long long w = v % range; if (w < 0) w += range;
    return w;
}
#if G3_KERNEL == 21
// take: S = out, flat source index into a
__device__ __forceinline__ long long g3_k(const G3CA& A, const G3CA& I, const G3CA& O, long long ldim, long long cdim, long long rdim,
                                          long long range, const char* ip, long long i) {
    long long w = g3_idx(O, I, ip, i, range); if (w < 0) return -1;
    long long li = i / (rdim * cdim), ri = i % rdim;
    return (li * range + w) * rdim + ri;
}
#define G3_TP G3CA A, G3CA I, G3CA O, long long ldim, long long cdim, long long rdim, long long range
extern "C" __global__ void cupy_take__zero(G3_TP, long long cap, int* claims, const char* ip, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; for (long long j = i; j < cap; j += (long long)gridDim.x * blockDim.x) claims[j] = 0;
}
extern "C" __global__ void cupy_take__count(G3_TP, long long cap, int* claims, const char* ip, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= O.size) return;
    long long k = g3_k(A, I, O, ldim, cdim, rdim, range, ip, i);
    if (k < 0 || k >= cap || k >= A.size) { g3_fail(failure, 1); return; }
    atomicAdd(claims + k, 1);
}
extern "C" __global__ void cupy_take__tl(G3_TP, long long cap, const int* claims, const char* ip, const float* d_a, float* d_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= O.size) return;
    long long k = g3_k(A, I, O, ldim, cdim, rdim, range, ip, i);
    if (k < 0 || k >= cap || k >= A.size) { g3_fail(failure, 1); return; }
    if (d_a == d_o) { g3_fail(failure, 0); return; }
    *(float*)((char*)d_o + g3_boff(O, O, i)) = *(const float*)((const char*)d_a + g3_off(A, k));
}
extern "C" __global__ void cupy_take__ad(G3_TP, long long cap, const int* claims, const char* ip, float* b_a, float* b_o, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= O.size) return;
    long long k = g3_k(A, I, O, ldim, cdim, rdim, range, ip, i);
    if (k < 0 || k >= cap || k >= A.size) { g3_fail(failure, 1); return; }
    if (b_a == b_o || claims[k] != 1) { g3_fail(failure, 0); return; }     // repeated index: a fan-in, refused
    float* bo = (float*)((char*)b_o + g3_boff(O, O, i));
    g3_acc((float*)((char*)b_a + g3_off(A, k)), G3D(*bo)); *bo = 0.0f;
}
#elif G3_KERNEL == 22
// scatter_update: S = v, flat destination index into a
__device__ __forceinline__ long long g3_k(const G3CA& V, const G3CA& I, long long cdim, long long rdim, long long adim, const char* ip, long long i) {
    long long w = g3_idx(V, I, ip, i, adim); if (w < 0) return -1;
    long long li = i / (rdim * cdim), ri = i % rdim;
    return (li * adim + w) * rdim + ri;
}
#define G3_SP G3CA V, G3CA I, long long cdim, long long rdim, long long adim, G3CA A
extern "C" __global__ void cupy_scatter_update__zero(G3_SP, long long cap, int* claims, const char* ip, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; for (long long j = i; j < cap; j += (long long)gridDim.x * blockDim.x) claims[j] = 0;
}
extern "C" __global__ void cupy_scatter_update__count(G3_SP, long long cap, int* claims, const char* ip, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= V.size) return;
    long long k = g3_k(V, I, cdim, rdim, adim, ip, i);
    if (k < 0 || k >= cap || k >= A.size) { g3_fail(failure, 1); return; }
    atomicAdd(claims + k, 1);
}
extern "C" __global__ void cupy_scatter_update__tl(G3_SP, long long cap, const int* claims, const char* ip, const float* d_v, float* d_a, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= V.size) return;
    long long k = g3_k(V, I, cdim, rdim, adim, ip, i);
    if (k < 0 || k >= cap || k >= A.size) { g3_fail(failure, 1); return; }
    if (d_v == d_a || claims[k] != 1) { g3_fail(failure, 0); return; }    // repeated index: order-dependent forward
    *(float*)((char*)d_a + g3_off(A, k)) = *(const float*)((const char*)d_v + g3_boff(V, V, i));
}
extern "C" __global__ void cupy_scatter_update__ad(G3_SP, long long cap, const int* claims, const char* ip, float* b_v, float* b_a, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= V.size) return;
    long long k = g3_k(V, I, cdim, rdim, adim, ip, i);
    if (k < 0 || k >= cap || k >= A.size) { g3_fail(failure, 1); return; }
    if (b_v == b_a || claims[k] != 1) { g3_fail(failure, 0); return; }
    float* ba = (float*)((char*)b_a + g3_off(A, k));
    g3_acc((float*)((char*)b_v + g3_boff(V, V, i)), G3D(*ba)); *ba = 0.0f;
}
#endif
