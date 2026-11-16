// SPDX-License-Identifier: Apache-2.0
// g3-radland: rrfs_radiation_unpack (rrfs/resident_radiation_rrfs.py:28, captured .cu 66409514). One launch moves up to
// 23 word blocks (blockIdx.y = op): for e < n, row = e / w, col = e % w: dst[row*a + col*b] = src ? src[row*c + col*f] : v.
// Derivative, per op: a copy (TL d_dst = d_src; AD b_src += b_dst, b_dst = 0) or, for a fill (src NULL, read from the
// pre-launch primal argument), a constant (d_dst = 0; b_dst consumed). Unused ops (n = 0; NULL pointers on the LW
// launch) do nothing. The AD is the exact transpose only if no two words share a destination or a source; the first
// thread of each op checks that its index maps are injective and that its spans do not overlap another op's
// destination or source span in a way the transpose cannot sum exactly (g3_pair_ok), else refusal slot 0. The TL runs
// in the forward geometry (op = blockIdx.y); the AD runs one thread per element index over all ops in order.
#include "g3_common.cuh"
#define G3_OPS 23
struct G3Op { const unsigned int* sp; float* d; float* s; long long a, b, c, f, n, w; };
__device__ __forceinline__ void g3_span(long long n, long long w, long long x, long long y, long long& lo, long long& hi) {
    long long rows = w > 0 ? (n + w - 1) / w : 0;
    lo = 0; hi = (rows > 0 && w > 0) ? (rows - 1) * x + (w - 1) * y + 1 : 0;
}
__device__ __forceinline__ bool g3_injective(long long n, long long w, long long x, long long y) {
    if (n <= 0) return true;
    if (x < 0 || y < 0 || w <= 0) return false;
    long long rows = (n + w - 1) / w;
    if (rows == 1) return w == 1 || y > 0;
    if (w == 1) return x > 0;
    return (y > 0 && x >= w * y) || (x > 0 && y >= rows * x);
}
// Two word sets of the launch may share words only in the ways the transpose below handles exactly:
//   source/source: the same map (a fan-out: the AD thread of element e adds both cotangents, in op order);
//   any pair: 1-D maps (one row) with the same stride whose origins differ by a non-multiple of it (interleaved).
// Everything else that overlaps by span is refused (slot 0). kind: 0 dst-dst, 1 src-src, 2 dst-src.
__device__ bool g3_pair_ok(const float* p, long long pn, long long pw, long long px, long long py,
                           const float* q, long long qn, long long qw, long long qx, long long qy, int kind) {
    long long pl, ph, ql, qh; g3_span(pn, pw, px, py, pl, ph); g3_span(qn, qw, qx, qy, ql, qh);
    if (!(p + pl < q + qh && q + ql < p + ph)) return true;                          // disjoint spans
    if (kind == 1 && p == q && pn == qn && pw == qw && px == qx && py == qy) return true;
    long long pr = (pn + pw - 1) / pw, qr = (qn + qw - 1) / qw;
    if (pr == 1 && qr == 1 && py == qy && py > 1 && ((long long)(p - q)) % py != 0) return true;
    // the same 2-D transpose map (unit row stride, rows <= column stride): words are (origin + r + col*y), r < rows, so
    // two origins whose offset falls outside [-(rows-1), rows-1] modulo y never share a word.
    if (pn == qn && pw == qw && px == qx && py == qy && px == 1 && pr <= py) {
        long long d = (((long long)(p - q)) % py + py) % py;
        if (d >= pr && d + pr <= py) return true;
    }
    return false;
}
__device__ bool g3_check(const G3Op* o, int k) {
    const G3Op& p = o[k];
    if (p.n <= 0) return true;
    if (!g3_injective(p.n, p.w, p.a, p.b)) return false;
    if (p.sp && !g3_injective(p.n, p.w, p.c, p.f)) return false;
    for (int j = 0; j < G3_OPS; ++j) {
        const G3Op& q = o[j]; if (q.n <= 0) continue;
        if (j != k && !g3_pair_ok(p.d, p.n, p.w, p.a, p.b, q.d, q.n, q.w, q.a, q.b, 0)) return false;
        if (q.sp && !g3_pair_ok(p.d, p.n, p.w, p.a, p.b, q.s, q.n, q.w, q.c, q.f, 2)) return false;
        if (j != k && p.sp && q.sp && !g3_pair_ok(p.s, p.n, p.w, p.c, p.f, q.s, q.n, q.w, q.c, q.f, 1)) return false;
    }
    return true;
}
#define G3A(N) const unsigned int* sp##N, float* d##N, float* s##N, long long a##N, long long b##N, long long c##N, \
               long long f##N, long long n##N, long long w##N
#define G3O(N) {sp##N, d##N, s##N, a##N, b##N, c##N, f##N, n##N, w##N}
#define G3_ARGS G3A(0), G3A(1), G3A(2), G3A(3), G3A(4), G3A(5), G3A(6), G3A(7), G3A(8), G3A(9), G3A(10), G3A(11), G3A(12), \
                G3A(13), G3A(14), G3A(15), G3A(16), G3A(17), G3A(18), G3A(19), G3A(20), G3A(21), G3A(22)
#define G3_LIST {G3O(0), G3O(1), G3O(2), G3O(3), G3O(4), G3O(5), G3O(6), G3O(7), G3O(8), G3O(9), G3O(10), G3O(11), G3O(12), \
                 G3O(13), G3O(14), G3O(15), G3O(16), G3O(17), G3O(18), G3O(19), G3O(20), G3O(21), G3O(22)}
extern "C" __global__ void rrfs_radiation_unpack__tl(G3_ARGS, int* failure) {
    const G3Op o[G3_OPS] = G3_LIST;
    int k = blockIdx.y; if (k >= G3_OPS) return;
    const G3Op& p = o[k];
    if (blockIdx.x == 0 && threadIdx.x == 0 && !g3_check(o, k)) g3_fail(failure, 0);
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (e >= p.n) return;
    long long row = e / p.w, col = e % p.w;
    p.d[row * p.a + col * p.b] = p.sp ? p.s[row * p.c + col * p.f] : 0.0f;
}
extern "C" __global__ void rrfs_radiation_unpack__ad(G3_ARGS, int* failure) {
    const G3Op o[G3_OPS] = G3_LIST;
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.x < G3_OPS && !g3_check(o, threadIdx.x)) g3_fail(failure, 0);
    for (int k = 0; k < G3_OPS; ++k) {           // ops in order: a word read by several ops gets its sums in one thread
        const G3Op& p = o[k]; if (e >= p.n) continue;
        long long row = e / p.w, col = e % p.w;
        float* bd = p.d + row * p.a + col * p.b;
        if (p.sp) g3_acc(p.s + row * p.c + col * p.f, *bd);
        *bd = 0.0f;
    }
}
