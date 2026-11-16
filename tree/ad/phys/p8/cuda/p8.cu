// SPDX-License-Identifier: Apache-2.0
// Autodiff physics rung, unit P8 (private): derivative rules for the small
// commands in the hrrrv5 physics slots, and the parallel transpose of the RRFS
// copy-row kernels (rrfs_post_transport_copy, rrfs_glue_copy_plan,
// cupy_concatenate_same_size).
//
// Shadow arena convention (same as ad/e2e U7): one shadow element per forward
// 4-byte word; a forward 8-byte element's shadow sits at the index of its
// first word. Every index below is an element index into one shadow arena.
// Built by build-cubin.sh with the strict forward flags.
#include <cstdint>

struct P8Row {             // one strided 4-D copy row, forward form of the RRFS copy kernels
    long long dst, src;    // arena element index of the view origin; src < 0: fill row (no source)
    long long ds[4], ss[4];
    long long n1, n2, n3;  // the forward kernel's decomposition: i3 = e % n3, i2, i1, i0 = rest
    long long count;
};

__device__ __forceinline__ void p8_offsets(const P8Row &r, long long e, long long &d, long long &s) {
    long long q = e;
    const long long i3 = q % r.n3; q /= r.n3;
    const long long i2 = q % r.n2; q /= r.n2;
    const long long i1 = q % r.n1; q /= r.n1;
    d = r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2] + i3 * r.ds[3];
    s = r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2] + i3 * r.ss[3];
}

// Tangent of a copy-row launch: d_dst = d_src (fill rows: 0). grid.y = row.
#define P8_ROWS_TL(T) \
extern "C" __global__ void p8_rows_tl_##T(T *a, const P8Row *rows) { \
    const P8Row r = rows[blockIdx.y]; \
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < r.count; \
         e += (long long)gridDim.x * blockDim.x) { \
        long long d, s; p8_offsets(r, e, d, s); \
        a[d] = r.src >= 0 ? a[s] : (T)0; \
    } \
}
P8_ROWS_TL(float)
P8_ROWS_TL(double)

// Parallel transpose, gather form. The host lists, for every distinct source
// element j, the destination visits that feed it in the exact order of the
// serial reference (rows last to first, elements ascending). Item = (dst id << 1)
// | repeat, where repeat marks a destination already visited earlier in that
// order (the serial kernel has zeroed it by then, so it contributes +0).
// Sum in binary64 starting from the old source value, one rounding to the
// element type: the same operations as glue_copy_add / glue_copy_flush.
#define P8_GATHER(T) \
extern "C" __global__ void p8_gather_ad_##T(T *a, const long long *src_word, const long long *start, \
        const long long *item, const long long *dst_word, long long nsrc, int *failure) { \
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (j >= nsrc) return; \
    T *p = a + src_word[j]; \
    const T old = *p; \
    double sum = (double)old; \
    bool bad = false; \
    for (long long k = start[j]; k < start[j + 1]; ++k) { \
        const long long it = item[k]; \
        const T seed = (it & 1) ? (T)0 : a[dst_word[it >> 1]]; \
        sum += (double)seed; \
        bad |= !isfinite((double)seed) || !isfinite(sum); \
        if (k == start[j]) bad |= !isfinite((double)old); \
    } \
    const T out = (T)sum; \
    bad |= !isfinite((double)out); \
    *p = out; \
    if (bad) atomicExch(failure, 1); \
}
P8_GATHER(float)
P8_GATHER(double)

// Direct form, used when the host build proved the launch injective: every
// destination and every source element is visited exactly once and no source is
// a destination. Then each source gets one contribution and the serial result
// is old + seed rounded once, which this computes in the forward's own
// coalesced traversal. Bitwise equal to the serial and gather forms.
#define P8_DIRECT(T) \
extern "C" __global__ void p8_rows_ad_direct_##T(T *a, const P8Row *rows, int *failure) { \
    const P8Row r = rows[blockIdx.y]; \
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < r.count; \
         e += (long long)gridDim.x * blockDim.x) { \
        long long d, s; p8_offsets(r, e, d, s); \
        const T seed = a[d]; a[d] = (T)0; \
        if (r.src >= 0) { \
            const T old = a[s]; const double sum = (double)old + (double)seed; const T out = (T)sum; a[s] = out; \
            if (!isfinite((double)old) || !isfinite((double)seed) || !isfinite(sum) || !isfinite((double)out)) atomicExch(failure, 1); \
        } else if (!isfinite((double)seed)) atomicExch(failure, 1); \
    } \
}
P8_DIRECT(float)
P8_DIRECT(double)

// After the gather: every destination word's cotangent is consumed (b_dst = 0).
// Seeds of destinations with no source (fill rows) are only checked.
#define P8_CLEAR(T) \
extern "C" __global__ void p8_clear_dst_##T(T *a, const long long *dst_word, const unsigned char *sourceless, \
        long long ndst, int *failure) { \
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (k >= ndst) return; \
    T *p = a + dst_word[k]; \
    if (sourceless[k] && !isfinite((double)*p)) atomicExch(failure, 1); \
    *p = (T)0; \
}
P8_CLEAR(float)
P8_CLEAR(double)

// Strided view (rank <= 4, element strides in arena elements).
struct P8View { long long base, size; int ndim, pad; long long shape[4], stride[4]; };
__device__ __forceinline__ long long p8_at(const P8View &v, long long i) {
    long long off = v.base;
    for (int k = v.ndim - 1; k >= 0; --k) { const long long n = v.shape[k]; off += (i % n) * v.stride[k]; i /= n; }
    return off;
}

// Elementwise copy (cupy_copy between float types; casts are linear and exact
// on the shadow). in_place (dst and src the same words): identity, nothing to do.
#define P8_VIEW(T) \
extern "C" __global__ void p8_view_tl_##T(T *a, P8View dst, P8View src) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= dst.size) return; \
    a[p8_at(dst, i)] = a[p8_at(src, i)]; \
} \
extern "C" __global__ void p8_view_ad_##T(T *a, P8View dst, P8View src) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= dst.size) return; \
    const long long d = p8_at(dst, i), s = p8_at(src, i); \
    const T b = a[d]; a[d] = (T)0; a[s] += b; \
}
P8_VIEW(float)
P8_VIEW(double)

// cupy_copy_where (float32, bool mask, float32): out[i] = mask[i] ? in[i] : out[i].
// The mask is the forward's own bool view (primal bytes); executed arm only.
#define P8_WHERE(T) \
extern "C" __global__ void p8_where_tl_##T(T *a, P8View dst, P8View src, const unsigned char *mask, P8View m) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= dst.size || !mask[p8_at(m, i)]) return; \
    a[p8_at(dst, i)] = a[p8_at(src, i)]; \
} \
extern "C" __global__ void p8_where_ad_##T(T *a, P8View dst, P8View src, const unsigned char *mask, P8View m) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= dst.size || !mask[p8_at(m, i)]) return; \
    const long long d = p8_at(dst, i), s = p8_at(src, i); \
    const T b = a[d]; a[d] = (T)0; a[s] += b; \
}
P8_WHERE(float)
P8_WHERE(double)

// cupy_maximum__float_float32_float32: out = (isnan(c)|isnan(x)) ? NaN : fmaxf(c, x),
// c a constant. Executed arm: x wins only when it is strictly larger and not NaN
// (at a tie the constant's arm is taken: zero derivative). xp = primal x before the launch.
#define P8_MAX(T) \
extern "C" __global__ void p8_max_tl_##T(T *a, P8View out, P8View x, const float *xp, P8View xv, float c) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= out.size) return; \
    const float v = xp[p8_at(xv, i)]; \
    const bool win = !isnan(v) && !isnan(c) && v > c; \
    a[p8_at(out, i)] = win ? a[p8_at(x, i)] : (T)0; \
} \
extern "C" __global__ void p8_max_ad_##T(T *a, P8View out, P8View x, const float *xp, P8View xv, float c) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= out.size) return; \
    const float v = xp[p8_at(xv, i)]; \
    const bool win = !isnan(v) && !isnan(c) && v > c; \
    const long long o = p8_at(out, i), s = p8_at(x, i); \
    const T b = a[o]; a[o] = (T)0; if (win) a[s] += b; \
}
P8_MAX(float)
P8_MAX(double)

// Inactive outputs (fills, integer and bool results, held or immutable
// uploads): the shadow of every written word is zero in both directions.
#define P8_ZERO(T) \
extern "C" __global__ void p8_zero_##T(T *a, long long first, long long n) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) a[first + i] = (T)0; \
}
P8_ZERO(float)
P8_ZERO(double)

// Test support: reproducible uniform [-1,1) values from (seed, element), NaN fill,
// and a fixed-grid dot product (partials added in order on the host).
__device__ __forceinline__ double p8_u01(unsigned long long seed, unsigned long long k) {
    unsigned long long x = seed * 0x9E3779B97F4A7C15ULL ^ (k + 0x632BE59BD9B4E019ULL);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}
#define P8_TEST(T) \
extern "C" __global__ void p8_fill_random_##T(T *a, long long first, long long n, unsigned long long seed) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) a[first + i] = (T)(2.0 * p8_u01(seed, (unsigned long long)(first + i)) - 1.0); \
} \
extern "C" __global__ void p8_sub_##T(T *a, const T *b, long long n) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) a[i] -= b[i]; \
} \
extern "C" __global__ void p8_fill_nan_##T(T *a, long long first, long long n) { \
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) a[first + i] = (T)NAN; \
} \
extern "C" __global__ void p8_dot3_##T(const T *x, const T *y, long long n, double *part) { \
    __shared__ double s[3][256]; \
    double ab = 0, aa = 0, bb = 0; \
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) { \
        const double u = (double)x[i], v = (double)y[i]; ab += u * v; aa += u * u; bb += v * v; \
    } \
    s[0][threadIdx.x] = ab; s[1][threadIdx.x] = aa; s[2][threadIdx.x] = bb; __syncthreads(); \
    for (int w = 128; w > 0; w >>= 1) { \
        if (threadIdx.x < w) for (int c = 0; c < 3; ++c) s[c][threadIdx.x] += s[c][threadIdx.x + w]; \
        __syncthreads(); \
    } \
    if (threadIdx.x == 0) for (int c = 0; c < 3; ++c) part[3 * blockIdx.x + c] = s[c][0]; \
}
P8_TEST(float)
P8_TEST(double)
