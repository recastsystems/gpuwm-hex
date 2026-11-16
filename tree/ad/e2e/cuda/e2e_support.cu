// SPDX-License-Identifier: Apache-2.0
// Device support for ad-e2e (U6): deterministic seed fills and fixed-order
// reductions. Every reduction uses a fixed grid (E2E_BLOCKS x 256) and a fixed
// shared-memory tree, so the same inputs give the same partial sums bit for bit.
// The host adds the per-block partials in index order.
#define E2E_THREADS 256

__device__ __forceinline__ unsigned long long e2e_mix(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
// Uniform in [-1, 1), from the 53 high bits of a counter hash.
__device__ __forceinline__ double e2e_unit(unsigned long long seed, unsigned long long i) {
    return (double)(e2e_mix(seed ^ e2e_mix(i)) >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

extern "C" __global__ void e2e_fill_f32(float *x, unsigned long long n, unsigned long long seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        x[i] = (float)e2e_unit(seed, i);
}
extern "C" __global__ void e2e_fill_f64(double *x, unsigned long long n, unsigned long long seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        x[i] = e2e_unit(seed, i);
}

__device__ __forceinline__ void e2e_block_sum(double acc, double *partial) {
    __shared__ double s[E2E_THREADS];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int width = E2E_THREADS / 2; width > 0; width >>= 1) {
        if (threadIdx.x < width) s[threadIdx.x] += s[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

// sum a * (b - c), with the difference formed in double (exact for f32 inputs).
// Unchanged words give an exact zero, so untouched memory adds nothing.
extern "C" __global__ void e2e_dotdiff_f32(const float *a, const float *b, const float *c,
                                           unsigned long long n, double *partial) {
    double acc = 0.0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        acc += (double)a[i] * ((double)b[i] - (c ? (double)c[i] : 0.0));
    e2e_block_sum(acc, partial);
}
extern "C" __global__ void e2e_dotdiff_f64(const double *a, const double *b, const double *c,
                                           unsigned long long n, double *partial) {
    double acc = 0.0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        acc += a[i] * (b[i] - (c ? c[i] : 0.0));
    e2e_block_sum(acc, partial);
}
// sum (b - c)^2; with c == 0 buffer this is a squared norm.
extern "C" __global__ void e2e_sqdiff_f32(const float *b, const float *c, unsigned long long n, double *partial) {
    double acc = 0.0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        double d = (double)b[i] - (c ? (double)c[i] : 0.0);
        acc += d * d;
    }
    e2e_block_sum(acc, partial);
}
extern "C" __global__ void e2e_sqdiff_f64(const double *b, const double *c, unsigned long long n, double *partial) {
    double acc = 0.0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        double d = b[i] - (c ? c[i] : 0.0);
        acc += d * d;
    }
    e2e_block_sum(acc, partial);
}
// Word counts: [0] words whose bits differ, [1] words of a with nonzero magnitude
// bits (sign ignored), [2] words of a equal to negative zero (f32 view).
extern "C" __global__ void e2e_words_u32(const unsigned *a, const unsigned *b, unsigned long long n,
                                         unsigned long long *counts) {
    unsigned long long differ = 0, nonzero = 0, negzero = 0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned x = a[i];
        if (b && x != b[i]) ++differ;
        if (x & 0x7fffffffu) ++nonzero;
        if (x == 0x80000000u) ++negzero;
    }
    if (differ) atomicAdd(counts, differ);
    if (nonzero) atomicAdd(counts + 1, nonzero);
    if (negzero) atomicAdd(counts + 2, negzero);
}

// Copy adjoint: the destination's cotangent moves into the source (src += dst; dst = 0).
extern "C" __global__ void e2e_move_add_f32(float *src, float *dst, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) { src[i] += dst[i]; dst[i] = 0.0f; }
}
extern "C" __global__ void e2e_move_add_f64(double *src, double *dst, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) { src[i] += dst[i]; dst[i] = 0.0; }
}
// Linear scale by a constant: x *= c (in-place multiply, tangent and adjoint alike).
extern "C" __global__ void e2e_scale_f32(float *x, double c, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) x[i] = (float)(c * (double)x[i]);
}
extern "C" __global__ void e2e_scale_f64(double *x, double c, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) x[i] = c * x[i];
}
// Out-of-place scale adjoint: in += c * out; out = 0.
extern "C" __global__ void e2e_axpy_clear_f32(float *in, float *out, double c, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) { in[i] = (float)((double)in[i] + c * (double)out[i]); out[i] = 0.0f; }
}
extern "C" __global__ void e2e_axpy_clear_f64(double *in, double *out, double c, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) { in[i] += c * out[i]; out[i] = 0.0; }
}

// Strided element copy (CuPy CArray views, up to 4 dimensions, element strides).
// Tangent: dst = src. Adjoint: src += dst; dst = 0. The dispatcher refuses
// broadcast sources and overlapping views, so no two threads touch one word.
struct E2EView { long long shape[4]; long long ss[4]; long long ds[4]; int ndim; };
template <class T>
__device__ void e2e_strided(T *src, T *dst, unsigned long long n, E2EView v, int adjoint) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long rest = i;
        long long so = 0, d = 0;
        for (int k = v.ndim - 1; k >= 0; --k) {
            long long idx = (long long)(rest % (unsigned long long)v.shape[k]);
            rest /= (unsigned long long)v.shape[k];
            so += idx * v.ss[k];
            d += idx * v.ds[k];
        }
        if (!adjoint) dst[d] = src[so];
        else { src[so] += dst[d]; dst[d] = (T)0; }
    }
}
extern "C" __global__ void e2e_strided_f32(float *src, float *dst, unsigned long long n, E2EView v, int adjoint) { e2e_strided(src, dst, n, v, adjoint); }
extern "C" __global__ void e2e_strided_f64(double *src, double *dst, unsigned long long n, E2EView v, int adjoint) { e2e_strided(src, dst, n, v, adjoint); }

// U9: adjoint of a broadcast strided copy (source stride 0 on some dims of
// length > 1). One thread per source element (the kept dims); it walks the
// broadcast dims in index order, so the sum is fixed-order and race free.
template <class T>
__device__ void e2e_bcast_ad(T *src, T *dst, E2EView v) {
    unsigned long long keep = 1, rep = 1;
    for (int k = 0; k < v.ndim; ++k) {
        if (v.shape[k] > 1 && v.ss[k] == 0) rep *= (unsigned long long)v.shape[k];
        else keep *= (unsigned long long)v.shape[k];
    }
    for (unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; t < keep;
         t += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long rest = t;
        long long so = 0, d0 = 0;
        for (int k = v.ndim - 1; k >= 0; --k) {
            if (v.shape[k] > 1 && v.ss[k] == 0) continue;
            long long idx = (long long)(rest % (unsigned long long)v.shape[k]);
            rest /= (unsigned long long)v.shape[k];
            so += idx * v.ss[k];
            d0 += idx * v.ds[k];
        }
        double acc = 0.0;
        for (unsigned long long b = 0; b < rep; ++b) {
            unsigned long long r = b;
            long long d = d0;
            for (int k = v.ndim - 1; k >= 0; --k) {
                if (!(v.shape[k] > 1 && v.ss[k] == 0)) continue;
                long long idx = (long long)(r % (unsigned long long)v.shape[k]);
                r /= (unsigned long long)v.shape[k];
                d += idx * v.ds[k];
            }
            acc += (double)dst[d];
            dst[d] = (T)0;
        }
        src[so] = (T)((double)src[so] + acc);
    }
}
extern "C" __global__ void e2e_bcast_ad_f32(float *src, float *dst, E2EView v) { e2e_bcast_ad(src, dst, v); }
extern "C" __global__ void e2e_bcast_ad_f64(double *src, double *dst, E2EView v) { e2e_bcast_ad(src, dst, v); }

// ---------------------------------------------------------------------------------------------------------------
// readbacks-device (round 3): the precision round trips and the failure checks that used to
// go through host memory on every derivative entry, done on the device. Every conversion is the one the host did:
// float -> double is exact; double -> float is round-to-nearest-even (Rust `as f32`, PTX cvt.rn.f32.f64, no ftz);
// the accumulate forms add in double, then round once, exactly as `(cur as f64 + acc) as f32`.
#define E2E_LOOP(i, n) for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < (n); \
                            i += (unsigned long long)gridDim.x * blockDim.x)
extern "C" __global__ void e2e_widen_f32_f64(const float *s, double *d, unsigned long long n) {
    E2E_LOOP(i, n) d[i] = (double)s[i];
}
extern "C" __global__ void e2e_narrow_f64_f32(const double *s, float *d, unsigned long long n) {
    E2E_LOOP(i, n) d[i] = __double2float_rn(s[i]);
}
// cur = replace ? (float)acc : (float)((double)cur + acc)
extern "C" __global__ void e2e_narrow_add_f64_f32(float *cur, const double *acc, unsigned long long n, int replace) {
    E2E_LOOP(i, n) cur[i] = __double2float_rn(replace ? acc[i] : __dadd_rn((double)cur[i], acc[i]));
}
// The TEMPO scatter: sh = (float)((accumulate ? (double)sh : 0.0) + r) (the TL path adds to +0.0 as the host did, so a
// -0 result stays +0), a non-finite result sets *bad, and the block partials of sum r^2 go to partial[blockIdx.x].
extern "C" __global__ void e2e_add_round_sq_f64_f32(float *sh, const double *r, unsigned long long n, int accumulate,
                                                    double *partial, unsigned *bad) {
    double acc = 0.0;
    unsigned b = 0;
    E2E_LOOP(i, n) {
        double x = r[i];
        acc += x * x;
        float w = __double2float_rn(__dadd_rn(accumulate ? (double)sh[i] : 0.0, x));
        if (!isfinite(w)) b = 1;
        sh[i] = w;
    }
    if (b) atomicOr(bad, 1u);
    e2e_block_sum(acc, partial);
}
// Rows of a [rows][nc] float64 vector to a column tile [rows][n] (dir 0) and back (dir 1).
extern "C" __global__ void e2e_tile_f64(const double *src, double *dst, unsigned long long rows, unsigned long long nc,
                                        unsigned long long c0, unsigned long long n, int dir) {
    E2E_LOOP(i, rows * n) {
        unsigned long long q = i / n, t = i % n;
        if (dir == 0) dst[i] = src[q * nc + c0 + t]; else dst[q * nc + c0 + t] = src[i];
    }
}
// adjoint-speed round 4 (readbacks-device, the last TEMPO readback): the count pass writes per-lane marks
// mk[(i * T + t) * 3 + f] (i <= steps, t < T = nc * 32; f = node counter, phase counter, record counter). The reverse
// sizes its column tiles from them. The host used to download all 3 x (steps + 1) x T ints and scan them on one core
// (x1.2562: 3 x 27 x 81,984 ints and 2.1 M lane visits per TEMPO launch; 64x that at 165k cells). Here, per (segment,
// column), cap is the largest over the 32 lanes of (u64)(i64)(node[sg+1] - node[sg]) + 64 (wrapping: the host's
// `as usize` then + 64), and pr[0] / pr[1] the largest (u64)(i64) phase delta / record mark over everything, maxed in
// the same unsigned order (pr zeroed by the caller, as the host's 0 start). Integer maxima: any order gives the host's
// values exactly. One warp is one (segment, column): T and the grid stride are multiples of 32, the block is 256.
extern "C" __global__ void e2e_tempo_marks(const int *mk, unsigned long long steps, unsigned long long nc,
                                           unsigned long long *cap, unsigned long long *pr) {
    const unsigned long long T = nc * 32, n = steps * T;
    unsigned long long pm = 0, rm = 0;
    E2E_LOOP(i, n) {
        const unsigned long long sg = i / T, t = i % T;
        const int *a = mk + (sg * T + t) * 3, *b = mk + ((sg + 1) * T + t) * 3;
        unsigned long long d0 = (unsigned long long)(long long)(b[0] - a[0]) + 64ull;
        const unsigned long long d1 = (unsigned long long)(long long)(b[1] - a[1]);
        const unsigned long long d2 = (unsigned long long)(long long)b[2];
        for (int o = 16; o > 0; o >>= 1) { const unsigned long long x = __shfl_xor_sync(0xffffffffu, d0, o); d0 = x > d0 ? x : d0; }
        if ((threadIdx.x & 31) == 0) cap[sg * nc + t / 32] = d0;
        pm = d1 > pm ? d1 : pm; rm = d2 > rm ? d2 : rm;
    }
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long x = __shfl_xor_sync(0xffffffffu, pm, o), y = __shfl_xor_sync(0xffffffffu, rm, o);
        pm = x > pm ? x : pm; rm = y > rm ? y : rm;
    }
    if ((threadIdx.x & 31) == 0) { if (pm) atomicMax(pr, pm); if (rm) atomicMax(pr + 1, rm); }
}
// P3 failure slots: four int codes per column; cnt[s] += columns whose code s is nonzero (integer counts, any order).
extern "C" __global__ void e2e_codes4_i32(const int *fl, unsigned long long ncols, unsigned long long *cnt) {
    unsigned long long c[4] = {0, 0, 0, 0};
    E2E_LOOP(i, ncols) { for (int s = 0; s < 4; ++s) if (fl[4 * i + s] != 0) ++c[s]; }
    for (int s = 0; s < 4; ++s) if (c[s]) atomicAdd(cnt + s, c[s]);
}
// Any a[i * stride + lane] != 0 (i < n) sets *flag.
extern "C" __global__ void e2e_any_lane_i32(const int *a, unsigned long long n, unsigned long long stride,
                                            unsigned long long lane, unsigned *flag) {
    unsigned b = 0;
    E2E_LOOP(i, n) if (a[i * stride + lane] != 0) b = 1;
    if (b) atomicOr(flag, 1u);
}
// The first nonzero a[i] (smallest base + i) and its value: key = ((0xffffffff - (base + i)) << 32) | a[i], max kept
// (0 means none), so the readback names the same column and bits the host scan did.
extern "C" __global__ void e2e_first_i32(const int *a, unsigned long long n, unsigned long long base, unsigned long long *key) {
    unsigned long long best = 0;
    E2E_LOOP(i, n) if (a[i] != 0) {
        unsigned long long k = ((0xffffffffULL - (base + i)) << 32) | (unsigned)a[i];
        if (k > best) best = k;
    }
    if (best) atomicMax(key, best);
}
// Words whose bits differ: float32 a vs float32 b (w == 4), or (float)a vs b with a float64 (w == 8).
extern "C" __global__ void e2e_diff_words(const void *a, const float *b, unsigned long long n, int w, unsigned long long *count) {
    unsigned long long d = 0;
    E2E_LOOP(i, n) {
        float x = w == 8 ? __double2float_rn(((const double *)a)[i]) : ((const float *)a)[i];
        if (__float_as_uint(x) != __float_as_uint(b[i])) ++d;
    }
    if (d) atomicAdd(count, d);
}

// tape-dedupe-and-pack (blocks): out[k] = 1 when the 32-bit words of pair k differ. tab holds (a, b, words) per pair; one
// thread block per pair (grid-stride over pairs).
extern "C" __global__ void e2e_pairs_differ(const unsigned long long *tab, unsigned long long n, unsigned *out) {
    for (unsigned long long k = blockIdx.x; k < n; k += gridDim.x) {
        const unsigned *a = (const unsigned *)tab[3 * k], *b = (const unsigned *)tab[3 * k + 1];
        const unsigned long long w = tab[3 * k + 2];
        int dif = 0;
        for (unsigned long long i = threadIdx.x; i < w; i += blockDim.x) dif |= a[i] != b[i];
        dif = __syncthreads_or(dif);
        if (threadIdx.x == 0) out[k] = dif ? 1u : 0u;
    }
}
// tape-dedupe-and-pack (blocks): copy run k's words to dst + tab[3k] from tab[3k + 1] (tab[3k + 2] words each).
extern "C" __global__ void e2e_runs_copy(unsigned long long dst, const unsigned long long *tab, unsigned long long n) {
    for (unsigned long long k = blockIdx.x; k < n; k += gridDim.x) {
        unsigned *o = (unsigned *)(dst + tab[3 * k]);
        const unsigned *s = (const unsigned *)tab[3 * k + 1];
        const unsigned long long w = tab[3 * k + 2];
        for (unsigned long long i = threadIdx.x; i < w; i += blockDim.x) o[i] = s[i];
    }
}
