// SPDX-License-Identifier: Apache-2.0
// Physics-on window, lane L1 (autodiff program): support kernels for the composed
// dispatcher in the P1 runner (src/runner/window.rs).
// The copy-row, view, where, max, zero and dot rules come from P8 (cuda/p8.cu,
// compiled beside this file); this file only adds what P8 has no kernel for.
// Shadow arena convention (P8 / U7): one float shadow element per forward 4-byte word.
#include <cstdint>

struct P8View { long long base, size; int ndim, pad; long long shape[4], stride[4]; };
__device__ __forceinline__ long long win_at(const P8View &v, long long i) {
    long long off = v.base;
    for (int k = v.ndim - 1; k >= 0; --k) { const long long n = v.shape[k]; off += (i % n) * v.stride[k]; i /= n; }
    return off;
}

// Strided zero: the shadow of every word an inactive writer overwrites (int -> float copies, fills).
extern "C" __global__ void win_view_zero_f32(float *a, P8View v) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < v.size;
         i += (long long)gridDim.x * blockDim.x)
        a[win_at(v, i)] = 0.0f;
}

// Reproducible uniform [-1,1) values from (seed, absolute element index), so a direction can be
// regenerated in chunks: out[i] = u(seed, first + i). Same mixer as P8's p8_u01.
__device__ __forceinline__ double win_u01(unsigned long long seed, unsigned long long k) {
    unsigned long long x = seed * 0x9E3779B97F4A7C15ULL ^ (k + 0x632BE59BD9B4E019ULL);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}
// scale (nullable): per-word weight, the relative direction (|primal| at the boundary the direction lives on).
extern "C" __global__ void win_fill_f32(float *out, long long first, long long n, unsigned long long seed, const float *scale) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float u = (float)(2.0 * win_u01(seed, (unsigned long long)(first + i)) - 1.0);
        out[i] = scale ? u * scale[i] : u;
    }
}

// Fixed-grid dot products in binary64 (grid 1024 x 256; partials added in index order on the host):
// part[3b] = sum a*s, part[3b+1] = sum a*(s - b), part[3b+2] = count of non-finite s.
extern "C" __global__ void win_dot2_f32(const float *a, const float *s, const float *b, long long n, double *part) {
    __shared__ double sh[3][256];
    double p = 0, q = 0, bad = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double u = (double)a[i], v = (double)s[i], w = (double)b[i];
        // a = 0 exactly (fixed allocations, zero weights): the word is not in the direction, skip it (0 * inf).
        if (u == 0.0) continue;
        if (!isfinite(v)) bad += 1.0;
        p += u * v; q += u * (v - w);
    }
    sh[0][threadIdx.x] = p; sh[1][threadIdx.x] = q; sh[2][threadIdx.x] = bad; __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) for (int c = 0; c < 3; ++c) sh[c][threadIdx.x] += sh[c][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) for (int c = 0; c < 3; ++c) part[3 * blockIdx.x + c] = sh[c][0];
}

// Diagnostic (--trace-norm 1 in the reverse): count of non-finite words (order-free count, never a result).
extern "C" __global__ void win_nonfinite_f32(const float *a, long long n, unsigned long long *count) {
    unsigned long long c = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        if (!isfinite(a[i])) ++c;
    if (c) atomicAdd(count, c);
}

// ---------------------------------------------------------------------------------------------------------------------
// Frozen FCT limiter (`ad-binder window --fct-frozen 1`, off by default; a deliberate choice). The flux
// limiter's coefficients are taken from the forward trajectory and held constant in the tangent and the adjoint, the
// standard treatment of a non-differentiable limiter in variational systems. This is an APPROXIMATION of the forward's
// derivative, named in every receipt: the limiter weights' own dependence on the state is dropped. Window rules:
//   fct_scale          -> its outputs scale_in / scale_out (the limiter coefficients) carry zero tangent; their cotangent
//                         is dropped (wrules R::Zero), so no derivative passes through the 1/(sum + 1e-20) ratio that
//                         overflowed float32 at degenerate tracer bounds (trace4, L1 2026-11-05 07:15Z);
//   fct_minmax_source  -> the kernels below: source_old = old + dt*source/rho_old keeps its exact derivative (it is
//                         transported state, read by fct_vertical_low_order and fct_edge_residual); the limiter bounds
//                         minimum / maximum carry none;
//   the other 7 FCT kernels (fct_limit_horizontal/vertical, the low-order, residual, finish and edge-value kernels)
//                      -> their ad.entry/v1 entries, live; fct_limit_* see a zero coefficient tangent, i.e. they are
//                         linear in the antidiffusive flux with the trajectory's frozen min(scale_out, scale_in) weights.
// Q3(t,k,c) = (t*nl + k)*nc + c; C2(k,c) = k*nc + c. One thread per (k,c) owns every tracer and the rho_old word, so the
// reverse needs no atomics and is bitwise repeatable. e_* are shadow element indices; source/rho are the primal inputs
// as they were at the launch (taped).
__device__ __forceinline__ float win_fct_current_v(float dt, float src, float r) { return __fdiv_rn(__fmul_rn(dt, src), r); }
extern "C" __global__ void win_fct_minmax_frozen_tl(float *a, int nt, int nl, int nc, float dt, const float *source, const float *rho,
                                                    long long e_old, long long e_src, long long e_rho, long long e_so,
                                                    long long e_min, long long e_max) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)nc * nl) return;
    const int c = (int)(id % nc), k = (int)(id / nc);
    const float r = rho[(long long)k * nc + c];
    const double drho = (double)a[e_rho + (long long)k * nc + c];
    for (int t = 0; t < nt; ++t) {
        const long long i = ((long long)t * nl + k) * nc + c;
        const double v = (double)win_fct_current_v(dt, source[i], r);
        const double d = (double)a[e_old + i] + ((double)dt * (double)a[e_src + i] - v * drho) / (double)r;
        a[e_so + i] = (float)d;
        a[e_min + i] = 0.0f;
        a[e_max + i] = 0.0f;
    }
}
extern "C" __global__ void win_fct_minmax_frozen_ad(float *a, int nt, int nl, int nc, float dt, const float *source, const float *rho,
                                                    long long e_old, long long e_src, long long e_rho, long long e_so,
                                                    long long e_min, long long e_max) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)nc * nl) return;
    const int c = (int)(id % nc), k = (int)(id / nc);
    const float r = rho[(long long)k * nc + c];
    double brho = 0.0;
    for (int t = 0; t < nt; ++t) {
        const long long i = ((long long)t * nl + k) * nc + c;
        const double g = (double)a[e_so + i];
        a[e_so + i] = 0.0f;
        a[e_min + i] = 0.0f;
        a[e_max + i] = 0.0f;
        const double v = (double)win_fct_current_v(dt, source[i], r);
        a[e_old + i] = (float)((double)a[e_old + i] + g);
        a[e_src + i] = (float)((double)a[e_src + i] + g * (double)dt / (double)r);
        brho -= g * v / (double)r;
    }
    a[e_rho + (long long)k * nc + c] = (float)((double)a[e_rho + (long long)k * nc + c] + brho);
}

// Taylor check (`--taylor h1,h2,..`): primal += h * x on the words where x is nonzero (x = the masked direction chunk).
extern "C" __global__ void win_axpy_primal_f32(float *y, const float *x, double h, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float u = x[i];
        if (u != 0.0f) y[i] = (float)((double)y[i] + h * (double)u);
    }
}

// Float32-shadow adapter (`--native-schemes 1`): window float32 shadows <-> an L2 row's float64 segment shadows.
extern "C" __global__ void win_widen_f32_f64(const float *a, long long e0, double *d, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        d[i] = (double)a[e0 + i];
}
extern "C" __global__ void win_narrow_f64_f32(float *a, long long e0, const double *d, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        a[e0 + i] = (float)d[i];
}
// 60 km tape (AD_P3_STRIPE=1): the same moves for `rows` runs of n words, `pitch` words apart on both sides (a banded
// MYNN-EDMF array's row stripes: the window's float32 shadow and the region have the same element pitch, ni). One launch
// instead of one per row; the same element writes, none shared between runs.
extern "C" __global__ void win_widen2_f32_f64(const float *a, long long e0, double *d, long long n, long long rows, long long pitch) {
    const long long tot = n * rows;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const long long k = i / n, o = k * pitch + (i - k * n);
        d[o] = (double)a[e0 + o];
    }
}
extern "C" __global__ void win_narrow2_f64_f32(float *a, long long e0, const double *d, long long n, long long rows, long long pitch) {
    const long long tot = n * rows;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
        const long long k = i / n, o = k * pitch + (i - k * n);
        a[e0 + o] = (float)d[o];
    }
}

// NOAA's model-top damping (hexcore/cuda_upper_damping_v841.py), on levels k >= nl - levels and columns (cells or
// edges) c < solved, the words the forward writes. The constants are formed exactly as the forward forms them.
//
// upper_cam_f32: values = max(values, floor_k), floor_k = 4 * 2.0833 * length * coefficient * (1 - (nl-1-k)/levels).
// The derivative is the 0/1 mask (pre >= floor_k) on those words; a diagonal map, so the tangent and the adjoint are
// the same launch. `pre` is the values array as it was before the forward launch (taped).
extern "C" __global__ void win_upper_cam(float *a, int nl, int nc, int solved, int levels, float length, float coef,
                                         long long e_v, const float *pre) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)nl * nc) return;
    const int k = (int)(id / nc), c = (int)(id % nc);
    if (c >= solved || k < nl - levels) return;
    const float base = __fmul_rn(__fmul_rn(__fmul_rn(4.0f, 2.0833f), length), coef);
    const float floor_k = __fmul_rn(base, __fsub_rn(1.0f, __fdiv_rn((float)(nl - 1 - k), (float)levels)));
    if (!(pre[id] >= floor_k)) a[e_v + id] = 0.0f;
}

// upper_rayleigh_f32: tendency -= rho * velocity * c_k, c_k = (k - (nl - levels) + 1) / (levels * days * 86400).
// Tangent: dt -= c_k (rho dv + v drho). Adjoint (the transpose): vbar -= c_k rho tbar, rhobar -= c_k v tbar, tbar kept.
// rho and velocity are read only by the forward; their primal values are taped before it (rho_p, vel_p).
extern "C" __global__ void win_upper_rayleigh(float *a, int nl, int ne, int solved, int levels, float days,
                                              long long e_t, long long e_rho, long long e_vel,
                                              const float *rho_p, const float *vel_p, int adjoint) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (long long)nl * ne) return;
    const int k = (int)(id / ne), e = (int)(id % ne);
    if (e >= solved || k < nl - levels) return;
    const float inverse = __fdiv_rn(1.0f, __fmul_rn((float)levels, __fmul_rn(days, 86400.0f)));
    const double c = (double)__fmul_rn((float)(k - (nl - levels) + 1), inverse);
    const double r = (double)rho_p[id], v = (double)vel_p[id];
    if (!adjoint) {
        a[e_t + id] = (float)((double)a[e_t + id] - c * (r * (double)a[e_vel + id] + v * (double)a[e_rho + id]));
    } else {
        const double tb = (double)a[e_t + id];
        a[e_vel + id] = (float)((double)a[e_vel + id] - c * r * tb);
        a[e_rho + id] = (float)((double)a[e_rho + id] - c * v * tb);
    }
}
