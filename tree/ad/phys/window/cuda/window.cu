// SPDX-License-Identifier: Apache-2.0
// Physics-on window, lane L1 (private autodiff program): support kernels for the composed
// dispatcher in the P1 runner (tree/ad/phys/binder/runtime/src/binder/phys/window.rs).
// The copy-row, view, where, max, zero and dot rules come from P8 (tree/ad/phys/p8/cuda/p8.cu,
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
