// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// Tangent-linear (TL) and adjoint (AD) entries for six hex transport / FCT
// kernels of tree/src/hexcore/cuda_transport.py at pin 16251589:
//   transport_edge_values   (:51)   transport_vertical_flux (:128)
//   fct_minmax_source       (:239)  fct_scale               (:395)
//   fct_limit_horizontal    (:419)  fct_limit_vertical      (:442)
// The forward kernels are a transcription of MPAS-A (BSD-3-Clause, see
// LICENSE-MPAS-BSD-3-Clause.txt); this derivative code is new, hand written
// from that forward source. The forward is never edited or recompiled here.
//
// Derivative policy (every min/max, sign, abs and clamp):
//   executed arm. The derivative follows the operand the forward actually
//   returned; exact ties take the LEFT operand, matching mpas_min/mpas_max
//   (fallback "right < left ? right : left", and fminf/fmaxf return a value
//   equal to the left operand on a tie). The limiter factors are
//   differentiated (never frozen): d(min(a,b)) follows the winner and
//   d(clamp) is the slope of the executed clamp arm.
//   copysign(1,v) and the sign arm of |v| are piecewise constant: zero
//   derivative; |v| at v == 0 takes the zero subgradient.
// Types: derivative arithmetic is always double inside each entry, with one
// rounding on store (COMMON-AD 44.1, 39.4/40.2 form). AD_REAL is the shadow
// storage type: float = production build, double = float64-promoted build.
// The primal stays float32 in both.
//
// Primal binding: every primal argument is the PRE-launch value of that
// argument (for in-place arguments this is the value before the forward
// overwrote it). TL entries of in-place kernels therefore run before the
// forward launch, or with the undo-log copy of the overwritten argument.
//
// Adjoint seeds of pure outputs are read only; the dispatcher zeroes them
// after the reverse entries (spec "after":"zero"). Seeds of in-place
// arguments are converted in place into the input cotangent (spec io
// "inplace"). Every sum is in a fixed order: no atomics, two reverse runs
// are bitwise identical.

#include "u2_fp32_helpers.cuh"

#ifndef AD_REAL
#define AD_REAL float
#endif
typedef AD_REAL T;   // derivative storage (shadow) type
typedef double T64;  // shadow type of fields declared "shadow_dtype": "f64" (COMMON-AD 49.2)
typedef double R;    // derivative arithmetic: always float64 inside, one rounding on store (COMMON-AD 44.1, 39.4/40.2 form)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define Q3(q,k,c,nl,nc) ((((q)*(nl) + (k))*(nc)) + (c))
#define QE3(q,k,e,nl,ne) ((((q)*(nl) + (k))*(ne)) + (e))
#define QI3(q,k,c,nl,nc) ((((q)*((nl)+1) + (k))*(nc)) + (c))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,width) ((e)*(width) + (s))

__constant__ float mpas_third_order_denominator = 12.0f;

// Arm predicates that mirror mpas_min / mpas_max exactly (IEEE compare of
// the decoded values, so an FTZ compile cannot hide a subnormal ordering).
__device__ __forceinline__ bool u2_min_right(float l, float r) {
    if (isnan(l)) return !isnan(r);
    return mpas_f32_to_f64_ieee(r) < mpas_f32_to_f64_ieee(l);
}
__device__ __forceinline__ bool u2_max_right(float l, float r) {
    if (isnan(l)) return !isnan(r);
    return mpas_f32_to_f64_ieee(r) > mpas_f32_to_f64_ieee(l);
}
__device__ __forceinline__ bool u2_pos(float x) { return mpas_f32_to_f64_ieee(x) > 0.0; }
__device__ __forceinline__ bool u2_neg(float x) { return mpas_f32_to_f64_ieee(x) < 0.0; }
__device__ __forceinline__ R rr(float x) { return (R)mpas_f32_to_f64_ieee(x); }
__device__ __forceinline__ float u2_plus(float x) { return mpas_max(0.0f, x); }
__device__ __forceinline__ float u2_minus(float x) { return mpas_min(0.0f, x); }

// =====================================================================
// transport_edge_values: value(t,k,e) = sum_s w(e,s,k) * stage(t,k,cell_s)
// w = adv + coefficient*sign(velocity)*adv3. Active: stage, velocity
// (zero derivative, executed sign arm). adv, adv3, coefficient: geometry /
// namelist, inactive.
// =====================================================================
__device__ __forceinline__ float u2_edge_weight(float a, float a3, float coef, float v) {
    return mpas_add(a, mpas_mul(mpas_mul(coef, mpas_copysign(1.0f, v)), a3));
}

extern "C" __global__ void transport_edge_values__tl(
    int nt, int nl, int nc, int ne, int width, float coef,
    const float *stage, const float *velocity, const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells,
    const T *d_stage, T *d_edge_values)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= ne * nl) return;
    const int e = id % ne, k = id / ne;
    const float v = velocity[E2(k, e, ne)];
    for (int t = 0; t < nt; ++t) {
        R z = (R)0;
        for (int s = 0; s < n_adv[e]; ++s) {
            const int a = ADV(e, s, width);
            const float w = u2_edge_weight(adv[a], adv3[a], coef, v);
            z = z + rr(w) * d_stage[Q3(t, k, adv_cells[a], nl, nc)];
        }
        d_edge_values[QE3(t, k, e, nl, ne)] = z;
    }
}

// Gather transpose. inv_off[nc+1], inv_ent[]: for cell c the (e*width+s)
// stencil entries whose adv_cells == c, ascending (derived once from the
// fixed integer topology).
extern "C" __global__ void transport_edge_values__ad(
    int nt, int nl, int nc, int ne, int width, float coef,
    const float *stage, const float *velocity, const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells,
    const int *inv_off, const int *inv_ent,
    const T *b_edge_values, T *b_stage)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    for (int t = 0; t < nt; ++t) {
        R z = (R)0;
        for (int p = inv_off[c]; p < inv_off[c + 1]; ++p) {
            const int a = inv_ent[p], e = a / width;
            const float w = u2_edge_weight(adv[a], adv3[a], coef, velocity[E2(k, e, ne)]);
            z = z + rr(w) * b_edge_values[QE3(t, k, e, nl, ne)];
        }
        const int i = Q3(t, k, c, nl, nc);
        b_stage[i] = b_stage[i] + z;
    }
    // velocity: zero derivative on the executed copysign arm; its cotangent
    // is left unchanged (accumulate zero).
}

// Branch record: sign arm of velocity for every output element.
extern "C" __global__ void transport_edge_values__arms(
    int nt, int nl, int nc, int ne, const float *velocity, unsigned char *arms)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= ne * nl) return;
    const int e = id % ne, k = id / ne;
    const unsigned char code = (__float_as_uint(velocity[E2(k, e, ne)]) >> 31) ? 1 : 0;
    for (int t = 0; t < nt; ++t) arms[QE3(t, k, e, nl, ne)] = code;
}

// =====================================================================
// transport_vertical_flux. Active: stage, velocity. fzm, fzp, coefficient
// inactive (geometry / namelist).
// =====================================================================
__device__ __forceinline__ R u2_dabs(float v, R dv) {
    return u2_pos(v) ? dv : (u2_neg(v) ? -dv : (R)0);
}

extern "C" __global__ void transport_vertical_flux__tl(
    int nt, int nl, int nc, float coef,
    const float *stage, const float *velocity, const float *fzm, const float *fzp,
    const T *d_stage, const T *d_velocity, T *d_vertical_flux)
{
    const int total = nt * (nl + 1) * nc;
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= total) return;
    const int c = id % nc, rest = id / nc, i = rest % (nl + 1), t = rest / (nl + 1);
    R result = (R)0;
    if (i > 0 && i < nl) {
        const float vel = velocity[C2(i, c, nc)];
        const R dvel = d_velocity[C2(i, c, nc)];
        if (i == 1 || i == nl - 1) {
            const float inner = mpas_add(mpas_mul(fzm[i], stage[Q3(t, i, c, nl, nc)]),
                                         mpas_mul(fzp[i], stage[Q3(t, i - 1, c, nl, nc)]));
            const R dinner = rr(fzm[i]) * d_stage[Q3(t, i, c, nl, nc)]
                           + rr(fzp[i]) * d_stage[Q3(t, i - 1, c, nl, nc)];
            result = dvel * rr(inner) + rr(vel) * dinner;
        } else {
            const float qim2 = stage[Q3(t, i - 2, c, nl, nc)], qim1 = stage[Q3(t, i - 1, c, nl, nc)];
            const float qi = stage[Q3(t, i, c, nl, nc)], qip1 = stage[Q3(t, i + 1, c, nl, nc)];
            const R dqim2 = d_stage[Q3(t, i - 2, c, nl, nc)], dqim1 = d_stage[Q3(t, i - 1, c, nl, nc)];
            const R dqi = d_stage[Q3(t, i, c, nl, nc)], dqip1 = d_stage[Q3(t, i + 1, c, nl, nc)];
            const float S = mpas_sub(mpas_mul(7.0f, mpas_add(qi, qim1)), mpas_add(qip1, qim2));
            const float T = mpas_sub(mpas_sub(qip1, qim2), mpas_mul(3.0f, mpas_sub(qi, qim1)));
            const R dS = (R)7 * (dqi + dqim1) - (dqip1 + dqim2);
            const R dT = (dqip1 - dqim2) - (R)3 * (dqi - dqim1);
            const R den = rr(mpas_third_order_denominator);
            const R dbase = (dvel * rr(S) + rr(vel) * dS) / den;
            const R dcorr = rr(coef) * (u2_dabs(vel, dvel) * rr(T) + rr(mpas_abs(vel)) * dT) / den;
            result = dbase + dcorr;
        }
    }
    d_vertical_flux[QI3(t, i, c, nl, nc)] = result;
}

// Cotangents of the two brackets of interface i for tracer t (interior
// interfaces: base S and correction T; boundary interfaces 1 and nl-1: the
// two-point sum).
__device__ __forceinline__ void u2_vf_brackets(int t, int i, int c, int nl, int nc, float coef,
        const float *velocity, const T *b_vertical_flux, R *bS, R *bT) {
    const float vel = velocity[C2(i, c, nc)];
    const R g = b_vertical_flux[QI3(t, i, c, nl, nc)];
    const R den = rr(mpas_third_order_denominator);
    *bS = (g / den) * rr(vel);
    *bT = (rr(coef) * g / den) * rr(mpas_abs(vel));
}

// Reverse entry 1 of 2: gather. One thread per (tracer, level, cell) sums, in
// double and in fixed interface order, every use of stage(t,k,c), then rounds
// once into b_stage.
extern "C" __global__ void transport_vertical_flux__ad_stage(
    int nt, int nl, int nc, float coef,
    const float *stage, const float *velocity, const float *fzm, const float *fzp,
    const T *b_vertical_flux, T *b_stage)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nt * nl * nc) return;
    const int c = id % nc, rest = id / nc, k = rest % nl, t = rest / nl;
    R acc = (R)0;
    for (int i = k - 1; i <= k + 2; ++i) {           // interfaces that read level k
        if (i < 1 || i > nl - 1) continue;
        if (i == 1 || i == nl - 1) {
            const R gi = (R)b_vertical_flux[QI3(t, i, c, nl, nc)] * rr(velocity[C2(i, c, nc)]);
            if (k == i) acc = acc + gi * rr(fzm[i]);
            else if (k == i - 1) acc = acc + gi * rr(fzp[i]);
        } else {
            R bS, bT;
            u2_vf_brackets(t, i, c, nl, nc, coef, velocity, b_vertical_flux, &bS, &bT);
            if (k == i)          acc = acc + (R)7 * bS - (R)3 * bT;
            else if (k == i - 1) acc = acc + (R)7 * bS + (R)3 * bT;
            else if (k == i + 1) acc = acc + (-bS + bT);
            else                 acc = acc + (-bS - bT);    // k == i - 2
        }
    }
    const int q = Q3(t, k, c, nl, nc);
    b_stage[q] = (R)b_stage[q] + acc;
}

// Reverse entry 2 of 2: one thread per (interface, cell) sums over tracers.
extern "C" __global__ void transport_vertical_flux__ad_velocity(
    int nt, int nl, int nc, float coef,
    const float *stage, const float *velocity, const float *fzm, const float *fzp,
    const T *b_vertical_flux, T *b_velocity)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (nl + 1) * nc) return;
    const int c = id % nc, i = id / nc;
    if (i < 1 || i > nl - 1) return;
    const float vel = velocity[C2(i, c, nc)];
    const R den = rr(mpas_third_order_denominator);
    R bvel = (R)0;
    for (int t = 0; t < nt; ++t) {
        const R g = b_vertical_flux[QI3(t, i, c, nl, nc)];
        if (i == 1 || i == nl - 1) {
            const float inner = mpas_add(mpas_mul(fzm[i], stage[Q3(t, i, c, nl, nc)]),
                                         mpas_mul(fzp[i], stage[Q3(t, i - 1, c, nl, nc)]));
            bvel = bvel + g * rr(inner);
        } else {
            const float qim2 = stage[Q3(t, i - 2, c, nl, nc)], qim1 = stage[Q3(t, i - 1, c, nl, nc)];
            const float qi = stage[Q3(t, i, c, nl, nc)], qip1 = stage[Q3(t, i + 1, c, nl, nc)];
            const float sb = mpas_sub(mpas_mul(7.0f, mpas_add(qi, qim1)), mpas_add(qip1, qim2));
            const float tb = mpas_sub(mpas_sub(qip1, qim2), mpas_mul(3.0f, mpas_sub(qi, qim1)));
            bvel = bvel + (g / den) * rr(sb) + u2_dabs(vel, (rr(coef) * g / den) * rr(tb));
        }
    }
    b_velocity[C2(i, c, nc)] = (R)b_velocity[C2(i, c, nc)] + bvel;
}

extern "C" __global__ void transport_vertical_flux__arms(
    int nt, int nl, int nc, const float *velocity, unsigned char *arms)
{
    const int total = nt * (nl + 1) * nc;
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= total) return;
    const int c = id % nc, rest = id / nc, i = rest % (nl + 1);
    unsigned char code = 0;
    if (i > 1 && i < nl - 1) {
        const float v = velocity[C2(i, c, nc)];
        code = u2_pos(v) ? 1 : (u2_neg(v) ? 2 : 3);
    }
    arms[id] = code;
}

// =====================================================================
// fct_minmax_source. current = old + dt*source/rho_old; minimum/maximum
// over {self, k-1, k+1, horizontal neighbours in slot order}, running
// mpas_min / mpas_max. Winner codes: 0 self, 1 level k-1, 2 level k+1,
// 3+s neighbour slot s. Active: old, source, rho_old. dt inactive.
// =====================================================================
__device__ __forceinline__ float u2_current(const float *old, const float *src, const float *rho,
                                            float dt, int t, int k, int c, int nl, int nc) {
    return mpas_add(old[Q3(t, k, c, nl, nc)],
                    mpas_div(mpas_mul(dt, src[Q3(t, k, c, nl, nc)]), rho[C2(k, c, nc)]));
}
__device__ __forceinline__ R u2_dcurrent(const float *src, const float *rho, float dt,
                                         const T *dold, const T *dsrc, const T *drho,
                                         int t, int k, int c, int nl, int nc) {
    const int i = Q3(t, k, c, nl, nc);
    const float r = rho[C2(k, c, nc)];
    const float v = mpas_div(mpas_mul(dt, src[i]), r);
    return dold[i] + (rr(dt) * dsrc[i] - rr(v) * drho[C2(k, c, nc)]) / rr(r);
}
__device__ __forceinline__ void u2_minmax_winners(
    int t, int k, int c, int nl, int nc, int max_edges, float dt,
    const int *n_edges_on_cell, const int *cells_on_cell,
    const float *old, const float *src, const float *rho,
    unsigned char *wmin, unsigned char *wmax)
{
    float low = u2_current(old, src, rho, dt, t, k, c, nl, nc), high = low;
    unsigned char lo = 0, hi = 0;
    if (k > 0) {
        const float v = u2_current(old, src, rho, dt, t, k - 1, c, nl, nc);
        if (u2_min_right(low, v)) lo = 1;
        if (u2_max_right(high, v)) hi = 1;
        low = mpas_min(low, v); high = mpas_max(high, v);
    }
    if (k + 1 < nl) {
        const float v = u2_current(old, src, rho, dt, t, k + 1, c, nl, nc);
        if (u2_min_right(low, v)) lo = 2;
        if (u2_max_right(high, v)) hi = 2;
        low = mpas_min(low, v); high = mpas_max(high, v);
    }
    for (int s = 0; s < n_edges_on_cell[c]; ++s) {
        const int n = cells_on_cell[CES(c, s, max_edges)];
        const float v = u2_current(old, src, rho, dt, t, k, n, nl, nc);
        if (u2_min_right(low, v)) lo = (unsigned char)(3 + s);
        if (u2_max_right(high, v)) hi = (unsigned char)(3 + s);
        low = mpas_min(low, v); high = mpas_max(high, v);
    }
    *wmin = lo; *wmax = hi;
}

// Branch record / AD workspace: winner codes for every (t,k,c).
extern "C" __global__ void fct_minmax_source__arms(
    int nt, int nl, int nc, int max_edges, float dt,
    const int *n_edges_on_cell, const int *cells_on_cell,
    const float *old, const float *source, const float *rho_old,
    unsigned char *win_min, unsigned char *win_max)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    for (int t = 0; t < nt; ++t) {
        const int i = Q3(t, k, c, nl, nc);
        u2_minmax_winners(t, k, c, nl, nc, max_edges, dt, n_edges_on_cell, cells_on_cell,
                          old, source, rho_old, &win_min[i], &win_max[i]);
    }
}

extern "C" __global__ void fct_minmax_source__tl(
    int nt, int nl, int nc, int max_edges, float dt,
    const int *n_edges_on_cell, const int *cells_on_cell,
    const float *old, const float *source, const float *rho_old,
    const T *d_old, const T *d_source, const T *d_rho_old,
    T *d_source_old, T *d_minimum, T *d_maximum)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    for (int t = 0; t < nt; ++t) {
        unsigned char lo, hi;
        u2_minmax_winners(t, k, c, nl, nc, max_edges, dt, n_edges_on_cell, cells_on_cell,
                          old, source, rho_old, &lo, &hi);
        const int i = Q3(t, k, c, nl, nc);
        d_source_old[i] = u2_dcurrent(source, rho_old, dt, d_old, d_source, d_rho_old, t, k, c, nl, nc);
        unsigned char w[2] = {lo, hi};
        R out[2];
        for (int m = 0; m < 2; ++m) {
            int kk = k, cc = c;
            if (w[m] == 1) kk = k - 1;
            else if (w[m] == 2) kk = k + 1;
            else if (w[m] >= 3) cc = cells_on_cell[CES(c, w[m] - 3, max_edges)];
            out[m] = u2_dcurrent(source, rho_old, dt, d_old, d_source, d_rho_old, t, kk, cc, nl, nc);
        }
        d_minimum[i] = out[0];
        d_maximum[i] = out[1];
    }
}

// Reverse entry 2 of 2 (entry 1 is __arms into the workspace win_min /
// win_max). One thread per (cell, level) owns b_old, b_source at (t,k,c) and
// b_rho_old at (k,c). inv_off/inv_nbr/inv_slot: for cell c, the (neighbour n,
// slot s) pairs with cells_on_cell[n,s] == c, ascending in n.
extern "C" __global__ void fct_minmax_source__ad(
    int nt, int nl, int nc, int max_edges, float dt,
    const float *source, const float *rho_old,
    const unsigned char *win_min, const unsigned char *win_max,
    const int *inv_off, const int *inv_nbr, const int *inv_slot,
    const T *b_source_old, const T *b_minimum, const T *b_maximum,
    T *b_old, T *b_source, T *b_rho_old)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    const float r = rho_old[C2(k, c, nc)];
    R brho = (R)0;
    for (int t = 0; t < nt; ++t) {
        const int j = Q3(t, k, c, nl, nc);
        R g = b_source_old[j];
        if (win_min[j] == 0) g = g + b_minimum[j];
        if (win_max[j] == 0) g = g + b_maximum[j];
        if (k + 1 < nl) {           // element (k+1) sees this one as candidate "level k-1"
            const int i = Q3(t, k + 1, c, nl, nc);
            if (win_min[i] == 1) g = g + b_minimum[i];
            if (win_max[i] == 1) g = g + b_maximum[i];
        }
        if (k > 0) {                // element (k-1) sees this one as candidate "level k+1"
            const int i = Q3(t, k - 1, c, nl, nc);
            if (win_min[i] == 2) g = g + b_minimum[i];
            if (win_max[i] == 2) g = g + b_maximum[i];
        }
        for (int p = inv_off[c]; p < inv_off[c + 1]; ++p) {
            const int i = Q3(t, k, inv_nbr[p], nl, nc);
            const unsigned char code = (unsigned char)(3 + inv_slot[p]);
            if (win_min[i] == code) g = g + b_minimum[i];
            if (win_max[i] == code) g = g + b_maximum[i];
        }
        const float v = mpas_div(mpas_mul(dt, source[j]), r);
        b_old[j] = b_old[j] + g;
        b_source[j] = b_source[j] + g * rr(dt) / rr(r);
        brho = brho - g * rr(v) / rr(r);
    }
    b_rho_old[C2(k, c, nc)] += brho;
}

// =====================================================================
// fct_scale (in place on scale_in, scale_out). Active: minimum, maximum,
// target_density, mass, scale_in, scale_out.
// scale_in'  = min(1, max(0, (maximum*rho - mass)/(scale_in  + 1e-20)))
// scale_out' = min(1, max(0, (minimum*rho - mass)/(scale_out - 1e-20)))
// Clamp arm codes: 0 lower clamp (max picked 0), 1 linear, 2 upper clamp.
// =====================================================================
__device__ __forceinline__ unsigned char u2_clamp_arm(float x) {
    if (!u2_max_right(0.0f, x)) return 0;          // max(0,x) returned 0
    const float m = mpas_max(0.0f, x);
    return u2_min_right(1.0f, m) ? 1 : 2;          // min(1,m) returned m ?
}

extern "C" __global__ void fct_scale__arms(
    int nt, int nl, int nc, const float *minimum, const float *maximum,
    const float *target_density, const float *mass, const float *scale_in,
    const float *scale_out, unsigned char *arm_in, unsigned char *arm_out)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    const float rho = target_density[C2(k, c, nc)];
    for (int t = 0; t < nt; ++t) {
        const int i = Q3(t, k, c, nl, nc);
        const float inc = mpas_div(mpas_sub(mpas_mul(maximum[i], rho), mass[i]), mpas_add(scale_in[i], 1.0e-20f));
        const float out = mpas_div(mpas_sub(mpas_mul(minimum[i], rho), mass[i]), mpas_sub(scale_out[i], 1.0e-20f));
        arm_in[i] = u2_clamp_arm(inc);
        arm_out[i] = u2_clamp_arm(out);
    }
}

extern "C" __global__ void fct_scale__tl(
    int nt, int nl, int nc, const float *minimum, const float *maximum,
    const float *target_density, const float *mass, const float *scale_in,
    const float *scale_out,
    const T *d_minimum, const T *d_maximum, const T *d_target_density, const T64 *d_mass,
    const T *d_scale_in, const T *d_scale_out, T *d_scale_in_new, T *d_scale_out_new)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    const float rho = target_density[C2(k, c, nc)];
    const R drho = d_target_density[C2(k, c, nc)];
    for (int t = 0; t < nt; ++t) {
        const int i = Q3(t, k, c, nl, nc);
        const float din = mpas_add(scale_in[i], 1.0e-20f), dout = mpas_sub(scale_out[i], 1.0e-20f);
        const float inc = mpas_div(mpas_sub(mpas_mul(maximum[i], rho), mass[i]), din);
        const float out = mpas_div(mpas_sub(mpas_mul(minimum[i], rho), mass[i]), dout);
        R ti = (R)0, to = (R)0;
        if (u2_clamp_arm(inc) == 1) {
            const R dnum = d_maximum[i] * rr(rho) + rr(maximum[i]) * drho - d_mass[i];
            ti = (dnum - rr(inc) * d_scale_in[i]) / rr(din);
        }
        if (u2_clamp_arm(out) == 1) {
            const R dnum = d_minimum[i] * rr(rho) + rr(minimum[i]) * drho - d_mass[i];
            to = (dnum - rr(out) * d_scale_out[i]) / rr(dout);
        }
        d_scale_in_new[i] = ti;
        d_scale_out_new[i] = to;
    }
}

// In place: b_scale_in / b_scale_out hold the seeds (cotangents of the new
// factors) on entry and the cotangents of the pre-launch factors on exit.
extern "C" __global__ void fct_scale__ad(
    int nt, int nl, int nc, const float *minimum, const float *maximum,
    const float *target_density, const float *mass, const float *scale_in,
    const float *scale_out,
    T *b_minimum, T *b_maximum, T *b_target_density, T64 *b_mass,
    T *b_scale_in, T *b_scale_out)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    const float rho = target_density[C2(k, c, nc)];
    R brho = (R)0;
    for (int t = 0; t < nt; ++t) {
        const int i = Q3(t, k, c, nl, nc);
        const float din = mpas_add(scale_in[i], 1.0e-20f), dout = mpas_sub(scale_out[i], 1.0e-20f);
        const float inc = mpas_div(mpas_sub(mpas_mul(maximum[i], rho), mass[i]), din);
        const float out = mpas_div(mpas_sub(mpas_mul(minimum[i], rho), mass[i]), dout);
        const R gi = b_scale_in[i], go = b_scale_out[i];
        R bsi = (R)0, bso = (R)0, bmass = (R)0;
        if (u2_clamp_arm(inc) == 1) {
            const R bn = gi / rr(din);
            b_maximum[i] = b_maximum[i] + bn * rr(rho);
            brho = brho + bn * rr(maximum[i]);
            bmass = bmass - bn;
            bsi = -bn * rr(inc);
        }
        if (u2_clamp_arm(out) == 1) {
            const R bn = go / rr(dout);
            b_minimum[i] = b_minimum[i] + bn * rr(rho);
            brho = brho + bn * rr(minimum[i]);
            bmass = bmass - bn;
            bso = -bn * rr(out);
        }
        b_mass[i] = (R)b_mass[i] + bmass;
        b_scale_in[i] = bsi;
        b_scale_out[i] = bso;
    }
    b_target_density[C2(k, c, nc)] += brho;
}

// =====================================================================
// Limiters. residual' = plus(f)*min(A,B) + minus(f)*min(C,D).
// Arm byte: bits 0-1 sign arm of f (0 zero, 1 positive, 2 negative),
// bit 2 min(A,B) took B, bit 3 min(C,D) took D.
// =====================================================================
__device__ __forceinline__ unsigned char u2_limit_arm(float f, float A, float B, float Cc, float D) {
    unsigned char code = u2_pos(f) ? 1 : (u2_neg(f) ? 2 : 0);
    if (u2_min_right(A, B)) code |= 4;
    if (u2_min_right(Cc, D)) code |= 8;
    return code;
}
__device__ __forceinline__ R u2_limit_tl(float f, float A, float B, float Cc, float D,
                                         R df, R dA, R dB, R dC, R dD) {
    const float wp = mpas_min(A, B), wn = mpas_min(Cc, D);
    const R dwp = u2_min_right(A, B) ? dB : dA;
    const R dwn = u2_min_right(Cc, D) ? dD : dC;
    R r = (R)0;
    if (u2_pos(f)) r = df * rr(wp) + rr(u2_plus(f)) * dwp;
    else if (u2_neg(f)) r = df * rr(wn) + rr(u2_minus(f)) * dwn;
    return r;
}

// ---- fct_limit_horizontal: A=scale_out[c0], B=scale_in[c1], C=scale_in[c0], D=scale_out[c1]
extern "C" __global__ void fct_limit_horizontal__arms(
    int nt, int nl, int nc, int ne, const int *cells_on_edge,
    const float *scale_in, const float *scale_out, const float *residual, unsigned char *arms)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= ne * nl) return;
    const int e = id % ne, k = id / ne;
    const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
    for (int t = 0; t < nt; ++t) {
        const int z = QE3(t, k, e, nl, ne);
        arms[z] = u2_limit_arm(residual[z], scale_out[Q3(t, k, c0, nl, nc)], scale_in[Q3(t, k, c1, nl, nc)],
                               scale_in[Q3(t, k, c0, nl, nc)], scale_out[Q3(t, k, c1, nl, nc)]);
    }
}

extern "C" __global__ void fct_limit_horizontal__tl(
    int nt, int nl, int nc, int ne, const int *cells_on_edge,
    const float *scale_in, const float *scale_out, const float *residual,
    const T *d_scale_in, const T *d_scale_out, const T *d_residual, T *d_residual_new)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= ne * nl) return;
    const int e = id % ne, k = id / ne;
    const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
    for (int t = 0; t < nt; ++t) {
        const int z = QE3(t, k, e, nl, ne);
        const int x0 = Q3(t, k, c0, nl, nc), x1 = Q3(t, k, c1, nl, nc);
        d_residual_new[z] = u2_limit_tl(residual[z], scale_out[x0], scale_in[x1], scale_in[x0], scale_out[x1],
                                        d_residual[z], d_scale_out[x0], d_scale_in[x1], d_scale_in[x0], d_scale_out[x1]);
    }
}

// Reverse entry 1 of 2: cell gather into b_scale_in / b_scale_out (reads the
// residual seed, does not modify it). inv_off/inv_edge: for cell c the edges
// with c in cells_on_edge, ascending.
extern "C" __global__ void fct_limit_horizontal__ad_scale(
    int nt, int nl, int nc, int ne, const int *cells_on_edge,
    const float *scale_in, const float *scale_out, const float *residual,
    const int *inv_off, const int *inv_edge,
    const T *b_residual, T *b_scale_in, T *b_scale_out)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nc * nl) return;
    const int c = id % nc, k = id / nc;
    for (int t = 0; t < nt; ++t) {
        R bi = (R)0, bo = (R)0;
        for (int p = inv_off[c]; p < inv_off[c + 1]; ++p) {
            const int e = inv_edge[p];
            const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
            const int z = QE3(t, k, e, nl, ne);
            const int x0 = Q3(t, k, c0, nl, nc), x1 = Q3(t, k, c1, nl, nc);
            const float f = residual[z];
            const float A = scale_out[x0], B = scale_in[x1], Cc = scale_in[x0], D = scale_out[x1];
            const R g = b_residual[z];
            if (u2_pos(f)) {
                const R gp = g * rr(u2_plus(f));
                const bool right = u2_min_right(A, B);
                if (c == c0 && !right) bo = bo + gp;     // A = scale_out[c0]
                if (c == c1 && right)  bi = bi + gp;     // B = scale_in[c1]
            } else if (u2_neg(f)) {
                const R gn = g * rr(u2_minus(f));
                const bool right = u2_min_right(Cc, D);
                if (c == c0 && !right) bi = bi + gn;     // C = scale_in[c0]
                if (c == c1 && right)  bo = bo + gn;     // D = scale_out[c1]
            }
        }
        const int i = Q3(t, k, c, nl, nc);
        b_scale_in[i] = b_scale_in[i] + bi;
        b_scale_out[i] = b_scale_out[i] + bo;
    }
}

// Reverse entry 2 of 2: edge-local, converts the residual seed in place into
// the cotangent of the pre-launch residual.
extern "C" __global__ void fct_limit_horizontal__ad_residual(
    int nt, int nl, int nc, int ne, const int *cells_on_edge,
    const float *scale_in, const float *scale_out, const float *residual, T *b_residual)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= ne * nl) return;
    const int e = id % ne, k = id / ne;
    const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
    for (int t = 0; t < nt; ++t) {
        const int z = QE3(t, k, e, nl, ne);
        const int x0 = Q3(t, k, c0, nl, nc), x1 = Q3(t, k, c1, nl, nc);
        const float f = residual[z];
        R coefv = (R)0;
        if (u2_pos(f)) coefv = rr(mpas_min(scale_out[x0], scale_in[x1]));
        else if (u2_neg(f)) coefv = rr(mpas_min(scale_in[x0], scale_out[x1]));
        b_residual[z] = b_residual[z] * coefv;
    }
}

// ---- fct_limit_vertical (interfaces 1..nl-1; 0 and nl pass through); one thread per (tracer, cell):
// A=scale_out[k-1], B=scale_in[k], C=scale_out[k], D=scale_in[k-1]
extern "C" __global__ void fct_limit_vertical__arms(
    int nt, int nl, int nc, const float *scale_in, const float *scale_out,
    const float *residual, unsigned char *arms)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nt * nc) return;
    const int c = id % nc;
    {   const int t = id / nc;   // one thread per (tracer, cell) owns that column
        arms[QI3(t, 0, c, nl, nc)] = 0; arms[QI3(t, nl, c, nl, nc)] = 0;
        for (int i = 1; i < nl; ++i) {
            const int qi = QI3(t, i, c, nl, nc);
            arms[qi] = u2_limit_arm(residual[qi], scale_out[Q3(t, i - 1, c, nl, nc)], scale_in[Q3(t, i, c, nl, nc)],
                                    scale_out[Q3(t, i, c, nl, nc)], scale_in[Q3(t, i - 1, c, nl, nc)]);
        }
    }
}

extern "C" __global__ void fct_limit_vertical__tl(
    int nt, int nl, int nc, const float *scale_in, const float *scale_out, const float *residual,
    const T *d_scale_in, const T *d_scale_out, const T *d_residual, T *d_residual_new)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nt * nc) return;
    const int c = id % nc;
    {   const int t = id / nc;   // one thread per (tracer, cell) owns that column
        d_residual_new[QI3(t, 0, c, nl, nc)] = d_residual[QI3(t, 0, c, nl, nc)];
        d_residual_new[QI3(t, nl, c, nl, nc)] = d_residual[QI3(t, nl, c, nl, nc)];
        for (int i = 1; i < nl; ++i) {
            const int qi = QI3(t, i, c, nl, nc);
            const int lo = Q3(t, i - 1, c, nl, nc), up = Q3(t, i, c, nl, nc);
            d_residual_new[qi] = u2_limit_tl(residual[qi], scale_out[lo], scale_in[up], scale_out[up], scale_in[lo],
                                             d_residual[qi], d_scale_out[lo], d_scale_in[up], d_scale_out[up], d_scale_in[lo]);
        }
    }
}

// One thread per cell owns the column. Pass 1 gathers, per level and in
// fixed order (interface k, then k+1), the scale cotangents in double and
// rounds once; pass 2 converts b_residual in place (interfaces 1..nl-1; 0 and
// nl unchanged = identity).
__device__ __forceinline__ void u2_lv_contrib(int t, int i, int c, int nl, int nc,
        const float *scale_in, const float *scale_out, const float *residual, const T *b_residual,
        R *b_in_up, R *b_out_up, R *b_in_lo, R *b_out_lo) {
    const int qi = QI3(t, i, c, nl, nc);
    const int lo = Q3(t, i - 1, c, nl, nc), up = Q3(t, i, c, nl, nc);
    const float f = residual[qi];
    const float A = scale_out[lo], B = scale_in[up], Cc = scale_out[up], D = scale_in[lo];
    const R g = b_residual[qi];
    *b_in_up = 0; *b_out_up = 0; *b_in_lo = 0; *b_out_lo = 0;
    if (u2_pos(f)) {
        const R gp = g * rr(u2_plus(f));
        if (u2_min_right(A, B)) *b_in_up = gp; else *b_out_lo = gp;
    } else if (u2_neg(f)) {
        const R gn = g * rr(u2_minus(f));
        if (u2_min_right(Cc, D)) *b_in_lo = gn; else *b_out_up = gn;
    }
}

extern "C" __global__ void fct_limit_vertical__ad(
    int nt, int nl, int nc, const float *scale_in, const float *scale_out, const float *residual,
    T *b_residual, T *b_scale_in, T *b_scale_out)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nt * nc) return;
    const int c = id % nc;
    {   const int t = id / nc;   // one thread per (tracer, cell) owns that column
        for (int k = 0; k < nl; ++k) {
            R bi = (R)0, bo = (R)0, x0, x1, x2, x3;
            if (k >= 1) {            // interface k: level k is "up"
                u2_lv_contrib(t, k, c, nl, nc, scale_in, scale_out, residual, b_residual, &x0, &x1, &x2, &x3);
                bi = bi + x0; bo = bo + x1;
            }
            if (k + 1 <= nl - 1) {   // interface k+1: level k is "lo"
                u2_lv_contrib(t, k + 1, c, nl, nc, scale_in, scale_out, residual, b_residual, &x0, &x1, &x2, &x3);
                bi = bi + x2; bo = bo + x3;
            }
            const int q = Q3(t, k, c, nl, nc);
            b_scale_in[q] = (R)b_scale_in[q] + bi;
            b_scale_out[q] = (R)b_scale_out[q] + bo;
        }
        for (int i = 1; i < nl; ++i) {
            const int qi = QI3(t, i, c, nl, nc);
            const int lo = Q3(t, i - 1, c, nl, nc), up = Q3(t, i, c, nl, nc);
            const float f = residual[qi];
            R nb = (R)0;
            if (u2_pos(f)) nb = (R)b_residual[qi] * rr(mpas_min(scale_out[lo], scale_in[up]));
            else if (u2_neg(f)) nb = (R)b_residual[qi] * rr(mpas_min(scale_out[up], scale_in[lo]));
            b_residual[qi] = nb;
        }
    }
}
