// SPDX-License-Identifier: Apache-2.0
// L4 speed (ad/window-L4-speed): owner-computes reverse of transport_edge_values_mono_v841.
// The tangent, the old reverse and ad_clear are the unchanged U2 source (included below).
// The old reverse searched every edge for every cell (O(nedges*ncells) per tracer) and summed
// the scalar coefficient on one thread. Here:
//   l4_inv_*      cell -> stencil items j = e*width + s, ascending (l4_common.cuh)
//   l4_vel        one thread per (k,e): b_velocity, same terms, same order (bitwise)
//   l4_adv        one thread per (e,s): b_adv, b_adv3, same terms, same order (bitwise)
//   l4_stage      one thread per (q,k,c): gather over the cell's items in (e,s) order (bitwise)
//   l4_coef_*     per-edge binary64 partials, then one fixed-order block sum (binary64 reassociation only)
// No float atomics. Failure slots keep their U2 numbers.
#include "transport_edge_values_mono_v841.ad.cu"
#include "l4_common.cuh"

#define TEVM(name) transport_edge_values_mono_v841__l4_##name

extern "C" __global__ void TEVM(inv_count)(int ncells, int nedges, int width, const int *n_adv, const int *adv_cells, int *ws)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges) return;
    const int count = n_adv[e]; if (count < 0 || count > width) return;
    for (int s = 0; s < count; ++s) { const int c = adv_cells[ADV(e, s, width)]; if (c >= 0 && c < ncells) atomicAdd(&ws[c], 1); }
}
extern "C" __global__ void TEVM(inv_scan)(int ncells, int *ws) { l4_scan_block(ws, ncells); }
extern "C" __global__ void TEVM(inv_fill)(int ncells, int nedges, int width, const int *n_adv, const int *adv_cells, int *ws)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges) return;
    const int count = n_adv[e]; if (count < 0 || count > width) return;
    int *cur = l4_cursor(ws, ncells), *list = l4_list(ws, ncells);
    for (int s = 0; s < count; ++s) {
        const int j = ADV(e, s, width), c = adv_cells[j];
        if (c >= 0 && c < ncells) list[ws[c] + atomicAdd(&cur[c], 1)] = j;
    }
}
extern "C" __global__ void TEVM(inv_sort)(int ncells, int *ws)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c < ncells) l4_sort_key(ws, ncells, c);
}

// Edge validity exactly as the old reverse: count in range and every stencil cell in range.
__device__ __forceinline__ bool tevm_edge_valid(int e, int ncells, int width, const int *n_adv, const int *adv_cells, Failure *failure)
{
    const int count = n_adv[e];
    if (count < 0) { ad_fail(failure, 14); return false; }
    if (count > width) { ad_fail(failure, 29); return false; }
    bool valid = true;
    for (int s = 0; s < count; ++s) { const int c = adv_cells[ADV(e, s, width)]; if (c < 0) { ad_fail(failure, 15); valid = false; } if (c >= ncells) { ad_fail(failure, 30); valid = false; } }
    return valid;
}

extern "C" __global__ void TEVM(vel)(
 int ntracers, int nlev, int ncells, int nedges, int width, float coefficient, const float *stage, const float *velocity,
 const float *adv, const float *adv3, const int *n_adv, const int *adv_cells, const float *b_edge_values,
 float *b_velocity, Failure *failure)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nlev * nedges) return;
    const int k = (int)(t / nedges), e = (int)(t % nedges);
    if (!tevm_edge_valid(e, ncells, width, n_adv, adv_cells, failure)) return;
    const int count = n_adv[e];
    if (!fin(coefficient)) ad_fail(failure, 16);
    const int vi = E2(k, e, nedges); const float vel = velocity[vi], sg = mpas_copysign(1.0f, vel); double bv = (double)b_velocity[vi];
    if (!fin(vel)) ad_fail(failure, 17);
    for (int q = 0; q < ntracers; ++q) {
        const int o = QE3(q, k, e, nlev, nedges); const float bo = b_edge_values[o]; float v = 0.0f; if (!fin(bo)) ad_fail(failure, 18);
        for (int s = 0; s < count; ++s) {
            const int j = ADV(e, s, width), i = Q3(q, k, adv_cells[j], nlev, ncells);
            const float w = weight(coefficient, sg, adv[j], adv3[j]), x = stage[i];
            { if (!fin(adv[j])) ad_fail(failure, 19); if (!fin(adv3[j])) ad_fail(failure, 20); if (!fin(x)) ad_fail(failure, 21); }
            const float part = mpas_mul(w, x); v = (count == 10 && s == 0) ? part : mpas_add(v, part);
        }
        bv += (double)mpas_mul(bo, v);
    }
    b_velocity[vi] = (float)bv; if (!fin(b_velocity[vi])) ad_fail(failure, 24);
}

extern "C" __global__ void TEVM(adv)(
 int ntracers, int nlev, int ncells, int nedges, int width, float coefficient, const float *stage, const float *velocity,
 const int *n_adv, const int *adv_cells, const float *b_edge_values, float *b_adv, float *b_adv3, Failure *failure)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nedges * width) return;
    const int e = (int)(t / width), s = (int)(t % width);
    if (!tevm_edge_valid(e, ncells, width, n_adv, adv_cells, failure)) return;
    const int count = n_adv[e]; if (s >= count) return;
    const int j = ADV(e, s, width), c = adv_cells[j]; double ba = (double)b_adv[j], bt = (double)b_adv3[j];
    for (int k = 0; k < nlev; ++k) {
        const float vel = velocity[E2(k, e, nedges)], sg = mpas_copysign(1.0f, vel);
        for (int q = 0; q < ntracers; ++q) {
            const float bo = b_edge_values[QE3(q, k, e, nlev, nedges)], x = stage[Q3(q, k, c, nlev, ncells)];
            const float bw = (count == 10) ? mpas_mul(mpas_mul(bo, vel), x) : mpas_mul(mpas_mul(bo, x), vel);
            ba += (double)bw; bt += (double)mpas_mul(mpas_mul(bw, coefficient), sg);
        }
    }
    b_adv[j] = (float)ba; b_adv3[j] = (float)bt; { if (!fin(b_adv[j])) ad_fail(failure, 22); if (!fin(b_adv3[j])) ad_fail(failure, 23); }
}

extern "C" __global__ void TEVM(stage)(
 int ntracers, int nlev, int ncells, int nedges, int width, float coefficient, const float *velocity,
 const float *adv, const float *adv3, const int *n_adv, const float *b_edge_values, float *b_stage, const int *ws, Failure *failure)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)ntracers * nlev * ncells) return;
    const int c = (int)(t % ncells), k = (int)((t / ncells) % nlev), q = (int)(t / ((long long)ncells * nlev));
    const int i = Q3(q, k, c, nlev, ncells);
    const int *off = ws, *list = ws + 2 * ncells + 1;
    double bs = (double)b_stage[i];
    for (int p = off[c]; p < off[c + 1]; ++p) {
        const int j = list[p], e = j / width; const int count = n_adv[e];
        const float vel = velocity[E2(k, e, nedges)], sg = mpas_copysign(1.0f, vel), w = weight(coefficient, sg, adv[j], adv3[j]);
        const int o = QE3(q, k, e, nlev, nedges);
        bs += (double)((count == 10) ? mpas_mul(mpas_mul(b_edge_values[o], vel), w) : mpas_mul(b_edge_values[o], mpas_mul(vel, w)));
    }
    b_stage[i] = (float)bs; if (!fin(b_stage[i])) ad_fail(failure, 25);
}

extern "C" __global__ void TEVM(coef_partial)(
 int ntracers, int nlev, int ncells, int nedges, int width, const float *stage, const float *velocity,
 const float *adv3, const int *n_adv, const int *adv_cells, const float *b_edge_values, double *partial)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges) return;
    const int count = n_adv[e]; double sum = 0.0;
    if (count >= 0 && count <= width) {
        for (int k = 0; k < nlev; ++k) {
            const float vel = velocity[E2(k, e, nedges)], sg = mpas_copysign(1.0f, vel);
            for (int q = 0; q < ntracers; ++q) {
                const float bo = b_edge_values[QE3(q, k, e, nlev, nedges)];
                for (int s = 0; s < count; ++s) {
                    const int j = ADV(e, s, width), c = adv_cells[j]; if (c < 0 || c >= ncells) continue;
                    const float x = stage[Q3(q, k, c, nlev, ncells)], bw = (count == 10) ? mpas_mul(mpas_mul(bo, vel), x) : mpas_mul(mpas_mul(bo, x), vel);
                    sum += (double)mpas_mul(mpas_mul(bw, sg), adv3[j]);
                }
            }
        }
    }
    partial[e] = sum;
}
extern "C" __global__ void TEVM(coef_reduce)(int nedges, const double *partial, float *b_coefficient, Failure *failure)
{
    const double total = l4_block_sum(partial, nedges);
    if (threadIdx.x == 0) { *b_coefficient = (float)((double)*b_coefficient + total); if (!fin(*b_coefficient)) ad_fail(failure, 26); }
}
