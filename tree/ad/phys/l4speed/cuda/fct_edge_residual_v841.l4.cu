// SPDX-License-Identifier: Apache-2.0
// L4 speed (ad/window-L4-speed): owner-computes reverse of fct_edge_residual_v841.
// The tangent, the old reverse and ad_clear are the unchanged U2 source (included below).
// The old reverse searched every edge for every cell (O(nedges*ncells) per tracer) and summed
// dt on one thread. Here:
//   l4_edge      one thread per edge: the old per-edge owner loop, unchanged (bitwise)
//   l4_inv_*     cell -> edge sides j = 2e + side (edges with both cells in range), ascending
//   l4_src       one thread per (q,k,c): gather in (e, side) order, as the old search (bitwise)
//   l4_dt_*      per-edge binary64 partials, then one fixed-order block sum (binary64 reassociation only)
// No float atomics. Failure slots keep their U2 numbers.
#include "fct_edge_residual_v841.ad.cu"
#include "l4_common.cuh"

#define FER(name) fct_edge_residual_v841__l4_##name

__device__ __forceinline__ bool fer_valid(int e, int ncells, const int *cells_on_edge) {
    const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
    return !(c0 < 0 || c0 >= ncells || c1 < 0 || c1 >= ncells);
}
extern "C" __global__ void FER(inv_count)(int ncells, int nedges, const int *cells_on_edge, int *ws)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges || !fer_valid(e, ncells, cells_on_edge)) return;
    atomicAdd(&ws[cells_on_edge[2 * e]], 1); atomicAdd(&ws[cells_on_edge[2 * e + 1]], 1);
}
extern "C" __global__ void FER(inv_scan)(int ncells, int *ws) { l4_scan_block(ws, ncells); }
extern "C" __global__ void FER(inv_fill)(int ncells, int nedges, const int *cells_on_edge, int *ws)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges || !fer_valid(e, ncells, cells_on_edge)) return;
    int *cur = l4_cursor(ws, ncells), *list = l4_list(ws, ncells);
    for (int side = 0; side < 2; ++side) { const int c = cells_on_edge[2 * e + side]; list[ws[c] + atomicAdd(&cur[c], 1)] = 2 * e + side; }
}
extern "C" __global__ void FER(inv_sort)(int ncells, int *ws)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c < ncells) l4_sort_key(ws, ncells, c);
}

extern "C" __global__ void FER(edge)(
    int ntracers, int nlev, int ncells, int nedges, float dt,
    const float *source_old, const float *velocity, const float *dv_edge,
    const int *cells_on_edge, const float *high_flux,
    const float *b_upwind_flux, const float *b_horizontal_residual,
    float *b_velocity, float *b_dv_edge, float *b_high_flux, Failure *failure)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges) return;
    {
        int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
        if(c0<0){ad_fail(failure,17);return;}if(c0>=ncells){ad_fail(failure,37);return;}if(c1<0){ad_fail(failure,38);return;}if(c1>=ncells){ad_fail(failure,39);return;}
        if (!ad_finite(dt)) ad_fail(failure,18);
        if (!ad_finite(dv_edge[e])) ad_fail(failure,19);
        double bdv=(double)b_dv_edge[e];
        for (int k=0;k<nlev;++k) {
            int vi=E2(k,e,nedges); float v=velocity[vi]; double bv=(double)b_velocity[vi];
            if (!ad_finite(v)) ad_fail(failure,20);
            for (int q=0;q<ntracers;++q) {
                int i=QE3(q,k,e,nlev,nedges), a=Q3(q,k,c0,nlev,ncells), b=Q3(q,k,c1,nlev,ncells);
                float x=source_old[a],y=source_old[b],bu=b_upwind_flux[i],br=b_horizontal_residual[i];
                {if(!ad_finite(x))ad_fail(failure,21);if(!ad_finite(y))ad_fail(failure,22);}
                if (!ad_finite(high_flux[i])) ad_fail(failure,23);
                if (!ad_finite(bu)) ad_fail(failure,24);
                if (!ad_finite(br)) ad_fail(failure,25);
                double bar=(double)bu-(double)br;
                float sum=mpas_add(mpas_mul(pval(v),x),mpas_mul(mval(v),y));
                bdv+=bar*(double)dt*(double)sum;
                double bs=bar*(double)mpas_mul(dv_edge[e],dt);
                bv+=bs*(double)(ad_pos(v)?x:(ad_neg(v)?y:0.0f));
                b_high_flux[i]=(float)((double)b_high_flux[i]+(double)dt*(double)br);
                {if(!ad_finite(sum))ad_fail(failure,26);if(!ad_finite((float)bv))ad_fail(failure,27);if(!ad_finite((float)bdv))ad_fail(failure,28);if(!ad_finite(b_high_flux[i]))ad_fail(failure,29);}
            }
            b_velocity[vi]=(float)bv;
            if (!ad_finite(b_velocity[vi])) ad_fail(failure,30);
        }
        b_dv_edge[e]=(float)bdv;
        if (!ad_finite(b_dv_edge[e])) ad_fail(failure,31);
    }
}

extern "C" __global__ void FER(src)(
    int ntracers, int nlev, int ncells, int nedges, float dt, const float *velocity, const float *dv_edge,
    const float *b_upwind_flux, const float *b_horizontal_residual, float *b_source_old, const int *ws, Failure *failure)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)ntracers * nlev * ncells) return;
    const int c = (int)(t % ncells), k = (int)((t / ncells) % nlev), q = (int)(t / ((long long)ncells * nlev));
    const int *off = ws, *list = ws + 2 * ncells + 1;
    const int ci = Q3(q, k, c, nlev, ncells);
    double sum = (double)b_source_old[ci];
    for (int p = off[c]; p < off[c + 1]; ++p) {
        const int j = list[p], e = j >> 1, side = j & 1;
        const float factor = mpas_mul(dv_edge[e], dt);
        const int i = QE3(q, k, e, nlev, nedges);
        const float v = velocity[E2(k, e, nedges)];
        const double bs = ((double)b_upwind_flux[i] - (double)b_horizontal_residual[i]) * (double)factor;
        if (side == 0) sum += bs * (double)pval(v); else sum += bs * (double)mval(v);
    }
    b_source_old[ci] = (float)sum; if (!ad_finite(b_source_old[ci])) ad_fail(failure, 32);
}

extern "C" __global__ void FER(dt_partial)(
    int ntracers, int nlev, int ncells, int nedges, const float *source_old, const float *velocity, const float *dv_edge,
    const int *cells_on_edge, const float *high_flux, const float *b_upwind_flux, const float *b_horizontal_residual, double *partial)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nedges) return;
    double total = 0.0;
    if (fer_valid(e, ncells, cells_on_edge)) {
        const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
        for (int q=0;q<ntracers;++q) for (int k=0;k<nlev;++k) {
            int i=QE3(q,k,e,nlev,nedges); float v=velocity[E2(k,e,nedges)];
            float sum=mpas_add(mpas_mul(pval(v),source_old[Q3(q,k,c0,nlev,ncells)]),mpas_mul(mval(v),source_old[Q3(q,k,c1,nlev,ncells)]));
            double bar=(double)b_upwind_flux[i]-(double)b_horizontal_residual[i];
            total+=bar*(double)dv_edge[e]*(double)sum+(double)b_horizontal_residual[i]*(double)high_flux[i];
        }
    }
    partial[e] = total;
}
extern "C" __global__ void FER(dt_reduce)(int nedges, const double *partial, float *b_dt, Failure *failure)
{
    const double total = l4_block_sum(partial, nedges);
    if (threadIdx.x == 0) { *b_dt = (float)((double)*b_dt + total); if (!ad_finite(*b_dt)) ad_fail(failure, 33); }
}
