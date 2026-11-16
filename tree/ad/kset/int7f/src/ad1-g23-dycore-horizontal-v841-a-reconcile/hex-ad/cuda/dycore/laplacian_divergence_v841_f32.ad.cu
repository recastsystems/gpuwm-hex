// SPDX-License-Identifier: Apache-2.0
// Direct hand oracle for the fixed-geometry cell divergence stencil.
#include "horizontal_v841_ad.cuh"
#include "horizontal_v841_branches.cuh"
// Replay the primal row for the transpose: the strict add's fast arm can
// flush a subnormal operand, so even a geometry stencil has state masks.
__device__ __forceinline__ float lapdiv_weight(int cell,int edge,const int *coe,
                                               const float *dv,const float *area) {
    float x=ad_probe_div(nullptr,0,1.0f,area[cell]);
    x=ad_probe_mul(nullptr,0,x,dv[edge]);
    return ad_probe_mul(nullptr,0,x,coe[2*edge]==cell?1.0f:-1.0f);
}
__device__ __forceinline__ bool lapdiv_live(int cell,int k,int target,int nedges,int max_edges,
        const float *u,const int *eoc,const int *neoc,const int *coe,
        const float *dv,const float *area) {
    float value=0.0f; bool live=false;
    for(int s=0;s<neoc[cell];++s) {
        int e=eoc[cell*max_edges+s];
        float term=ad_probe_mul(nullptr,0,lapdiv_weight(cell,e,coe,dv,area),u[k*nedges+e]);
        bool l=ad_probe_add_operand_live(value,term,false,true);
        bool r=ad_probe_add_operand_live(value,term,false,false);
        if(s==target)live=r;else if(s>target)live=live&&l;
        value=ad_probe_addsub(nullptr,0,value,term,false);
    }
    return live;
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void laplacian_divergence_v841_f32__tl(
    const float *delsq_u, const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge, const float *dv_edge, const float *area_cell,
    int nlev, int ncells, int nedges, int max_edges, float *delsq_divergence,
    const float *d_delsq_u, float *d_delsq_divergence, AdFailureRecord *failure, unsigned *branch_record)
{
    // Count executed primal branches in shared memory, then combine counts.
    __shared__ unsigned ad_branch_local[28];
    for (unsigned q=threadIdx.x; q<28u; q+=blockDim.x) ad_branch_local[q]=0u;
    __syncthreads();
    ad_probe_laplacian_divergence_v841_f32(delsq_u, edges_on_cell, n_edges_on_cell, cells_on_edge, dv_edge, area_cell, nlev, ncells, nedges, max_edges, delsq_divergence, ad_branch_local);
    __syncthreads();
    for (unsigned q=threadIdx.x; q<28u; q+=blockDim.x)
        if (ad_branch_local[q]) atomicAdd(&branch_record[q],ad_branch_local[q]);

    (void)delsq_divergence;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    AD_LOOP(nlev * ncells) {
        const int cell = ad_i % ncells, k = ad_i / ncells;
        const float area = area_cell[cell];
        AD_TEST(area, 6, cell, 225);
        if (area == 0.0f) { ad_fail(failure, ad_thread, 6, cell, 225); continue; }
        float sum = 0.0f, primal = 0.0f;
        for (int s = 0; s < n_edges_on_cell[cell]; ++s) {
            const int edge = edges_on_cell[cell * max_edges + s];
            const float w = lapdiv_weight(cell,edge,cells_on_edge,dv_edge,area_cell);
            const int i = k * nedges + edge;
            AD_TEST(delsq_u[i], 1, i, 225);
            AD_TEST(d_delsq_u[i], 1, i, 225);
            AD_TEST(w, 5, edge, 225);
            const float term=ad_probe_mul(nullptr,0,w,delsq_u[i]);
            sum = __fadd_rn(ad_probe_add_operand_live(primal,term,false,true)?sum:0.0f,
                ad_probe_add_operand_live(primal,term,false,false)?__fmul_rn(ad_probe_mul_weight(w,delsq_u[i],false),d_delsq_u[i]):0.0f);
            primal=ad_probe_addsub(nullptr,0,primal,term,false);
        }
        AD_TEST(sum, 11, ad_i, 228);
        d_delsq_divergence[ad_i] = sum;
    }
}
#endif

#if !defined(AD_ONLY_TL)
extern "C" __global__ void laplacian_divergence_v841_f32__ad(
    const float *delsq_u, const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge, const float *dv_edge, const float *area_cell,
    int nlev, int ncells, int nedges, int max_edges, float *delsq_divergence,
    float *b_delsq_u, const float *seed_delsq_divergence, float *b_delsq_divergence, AdFailureRecord *failure)
{
    (void)delsq_divergence;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    // Snapshots are disjoint from the consumed cotangents and input cotangents.
    AD_LOOP(nlev * ncells) { b_delsq_divergence[ad_i] = 0.0f; }
    AD_LOOP(nlev * nedges) {
        const int edge = ad_i % nedges, k = ad_i / nedges;
        double acc = b_delsq_u[ad_i];
        AD_TEST(acc, 1, ad_i, 225);
        AD_TEST(delsq_u[ad_i], 1, ad_i, 225);
        for (int q = 0; q < 2; ++q) {
            const int cell = cells_on_edge[2 * edge + q];
            if (cell < 0 || cell >= ncells || (q && cell == cells_on_edge[2 * edge])) continue;
            const float area = area_cell[cell];
            AD_TEST(area, 6, cell, 225);
            if (area == 0.0f) { ad_fail(failure, ad_thread, 6, cell, 225); continue; }
            for (int s = 0; s < n_edges_on_cell[cell]; ++s) {
                if (edges_on_cell[cell * max_edges + s] != edge) continue;
                const float w = lapdiv_weight(cell,edge,cells_on_edge,dv_edge,area_cell);
                const float by = seed_delsq_divergence[k * ncells + cell];
                AD_TEST(w, 5, edge, 225); AD_TEST(by, 11, k * ncells + cell, 225);
                if(lapdiv_live(cell,k,s,nedges,max_edges,delsq_u,edges_on_cell,
                               n_edges_on_cell,cells_on_edge,dv_edge,area_cell))
                    acc += (double)ad_probe_mul_weight(w,delsq_u[ad_i],false) * (double)by;
            }
        }
        AD_TEST(acc, 1, ad_i, 225);
        b_delsq_u[ad_i] = acc;
    }
}
#endif

