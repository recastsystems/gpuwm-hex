// SPDX-License-Identifier: Apache-2.0
// Direct hand oracle for the fixed-geometry vertex vorticity stencil.
#include "horizontal_v841_ad.cuh"
#include "horizontal_v841_branches.cuh"
__device__ __forceinline__ float lapvort_weight(int v,int e,const int *voe,
                                                const float *dc,const float *area) {
    float x=ad_probe_div(nullptr,0,1.0f,area[v]);
    x=ad_probe_mul(nullptr,0,x,dc[e]);
    return ad_probe_mul(nullptr,0,x,voe[2*e]==v?-1.0f:1.0f);
}
__device__ __forceinline__ bool lapvort_live(int v,int k,int target,int nedges,int degree,
        const float *u,const int *eov,const int *voe,const float *dc,const float *area) {
    float value=0.0f; bool live=false;
    for(int s=0;s<degree;++s) {
        int e=eov[v*degree+s]; if(e<0)continue;
        float term=ad_probe_mul(nullptr,0,lapvort_weight(v,e,voe,dc,area),u[k*nedges+e]);
        bool l=ad_probe_add_operand_live(value,term,false,true);
        bool r=ad_probe_add_operand_live(value,term,false,false);
        if(s==target)live=r;else if(s>target)live=live&&l;
        value=ad_probe_addsub(nullptr,0,value,term,false);
    }
    return live;
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void laplacian_vorticity_v841_f32__tl(
    const float *delsq_u, const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *area_triangle, int nlev, int nedges,
    int nvertices, int vertex_degree, float *delsq_vorticity,
    const float *d_delsq_u, float *d_delsq_vorticity, AdFailureRecord *failure, unsigned *branch_record)
{
    // Count executed primal branches in shared memory, then combine counts.
    __shared__ unsigned ad_branch_local[32];
    for (unsigned q=threadIdx.x; q<32u; q+=blockDim.x) ad_branch_local[q]=0u;
    __syncthreads();
    ad_probe_laplacian_vorticity_v841_f32(delsq_u, edges_on_vertex, vertices_on_edge, dc_edge, area_triangle, nlev, nedges, nvertices, vertex_degree, delsq_vorticity, ad_branch_local);
    __syncthreads();
    for (unsigned q=threadIdx.x; q<32u; q+=blockDim.x)
        if (ad_branch_local[q]) atomicAdd(&branch_record[q],ad_branch_local[q]);

    (void)delsq_vorticity;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    AD_LOOP(nlev * nvertices) {
        const int vertex = ad_i % nvertices, k = ad_i / nvertices;
        const float area = area_triangle[vertex];
        AD_TEST(area, 5, vertex, 252);
        if (area == 0.0f) { ad_fail(failure, ad_thread, 5, vertex, 252); continue; }
        float sum = 0.0f, primal = 0.0f;
        for (int s = 0; s < vertex_degree; ++s) {
            const int edge = edges_on_vertex[vertex * vertex_degree + s];
            if (edge < 0) continue;
            const float w = lapvort_weight(vertex,edge,vertices_on_edge,dc_edge,area_triangle);
            const int i = k * nedges + edge;
            AD_TEST(delsq_u[i], 1, i, 252); AD_TEST(d_delsq_u[i], 1, i, 252);
            AD_TEST(w, 4, edge, 252);
            const float term=ad_probe_mul(nullptr,0,w,delsq_u[i]);
            sum = __fadd_rn(ad_probe_add_operand_live(primal,term,false,true)?sum:0.0f,
                ad_probe_add_operand_live(primal,term,false,false)?__fmul_rn(ad_probe_mul_weight(w,delsq_u[i],false),d_delsq_u[i]):0.0f);
            primal=ad_probe_addsub(nullptr,0,primal,term,false);
        }
        AD_TEST(sum, 10, ad_i, 255);
        d_delsq_vorticity[ad_i] = sum;
    }
}
#endif

#if !defined(AD_ONLY_TL)
extern "C" __global__ void laplacian_vorticity_v841_f32__ad(
    const float *delsq_u, const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *area_triangle, int nlev, int nedges,
    int nvertices, int vertex_degree, float *delsq_vorticity,
    float *b_delsq_u, const float *seed_delsq_vorticity, float *b_delsq_vorticity, AdFailureRecord *failure)
{
    (void)delsq_vorticity;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    // Snapshots are disjoint from the consumed cotangents and input cotangents.
    AD_LOOP(nlev * nvertices) { b_delsq_vorticity[ad_i] = 0.0f; }
    AD_LOOP(nlev * nedges) {
        const int edge = ad_i % nedges, k = ad_i / nedges;
        double acc = b_delsq_u[ad_i];
        AD_TEST(acc, 1, ad_i, 252); AD_TEST(delsq_u[ad_i], 1, ad_i, 252);
        for (int q = 0; q < 2; ++q) {
            const int vertex = vertices_on_edge[2 * edge + q];
            if (vertex < 0 || vertex >= nvertices || (q && vertex == vertices_on_edge[2 * edge])) continue;
            const float area = area_triangle[vertex];
            AD_TEST(area, 5, vertex, 252);
            if (area == 0.0f) { ad_fail(failure, ad_thread, 5, vertex, 252); continue; }
            for (int s = 0; s < vertex_degree; ++s) {
                if (edges_on_vertex[vertex * vertex_degree + s] != edge) continue;
                const float w = lapvort_weight(vertex,edge,vertices_on_edge,dc_edge,area_triangle);
                const float by = seed_delsq_vorticity[k * nvertices + vertex];
                AD_TEST(w, 4, edge, 252); AD_TEST(by, 10, k * nvertices + vertex, 252);
                if(lapvort_live(vertex,k,s,nedges,vertex_degree,delsq_u,edges_on_vertex,
                                vertices_on_edge,dc_edge,area_triangle))
                    acc += (double)ad_probe_mul_weight(w,delsq_u[ad_i],false) * (double)by;
            }
        }
        AD_TEST(acc, 1, ad_i, 252);
        b_delsq_u[ad_i] = acc;
    }
}
#endif

