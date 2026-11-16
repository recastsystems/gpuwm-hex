// SPDX-License-Identifier: Apache-2.0
#include "horizontal_v841_ad.cuh"
#include "horizontal_v841_branches.cuh"

// Replay the fixed-degree row so a gather reverse can reproduce the masks of
// each accumulation without atomics or saved state. A null record suppresses
// diagnostic counting; the tangent entry records the complete row separately.
struct VertexReplay { float vort,vort_weight,ke_weight; bool vortex_term, kinetic_term; };
__device__ __forceinline__ VertexReplay vertex_replay(
    int v,int k,int target,int nedges,int degree,const float *u,const float *ke,
    const int *eov,const int *voe,const float *dc,const float *inv) {
    float sv=0.0f,sk=0.0f;bool mv=false,mk=false;
    for(int s=0;s<degree;++s) {
        int e=eov[v*degree+s];if(e<0)continue;
        float sign=voe[2*e]==v?-1.0f:1.0f;
        float w=ad_probe_mul(nullptr,0,sign,dc[e]);
        float tv=ad_probe_mul(nullptr,0,w,u[k*nedges+e]);
        float tk=ke[k*nedges+e];
        bool lv=ad_probe_add_operand_live(sv,tv,false,true);
        bool rv=ad_probe_add_operand_live(sv,tv,false,false);
        bool lk=ad_probe_add_operand_live(sk,tk,false,true);
        bool rk=ad_probe_add_operand_live(sk,tk,false,false);
        if(s==target){mv=rv;mk=rk;}else if(s>target){mv=mv&&lv;mk=mk&&lk;}
        sv=ad_probe_addsub(nullptr,0,sv,tv,false);
        sk=ad_probe_addsub(nullptr,0,sk,tk,false);
    }
    float r=ad_probe_mul(nullptr,0,0.25f,inv[v]);
    return {ad_probe_mul(nullptr,0,sv,inv[v]),ad_probe_mul_weight(sv,inv[v],true),
            ad_probe_mul_weight(sk,r,true),mv,mk};
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void vertex_diagnostics_v841_f32__tl(
    const float *normal_velocity, const float *ke_edge,
    const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *inv_area_triangle, const float *f_vertex,
    int nlev, int nedges, int nvertices, int vertex_degree,
    float *vorticity, float *ke_vertex, float *pv_vertex,
    const float *d_normal_velocity, const float *d_ke_edge, const float *d_f_vertex,
    float *d_vorticity, float *d_ke_vertex, float *d_pv_vertex,
    AdFailureRecord *failure, unsigned *branch_record)
{
    // Count executed primal branches in shared memory, then combine counts.
    __shared__ unsigned ad_branch_local[30];
    for (unsigned q=threadIdx.x; q<30u; q+=blockDim.x) ad_branch_local[q]=0u;
    __syncthreads();
    ad_probe_vertex_diagnostics_v841_f32(normal_velocity, ke_edge, edges_on_vertex, vertices_on_edge, dc_edge, inv_area_triangle, f_vertex, nlev, nedges, nvertices, vertex_degree, vorticity, ke_vertex, pv_vertex, ad_branch_local);
    __syncthreads();
    for (unsigned q=threadIdx.x; q<30u; q+=blockDim.x)
        if (ad_branch_local[q]) atomicAdd(&branch_record[q],ad_branch_local[q]);

    (void)vorticity; (void)ke_vertex; (void)pv_vertex;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    AD_LOOP(nlev * nvertices) {
        const int v = ad_i % nvertices, k = ad_i / nvertices;
        float dvort = 0.0f, dke = 0.0f, svort=0.0f, ske=0.0f;
        AD_TEST(inv_area_triangle[v], 6, v, 88); AD_TEST(f_vertex[v], 7, v, 94);
        AD_TEST(d_f_vertex[v], 7, v, 94);
        for (int s = 0; s < vertex_degree; ++s) {
            const int edge = edges_on_vertex[v * vertex_degree + s];
            if (edge < 0) continue;
            const int e = k * nedges + edge;
            const float sign = vertices_on_edge[2 * edge] == v ? -1.0f : 1.0f;
            AD_TEST(normal_velocity[e], 1, e, 83); AD_TEST(ke_edge[e], 2, e, 86);
            AD_TEST(d_normal_velocity[e], 1, e, 83); AD_TEST(d_ke_edge[e], 2, e, 86);
            AD_TEST(dc_edge[edge], 5, edge, 83);
            const float w=ad_probe_mul(nullptr,0,sign,dc_edge[edge]);
            const float tv=ad_probe_mul(nullptr,0,w,normal_velocity[e]);
            const float tk=ke_edge[e];
            dvort = __fadd_rn(ad_probe_add_operand_live(svort,tv,false,true)?dvort:0.0f,
                ad_probe_add_operand_live(svort,tv,false,false)?__fmul_rn(ad_probe_mul_weight(w,normal_velocity[e],false),d_normal_velocity[e]):0.0f);
            dke = __fadd_rn(ad_probe_add_operand_live(ske,tk,false,true)?dke:0.0f,
                ad_probe_add_operand_live(ske,tk,false,false)?d_ke_edge[e]:0.0f);
            svort=ad_probe_addsub(nullptr,0,svort,tv,false);
            ske=ad_probe_addsub(nullptr,0,ske,tk,false);
        }
        dvort = __fmul_rn(dvort, ad_probe_mul_weight(svort,inv_area_triangle[v],true));
        const float primal_r=ad_probe_mul(nullptr,0,0.25f,inv_area_triangle[v]);
        dke = __fmul_rn(dke, ad_probe_mul_weight(ske,primal_r,true));
        const float primal_vort=ad_probe_mul(nullptr,0,svort,inv_area_triangle[v]);
        const float dpv = __fadd_rn(ad_probe_add_operand_live(f_vertex[v],primal_vort,false,true)?d_f_vertex[v]:0.0f,
            ad_probe_add_operand_live(f_vertex[v],primal_vort,false,false)?dvort:0.0f);
        AD_TEST(dvort, 12, ad_i, 92); AD_TEST(dke, 13, ad_i, 93); AD_TEST(dpv, 14, ad_i, 94);
        d_vorticity[ad_i] = dvort; d_ke_vertex[ad_i] = dke; d_pv_vertex[ad_i] = dpv;
    }
}
#endif

#if !defined(AD_ONLY_TL)
extern "C" __global__ void vertex_diagnostics_v841_f32__ad(
    const float *normal_velocity, const float *ke_edge,
    const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *inv_area_triangle, const float *f_vertex,
    int nlev, int nedges, int nvertices, int vertex_degree,
    float *vorticity, float *ke_vertex, float *pv_vertex,
    float *b_normal_velocity, float *b_ke_edge, float *b_f_vertex,
    const float *seed_vorticity, const float *seed_ke_vertex, const float *seed_pv_vertex,
    float *b_vorticity, float *b_ke_vertex, float *b_pv_vertex, AdFailureRecord *failure)
{
    (void)vorticity; (void)ke_vertex; (void)pv_vertex;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    // Snapshots are disjoint from the consumed cotangents and input cotangents.
    AD_LOOP(nlev * nvertices) { b_vorticity[ad_i] = 0.0f; b_ke_vertex[ad_i] = 0.0f; b_pv_vertex[ad_i] = 0.0f; }
    AD_LOOP(nlev * nedges) {
        const int edge = ad_i % nedges, k = ad_i / nedges;
        double bu = b_normal_velocity[ad_i], bk = b_ke_edge[ad_i];
        AD_TEST(normal_velocity[ad_i], 1, ad_i, 83); AD_TEST(ke_edge[ad_i], 2, ad_i, 86);
        for (int q = 0; q < 2; ++q) {
            const int v = vertices_on_edge[2 * edge + q];
            if (v < 0 || v >= nvertices || (q && v == vertices_on_edge[2 * edge])) continue;
            for (int s = 0; s < vertex_degree; ++s) {
                if (edges_on_vertex[v * vertex_degree + s] != edge) continue;
                const int j = k * nvertices + v;
                const float sign = vertices_on_edge[2 * edge] == v ? -1.0f : 1.0f;
                const VertexReplay replay=vertex_replay(v,k,s,nedges,vertex_degree,normal_velocity,ke_edge,
                    edges_on_vertex,vertices_on_edge,dc_edge,inv_area_triangle);
                const float bpv=ad_probe_add_operand_live(f_vertex[v],replay.vort,false,false)?seed_pv_vertex[j]:0.0f;
                const double bvor = (double)seed_vorticity[j] + (double)bpv;
                AD_TEST(seed_vorticity[j], 12, j, 92); AD_TEST(seed_pv_vertex[j], 14, j, 94);
                AD_TEST(seed_ke_vertex[j], 13, j, 93); AD_TEST(inv_area_triangle[v], 6, v, 88);
                AD_TEST(dc_edge[edge], 5, edge, 83);
                const float w=ad_probe_mul(nullptr,0,sign,dc_edge[edge]);
                if(replay.vortex_term) bu += (double)ad_probe_mul_weight(w,normal_velocity[ad_i],false) * (double)replay.vort_weight * bvor;
                if(replay.kinetic_term) bk += (double)replay.ke_weight * (double)seed_ke_vertex[j];
            }
        }
        AD_TEST(bu, 1, ad_i, 83); AD_TEST(bk, 2, ad_i, 86);
        b_normal_velocity[ad_i] = bu; b_ke_edge[ad_i] = bk;
    }
    AD_LOOP(nvertices) {
        double b = b_f_vertex[ad_i];
        AD_TEST(f_vertex[ad_i], 7, ad_i, 94);
        for (int k = 0; k < nlev; ++k) {
            const int j = k * nvertices + ad_i;
            AD_TEST(seed_pv_vertex[j], 14, j, 94);
            const VertexReplay replay=vertex_replay(ad_i,k,-1,nedges,vertex_degree,normal_velocity,ke_edge,
                edges_on_vertex,vertices_on_edge,dc_edge,inv_area_triangle);
            if(ad_probe_add_operand_live(f_vertex[ad_i],replay.vort,false,true)) b += (double)seed_pv_vertex[j];
        }
        AD_TEST(b, 7, ad_i, 94); b_f_vertex[ad_i] = b;
    }
}
#endif

