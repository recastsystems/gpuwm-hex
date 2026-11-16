// SPDX-License-Identifier: Apache-2.0
// The coefficients rho_edge, kdiff and del2 are active: this is multilinear,
// and a fixed-weight stencil alone cannot represent the full Jacobian.
#include "horizontal_v841_ad.cuh"
#include "horizontal_v841_branches.cuh"
__device__ __forceinline__ float lap2_factor(int c, int e, const int *coe,
    const float *dc, const float *dv, const float *area) {
    const float sign = coe[2 * e] == c ? 1.0f : -1.0f;
    float x=ad_probe_div(nullptr,0,1.0f,area[c]);
    x=ad_probe_mul(nullptr,0,x,sign);x=ad_probe_mul(nullptr,0,x,dv[e]);
    return ad_probe_mul(nullptr,0,x,ad_probe_div(nullptr,0,1.0f,dc[e]));
}
struct Lap2Primal { float factor,grad,fg,flux,ks,flux0,flux1,term; };
__device__ __forceinline__ Lap2Primal lap2_primal(int c,int e,int k,int ncells,int nedges,
    const float *theta,const float *rho,const float *kd,const float *del2,const int *coe,
    const float *dc,const float *dv,const float *area) {
    int i0=k*ncells+coe[2*e],i1=k*ncells+coe[2*e+1],ie=k*nedges+e;
    float factor=lap2_factor(c,e,coe,dc,dv,area);
    float grad=ad_probe_addsub(nullptr,0,theta[i1],theta[i0],true);
    float fg=ad_probe_mul(nullptr,0,factor,grad),flux=ad_probe_mul(nullptr,0,fg,rho[ie]);
    float ks=ad_probe_addsub(nullptr,0,kd[i0],kd[i1],false);
    float flux0=ad_probe_mul(nullptr,0,flux,0.5f),flux1=ad_probe_mul(nullptr,0,flux0,ks);
    float term=ad_probe_mul(nullptr,0,flux1,del2[e]);
    return {factor,grad,fg,flux,ks,flux0,flux1,term};
}
struct Lap2Live { bool lap,tend; };
__device__ __forceinline__ Lap2Live lap2_live(int c,int k,int target,int ncells,int nedges,int stride,
    const float *theta,const float *rho,const float *kd,const float *del2,const int *eoc,const int *neoc,
    const int *coe,const float *dc,const float *dv,const float *area) {
    float lap=0.0f,tend=0.0f;bool ml=false,mt=false;
    for(int s=0;s<neoc[c];++s) {
        int e=eoc[c*stride+s];
        Lap2Primal p=lap2_primal(c,e,k,ncells,nedges,theta,rho,kd,del2,coe,dc,dv,area);
        bool ll=ad_probe_add_operand_live(lap,p.flux,false,true),rl=ad_probe_add_operand_live(lap,p.flux,false,false);
        bool lt=ad_probe_add_operand_live(tend,p.term,false,true),rt=ad_probe_add_operand_live(tend,p.term,false,false);
        if(s==target){ml=rl;mt=rt;}else if(s>target){ml=ml&&ll;mt=mt&&lt;}
        lap=ad_probe_addsub(nullptr,0,lap,p.flux,false);tend=ad_probe_addsub(nullptr,0,tend,p.term,false);
    }
    return {ml,mt};
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void theta_filter_lap2_v841_f32__tl(
    const float *theta_m, const float *rho_edge, const float *kdiff, const float *del2,
    const int *edges_on_cell, const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dc_edge, const float *dv_edge, const float *area_cell,
    int nlev, int ncells, int nedges, int max_edges, float *delsq_theta, float *tendency,
    const float *d_theta_m, const float *d_rho_edge, const float *d_kdiff, const float *d_del2,
    float *d_delsq_theta, float *d_tendency, AdFailureRecord *failure, unsigned *branch_record)
{
    // Count executed primal branches in shared memory, then combine counts.
    __shared__ unsigned ad_branch_local[28];
    for (unsigned q=threadIdx.x; q<28u; q+=blockDim.x) ad_branch_local[q]=0u;
    __syncthreads();
    ad_probe_theta_filter_lap2_v841_f32(theta_m, rho_edge, kdiff, del2, edges_on_cell, n_edges_on_cell, cells_on_edge, dc_edge, dv_edge, area_cell, nlev, ncells, nedges, max_edges, delsq_theta, tendency, ad_branch_local);
    __syncthreads();
    for (unsigned q=threadIdx.x; q<28u; q+=blockDim.x)
        if (ad_branch_local[q]) atomicAdd(&branch_record[q],ad_branch_local[q]);

    (void)delsq_theta; (void)tendency;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    AD_LOOP(nlev * ncells) {
        const int c = ad_i % ncells, k = ad_i / ncells;
        float dl = 0.0f, dt = 0.0f, pl=0.0f, pt=0.0f;
        AD_TEST(area_cell[c], 10, c, 288);
        if (area_cell[c] == 0.0f) { ad_fail(failure, ad_thread, 10, c, 288); continue; }
        for (int s = 0; s < n_edges_on_cell[c]; ++s) {
            const int e = edges_on_cell[c * max_edges + s];
            const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
            const int i0 = k * ncells + c0, i1 = k * ncells + c1, ie = k * nedges + e;
            AD_TEST(dc_edge[e], 8, e, 290); AD_TEST(dv_edge[e], 9, e, 289);
            if (dc_edge[e] == 0.0f) { ad_fail(failure, ad_thread, 8, e, 290); continue; }
            AD_TEST(theta_m[i0], 1, i0, 292); AD_TEST(theta_m[i1], 1, i1, 292);
            AD_TEST(rho_edge[ie], 2, ie, 294); AD_TEST(kdiff[i0], 3, i0, 298);
            AD_TEST(kdiff[i1], 3, i1, 299); AD_TEST(del2[e], 4, e, 299);
            AD_TEST(d_theta_m[i0], 1, i0, 292); AD_TEST(d_theta_m[i1], 1, i1, 292);
            AD_TEST(d_rho_edge[ie], 2, ie, 294); AD_TEST(d_kdiff[i0], 3, i0, 298);
            AD_TEST(d_kdiff[i1], 3, i1, 299); AD_TEST(d_del2[e], 4, e, 299);
            Lap2Primal p=lap2_primal(c,e,k,ncells,nedges,theta_m,rho_edge,kdiff,del2,cells_on_edge,dc_edge,dv_edge,area_cell);
            const float dgrad=(ad_probe_add_operand_live(theta_m[i1],theta_m[i0],true,true)?d_theta_m[i1]:0.0f)
                -(ad_probe_add_operand_live(theta_m[i1],theta_m[i0],true,false)?d_theta_m[i0]:0.0f);
            const float dfg=ad_probe_mul_weight(p.factor,p.grad,false)*dgrad;
            const float dflux=ad_probe_mul_weight(p.fg,rho_edge[ie],true)*dfg+
                              ad_probe_mul_weight(p.fg,rho_edge[ie],false)*d_rho_edge[ie];
            const float dks=(ad_probe_add_operand_live(kdiff[i0],kdiff[i1],false,true)?d_kdiff[i0]:0.0f)
                +(ad_probe_add_operand_live(kdiff[i0],kdiff[i1],false,false)?d_kdiff[i1]:0.0f);
            const float dflux0=ad_probe_mul_weight(p.flux,0.5f,true)*dflux;
            const float dflux1=ad_probe_mul_weight(p.flux0,p.ks,true)*dflux0+
                               ad_probe_mul_weight(p.flux0,p.ks,false)*dks;
            const float dterm=ad_probe_mul_weight(p.flux1,del2[e],true)*dflux1+
                              ad_probe_mul_weight(p.flux1,del2[e],false)*d_del2[e];
            dl=__fadd_rn(ad_probe_add_operand_live(pl,p.flux,false,true)?dl:0.0f,
                         ad_probe_add_operand_live(pl,p.flux,false,false)?dflux:0.0f);
            dt=__fadd_rn(ad_probe_add_operand_live(pt,p.term,false,true)?dt:0.0f,
                         ad_probe_add_operand_live(pt,p.term,false,false)?dterm:0.0f);
            pl=ad_probe_addsub(nullptr,0,pl,p.flux,false);pt=ad_probe_addsub(nullptr,0,pt,p.term,false);
        }
        AD_TEST(dl, 15, ad_i, 302); AD_TEST(dt, 16, ad_i, 303);
        d_delsq_theta[ad_i] = dl; d_tendency[ad_i] = dt;
    }
}
#endif

struct Lap2Bar { double theta, rho, kdiff, del2; };
__device__ __forceinline__ Lap2Bar lap2_bar(int c, int e, int k, int target_slot, int ncells, int nedges, int stride,
    const float *theta, const float *rho, const float *kd, const float *del2,
    const int *eoc,const int *neoc,const int *coe, const float *dc, const float *dv, const float *area,
    const float *bl, const float *bt)
{
    const int c0 = coe[2 * e], c1 = coe[2 * e + 1];
    const int i0 = k * ncells + c0, i1 = k * ncells + c1, ie = k * nedges + e;
    Lap2Primal p=lap2_primal(c,e,k,ncells,nedges,theta,rho,kd,del2,coe,dc,dv,area);
    Lap2Live live=lap2_live(c,k,target_slot,ncells,nedges,stride,theta,rho,kd,del2,eoc,neoc,coe,dc,dv,area);
    const float ylap = live.lap?bl[k * ncells + c]:0.0f, ytend = live.tend?bt[k * ncells + c]:0.0f;
    const double bflux1=(double)ytend*ad_probe_mul_weight(p.flux1,del2[e],true);
    const double bf=(double)ylap+(bflux1*ad_probe_mul_weight(p.flux0,p.ks,true))*ad_probe_mul_weight(p.flux,0.5f,true);
    Lap2Bar r = { (bf*ad_probe_mul_weight(p.fg,rho[ie],true))*ad_probe_mul_weight(p.factor,p.grad,false),
                  bf*ad_probe_mul_weight(p.fg,rho[ie],false),
                  bflux1*ad_probe_mul_weight(p.flux0,p.ks,false),
                  ytend*ad_probe_mul_weight(p.flux1,del2[e],false) };
    return r;
}
#if !defined(AD_ONLY_TL)
extern "C" __global__ void theta_filter_lap2_v841_f32__ad(
    const float *theta_m, const float *rho_edge, const float *kdiff, const float *del2,
    const int *edges_on_cell, const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dc_edge, const float *dv_edge, const float *area_cell,
    int nlev, int ncells, int nedges, int max_edges, float *delsq_theta, float *tendency,
    float *b_theta_m, float *b_rho_edge, float *b_kdiff, float *b_del2,
    const float *seed_delsq_theta, const float *seed_tendency, float *b_delsq_theta, float *b_tendency, AdFailureRecord *failure)
{
    (void)delsq_theta; (void)tendency;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    // Snapshots are disjoint from the consumed cotangents and input cotangents.
    AD_LOOP(nlev * ncells) { b_delsq_theta[ad_i] = 0.0f; b_tendency[ad_i] = 0.0f; }
    AD_LOOP(nlev * ncells) {
        const int target = ad_i % ncells, k = ad_i / ncells;
        double bx = b_theta_m[ad_i], bk = b_kdiff[ad_i];
        AD_TEST(theta_m[ad_i], 1, ad_i, 292); AD_TEST(kdiff[ad_i], 3, ad_i, 298);
        for (int s = 0; s < n_edges_on_cell[target]; ++s) {
            const int e = edges_on_cell[target * max_edges + s];
            const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
            for (int q = 0; q < 2; ++q) {
                const int c = q ? c1 : c0;
                if (c < 0 || c >= ncells || (q && c1 == c0)) continue;
                for (int t = 0; t < n_edges_on_cell[c]; ++t) {
                    if (edges_on_cell[c * max_edges + t] != e) continue;
                    const int j = k * ncells + c;
                    AD_TEST(area_cell[c], 10, c, 288); AD_TEST(dc_edge[e], 8, e, 290);
                    if (area_cell[c] == 0.0f || dc_edge[e] == 0.0f) {
                        if (area_cell[c] == 0.0f) ad_fail(failure, ad_thread, 10, c, 288);
                        if (dc_edge[e] == 0.0f) ad_fail(failure, ad_thread, 8, e, 290);
                        continue;
                    }
                    AD_TEST(seed_delsq_theta[j], 15, j, 302); AD_TEST(seed_tendency[j], 16, j, 303);
                    const Lap2Bar b = lap2_bar(c,e,k,t,ncells,nedges,max_edges,theta_m,rho_edge,kdiff,del2,
                        edges_on_cell,n_edges_on_cell,cells_on_edge,dc_edge,dv_edge,area_cell,seed_delsq_theta,seed_tendency);
                    int i0=k*ncells+c0,i1=k*ncells+c1;
                    if (target == c0 && ad_probe_add_operand_live(theta_m[i1],theta_m[i0],true,false)) bx -= b.theta;
                    if (target == c1 && ad_probe_add_operand_live(theta_m[i1],theta_m[i0],true,true)) bx += b.theta;
                    if (target == c0 && ad_probe_add_operand_live(kdiff[i0],kdiff[i1],false,true)) bk += b.kdiff;
                    if (target == c1 && ad_probe_add_operand_live(kdiff[i0],kdiff[i1],false,false)) bk += b.kdiff;
                }
            }
        }
        AD_TEST(bx, 1, ad_i, 292); AD_TEST(bk, 3, ad_i, 298);
        b_theta_m[ad_i] = bx; b_kdiff[ad_i] = bk;
    }
    AD_LOOP(nlev * nedges) {
        const int e = ad_i % nedges, k = ad_i / nedges;
        const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
        double br = b_rho_edge[ad_i];
        AD_TEST(rho_edge[ad_i], 2, ad_i, 294);
        for (int q = 0; q < 2; ++q) {
            const int c = q ? c1 : c0;
            if (c < 0 || c >= ncells || (q && c1 == c0)) continue;
            for (int t = 0; t < n_edges_on_cell[c]; ++t) {
                if (edges_on_cell[c * max_edges + t] != e) continue;
                if (area_cell[c] == 0.0f || dc_edge[e] == 0.0f) {
                    if (area_cell[c] == 0.0f) ad_fail(failure, ad_thread, 10, c, 288);
                    if (dc_edge[e] == 0.0f) ad_fail(failure, ad_thread, 8, e, 290);
                    continue;
                }
                const Lap2Bar b = lap2_bar(c,e,k,t,ncells,nedges,max_edges,theta_m,rho_edge,kdiff,del2,
                    edges_on_cell,n_edges_on_cell,cells_on_edge,dc_edge,dv_edge,area_cell,seed_delsq_theta,seed_tendency);
                br += b.rho;
            }
        }
        AD_TEST(br, 2, ad_i, 294); b_rho_edge[ad_i] = br;
        if (k == 0) {
            double bd = b_del2[e]; AD_TEST(del2[e], 4, e, 299);
            for (int kk = 0; kk < nlev; ++kk) for (int q = 0; q < 2; ++q) {
                const int c = q ? c1 : c0;
                if (c < 0 || c >= ncells || (q && c1 == c0)) continue;
                for (int t = 0; t < n_edges_on_cell[c]; ++t) {
                    if (edges_on_cell[c * max_edges + t] != e) continue;
                    if (area_cell[c] == 0.0f || dc_edge[e] == 0.0f) {
                        if (area_cell[c] == 0.0f) ad_fail(failure, ad_thread, 10, c, 288);
                        if (dc_edge[e] == 0.0f) ad_fail(failure, ad_thread, 8, e, 290);
                        continue;
                    }
                    const Lap2Bar b = lap2_bar(c,e,kk,t,ncells,nedges,max_edges,theta_m,rho_edge,kdiff,del2,
                        edges_on_cell,n_edges_on_cell,cells_on_edge,dc_edge,dv_edge,area_cell,seed_delsq_theta,seed_tendency);
                    bd += b.del2;
                }
            }
            AD_TEST(bd, 4, e, 299); b_del2[e] = bd;
        }
    }
}
#endif

