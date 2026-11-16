// SPDX-License-Identifier: Apache-2.0
// del4 and h4 are active parameters. The output tendency is overwritten in place.
#include "horizontal_v841_ad.cuh"
#include "horizontal_v841_branches.cuh"
__device__ __forceinline__ float lap4_geometry(int c, int e, const int *coe,
    const float *dc, const float *dv, const float *area) {
    const float sign = coe[2 * e] == c ? 1.0f : -1.0f;
    return ((dv[e] * sign) * (1.0f / dc[e])) * (1.0f / area[c]);
}
struct Lap4Primal { float inv,idc,q,a,b,c,factor,grad,term; };
__device__ __forceinline__ Lap4Primal lap4_primal(int c,int e,int k,int ncells,
    const float *theta,const float *del4,const int *coe,const float *dc,const float *dv,
    const float *area,float h4) {
    float inv=ad_probe_div(nullptr,0,1.0f,area[c]),idc=ad_probe_div(nullptr,0,1.0f,dc[e]);
    float q=ad_probe_mul(nullptr,0,h4,inv),a=ad_probe_mul(nullptr,0,del4[e],q);
    float b=ad_probe_mul(nullptr,0,a,dv[e]),cs=ad_probe_mul(nullptr,0,b,coe[2*e]==c?1.0f:-1.0f);
    float factor=ad_probe_mul(nullptr,0,cs,idc);
    float grad=ad_probe_addsub(nullptr,0,theta[k*ncells+coe[2*e+1]],theta[k*ncells+coe[2*e]],true);
    return {inv,idc,q,a,b,cs,factor,grad,ad_probe_mul(nullptr,0,factor,grad)};
}
struct Lap4Live { bool input,term; };
__device__ __forceinline__ Lap4Live lap4_live(int c,int k,int target,int ncells,int stride,
    const float *theta,const float *del4,const int *eoc,const int *neoc,const int *coe,
    const float *dc,const float *dv,const float *area,float h4,const float *tendency) {
    float value=tendency[k*ncells+c];bool mi=true,mt=false;
    for(int s=0;s<neoc[c];++s) {
        int e=eoc[c*stride+s];
        Lap4Primal p=lap4_primal(c,e,k,ncells,theta,del4,coe,dc,dv,area,h4);
        bool l=ad_probe_add_operand_live(value,p.term,true,true),r=ad_probe_add_operand_live(value,p.term,true,false);
        mi=mi&&l;if(s==target)mt=r;else if(s>target)mt=mt&&l;
        value=ad_probe_addsub(nullptr,0,value,p.term,true);
    }
    return {mi,mt};
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void theta_filter_lap4_v841_f32__tl(
    const float *delsq_theta, const float *del4, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge, const float *dc_edge,
    const float *dv_edge, const float *area_cell, float h4, int nlev, int ncells,
    int max_edges, float *tendency,
    const float *d_delsq_theta, const float *d_del4, const float *d_h4,
    const float *d_tendency_in, float *d_tendency_out, int nedges, AdFailureRecord *failure, unsigned *branch_record)
{
    // Count executed primal branches in shared memory, then combine counts.
    __shared__ unsigned ad_branch_local[28];
    for (unsigned q=threadIdx.x; q<28u; q+=blockDim.x) ad_branch_local[q]=0u;
    __syncthreads();
    ad_probe_theta_filter_lap4_v841_f32(delsq_theta, del4, edges_on_cell, n_edges_on_cell, cells_on_edge, dc_edge, dv_edge, area_cell, h4, nlev, ncells, max_edges, tendency, ad_branch_local);
    __syncthreads();
    for (unsigned q=threadIdx.x; q<28u; q+=blockDim.x)
        if (ad_branch_local[q]) atomicAdd(&branch_record[q],ad_branch_local[q]);

    (void)nedges;
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    AD_LOOP(nlev * ncells) {
        const int c = ad_i % ncells, k = ad_i / ncells;
        float dt = d_tendency_in[ad_i], compensation = 0.0f, pt=tendency[ad_i];
        AD_TEST(h4, 9, 0, 333); AD_TEST(*d_h4, 9, 0, 333);
        AD_TEST(dt, 13, ad_i, 328); AD_TEST(area_cell[c], 8, c, 334);
        if (area_cell[c] == 0.0f) { ad_fail(failure, ad_thread, 8, c, 334); continue; }
        for (int s = 0; s < n_edges_on_cell[c]; ++s) {
            const int e = edges_on_cell[c * max_edges + s];
            const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
            const int i0 = k * ncells + c0, i1 = k * ncells + c1;
            AD_TEST(dc_edge[e], 6, e, 335); AD_TEST(dv_edge[e], 7, e, 334);
            if (dc_edge[e] == 0.0f) { ad_fail(failure, ad_thread, 6, e, 335); continue; }
            AD_TEST(delsq_theta[i0], 1, i0, 336); AD_TEST(delsq_theta[i1], 1, i1, 336);
            AD_TEST(del4[e], 2, e, 333); AD_TEST(d_del4[e], 2, e, 333);
            AD_TEST(d_delsq_theta[i0], 1, i0, 336); AD_TEST(d_delsq_theta[i1], 1, i1, 336);
            const float sign = cells_on_edge[2 * e] == c ? 1.0f : -1.0f;
            const Lap4Primal p=lap4_primal(c,e,k,ncells,delsq_theta,del4,cells_on_edge,dc_edge,dv_edge,area_cell,h4);
            const float dq=*d_h4*ad_probe_mul_weight(h4,p.inv,true);
            const float da=d_del4[e]*ad_probe_mul_weight(del4[e],p.q,true)+dq*ad_probe_mul_weight(del4[e],p.q,false);
            const float db=da*ad_probe_mul_weight(p.a,dv_edge[e],true);
            const float dcs=db*ad_probe_mul_weight(p.b,sign,true);
            const float dfactor=dcs*ad_probe_mul_weight(p.c,p.idc,true);
            const float dgrad=(ad_probe_add_operand_live(delsq_theta[i1],delsq_theta[i0],true,true)?d_delsq_theta[i1]:0.0f)
                -(ad_probe_add_operand_live(delsq_theta[i1],delsq_theta[i0],true,false)?d_delsq_theta[i0]:0.0f);
            if(!ad_probe_add_operand_live(pt,p.term,true,true)){dt=0.0f;compensation=0.0f;}
            const float term=ad_probe_add_operand_live(pt,p.term,true,false)?
                -(dfactor*ad_probe_mul_weight(p.factor,p.grad,true)+dgrad*ad_probe_mul_weight(p.factor,p.grad,false)):0.0f;
            const float corrected = __fsub_rn(term, compensation);
            const float next = __fadd_rn(dt, corrected);
            compensation = __fsub_rn(__fsub_rn(next, dt), corrected);
            dt = next;
            pt=ad_probe_addsub(nullptr,0,pt,p.term,true);
        }
        AD_TEST(dt, 13, ad_i, 340); d_tendency_out[ad_i] = dt;
    }
}
#endif

struct Lap4Bar { double theta, del4, h4; };
__device__ __forceinline__ Lap4Bar lap4_bar(int c,int e,int k,int slot,int ncells,int stride,
    const float *theta,const float *del4,const int *eoc,const int *neoc,const int *coe,
    const float *dc,const float *dv,const float *area,float h4,const float *tendency,const float *by)
{
    const int c0 = coe[2 * e], c1 = coe[2 * e + 1];
    const float sign = coe[2 * e] == c ? 1.0f : -1.0f;
    const Lap4Primal p=lap4_primal(c,e,k,ncells,theta,del4,coe,dc,dv,area,h4);
    const Lap4Live live=lap4_live(c,k,slot,ncells,stride,theta,del4,eoc,neoc,coe,dc,dv,area,h4,tendency);
    const double seed = live.term?by[k * ncells + c]:0.0;
    const double bf=-seed*ad_probe_mul_weight(p.factor,p.grad,true);
    const double bc=bf*ad_probe_mul_weight(p.c,p.idc,true);
    const double bb=bc*ad_probe_mul_weight(p.b,sign,true);
    const double ba=bb*ad_probe_mul_weight(p.a,dv[e],true);
    Lap4Bar b = { -seed*ad_probe_mul_weight(p.factor,p.grad,false),
                  ba*ad_probe_mul_weight(del4[e],p.q,true),
                  (ba*ad_probe_mul_weight(del4[e],p.q,false))*ad_probe_mul_weight(h4,p.inv,true) };
    return b;
}
// The output seed and input cotangent are separate, per section 36. This
// avoids a race between gather reads and the input-cotangent write.
#if !defined(AD_ONLY_TL)
extern "C" __global__ void theta_filter_lap4_v841_f32__ad(
    const float *delsq_theta, const float *del4, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge, const float *dc_edge,
    const float *dv_edge, const float *area_cell, float h4, int nlev, int ncells,
    int max_edges, float *tendency,
    float *b_delsq_theta, float *b_del4, float *b_h4,
    const float *seed_tendency, float *b_tendency_in, int nedges, float *b_tendency, AdFailureRecord *failure)
{
    const int ad_thread = blockDim.x * blockIdx.x + threadIdx.x;
    // Snapshots are disjoint from the consumed cotangents and input cotangents.
    AD_LOOP(nlev * ncells) { b_tendency[ad_i] = 0.0f; }
    AD_LOOP(nlev * ncells) {
        const int target = ad_i % ncells, k = ad_i / ncells;
        double bx = b_delsq_theta[ad_i];
        AD_TEST(delsq_theta[ad_i], 1, ad_i, 336); AD_TEST(seed_tendency[ad_i], 13, ad_i, 340);
        for (int s = 0; s < n_edges_on_cell[target]; ++s) {
            const int e = edges_on_cell[target * max_edges + s];
            const int c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
            for (int q = 0; q < 2; ++q) {
                const int c = q ? c1 : c0;
                if (c < 0 || c >= ncells || (q && c1 == c0)) continue;
                for (int t = 0; t < n_edges_on_cell[c]; ++t) {
                    if (edges_on_cell[c * max_edges + t] != e) continue;
                    if (area_cell[c] == 0.0f || dc_edge[e] == 0.0f) {
                        if (area_cell[c] == 0.0f) ad_fail(failure, ad_thread, 8, c, 334);
                        if (dc_edge[e] == 0.0f) ad_fail(failure, ad_thread, 6, e, 335);
                        continue;
                    }
                    const Lap4Bar b=lap4_bar(c,e,k,t,ncells,max_edges,delsq_theta,del4,edges_on_cell,n_edges_on_cell,
                                              cells_on_edge,dc_edge,dv_edge,area_cell,h4,tendency,seed_tendency);
                    int i0=k*ncells+c0,i1=k*ncells+c1;
                    if(target==c0 && ad_probe_add_operand_live(delsq_theta[i1],delsq_theta[i0],true,false))bx-=b.theta;
                    if(target==c1 && ad_probe_add_operand_live(delsq_theta[i1],delsq_theta[i0],true,true))bx+=b.theta;
                }
            }
        }
        AD_TEST(bx, 1, ad_i, 336); b_delsq_theta[ad_i] = bx;
        const Lap4Live input_live=lap4_live(target,k,-1,ncells,max_edges,delsq_theta,del4,edges_on_cell,
            n_edges_on_cell,cells_on_edge,dc_edge,dv_edge,area_cell,h4,tendency);
        b_tendency_in[ad_i]=__fadd_rn(b_tendency_in[ad_i],input_live.input?seed_tendency[ad_i]:0.0f);
    }
    AD_LOOP(nedges) {
        const int e = ad_i, c0 = cells_on_edge[2 * e], c1 = cells_on_edge[2 * e + 1];
        double bd = b_del4[e]; AD_TEST(del4[e], 2, e, 333);
        for (int k = 0; k < nlev; ++k) for (int q = 0; q < 2; ++q) {
            const int c = q ? c1 : c0;
            if (c < 0 || c >= ncells || (q && c1 == c0)) continue;
            for (int t = 0; t < n_edges_on_cell[c]; ++t) {
                if (edges_on_cell[c * max_edges + t] != e) continue;
                if (area_cell[c] == 0.0f || dc_edge[e] == 0.0f) {
                    if (area_cell[c] == 0.0f) ad_fail(failure, ad_thread, 8, c, 334);
                    if (dc_edge[e] == 0.0f) ad_fail(failure, ad_thread, 6, e, 335);
                    continue;
                }
                const Lap4Bar b=lap4_bar(c,e,k,t,ncells,max_edges,delsq_theta,del4,edges_on_cell,n_edges_on_cell,
                                          cells_on_edge,dc_edge,dv_edge,area_cell,h4,tendency,seed_tendency);
                bd += b.del4;
            }
        }
        AD_TEST(bd, 2, e, 333); b_del4[e] = bd;
    }
    // Only block zero handles the scalar cotangent. Its threads traverse
    // fixed strides and use a fixed tree, including non-power-of-two blocks.
    // Other blocks can finish independently: none touches b_h4.
    if(blockIdx.x==0) {
        __shared__ double sums[1024];
        const int t=threadIdx.x;
        double sum=0.0;
        for(int i=t;i<nlev*ncells;i+=blockDim.x) {
            const int c=i%ncells,k=i/ncells;
            for(int s=0;s<n_edges_on_cell[c];++s) {
                const int e=edges_on_cell[c*max_edges+s];
                if(area_cell[c]==0.0f || dc_edge[e]==0.0f) {
                    if (area_cell[c] == 0.0f) ad_fail(failure,t,8,c,334);
                    if (dc_edge[e] == 0.0f) ad_fail(failure,t,6,e,335);
                    continue;
                }
                const Lap4Bar b=lap4_bar(c,e,k,s,ncells,max_edges,delsq_theta,del4,edges_on_cell,n_edges_on_cell,
                                         cells_on_edge,dc_edge,dv_edge,area_cell,h4,tendency,seed_tendency);
                sum+=(double)b.h4;
            }
        }
        sums[t]=sum;__syncthreads();
        for(int active=blockDim.x;active>1;active=(active+1)/2) {
            const int next=(active+1)/2;
            if(t<next && t+next<active)sums[t]+=sums[t+next];
            __syncthreads();
        }
        if(t==0){float out=(float)((double)*b_h4+sums[0]);
            if(!isfinite(out))ad_fail(failure,0,9,0,333);else *b_h4=out;}
    }
}
#endif

