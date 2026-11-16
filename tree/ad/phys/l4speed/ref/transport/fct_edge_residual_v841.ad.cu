// SPDX-License-Identifier: Apache-2.0
// Hand rule for cuda_transport_v841.py:295. See its spec for the ordered ABI.
#include "helpers.cuh"
#define HEX_AD_TRACE_BASE 3
#include "trace_helpers.cuh"

__device__ __forceinline__ bool ad_finite(float a) {
    return (__float_as_uint(a) & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ float pval(float x,unsigned int *rec=nullptr) { return ad_trace_max(0.0f,x,rec); }
__device__ __forceinline__ float mval(float x,unsigned int *rec=nullptr) { return ad_trace_min(0.0f,x,rec); }

#if !defined(AD_AD_ONLY)
extern "C" __global__ void fct_edge_residual_v841__tl(
    int ntracers, int nlev, int ncells, int nedges, float dt,
    const float *source_old, const float *velocity, const float *dv_edge,
    const int *cells_on_edge, const float *high_flux,
    const float *upwind_flux, const float *horizontal_residual,
    const float *d_dt_ptr, const float *d_source_old, const float *d_velocity,
    const float *d_dv_edge, const float *d_high_flux,
    float *d_upwind_flux, float *d_horizontal_residual, Failure *failure, unsigned int *branch_record)
{
    (void)upwind_flux; (void)horizontal_residual;
    const float d_dt=*d_dt_ptr;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    unsigned int branch_local[38]={0};
    for (int e=thread; e<nedges; e+=stride) {
        const int c0=cells_on_edge[2*e], c1=cells_on_edge[2*e+1];
        if(c0<0){ad_fail(failure,0);continue;}if(c0>=ncells){ad_fail(failure,34);continue;}if(c1<0){ad_fail(failure,35);continue;}if(c1>=ncells){ad_fail(failure,36);continue;}
        {if(!ad_finite(dt))ad_fail(failure,1);if(!ad_finite(d_dt))ad_fail(failure,2);}
        {if(!ad_finite(dv_edge[e]))ad_fail(failure,3);if(!ad_finite(d_dv_edge[e]))ad_fail(failure,4);}
        for (int q=0;q<ntracers;++q) for (int k=0;k<nlev;++k) {
            const int i=QE3(q,k,e,nlev,nedges), vindex=E2(k,e,nedges);
            const int a=Q3(q,k,c0,nlev,ncells), b=Q3(q,k,c1,nlev,ncells);
            float v=velocity[vindex], dv=d_velocity[vindex], x=source_old[a], y=source_old[b];
            ad_branch(branch_local,1,ad_pos(v));ad_branch(branch_local,2,ad_neg(v));
            {if(!ad_finite(v))ad_fail(failure,5);if(!ad_finite(dv))ad_fail(failure,6);}
            {if(!ad_finite(x))ad_fail(failure,7);if(!ad_finite(y))ad_fail(failure,8);if(!ad_finite(d_source_old[a]))ad_fail(failure,9);if(!ad_finite(d_source_old[b]))ad_fail(failure,10);}
            {if(!ad_finite(high_flux[i]))ad_fail(failure,11);if(!ad_finite(d_high_flux[i]))ad_fail(failure,12);}
            float p=pval(v,branch_local), m=mval(v,branch_local);
            float sum=ad_trace_add(ad_trace_mul(p,x,branch_local),ad_trace_mul(m,y,branch_local),branch_local);
            float dsum=mpas_add(mpas_add(mpas_mul(ad_pos(v)?dv:0.0f,x),mpas_mul(p,d_source_old[a])),
                                mpas_add(mpas_mul(ad_neg(v)?dv:0.0f,y),mpas_mul(m,d_source_old[b])));
            float factor=ad_trace_mul(dv_edge[e],dt,branch_local);float primal_upwind=ad_trace_mul(factor,sum,branch_local);(void)ad_trace_sub(ad_trace_mul(dt,high_flux[i],branch_local),primal_upwind,branch_local);
            float dfactor=mpas_add(mpas_mul(d_dv_edge[e],dt),mpas_mul(dv_edge[e],d_dt));
            float du=mpas_add(mpas_mul(dfactor,sum),mpas_mul(factor,dsum));
            float dr=mpas_sub(mpas_add(mpas_mul(d_dt,high_flux[i]),mpas_mul(dt,d_high_flux[i])),du);
            {if(!ad_finite(sum))ad_fail(failure,13);if(!ad_finite(factor))ad_fail(failure,14);if(!ad_finite(du))ad_fail(failure,15);}
            if (!ad_finite(dr)) ad_fail(failure,16);
            d_upwind_flux[i]=du; d_horizontal_residual[i]=dr;
        }
    }
    ad_branch_flush(branch_record,branch_local,19);
}

#endif

// Each edge owner writes edge gradients; each cell owner gathers its incident
// edge contributions in increasing edge order. There are no atomic writes.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void fct_edge_residual_v841__ad(
    int ntracers, int nlev, int ncells, int nedges, float dt,
    const float *source_old, const float *velocity, const float *dv_edge,
    const int *cells_on_edge, const float *high_flux,
    float *b_upwind_flux, float *b_horizontal_residual,
    float *b_dt, float *b_source_old, float *b_velocity,
    float *b_dv_edge, float *b_high_flux, Failure *failure)
{
    const int thread=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=gridDim.x*blockDim.x;
    for (int e=thread;e<nedges;e+=stride) {
        int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
        if(c0<0){ad_fail(failure,17);continue;}if(c0>=ncells){ad_fail(failure,37);continue;}if(c1<0){ad_fail(failure,38);continue;}if(c1>=ncells){ad_fail(failure,39);continue;}
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
    for (int c=thread;c<ncells;c+=stride) for (int q=0;q<ntracers;++q)
      for(int base=0;base<nlev;base+=64){
        double sum[64];int levels=min(64,nlev-base);
        for(int l=0;l<levels;++l)sum[l]=(double)b_source_old[Q3(q,base+l,c,nlev,ncells)];
        for (int e=0;e<nedges;++e) {
            int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
            if (c0!=c && c1!=c) continue;
            if (c0<0 || c0>=ncells || c1<0 || c1>=ncells) continue;
            float factor=mpas_mul(dv_edge[e],dt);
            for(int l=0;l<levels;++l){int k=base+l,i=QE3(q,k,e,nlev,nedges);
                float v=velocity[E2(k,e,nedges)];
                double bs=((double)b_upwind_flux[i]-(double)b_horizontal_residual[i])*(double)factor;
                if(c0==c)sum[l]+=bs*(double)pval(v);
                if(c1==c)sum[l]+=bs*(double)mval(v);
            }
        }
        for(int l=0;l<levels;++l){int ci=Q3(q,base+l,c,nlev,ncells);b_source_old[ci]=(float)sum[l];if(!ad_finite(b_source_old[ci]))ad_fail(failure,32);}
    }
    if (thread==0) {
        double total=(double)*b_dt;
        for (int e=0;e<nedges;++e) {
            int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
            if (c0<0 || c0>=ncells || c1<0 || c1>=ncells) continue;
            for (int q=0;q<ntracers;++q) for (int k=0;k<nlev;++k) {
                int i=QE3(q,k,e,nlev,nedges); float v=velocity[E2(k,e,nedges)];
                float sum=mpas_add(mpas_mul(pval(v),source_old[Q3(q,k,c0,nlev,ncells)]),mpas_mul(mval(v),source_old[Q3(q,k,c1,nlev,ncells)]));
                double bar=(double)b_upwind_flux[i]-(double)b_horizontal_residual[i];
                total+=bar*(double)dv_edge[e]*(double)sum+(double)b_horizontal_residual[i]*(double)high_flux[i];
            }
        }
        *b_dt=(float)total;
        if (!ad_finite(*b_dt)) ad_fail(failure,33);
    }
}
#endif

// Runs after all reverse readers on the same stream.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void fct_edge_residual_v841__ad_clear(int ntracers, int nlev, int nedges, float *b_upwind_flux, float *b_horizontal_residual)
{
 int c = blockIdx.x * blockDim.x + threadIdx.x;
 if (c < nedges) for (int q=0; q<ntracers; ++q) for (int k=0; k<nlev; ++k) { int i=QE3(q,k,c,nlev,nedges); b_upwind_flux[i] = 0.0f; b_horizontal_residual[i] = 0.0f; }
}
#endif
