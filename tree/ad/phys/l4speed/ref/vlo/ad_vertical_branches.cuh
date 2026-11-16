// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ad_interpolate_branches.cuh"

// Slots 0--8 are the transport predicates. Slots 9--20 are the
// fp32 arithmetic helper predicates, 21--22 are min/max fallback guards,
// and 23--24 are the independent lower-residual extrema.
__device__ __forceinline__ float ad_vr_extreme(float x,bool maximum,int winner,unsigned *r) {
    bool chosen=maximum ? ad_pos(x) : ad_neg(x);
    ad_ib_record(r,winner,chosen);
    bool sub=mpas_is_subnormal_f32(x);
    ad_ib_record(r,maximum?22:21,sub);
    if(sub) { (void)ad_ib_decode(0.0f,r+16); (void)ad_ib_decode(x,r+16); }
    return maximum?mpas_max(0.0f,x):mpas_min(0.0f,x);
}
__device__ __forceinline__ float ad_vr_flux(float dt,float velocity,float a,float b,unsigned *r) {
    float p=ad_vr_extreme(velocity,true,4,r),m=ad_vr_extreme(velocity,false,5,r);
    unsigned *h=r+16;
    return ad_ib_mul(dt,ad_ib_addsub(ad_ib_mul(p,a,h),ad_ib_mul(m,b,h),false,h),h);
}
__device__ __forceinline__ float ad_vr_plain_flux(float dt,float v,float a,float b) {
    return mpas_mul(dt,mpas_add(mpas_mul(ad_plus(v),a),mpas_mul(ad_minus(v),b)));
}
__device__ __forceinline__ void ad_vr_trace(int t,int c,int nl,int nc,float dt,
        const float *s,const float *rho,const float *velocity,const float *rdzw,
        const float *high,const float *res,unsigned *r) {
    unsigned *h=r+16;
    for(int k=0;k<nl;++k){ad_ib_record(r,1,true);int q=ad_q(t,k,c,nl,nc);(void)ad_ib_mul(s[q],rho[k*nc+c],h);}
    ad_ib_record(r,1,false);
    for(int k=0;k<=nl;++k){ad_ib_record(r,2,true);(void)ad_ib_mul(dt,high[ad_qi(t,k,c,nl,nc)],h);}
    ad_ib_record(r,2,false);
    for(int k=0;k+1<nl;++k){
        ad_ib_record(r,3,true);int a=ad_q(t,k,c,nl,nc),b=ad_q(t,k+1,c,nl,nc);
        float f=ad_vr_flux(dt,velocity[(k+1)*nc+c],s[a],s[b],r);
        float initial=mpas_mul(s[a],rho[k*nc+c]);
        (void)ad_ib_addsub(initial,ad_ib_mul(f,rdzw[k],h),true,h);
    }
    ad_ib_record(r,3,false);
    for(int k=1;k<nl;++k){
        ad_ib_record(r,6,true);int a=ad_q(t,k-1,c,nl,nc),b=ad_q(t,k,c,nl,nc);
        float f=ad_vr_flux(dt,velocity[k*nc+c],s[a],s[b],r);
        float before=mpas_mul(s[b],rho[k*nc+c]);
        if(k+1<nl){int next=ad_q(t,k+1,c,nl,nc);float upper=ad_vr_plain_flux(dt,velocity[(k+1)*nc+c],s[b],s[next]);before=mpas_sub(before,mpas_mul(upper,rdzw[k]));}
        (void)ad_ib_addsub(before,ad_ib_mul(f,rdzw[k],h),false,h);
        (void)ad_ib_addsub(mpas_mul(dt,high[ad_qi(t,k,c,nl,nc)]),f,true,h);
    }
    ad_ib_record(r,6,false);
    for(int k=0;k<nl;++k){
        float u=res[ad_qi(t,k+1,c,nl,nc)],l=res[ad_qi(t,k,c,nl,nc)];
        float mn_u=ad_vr_extreme(u,false,7,r),mx_l=ad_vr_extreme(l,true,23,r);
        float mx_u=ad_vr_extreme(u,true,8,r),mn_l=ad_vr_extreme(l,false,24,r);
        (void)ad_ib_mul(-rdzw[k],ad_ib_addsub(mn_u,mx_l,true,h),h);
        (void)ad_ib_mul(-rdzw[k],ad_ib_addsub(mx_u,mn_l,true,h),h);
    }
}
