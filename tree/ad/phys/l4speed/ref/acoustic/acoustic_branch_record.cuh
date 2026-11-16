// SPDX-License-Identifier: Apache-2.0
// Branch observation for the pinned mpas FP32 primal helpers. Counters are
// unsigned integer additions, so concurrent scheduling cannot change them.
#pragma once
#include "acoustic_mpas_fp32.cuh"
__device__ __forceinline__ void abr_count(unsigned *p,int slot,bool arm) {
    if(p) ++p[2*slot+(arm?1:0)];
}
__device__ __forceinline__ void abr_fallback(unsigned *p,int base,float left,float right,int op) {
    unsigned lm=mpas_f32_magnitude_bits(left),rm=mpas_f32_magnitude_bits(right);
    bool ls=lm!=0u && (lm&0x7f800000u)==0u,rs=rm!=0u && (rm&0x7f800000u)==0u;
    abr_count(p,base+0,ls);abr_count(p,base+0,rs);
    if(ls)abr_count(p,base+12,(__float_as_uint(left)&0x80000000u)!=0u);
    if(rs)abr_count(p,base+12,(__float_as_uint(right)&0x80000000u)!=0u);
    double l=mpas_f32_to_f64_ieee(left),r=mpas_f32_to_f64_ieee(right),v;
    if(op==0)v=l+r;else if(op==1)v=l-r;else if(op==2)v=l*r;else v=l/r;
    double m=fabs(v);bool small=m>0.0 && m<1.1754943508222875e-38;abr_count(p,base+1,small);
    if(small)abr_count(p,base+13,__double_as_longlong(v)<0LL);
}
__device__ __forceinline__ float abr_add(float l,float r,unsigned *p,int base) {
    float v=mpas_add(l,r);if(!p)return v;
    float ordinary=l+r;unsigned exp=mpas_f32_magnitude_bits(ordinary)>>23;
    bool fast=exp>24u;abr_count(p,base+2,fast);if(fast)return v;
    unsigned lm=mpas_f32_magnitude_bits(l),rm=mpas_f32_magnitude_bits(r),le=lm>>23,re=rm>>23;
    bool sub=(le==0u&&lm!=0u)||(re==0u&&rm!=0u);abr_count(p,base+3,sub);
    if(sub){abr_fallback(p,base,l,r,0);return v;}
    bool cancel=le<=23u&&re<=23u&&exp==0u&&lm!=rm&&lm!=0u&&rm!=0u;
    abr_count(p,base+4,cancel);if(cancel)abr_fallback(p,base,l,r,0);return v;
}
__device__ __forceinline__ float abr_sub(float l,float r,unsigned *p,int base) {
    float v=mpas_sub(l,r);if(!p)return v;
    float ordinary=l-r;unsigned exp=mpas_f32_magnitude_bits(ordinary)>>23;
    bool fast=exp>24u;abr_count(p,base+5,fast);if(fast)return v;
    unsigned lm=mpas_f32_magnitude_bits(l),rm=mpas_f32_magnitude_bits(r),le=lm>>23,re=rm>>23;
    bool sub=(le==0u&&lm!=0u)||(re==0u&&rm!=0u);abr_count(p,base+6,sub);
    if(sub){abr_fallback(p,base,l,r,1);return v;}
    bool cancel=le<=23u&&re<=23u&&exp==0u&&lm!=rm&&lm!=0u&&rm!=0u;
    abr_count(p,base+7,cancel);if(cancel)abr_fallback(p,base,l,r,1);return v;
}
__device__ __forceinline__ float abr_mul(float l,float r,unsigned *p,int base) {
    float v=mpas_mul(l,r);if(!p)return v;
    float ordinary=l*r;bool small=(mpas_f32_magnitude_bits(ordinary)&0x7f800000u)==0u;
    abr_count(p,base+8,small);if(small){bool nonzero=mpas_f32_magnitude_bits(l)!=0u&&mpas_f32_magnitude_bits(r)!=0u;
        abr_count(p,base+9,nonzero);if(nonzero)abr_fallback(p,base,l,r,2);}
    return v;
}
__device__ __forceinline__ float abr_div(float l,float r,unsigned *p,int base) {
    float v=mpas_div(l,r);if(!p)return v;
    unsigned lm=mpas_f32_magnitude_bits(l),rm=mpas_f32_magnitude_bits(r),le=lm>>23,re=rm>>23;
    bool sub=(le==0u&&lm!=0u)||(re==0u&&rm!=0u);abr_count(p,base+10,sub);
    if(sub){abr_fallback(p,base,l,r,3);return v;}
    float ordinary=l/r;bool small=lm!=0u&&rm!=0u&&le+126u<=re&&(mpas_f32_magnitude_bits(ordinary)&0x7f800000u)==0u;
    abr_count(p,base+11,small);if(small)abr_fallback(p,base,l,r,3);return v;
}
