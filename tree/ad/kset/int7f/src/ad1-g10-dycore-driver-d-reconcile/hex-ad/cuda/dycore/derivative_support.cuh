// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef __CUDACC_RTC__
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#include <math.h>
#endif

// Device refusal buffer: one zero-initialised int32 per source_id.  Integer
// exchanges are deterministic; the host maps each nonzero slot to its source.
__device__ __forceinline__ void ad_refuse(int *failure, uint64_t, uint64_t, uint64_t source) {
    int slot;
    switch (source) {
      case 1632: case 164801: case 1660: case 1679: slot=0; break;
      case 164802: case 1661: case 1680: slot=1; break;
      case 1662: case 1681: slot=2; break;
      case 1683: slot=3; break;
      case 1684: slot=4; break;
      default: slot=0; break;
    }
    atomicExch(failure+slot,1);
}
__device__ __forceinline__ float ad_add(float a,float b) { return __fadd_rn(a,b); }
__device__ __forceinline__ float ad_sub(float a,float b) { return __fsub_rn(a,b); }
__device__ __forceinline__ float ad_mul(float a,float b) { return __fmul_rn(a,b); }
__device__ __forceinline__ float ad_div(float a,float b) { return __fdiv_rn(a,b); }
__device__ __forceinline__ unsigned ad_mag(float x) { return __float_as_uint(x)&0x7fffffffu; }
__device__ __forceinline__ bool ad_subnormal(float x) {
    const unsigned u=ad_mag(x); return u!=0u && (u&0x7f800000u)==0u;
}
__device__ __forceinline__ float ad_flush(float x) {
    return ad_subnormal(x)?__uint_as_float(__float_as_uint(x)&0x80000000u):x;
}
// Reconstruct the exact predicates of cuda_fp32.py while compiled with FTZ
// disabled: explicitly emulate the forward helper's FTZ ordinary operation.
__device__ __forceinline__ float ad_forward_add(float a,float b,bool sub) {
    float o=ad_flush(sub?ad_sub(ad_flush(a),ad_flush(b)):ad_add(ad_flush(a),ad_flush(b)));
    if ((ad_mag(o)>>23)>24u) return o;
    return sub?ad_sub(a,b):ad_add(a,b);
}
__device__ __forceinline__ void ad_add_partials(float a,float b,bool sub,float &pa,float &pb) {
    const float ordinary=ad_flush(sub?ad_sub(ad_flush(a),ad_flush(b)):ad_add(ad_flush(a),ad_flush(b)));
    const bool early=(ad_mag(ordinary)>>23)>24u;
    pa=early&&ad_subnormal(a)?0.0f:1.0f; pb=early&&ad_subnormal(b)?0.0f:1.0f;
}
__device__ __forceinline__ void ad_mul_partials(float a,float b,float &pa,float &pb) {
    const bool z=ad_mag(a)==0u || ad_mag(b)==0u;
    pa=z?ad_flush(b):b; pb=z?ad_flush(a):a;
}
// Each branch slot is [not taken,taken]; integer increments are deterministic.
__device__ __forceinline__ void ad_count(uint32_t *r,int slot,bool yes) {
    atomicAdd(r+2*slot+(yes?1:0),1u);
}
// Decode/pack helpers are common to all recorded arithmetic call sites.
// common+0: decode subnormal; +1: decoded sign; +2: narrow pack; +3: pack sign.
__device__ __forceinline__ double ad_record_decode(uint32_t *r,int common,float x) {
    bool sub=ad_subnormal(x); ad_count(r,common,sub);
    if(sub) ad_count(r,common+1,(__float_as_uint(x)&0x80000000u)!=0);
    return (double)x;
}
__device__ __forceinline__ void ad_record_pack(uint32_t *r,int common,double v) {
    double a=fabs(v); bool sub=a>0.0 && a<1.1754943508222875e-38;
    ad_count(r,common+2,sub); if(sub) ad_count(r,common+3,signbit(v));
}
// base: source predicates in source order. Only reached predicates increment.
__device__ __forceinline__ void ad_record_add(uint32_t *r,int base,int common,float a,float b,bool sub) {
    float o=ad_flush(sub?ad_sub(ad_flush(a),ad_flush(b)):ad_add(ad_flush(a),ad_flush(b)));
    unsigned am=ad_mag(a),bm=ad_mag(b),oe=ad_mag(o)>>23,ae=am>>23,be=bm>>23;
    bool early=oe>24u; ad_count(r,base,early); if(early)return;
    bool tiny=(ae==0u&&am!=0u)||(be==0u&&bm!=0u); ad_count(r,base+1,tiny);
    if(tiny){double x=ad_record_decode(r,common,a),y=ad_record_decode(r,common,b);ad_record_pack(r,common,sub?x-y:x+y);return;}
    bool cancel=ae<=23u&&be<=23u&&oe==0u&&am!=bm&&am!=0u&&bm!=0u;
    ad_count(r,base+2,cancel);
    if(cancel){double x=ad_record_decode(r,common,a),y=ad_record_decode(r,common,b);ad_record_pack(r,common,sub?x-y:x+y);}
}
__device__ __forceinline__ void ad_record_mul(uint32_t *r,int base,int common,float a,float b) {
    float o=ad_flush(ad_mul(ad_flush(a),ad_flush(b))); unsigned am=ad_mag(a),bm=ad_mag(b);
    bool narrow=(ad_mag(o)&0x7f800000u)==0u; ad_count(r,base,narrow); if(!narrow)return;
    bool nonzero=am!=0u&&bm!=0u; ad_count(r,base+1,nonzero);
    if(nonzero){double x=ad_record_decode(r,common,a),y=ad_record_decode(r,common,b);ad_record_pack(r,common,x*y);}
}
__device__ __forceinline__ void ad_record_div(uint32_t *r,int base,int common,float a,float b) {
    unsigned am=ad_mag(a),bm=ad_mag(b),ae=am>>23,be=bm>>23;
    bool tiny=(ae==0u&&am!=0u)||(be==0u&&bm!=0u); ad_count(r,base,tiny);
    if(tiny){double x=ad_record_decode(r,common,a),y=ad_record_decode(r,common,b);ad_record_pack(r,common,x/y);return;}
    float o=ad_flush(ad_div(ad_flush(a),ad_flush(b)));
    bool narrow=am!=0u&&bm!=0u&&ae+126u<=be&&(ad_mag(o)&0x7f800000u)==0u;
    ad_count(r,base+1,narrow);
    if(narrow){double x=ad_record_decode(r,common,a),y=ad_record_decode(r,common,b);ad_record_pack(r,common,x/y);}
}
