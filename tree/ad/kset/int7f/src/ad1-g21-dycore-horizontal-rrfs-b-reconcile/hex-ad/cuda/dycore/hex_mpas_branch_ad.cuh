// SPDX-License-Identifier: Apache-2.0
// Trace whether a primal guarded arithmetic helper took its fallback arm.
// These wrappers return the unchanged helper result; the boolean records the
// aggregate predicate for one forward output point, not derivative math.
#pragma once
#include "hex_mpas_fp32_ad.cuh"
__device__ __forceinline__ float ad_trace_add(float a,float b,bool &hit){
#if MPAS_FTZ_FALLBACK_ENABLED
 const float x=a+b;unsigned r=mpas_f32_magnitude_bits(x), re=r>>23;
 if(re<=24u){unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b),ae=am>>23,be=bm>>23;
  hit|=(ae==0u&&am!=0u)||(be==0u&&bm!=0u)||(ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u);}
 #endif
 return mpas_add(a,b);
}
__device__ __forceinline__ float ad_trace_sub(float a,float b,bool &hit){
#if MPAS_FTZ_FALLBACK_ENABLED
 const float x=a-b;unsigned r=mpas_f32_magnitude_bits(x), re=r>>23;
 if(re<=24u){unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b),ae=am>>23,be=bm>>23;
  hit|=(ae==0u&&am!=0u)||(be==0u&&bm!=0u)||(ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u);}
 #endif
 return mpas_sub(a,b);
}
__device__ __forceinline__ float ad_trace_mul(float a,float b,bool &hit){
#if MPAS_FTZ_FALLBACK_ENABLED
 const float x=a*b;unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b);
 hit|=((mpas_f32_magnitude_bits(x)&0x7f800000u)==0u&&am!=0u&&bm!=0u);
 #endif
 return mpas_mul(a,b);
}
__device__ __forceinline__ float ad_trace_sqrt(float a,bool &hit){
#if MPAS_FTZ_FALLBACK_ENABLED
 hit|=mpas_is_subnormal_f32(a);
#endif
 return mpas_sqrt(a);}
__device__ __forceinline__ float ad_trace_min(float a,float b,bool &hit){
#if MPAS_FTZ_FALLBACK_ENABLED
 hit|=mpas_is_subnormal_f32(a)||mpas_is_subnormal_f32(b);
#endif
 return mpas_min(a,b);}
__device__ __forceinline__ void ad_count_branch(unsigned int *r,int slot,bool arm){atomicAdd(&r[2*slot+(arm?1:0)],1u);}
// Each wrapper records exactly one primal call site, including each loop
// execution, and returns the forward helper's unchanged value.
__device__ __forceinline__ float ad_site_add(float a,float b,unsigned *r,int slot){bool hit=false;float v=ad_trace_add(a,b,hit);ad_count_branch(r,slot,hit);return v;}
__device__ __forceinline__ float ad_site_sub(float a,float b,unsigned *r,int slot){bool hit=false;float v=ad_trace_sub(a,b,hit);ad_count_branch(r,slot,hit);return v;}
__device__ __forceinline__ float ad_site_mul(float a,float b,unsigned *r,int slot){bool hit=false;float v=ad_trace_mul(a,b,hit);ad_count_branch(r,slot,hit);return v;}
__device__ __forceinline__ float ad_site_sqrt(float a,unsigned *r,int slot){bool hit=false;float v=ad_trace_sqrt(a,hit);ad_count_branch(r,slot,hit);return v;}
__device__ __forceinline__ float ad_site_min(float a,float b,unsigned *r,int slot){bool hit=false;float v=ad_trace_min(a,b,hit);ad_count_branch(r,slot,hit);return v;}
