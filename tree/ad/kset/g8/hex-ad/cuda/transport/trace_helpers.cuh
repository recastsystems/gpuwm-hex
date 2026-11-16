// SPDX-License-Identifier: Apache-2.0
// Branch records for the original MPAS floating-point helpers. The native
// helpers still compute each result; these functions mirror their predicates.
#ifndef HEX_AD_TRACE_HELPERS
#define HEX_AD_TRACE_HELPERS
#ifndef HEX_AD_TRACE_BASE
#error HEX_AD_TRACE_BASE must equal the first helper branch slot in the spec
#endif
// The strict certified CuPy route executes the helper's ordinary FP32
// operations with FTZ enabled. Derivative arithmetic can use FTZ disabled;
// use the forward mode just for the replayed branch predicates.
#ifndef HEX_AD_FORWARD_FTZ
#define HEX_AD_FORWARD_FTZ 1
#endif
// Reproduce an FTZ operation in CUDA C++ while compiling derivative math with
// FTZ disabled.  Keep the sign of a flushed operand or result, including -0.
__device__ __forceinline__ float ad_flush_f32(float x){
 unsigned int bits=__float_as_uint(x);
 return (bits&0x7fffffffu)<0x00800000u?__uint_as_float(bits&0x80000000u):x;
}
__device__ __forceinline__ float ad_ftz_add(float a,float b){return ad_flush_f32(__fadd_rn(ad_flush_f32(a),ad_flush_f32(b)));}
__device__ __forceinline__ float ad_ftz_sub(float a,float b){return ad_flush_f32(__fsub_rn(ad_flush_f32(a),ad_flush_f32(b)));}
__device__ __forceinline__ float ad_ftz_mul(float a,float b){return ad_flush_f32(__fmul_rn(ad_flush_f32(a),ad_flush_f32(b)));}
__device__ __forceinline__ float ad_ftz_div(float a,float b){return ad_flush_f32(__fdiv_rn(ad_flush_f32(a),ad_flush_f32(b)));}
__device__ __forceinline__ float ad_original_add(float a,float b){
#if HEX_AD_FORWARD_FTZ
 return ad_ftz_add(a,b);
#else
 return __fadd_rn(a,b);
#endif
}
__device__ __forceinline__ float ad_original_sub(float a,float b){
#if HEX_AD_FORWARD_FTZ
 return ad_ftz_sub(a,b);
#else
 return __fsub_rn(a,b);
#endif
}
__device__ __forceinline__ float ad_original_mul(float a,float b){
#if HEX_AD_FORWARD_FTZ
 return ad_ftz_mul(a,b);
#else
 return __fmul_rn(a,b);
#endif
}
__device__ __forceinline__ float ad_original_div(float a,float b){
#if HEX_AD_FORWARD_FTZ
 return ad_ftz_div(a,b);
#else
 return __fdiv_rn(a,b);
#endif
}
__device__ __forceinline__ double ad_trace_decode(float x,unsigned int *r){
 unsigned bits=__float_as_uint(x),mag=bits&0x7fffffffu;bool sub=mag!=0u&&(mag&0x7f800000u)==0u;ad_branch(r,HEX_AD_TRACE_BASE+0,sub);
 if(sub)ad_branch(r,HEX_AD_TRACE_BASE+1,(bits&0x80000000u)!=0u);return mpas_f32_to_f64_ieee(x);
}
__device__ __forceinline__ void ad_trace_pack(double x,unsigned int *r){
 double mag=fabs(x);bool sub=mag>0.0&&mag<1.1754943508222875e-38;ad_branch(r,HEX_AD_TRACE_BASE+2,sub);
 if(sub)ad_branch(r,HEX_AD_TRACE_BASE+3,__double_as_longlong(x)<0LL);
}
__device__ __forceinline__ float ad_trace_add(float a,float b,unsigned int *r){
 if(!r)return mpas_add(a,b);
#if MPAS_FTZ_FALLBACK_ENABLED
 float o=ad_original_add(a,b);unsigned re=mpas_f32_magnitude_bits(o)>>23;bool fast=re>24u;ad_branch(r,HEX_AD_TRACE_BASE+4,fast);
 if(!fast){unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b),ae=am>>23,be=bm>>23;bool tiny=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);ad_branch(r,HEX_AD_TRACE_BASE+5,tiny);
  if(tiny){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x+y,r);}else{bool cancel=ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u;ad_branch(r,HEX_AD_TRACE_BASE+6,cancel);if(cancel){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x+y,r);}}
 }
#endif
 return mpas_add(a,b);
}
__device__ __forceinline__ float ad_trace_sub(float a,float b,unsigned int *r){
 if(!r)return mpas_sub(a,b);
#if MPAS_FTZ_FALLBACK_ENABLED
 float o=ad_original_sub(a,b);unsigned re=mpas_f32_magnitude_bits(o)>>23;bool fast=re>24u;ad_branch(r,HEX_AD_TRACE_BASE+7,fast);
 if(!fast){unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b),ae=am>>23,be=bm>>23;bool tiny=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);ad_branch(r,HEX_AD_TRACE_BASE+8,tiny);
  if(tiny){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x-y,r);}else{bool cancel=ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u;ad_branch(r,HEX_AD_TRACE_BASE+9,cancel);if(cancel){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x-y,r);}}
 }
#endif
 return mpas_sub(a,b);
}
__device__ __forceinline__ float ad_trace_mul(float a,float b,unsigned int *r){
 if(!r)return mpas_mul(a,b);
#if MPAS_FTZ_FALLBACK_ENABLED
 float o=ad_original_mul(a,b);bool envelope=(mpas_f32_magnitude_bits(o)&0x7f800000u)==0u;ad_branch(r,HEX_AD_TRACE_BASE+10,envelope);
 if(envelope){unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b);bool nonzero=am!=0u&&bm!=0u;ad_branch(r,HEX_AD_TRACE_BASE+11,nonzero);if(nonzero){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x*y,r);}}
#endif
 return mpas_mul(a,b);
}
__device__ __forceinline__ float ad_trace_div(float a,float b,unsigned int *r){
 if(!r)return mpas_div(a,b);
#if MPAS_FTZ_FALLBACK_ENABLED
 unsigned am=mpas_f32_magnitude_bits(a),bm=mpas_f32_magnitude_bits(b),ae=am>>23,be=bm>>23;bool tiny=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);ad_branch(r,HEX_AD_TRACE_BASE+12,tiny);
 if(tiny){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x/y,r);}else{float o=ad_original_div(a,b);bool result=am!=0u&&bm!=0u&&ae+126u<=be&&(mpas_f32_magnitude_bits(o)&0x7f800000u)==0u;ad_branch(r,HEX_AD_TRACE_BASE+13,result);if(result){double x=ad_trace_decode(a,r),y=ad_trace_decode(b,r);ad_trace_pack(x/y,r);}}
#endif
 return mpas_div(a,b);
}
__device__ __forceinline__ float ad_trace_min(float a,float b,unsigned int *r){
 if(!r)return mpas_min(a,b);
#if MPAS_FTZ_FALLBACK_ENABLED
 bool tiny=mpas_is_subnormal_f32(a)||mpas_is_subnormal_f32(b);ad_branch(r,HEX_AD_TRACE_BASE+14,tiny);if(tiny){ad_trace_decode(a,r);ad_trace_decode(b,r);}
#endif
 return mpas_min(a,b);
}
__device__ __forceinline__ float ad_trace_max(float a,float b,unsigned int *r){
 if(!r)return mpas_max(a,b);
#if MPAS_FTZ_FALLBACK_ENABLED
 bool tiny=mpas_is_subnormal_f32(a)||mpas_is_subnormal_f32(b);ad_branch(r,HEX_AD_TRACE_BASE+15,tiny);if(tiny){ad_trace_decode(a,r);ad_trace_decode(b,r);}
#endif
 return mpas_max(a,b);
}
#endif
