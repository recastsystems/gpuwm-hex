// SPDX-License-Identifier: MIT
// Branch counters for the pinned guarded FP32 helpers. Ordinary operations
// on the captured route flush subnormal inputs/results; reproduce that bit
// behaviour before testing each source predicate. The returned value is IEEE
// rounded, like the guarded forward helper.
#pragma once
#include "common.cuh"
__device__ __forceinline__ unsigned ad_mag(float x) {return __float_as_uint(x)&0x7fffffffu;}
__device__ __forceinline__ float ad_flush(float x) {
    unsigned b=__float_as_uint(x); return ((b&0x7f800000u)==0u)?__uint_as_float(b&0x80000000u):x;
}
__device__ __forceinline__ void ad_branch(uint32_t *record, int slot, bool arm) {
    if (!record || slot < 0) return;
    unsigned mask=__activemask(), group=__match_any_sync(mask,(unsigned)arm);
    int lane=threadIdx.x&31;
    if (lane==(__ffs(group)-1)) atomicAdd(record+2*slot+(arm?1:0),__popc(group));
}
#ifndef AD_TRACE_ADD_SUBNORMAL
#define AD_TRACE_ADD_SUBNORMAL -1
#endif
#ifndef AD_TRACE_ADD_CANCEL
#define AD_TRACE_ADD_CANCEL -1
#endif
#ifndef AD_TRACE_SUB_SUBNORMAL
#define AD_TRACE_SUB_SUBNORMAL -1
#endif
#ifndef AD_TRACE_SUB_CANCEL
#define AD_TRACE_SUB_CANCEL -1
#endif
#ifndef AD_TRACE_MUL_NONZERO
#define AD_TRACE_MUL_NONZERO -1
#endif
#ifndef AD_TRACE_DIV_UNDERFLOW
#define AD_TRACE_DIV_UNDERFLOW -1
#endif
// Each slot records one source conditional, and early returns preserve the
// source's evaluation order. Slots for absent operations are disabled.
__device__ __forceinline__ float ad_add(float a,float b,uint32_t *r,int slot,bool sub=false) {
    float ordinary=ad_flush(sub ? ad_flush(a)-ad_flush(b) : ad_flush(a)+ad_flush(b));
    unsigned result=ad_mag(ordinary)>>23, ma=ad_mag(a),mb=ad_mag(b),ea=ma>>23,eb=mb>>23;
    bool normal_result=result>24u;
    ad_branch(r,slot,normal_result);
    if (normal_result) return ordinary;
    bool subnormal_input=(ea==0u && ma!=0u)||(eb==0u && mb!=0u);
    ad_branch(r,sub?AD_TRACE_SUB_SUBNORMAL:AD_TRACE_ADD_SUBNORMAL,subnormal_input);
    if (subnormal_input) return (float)(sub ? (double)a-(double)b : (double)a+(double)b);
    bool cancel=ea<=23u && eb<=23u && result==0u && ma!=mb && ma!=0u && mb!=0u;
    ad_branch(r,sub?AD_TRACE_SUB_CANCEL:AD_TRACE_ADD_CANCEL,cancel);
    return cancel ? (float)(sub ? (double)a-(double)b : (double)a+(double)b) : ordinary;
}
__device__ __forceinline__ float ad_mul(float a,float b,uint32_t *r,int slot) {
    float ordinary=ad_flush(ad_flush(a)*ad_flush(b));
    bool envelope=(ad_mag(ordinary)&0x7f800000u)==0u;
    ad_branch(r,slot,envelope);
    if (envelope) {
        bool nonzero=ad_mag(a)!=0u && ad_mag(b)!=0u;
        ad_branch(r,AD_TRACE_MUL_NONZERO,nonzero);
        if (nonzero) return (float)((double)a*(double)b);
    }
    return ordinary;
}
__device__ __forceinline__ float ad_div(float a,float b,uint32_t *r,int slot) {
    unsigned ma=ad_mag(a),mb=ad_mag(b),ea=ma>>23,eb=mb>>23;
    bool subnormal=(ea==0u && ma!=0u)||(eb==0u && mb!=0u);
    ad_branch(r,slot,subnormal);
    if (subnormal) return (float)((double)a/(double)b);
    float ordinary=ad_flush(ad_flush(a)/ad_flush(b));
    bool underflow=ma!=0u && mb!=0u && ea+126u<=eb && (ad_mag(ordinary)&0x7f800000u)==0u;
    ad_branch(r,AD_TRACE_DIV_UNDERFLOW,underflow);
    return underflow ? (float)((double)a/(double)b) : ordinary;
}
