// SPDX-License-Identifier: Apache-2.0
// Counts the actual guarded helper predicates in cuda_fp32.py for an
// independent replay of the unchanged forward's arithmetic.  The ordinary
// operations explicitly use the forward's FTZ setting; fallback conversions
// reproduce its IEEE subnormal handling. The caller combines per-block counts.
#ifndef HEX_AD_BRANCH_OPS_CUH
#define HEX_AD_BRANCH_OPS_CUH
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int int32_t;
struct AdBranchRecorder {
    unsigned int* counts;
    __device__ __forceinline__ void count(int slot, bool arm) const {
        if (slot >= 0) atomicAdd(counts + 2*slot + (arm ? 1 : 0), 1u);
    }
    __device__ __forceinline__ unsigned mag(float x) const {return __float_as_uint(x)&0x7fffffffu;}
    __device__ __forceinline__ double decode(float x) const {
        unsigned bits=__float_as_uint(x), m=bits&0x7fffffffu;
        bool sub=m!=0u && (m&0x7f800000u)==0u;
        count(AD_BR_DECODE_SUBNORMAL,sub);
        if(sub){bool sign=(bits&0x80000000u)!=0u;count(AD_BR_DECODE_SIGN,sign);
            double d=ldexp((double)(m&0x007fffffu),-149);return sign?-d:d;}
        return (double)x;
    }
    __device__ __forceinline__ float encode(double x) const {
        double m=fabs(x);bool sub=m>0.0 && m<1.1754943508222875e-38;
        count(AD_BR_ENCODE_SUBNORMAL,sub);
        if(sub){unsigned mantissa=(unsigned)rint(m*7.1362384635297994e+44);
            bool sign=__double_as_longlong(x)<0LL;count(AD_BR_ENCODE_SIGN,sign);
            return __uint_as_float((sign?0x80000000u:0u)|mantissa);}
        return __double2float_rn(x);
    }
    __device__ __forceinline__ float add(float l,float r) const {
        float v;asm("add.rn.ftz.f32 %0,%1,%2;":"=f"(v):"f"(l),"f"(r));
        unsigned re=mag(v)>>23;bool fast=re>24u;count(AD_BR_ADD_FAST,fast);if(fast)return v;
        unsigned lm=mag(l),rm=mag(r),le=lm>>23,rexp=rm>>23;
        bool sub=(le==0u&&lm!=0u)||(rexp==0u&&rm!=0u);count(AD_BR_ADD_SUBNORMAL,sub);
        if(sub)return encode(decode(l)+decode(r));
        bool cancel=le<=23u&&rexp<=23u&&re==0u&&lm!=rm&&lm!=0u&&rm!=0u;
        count(AD_BR_ADD_CANCEL,cancel);if(cancel)return encode(decode(l)+decode(r));return v;
    }
    __device__ __forceinline__ float sub(float l,float r) const {
        float v;asm("sub.rn.ftz.f32 %0,%1,%2;":"=f"(v):"f"(l),"f"(r));
        unsigned re=mag(v)>>23;bool fast=re>24u;count(AD_BR_SUB_FAST,fast);if(fast)return v;
        unsigned lm=mag(l),rm=mag(r),le=lm>>23,rexp=rm>>23;
        bool subnormal=(le==0u&&lm!=0u)||(rexp==0u&&rm!=0u);count(AD_BR_SUB_SUBNORMAL,subnormal);
        if(subnormal)return encode(decode(l)-decode(r));
        bool cancel=le<=23u&&rexp<=23u&&re==0u&&lm!=rm&&lm!=0u&&rm!=0u;
        count(AD_BR_SUB_CANCEL,cancel);if(cancel)return encode(decode(l)-decode(r));return v;
    }
    __device__ __forceinline__ float mul(float l,float r) const {
        float v;asm("mul.rn.ftz.f32 %0,%1,%2;":"=f"(v):"f"(l),"f"(r));
        bool envelope=(mag(v)&0x7f800000u)==0u;count(AD_BR_MUL_ENVELOPE,envelope);
        if(envelope){bool nonzero=mag(l)!=0u&&mag(r)!=0u;count(AD_BR_MUL_NONZERO,nonzero);
            if(nonzero)return encode(decode(l)*decode(r));}return v;
    }
    __device__ __forceinline__ float div(float l,float r) const {
        unsigned lm=mag(l),rm=mag(r),le=lm>>23,re=rm>>23;
        bool subnormal=(le==0u&&lm!=0u)||(re==0u&&rm!=0u);count(AD_BR_DIV_SUBNORMAL,subnormal);
        if(subnormal)return encode(decode(l)/decode(r));
        float v;asm("div.rn.ftz.f32 %0,%1,%2;":"=f"(v):"f"(l),"f"(r));
        bool envelope=lm!=0u&&rm!=0u&&le+126u<=re&&(mag(v)&0x7f800000u)==0u;
        count(AD_BR_DIV_ENVELOPE,envelope);if(envelope)return encode(decode(l)/decode(r));return v;
    }
};
#endif
