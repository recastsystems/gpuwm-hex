// SPDX-License-Identifier: MIT
// Predicate reconstruction for cuda_fp32.py. Compile this entry with the
// captured forward's FTZ setting. Counts are local to a CTA and reduced to
// uint32[slot][arm] by the derivative branch entry.
#pragma once
__device__ __forceinline__ bool hb_arm(unsigned *r, int slot, bool arm) {
    atomicAdd(r + 2 * slot + (arm ? 1 : 0), 1u);
    return arm;
}
// The fast-capture configuration compiles the forward's IEEE fallback out.
// Mirror that compile-time choice so the branch replica also contracts the
// same multiply/add expressions under the captured fast compiler flags.
#ifndef MPAS_FTZ_FALLBACK_ENABLED
#define MPAS_FTZ_FALLBACK_ENABLED 1
#endif
#if MPAS_FTZ_FALLBACK_ENABLED
__device__ __forceinline__ unsigned hb_mag(float x) { return __float_as_uint(x) & 0x7fffffffu; }
__device__ __forceinline__ double hb_decode(unsigned *r, float x) {
    unsigned bits=__float_as_uint(x), m=bits&0x7fffffffu;
    if (hb_arm(r,0,m!=0u && (m&0x7f800000u)==0u)) {
        double d=ldexp((double)(m&0x007fffffu),-149);
        return hb_arm(r,1,(bits&0x80000000u)!=0u) ? -d : d;
    }
    return (double)x;
}
__device__ __forceinline__ float hb_pack(unsigned *r, double x) {
    double m=fabs(x);
    if (hb_arm(r,2,m>0.0 && m<1.1754943508222875e-38)) {
        unsigned mantissa=(unsigned)rint(m*7.1362384635297994e+44);
        unsigned sign=hb_arm(r,3,__double_as_longlong(x)<0LL) ? 0x80000000u : 0u;
        return __uint_as_float(sign|mantissa);
    }
    return __double2float_rn(x);
}
__device__ __forceinline__ float hb_add(unsigned *r,float a,float b) {
    float o=a+b; unsigned re=hb_mag(o)>>23;
    if(hb_arm(r,4,re>24u)) return o;
    unsigned am=hb_mag(a),bm=hb_mag(b),ae=am>>23,be=bm>>23;
    if(hb_arm(r,5,(ae==0u&&am!=0u)||(be==0u&&bm!=0u))) return hb_pack(r,hb_decode(r,a)+hb_decode(r,b));
    if(hb_arm(r,6,ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u)) return hb_pack(r,hb_decode(r,a)+hb_decode(r,b));
    return o;
}
__device__ __forceinline__ float hb_mul(unsigned *r,float a,float b) {
    float o=a*b;
    if(hb_arm(r,7,(hb_mag(o)&0x7f800000u)==0u)) {
        unsigned am=hb_mag(a),bm=hb_mag(b);
        if(hb_arm(r,8,am!=0u&&bm!=0u)) return hb_pack(r,hb_decode(r,a)*hb_decode(r,b));
    }
    return o;
}
__device__ __forceinline__ float hb_sub(unsigned *r,float a,float b) {
    float o=a-b; unsigned re=hb_mag(o)>>23;
    if(hb_arm(r,9,re>24u)) return o;
    unsigned am=hb_mag(a),bm=hb_mag(b),ae=am>>23,be=bm>>23;
    if(hb_arm(r,10,(ae==0u&&am!=0u)||(be==0u&&bm!=0u))) return hb_pack(r,hb_decode(r,a)-hb_decode(r,b));
    if(hb_arm(r,11,ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u)) return hb_pack(r,hb_decode(r,a)-hb_decode(r,b));
    return o;
}
__device__ __forceinline__ float hb_div(unsigned *r,float a,float b) {
    unsigned am=hb_mag(a),bm=hb_mag(b),ae=am>>23,be=bm>>23;
    if(hb_arm(r,12,(ae==0u&&am!=0u)||(be==0u&&bm!=0u))) return hb_pack(r,hb_decode(r,a)/hb_decode(r,b));
    float o=a/b;
    if(hb_arm(r,13,am!=0u&&bm!=0u&&ae+126u<=be&&(hb_mag(o)&0x7f800000u)==0u)) return hb_pack(r,hb_decode(r,a)/hb_decode(r,b));
    return o;
}

#else
// In fast mode cuda_fp32.py has no helper predicates at these sites.
__device__ __forceinline__ float hb_add(unsigned *,float a,float b) { return a+b; }
__device__ __forceinline__ float hb_mul(unsigned *,float a,float b) { return a*b; }
__device__ __forceinline__ float hb_sub(unsigned *,float a,float b) { return a-b; }
__device__ __forceinline__ float hb_div(unsigned *,float a,float b) { return a/b; }
#endif
