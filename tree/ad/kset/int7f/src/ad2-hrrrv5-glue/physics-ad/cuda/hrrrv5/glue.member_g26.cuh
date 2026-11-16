#include "glue.member_magnitude.cuh"
// SPDX-License-Identifier: Apache-2.0
// Deterministic counts for the pinned MPAS guarded-f32 helper predicates.
// The ordinary forward is unchanged. These helpers recompute its primal
// arithmetic solely to record the helper arms from the derivative entry.
#pragma once
struct PrepBranchRecorder {
    unsigned int *record;
    int base;
    mutable unsigned int totals[52];
    __device__ __forceinline__ PrepBranchRecorder(unsigned int *out, int first)
        : record(out), base(first), totals{} {}
    __device__ __forceinline__ void count(int slot, bool arm) const {
        ++totals[2 * slot + (arm ? 1 : 0)];
    }
    __device__ __forceinline__ void count_n(int slot, bool arm, unsigned n) const {
        totals[2 * slot + (arm ? 1 : 0)] += n;
    }
    __device__ __forceinline__ void flush(int slots) const {
        // One integer update per active warp and counter.  The active mask
        // also covers padded lanes and the wind kernel's early-return arms.
        const unsigned mask = __activemask();
        const int leader = __ffs(mask) - 1;
        const int lane = threadIdx.x & 31;
        for (int i = 0; i < 2 * slots; ++i) {
            const unsigned total = __reduce_add_sync(mask, totals[i]);
            if (lane == leader && total) atomicAdd(record + i, total);
        }
    }
    __device__ __forceinline__ unsigned mag(float x) const {
        return glue_f32_magnitude_bits(x);
    }
    // Match the forward image's FTZ/DAZ arithmetic while compiling the
    // derivative with --ftz=false.  Preserve the sign of flushed zero.
    __device__ __forceinline__ float ftz(float x) const {
        unsigned bits = __float_as_uint(x);
        return (bits & 0x7f800000u) == 0u
            ? __uint_as_float(bits & 0x80000000u) : x;
    }
    __device__ __forceinline__ double decode(float x) const {
        unsigned bits=__float_as_uint(x), m=bits&0x7fffffffu;
        bool sub=m!=0u && (m&0x7f800000u)==0u; count(base+0,sub);
        if (sub) {
            double decoded=ldexp((double)(m&0x007fffffu),-149);
            bool neg=(bits&0x80000000u)!=0u; count(base+12,neg);
            return neg ? -decoded : decoded;
        }
        return (double)x;
    }
    __device__ __forceinline__ float pack(double x) const {
        double a=fabs(x); bool sub=a>0.0 && a<1.1754943508222875e-38;
        count(base+1,sub);
        if (sub) {
            unsigned m=(unsigned)rint(a*7.1362384635297994e+44);
            bool neg=__double_as_longlong(x)<0LL; count(base+13,neg);
            return __uint_as_float((neg?0x80000000u:0u)|m);
        }
        return __double2float_rn(x);
    }
    __device__ __forceinline__ float add(float a,float b) const {
        float r = ftz(__fadd_rn(ftz(a), ftz(b)));
        unsigned re=mag(r)>>23; bool fast=re>24u; count(base+2,fast);
        if (fast) return r;
        unsigned am=mag(a),bm=mag(b),ae=am>>23,be=bm>>23;
        bool sub=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);count(base+3,sub);
        if (sub) return pack(decode(a)+decode(b));
        bool cancel=ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u;
        count(base+4,cancel); if (cancel) return pack(decode(a)+decode(b));
        return r;
    }
    __device__ __forceinline__ float sub(float a,float b) const {
        float r = ftz(__fsub_rn(ftz(a), ftz(b)));
        unsigned re=mag(r)>>23; bool fast=re>24u; count(base+5,fast);
        if (fast) return r;
        unsigned am=mag(a),bm=mag(b),ae=am>>23,be=bm>>23;
        bool subn=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);count(base+6,subn);
        if (subn) return pack(decode(a)-decode(b));
        bool cancel=ae<=23u&&be<=23u&&re==0u&&am!=bm&&am!=0u&&bm!=0u;
        count(base+7,cancel); if (cancel) return pack(decode(a)-decode(b));
        return r;
    }
    __device__ __forceinline__ float mul(float a,float b) const {
        float r = ftz(__fmul_rn(ftz(a), ftz(b)));
        bool env=(mag(r)&0x7f800000u)==0u; count(base+8,env);
        if (env) { bool nz=mag(a)!=0u&&mag(b)!=0u; count(base+9,nz);
            if (nz) return pack(decode(a)*decode(b)); }
        return r;
    }
    __device__ __forceinline__ float div(float a,float b) const {
        unsigned am=mag(a),bm=mag(b),ae=am>>23,be=bm>>23;
        bool subn=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);count(base+10,subn);
        if (subn) return pack(decode(a)/decode(b));
        float r = ftz(__fdiv_rn(ftz(a), ftz(b)));
        bool env=am!=0u&&bm!=0u&&ae+126u<=be&&(mag(r)&0x7f800000u)==0u;
        count(base+11,env); if (env) return pack(decode(a)/decode(b));
        return r;
    }
};
