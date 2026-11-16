#include "glue.member_magnitude.cuh"
// SPDX-License-Identifier: Apache-2.0
// Native arithmetic helpers copied from pinned hex cuda_fp32.py.
#pragma once

#ifndef MPAS_FTZ_FALLBACK_ENABLED
#define MPAS_FTZ_FALLBACK_ENABLED 1
#endif

__device__ __forceinline__ unsigned int mpas_f32_magnitude_bits(
    const float value)
{
    return glue_f32_magnitude_bits(value);
}

__device__ __forceinline__ bool mpas_is_subnormal_f32(const float value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int magnitude = mpas_f32_magnitude_bits(value);
    return magnitude != 0u && (magnitude & 0x7f800000u) == 0u;
#else
    return false;
#endif
}

// A plain float/double cast is one of the measured DAZ mechanisms, so decode
// a subnormal float from its integer mantissa instead.
__device__ __forceinline__ double mpas_f32_to_f64_ieee(const float value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int bits = __float_as_uint(value);
    const unsigned int magnitude = bits & 0x7fffffffu;
    if (magnitude != 0u && (magnitude & 0x7f800000u) == 0u) {
        const double decoded = ldexp(
            (double)(magnitude & 0x007fffffu), -149);
        return (bits & 0x80000000u) != 0u ? -decoded : decoded;
    }
#endif
    return (double)value;
}

// __double2float_rn flushes an FP32-subnormal result on the certified route,
// so pack that narrow result from the rounded integer mantissa.
__device__ __forceinline__ float mpas_f64_to_f32_ieee(const double value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const double magnitude = fabs(value);
    if (magnitude > 0.0 && magnitude < 1.1754943508222875e-38) {
        const double scaled = rint(
            magnitude * 7.1362384635297994e+44);  // 2^149
        const unsigned int mantissa = (unsigned int)scaled;
        const unsigned int sign =
            (__double_as_longlong(value) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(sign | mantissa);
    }
#endif
    return __double2float_rn(value);
}

__device__ __forceinline__ float mpas_add(
    const float left, const float right)
{
    const float ordinary = left + right;
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int result_magnitude = mpas_f32_magnitude_bits(ordinary);
    const unsigned int result_exponent = result_magnitude >> 23;
    // A result at exponent field >=25 cannot have been rounded differently
    // by an FP32 subnormal operand, which is then below half an ulp.
    if (result_exponent > 24u) return ordinary;
    const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
    const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
    const unsigned int left_exponent = left_magnitude >> 23;
    const unsigned int right_exponent = right_magnitude >> 23;
    if ((left_exponent == 0u && left_magnitude != 0u)
            || (right_exponent == 0u && right_magnitude != 0u)) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) + mpas_f32_to_f64_ieee(right));
    }
#endif
#if MPAS_FTZ_FALLBACK_ENABLED
    // Two normal binary32 values can cancel into the subnormal range only
    // while both exponent fields are <= 23; above that, one ulp is already
    // at least the smallest normal value.
    if (left_exponent <= 23u && right_exponent <= 23u
            && result_exponent == 0u
            && left_magnitude != right_magnitude
            && left_magnitude != 0u && right_magnitude != 0u) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) + mpas_f32_to_f64_ieee(right));
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_sub(
    const float left, const float right)
{
    const float ordinary = left - right;
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int result_magnitude = mpas_f32_magnitude_bits(ordinary);
    const unsigned int result_exponent = result_magnitude >> 23;
    if (result_exponent > 24u) return ordinary;
    const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
    const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
    const unsigned int left_exponent = left_magnitude >> 23;
    const unsigned int right_exponent = right_magnitude >> 23;
    if ((left_exponent == 0u && left_magnitude != 0u)
            || (right_exponent == 0u && right_magnitude != 0u)) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) - mpas_f32_to_f64_ieee(right));
    }
#endif
#if MPAS_FTZ_FALLBACK_ENABLED
    if (left_exponent <= 23u && right_exponent <= 23u
            && result_exponent == 0u
            && left_magnitude != right_magnitude
            && left_magnitude != 0u && right_magnitude != 0u) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) - mpas_f32_to_f64_ieee(right));
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_mul(
    const float left, const float right)
{
    const float ordinary = left * right;
#if MPAS_FTZ_FALLBACK_ENABLED
    // A normal finite/overflow result proves neither operand was lost to DAZ
    // and no FTZ result occurred.  Inspect operands only on the rare zero or
    // subnormal-result envelope.
    if ((mpas_f32_magnitude_bits(ordinary) & 0x7f800000u) == 0u) {
        const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
        const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
        if (left_magnitude != 0u && right_magnitude != 0u) {
            return mpas_f64_to_f32_ieee(
                mpas_f32_to_f64_ieee(left) * mpas_f32_to_f64_ieee(right));
        }
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_div(
    const float left, const float right)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    const unsigned int left_magnitude = mpas_f32_magnitude_bits(left);
    const unsigned int right_magnitude = mpas_f32_magnitude_bits(right);
    const unsigned int left_exponent = left_magnitude >> 23;
    const unsigned int right_exponent = right_magnitude >> 23;
    if ((left_exponent == 0u && left_magnitude != 0u)
            || (right_exponent == 0u && right_magnitude != 0u)) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) / mpas_f32_to_f64_ieee(right));
    }
#endif
    const float ordinary = left / right;
#if MPAS_FTZ_FALLBACK_ENABLED
    if (left_magnitude != 0u && right_magnitude != 0u
            && left_exponent + 126u <= right_exponent
            && (mpas_f32_magnitude_bits(ordinary) & 0x7f800000u) == 0u) {
        return mpas_f64_to_f32_ieee(
            mpas_f32_to_f64_ieee(left) / mpas_f32_to_f64_ieee(right));
    }
#endif
    return ordinary;
}

__device__ __forceinline__ float mpas_sqrt(const float value)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    if (mpas_is_subnormal_f32(value)) {
        return mpas_f64_to_f32_ieee(sqrt(mpas_f32_to_f64_ieee(value)));
    }
#endif
    return sqrtf(value);
}

__device__ __forceinline__ float mpas_abs(const float value)
{
    return __uint_as_float(__float_as_uint(value) & 0x7fffffffu);
}

__device__ __forceinline__ float mpas_copysign(
    const float magnitude, const float sign)
{
    return __uint_as_float(
        (__float_as_uint(magnitude) & 0x7fffffffu)
        | (__float_as_uint(sign) & 0x80000000u));
}

__device__ __forceinline__ float mpas_min(
    const float left, const float right)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    if (mpas_is_subnormal_f32(left) || mpas_is_subnormal_f32(right)) {
        return mpas_f32_to_f64_ieee(right) < mpas_f32_to_f64_ieee(left)
            ? right : left;
    }
#endif
    return fminf(left, right);
}

__device__ __forceinline__ float mpas_max(
    const float left, const float right)
{
#if MPAS_FTZ_FALLBACK_ENABLED
    if (mpas_is_subnormal_f32(left) || mpas_is_subnormal_f32(right)) {
        return mpas_f32_to_f64_ieee(right) > mpas_f32_to_f64_ieee(left)
            ? right : left;
    }
#endif
    return fmaxf(left, right);
}

// One deterministic integer flag per distinct source, as required by ad.entry/v1.
// The host constructs the detailed AdFailureRecord after reading these slots.
__device__ __forceinline__ void ad_fail(int *slots, unsigned long long field, unsigned long long index, unsigned long long source) {
    (void)field; (void)index;
    if (slots) atomicExch(&slots[source - 1], 1);
}

// Branch counts are integer atomics and do not change the ordinary forward.
// All active lanes at this program point have the same slot and record pointer.
__device__ __forceinline__ bool ad_branch(unsigned int *record, int slot, bool choice) {
    unsigned int lanes = __activemask();
    unsigned int yes = __ballot_sync(lanes, choice);
    unsigned int no = lanes & ~yes;
    unsigned int lane = threadIdx.x & 31u;
    if (yes && lane == (unsigned int)(__ffs(yes) - 1)) atomicAdd(&record[2*slot+1], __popc(yes));
    if (no && lane == (unsigned int)(__ffs(no) - 1)) atomicAdd(&record[2*slot], __popc(no));
    return choice;
}
// Slots relative to base follow source order: f32->f64 subnormal/sign,
// f64->f32 subnormal/sign, add early/input/cancellation, multiply envelope/
// operands, and (if used) divide input/envelope. They count evaluations, not
// merely the number of cells reaching a helper.
__device__ __forceinline__ double ad_br_to64(float x,unsigned int *r,int b) {
    unsigned int bits=__float_as_uint(x), m=bits&0x7fffffffu;
    if(ad_branch(r,b,m!=0u && (m&0x7f800000u)==0u)) {
        double d=ldexp((double)(m&0x007fffffu),-149);
        return ad_branch(r,b+1,(bits&0x80000000u)!=0u) ? -d : d;
    }
    return (double)x;
}
__device__ __forceinline__ float ad_br_to32(double x,unsigned int *r,int b) {
    double m=fabs(x);
    if(ad_branch(r,b+2,m>0.0 && m<1.1754943508222875e-38)) {
        unsigned int mant=(unsigned int)rint(m*7.1362384635297994e+44);
        unsigned int sign=ad_branch(r,b+3,__double_as_longlong(x)<0LL) ? 0x80000000u:0u;
        return __uint_as_float(sign|mant);
    }
    return __double2float_rn(x);
}
__device__ __forceinline__ float ad_br_add(float x,float y,unsigned int *r,int b) {
    float z=x+y; unsigned int rm=mpas_f32_magnitude_bits(z), re=rm>>23;
    if(ad_branch(r,b+4,re>24u)) return z;
    unsigned int xm=mpas_f32_magnitude_bits(x),ym=mpas_f32_magnitude_bits(y),xe=xm>>23,ye=ym>>23;
    if(ad_branch(r,b+5,(xe==0u && xm!=0u)||(ye==0u && ym!=0u)))
        return ad_br_to32(ad_br_to64(x,r,b)+ad_br_to64(y,r,b),r,b);
    if(ad_branch(r,b+6,xe<=23u&&ye<=23u&&re==0u&&xm!=ym&&xm!=0u&&ym!=0u))
        return ad_br_to32(ad_br_to64(x,r,b)+ad_br_to64(y,r,b),r,b);
    return z;
}
__device__ __forceinline__ float ad_br_mul(float x,float y,unsigned int *r,int b) {
    float z=x*y;
    if(ad_branch(r,b+7,(mpas_f32_magnitude_bits(z)&0x7f800000u)==0u)) {
        unsigned int xm=mpas_f32_magnitude_bits(x),ym=mpas_f32_magnitude_bits(y);
        if(ad_branch(r,b+8,xm!=0u&&ym!=0u))
            return ad_br_to32(ad_br_to64(x,r,b)*ad_br_to64(y,r,b),r,b);
    }
    return z;
}
__device__ __forceinline__ float ad_br_div(float x,float y,unsigned int *r,int b) {
    unsigned int xm=mpas_f32_magnitude_bits(x),ym=mpas_f32_magnitude_bits(y),xe=xm>>23,ye=ym>>23;
    if(ad_branch(r,b+9,(xe==0u&&xm!=0u)||(ye==0u&&ym!=0u)))
        return ad_br_to32(ad_br_to64(x,r,b)/ad_br_to64(y,r,b),r,b);
    float z=x/y;
    if(ad_branch(r,b+10,xm!=0u&&ym!=0u&&xe+126u<=ye&&(mpas_f32_magnitude_bits(z)&0x7f800000u)==0u))
        return ad_br_to32(ad_br_to64(x,r,b)/ad_br_to64(y,r,b),r,b);
    return z;
}
