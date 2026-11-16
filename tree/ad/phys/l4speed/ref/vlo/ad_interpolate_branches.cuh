// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ad_mpas_f32.cuh"

// Re-execute the pinned forward helper predicates and operations only for
// branch recording. The counters are integer and order-independent.
__device__ __forceinline__ void ad_ib_record(unsigned int *r, int slot, bool arm) {
    atomicAdd(&r[2 * slot + (arm ? 1 : 0)], 1u);
}
__device__ __forceinline__ double ad_ib_decode(float v, unsigned int *r) {
    unsigned bits=__float_as_uint(v), mag=bits & 0x7fffffffu;
    bool sub=mag!=0u && (mag&0x7f800000u)==0u;
    ad_ib_record(r,1,sub);
    if (sub) ad_ib_record(r,11,(bits & 0x80000000u)!=0u);
    return mpas_f32_to_f64_ieee(v);
}
__device__ __forceinline__ float ad_ib_pack(double v, unsigned int *r) {
    double mag=fabs(v); bool sub=mag>0.0 && mag<1.1754943508222875e-38;
    ad_ib_record(r,2,sub);
    if (sub) ad_ib_record(r,12,__double_as_longlong(v)<0LL);
    return mpas_f64_to_f32_ieee(v);
}
__device__ __forceinline__ float ad_ib_addsub(float left, float right, bool minus, unsigned int *r) {
    float ordinary=minus ? left-right : left+right;
    unsigned re=mpas_f32_magnitude_bits(ordinary)>>23;
    int slot=minus ? 6 : 3;
    bool fast=re>24u; ad_ib_record(r,slot,fast); if (fast) return ordinary;
    unsigned lm=mpas_f32_magnitude_bits(left), rm=mpas_f32_magnitude_bits(right);
    unsigned le=lm>>23, rexp=rm>>23;
    bool sub=(le==0u && lm!=0u)||(rexp==0u && rm!=0u);
    ad_ib_record(r,slot+1,sub);
    if (sub) {
        double l=ad_ib_decode(left,r), q=ad_ib_decode(right,r);
        return ad_ib_pack(minus ? l-q : l+q,r);
    }
    bool cancel=le<=23u && rexp<=23u && re==0u && lm!=rm && lm!=0u && rm!=0u;
    ad_ib_record(r,slot+2,cancel);
    if (cancel) {
        double l=ad_ib_decode(left,r), q=ad_ib_decode(right,r);
        return ad_ib_pack(minus ? l-q : l+q,r);
    }
    return ordinary;
}
__device__ __forceinline__ float ad_ib_mul(float left, float right, unsigned int *r) {
    float ordinary=left*right;
    bool small=(mpas_f32_magnitude_bits(ordinary)&0x7f800000u)==0u;
    ad_ib_record(r,9,small);
    if (small) {
        unsigned lm=mpas_f32_magnitude_bits(left), rm=mpas_f32_magnitude_bits(right);
        bool nonzero=lm!=0u && rm!=0u; ad_ib_record(r,10,nonzero);
        if (nonzero) {
            double l=ad_ib_decode(left,r), q=ad_ib_decode(right,r);
            return ad_ib_pack(l*q,r);
        }
    }
    return ordinary;
}
__device__ __forceinline__ void ad_ib_interpolate(float weight, float old_rho, float new_rho, unsigned int *r) {
    float complement=ad_ib_addsub(1.0f,weight,true,r);
    float left=ad_ib_mul(complement,old_rho,r);
    float right=ad_ib_mul(weight,new_rho,r);
    (void)ad_ib_addsub(left,right,false,r);
}
