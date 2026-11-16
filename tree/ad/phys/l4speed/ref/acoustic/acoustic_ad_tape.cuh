// SPDX-License-Identifier: Apache-2.0
// Small, deterministic CUDA-local operation tape for the hand-transcribed
// acoustic sweeps.  Storage is caller-owned and reused after each output
// element or column; it is not checkpoint state and does not change F.
#pragma once
#include "acoustic_branch_record.cuh"

// Section 35 device contract: one shared uint32 slot per source_id.
// Concurrent reports use integer atomics; any nonzero slot refuses the call.
typedef unsigned AcousticFailure;
__device__ __forceinline__ void acoustic_fail(AcousticFailure *f, unsigned long long field,
                                              unsigned long long index, unsigned long long source) {
    (void)field; (void)index;
    atomicExch(f + (source - 1), 1u);
}
struct AcousticField { const float *x; float *d; unsigned long long id; const float *seed; double *accum; };
struct AcousticDual { float v, d; };
struct AcousticTangent {
    typedef AcousticDual V;
    typedef float S;
    AcousticFailure *failure; unsigned *branches; int branch_base;
    __device__ void record(int slot,bool arm) {abr_count(branches,slot,arm);}
    __device__ V constant(float x) { return {x,0.0f}; }
    __device__ V input(AcousticField a,int i) {
        float x=a.x[i],d=a.d[i];
        if(!isfinite(x)||!isfinite(d)) acoustic_fail(failure,a.id,i,1);
        return {x,d};
    }
    __device__ V scalar(float x,float *p,unsigned long long id) {
        float d=*p; if(!isfinite(x)||!isfinite(d)) acoustic_fail(failure,id,0,1);
        return {x,d};
    }
    __device__ V add(V a,V b) { return checked({abr_add(a.v,b.v,branches,branch_base),mpas_add(a.d,b.d)}); }
    __device__ V sub(V a,V b) { return checked({abr_sub(a.v,b.v,branches,branch_base),mpas_sub(a.d,b.d)}); }
    __device__ V mul(V a,V b) { return checked({abr_mul(a.v,b.v,branches,branch_base),mpas_add(mpas_mul(a.d,b.v),mpas_mul(a.v,b.d))}); }
    __device__ V div(V a,V b) {
        if(mpas_f32_magnitude_bits(b.v)==0u) acoustic_fail(failure,0,0,2);
        float v=abr_div(a.v,b.v,branches,branch_base);
        return checked({v,mpas_div(mpas_sub(a.d,mpas_mul(v,b.d)),b.v)});
    }
    __device__ V checked(V a) { if(!isfinite(a.v)||!isfinite(a.d)) acoustic_fail(failure,0,0,3);return a; }
    __device__ void output(AcousticField a,int i,V v) { a.d[i]=v.d; }
    __device__ void finish() {}
};
// The bucket lives in every tape slot so repeated reads of one device value
// share one node. Its cotangent is accumulated in f64 and written once.
struct AcousticNode { double bar; float left_weight, right_weight; int left,right; unsigned long long destination; int bucket; };
static_assert(sizeof(AcousticNode)==40,"AcousticNode ABI");
struct AcousticValue { float v; int id; };
struct AcousticReverse {
    typedef AcousticValue V;
    typedef double S;
    AcousticNode *nodes; int count,capacity; AcousticFailure *failure;
    __device__ void reset() { for(int i=0;i<capacity;++i) nodes[i].bucket=-1; }
    __device__ void record(int,bool) {}
    __device__ V constant(float x) { return {x,-1}; }
    __device__ V push(float x,int l,int r,float dl,float dr,unsigned long long dst) {
        if(!isfinite(x)||!isfinite(dl)||!isfinite(dr)) acoustic_fail(failure,0,0,3);
        if(count>=capacity) { acoustic_fail(failure,0,count,4);return {x,-1}; }
        int id=count++;AcousticNode &n=nodes[id];
        n.bar=0.0;n.left_weight=dl;n.right_weight=dr;n.left=l;n.right=r;n.destination=dst;
        return {x,id};
    }
    __device__ V leaf(float x,unsigned long long dst) {
        unsigned long long h=(dst>>3)^(dst>>19)^(dst>>37);
        int bucket=(int)(h%(unsigned long long)capacity);
        for(int i=nodes[bucket].bucket;i>=0;i=nodes[i].left)
            if(nodes[i].destination==dst) return {x,i};
        V v=push(x,nodes[bucket].bucket,-1,0.0f,0.0f,dst);
        if(v.id>=0)nodes[bucket].bucket=v.id;
        return v;
    }
    __device__ V input(AcousticField a,int i) {
        double d=a.accum?a.accum[i]:mpas_f32_to_f64_ieee(a.d[i]);
        if(!isfinite(a.x[i])||!isfinite(d)) acoustic_fail(failure,a.id,i,1);
        unsigned long long dst=a.accum?(((unsigned long long)(a.accum+i))|1ULL):(unsigned long long)(a.d+i);
        return leaf(a.x[i],dst);
    }
    __device__ V scalar(float x,double *d,unsigned long long id) {
        if(!isfinite(x)||!isfinite(*d)) acoustic_fail(failure,id,0,1);
        return leaf(x,((unsigned long long)d)|1ULL);
    }
    __device__ V add(V a,V b) { return push(mpas_add(a.v,b.v),a.id,b.id,1.0f,1.0f,0ULL); }
    __device__ V sub(V a,V b) { return push(mpas_sub(a.v,b.v),a.id,b.id,1.0f,-1.0f,0ULL); }
    __device__ V mul(V a,V b) { return push(mpas_mul(a.v,b.v),a.id,b.id,b.v,a.v,0ULL); }
    __device__ V div(V a,V b) {
        if(mpas_f32_magnitude_bits(b.v)==0u) acoustic_fail(failure,0,0,2);
        const float v=mpas_div(a.v,b.v);
        return push(v,a.id,b.id,mpas_div(1.0f,b.v),mpas_sub(0.0f,mpas_div(v,b.v)),0ULL);
    }
    __device__ void output(AcousticField a,int i,V v) {
        float seed=a.seed ? a.seed[i] : a.d[i];if(!isfinite(seed)) acoustic_fail(failure,a.id,i,1);
        if(v.id>=0) nodes[v.id].bar+=mpas_f32_to_f64_ieee(seed);
        if(!a.seed) a.d[i]=0.0f;
    }
    __device__ void finish() {
        for(int i=count-1;i>=0;--i) {
            AcousticNode &n=nodes[i];
            if(n.destination) {
                if(n.destination&1ULL) {
                    double *dst=(double *)(n.destination&~1ULL);double z=*dst+n.bar;
                    if(!isfinite(z)) acoustic_fail(failure,0,i,3);*dst=z;
                } else {
                    float *dst=(float *)n.destination;float z=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(*dst)+n.bar);
                    if(!isfinite(z)) acoustic_fail(failure,0,i,3);*dst=z;
                }
            } else {
                if(n.left>=0) nodes[n.left].bar+=n.bar*mpas_f32_to_f64_ieee(n.left_weight);
                if(n.right>=0) nodes[n.right].bar+=n.bar*mpas_f32_to_f64_ieee(n.right_weight);
            }
        }
    }
};
