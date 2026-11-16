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
#ifdef ACOUSTIC_DIAGNOSTIC_F64_SHADOW
typedef double AcousticShadow;
typedef double AcousticPrimal;
__device__ __forceinline__ double acoustic_shadow_decode(double x) { return x; }
__device__ __forceinline__ double acoustic_shadow_encode(double x) { return x; }
__device__ __forceinline__ double acoustic_primal_decode(double x) { return x; }
__device__ __forceinline__ double acoustic_input(float x) { return mpas_f32_to_f64_ieee(x); }
__device__ __forceinline__ double acoustic_add(double a,double b,unsigned *,int) { return a+b; }
__device__ __forceinline__ double acoustic_sub(double a,double b,unsigned *,int) { return a-b; }
__device__ __forceinline__ double acoustic_mul(double a,double b,unsigned *,int) { return a*b; }
__device__ __forceinline__ double acoustic_div(double a,double b,unsigned *,int) { return a/b; }
#else
typedef float AcousticShadow;
typedef float AcousticPrimal;
__device__ __forceinline__ double acoustic_shadow_decode(float x) { return mpas_f32_to_f64_ieee(x); }
__device__ __forceinline__ float acoustic_shadow_encode(double x) { return mpas_f64_to_f32_ieee(x); }
__device__ __forceinline__ double acoustic_primal_decode(float x) { return mpas_f32_to_f64_ieee(x); }
__device__ __forceinline__ float acoustic_input(float x) { return x; }
__device__ __forceinline__ float acoustic_add(float a,float b,unsigned *p,int s) { return abr_add(a,b,p,s); }
__device__ __forceinline__ float acoustic_sub(float a,float b,unsigned *p,int s) { return abr_sub(a,b,p,s); }
__device__ __forceinline__ float acoustic_mul(float a,float b,unsigned *p,int s) { return abr_mul(a,b,p,s); }
__device__ __forceinline__ float acoustic_div(float a,float b,unsigned *p,int s) { return abr_div(a,b,p,s); }
#endif
struct AcousticField { const float *x; AcousticShadow *d; unsigned long long id; const AcousticShadow *seed; double *accum; };
struct AcousticDual { AcousticPrimal v; double d; };
struct AcousticTangent {
    typedef AcousticDual V;
    typedef AcousticShadow S;
    AcousticFailure *failure; unsigned *branches; int branch_base;
    __device__ void record(int slot,bool arm) {abr_count(branches,slot,arm);}
    __device__ V constant(float x) { return {acoustic_input(x),0.0f}; }
    __device__ V input(AcousticField a,int i) {
        float x=a.x[i];double d=acoustic_shadow_decode(a.d[i]);
        if(!isfinite(x)||!isfinite(d)) acoustic_fail(failure,a.id,i,1);
        return {acoustic_input(x),d};
    }
    __device__ V scalar(float x,AcousticShadow *p,unsigned long long id) {
        double d=acoustic_shadow_decode(*p); if(!isfinite(x)||!isfinite(d)) acoustic_fail(failure,id,0,1);
        return {acoustic_input(x),d};
    }
    __device__ V add(V a,V b) { return checked({acoustic_add(a.v,b.v,branches,branch_base),a.d+b.d}); }
    __device__ V sub(V a,V b) { return checked({acoustic_sub(a.v,b.v,branches,branch_base),a.d-b.d}); }
    __device__ V mul(V a,V b) { return checked({acoustic_mul(a.v,b.v,branches,branch_base),a.d*acoustic_primal_decode(b.v)+acoustic_primal_decode(a.v)*b.d}); }
    __device__ V div(V a,V b) {
        if(acoustic_primal_decode(b.v)==0.0) acoustic_fail(failure,0,0,2);
        AcousticPrimal v=acoustic_div(a.v,b.v,branches,branch_base);
        const double den=acoustic_primal_decode(b.v);
        return checked({v,a.d*(1.0/den)+b.d*(-acoustic_primal_decode(v)/den)});
    }
    __device__ V checked(V a) { if(!isfinite(a.v)||!isfinite(a.d)) acoustic_fail(failure,0,0,3);return a; }
    __device__ void output(AcousticField a,int i,V v) { a.d[i]=acoustic_shadow_encode(v.d); }
    __device__ void finish() {}
};
// The bucket lives in every tape slot so repeated reads of one device value
// share one node. Its cotangent is accumulated in f64 and written once.
struct AcousticNode { double bar, left_weight, right_weight; int left,right; unsigned long long destination; int bucket; };
static_assert(sizeof(AcousticNode)==48,"AcousticNode ABI");
struct AcousticValue { AcousticPrimal v; int id; };
struct AcousticReverse {
    typedef AcousticValue V;
    typedef double S;
    AcousticNode *nodes; int count,capacity; AcousticFailure *failure;
    __device__ void reset() { for(int i=0;i<capacity;++i) nodes[i].bucket=-1; }
    __device__ void record(int,bool) {}
    __device__ V constant(float x) { return {acoustic_input(x),-1}; }
    __device__ V push(AcousticPrimal x,int l,int r,double dl,double dr,unsigned long long dst) {
        if(!isfinite(x)||!isfinite(dl)||!isfinite(dr)) acoustic_fail(failure,0,0,3);
        if(count>=capacity) { acoustic_fail(failure,0,count,4);return {x,-1}; }
        int id=count++;AcousticNode &n=nodes[id];
        n.bar=0.0;n.left_weight=dl;n.right_weight=dr;n.left=l;n.right=r;n.destination=dst;
        return {x,id};
    }
    __device__ V leaf(AcousticPrimal x,unsigned long long dst) {
        unsigned long long h=(dst>>3)^(dst>>19)^(dst>>37);
        int bucket=(int)(h%(unsigned long long)capacity);
        for(int i=nodes[bucket].bucket;i>=0;i=nodes[i].left)
            if(nodes[i].destination==dst) return {x,i};
        V v=push(x,nodes[bucket].bucket,-1,0.0f,0.0f,dst);
        if(v.id>=0)nodes[bucket].bucket=v.id;
        return v;
    }
    __device__ V input(AcousticField a,int i) {
        double d=a.accum?a.accum[i]:acoustic_shadow_decode(a.d[i]);
        if(!isfinite(a.x[i])||!isfinite(d)) acoustic_fail(failure,a.id,i,1);
        unsigned long long dst=a.accum?(((unsigned long long)(a.accum+i))|1ULL):(unsigned long long)(a.d+i);
        return leaf(acoustic_input(a.x[i]),dst);
    }
    __device__ V scalar(float x,double *d,unsigned long long id) {
        if(!isfinite(x)||!isfinite(*d)) acoustic_fail(failure,id,0,1);
        return leaf(acoustic_input(x),((unsigned long long)d)|1ULL);
    }
    __device__ V add(V a,V b) { return push(acoustic_add(a.v,b.v,nullptr,0),a.id,b.id,1.0,1.0,0ULL); }
    __device__ V sub(V a,V b) { return push(acoustic_sub(a.v,b.v,nullptr,0),a.id,b.id,1.0,-1.0,0ULL); }
    __device__ V mul(V a,V b) { return push(acoustic_mul(a.v,b.v,nullptr,0),a.id,b.id,acoustic_primal_decode(b.v),acoustic_primal_decode(a.v),0ULL); }
    __device__ V div(V a,V b) {
        if(acoustic_primal_decode(b.v)==0.0) acoustic_fail(failure,0,0,2);
        const AcousticPrimal v=acoustic_div(a.v,b.v,nullptr,0);
        const double den=acoustic_primal_decode(b.v);
        return push(v,a.id,b.id,1.0/den,-acoustic_primal_decode(v)/den,0ULL);
    }
    __device__ void output(AcousticField a,int i,V v) {
        double seed=acoustic_shadow_decode(a.seed ? a.seed[i] : a.d[i]);if(!isfinite(seed)) acoustic_fail(failure,a.id,i,1);
        if(v.id>=0) nodes[v.id].bar+=seed;
        if(!a.seed) a.d[i]=acoustic_shadow_encode(0.0);
    }
    __device__ void finish() {
        for(int i=count-1;i>=0;--i) {
            AcousticNode &n=nodes[i];
            if(n.destination) {
                if(n.destination&1ULL) {
                    double *dst=(double *)(n.destination&~1ULL);double z=*dst+n.bar;
                    if(!isfinite(z)) acoustic_fail(failure,0,i,3);*dst=z;
                } else {
                    AcousticShadow *dst=(AcousticShadow *)n.destination;AcousticShadow z=acoustic_shadow_encode(acoustic_shadow_decode(*dst)+n.bar);
                    if(!isfinite(z)) acoustic_fail(failure,0,i,3);*dst=z;
                }
            } else {
                if(n.left>=0) nodes[n.left].bar+=n.bar*n.left_weight;
                if(n.right>=0) nodes[n.right].bar+=n.bar*n.right_weight;
            }
        }
    }
};
