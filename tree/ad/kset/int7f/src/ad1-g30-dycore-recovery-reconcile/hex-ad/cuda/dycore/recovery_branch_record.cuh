// SPDX-License-Identifier: Apache-2.0
// The count in each arm is an exact, order-independent integer reduction.
// Slots 0..10 follow cuda_fp32.py lines 42,45,57,62,78,83,93,142,145,162,170.
#pragma once

struct RecoveryBranchCounts {
  unsigned int counts[34];
  __device__ __forceinline__ RecoveryBranchCounts() : counts{} {}
  __device__ __forceinline__ void mark(int slot, bool arm) { ++counts[2*slot+(arm?1:0)]; }
  __device__ __forceinline__ void publish(unsigned int *out, int slots) {
    unsigned int mask=__activemask();
    int first=__ffs(mask)-1;
    for(int i=0;i<2*slots;++i){
      unsigned int total=__reduce_add_sync(mask,counts[i]);
      if((threadIdx.x & 31)==first && total) atomicAdd(out+i,total);
    }
  }
};

template<class T> struct RecoveryBranchTrace : RecoveryBranchCounts {
  __device__ __forceinline__ T add(T a,T b){return a+b;}
  __device__ __forceinline__ T mul(T a,T b){return a*b;}
  __device__ __forceinline__ T div(T a,T b){return a/b;}
  __device__ __forceinline__ T sign(T x,int slot){this->mark(slot,signbit(x));return copysign(T(1),x);}
};

template<> struct RecoveryBranchTrace<float> : RecoveryBranchCounts {
  __device__ __forceinline__ unsigned mag(float x){return __float_as_uint(x)&0x7fffffffu;}
  __device__ __forceinline__ double decode(float x){
    unsigned v=mag(x); bool sub=v!=0u && (v&0x7f800000u)==0u; this->mark(0,sub);
    if(sub)this->mark(1,(__float_as_uint(x)&0x80000000u)!=0u);
    return mpas_f32_to_f64_ieee(x);
  }
  __device__ __forceinline__ float encode(double x){
    double v=fabs(x);bool sub=v>0.0 && v<1.1754943508222875e-38;this->mark(2,sub);
    if(sub)this->mark(3,__double_as_longlong(x)<0LL);
    return mpas_f64_to_f32_ieee(x);
  }
  __device__ __forceinline__ float add(float a,float b){
    float ordinary=a+b;
#if MPAS_FTZ_FALLBACK_ENABLED
    unsigned result=mag(ordinary)>>23;bool big=result>24u;this->mark(4,big);
    if(!big){unsigned am=mag(a),bm=mag(b),ae=am>>23,be=bm>>23;
      bool sub=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);this->mark(5,sub);
      if(sub)return encode(decode(a)+decode(b));
      bool cancel=ae<=23u&&be<=23u&&result==0u&&am!=bm&&am!=0u&&bm!=0u;this->mark(6,cancel);
      if(cancel)return encode(decode(a)+decode(b));
    }
#endif
    return ordinary;
  }
  __device__ __forceinline__ float mul(float a,float b){
    float ordinary=a*b;
#if MPAS_FTZ_FALLBACK_ENABLED
    bool zeroexp=(mag(ordinary)&0x7f800000u)==0u;this->mark(7,zeroexp);
    if(zeroexp){bool nonzero=mag(a)!=0u&&mag(b)!=0u;this->mark(8,nonzero);
      if(nonzero)return encode(decode(a)*decode(b));}
#endif
    return ordinary;
  }
  __device__ __forceinline__ float div(float a,float b){
#if MPAS_FTZ_FALLBACK_ENABLED
    unsigned am=mag(a),bm=mag(b),ae=am>>23,be=bm>>23;
    bool sub=(ae==0u&&am!=0u)||(be==0u&&bm!=0u);this->mark(9,sub);
    if(sub)return encode(decode(a)/decode(b));
#endif
    float ordinary=a/b;
#if MPAS_FTZ_FALLBACK_ENABLED
    bool small=am!=0u&&bm!=0u&&ae+126u<=be&&(mag(ordinary)&0x7f800000u)==0u;this->mark(10,small);
    if(small)return encode(decode(a)/decode(b));
#endif
    return ordinary;
  }
  __device__ __forceinline__ float sign(float x,int slot){this->mark(slot,(__float_as_uint(x)&0x80000000u)!=0u);return mpas_copysign(1.0f,x);}
};
