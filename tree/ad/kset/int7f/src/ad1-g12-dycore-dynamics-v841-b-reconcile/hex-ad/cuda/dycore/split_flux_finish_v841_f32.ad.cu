// SPDX-License-Identifier: MIT
// Interim hand implementation, including the native piecewise flush rules.
// Compile with --ftz=false --fmad=false. Device arrays and refusal slots are
// disjoint, and nlevels*nowners must fit the allocated contiguous arrays.
__device__ __forceinline__ void refuse(int* r, long long, unsigned long long) { atomicExch(r, 1); }
// The default follows the fast forward. Define AD_NUMERICS_STRICT for the
// strict helper. The derivative itself is always compiled
// with FTZ disabled so that its arithmetic does not introduce extra masks.
__device__ __forceinline__ bool is_subnormal(float x) {
    const unsigned u=__float_as_uint(x)&0x7fffffffu;
    return u!=0u && (u&0x7f800000u)==0u;
}
__device__ __forceinline__ bool live_product(float x,float reciprocal) {
#ifndef AD_NUMERICS_STRICT
    if (is_subnormal(x) || is_subnormal(reciprocal)) return false;
    // This unflushed result identifies the FTZ instruction's output mask.
    return !is_subnormal(__fmul_rn(x,reciprocal));
#else
    // When x is zero the strict helper does not enter its FP64 fallback.
    // Its ordinary multiplication therefore still flushes a subnormal
    // reciprocal, and its derivative with respect to x is zero. For a
    // nonzero x that same reciprocal enters the strict fallback instead.
    if ((__float_as_uint(x)&0x7fffffffu)==0u && is_subnormal(reciprocal)) return false;
    return true;
#endif
}


__device__ __forceinline__ void record_finish(unsigned* r,float x,float reciprocal) {
    if (!r) return;
    float a=is_subnormal(x)?__uint_as_float(__float_as_uint(x)&0x80000000u):x;
    float b=is_subnormal(reciprocal)?__uint_as_float(__float_as_uint(reciprocal)&0x80000000u):reciprocal;
    float raw=__fmul_rn(a,b);
#ifdef AD_NUMERICS_STRICT
    bool envelope=(__float_as_uint(raw)&0x7f800000u)==0u;
    atomicAdd(r+0*2+(unsigned)envelope,1u);
    if(envelope){
        bool both=(__float_as_uint(x)&0x7fffffffu)&&(__float_as_uint(reciprocal)&0x7fffffffu);
        atomicAdd(r+1*2+(unsigned)both,1u);
        if(both){
            bool xs=is_subnormal(x),rs=is_subnormal(reciprocal);
            atomicAdd(r+5*2+(unsigned)xs,1u); atomicAdd(r+6*2+(unsigned)rs,1u);
            if(xs)atomicAdd(r+7*2+(__float_as_uint(x)>>31),1u);
            if(rs)atomicAdd(r+8*2+(__float_as_uint(reciprocal)>>31),1u);
            double v=(double)x*(double)reciprocal, m=fabs(v);
            bool pack=m>0.0&&m<1.1754943508222875e-38;
            atomicAdd(r+9*2+(unsigned)pack,1u);
            if(pack)atomicAdd(r+10*2+(unsigned)(__double_as_longlong(v)<0LL),1u);
        }
    }
#endif
    atomicAdd(r+2*2+(unsigned)is_subnormal(x),1u);
    atomicAdd(r+3*2+(unsigned)is_subnormal(reciprocal),1u);
    atomicAdd(r+4*2+(unsigned)is_subnormal(raw),1u);
}

#ifndef AD_AD_ONLY
extern "C" __global__ void split_flux_finish_v841_f32__tl(const int nlevels, const int nowners, const float reciprocal, const float* accumulator, float* average, const float* d_accumulator, float* d_average, int* failures, unsigned* branch_record) {
    const long long n=(long long)nlevels*nowners;
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(long long)gridDim.x*blockDim.x) {
        record_finish(branch_record,accumulator[i],reciprocal); d_average[i]=live_product(accumulator[i],reciprocal)?__fmul_rn(d_accumulator[i],reciprocal):0.0f;
        if (!isfinite(reciprocal) || !isfinite(accumulator[i]) || !isfinite(d_accumulator[i]) || !isfinite(d_average[i])) refuse(failures,i,1ULL);
    }
}
#endif
#ifndef AD_TL_ONLY
extern "C" __global__ void split_flux_finish_v841_f32__ad(const int nlevels, const int nowners, const float reciprocal, const float* accumulator, float* average, float* b_accumulator, const float* b_average, int* failures, unsigned* branch_record) {
    const long long n=(long long)nlevels*nowners;
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(long long)gridDim.x*blockDim.x) {
        bool invalid=!isfinite(reciprocal) || !isfinite(accumulator[i]) || !isfinite(b_accumulator[i]) || !isfinite(b_average[i]);
        record_finish(branch_record,accumulator[i],reciprocal); const float seed=b_average[i];
        if(live_product(accumulator[i],reciprocal)) b_accumulator[i]=__fadd_rn(b_accumulator[i],__fmul_rn(seed,reciprocal));
        if (invalid || !isfinite(b_accumulator[i])) refuse(failures,i,1ULL);
    }
}

#endif
