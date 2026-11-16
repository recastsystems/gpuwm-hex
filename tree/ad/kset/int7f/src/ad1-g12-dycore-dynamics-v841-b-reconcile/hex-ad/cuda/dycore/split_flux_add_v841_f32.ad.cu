// SPDX-License-Identifier: MIT
// Interim hand implementation of the corresponding fixed-linear opspec.
// Compile with --ftz=false --fmad=false. Device arrays and refusal slots are
// disjoint, and nlevels*nowners must fit the allocated contiguous arrays.
__device__ __forceinline__ void refuse(int* r, long long, unsigned long long) { atomicExch(r, 1); }
__device__ __forceinline__ unsigned bits_abs(float x) {return __float_as_uint(x)&0x7fffffffu;}
__device__ __forceinline__ bool subnormal(float x) {unsigned m=bits_abs(x);return m!=0u&&(m&0x7f800000u)==0u;}
__device__ __forceinline__ float flushed(float x) {unsigned b=__float_as_uint(x);return subnormal(x)?__uint_as_float(b&0x80000000u):x;}
__device__ __forceinline__ void add_live(float a,float b,bool* la,bool* lb) {
    float ordinary=__fadd_rn(flushed(a),flushed(b));
#ifndef AD_NUMERICS_STRICT
    bool out_live=!subnormal(ordinary);
    *la=out_live&&!subnormal(a); *lb=out_live&&!subnormal(b);
#else
    // A strict-helper early return can still execute DAZ on a tiny operand.
    // Its fallback handles the remaining subnormal-operand cases.
    if ((bits_abs(flushed(ordinary))>>23)>24u) { *la=!subnormal(a); *lb=!subnormal(b); }
    else { *la=true; *lb=true; }
#endif
}
// One pair of counters per declared predicate. Integer atomic adds are exact
// and deterministic; the optional pointer is disjoint from all fields.
__device__ __forceinline__ void record_add(unsigned* r,float a,float b) {
    if (!r) return;
    float raw=__fadd_rn(flushed(a),flushed(b));
    unsigned oe=bits_abs(flushed(raw))>>23, am=bits_abs(a), bm=bits_abs(b), ae=am>>23, be=bm>>23;
#ifdef AD_NUMERICS_STRICT
    bool early=oe>24u, fallback=false;
    atomicAdd(r+0*2+(unsigned)early,1u);
    if (!early) {
        bool sub=(ae==0u&&am)||(be==0u&&bm);
        atomicAdd(r+1*2+(unsigned)sub,1u);
        if (!sub) {
            bool cancel=ae<=23u&&be<=23u&&oe==0u&&am!=bm&&am&&bm;
            atomicAdd(r+2*2+(unsigned)cancel,1u); fallback=cancel;
        } else fallback=true;
    }
    // Branches inside the original double decode and narrow-result packing.
    // These preserve the real value; sign selection has unit derivative.
    if (fallback) {
        bool as=subnormal(a),bs=subnormal(b);
        atomicAdd(r+6*2+(unsigned)as,1u); atomicAdd(r+7*2+(unsigned)bs,1u);
        if(as) atomicAdd(r+8*2+(__float_as_uint(a)>>31),1u);
        if(bs) atomicAdd(r+9*2+(__float_as_uint(b)>>31),1u);
        double v=(double)a+(double)b, m=fabs(v);
        bool pack=m>0.0&&m<1.1754943508222875e-38;
        atomicAdd(r+10*2+(unsigned)pack,1u);
        if(pack) atomicAdd(r+11*2+(unsigned)(__double_as_longlong(v)<0LL),1u);
    }
#endif
    atomicAdd(r+3*2+(unsigned)subnormal(a),1u);
    atomicAdd(r+4*2+(unsigned)subnormal(b),1u);
    atomicAdd(r+5*2+(unsigned)subnormal(raw),1u);
}
// The overwritten input is required for the executed DAZ/FTZ mask. It cannot
// be recovered from the rounded output, so this augmented forward saves it.
#if !defined(AD_TL_ONLY) && !defined(AD_AD_ONLY)
extern "C" __global__ void split_flux_add_v841_f32__aug(const int nlevels,const int nowners,const float* current,float* accumulator,float* saved_accumulator) {
    const long long n=(long long)nlevels*nowners;
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
        const float old=accumulator[i]; saved_accumulator[i]=old;
#ifndef AD_NUMERICS_STRICT
        accumulator[i]=flushed(__fadd_rn(flushed(current[i]),flushed(old)));
#else
        accumulator[i]=__fadd_rn(current[i],old);
#endif
    }
}
#endif

#if !defined(AD_AD_ONLY) && !defined(AD_AUG_ONLY)
extern "C" __global__ void split_flux_add_v841_f32__tl(const int nlevels, const int nowners, const float* current, float* accumulator, const float* d_current, float* d_accumulator, const float* saved_accumulator, int* failures, unsigned* branch_record) {
    const long long n=(long long)nlevels*nowners;
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(long long)gridDim.x*blockDim.x) {
        bool la,lb;add_live(current[i],saved_accumulator[i],&la,&lb); record_add(branch_record,current[i],saved_accumulator[i]);
        d_accumulator[i]=__fadd_rn(la?d_current[i]:0.0f,lb?d_accumulator[i]:0.0f);
        if (!isfinite(current[i]) || !isfinite(accumulator[i]) || !isfinite(saved_accumulator[i]) || !isfinite(d_current[i]) || !isfinite(d_accumulator[i])) refuse(failures,i,1ULL);
    }
}
#endif
#if !defined(AD_TL_ONLY) && !defined(AD_AUG_ONLY)
extern "C" __global__ void split_flux_add_v841_f32__ad(const int nlevels, const int nowners, const float* current, float* accumulator, float* b_current, const float* b_accumulator_out, float* b_accumulator_in, const float* saved_accumulator, int* failures, unsigned* branch_record) {
    const long long n=(long long)nlevels*nowners;
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(long long)gridDim.x*blockDim.x) {
        bool invalid=!isfinite(current[i]) || !isfinite(accumulator[i]) || !isfinite(saved_accumulator[i]) || !isfinite(b_current[i]) || !isfinite(b_accumulator_out[i]) || !isfinite(b_accumulator_in[i]);
        bool la,lb;add_live(current[i],saved_accumulator[i],&la,&lb); record_add(branch_record,current[i],saved_accumulator[i]);
        const float seed=b_accumulator_out[i]; if(la) b_current[i]=__fadd_rn(b_current[i],seed); if(lb) b_accumulator_in[i]=__fadd_rn(b_accumulator_in[i],seed);
        if (invalid || !isfinite(b_current[i]) || !isfinite(b_accumulator_in[i])) refuse(failures,i,1ULL);
    }
}

#endif
