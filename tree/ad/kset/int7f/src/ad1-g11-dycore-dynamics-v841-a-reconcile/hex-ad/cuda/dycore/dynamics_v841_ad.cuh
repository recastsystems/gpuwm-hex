// SPDX-License-Identifier: Apache-2.0
// MPAS helpers from pinned tree/src/hexcore/cuda_fp32.py; derivative helpers follow.
#pragma once

#ifndef MPAS_FTZ_FALLBACK_ENABLED
#define MPAS_FTZ_FALLBACK_ENABLED 1
#endif

__device__ __forceinline__ unsigned int mpas_f32_magnitude_bits(
    const float value)
{
    return __float_as_uint(value) & 0x7fffffffu;
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

// The pinned helpers either execute the guarded IEEE operation or the native
// FTZ operation.  In the latter, a subnormal operand is a constant zero.  A
// decoded/repacked subnormal on the guarded path retains the derivative of
// the real arithmetic; rounding itself is not differentiated.
using AdFailureRecord = unsigned int;  // device failure slots; host builds the detailed record
struct AdCheck { AdFailureRecord *records; int slot; unsigned long long source; unsigned int failed = 0u; };
__device__ __forceinline__ void ad_fail(AdCheck &c, unsigned long long field,
                                      unsigned long long index) {
    (void)index;
    atomicExch(&c.records[32u + (unsigned)field], 1u);
    c.failed = 1u;
}
__device__ __forceinline__ float ad_read(const float *p, int i, int id, AdCheck &c) {
    const float x = p[i];
    if (!isfinite(x)) { atomicExch(&c.records[(unsigned)id], 1u); c.failed = 1u; }
    return x;
}
struct AdJet { float v, d; };
__device__ __forceinline__ AdJet ad_const(float x) { return {x, 0.0f}; }
// sel==0 uses a tangent array; otherwise it seeds one input element for a
// deterministic gather of one transpose row.  No seed pointer is read in
// basis mode.
__device__ __forceinline__ AdJet ad_input(const float *x, const float *dx, int i,
    int id, int sel, int target, AdCheck &c) {
    const float v = ad_read(x, i, id, c);
    const float d = sel ? (sel == id && target == i ? 1.0f : 0.0f) : ad_read(dx, i, id, c);
    return {v, d};
}
__device__ __forceinline__ bool ad_subnormal(float x) {
    const unsigned u = __float_as_uint(x) & 0x7fffffffu;
    return u && ((u & 0x7f800000u) == 0u);
}
__device__ __forceinline__ AdJet ad_add(AdJet a, AdJet b) {
    const float ordinary = a.v + b.v;
    const unsigned exp = (__float_as_uint(ordinary) & 0x7fffffffu) >> 23;
    const bool native_big = exp > 24u;
    const float da = native_big && ad_subnormal(a.v) ? 0.0f : a.d;
    const float db = native_big && ad_subnormal(b.v) ? 0.0f : b.d;
    return {mpas_add(a.v, b.v), mpas_add(da, db)};
}
__device__ __forceinline__ AdJet ad_sub(AdJet a, AdJet b) {
    const float ordinary = a.v - b.v;
    const unsigned exp = (__float_as_uint(ordinary) & 0x7fffffffu) >> 23;
    const bool native_big = exp > 24u;
    const float da = native_big && ad_subnormal(a.v) ? 0.0f : a.d;
    const float db = native_big && ad_subnormal(b.v) ? 0.0f : b.d;
    return {mpas_sub(a.v, b.v), mpas_sub(da, db)};
}
__device__ __forceinline__ AdJet ad_mul(AdJet a, AdJet b) {
    const float ordinary = a.v * b.v;
    const unsigned magnitude = __float_as_uint(ordinary) & 0x7fffffffu;
    const bool guarded = (magnitude & 0x7f800000u) == 0u
        && (__float_as_uint(a.v) & 0x7fffffffu) != 0u
        && (__float_as_uint(b.v) & 0x7fffffffu) != 0u;
    const float av = !guarded && ad_subnormal(a.v) ? 0.0f : a.v;
    const float bv = !guarded && ad_subnormal(b.v) ? 0.0f : b.v;
    return {mpas_mul(a.v, b.v), mpas_add(mpas_mul(a.d, bv), mpas_mul(av, b.d))};
}
// The extra counters contain executed *primal* helper predicates. Integer
// additions are order independent; they never accumulate floating gradients.
#ifndef AD_B78
#define AD_B78 -1
#define AD_B83 -1
#define AD_B93 -1
#define AD_B111 -1
#define AD_B116 -1
#define AD_B123 -1
#define AD_B142 -1
#define AD_B145 -1
#define AD_B42 -1
#define AD_B45 -1
#define AD_B57 -1
#define AD_B62 -1
#endif
__device__ __forceinline__ void ad_branch(unsigned *r,int slot,bool arm) {
    if (r && slot >= 0) atomicAdd(&r[2*slot+(arm?1:0)],1u);
}
__device__ __forceinline__ void ad_guard_trace(float left,float right,int op,unsigned *r) {
    const float x[2]={left,right};
    for (int i=0;i<2;++i) {
        unsigned bits=__float_as_uint(x[i]),mag=bits&0x7fffffffu;
        bool sub=mag && (mag&0x7f800000u)==0u;
        ad_branch(r,AD_B42,sub);
        if(sub) ad_branch(r,AD_B45,(bits&0x80000000u)!=0u);
    }
    double a=mpas_f32_to_f64_ieee(left),b=mpas_f32_to_f64_ieee(right);
    double value=op==0?a+b:op==1?a-b:a*b;
    double mag=fabs(value);
    bool pack=mag>0.0 && mag<1.1754943508222875e-38;
    ad_branch(r,AD_B57,pack);
    if(pack) ad_branch(r,AD_B62,__double_as_longlong(value)<0LL);
}
__device__ __forceinline__ void ad_addsub_trace(float left,float right,bool sub,unsigned *r) {
    float ordinary=sub?left-right:left+right;
    unsigned ex=(__float_as_uint(ordinary)&0x7fffffffu)>>23;
    bool big=ex>24u;
    ad_branch(r,sub?AD_B111:AD_B78,big);
    if(big)return;
    unsigned am=__float_as_uint(left)&0x7fffffffu,bm=__float_as_uint(right)&0x7fffffffu;
    unsigned ae=am>>23,be=bm>>23;
    bool subnormal=(ae==0u&&am)||(be==0u&&bm);
    ad_branch(r,sub?AD_B116:AD_B83,subnormal);
    if(subnormal){ad_guard_trace(left,right,sub?1:0,r);return;}
    bool cancel=ae<=23u&&be<=23u&&ex==0u&&am!=bm&&am&&bm;
    ad_branch(r,sub?AD_B123:AD_B93,cancel);
    if(cancel)ad_guard_trace(left,right,sub?1:0,r);
}
__device__ __forceinline__ AdJet ad_branch_add(AdJet a,AdJet b,unsigned*r) {
    if(r)ad_addsub_trace(a.v,b.v,false,r);return ad_add(a,b);
}
__device__ __forceinline__ AdJet ad_branch_sub(AdJet a,AdJet b,unsigned*r) {
    if(r)ad_addsub_trace(a.v,b.v,true,r);return ad_sub(a,b);
}
__device__ __forceinline__ AdJet ad_branch_mul(AdJet a,AdJet b,unsigned*r) {
    if(r){
        unsigned vm=__float_as_uint(a.v*b.v)&0x7fffffffu;
        bool envelope=(vm&0x7f800000u)==0u;
        ad_branch(r,AD_B142,envelope);
        if(envelope){
            bool nonzero=(__float_as_uint(a.v)&0x7fffffffu)&&(__float_as_uint(b.v)&0x7fffffffu);
            ad_branch(r,AD_B145,nonzero);
            if(nonzero)ad_guard_trace(a.v,b.v,2,r);
        }
    }
    return ad_mul(a,b);
}

__device__ __forceinline__ void ad_accumulate(float *p, int i, double add, int id, AdCheck &c) {
    const float before = ad_read(p, i, id, c);
    // Sum into the existing cotangent in float64, rounding once on output.
    const float after = mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(before) + add);
    if (!isfinite(add) || !isfinite(after)) ad_fail(c, id, i);
    if (!c.failed) p[i] = after;
}

struct ThetaData {
    int nl, nc, ne, me, corrected, w;
    const int *count, *edges;
    const float *sign, *area, *rdz, *edge, *vertical;
    const float *ru, *ru_saved, *theta, *dv;
    const int *cells;
};
struct ThetaDirection {
    const float *edge, *vertical, *ru, *ru_saved, *theta;
};
struct ThetaReverse {
    const float *output;
    float *edge, *vertical, *ru, *ru_saved, *theta;
    // Canonical, sorted, duplicate-free inverse incidence.  The scheduler
    // validates these against the pinned forward topology before launch.
    const int *edge_offsets, *edge_owners, *theta_offsets, *theta_owners;
};

#define ad_add(a,b) ad_branch_add((a),(b),branch_records)
#define ad_sub(a,b) ad_branch_sub((a),(b),branch_records)
#define ad_mul(a,b) ad_branch_mul((a),(b),branch_records)
__device__ __forceinline__ AdJet theta_eval(
    const ThetaData &s, const ThetaDirection &d, int k, int cell,
    int sel, int target, AdCheck &check, unsigned *branch_records=nullptr) {
    if (s.w && (k == 0 || k == s.nl)) return ad_const(0.0f);
    const int ci = k * s.nc + cell;
    AdJet value = ad_const(0.0f);
    for (int slot = 0; slot < s.count[cell]; ++slot) {
        const int pos = cell * s.me + slot, e = s.edges[pos], ei = k * s.ne + e;
        const float sign = ad_read(s.sign, pos, 7, check);
        value = ad_sub(value, ad_mul(ad_const(sign), ad_input(s.edge, d.edge, ei, 10, sel, target, check)));
    }
    if (s.corrected) {
        for (int slot = 0; slot < s.count[cell]; ++slot) {
            const int pos = cell * s.me + slot, e = s.edges[pos], ei = k * s.ne + e;
            const int c0 = s.cells[2 * e], c1 = s.cells[2 * e + 1];
            const float sign = ad_read(s.sign, pos, 7, check), dv = ad_read(s.dv, e, 15, check);
            AdJet flux = (ad_mul)(ad_const(sign), ad_const(dv)); // geometry-only predicate
            flux = ad_mul(flux, ad_sub(ad_input(s.ru_saved, d.ru_saved, ei, 13, sel, target, check),
                                       ad_input(s.ru, d.ru, ei, 12, sel, target, check)));
            flux = ad_mul(flux, ad_const(0.5f));
            flux = ad_mul(flux, ad_add(ad_input(s.theta, d.theta, k*s.nc+c1, 14, sel, target, check),
                                       ad_input(s.theta, d.theta, k*s.nc+c0, 14, sel, target, check)));
            value = ad_sub(value, flux);
        }
    }
    value = ad_mul(value, ad_const(ad_read(s.area, cell, 8, check)));
    const AdJet vertical = ad_sub(ad_input(s.vertical, d.vertical, (k+1)*s.nc+cell, 11, sel, target, check),
                                  ad_input(s.vertical, d.vertical, ci, 11, sel, target, check));
    return ad_sub(value, ad_mul(ad_const(ad_read(s.rdz, k, 9, check)), vertical));
}

#undef ad_add
#undef ad_sub
#undef ad_mul

__device__ __forceinline__ int theta_extent(const ThetaData &s) {
    const int edge = (s.nl + (s.w ? 1 : 0)) * s.ne;
    const int vert = (s.nl + 1) * s.nc;
    return edge > vert ? edge : vert;
}
__device__ __forceinline__ void theta_validate_tl(
    const ThetaData &s, const ThetaDirection &d, int i, AdCheck &c) {
    const int edge = (s.nl + (s.w ? 1 : 0)) * s.ne;
    if (i < edge) { ad_read(s.edge, i, 10, c); ad_read(d.edge, i, 10, c); }
    if (i < (s.nl+1)*s.nc) { ad_read(s.vertical, i, 11, c); ad_read(d.vertical, i, 11, c); }
    if (s.corrected && i < s.nl*s.ne) {
        ad_read(s.ru, i, 12, c); ad_read(d.ru, i, 12, c);
        ad_read(s.ru_saved, i, 13, c); ad_read(d.ru_saved, i, 13, c);
    }
    if (s.corrected && i < s.nl*s.nc) { ad_read(s.theta, i, 14, c); ad_read(d.theta, i, 14, c); }
}
__device__ __forceinline__ void theta_store_tl(
    const ThetaData &s, const ThetaDirection &d, int k, int cell,
    float *out, AdFailureRecord *failures, int validation_extent, unsigned *branch_records) {
    const int i = k*s.nc+cell;
    AdCheck c{failures, validation_extent+i, 1ull};
    const AdJet value = theta_eval(s, d, k, cell, 0, 0, c, branch_records);
    if (!isfinite(value.v) || !isfinite(value.d)) ad_fail(c, 17, i);
    if (!c.failed) out[i] = value.d;
}
__device__ __forceinline__ float theta_basis(
    const ThetaData &s, const ThetaDirection &unused, int k, int cell,
    int field, int index, float seed, AdCheck &c) {
    AdJet x = theta_eval(s, unused, k, cell, field, index, c);
    if (!isfinite(x.v) || !isfinite(x.d)) ad_fail(c, 17, k*s.nc+cell);
    return mpas_mul(seed, x.d);
}
__device__ __forceinline__ void theta_reverse_element(
    const ThetaData &s, const ThetaReverse &b, int i, AdCheck &c) {
    const ThetaDirection none{nullptr,nullptr,nullptr,nullptr,nullptr};
    const int edge_levels = s.nl + (s.w ? 1 : 0);
    const int nout = (s.nl + (s.w ? 1 : 0)) * s.nc;
    // Validate every active primal and output cotangent, including levels
    // whose output is a constant, before the caller accepts any gradient.
    if (i < nout) ad_read(b.output, i, 17, c);
    if (i < edge_levels*s.ne) ad_read(s.edge, i, 10, c);
    if (i < (s.nl+1)*s.nc) ad_read(s.vertical, i, 11, c);
    if (s.corrected && i < s.nl*s.ne) { ad_read(s.ru, i, 12, c); ad_read(s.ru_saved, i, 13, c); }
    if (s.corrected && i < s.nl*s.nc) ad_read(s.theta, i, 14, c);
    if (i < edge_levels*s.ne) {
        const int k=i/s.ne, e=i%s.ne;
        double be=0.0, br=0.0, bs=0.0;
        for (int p=b.edge_offsets[e]; p<b.edge_offsets[e+1]; ++p) {
            const int cell=b.edge_owners[p];
            const float seed=ad_read(b.output, k*s.nc+cell, 17, c);
            be += mpas_f32_to_f64_ieee(theta_basis(s,none,k,cell,10,i,seed,c));
            if (s.corrected) {
                br += mpas_f32_to_f64_ieee(theta_basis(s,none,k,cell,12,i,seed,c));
                bs += mpas_f32_to_f64_ieee(theta_basis(s,none,k,cell,13,i,seed,c));
            }
        }
        ad_accumulate(b.edge,i,be,10,c);
        if (s.corrected) { ad_accumulate(b.ru,i,br,12,c); ad_accumulate(b.ru_saved,i,bs,13,c); }
    }
    if (i < (s.nl+1)*s.nc) {
        const int k=i/s.nc, cell=i%s.nc;
        double bv=0.0;
        if (k < s.nl) {
            const float seed=ad_read(b.output,k*s.nc+cell,17,c);
            bv += mpas_f32_to_f64_ieee(theta_basis(s,none,k,cell,11,i,seed,c));
        }
        if (k > 0) {
            const float seed=ad_read(b.output,(k-1)*s.nc+cell,17,c);
            bv += mpas_f32_to_f64_ieee(theta_basis(s,none,k-1,cell,11,i,seed,c));
        }
        ad_accumulate(b.vertical,i,bv,11,c);
    }
    if (s.corrected && i < s.nl*s.nc) {
        const int k=i/s.nc, cell=i%s.nc;
        double bt=0.0;
        for (int p=b.theta_offsets[cell]; p<b.theta_offsets[cell+1]; ++p) {
            const int owner=b.theta_owners[p];
            const float seed=ad_read(b.output,k*s.nc+owner,17,c);
            bt += mpas_f32_to_f64_ieee(theta_basis(s,none,k,owner,14,i,seed,c));
        }
        ad_accumulate(b.theta,i,bt,14,c);
    }
}

struct MomentumData {
    int nl, nc, ne, me, has_initial;
    const float *u, *rho, *pv, *kinetic, *div;
    const int *cells, *neighbors, *count;
    const float *weights, *invdc, *reference, *f, *initial;
};
struct MomentumDirection {
    const float *u, *rho, *pv, *kinetic, *div, *reference, *f, *initial;
};
struct MomentumReverse {
    const float *output;
    float *u, *rho, *pv, *kinetic, *div, *reference, *f, *initial;
    const int *edge_offsets, *edge_owners, *cell_offsets, *cell_owners;
};
__device__ __forceinline__ int momentum_extent(const MomentumData &s) {
    const int a=s.nl*s.ne, b=s.nl*s.nc;
    return a>b?a:b;
}
#define ad_add(a,b) ad_branch_add((a),(b),branch_records)
#define ad_sub(a,b) ad_branch_sub((a),(b),branch_records)
#define ad_mul(a,b) ad_branch_mul((a),(b),branch_records)
__device__ __forceinline__ AdJet momentum_eval(
    const MomentumData &s, const MomentumDirection &d, int k, int edge,
    int sel, int target, AdCheck &c, unsigned *branch_records=nullptr) {
    const int i=k*s.ne+edge, c0=s.cells[2*edge], c1=s.cells[2*edge+1];
    AdJet q=ad_const(0.0f);
    for (int slot=0;slot<s.count[edge];++slot) {
        const int pos=edge*s.me+slot, n=s.neighbors[pos], ni=k*s.ne+n;
        const float weight=ad_read(s.weights,pos,13,c);
        const AdJet work=ad_mul(ad_const(0.5f),
            ad_add(ad_input(s.pv,d.pv,i,7,sel,target,c), ad_input(s.pv,d.pv,ni,7,sel,target,c)));
        q=ad_add(q,ad_mul(ad_mul(ad_const(weight),ad_input(s.u,d.u,ni,5,sel,target,c)),work));
    }
    for (int slot=0;slot<s.count[edge];++slot) {
        const int pos=edge*s.me+slot, n=s.neighbors[pos], ni=k*s.ne+n;
        const float weight=ad_read(s.weights,pos,13,c);
        q=ad_sub(q,ad_mul(ad_mul(ad_const(weight),ad_input(s.reference,d.reference,ni,15,sel,target,c)),
                              ad_input(s.f,d.f,edge,16,sel,target,c)));
    }
    const AdJet gradient=ad_mul(
        ad_sub(ad_input(s.kinetic,d.kinetic,k*s.nc+c1,8,sel,target,c),
               ad_input(s.kinetic,d.kinetic,k*s.nc+c0,8,sel,target,c)),
        ad_const(ad_read(s.invdc,edge,14,c)));
    const AdJet horizontal=ad_mul(ad_input(s.rho,d.rho,i,6,sel,target,c),ad_sub(q,gradient));
    const AdJet prior=s.has_initial ? ad_add(ad_input(s.initial,d.initial,i,17,sel,target,c),horizontal) : horizontal;
    const AdJet loss=ad_mul(ad_mul(ad_input(s.u,d.u,i,5,sel,target,c),ad_const(0.5f)),
        ad_add(ad_input(s.div,d.div,k*s.nc+c0,9,sel,target,c),
               ad_input(s.div,d.div,k*s.nc+c1,9,sel,target,c)));
    return ad_sub(prior,loss);
}
#undef ad_add
#undef ad_sub
#undef ad_mul

__device__ __forceinline__ void momentum_validate_tl(
    const MomentumData &s, const MomentumDirection &d, int i, AdCheck &c) {
    if (i<s.nl*s.ne) {
        ad_read(s.u,i,5,c); ad_read(d.u,i,5,c); ad_read(s.rho,i,6,c); ad_read(d.rho,i,6,c);
        ad_read(s.pv,i,7,c); ad_read(d.pv,i,7,c); ad_read(s.reference,i,15,c); ad_read(d.reference,i,15,c);
        ad_read(s.initial,i,17,c); ad_read(d.initial,i,17,c);
    }
    if (i<s.nl*s.nc) {
        ad_read(s.kinetic,i,8,c); ad_read(d.kinetic,i,8,c); ad_read(s.div,i,9,c); ad_read(d.div,i,9,c);
    }
    if (i<s.ne) { ad_read(s.f,i,16,c); ad_read(d.f,i,16,c); }
}
__device__ __forceinline__ void momentum_store_tl(
    const MomentumData &s, const MomentumDirection &d, int k, int e,
    float *out, AdFailureRecord *failures, int extent, unsigned *branch_records) {
    const int i=k*s.ne+e;
    AdCheck c{failures,extent+i,1ull};
    const AdJet value=momentum_eval(s,d,k,e,0,0,c,branch_records);
    if (!isfinite(value.v) || !isfinite(value.d)) ad_fail(c,19,i);
    if (!c.failed) out[i]=value.d;
}
__device__ __forceinline__ float momentum_basis(
    const MomentumData &s, const MomentumDirection &none, int k, int e,
    int field, int index, float seed, AdCheck &c) {
    const AdJet value=momentum_eval(s,none,k,e,field,index,c);
    if (!isfinite(value.v) || !isfinite(value.d)) ad_fail(c,19,k*s.ne+e);
    return mpas_mul(seed,value.d);
}
__device__ __forceinline__ void momentum_reverse_element(
    const MomentumData &s, const MomentumReverse &b, int i, AdCheck &c) {
    const MomentumDirection none{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    if (i<s.nl*s.ne) {
        ad_read(s.u,i,5,c); ad_read(s.rho,i,6,c); ad_read(s.pv,i,7,c);
        ad_read(s.reference,i,15,c); ad_read(s.initial,i,17,c); ad_read(b.output,i,19,c);
        const int k=i/s.ne, edge=i%s.ne;
        double bu=0.0, bpv=0.0, bref=0.0;
        for (int p=b.edge_offsets[edge];p<b.edge_offsets[edge+1];++p) {
            const int owner=b.edge_owners[p];
            const float seed=ad_read(b.output,k*s.ne+owner,19,c);
            bu += mpas_f32_to_f64_ieee(momentum_basis(s,none,k,owner,5,i,seed,c));
            bpv += mpas_f32_to_f64_ieee(momentum_basis(s,none,k,owner,7,i,seed,c));
            bref += mpas_f32_to_f64_ieee(momentum_basis(s,none,k,owner,15,i,seed,c));
        }
        const float seed=ad_read(b.output,i,19,c);
        const float brho=momentum_basis(s,none,k,edge,6,i,seed,c);
        const float binit=momentum_basis(s,none,k,edge,17,i,seed,c);
        ad_accumulate(b.u,i,bu,5,c); ad_accumulate(b.pv,i,bpv,7,c);
        ad_accumulate(b.reference,i,bref,15,c); ad_accumulate(b.rho,i,mpas_f32_to_f64_ieee(brho),6,c);
        ad_accumulate(b.initial,i,mpas_f32_to_f64_ieee(binit),17,c);
    }
    if (i<s.nl*s.nc) {
        ad_read(s.kinetic,i,8,c); ad_read(s.div,i,9,c);
        const int k=i/s.nc, cell=i%s.nc;
        double bkin=0.0, bdiv=0.0;
        for (int p=b.cell_offsets[cell];p<b.cell_offsets[cell+1];++p) {
            const int owner=b.cell_owners[p];
            const float seed=ad_read(b.output,k*s.ne+owner,19,c);
            bkin += mpas_f32_to_f64_ieee(momentum_basis(s,none,k,owner,8,i,seed,c));
            bdiv += mpas_f32_to_f64_ieee(momentum_basis(s,none,k,owner,9,i,seed,c));
        }
        ad_accumulate(b.kinetic,i,bkin,8,c); ad_accumulate(b.div,i,bdiv,9,c);
    }
    if (i<s.ne) {
        ad_read(s.f,i,16,c);
        double bf=0.0;
        for(int k=0;k<s.nl;++k) {
            const float seed=ad_read(b.output,k*s.ne+i,19,c);
            bf += mpas_f32_to_f64_ieee(momentum_basis(s,none,k,i,16,i,seed,c));
        }
        ad_accumulate(b.f,i,bf,16,c);
    }
}


