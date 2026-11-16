#include "glue.member_magnitude.cuh"
// SPDX-License-Identifier: MIT
// Shared by this group's new derivative cubins only. Forward code is untouched.
#ifndef HEX_AD_DYCORE_COMMON_CUH
#define HEX_AD_DYCORE_COMMON_CUH

// The section 33 failure ABI is one zeroed int32 flag per distinct cause.
// A separate zeroed uint64 workspace records one plus a failing logical
// element; the source/field to slot mapping is in the spec. Both atomics are
// integer and order-independent. The host checks flags after synchronization
// and combines each flag, its spec metadata, and the workspace into the ABI
// failure record.
#define AD_FAILURE_CASE(source, field, slot) \
    case (((unsigned long long)(source) << 32) | (field)): at = slot; break;
__device__ __forceinline__ void ad_refuse(int *f, unsigned long long *detail,
        unsigned long long i, unsigned long long field,
        unsigned long long source) {
    int at = -1;
    switch ((source << 32) | field) { AD_FAILURE_PAIRS(AD_FAILURE_CASE) }
    if (at >= 0) { atomicOr(f + at, 1); atomicMax(detail + at, i + 1ULL); }
}
#undef AD_FAILURE_CASE

// The optional branch census uses integer counts: [slot][0=false,1=true].
// Integer addition is deterministic. These helpers mirror the source's
// guarded IEEE operations, including which fallback predicates execute.
__device__ __forceinline__ void ad_branch(unsigned int *r, int slot, bool arm) {
    if (r && slot >= 0) atomicAdd(r + 2 * slot + (arm ? 1 : 0), 1u);
}
__device__ __forceinline__ unsigned ad_mag(float x) {
    return glue_f32_magnitude_bits(x);
}
__device__ __forceinline__ double ad_record_decode(float x, unsigned *r, int slot) {
    unsigned bits = __float_as_uint(x), mag = bits & 0x7fffffffu;
    bool sub = mag != 0u && (mag & 0x7f800000u) == 0u;
    ad_branch(r, slot, sub);
    if (sub) {
        double d = ldexp((double)(mag & 0x007fffffu), -149);
        return (bits & 0x80000000u) ? -d : d;
    }
    return (double)x;
}
__device__ __forceinline__ float ad_record_pack(double x, unsigned *r, int slot) {
    double mag = fabs(x);
    bool sub = mag > 0.0 && mag < 1.1754943508222875e-38;
    ad_branch(r, slot, sub);
    if (sub) {
        unsigned mantissa = (unsigned)rint(mag * 7.1362384635297994e+44);
        unsigned sign = (__double_as_longlong(x) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(sign | mantissa);
    }
    return __double2float_rn(x);
}
__device__ __forceinline__ float ad_record_addsub(float a, float b, bool sub,
        unsigned *r, int slot, int decode_slot, int pack_slot) {
    float ordinary = sub ? a - b : a + b;
    unsigned result = ad_mag(ordinary), re = result >> 23;
    bool large = re > 24u;
    ad_branch(r, slot, large);
    if (large) return ordinary;
    unsigned am = ad_mag(a), bm = ad_mag(b), ae = am >> 23, be = bm >> 23;
    bool operand_sub = (ae == 0u && am != 0u) || (be == 0u && bm != 0u);
    ad_branch(r, slot + 1, operand_sub);
    if (operand_sub) {
        double da = ad_record_decode(a, r, decode_slot);
        double db = ad_record_decode(b, r, decode_slot);
        return ad_record_pack(sub ? da - db : da + db, r, pack_slot);
    }
    bool cancellation = ae <= 23u && be <= 23u && re == 0u &&
                        am != bm && am != 0u && bm != 0u;
    ad_branch(r, slot + 2, cancellation);
    if (cancellation) {
        double da = ad_record_decode(a, r, decode_slot);
        double db = ad_record_decode(b, r, decode_slot);
        return ad_record_pack(sub ? da - db : da + db, r, pack_slot);
    }
    return ordinary;
}
__device__ __forceinline__ float ad_record_mul(float a, float b,
        unsigned *r, int slot, int decode_slot, int pack_slot) {
    float ordinary = a * b;
    bool envelope = (ad_mag(ordinary) & 0x7f800000u) == 0u;
    ad_branch(r, slot, envelope);
    if (envelope) {
        bool nonzero = ad_mag(a) != 0u && ad_mag(b) != 0u;
        ad_branch(r, slot + 1, nonzero);
        if (nonzero) {
            double da = ad_record_decode(a, r, decode_slot);
            double db = ad_record_decode(b, r, decode_slot);
            return ad_record_pack(da * db, r, pack_slot);
        }
    }
    return ordinary;
}
__device__ __forceinline__ float ad_record_div(float a, float b,
        unsigned *r, int slot, int decode_slot, int pack_slot) {
    unsigned am=ad_mag(a), bm=ad_mag(b), ae=am >> 23, be=bm >> 23;
    bool operand_sub = (ae == 0u && am != 0u) || (be == 0u && bm != 0u);
    ad_branch(r, slot, operand_sub);
    if (operand_sub) {
        double da=ad_record_decode(a,r,decode_slot), db=ad_record_decode(b,r,decode_slot);
        return ad_record_pack(da / db,r,pack_slot);
    }
    float ordinary=a / b;
    bool envelope = am != 0u && bm != 0u && ae + 126u <= be &&
                    (ad_mag(ordinary) & 0x7f800000u) == 0u;
    ad_branch(r, slot + 1, envelope);
    if (envelope) {
        double da=ad_record_decode(a,r,decode_slot), db=ad_record_decode(b,r,decode_slot);
        return ad_record_pack(da / db,r,pack_slot);
    }
    return ordinary;
}

// Fixed reduction tree. Launch with block=(256,1,1), grid=ceil(n/512).
// Repeat on the successive output arrays until one value remains; then launch
// ad_dycore_accumulate once. Buffers for consecutive stages must be disjoint.
// This produces one deterministic Euclidean cotangent for a by-value scalar.
#if !defined(AD_MODE_TL)
extern "C" __global__ void ad_dycore_reduce(const int n, const float *x, float *y) {
    __shared__ float part[256];
    int t = threadIdx.x, i = blockIdx.x * 512 + t;
    float a = i < n ? x[i] : 0.0f;
    float b = i + 256 < n ? x[i + 256] : 0.0f;
    part[t] = a + b;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride) part[t] += part[t + stride];
        __syncthreads();
    }
    if (t == 0) y[blockIdx.x] = part[0];
}
extern "C" __global__ void ad_dycore_accumulate(const float *x, float *bar) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *bar += *x;
}
#endif
#endif
