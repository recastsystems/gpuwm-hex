// SPDX-License-Identifier: Apache-2.0
// P3 whole-launch MYNN-EDMF derivative types. Include after the captured MYNN prefix (primal routines).
#pragma once
#include "mynn_tape.cuh"

// Branch record: one bit per primal comparison or sign choice, in execution order, per column (thread of the
// launch). Off when mw_br_n is null. Used only to count switch flips between a base and a perturbed run.
__device__ unsigned int* mw_br_bits;
__device__ unsigned long long mw_br_words;      // words per column
__device__ unsigned long long* mw_br_n;         // comparisons per column
__device__ unsigned long long* mw_br_hash;      // order-sensitive hash of all outcomes per column
__device__ __forceinline__ bool mw_rec(bool r) {
    if (mw_br_n) {
        const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long n = mw_br_n[t];
        if (r && n < mw_br_words * 32ULL) mw_br_bits[t * mw_br_words + (n >> 5)] |= 1u << (n & 31);
        mw_br_hash[t] = (mw_br_hash[t] ^ (unsigned long long)(r ? 0x9e3779b97f4a7c15ULL : 0x6a09e667f3bcc909ULL)) * 0x100000001b3ULL + n;
        mw_br_n[t] = n + 1;
    }
    return r;
}

// Exact-type selectors so the integer-argument overloads never compete with the captured float routines.
template<class A, class B> struct mw_same { static constexpr bool v = false; };
template<class A> struct mw_same<A, A> { static constexpr bool v = true; };
template<class T> struct mw_is_int { static constexpr bool v = false; };
template<> struct mw_is_int<int> { static constexpr bool v = true; };
template<bool B, class T> struct mw_en {};
template<class T> struct mw_en<true, T> { using type = T; };

// Domain record: the first derivative-undefined operation (code from mw_ad_rules.inc) hit per column, with a
// dependent operand. The derivative policy refuses such a column (failure slot 1); this says which routine.
__device__ int* mw_dom;
template<class D> __device__ __forceinline__ void mw_domain(int code, D d) {
    if (d.depends && d.context) {
        d.context->fail(1);
        if (mw_dom) { const unsigned t = blockIdx.x * blockDim.x + threadIdx.x; if (!mw_dom[t]) mw_dom[t] = code; }
    }
}

// Trajectory freeze (gen_whole.py --freeze): the value is kept bitwise, the derivative dropped. Plain float/double/int
// pass through; the carrier overloads are in mw_ad_rules.inc.
template<class T> __device__ __forceinline__ T mw_frz(T x) { return x; }
// a12 truephys-term (gen_whole.py --l2-quot exact): plain float32 forms of the level-2 quotients, the generated
// statements' own operations (the carrier forms with their closed-form partials are in mw_ad_rules.inc).
__device__ __forceinline__ float mw_l2_sh(float shc, float rfc, float rf, float one) { return r6_div(r6_mul(shc, r6_sub(rfc, rf)), r6_sub(one, rf)); }
__device__ __forceinline__ float mw_l2_sm(float smc, float rf1, float rf2, float rf, float sh) { return r6_mul(r6_div(r6_mul(smc, r6_sub(rf1, rf)), r6_sub(rf2, rf)), sh); }
// fullphys mynn-exact (gen_whole.py --l25-quot exact): plain forms of the level-2.5 quotients, the generated statements'
// own double operations on the same words (float arguments widen exactly, so the forward
// is bitwise; double arguments keep a double instance double; the carrier forms with their closed-form partials are in mw_ad_rules.inc).
__device__ __forceinline__ double mw_l25_sm(double q3sq, double, double, float, double, double e3, double e4, double eden, double a1,
                                            double c13, float, float, float, float, float) {
    return mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, (double)a1), mynn_dsub(e3, mynn_dmul((double)c13, e4))), eden); }
__device__ __forceinline__ double mw_l25_sh(int tke_budget, double q3sq, double, double gmel, float, double e2, double,
                                            double, double eden, double c13e5c, double y, float, float, float, float, float) {
    const double X = mynn_dadd(mynn_dmul((double)c13e5c, gmel), e2), Y = mynn_dmul(q3sq, (double)y);
    return mynn_ddiv(tke_budget == 1 ? mynn_dmul(Y, X) : mynn_dmul(X, Y), eden); }
__device__ __forceinline__ float mw_l2_rf(float ri1, float ri, float ri2, float ri3, float ri4) { return r6_mul(ri1, r6_sub(r6_add(ri, ri2), mynn_sqrt(r6_add(r6_sub(mynn_ipow<2>(ri), r6_mul(ri3, ri)), ri4)))); }

#define ADF MynnTangentF
#define ADD MynnTangentD
#define DT MynnTangent
#include "mw_ad_rules.inc"
#undef ADF
#undef ADD
#undef DT
#define ADF MynnTapeF
#define ADD MynnTapeD
#define DT MynnLinear
#include "mw_ad_rules.inc"
#undef ADF
#undef ADD
#undef DT

// Freeze every element of an array (closure outputs of a routine, gen_whole.py --freeze).
template<class T, int R> __device__ __noinline__ void mw_frz_arr(MynnArray<T, R> a) {
    if (!a.data) return;
    const int n = a.size();
    for (int q = 0; q < n; ++q) a.linear(q) = mw_frz(a.linear(q));
}
