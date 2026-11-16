// SPDX-License-Identifier: Apache-2.0
// x20 MYNN-EDMF stage M2 (pbl-frozen): primal-only recompute of f_get_pblh and f_scale_aware for the fused
// per-column reverse kernel (adjoint-20x, branch ad/x20-mynn-M2). Reverse of this stage: none.
//
// Why no reverse: under gen_whole.py --freeze closure (the pk120 and pkg18m2 bundles), every real output of
// these routines is frozen right after the call (pblh; psig_bl, psig_shcu; kpbl is integer control), so the
// taped nodes they make today all end with a zero adjoint. This header evaluates them in plain float with the
// same operations, in the same order, as the generated MR=float path of mynn_whole.gen.cu (b8d132a0), so the
// values are bitwise those of the forward and of the taped record.
//
// Needs, from the generated prefix: r6_add/r6_sub/r6_mul/r6_div(float,float), mynn_min/mynn_max<float>,
// mynn_tanh(float), mynn_pow(float,float), mynn_ipow<2>(float). Include it after them (the gen.cu includes it
// right after mw_types.cuh).
//
// Accessors: A is anything with float operator()(int k) at Fortran level index k (kts..kte+1 for zw), e.g. a
// register array lambda, a shared-memory row, or the generated MynnArray (via m2_pv for carrier elements).
// Only the true stage inputs are read: thv, zw, dz (levels), ust, landsea, dx. qke is NOT read: the pinned
// stable_method is 1, so the generated stable_method==0 branch that reads qke is dead (constexpr).
#pragma once

namespace mynn_m2 {
static constexpr float zero = 0.0f, one = 1.0f, ten = 10.0f, hundred = 100.0f, p5 = 0.5f;
static constexpr float sbl_lim = 200.f, sbl_damp = 170.f;
}

// f_get_pblh (module_bl_mynnedmf.F90, RRFS 44c0849 / c6d08709), stable_method = 1. Writes pblh and kpbl.
template<class A, class B, class C>
__device__ __forceinline__ void m2_get_pblh(int kts, int kte, A thv1, B zw1, C dz1, float ust, float landsea,
                                            float& pblh, int& kpbl) {
    using namespace mynn_m2;
    kpbl = 2;
    int k = kts + 1, kthv = 1;
    float minthv = 9.e9f;
    while (zw1(k) <= 200.f) {
        if (minthv > thv1(k)) { minthv = thv1(k); kthv = k; }
        k = k + 1;
    }
    (void)kthv;
    const float delt_thv = (r6_sub(landsea, 1.5f) >= zero) ? 0.75f : 1.25f;
    float pb = zero;
    for (k = kts + 1; k <= kte - 1; ++k) {
        const float tk = thv1(k);
        if (tk >= r6_add(minthv, delt_thv)) {
            pb = r6_sub(zw1(k), r6_mul(dz1(k - 1), mynn_min<float>(r6_div(r6_sub(tk, r6_add(minthv, delt_thv)),
                                                   mynn_max<float>(r6_sub(tk, thv1(k - 1)), 1e-6f)), one)));
        }
        if (k == kte - 1) pb = zw1(kts + 1);
        if (pb > zero) break;
    }
    float wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pb, hundred), hundred))), p5);
    const float cpblh = r6_add(r6_mul(400.f, r6_sub(one, wt)), r6_mul(700.f, wt));
    float st = mynn_max<float>(r6_mul(ust, cpblh), ten);
    st = mynn_max<float>(st, r6_sub(pb, sbl_lim));
    st = mynn_min<float>(st, r6_add(pb, sbl_lim));
    wt = r6_add(r6_mul(p5, mynn_tanh(r6_div(r6_sub(pb, sbl_lim), sbl_damp))), p5);
    pb = r6_add(r6_mul(st, r6_sub(one, wt)), r6_mul(pb, wt));
    for (k = kts + 1; k <= kte - 1; ++k) {
        if (zw1(k) >= pb) { kpbl = k - 1; break; }
    }
    pblh = pb;
}

// f_scale_aware. Writes psig_bl and psig_shcu.
__device__ __forceinline__ void m2_scale_aware(float dx, float pblh, float& psig_bl, float& psig_shcu) {
    using namespace mynn_m2;
    float dxdh = r6_div(mynn_max<float>(r6_mul(2.5f, dx), 10.f), mynn_min<float>(pblh, 3000.f));
    float a = r6_add(mynn_ipow<2>(dxdh), r6_mul(0.106f, mynn_pow(dxdh, 0.667f)));
    float b = r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.066f, mynn_pow(dxdh, 0.667f))), 0.071f);
    float pb = r6_div(a, b);
    dxdh = r6_div(mynn_max<float>(r6_mul(2.5f, dx), 10.f), mynn_min<float>(r6_add(pblh, 500.f), 3500.f));
    a = r6_add(mynn_ipow<2>(dxdh), r6_mul(0.145f, mynn_pow(dxdh, 0.667f)));
    b = r6_add(r6_add(mynn_ipow<2>(dxdh), r6_mul(0.172f, mynn_pow(dxdh, 0.667f))), 0.170f);
    float ps = r6_div(a, b);
    if (pb > one) pb = one;
    if (pb < zero) pb = zero;
    if (ps > one) ps = one;
    if (ps < zero) ps = zero;
    psig_bl = pb; psig_shcu = ps;
}

// The whole stage as the fused kernel calls it, once per f_mynnedmf call site (step-1 init block and the
// every-step block): pblh, kpbl, then psig_bl/psig_shcu (1, 1 when scaleaware <= 0, as the driver does).
template<class A, class B, class C>
__device__ __forceinline__ void m2_pbl_stage(int kts, int kte, A thv1, B zw1, C dz1, float ust, float landsea,
                                             float dx, bool scaleaware, float& pblh, int& kpbl,
                                             float& psig_bl, float& psig_shcu) {
    m2_get_pblh(kts, kte, thv1, zw1, dz1, ust, landsea, pblh, kpbl);
    if (scaleaware) m2_scale_aware(dx, pblh, psig_bl, psig_shcu);
    else { psig_bl = mynn_m2::one; psig_shcu = mynn_m2::one; }
}

// ---- swap-in glue for the generated tape kernel (proof only; the fused kernel calls the functions above) ----
// Primal value of a column element: identity for float, .v for the AD carriers (MynnTangentF, MynnTapeF).
__device__ __forceinline__ float m2_pv(float x) { return x; }
template<class T> __device__ __forceinline__ float m2_pv(const T& x) { return x.v; }

// Drop-in for "f_get_pblh(kts, kte, pblh, thv1, qke1, ust, zw1, dz1, xland, kpbl);" in MynnColumn: the frozen
// output is written as a constant carrier (no tape node, no count-pass node), kpbl as before.
template<class MR, class Arr>
__device__ __forceinline__ void m2_get_pblh_mr(int kts, int kte, MR& pblh, Arr thv1, MR ust, Arr zw1, Arr dz1,
                                               MR landsea, int& kpbl) {
    thv1.rebase({kts}); zw1.rebase({kts}); dz1.rebase({kts});
    float pb; int kp;
    m2_get_pblh(kts, kte, [&](int k) { return m2_pv(thv1(k)); }, [&](int k) { return m2_pv(zw1(k)); },
                [&](int k) { return m2_pv(dz1(k)); }, m2_pv(ust), m2_pv(landsea), pb, kp);
    pblh = MR(pb); kpbl = kp;
}
template<class MR>
__device__ __forceinline__ void m2_scale_aware_mr(MR dx, MR pblh, MR& psig_bl, MR& psig_shcu) {
    float a, b;
    m2_scale_aware(m2_pv(dx), m2_pv(pblh), a, b);
    psig_bl = MR(a); psig_shcu = MR(b);
}
