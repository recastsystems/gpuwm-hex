// SPDX-License-Identifier: Apache-2.0
// x20 MYNN-EDMF stage M7 (turb-prod): hand reverse of the LIVE remainder of f_mym_turbulence
// (MYNN-EDMF c6d08709, captured source 70da4af7, module_bl_mynnedmf.F90:2757-3227), under gen_whole.py --freeze closure.
//
// Shape (06:50Z coordinator ruling): __device__ functions only, register-resident, recompute-not-load, no tape.
// The integrator inlines them into the one fused per-column MYNN reverse kernel.
//
// What is live under --freeze closure (gen_whole.py FREEZE_STMT/FREEZE_AFTER):
//   frozen  : el (after f_mym_length), sm, sh, qdiv, gamt, gamq, gamv (frozen before `elq = el*qkw`),
//             dfm/dfh/dfq (frozen where formed, and after the call), tkeprod_dn/up (plume groups), so the
//             SPP block and dfm/dfh/dfq carry no derivative and are not reversed here.
//   live in : qkw(k) (f_mym_length), gm/gh/dtl/dqw(k) (f_mym_level2), dz(k) (the tcd/qcd vertical difference).
//   live out: pdk, pdt, pdq, pdc (k = kts+1..kte; kts is the constant 0), tcd, qcd (after the difference),
//             qshear1/qbuoy1 when tke_budget == 1.
//   Forward (per level k = kts+1..kte, float, fmad off):
//     elq = el*qkw;  elh = (float)((double)elq*qdiv)
//     pdk = elq*((sm*gm + sh*gh) + gamv) + p5*tkdn + p5*tkup
//     A = sh*dtl + gamt;  B = sh*dqw + gamq
//     pdt = (elh*A)*dtl;  pdq = (elh*B)*dqw;  pdc = ((elh*A)*dqw)*p5 + ((elh*B)*dtl)*p5
//     pre_t = elq*gamt;  pre_q = elq*gamq          (tcd/qcd before the difference)
//     qshear = (elq*sm)*gm;  qbuoy = elq*(gh*sh + gamv) + p5*tkdn + p5*tkup   (tke_budget == 1 only)
//   then pre_t(kts) = pre_t(kte) = 0 (constants) and, for k = kts..kte-1, tcd(k) = (pre_t(k+1) - pre_t(k))/dz(k);
//   tcd(kte) = 0. Same for qcd.
// With closure < 3 (the hrrrv5 deck: 2.6) gamt = gamq = gamv = 0 exactly, so tcd/qcd carry no derivative at all;
// m7_has_gam() lets the integrator skip that half.
//
// Precision: primal weights are the forward's float intermediates recomputed with __f*_rn (equal to r6_* on the
// default IEEE path), derivative arithmetic in double, as the generated tape. Sums are reassociated, so the words
// differ from the tape in the last bits; graded on dot/Taylor (stage map, "Gate words").

#pragma once

// Primal values the reverse needs at one level. All but qdiv/gamt/gamq/gamv are arena arrays the fused kernel's
// primal recompute already holds; qdiv, gamt, gamq, gamv are locals of the forward's f_mym_turbulence level loop and
// must be kept per level by the primal recompute at the frozen point (`elq = r6_mul(el(k), qkw(k));`).
struct M7Prim {
    float el, qkw, sm, sh, gm, gh, dtl, dqw;  // final sm/sh (after every clamp), el frozen
    double qdiv;                              // frozen; 1 when closure >= 3
    float gamt, gamq, gamv;                   // frozen; exactly 0 when closure < 3
};

// Output cotangents arriving at one level (from M8 predict and M9/M10 tendencies/mixing).
struct M7Bar {
    double pdk, pdt, pdq, pdc;   // level k = kts+1..kte
    double pre_t, pre_q;         // cotangent of tcd/qcd BEFORE the difference: from m7_diff_rev_gather
    double qshear, qbuoy;        // only read when tke_budget == 1
};

// Input cotangents produced at one level (accumulated: +=).
struct M7In { double qkw, gm, gh, dtl, dqw; };

__device__ __forceinline__ bool m7_has_gam(float closure) { return closure >= 3.0f; }

// Reverse of the level-local production terms at level k (kts+1 <= k <= kte). No branches but tke_budget.
__device__ __forceinline__ void m7_rev_level(const M7Prim& p, const M7Bar& b, int tke_budget, M7In& o) {
    const float p5 = 0.5f;
    const float elq = __fmul_rn(p.el, p.qkw);
    const float elh = (float)__dmul_rn((double)elq, p.qdiv);
    const float S = __fadd_rn(__fadd_rn(__fmul_rn(p.sm, p.gm), __fmul_rn(p.sh, p.gh)), p.gamv);
    const float A = __fadd_rn(__fmul_rn(p.sh, p.dtl), p.gamt);
    const float B = __fadd_rn(__fmul_rn(p.sh, p.dqw), p.gamq);
    const float elhA = __fmul_rn(elh, A), elhB = __fmul_rn(elh, B);
    // pdk
    double elq_b = b.pdk * (double)S;
    double gm_b = b.pdk * (double)elq * (double)p.sm;
    double gh_b = b.pdk * (double)elq * (double)p.sh;
    // pdt, pdq, pdc
    const double pdc5 = b.pdc * (double)p5;
    const double elhA_b = b.pdt * (double)p.dtl + pdc5 * (double)p.dqw;
    const double elhB_b = b.pdq * (double)p.dqw + pdc5 * (double)p.dtl;
    double dtl_b = b.pdt * (double)elhA + pdc5 * (double)elhB;
    double dqw_b = b.pdq * (double)elhB + pdc5 * (double)elhA;
    const double elh_b = elhA_b * (double)A + elhB_b * (double)B;
    dtl_b += elhA_b * (double)elh * (double)p.sh;   // through A
    dqw_b += elhB_b * (double)elh * (double)p.sh;   // through B
    elq_b += elh_b * p.qdiv;
    // tcd/qcd before the difference (zero weight when closure < 3)
    elq_b += b.pre_t * (double)p.gamt + b.pre_q * (double)p.gamq;
    if (tke_budget == 1) {
        const float elqsm = __fmul_rn(elq, p.sm);
        elq_b += b.qshear * (double)p.sm * (double)p.gm;
        gm_b += b.qshear * (double)elqsm;
        const float Q = __fadd_rn(__fmul_rn(p.gh, p.sh), p.gamv);
        elq_b += b.qbuoy * (double)Q;
        gh_b += b.qbuoy * (double)elq * (double)p.sh;
    }
    o.qkw += elq_b * (double)p.el;
    o.gm += gm_b; o.gh += gh_b; o.dtl += dtl_b; o.dqw += dqw_b;
}

// Primal pre_t/pre_q at level k (0 at kts and kte: the forward overwrites them with the constant 0).
__device__ __forceinline__ float m7_pre(float el, float qkw, float gam, int k, int kts, int kte) {
    return (k <= kts || k >= kte) ? 0.0f : __fmul_rn(__fmul_rn(el, qkw), gam);
}

// Reverse of the vertical difference tcd(k) = (pre(k+1) - pre(k))/dz(k), k = kts..kte-1, in GATHER form so lanes
// that own different levels never write the same word:
//   pre_b(k) = out_b(k-1)/dz(k-1) - out_b(k)/dz(k)   for kts < k < kte (pre(kts), pre(kte) are constants)
//   dz_b(k) += -out_b(k)*out(k)/dz(k)                 for kts <= k < kte
// out_b(k) is the cotangent of the final tcd(k) (out_b(kte) is ignored: tcd(kte) is the constant 0).
// pre_k, pre_kp1: primal pre at k and k+1 (m7_pre). Returns pre_b(k); adds dz_b(k).
__device__ __forceinline__ double m7_diff_rev_gather(int k, int kts, int kte, double out_b_km1, double out_b_k,
                                                     float dz_km1, float dz_k, float pre_k, float pre_kp1,
                                                     double& dz_b_k) {
    double pre_b = 0.0;
    if (k < kte) {
        const float out = __fdiv_rn(__fsub_rn(pre_kp1, pre_k), dz_k);
        const double q = out_b_k / (double)dz_k;
        if (k > kts) pre_b -= q;
        dz_b_k -= q * (double)out;
    }
    if (k > kts && k < kte) pre_b += out_b_km1 / (double)dz_km1;
    return pre_b;
}

// Whole-column driver for one column, levels strided over the column's lanes (lane_sub/lane_width as in the
// forward; one lane per column gives the plain loop). Accessors are callables (k) -> value so the integrator can
// pass arena arrays, registers or shared memory without copies. Writes are += into the cotangent accessors.
//   P(k)  -> M7Prim      primal at level k
//   DZ(k) -> float       dz(k)
//   OB(k) -> M7Bar with pdk/pdt/pdq/pdc/qshear/qbuoy filled (pre_t/pre_q ignored)
//   TB(k), QB(k) -> double  cotangent of the final tcd(k), qcd(k)
//   ACC(k, M7In) adds the level's input cotangents; ACCDZ(k, double) adds dz_b(k)
template<class FP, class FDZ, class FOB, class FTB, class FQB, class FACC, class FACCDZ>
__device__ __forceinline__ void m7_rev_column(int kts, int kte, float closure, int tke_budget, int lane_sub,
                                              int lane_width, FP P, FDZ DZ, FOB OB, FTB TB, FQB QB, FACC ACC,
                                              FACCDZ ACCDZ) {
    const bool gam = m7_has_gam(closure);
    for (int k = kts + lane_sub; k <= kte; k += lane_width) {
        M7In o = {0.0, 0.0, 0.0, 0.0, 0.0};
        double dz_b = 0.0;
        M7Bar b = OB(k);
        b.pre_t = 0.0; b.pre_q = 0.0;
        if (gam) {
            const M7Prim pk = P(k);
            const float dzk = DZ(k), dzm = (k > kts) ? DZ(k - 1) : 1.0f;
            const float pt_k = m7_pre(pk.el, pk.qkw, pk.gamt, k, kts, kte);
            const float pq_k = m7_pre(pk.el, pk.qkw, pk.gamq, k, kts, kte);
            float pt_n = 0.0f, pq_n = 0.0f;
            if (k + 1 < kte) {
                const M7Prim pn = P(k + 1);
                pt_n = m7_pre(pn.el, pn.qkw, pn.gamt, k + 1, kts, kte);
                pq_n = m7_pre(pn.el, pn.qkw, pn.gamq, k + 1, kts, kte);
            }
            const double tbm = (k > kts) ? TB(k - 1) : 0.0, qbm = (k > kts) ? QB(k - 1) : 0.0;
            b.pre_t = m7_diff_rev_gather(k, kts, kte, tbm, TB(k), dzm, dzk, pt_k, pt_n, dz_b);
            b.pre_q = m7_diff_rev_gather(k, kts, kte, qbm, QB(k), dzm, dzk, pq_k, pq_n, dz_b);
            if (k > kts) m7_rev_level(pk, b, tke_budget, o);
        } else if (k > kts) {
            m7_rev_level(P(k), b, tke_budget, o);
        }
        if (k > kts) ACC(k, o);
        if (dz_b != 0.0) ACCDZ(k, dz_b);
    }
}
