// SPDX-License-Identifier: Apache-2.0
// x20 MYNN-EDMF stage M10 "out": the reverse of everything MYNN does after f_mynn_tendencies/f_moisture_check.
// Forward routines covered (c6d08709 capture 70da4af7, MPAS driver order):
//   f_mynn_mix_chem       (mix_chem)               -- off in the hrrrv5 deck (mix_chem 0, nchem 0); not reversed here,
//                                                     the generated tape keeps it if a deck turns it on.
//   f_mynn_mix_scalars    (bl_mynn_mixscalars==1)  -- off in the hrrrv5 deck (0); same.
//   f_retrieve_exchange_coeffs (exch_m, exch_h)    -- frozen under --freeze closure (group 7): no reverse.
//   TKE budget block (bl_mynn_tkebudget==1)        -- off in the hrrrv5 deck (0).
//   f_mynnedmf_post_run   (specific -> mixing-ratio tendencies, qv/qc/qi/qs)  -- LIVE: hand reverse below.
//   driver unpack          (column -> launch arrays)                          -- copy routing, table M10_ROUTE.
// Shape: register-resident, level-local, recompute-not-load. Per level the reverse needs only the stage's true inputs
// (qv, qc, qi, qs at the launch, and the four specific-humidity tendencies M9 hands over) and delt; every forward
// intermediate (1+qv, sq, sqv, 1-sqv, rq) is recomputed bitwise with the captured float routines. No tape, no global
// round trip. The primal it recomputes is bitwise the forward's (same r6_* calls, same order).
// The derivative is the transpose of the generated tangent's own rule set (mw_ad_rules.inc: r6_div through mw_quot with
// w = (double)a/(double)b, r6_mul through mynn_ad_multiply with float operand values, passive delt), so the dot test
// against the unchanged TL entry grades it.
#pragma once

// fmask bits: which hydrometeor arms post_run runs (f_qc, f_qi, f_qs launch scalars).
#define M10_FQC 1u
#define M10_FQI 2u
#define M10_FQS 4u

// Jacobian of one level of f_mynnedmf_post_run plus its primal outputs. x index 0,1,2 = qc, qi, qs.
struct M10PostLevel {
    float outv, outx[3];                 // primal dqv, dqc, dqi, dqs out (bitwise the forward)
    double vq, vD;                       // d outv / d qv, d outv / d dqv_in
    double xX[3], xq[3], xDx[3], xD[3];  // d outx / d qx, / d qv, / d dqx_in, / d dqv_in
};

__device__ __forceinline__ M10PostLevel m10_post_level(float q, const float X[3], float D, const float Dx[3], float dt,
                                                       unsigned fmask) {
    M10PostLevel J;
    // qv arm: sq = qv/(1+qv); sqv = sq + dqv*delt; rq = sqv/(1-sqv); dqv = (rq - qv)/delt
    const float a1 = r6_add(1.f, q);
    const float sq = r6_div(q, a1);
    const float sqv = r6_add(sq, r6_mul(D, dt));
    const float s1 = r6_sub(1.f, sqv);
    const float rq = r6_div(sqv, s1);
    J.outv = r6_div(r6_sub(rq, q), dt);
    const double A1 = (double)a1, S1 = (double)s1, DT = (double)dt;
    const double w1 = (double)q / A1;    // r6_div(qv, a1) rule weight
    const double w2 = (double)sqv / S1;  // r6_div(sqv, s1) rule weight
    const double Sq = (1.0 - w1) / A1;   // d sqv / d qv   (d sq = (dq - dq*w1)/a1)
    const double g = (1.0 + w2) / S1;    // d rq / d sqv   (d rq = (dsqv - (-dsqv)*w2)/s1)
    J.vq = (g * Sq - 1.0) / DT;
    J.vD = g * DT / DT;
    // hydrometeor arms: sq = X/(1+qv); sqx = sq + dqx*delt; rq = sqx*(1+sqv); dqx = (rq - X)/delt
    const float b1 = r6_add(1.f, sqv);
    const double B1 = (double)b1;
#pragma unroll
    for (int x = 0; x < 3; ++x) {
        if (!(fmask & (1u << x))) { J.outx[x] = Dx[x]; J.xX[x] = J.xq[x] = J.xD[x] = 0.0; J.xDx[x] = 1.0; continue; }
        const float sqx0 = r6_div(X[x], a1);
        const float sqx = r6_add(sqx0, r6_mul(Dx[x], dt));
        const float rqx = r6_mul(sqx, b1);
        J.outx[x] = r6_div(r6_sub(rqx, X[x]), dt);
        const double wx = (double)X[x] / A1;
        J.xX[x] = (B1 / A1 - 1.0) / DT;
        J.xq[x] = (B1 * (-wx / A1) + (double)sqx * Sq) / DT;
        J.xDx[x] = B1 * DT / DT;
        J.xD[x] = (double)sqx * DT / DT;
    }
    return J;
}

// Reverse of one level: accumulate J^T * (ob_v, ob_x) into the input cotangents.
__device__ __forceinline__ void m10_post_level_rev(const M10PostLevel& J, unsigned fmask, double ob_v, const double ob_x[3],
                                                   double& bq, double bX[3], double& bD, double bDx[3]) {
    bq += J.vq * ob_v;
    bD += J.vD * ob_v;
#pragma unroll
    for (int x = 0; x < 3; ++x) {
        if (!(fmask & (1u << x))) { bDx[x] += ob_x[x]; continue; }
        bX[x] += J.xX[x] * ob_x[x];
        bq += J.xq[x] * ob_x[x];
        bDx[x] += J.xDx[x] * ob_x[x];
        bD += J.xD[x] * ob_x[x];
    }
}

// The fused-kernel stage function: one column, levels 0..nz-1, each array a level-strided view (stride ls; in the
// fused kernel these are registers/shared rows, in a standalone launch the [slot][column] arrays with ls = ncol).
// Inputs : qv, qx[3] (launch values), dqv_in, dqx_in[3] (M9's specific tendencies, recomputed or handed over in
//          registers), the output cotangents ob_v, ob_x[3] (rqvblten, rqcblten, rqiblten, rqsblten bars).
// Outputs: bar_qv, bar_qx[3] += (accumulated: the launch arrays are also read upstream); bar_dqv_in, bar_dqx_in[3] =
//          (written: M9's output cotangents for the four tendencies).
__device__ __forceinline__ void m10_out_rev_column(int nz, int ls, float delt, unsigned fmask,
        const float* qv, const float* const qx[3], const float* dqv_in, const float* const dqx_in[3],
        const double* ob_v, const double* const ob_x[3],
        double* bar_qv, double* const bar_qx[3], double* bar_dqv_in, double* const bar_dqx_in[3]) {
    for (int k = 0; k < nz; ++k) {
        const int o = k * ls;
        float X[3], Dx[3]; double obx[3], bX[3] = {0.0, 0.0, 0.0}, bDx[3] = {0.0, 0.0, 0.0};
#pragma unroll
        for (int x = 0; x < 3; ++x) { X[x] = qx[x][o]; Dx[x] = dqx_in[x][o]; obx[x] = ob_x[x][o]; }
        const M10PostLevel J = m10_post_level(qv[o], X, dqv_in[o], Dx, delt, fmask);
        double bq = 0.0, bD = 0.0;
        m10_post_level_rev(J, fmask, ob_v[o], obx, bq, bX, bD, bDx);
        bar_qv[o] += bq; bar_dqv_in[o] = bD;
#pragma unroll
        for (int x = 0; x < 3; ++x) { bar_qx[x][o] += bX[x]; bar_dqx_in[x][o] = bDx[x]; }
    }
}

// Driver unpack routing (the reverse of the copies column -> launch array). COPY: the launch array's output cotangent
// becomes the cotangent of the column array M9/M8/M7/M3 produced (a move, no arithmetic). FROZEN: the value is frozen
// under --freeze closure, its output cotangent is dropped. POST: through m10_out_rev_column. Arrays the launch does not
// write keep the identity part of J = I + K (their xbar includes ybar), which the fused kernel's epilogue adds.
enum M10Route : unsigned char { M10_COPY = 0, M10_FROZEN = 1, M10_POST = 2 };
struct M10RouteRow { const char* launch; const char* column; M10Route r; };
static constexpr M10RouteRow M10_ROUTE[] = {
    {"pblh", "pblh1", M10_FROZEN}, {"cldfra_bl", "cldfrabl1", M10_COPY}, {"qc_bl", "qcbl1", M10_COPY},
    {"qi_bl", "qibl1", M10_COPY}, {"el_pbl", "elpbl1", M10_FROZEN}, {"qke", "qke1", M10_COPY},
    {"qke_adv", "qkeadv1", M10_COPY}, {"cov", "cov1", M10_COPY}, {"tsq", "tsq1", M10_COPY}, {"qsq", "qsq1", M10_COPY},
    {"sh3d", "sh1", M10_FROZEN}, {"sm3d", "sm1", M10_FROZEN}, {"rublten", "rublten1", M10_COPY},
    {"rvblten", "rvblten1", M10_COPY}, {"rthblten", "rthblten1", M10_COPY}, {"rqvblten", "rqvblten1", M10_POST},
    {"rqcblten", "rqcblten1", M10_POST}, {"rqiblten", "rqiblten1", M10_POST}, {"rqsblten", "rqsblten1", M10_POST},
    {"rqozblten", "rqozblten1", M10_COPY}, {"rncblten", "rncblten1", M10_COPY}, {"rniblten", "rniblten1", M10_COPY},
    {"rnifablten", "rnifablten1", M10_COPY}, {"rnwfablten", "rnwfablten1", M10_COPY}, {"rnbcablten", "rnbcablten1", M10_COPY},
    {"edmf_a", "edmfa1", M10_FROZEN}, {"edmf_w", "edmfw1", M10_FROZEN}, {"edmf_qt", "edmfqt1", M10_FROZEN},
    {"edmf_thl", "edmfthl1", M10_FROZEN}, {"edmf_ent", "edmfent1", M10_FROZEN}, {"edmf_qc", "edmfqc1", M10_FROZEN},
    {"sub_thl", "subthl1", M10_FROZEN}, {"sub_sqv", "subsqv1", M10_FROZEN}, {"det_thl", "detthl1", M10_FROZEN},
    {"det_sqv", "detsqv1", M10_FROZEN}, {"maxwidth", "maxwidth1", M10_FROZEN}, {"maxmf", "maxmf1", M10_FROZEN},
    {"ztop_plume", "ztopplume1", M10_FROZEN}, {"excess_h", "excessh1", M10_FROZEN}, {"excess_q", "excessq1", M10_FROZEN},
    {"maxwidth_dd", "maxwidth_dd1", M10_FROZEN}, {"maxmf_dd", "maxmf_dd1", M10_FROZEN},
    {"maxtkeprod", "maxtkeprod1", M10_FROZEN}, {"cldtop_cooling", "cldtop_cooling1", M10_FROZEN},
    {"ent_eff", "ent_eff1", M10_FROZEN}, {"exch_h", "exchh1", M10_FROZEN}, {"exch_m", "exchm1", M10_FROZEN},
};

#ifdef M10_TAPE_BRIDGE
// Swap-in bridge for the existing generated tape (proof vehicle only, not the fused target): f_mynnedmf_post_run for
// the tape carrier computes the primal with m10_post_level (bitwise) and records each output as at most two fused
// linear nodes plus one sum whose weights are J above, instead of the generated per-operation nodes. The reverse sweep
// then applies exactly m10_post_level_rev. The count pass sees the same pushes, so ring sizing stays consistent.
__device__ __forceinline__ int m10_lin2(MynnCtxRef c, int a, double wa, int b, double wb) {
    if (a >= 0 && b >= 0) return c->push2(a, wa, b, wb, 1);  // kind 1: bars[a] += wa*bar, bars[b] += wb*bar
    if (a >= 0) return c->push(a, wa, -1, 0.0);
    if (b >= 0) return c->push(b, wb, -1, 0.0);
    return -1;
}
__device__ __forceinline__ int m10_sum(MynnCtxRef c, int a, int b) {
    if (a >= 0 && b >= 0) return c->push(a, 1.0, b, 1.0);
    return a >= 0 ? a : b;
}
template<class ARR>
__device__ __forceinline__ void m10_post_run_tape(int kte, bool f_qc, bool f_qi, bool f_qs, MynnTapeF delt,
        ARR qv, ARR qc, ARR qi, ARR qs, ARR dqv, ARR dqc, ARR dqi, ARR dqs) {
    const unsigned fm = (f_qc ? M10_FQC : 0u) | (f_qi ? M10_FQI : 0u) | (f_qs ? M10_FQS : 0u);
    ARR outs[3] = {dqc, dqi, dqs};
    for (int k = 1; k <= kte; ++k) {
        const MynnTapeF Q = qv(k), Dv = dqv(k);
        const MynnTapeF Xc[3] = {qc(k), qi(k), qs(k)}, Dc[3] = {dqc(k), dqi(k), dqs(k)};
        const float X[3] = {Xc[0].v, Xc[1].v, Xc[2].v}, Dx[3] = {Dc[0].v, Dc[1].v, Dc[2].v};
        const M10PostLevel J = m10_post_level(Q.v, X, Dv.v, Dx, delt.v, fm);
        MynnCtxRef c = mynn_ad_context(Q.d.context, Dv.d.context);
#pragma unroll
        for (int x = 0; x < 3; ++x) c = mynn_ad_context(c, mynn_ad_context(Xc[x].d.context, Dc[x].d.context));
        const int iq = Q.d.id, iD = Dv.d.id;
        {
            const int id = c ? m10_lin2(c, iq, J.vq, iD, J.vD) : -1;
            dqv(k) = MynnTapeF(J.outv, MynnLinear(id, c, Q.d.depends || Dv.d.depends));
        }
#pragma unroll
        for (int x = 0; x < 3; ++x) {
            if (!(fm & (1u << x))) continue;
            int id = -1;
            if (c) {
                const int a = m10_lin2(c, Xc[x].d.id, J.xX[x], Dc[x].d.id, J.xDx[x]);
                const int b = m10_lin2(c, iq, J.xq[x], iD, J.xD[x]);
                id = m10_sum(c, a, b);
            }
            outs[x](k) = MynnTapeF(J.outx[x], MynnLinear(id, c, Xc[x].d.depends || Dc[x].d.depends || Q.d.depends || Dv.d.depends));
        }
    }
}
#endif
