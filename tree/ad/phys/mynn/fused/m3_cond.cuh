// SPDX-License-Identifier: Apache-2.0
// x20 MYNN stage M3 (sgs-cloud): register-resident reverse of f_mym_condensation, bl_mynn_cloudpdf = 2 (the hrrrv5
// deck; RRFS 44c0849 module_bl_mynnedmf.F90:3769-4158, captured source 70da4af7). No tape: each level recomputes its
// forward from its own inputs (zw dz th exner p qw qsq tsq cov qi qs qc pattern) in float with the captured
// primitives (r6_*, mynn_*), so the primal is bitwise the forward, then runs the hand reverse of that straight line
// in double. Derivative rules are the generator rules (mw_ad_rules.inc) term for term: product weights are the float
// operand values, quotient weight (double)a/(double)b, min/max/clamps follow the executed arm with the forward tie
// rule (mynn_max(a,b) = a>b?a:b, mynn_min(a,b) = a<b?a:b), a constant arm carries no derivative.
// Zero derivative by construction: pblh (frozen under --freeze closure), xland and closure (control only), k_tropo
// (integer control). Outputs per level: qc_bl, qi_bl, cldfra_bl, vt, vq, sgm; levels kts..kte-1, then the kte tail.
// Include after the captured MYNN prefix (primitives); needs nothing from MynnColumn.
#pragma once

namespace m3c {
constexpr float r_d = 287.f;
constexpr float r_v = 461.6f;
constexpr float cp = 1004.5f;
constexpr float cpv = ((4.f) * (r_v));
constexpr float cliq = 4190.f;
constexpr float cice = 2106.f;
constexpr float xlv = 2.50e6f;
constexpr float xlf = 3.50e5f;
constexpr float xls = ((xlv) + (xlf));
constexpr float p608 = ((((r_v) / (r_d))) - (1.f));
constexpr float ep_2 = ((r_d) / (r_v));
constexpr float t0c = 273.15f;
constexpr float tref = 300.0f;
constexpr float tice = 238.0f;
constexpr float tv0 = ((p608) * (tref));
constexpr float xlvcp = ((xlv) / (cp));
constexpr float zero = 0.0f, one = 1.0f, two = 2.0f, three = 3.0f, five = 5.0f, ten = 10.0f, hundred = 100.0f;
constexpr float p1 = 0.1f, p2 = 0.2f, p5 = 0.5f;
constexpr float tliq = 269.f;
constexpr float qlim_sfc = 0.008f, qlim_pbl = 0.020f, qlim_trp = 0.025f;
constexpr float clim_sfc = 0.010f, clim_pbl = 0.020f, clim_trp = 0.030f;
constexpr float rhcrit = 0.83f, rhmax = 1.10f;
// liquid (j) and ice (k) saturation-pressure polynomials of f_esat_blend / f_qsat_blend
__device__ __forceinline__ float cJ(int i) {
    switch (i) { case 0: return .611583699e03f; case 1: return .444606896e02f; case 2: return .143177157e01f;
                 case 3: return .264224321e-1f; case 4: return .299291081e-3f; case 5: return .203154182e-5f;
                 case 6: return .702620698e-8f; case 7: return .379534310e-11f; default: return (-(.321582393e-13f)); }
}
__device__ __forceinline__ float cK(int i) {
    switch (i) { case 0: return .609868993e03f; case 1: return .499320233e02f; case 2: return .184672631e01f;
                 case 3: return .402737184e-1f; case 4: return .565392987e-3f; case 5: return .521693933e-5f;
                 case 6: return .307839583e-7f; case 7: return .105785160e-9f; default: return .161444444e-12f; }
}
} // namespace m3c

// Horner polynomial: es = c0 + xc*(c1 + xc*(... + xc*c8)), the captured evaluation order.
// REV: a_xc += (d es / d xc) a_es through the reverse of the r6_mul/r6_add chain.
template<bool ICE, bool REV> __device__ __forceinline__ float m3_horner(float xc, double a_es, double& a_xc) {
    float h[9];
    h[8] = ICE ? m3c::cK(8) : m3c::cJ(8);
#pragma unroll
    for (int i = 7; i >= 0; --i) h[i] = r6_add(ICE ? m3c::cK(i) : m3c::cJ(i), r6_mul(xc, h[i + 1]));
    if (REV) {
        double ah = a_es;
#pragma unroll
        for (int i = 0; i < 8; ++i) { a_xc += ah * (double)h[i + 1]; ah = ah * (double)xc; }
    }
    return h[0];
}

// One arm of f_qsat_blend from its saturation pressure: es2 = min(es, 0.15 p); q = 0.622 es2 / max(p - es2, 1e-5).
template<bool REV> __device__ __forceinline__ float m3_q_from_es(float es, float p, double a_q, double& a_es, double& a_p) {
    const float p15 = r6_mul(p, 0.15f);
    const bool es_live = es < p15;
    const float es2 = mynn_min((float)(es), (float)(p15));
    const float num = r6_mul(0.622f, es2);
    const float dd = r6_sub(p, es2);
    const bool den_live = dd > 1e-5f;
    const float den = mynn_max((float)(dd), (float)(1e-5f));
    const float q = r6_div(num, den);
    if (REV) {
        const double a_num = a_q / (double)den;
        const double a_den = -(a_q * ((double)num / (double)den)) / (double)den;
        double a_es2 = a_num * 0.622f;
        if (den_live) { a_p += a_den; a_es2 -= a_den; }
        if (es_live) a_es += a_es2; else a_p += a_es2 * 0.15f;
    }
    return q;
}

template<bool REV> __device__ __forceinline__ float m3_qsat_blend(float t, float p, double a_q, double& a_t, double& a_p) {
    using namespace m3c;
    const float tm = r6_sub(t, t0c);
    const bool xc_live = !(r6_neg(80.f) > tm);
    const float xc = mynn_max((float)(r6_neg(80.f)), (float)(tm));
    double a_xc = 0.0, dmy = 0.0;
    float q;
    if (t >= r6_sub(t0c, 6.f)) {
        double a_es = 0.0;
        const float es = m3_horner<false, false>(xc, 0.0, dmy);
        q = m3_q_from_es<REV>(es, p, a_q, a_es, a_p);
        if (REV) m3_horner<false, true>(xc, a_es, a_xc);
    } else if (t <= tice) {
        double a_es = 0.0;
        const float es = m3_horner<true, false>(xc, 0.0, dmy);
        q = m3_q_from_es<REV>(es, p, a_q, a_es, a_p);
        if (REV) m3_horner<true, true>(xc, a_es, a_xc);
    } else {
        const float esl = m3_horner<false, false>(xc, 0.0, dmy);
        const float esi = m3_horner<true, false>(xc, 0.0, dmy);
        const float rslf = m3_q_from_es<false>(esl, p, 0.0, dmy, dmy);
        const float rsif = m3_q_from_es<false>(esi, p, 0.0, dmy, dmy);
        const float denc = r6_sub(r6_sub(t0c, 6.f), tice);
        const float chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), denc);
        const float omc = r6_sub(1.f, chi);
        q = r6_add(r6_mul(omc, rslf), r6_mul(chi, rsif));
        if (REV) {
            double a_esl = 0.0, a_esi = 0.0;
            const double a_rslf = a_q * (double)omc, a_rsif = a_q * (double)chi;
            const double a_chi = -(a_q * (double)rslf) + a_q * (double)rsif;
            a_t += -a_chi / (double)denc;
            m3_q_from_es<true>(esl, p, a_rslf, a_esl, a_p);
            m3_q_from_es<true>(esi, p, a_rsif, a_esi, a_p);
            m3_horner<false, true>(xc, a_esl, a_xc);
            m3_horner<true, true>(xc, a_esi, a_xc);
        }
    }
    if (REV && xc_live) a_t += a_xc;
    return q;
}

template<bool REV> __device__ __forceinline__ float m3_xl_blend(float t, double a_xl, double& a_t) {
    using namespace m3c;
    const float tm = r6_sub(t, t0c);
    if (t >= t0c) {
        const float s = r6_sub(cpv, cliq);
        if (REV) a_t += a_xl * (double)s;
        return r6_add(xlv, r6_mul(s, tm));
    } else if (t <= tice) {
        const float s = r6_sub(cpv, cice);
        if (REV) a_t += a_xl * (double)s;
        return r6_add(xls, r6_mul(s, tm));
    }
    const float sv = r6_sub(cpv, cliq), si = r6_sub(cpv, cice);
    const float xlvt = r6_add(xlv, r6_mul(sv, tm));
    const float xlst = r6_add(xls, r6_mul(si, tm));
    const float denc = r6_sub(t0c, tice);
    const float chi = r6_div(r6_sub(t0c, t), denc);
    const float omc = r6_sub(1.f, chi);
    if (REV) {
        const double a_xlvt = a_xl * (double)omc, a_xlst = a_xl * (double)chi;
        const double a_chi = -(a_xl * (double)xlvt) + a_xl * (double)xlst;
        a_t += a_xlvt * (double)sv + a_xlst * (double)si - a_chi / (double)denc;
    }
    return r6_add(r6_mul(omc, xlvt), r6_mul(chi, xlst));
}

// Level inputs (primal, float) and their cotangents (double); output values and cotangents.
struct M3Lev  { float zw, dz, th, ex, p, qw, qsq, tsq, cov, qi, qs, qc, pat; };
struct M3IBar { double zw, dz, th, ex, p, qw, qsq, tsq, cov, qi, qs, qc, pat; };
struct M3Out  { float qc_bl, qi_bl, cf, vt, vq, sgm; };
struct M3OBar { double qc_bl, qi_bl, cf, vt, vq, sgm; };
struct M3Col  { float pblh1, xland, closure; int spp_pbl, k_tropo; };

// min(max(v, 0) / d, 1), the blending-weight ramp; live = the derivative passes (v > 0 and the quotient < 1).
__device__ __forceinline__ float m3_ramp(float v, float d, bool& live) {
    const bool l1 = v > m3c::zero;
    const float m = mynn_max((float)(v), (float)(m3c::zero));
    const float w = r6_div(m, d);
    live = l1 && (w < m3c::one);
    return mynn_min((float)(w), (float)(m3c::one));
}

// One level of f_mym_condensation, case 2. Forward always (y = outputs, bitwise); REV: g += J^T yb.
template<bool REV> __device__ __forceinline__ void m3c2_level(const M3Lev& x, const M3Col& c, int k, M3Out& y,
                                                              const M3OBar& yb, M3IBar& g) {
    using namespace m3c;
    const float pblh2 = mynn_max((float)(c.pblh1), (float)(ten));
    const float zagl = r6_add(x.zw, r6_mul(p5, x.dz));
    const float t = r6_mul(x.th, x.ex);
    double dmy = 0.0;
    const float xl = m3_xl_blend<false>(t, 0.0, dmy);
    const float qsat = m3_qsat_blend<false>(t, x.p, 0.0, dmy, dmy);
    // rh
    const bool qsd_live = qsat > 1e-10f;
    const float qsd = mynn_max((float)(qsat), (float)(1e-10f));
    const float rhq = r6_div(x.qw, qsd);
    const bool rh_in = rhq < rhmax;
    const float rh_m = mynn_min((float)(rhq), (float)(rhmax));
    const bool rh_out = rh_m > 0.001f;
    const float rh = mynn_max((float)(rh_m), (float)(0.001f));
    const bool rh_live = rh_in && rh_out;
    // dqsl, bet, rsl, a, b
    const float t2 = mynn_ipow<2>(t);
    const float dq_n1 = r6_mul(qsat, ep_2);
    const float dq_n = r6_mul(dq_n1, xlv);
    const float dq_d = r6_mul(r_d, t2);
    const float dqsl = r6_div(dq_n, dq_d);
    const float bet = r6_mul(dqsl, x.ex);
    const float rs_n = r6_mul(xl, qsat);
    const float rs_d = r6_mul(r_v, t2);
    const float rsl = r6_div(rs_n, rs_d);
    const float cpm = r6_add(cp, r6_mul(x.qw, cpv));
    const float xlrsl = r6_mul(xl, rsl);
    const float q2 = r6_div(xlrsl, cpm);
    const float den1 = r6_add(one, q2);
    const float av = r6_div(one, den1);
    const float bv = r6_mul(av, rsl);
    // perturbed qw, qmq
    const float sppf = ((float)(c.spp_pbl));
    const float qp1 = r6_mul(x.qw, p5);
    const float qp2 = r6_mul(qp1, x.pat);
    const float qp3 = r6_mul(qp2, sppf);
    const float qw_pert = r6_add(x.qw, qp3);
    const float qmq = r6_sub(qw_pert, qsat);
    // variances (double, as the forward)
    const bool r3_live = !(zero > x.qsq), t3_live = !(zero > x.tsq);
    const double r3a = mynn_max((float)(zero), (float)(x.qsq));
    const double t3 = mynn_max((float)(zero), (float)(x.tsq));
    double c3 = x.cov;
    const double c3abs = mynn_abs(c3);
    const double c3sq_ = mynn_sqrt(mynn_dmul(t3, r3a));
    const bool c3_abs_arm = c3abs < c3sq_;
    const double c3m = mynn_min((double)(c3abs), (double)(c3sq_));
    c3 = mynn_sign(c3m, c3);
    const bool c3_same = signbit(c3) == signbit(c3m);
    const bool c3_neg = (double)x.cov < 0.0 || ((double)x.cov == 0.0 && signbit((double)x.cov));
    const int cl = (c.closure < 2.7f) ? 0 : ((c.closure >= 2.7f && c.closure < 3.0f) ? 1 : 2);
    const float bet2 = mynn_ipow<2>(bet);
    const float tbet = r6_mul(two, bet);
    double r3 = r3a;
    if (cl == 1) r3 = mynn_dadd(r3a, mynn_dmul(bet2, t3));
    else if (cl == 2) r3 = mynn_dsub(mynn_dadd(r3a, mynn_dmul(bet2, t3)), mynn_dmul(tbet, c3));
    const float s0 = mynn_sqrt(((float)(r3)));
    const bool sA_live = s0 > 1e-13f;
    const float sA = mynn_max((float)(s0), (float)(1e-13f));
    const float qwp5 = r6_mul(x.qw, p5);
    const bool sB_a = sA < qwp5;
    const float sB = mynn_min((float)(sA), (float)(qwp5));
    bool wdz_live; const float wdz = m3_ramp(r6_sub(x.dz, 100.f), 500.f, wdz_live);
    const float sBp2 = r6_mul(sB, p2);
    const float sBw = r6_mul(sBp2, wdz);
    const float sgm = r6_add(sB, sBw);
    // clim
    bool w5_live; const float w5 = m3_ramp(r6_sub(zagl, r6_add(pblh2, 10.f)), 250.f, w5_live);
    const float clim1 = r6_add(r6_mul(clim_pbl, r6_sub(one, w5)), r6_mul(clim_trp, w5));
    const float zsl1 = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(50.f))), (float)(150.f));
    bool w6_live; const float w6 = m3_ramp(r6_sub(zagl, zsl1), 150.f, w6_live);
    const float clim2 = r6_add(r6_mul(clim_sfc, r6_sub(one, w6)), r6_mul(clim1, w6));
    const float qwcl = r6_mul(x.qw, clim2);
    const bool sc1_a = sgm > qwcl;
    const float sc1 = mynn_max((float)(sgm), (float)(qwcl));
    const bool sc2_live = sc1 > 1e-13f;
    const float sc2 = mynn_max((float)(sc1), (float)(1e-13f));
    const bool qmq_pos = qmq >= zero;
    const float qw02 = r6_mul(0.02f, x.qw);
    const bool sc3_a = qmq_pos && (qw02 > sc2);
    const float sc3 = qmq_pos ? mynn_max((float)(qw02), (float)(sc2)) : sc2;
    const float q1_0 = r6_div(qmq, sc3);
    // rh adjustments above the PBL
    bool wa_live; const float wta = m3_ramp(r6_sub(zagl, pblh2), 300.f, wa_live);
    const float den3 = r6_sub(one, rhcrit);
    const float qis = r6_add(x.qi, x.qs);
    const bool blk_i = (qis > 1.e-10f) && (zagl > pblh2);
    float q1_1 = q1_0, mxi = 0.f, ci = 0.f;
    bool mxi_live = false, rai_live = false, rbi_rh = false, q1i_rh = false;
    if (blk_i) {
        const float lgi = r6_add(ten, mynn_log10(qis));
        mxi_live = lgi > zero; mxi = mynn_max((float)(lgi), (float)(zero));
        ci = r6_mul(wta, 0.037f);
        const float si_ = r6_add(rhcrit, r6_mul(ci, mxi));
        rai_live = si_ < rhmax; const float rai = mynn_min((float)(si_), (float)(rhmax));
        rbi_rh = rh > rai; const float rbi = mynn_max((float)(rh), (float)(rai));
        const float q1ri = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rbi, rhcrit)), den3));
        q1i_rh = q1ri > q1_0; q1_1 = mynn_max((float)(q1ri), (float)(q1_0));
    }
    const bool blk_c = (x.qc > 1.e-5f) && (zagl > pblh2);
    float q1_2 = q1_1, mxc = 0.f, cc = 0.f;
    bool mxc_live = false, rac_live = false, rbc_rh = false, q1c_rh = false;
    if (blk_c) {
        const float lgc = r6_add(five, mynn_log10(x.qc));
        mxc_live = lgc > zero; mxc = mynn_max((float)(lgc), (float)(zero));
        cc = r6_mul(wta, 0.07f);
        const float sc_ = r6_add(rhcrit, r6_mul(cc, mxc));
        rac_live = sc_ < rhmax; const float rac = mynn_min((float)(sc_), (float)(rhmax));
        rbc_rh = rh > rac; const float rbc = mynn_max((float)(rh), (float)(rac));
        const float q1rc = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rbc, rhcrit)), den3));
        q1c_rh = q1rc > q1_1; q1_2 = mynn_max((float)(q1rc), (float)(q1_1));
    }
    const float q1 = q1_2;
    // cloud fraction from q1 (cldfra_rh is zero in this generation; its tanh terms feed nothing)
    const float u2 = r6_div(r6_sub(zagl, r6_sub(c.pblh1, 100.f)), 200.f);
    const float u2m = mynn_max((float)(u2), (float)(zero));
    const bool w2b_live = (u2 > zero) && (u2m < one);
    const float wtb = mynn_min((float)(u2m), (float)(one));
    const float xa0 = r6_mul(3.6f, r6_add(q1, 0.05f));
    const float sa0 = r6_add(p5, r6_mul(0.35f, mynn_atan(xa0)));
    const float ma0 = mynn_min((float)(sa0), (float)(one));
    const bool cq0_live = (sa0 < one) && !(zero > ma0);
    const float cq0 = mynn_max((float)(zero), (float)(ma0));
    const float xa1 = r6_mul(1.6f, r6_add(q1, 0.55f));
    const float sa1 = r6_add(p5, r6_mul(0.39f, mynn_atan(xa1)));
    const float ma1 = mynn_min((float)(sa1), (float)(one));
    const bool cq1_live = (sa1 < one) && !(zero > ma1);
    const float cq1 = mynn_max((float)(zero), (float)(ma1));
    const float omwb = r6_sub(one, wtb);
    const float cfq = r6_add(r6_mul(cq0, omwb), r6_mul(cq1, wtb));
    const bool cf_live0 = cfq > zero;              // max(cldfra_qsq, cldfra_rh = 0)
    float cf = mynn_max((float)(cfq), (float)(zero));
    // qlim
    bool w3_live; const float w3 = m3_ramp(r6_sub(zagl, r6_add(pblh2, ten)), 300.f, w3_live);
    const float qlim1 = r6_add(r6_mul(qlim_pbl, r6_sub(one, w3)), r6_mul(qlim_trp, w3));
    bool w4_live; const float w4 = m3_ramp(r6_sub(zagl, zsl1), 200.f, w4_live);
    const float qlim2 = r6_add(r6_mul(qlim_sfc, r6_sub(one, w4)), r6_mul(qlim1, w4));
    const float qwql = r6_mul(x.qw, qlim2);
    const bool sq_a = sgm > qwql;
    const float sgmq = mynn_max((float)(sgm), (float)(qwql));
    const float qw025 = r6_mul(0.025f, x.qw);
    const bool mq_a = sgmq < qw025;
    const float mq = mynn_min((float)(sgmq), (float)(qw025));
    float qlw = r6_mul(mq, cf);
    float qli = r6_mul(mq, cf);
    const float cf_pre = cf;
    const bool cf_small = cf < 0.001f;
    if (cf_small) { qli = zero; qlw = zero; cf = zero; }
    const float dl = r6_sub(tliq, tice);
    const float u_l = r6_div(r6_sub(t, tice), dl);
    const float u_lm = mynn_max((float)(u_l), (float)(zero));
    const bool liq_live = (u_l > zero) && (u_lm < one);
    const float liq = mynn_min((float)(u_lm), (float)(one));
    const float oml = r6_sub(one, liq);
    float qc_bl = r6_mul(liq, qlw);
    float qi_bl = r6_mul(oml, qli);
    const bool tropo = k >= c.k_tropo;
    if (tropo) { cf = zero; qc_bl = zero; qi_bl = zero; }
    // vt, vq
    const float q1c = (r6_sub(c.xland, 1.5f) >= zero) ? r6_neg(2.5f) : r6_neg(2.0f);
    const bool q1k_live = q1 > q1c;
    const float q1k = mynn_max((float)(q1), (float)(q1c));
    int fb; float fng, fe = 0.f, f0 = 0.f;
    if (q1k >= one) { fb = 0; fng = one; }
    else if (q1k >= r6_neg(1.7f) && q1k < one) { fb = 1; fe = mynn_exp(r6_neg(r6_mul(0.4f, r6_sub(q1k, one)))); fng = fe; }
    else if (q1k >= r6_neg(2.5f) && q1k < r6_neg(1.7f)) { fb = 2; fe = mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1k, 1.7f)))); fng = r6_add(three, fe); }
    else { fb = 3; fe = mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1k, 2.5f)))); f0 = r6_add(23.9f, fe); fng = mynn_min((float)(f0), (float)(60.f)); }
    const bool cfm_live = cf < 0.6f;
    const float cfm0 = mynn_min((float)(cf), (float)(0.6f));
    const float zsl3 = mynn_min((float)(mynn_max((float)(r6_mul(p1, pblh2)), (float)(25.f))), (float)(hundred));
    const float wzr = r6_div(zagl, zsl3);
    const bool wz_live = wzr < one;
    const float wz = mynn_min((float)(wzr), (float)(one));
    const float cfmax = r6_mul(cfm0, wz);
    const float bt = r6_mul(bv, t);
    const float bb = r6_div(bt, x.th);
    const float qww = r6_add(one, r6_mul(p608, x.qw));
    const float alpha = r6_mul(p608, x.th);
    const float r1 = r6_div(x.th, t);
    const float r2 = r6_div(xl, cp);
    const float beta = r6_sub(r6_mul(r1, r2), r6_mul(1.61f, x.th));
    const float cb = r6_mul(cfmax, beta);
    const float cbb = r6_mul(cb, bb);
    const float cbbf = r6_mul(cbb, fng);
    const float cba = r6_mul(cb, av);
    const float cbaf = r6_mul(cba, fng);
    y.vt = r6_sub(r6_sub(qww, cbbf), one);
    y.vq = r6_sub(r6_add(alpha, cbaf), tv0);
    y.qc_bl = qc_bl; y.qi_bl = qi_bl; y.cf = cf; y.sgm = sgm;
    if (!REV) return;

    // ------------------------------------------------------------------ reverse (double)
    double a_qw = 0.0, a_th = 0.0, a_t = 0.0, a_zagl = 0.0, a_xl = 0.0, a_qsat = 0.0, a_av = 0.0, a_bv = 0.0;
    double a_cf = 0.0, a_q1 = 0.0, a_sgm = yb.sgm, a_rh = 0.0, a_p = 0.0, a_ex = 0.0, a_dz = 0.0;
    // vq = (alpha + ((cfmax beta) a) fng) - tv0; vt = (qww - ((cfmax beta) bb) fng) - 1
    const double a_cbaf = yb.vq, a_alpha = yb.vq, a_qww = yb.vt, a_cbbf = -yb.vt;
    const double a_cbb = a_cbbf * (double)fng, a_cba = a_cbaf * (double)fng;
    const double a_fng = a_cbbf * (double)cbb + a_cbaf * (double)cba;
    a_av += a_cba * (double)cb;
    const double a_cb = a_cba * (double)av + a_cbb * (double)bb;
    const double a_bb = a_cbb * (double)cb;
    const double a_cfmax = a_cb * (double)beta, a_beta = a_cb * (double)cfmax;
    const double a_r1 = a_beta * (double)r2, a_r2 = a_beta * (double)r1;
    a_th += -(a_beta * 1.61f);
    a_th += a_r1 / (double)t; a_t += -(a_r1 * ((double)x.th / (double)t)) / (double)t;
    a_xl += a_r2 / (double)cp;
    a_th += (double)p608 * a_alpha;
    a_qw += (double)p608 * a_qww;
    const double a_bt = a_bb / (double)x.th;
    a_th += -(a_bb * ((double)bt / (double)x.th)) / (double)x.th;
    a_bv += a_bt * (double)t; a_t += a_bt * (double)bv;
    const double a_cfm0 = a_cfmax * (double)wz;
    if (wz_live) a_zagl += (a_cfmax * (double)cfm0) / (double)zsl3;
    if (cfm_live) a_cf += a_cfm0;
    // fng (branch of q1k)
    double a_q1k = 0.0;
    if (fb == 1) a_q1k = -(a_fng * (double)fe) * 0.4f;
    else if (fb == 2) a_q1k = -(a_fng * (double)fe) * 3.8f;
    else if (fb == 3 && f0 < 60.f) a_q1k = -(a_fng * (double)fe) * 1.6f;
    if (q1k_live) a_q1 += a_q1k;
    // cldfra_bl, qc_bl, qi_bl (all constant zero at and above k_tropo)
    if (!tropo) {
        a_cf += yb.cf;
        const double a_liq = yb.qc_bl * (double)qlw - yb.qi_bl * (double)qli;
        const double a_qlw = yb.qc_bl * (double)liq, a_qli = yb.qi_bl * (double)oml;
        if (liq_live) a_t += a_liq / (double)dl;
        if (cf_small) { a_cf = 0.0; }
        else {
            const double a_mq = a_qlw * (double)cf_pre + a_qli * (double)cf_pre;
            a_cf += a_qlw * (double)mq + a_qli * (double)mq;
            // mq = min(sgmq, 0.025 qw); sgmq = max(sgm, qw qlim2)
            double a_sgmq = 0.0;
            if (mq_a) a_sgmq = a_mq; else a_qw += a_mq * 0.025f;
            if (sq_a) a_sgm += a_sgmq;
            else {
                a_qw += a_sgmq * (double)qlim2;
                const double a_qlim2 = a_sgmq * (double)x.qw;
                const double a_w4 = -(a_qlim2 * (double)qlim_sfc) + a_qlim2 * (double)qlim1;
                const double a_qlim1 = a_qlim2 * (double)w4;
                if (w4_live) a_zagl += a_w4 / 200.0;
                const double a_w3 = -(a_qlim1 * (double)qlim_pbl) + a_qlim1 * (double)qlim_trp;
                if (w3_live) a_zagl += a_w3 / 300.0;
            }
        }
    } else {
        a_cf = 0.0;   // cfmax read the zeroed cldfra_bl
    }
    // cf = max(cfq, 0); cfq = cq0 (1 - wtb) + cq1 wtb
    if (cf_live0 && a_cf != 0.0) {
        const double a_cq0 = a_cf * (double)omwb, a_cq1 = a_cf * (double)wtb;
        const double a_wtb = -(a_cf * (double)cq0) + a_cf * (double)cq1;
        if (w2b_live) a_zagl += a_wtb / 200.0;
        if (cq0_live) a_q1 += ((a_cq0 * 0.35f) / (1.0 + (double)xa0 * (double)xa0)) * 3.6f;
        if (cq1_live) a_q1 += ((a_cq1 * 0.39f) / (1.0 + (double)xa1 * (double)xa1)) * 1.6f;
    }
    // rh adjustments, reverse order (qc block, then qi block)
    double a_wta = 0.0;
    if (blk_c) {
        double a_q1rc = 0.0;
        if (q1c_rh) { a_q1rc = a_q1; a_q1 = 0.0; }
        const double a_rb = (a_q1rc * 3.0) / (double)den3;
        double a_ra = 0.0;
        if (rbc_rh) a_rh += a_rb; else a_ra = a_rb;
        if (rac_live) {
            a_wta += (a_ra * (double)mxc) * 0.07f;
            if (mxc_live) g.qc += (a_ra * (double)cc) / ((double)x.qc * 2.302585092994045684);
        }
    }
    if (blk_i) {
        double a_q1ri = 0.0;
        if (q1i_rh) { a_q1ri = a_q1; a_q1 = 0.0; }
        const double a_rb = (a_q1ri * 3.0) / (double)den3;
        double a_ra = 0.0;
        if (rbi_rh) a_rh += a_rb; else a_ra = a_rb;
        if (rai_live) {
            a_wta += (a_ra * (double)mxi) * 0.037f;
            if (mxi_live) { const double a_qis = (a_ra * (double)ci) / ((double)qis * 2.302585092994045684); g.qi += a_qis; g.qs += a_qis; }
        }
    }
    if (wa_live) a_zagl += a_wta / 300.0;
    // q1_0 = qmq / sgmc
    const double a_qmq = a_q1 / (double)sc3;
    double a_sc = -(a_q1 * ((double)qmq / (double)sc3)) / (double)sc3;
    if (sc3_a) { a_qw += a_sc * 0.02f; a_sc = 0.0; }
    if (!sc2_live) a_sc = 0.0;
    if (sc1_a) a_sgm += a_sc;
    else {
        a_qw += a_sc * (double)clim2;
        const double a_clim2 = a_sc * (double)x.qw;
        const double a_w6 = -(a_clim2 * (double)clim_sfc) + a_clim2 * (double)clim1;
        const double a_clim1 = a_clim2 * (double)w6;
        if (w6_live) a_zagl += a_w6 / 150.0;
        const double a_w5 = -(a_clim1 * (double)clim_pbl) + a_clim1 * (double)clim_trp;
        if (w5_live) a_zagl += a_w5 / 250.0;
    }
    // sgm = sB + (sB p2) wdz
    double a_sB = a_sgm;
    const double a_sBp2 = a_sgm * (double)wdz;
    if (wdz_live) a_dz += (a_sgm * (double)sBp2) / 500.0;
    a_sB += a_sBp2 * (double)p2;
    double a_sA = 0.0;
    if (sB_a) a_sA = a_sB; else a_qw += a_sB * (double)p5;
    double a_r3 = 0.0;
    if (sA_live) a_r3 = a_sA / (2.0 * (double)s0);
    double a_r3a = a_r3, a_t3 = 0.0, a_bet = 0.0, a_c3 = 0.0;
    if (cl >= 1) { a_t3 += a_r3 * (double)bet2; a_bet += (a_r3 * t3) * (2.0 * (double)bet); }
    if (cl == 2) { a_c3 += -(a_r3 * (double)tbet); a_bet += (-(a_r3 * c3)) * (double)two; }
    if (cl == 2 && a_c3 != 0.0) {
        const double a_m = c3_same ? a_c3 : -a_c3;
        if (c3_abs_arm) g.cov += c3_neg ? -a_m : a_m;
        else if (c3sq_ != 0.0) { const double a_tr = a_m / (2.0 * c3sq_); a_t3 += a_tr * r3a; a_r3a += a_tr * t3; }
    }
#ifdef M3_BREAK
    if (r3_live) g.qsq += a_r3a * (1.0 + M3_BREAK);   // test-only: proves the dot test sees an M3 weight error
#else
    if (r3_live) g.qsq += a_r3a;
#endif
    if (t3_live) g.tsq += a_t3;
    const double a_dqsl = a_bet * (double)x.ex; a_ex += a_bet * (double)dqsl;
    // qmq = qw_pert - qsat; qw_pert = qw + ((qw p5) pattern) spp
    a_qsat -= a_qmq;
    a_qw += a_qmq;
    if (sppf != 0.f) { const double a3 = a_qmq * (double)sppf; g.pat += a3 * (double)qp1; a_qw += (a3 * (double)x.pat) * (double)p5; }
    // b = a rsl; a = 1 / (1 + xl rsl / cpm)
    double a_rsl = a_bv * (double)av; a_av += a_bv * (double)rsl;
    const double a_den1 = -(a_av * (1.0 / (double)den1)) / (double)den1;
    const double a_xlrsl = a_den1 / (double)cpm;
    a_qw += (-(a_den1 * ((double)xlrsl / (double)cpm)) / (double)cpm) * (double)cpv;
    a_xl += a_xlrsl * (double)rsl; a_rsl += a_xlrsl * (double)xl;
    double a_t2 = 0.0;
    { const double a_n = a_rsl / (double)rs_d, a_d = -(a_rsl * ((double)rs_n / (double)rs_d)) / (double)rs_d;
      a_xl += a_n * (double)qsat; a_qsat += a_n * (double)xl; a_t2 += a_d * (double)r_v; }
    { const double a_n = a_dqsl / (double)dq_d, a_d = -(a_dqsl * ((double)dq_n / (double)dq_d)) / (double)dq_d;
      a_qsat += (a_n * (double)xlv) * (double)ep_2; a_t2 += a_d * (double)r_d; }
    a_t += a_t2 * (2.0 * (double)t);
    // rh = max(min(qw / max(qsat, 1e-10), rhmax), 0.001)
    if (rh_live) { a_qw += a_rh / (double)qsd; if (qsd_live) a_qsat += -(a_rh * ((double)x.qw / (double)qsd)) / (double)qsd; }
    m3_qsat_blend<true>(t, x.p, a_qsat, a_t, a_p);
    m3_xl_blend<true>(t, a_xl, a_t);
    a_th += a_t * (double)x.ex; a_ex += a_t * (double)x.th;
    g.zw += a_zagl; a_dz += a_zagl * (double)p5;
    g.dz += a_dz; g.th += a_th; g.ex += a_ex; g.p += a_p; g.qw += a_qw;
}

// k_tropo of f_mym_condensation (integer control; forward-exact).
template<class TH, class P> __device__ __forceinline__ int m3_k_tropo(int kts, int kte, TH th, P p) {
    int k;
    for (k = kte - 3; k >= kts; --k) {
        const float theta1 = th(k), theta2 = th(k + 2);
        const float ht1 = r6_mul(44307.692f, r6_sub(m3c::one, mynn_pow(r6_div(p(k), 101325.f), 0.190f)));
        const float ht2 = r6_mul(44307.692f, r6_sub(m3c::one, mynn_pow(r6_div(p(k + 2), 101325.f), 0.190f)));
        if (r6_div(r6_sub(theta2, theta1), r6_sub(ht2, ht1)) < r6_div(10.f, 1500.f) && ht1 < 19000.f && ht1 > 4000.f) break;
    }
    return mynn_max((int)(kts + 2), (int)(k + 2));
}

// Column reverse of M3 for the fused MYNN reverse kernel. The accessor A provides, per level k:
//   M3Lev in(k)                    primal inputs (launch-input snapshot or earlier-stage registers/shared)
//   float th(k), p(k)              primal theta and pressure (k_tropo)
//   void take_top(kte, &bvt, &bvq) returns the kte cotangents of vt and vq and zeroes the five arrays the forward
//                                  writes at kte (vt vq qc_bl qi_bl cldfra_bl; sgm(kte) is not written, leave it)
//   M3OBar take(k)                 returns and zeroes the six output cotangents at k (kts..kte-1)
//   void add(k, const M3IBar&)     accumulates the input cotangents at k
template<class A> __device__ void m3_cond_rev_column(A& acc, int kts, int kte, float pblh1, float xland, float closure, int spp_pbl) {
    M3Col c{pblh1, xland, closure, spp_pbl, 0};
    c.k_tropo = m3_k_tropo(kts, kte, [&](int k) { return acc.th(k); }, [&](int k) { return acc.p(k); });
    double tvt = 0.0, tvq = 0.0;
    acc.take_top(kte, tvt, tvq);
    for (int k = kte - 1; k >= kts; --k) {
        M3OBar b = acc.take(k);
        if (k == kte - 1) { b.vt += tvt; b.vq += tvq; }
        M3IBar g{}; M3Out y;
        m3c2_level<true>(acc.in(k), c, k, y, b, g);
        acc.add(k, g);
    }
}
