// Adapted from MYNN fused/m3_cond.cuh at 20a441f67. Explicit hand transpose.
// Production case 2 only. P=float for production primals, bars always double.
// Constants retain their source float32 values in the double diagnostic build.
#pragma once
#include "numeric.hpp"
namespace direct {
#ifdef CLOUD_COVER
static unsigned long long cloud_coverage[38][2]{},cloud_fng[4]{};
#endif
using std::signbit;
template<class P> inline P mynn_min(P a,P b){return a<b?a:b;}
template<class P> inline P mynn_max(P a,P b){return a>b?a:b;}
template<class P> inline P mynn_abs(P a){return std::abs(a);}
template<class P> inline P mynn_sign(P a,P b){return std::copysign(a,b);}
template<class P> inline P mynn_exp(P a){return std::exp(a);}
template<class P> inline P mynn_log10(P a){return std::log10(a);}
template<class P> inline P mynn_atan(P a){return std::atan(a);}
template<class P> inline P mynn_pow(P a,float b){return P(std::pow(a,P(b)));}
template<class A,class B> inline double mynn_dmul(A a,B b){return double(a)*double(b);}
template<class A,class B> inline double mynn_dadd(A a,B b){return double(a)+double(b);}
template<class A,class B> inline double mynn_dsub(A a,B b){return double(a)-double(b);}
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
inline float cJ(int i) {
    switch (i) { case 0: return .611583699e03f; case 1: return .444606896e02f; case 2: return .143177157e01f;
                 case 3: return .264224321e-1f; case 4: return .299291081e-3f; case 5: return .203154182e-5f;
                 case 6: return .702620698e-8f; case 7: return .379534310e-11f; default: return (-(.321582393e-13f)); }
}
inline float cK(int i) {
    switch (i) { case 0: return .609868993e03f; case 1: return .499320233e02f; case 2: return .184672631e01f;
                 case 3: return .402737184e-1f; case 4: return .565392987e-3f; case 5: return .521693933e-5f;
                 case 6: return .307839583e-7f; case 7: return .105785160e-9f; default: return .161444444e-12f; }
}
} // namespace m3c

// Horner polynomial: es = c0 + xc*(c1 + xc*(... + xc*c8)), the captured evaluation order.
// REV: a_xc += (d es / d xc) a_es through the reverse of the r6_mul/r6_add chain.
template<bool ICE, bool REV, class P> inline P m3_horner(P xc, double a_es, double& a_xc) {
    P h[9];
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
template<bool REV, class P> inline P m3_q_from_es(P es, P p, double a_q, double& a_es, double& a_p) {
    const P p15 = r6_mul(p, 0.15f);
    const bool es_live = es < p15;
    const P es2 = mynn_min((P)(es), (P)(p15));
    const P num = r6_mul(0.622f, es2);
    const P dd = r6_sub(p, es2);
    const bool den_live = dd > 1e-5f;
    const P den = mynn_max((P)(dd), (P)(1e-5f));
    const P q = r6_div(num, den);
    if (REV) {
        const double a_num = a_q / (double)den;
        const double a_den = -(a_q * ((double)num / (double)den)) / (double)den;
        double a_es2 = a_num * 0.622f;
        if (den_live) { a_p += a_den; a_es2 -= a_den; }
        if (es_live) a_es += a_es2; else a_p += a_es2 * 0.15f;
    }
    return q;
}

template<bool REV, class P> inline P m3_qsat_blend(P t, P p, double a_q, double& a_t, double& a_p) {
    using namespace m3c;
    const P tm = r6_sub(t, t0c);
    const bool xc_live = !(r6_neg(80.f) > tm);
    const P xc = mynn_max((P)(r6_neg(80.f)), (P)(tm));
    double a_xc = 0.0, dmy = 0.0;
    P q;
    if (t >= r6_sub(t0c, 6.f)) {
        double a_es = 0.0;
        const P es = m3_horner<false, false>(xc, 0.0, dmy);
        q = m3_q_from_es<REV>(es, p, a_q, a_es, a_p);
        if (REV) m3_horner<false, true>(xc, a_es, a_xc);
    } else if (t <= tice) {
        double a_es = 0.0;
        const P es = m3_horner<true, false>(xc, 0.0, dmy);
        q = m3_q_from_es<REV>(es, p, a_q, a_es, a_p);
        if (REV) m3_horner<true, true>(xc, a_es, a_xc);
    } else {
        const P esl = m3_horner<false, false>(xc, 0.0, dmy);
        const P esi = m3_horner<true, false>(xc, 0.0, dmy);
        const P rslf = m3_q_from_es<false>(esl, p, 0.0, dmy, dmy);
        const P rsif = m3_q_from_es<false>(esi, p, 0.0, dmy, dmy);
        const P denc = r6_sub(r6_sub(t0c, 6.f), tice);
        const P chi = r6_div(r6_sub(r6_sub(t0c, 6.f), t), denc);
        const P omc = r6_sub(1.f, chi);
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

template<bool REV, class P> inline P m3_xl_blend(P t, double a_xl, double& a_t) {
    using namespace m3c;
    const P tm = r6_sub(t, t0c);
    if (t >= t0c) {
        const P s = r6_sub(cpv, cliq);
        if (REV) a_t += a_xl * (double)s;
        return r6_add(xlv, r6_mul(s, tm));
    } else if (t <= tice) {
        const P s = r6_sub(cpv, cice);
        if (REV) a_t += a_xl * (double)s;
        return r6_add(xls, r6_mul(s, tm));
    }
    const P sv = r6_sub(cpv, cliq), si = r6_sub(cpv, cice);
    const P xlvt = r6_add(xlv, r6_mul(sv, tm));
    const P xlst = r6_add(xls, r6_mul(si, tm));
    const P denc = r6_sub(t0c, tice);
    const P chi = r6_div(r6_sub(t0c, t), denc);
    const P omc = r6_sub(1.f, chi);
    if (REV) {
        const double a_xlvt = a_xl * (double)omc, a_xlst = a_xl * (double)chi;
        const double a_chi = -(a_xl * (double)xlvt) + a_xl * (double)xlst;
        a_t += a_xlvt * (double)sv + a_xlst * (double)si - a_chi / (double)denc;
    }
    return r6_add(r6_mul(omc, xlvt), r6_mul(chi, xlst));
}

// Level inputs (primal, P) and their cotangents (double); output values and cotangents.
template<class P> struct M3Lev  { P zw, dz, th, ex, p, qw, qsq, tsq, cov, qi, qs, qc, pat; };
struct M3IBar { double zw, dz, th, ex, p, qw, qsq, tsq, cov, qi, qs, qc, pat; };
template<class P> struct M3Out  { P qc_bl, qi_bl, cf, vt, vq, sgm; };
struct M3OBar { double qc_bl, qi_bl, cf, vt, vq, sgm; };
template<class P> struct M3Col  { P pblh1, xland, closure; int spp_pbl, k_tropo; };

// min(max(v, 0) / d, 1), the blending-weight ramp; live = the derivative passes (v > 0 and the quotient < 1).
template<class P> inline P m3_ramp(P v, double d, bool& live) {
    const bool l1 = v > m3c::zero;
    const P m = mynn_max((P)(v), (P)(m3c::zero));
    const P w = r6_div(m, P(d));
    live = l1 && (w < m3c::one);
    return mynn_min((P)(w), (P)(m3c::one));
}

// One level of f_mym_condensation, case 2. Forward always (y = outputs, bitwise); REV: g += J^T yb.
template<bool REV, class P> inline void m3c2_level(const M3Lev<P>& x, const M3Col<P>& c, int k, M3Out<P>& y,
                                                              const M3OBar& yb, M3IBar& g) {
    using namespace m3c;
    const P pblh2 = mynn_max((P)(c.pblh1), (P)(ten));
    const P zagl = r6_add(x.zw, r6_mul(p5, x.dz));
    const P t = r6_mul(x.th, x.ex);
    double dmy = 0.0;
    const P xl = m3_xl_blend<false>(t, 0.0, dmy);
    const P qsat = m3_qsat_blend<false>(t, x.p, 0.0, dmy, dmy);
    // rh
    const bool qsd_live = qsat > 1e-10f;
    const P qsd = mynn_max((P)(qsat), (P)(1e-10f));
    const P rhq = r6_div(x.qw, qsd);
    const bool rh_in = rhq < rhmax;
    const P rh_m = mynn_min((P)(rhq), (P)(rhmax));
    const bool rh_out = rh_m > 0.001f;
    const P rh = mynn_max((P)(rh_m), (P)(0.001f));
    const bool rh_live = rh_in && rh_out;
    const P t2 = mynn_ipow<2>(t);
    const P rs_n = r6_mul(xl, qsat);
    const P rs_d = r6_mul(r_v, t2);
    const P rsl = r6_div(rs_n, rs_d);
    const P cpm = r6_add(cp, r6_mul(x.qw, cpv));
    const P xlrsl = r6_mul(xl, rsl);
    const P q2 = r6_div(xlrsl, cpm);
    const P den1 = r6_add(one, q2);
    const P av = r6_div(one, den1);
    const P bv = r6_mul(av, rsl);
    // perturbed qw, qmq
    const P sppf = ((P)(c.spp_pbl));
    const P qp1 = r6_mul(x.qw, p5);
    const P qp2 = r6_mul(qp1, x.pat);
    const P qp3 = r6_mul(qp2, sppf);
    const P qw_pert = r6_add(x.qw, qp3);
    const P qmq = r6_sub(qw_pert, qsat);
    const bool r3_live = !(zero > x.qsq);
    const double r3 = mynn_max(P(zero),P(x.qsq));
    const P s0 = mynn_sqrt(((P)(r3)));
    const bool sA_live = s0 > 1e-13f;
    const P sA = mynn_max((P)(s0), (P)(1e-13f));
    const P qwp5 = r6_mul(x.qw, p5);
    const bool sB_a = sA < qwp5;
    const P sB = mynn_min((P)(sA), (P)(qwp5));
    bool wdz_live; const P wdz = m3_ramp(r6_sub(x.dz, 100.f), 500.f, wdz_live);
    const P sBp2 = r6_mul(sB, p2);
    const P sBw = r6_mul(sBp2, wdz);
    const P sgm = r6_add(sB, sBw);
    // clim
    bool w5_live; const P w5 = m3_ramp(r6_sub(zagl, r6_add(pblh2, 10.f)), 250.f, w5_live);
    const P clim1 = r6_add(r6_mul(clim_pbl, r6_sub(one, w5)), r6_mul(clim_trp, w5));
    const P zsl1 = mynn_min((P)(mynn_max((P)(r6_mul(p1, pblh2)), (P)(50.f))), (P)(150.f));
    bool w6_live; const P w6 = m3_ramp(r6_sub(zagl, zsl1), 150.f, w6_live);
    const P clim2 = r6_add(r6_mul(clim_sfc, r6_sub(one, w6)), r6_mul(clim1, w6));
    const P qwcl = r6_mul(x.qw, clim2);
    const bool sc1_a = sgm > qwcl;
    const P sc1 = mynn_max((P)(sgm), (P)(qwcl));
    const bool sc2_live = sc1 > 1e-13f;
    const P sc2 = mynn_max((P)(sc1), (P)(1e-13f));
    const bool qmq_pos = qmq >= zero;
    const P qw02 = r6_mul(0.02f, x.qw);
    const bool sc3_a = qmq_pos && (qw02 > sc2);
    const P sc3 = qmq_pos ? mynn_max((P)(qw02), (P)(sc2)) : sc2;
    const P q1_0 = r6_div(qmq, sc3);
    // rh adjustments above the PBL
    bool wa_live; const P wta = m3_ramp(r6_sub(zagl, pblh2), 300.f, wa_live);
    const P den3 = r6_sub(one, rhcrit);
    const P qis = r6_add(x.qi, x.qs);
    const bool blk_i = (qis > 1.e-10f) && (zagl > pblh2);
    P q1_1 = q1_0, mxi = 0.f, ci = 0.f;
    bool mxi_live = false, rai_live = false, rbi_rh = false, q1i_rh = false;
    if (blk_i) {
        const P lgi = r6_add(ten, mynn_log10(qis));
        mxi_live = lgi > zero; mxi = mynn_max((P)(lgi), (P)(zero));
        ci = r6_mul(wta, 0.037f);
        const P si_ = r6_add(rhcrit, r6_mul(ci, mxi));
        rai_live = si_ < rhmax; const P rai = mynn_min((P)(si_), (P)(rhmax));
        rbi_rh = rh > rai; const P rbi = mynn_max((P)(rh), (P)(rai));
        const P q1ri = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rbi, rhcrit)), den3));
        q1i_rh = q1ri > q1_0; q1_1 = mynn_max((P)(q1ri), (P)(q1_0));
    }
    const bool blk_c = (x.qc > 1.e-5f) && (zagl > pblh2);
    P q1_2 = q1_1, mxc = 0.f, cc = 0.f;
    bool mxc_live = false, rac_live = false, rbc_rh = false, q1c_rh = false;
    if (blk_c) {
        const P lgc = r6_add(five, mynn_log10(x.qc));
        mxc_live = lgc > zero; mxc = mynn_max((P)(lgc), (P)(zero));
        cc = r6_mul(wta, 0.07f);
        const P sc_ = r6_add(rhcrit, r6_mul(cc, mxc));
        rac_live = sc_ < rhmax; const P rac = mynn_min((P)(sc_), (P)(rhmax));
        rbc_rh = rh > rac; const P rbc = mynn_max((P)(rh), (P)(rac));
        const P q1rc = r6_add(r6_neg(3.f), r6_div(r6_mul(three, r6_sub(rbc, rhcrit)), den3));
        q1c_rh = q1rc > q1_1; q1_2 = mynn_max((P)(q1rc), (P)(q1_1));
    }
    const P q1 = q1_2;
    // cloud fraction from q1 (cldfra_rh is zero in this generation; its tanh terms feed nothing)
    const P u2 = r6_div(r6_sub(zagl, r6_sub(c.pblh1, 100.f)), 200.f);
    const P u2m = mynn_max((P)(u2), (P)(zero));
    const bool w2b_live = (u2 > zero) && (u2m < one);
    const P wtb = mynn_min((P)(u2m), (P)(one));
    const P xa0 = r6_mul(3.6f, r6_add(q1, 0.05f));
    const P sa0 = r6_add(p5, r6_mul(0.35f, mynn_atan(xa0)));
    const P ma0 = mynn_min((P)(sa0), (P)(one));
    const bool cq0_live = (sa0 < one) && !(zero > ma0);
    const P cq0 = mynn_max((P)(zero), (P)(ma0));
    const P xa1 = r6_mul(1.6f, r6_add(q1, 0.55f));
    const P sa1 = r6_add(p5, r6_mul(0.39f, mynn_atan(xa1)));
    const P ma1 = mynn_min((P)(sa1), (P)(one));
    const bool cq1_live = (sa1 < one) && !(zero > ma1);
    const P cq1 = mynn_max((P)(zero), (P)(ma1));
    const P omwb = r6_sub(one, wtb);
    const P cfq = r6_add(r6_mul(cq0, omwb), r6_mul(cq1, wtb));
    const bool cf_live0 = cfq > zero;              // max(cldfra_qsq, cldfra_rh = 0)
    P cf = mynn_max((P)(cfq), (P)(zero));
    // qlim
    bool w3_live; const P w3 = m3_ramp(r6_sub(zagl, r6_add(pblh2, ten)), 300.f, w3_live);
    const P qlim1 = r6_add(r6_mul(qlim_pbl, r6_sub(one, w3)), r6_mul(qlim_trp, w3));
    bool w4_live; const P w4 = m3_ramp(r6_sub(zagl, zsl1), 200.f, w4_live);
    const P qlim2 = r6_add(r6_mul(qlim_sfc, r6_sub(one, w4)), r6_mul(qlim1, w4));
    const P qwql = r6_mul(x.qw, qlim2);
    const bool sq_a = sgm > qwql;
    const P sgmq = mynn_max((P)(sgm), (P)(qwql));
    const P qw025 = r6_mul(0.025f, x.qw);
    const bool mq_a = sgmq < qw025;
    const P mq = mynn_min((P)(sgmq), (P)(qw025));
    P qlw = r6_mul(mq, cf);
    P qli = r6_mul(mq, cf);
    const P cf_pre = cf;
    const bool cf_small = cf < 0.001f;
    if (cf_small) { qli = zero; qlw = zero; cf = zero; }
    const P dl = r6_sub(tliq, tice);
    const P u_l = r6_div(r6_sub(t, tice), dl);
    const P u_lm = mynn_max((P)(u_l), (P)(zero));
    const bool liq_live = (u_l > zero) && (u_lm < one);
    const P liq = mynn_min((P)(u_lm), (P)(one));
    const P oml = r6_sub(one, liq);
    P qc_bl = r6_mul(liq, qlw);
    P qi_bl = r6_mul(oml, qli);
    const bool tropo = k >= c.k_tropo;
    if (tropo) { cf = zero; qc_bl = zero; qi_bl = zero; }
    // vt, vq
    const P q1c = (r6_sub(c.xland, 1.5f) >= zero) ? r6_neg(2.5f) : r6_neg(2.0f);
    const bool q1k_live = q1 > q1c;
    const P q1k = mynn_max((P)(q1), (P)(q1c));
    int fb; P fng, fe = 0.f, f0 = 0.f;
    if (q1k >= one) { fb = 0; fng = one; }
    else if (q1k >= r6_neg(1.7f) && q1k < one) { fb = 1; fe = mynn_exp(r6_neg(r6_mul(0.4f, r6_sub(q1k, one)))); fng = fe; }
    else if (q1k >= r6_neg(2.5f) && q1k < r6_neg(1.7f)) { fb = 2; fe = mynn_exp(r6_neg(r6_mul(3.8f, r6_add(q1k, 1.7f)))); fng = r6_add(three, fe); }
    else { fb = 3; fe = mynn_exp(r6_neg(r6_mul(1.6f, r6_add(q1k, 2.5f)))); f0 = r6_add(23.9f, fe); fng = mynn_min((P)(f0), (P)(60.f)); }
    const bool cfm_live = cf < 0.6f;
    const P cfm0 = mynn_min((P)(cf), (P)(0.6f));
    const P zsl3 = mynn_min((P)(mynn_max((P)(r6_mul(p1, pblh2)), (P)(25.f))), (P)(hundred));
    const P wzr = r6_div(zagl, zsl3);
    const bool wz_live = wzr < one;
    const P wz = mynn_min((P)(wzr), (P)(one));
    const P cfmax = r6_mul(cfm0, wz);
    const P bt = r6_mul(bv, t);
    const P bb = r6_div(bt, x.th);
    const P qww = r6_add(one, r6_mul(p608, x.qw));
    const P alpha = r6_mul(p608, x.th);
    const P r1 = r6_div(x.th, t);
    const P r2 = r6_div(xl, cp);
    const P beta = r6_sub(r6_mul(r1, r2), r6_mul(1.61f, x.th));
    const P cb = r6_mul(cfmax, beta);
    const P cbb = r6_mul(cb, bb);
    const P cbbf = r6_mul(cbb, fng);
    const P cba = r6_mul(cb, av);
    const P cbaf = r6_mul(cba, fng);
    y.vt = r6_sub(r6_sub(qww, cbbf), one);
    y.vq = r6_sub(r6_add(alpha, cbaf), tv0);
    y.qc_bl = qc_bl; y.qi_bl = qi_bl; y.cf = cf; y.sgm = sgm;
#ifdef CLOUD_COVER
    ++cloud_coverage[0][qsd_live?1:0];
    ++cloud_coverage[1][rh_in?1:0];
    ++cloud_coverage[2][rh_out?1:0];
    ++cloud_coverage[3][r3_live?1:0];
    ++cloud_coverage[4][sA_live?1:0];
    ++cloud_coverage[5][sB_a?1:0];
    ++cloud_coverage[6][wdz_live?1:0];
    ++cloud_coverage[7][w5_live?1:0];
    ++cloud_coverage[8][w6_live?1:0];
    ++cloud_coverage[9][sc1_a?1:0];
    ++cloud_coverage[10][sc2_live?1:0];
    ++cloud_coverage[11][qmq_pos?1:0];
    ++cloud_coverage[12][sc3_a?1:0];
    ++cloud_coverage[13][wa_live?1:0];
    ++cloud_coverage[14][blk_i?1:0];
    ++cloud_coverage[15][blk_c?1:0];
    if(blk_i) ++cloud_coverage[16][mxi_live?1:0];
    if(blk_i) ++cloud_coverage[17][rai_live?1:0];
    if(blk_i) ++cloud_coverage[18][rbi_rh?1:0];
    if(blk_i) ++cloud_coverage[19][q1i_rh?1:0];
    if(blk_c) ++cloud_coverage[20][mxc_live?1:0];
    if(blk_c) ++cloud_coverage[21][rac_live?1:0];
    if(blk_c) ++cloud_coverage[22][rbc_rh?1:0];
    if(blk_c) ++cloud_coverage[23][q1c_rh?1:0];
    ++cloud_coverage[24][w2b_live?1:0];
    ++cloud_coverage[25][cq0_live?1:0];
    ++cloud_coverage[26][cq1_live?1:0];
    ++cloud_coverage[27][cf_live0?1:0];
    ++cloud_coverage[28][w3_live?1:0];
    ++cloud_coverage[29][w4_live?1:0];
    ++cloud_coverage[30][sq_a?1:0];
    ++cloud_coverage[31][mq_a?1:0];
    ++cloud_coverage[32][cf_small?1:0];
    ++cloud_coverage[33][liq_live?1:0];
    ++cloud_coverage[34][tropo?1:0];
    ++cloud_coverage[35][q1k_live?1:0];
    ++cloud_coverage[36][cfm_live?1:0];
    ++cloud_coverage[37][wz_live?1:0];
    ++cloud_fng[fb];
#endif
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
    if (r3_live) g.qsq += a_r3;
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
template<class TH, class PR> inline int m3_k_tropo(int kts, int kte, TH th, PR p) {
    using P = decltype(th(kts));
    int k;
    for (k = kte - 3; k >= kts; --k) {
        const P theta1 = th(k), theta2 = th(k + 2);
        const P ht1 = r6_mul(44307.692f, r6_sub(m3c::one, mynn_pow(r6_div(p(k), 101325.f), 0.190f)));
        const P ht2 = r6_mul(44307.692f, r6_sub(m3c::one, mynn_pow(r6_div(p(k + 2), 101325.f), 0.190f)));
        if (r6_div(r6_sub(theta2, theta1), r6_sub(ht2, ht1)) < r6_div(10.f, 1500.f) && ht1 < 19000.f && ht1 > 4000.f) break;
    }
    return mynn_max((int)(kts + 2), (int)(k + 2));
}


// Packed input fields: zw dz th exner p qw qsq tsq cov qi qs qc pattern,
// old_sgm thl qv sh el; each has MAX words, physical indices 1..n.
// Packed output/seed fields: qc_bl qi_bl cldfra_bl vt vq sgm.
constexpr int CNI=18*MAX, CNO=6*MAX;
template<class P> inline M3Lev<P> cloud_level_input(const P*x,int k){
 return {x[k],x[MAX+k],x[2*MAX+k],x[3*MAX+k],x[4*MAX+k],x[5*MAX+k],
 x[6*MAX+k],x[7*MAX+k],x[8*MAX+k],x[9*MAX+k],x[10*MAX+k],x[11*MAX+k],x[12*MAX+k]};
}
// Invalid derivative-domain preflight for case2/closure<2.7. It follows native
// sqrt dependency rules with every cut input active, including zero-output paths.
template<class P> inline bool cloud_valid(int n,const P*x,const M3Col<P>&c){
 if(n<4||n>=MAX||c.spp_pbl!=0||!(c.closure<P(2.7f)))return false;
 for(int k=1;k<n;++k){auto z=cloud_level_input(x,k);
  if(!(z.th>0&&z.ex>0&&z.p>0))return false;
  bool td=!(P(0)>z.tsq),rd=!(P(0)>z.qsq);
  double t=mynn_max(P(0),z.tsq),r=mynn_max(P(0),z.qsq);
  if(t*r<=0&&((td&&r!=0)||(rd&&t!=0)||(td&&rd)))return false;
  if(r<=0&&rd)return false;
 }
 return true;
}
// Additive bars; no aliasing between x, seed, bar or optional y.
// One column owner; top vt/vq seeds fold before the level transpose.
template<class P> inline bool cloud_backward(int n,const P*x,const M3Col<P>&cfg,
 const double*seed,double*bar,P*y=nullptr,bool planted=false){
 if(!cloud_valid(n,x,cfg))return false;
 M3Col<P> c=cfg;
 c.k_tropo=m3_k_tropo(1,n,[&](int k){return x[2*MAX+k];},[&](int k){return x[4*MAX+k];});
 for(int k=n-1;k>=1;--k){
  M3OBar b{seed[k],seed[MAX+k],seed[2*MAX+k],seed[3*MAX+k],seed[4*MAX+k],seed[5*MAX+k]};
  if(k==n-1){b.vt+=seed[3*MAX+n];b.vq+=seed[4*MAX+n];}
  M3IBar g{};M3Out<P> o;
  m3c2_level<true>(cloud_level_input(x,k),c,k,o,b,g);
  bar[k]+=g.zw;bar[MAX+k]+=g.dz;bar[2*MAX+k]+=g.th;
  bar[3*MAX+k]+=g.ex;bar[4*MAX+k]+=g.p;bar[5*MAX+k]+=g.qw;
  bar[6*MAX+k]+=g.qsq*((planted&&k==4)?1.001:1.0);
  bar[9*MAX+k]+=g.qi;bar[10*MAX+k]+=g.qs;bar[11*MAX+k]+=g.qc;
  if(y){y[k]=o.qc_bl;y[MAX+k]=o.qi_bl;y[2*MAX+k]=o.cf;y[3*MAX+k]=o.vt;y[4*MAX+k]=o.vq;y[5*MAX+k]=o.sgm;}
 }
 bar[13*MAX+n]+=seed[5*MAX+n];
 if(y){y[n]=y[MAX+n]=y[2*MAX+n]=P(0);y[3*MAX+n]=y[3*MAX+n-1];y[4*MAX+n]=y[4*MAX+n-1];y[5*MAX+n]=x[13*MAX+n];}
 return true;
}
} // namespace direct
