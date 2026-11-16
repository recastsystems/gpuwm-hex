// SPDX-License-Identifier: Apache-2.0
// x20 MYNN M3 proof hook (not the fused kernel): in the generated P3 tape entries (mynn_whole_ad / _count, carrier
// MynnTapeF) f_mym_condensation (bl_mynn_cloudpdf = 2) is replaced by m3c2_level: the primal is the stage's float
// recompute and each output's derivative is ONE linear combination of its level inputs, with the weights read off the
// stage's own reverse (m3c2_level<true> seeded with one output cotangent at a time). The tangent entry keeps the
// generated derivative, so the P3 dot test <J_gen dx, ybar> = <dx, J_M3^T ybar> checks the hand reverse against the
// generator's tangent, and the identity test checks the stage primal bitwise. Off with -DMW_M3_OFF.
#pragma once

template<class A, class F>
__device__ __noinline__ void m3_tape_swap(int kts, int kte, F xland, A dz, A zw, A qc, A qi, A qs, A p, A exner, A tsq, A qsq,
                                          A cov, A qw, A th, A pat, A qc_bl1, A qi_bl1, A cldfra_bl1, A vt, A vq, A sgm,
                                          F pblh1, F closure, int spp_pbl) {
    M3Col c{pblh1.v, xland.v, closure.v, spp_pbl, 0};
    c.k_tropo = m3_k_tropo(kts, kte, [&](int k) { return th(k).v; }, [&](int k) { return p(k).v; });
    for (int k = kts; k <= kte - 1; ++k) {
        const M3Lev x{zw(k).v, dz(k).v, th(k).v, exner(k).v, p(k).v, qw(k).v, qsq(k).v, tsq(k).v, cov(k).v,
                      qi(k).v, qs(k).v, qc(k).v, pat(k).v};
        decltype(zw(k).d) din[13] = {zw(k).d, dz(k).d, th(k).d, exner(k).d, p(k).d, qw(k).d, qsq(k).d, tsq(k).d, cov(k).d,
                                     qi(k).d, qs(k).d, qc(k).d, pat(k).d};
        // the generator's derivative-domain refusals inside this stage (failure slot 1, mw_dom code), so the refused
        // column set is unchanged: sqrt of the double product tsq*qsq (code 6) and sqrt of the float variance (code 2)
        // at an exact zero with a dependent operand.
        {
            const bool t3dep = !(m3c::zero > x.tsq) && tsq(k).d.depends, r3dep = !(m3c::zero > x.qsq) && qsq(k).d.depends;
            const float t3f = mynn_max((float)(m3c::zero), (float)(x.tsq)), r3f = mynn_max((float)(m3c::zero), (float)(x.qsq));
            const bool pdep = (t3dep && r3f != 0.f) || (r3dep && t3f != 0.f) || (t3dep && r3dep);
            if (mynn_dmul((double)t3f, (double)r3f) <= 0.0 && pdep) mw_domain(6, t3dep ? tsq(k).d : qsq(k).d);
            const bool cl0 = closure.v < 2.7f;
            if (r3f <= 0.f && (cl0 ? r3dep : (r3dep || t3dep))) mw_domain(2, r3dep ? qsq(k).d : tsq(k).d);
        }
        M3Out y; M3IBar g0{};
        m3c2_level<false>(x, c, k, y, M3OBar{}, g0);
        decltype(zw(k).d) dout[6];
        for (int o = 0; o < 6; ++o) {
            M3OBar b{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (o == 0) b.qc_bl = 1.0; else if (o == 1) b.qi_bl = 1.0; else if (o == 2) b.cf = 1.0;
            else if (o == 3) b.vt = 1.0; else if (o == 4) b.vq = 1.0; else b.sgm = 1.0;
            M3IBar g{}; M3Out yy;
            m3c2_level<true>(x, c, k, yy, b, g);
            const double w[13] = {g.zw, g.dz, g.th, g.ex, g.p, g.qw, g.qsq, g.tsq, g.cov, g.qi, g.qs, g.qc, g.pat};
            decltype(zw(k).d) acc;
            for (int j = 0; j < 13; ++j) if (w[j] != 0.0 && din[j].depends) acc = acc + din[j] * w[j];
            dout[o] = acc;
        }
        qc_bl1(k) = F(y.qc_bl, dout[0]);
        qi_bl1(k) = F(y.qi_bl, dout[1]);
        cldfra_bl1(k) = F(y.cf, dout[2]);
        vt(k) = F(y.vt, dout[3]);
        vq(k) = F(y.vq, dout[4]);
        sgm(k) = F(y.sgm, dout[5]);
    }
    vt(kte) = vt(kte - 1);
    vq(kte) = vq(kte - 1);
    qc_bl1(kte) = m3c::zero;
    qi_bl1(kte) = m3c::zero;
    cldfra_bl1(kte) = m3c::zero;
}
