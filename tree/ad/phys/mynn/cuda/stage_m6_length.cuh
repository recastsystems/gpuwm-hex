// SPDX-License-Identifier: Apache-2.0
// x20 MYNN stage M6 (length-frozen): f_mym_level2 (closure part), f_mym_length, f_boulac_length as PRIMAL-ONLY stages.
// Included INSIDE struct MynnColumn by tape_tools/patch_m6_primal.py, right before the generated f_mym_level2.
//
// Why: under gen_whole.py --freeze closure, sm/sh (level2) and el (length, boulac) are frozen right after each call
// (mw_frz_arr), so every tape node that feeds only them ends the reverse sweep with a zero adjoint. Their live outputs
// are the gradients dtl/dqw/dtv/gm/gh (level2) and qkw/qtw (length). This stage keeps exactly those live statements on
// the carrier (MR) and computes the frozen outputs from primal float values: no tape node, no reverse, no count-pass work.
// Values are bitwise those of before: the carrier rules compute primal values with the same captured float routines
// (mw_ad_rules.inc: "Primal: the captured float/double arithmetic of the generated source (bitwise by construction)"),
// and the frozen part runs the same generated code instantiated on MynnColumn<1, float, double>.
// Interface to the neighbouring stages is unchanged: same call signatures, same outputs (sm/sh/el as passive carriers).
// The only extra state is arena scratch for primal copies (released by the frame at return); no global hand-off.
template<class Q> struct m6_is_float { static constexpr bool v = false; };
template<> struct m6_is_float<float> { static constexpr bool v = true; };
__device__ static __forceinline__ float m6v(float x) { return x; }
template<class Q> __device__ static __forceinline__ float m6v(const Q& x) { return x.v; }

// Primal copy of a carrier array (same bounds); an absent optional array stays absent.
template<class A> __device__ MynnArray<float, 1> m6_prim(const A& a) {
    MynnArray<float, 1> f;
    if (!a.data) return f;
    f = mynn_array_any<float, 1, mynn_tile>(arena, {a.lo[0]}, {a.hi[0]});
    mynn_lane_sync();
    for (int k = a.lo[0] + lane_sub; k <= a.hi[0]; k += lane_width) f(k) = m6v(a(k));
    mynn_lane_sync();
    return f;
}
template<class A> __device__ MynnArray<float, 1> m6_scratch(const A& a) {
    MynnArray<float, 1> f;
    if (!a.data) return f;
    return mynn_array_any<float, 1, mynn_tile>(arena, {a.lo[0]}, {a.hi[0]});
}

// level2 closure (RRFS 44c0849 module_bl_mynnedmf.F90:1760-1800 and the ri statement of the gradient loop), primal.
// ri(k) = -gh(k)/max(duz, 1e-10) with duz == gm(k) bitwise (gm(k) = duz in both ess branches).
__device__ void m6_level2_closure(int kts, int kte, MynnArray<MR, 1> gm, MynnArray<MR, 1> gh, MynnArray<MR, 1> sm,
                                  MynnArray<MR, 1> sh) {
    const int lo = kts + 1, n = mynn_trip(lo, kte, 1);
    mynn_lane_sync();
    for (int it = lane_sub; it < n; it += lane_width) {
        const int k = lo + it;
        const float ri = r6_neg(r6_div(m6v(gh(k)), mynn_max(m6v(gm(k)), 1.0e-10f)));
        float a2fac;
        if (((ckmod) == (1))) {
            a2fac = r6_div(one, r6_add(one, mynn_max(zero, ri)));
        } else {
            a2fac = one;
        }
        const float rfc = r6_div(g1, r6_add(g1, g2));
        const float f1 = r6_add(r6_add(r6_mul(b1, r6_sub(g1, c1)), r6_mul(r6_mul(r6_mul(r6_mul(three, a2), a2fac), r6_sub(one, c2)), r6_sub(one, c5))), r6_mul(r6_mul(two, a1), r6_sub(three, r6_mul(two, c2))));
        const float f2 = r6_sub(r6_mul(b1, r6_add(g1, g2)), r6_mul(r6_mul(three, a1), r6_sub(one, c2)));
        const float rf1 = r6_div(r6_mul(b1, r6_sub(g1, c1)), f1);
        const float rf2 = r6_div(r6_mul(b1, g1), f2);
        const float smc = r6_div(r6_mul(r6_div(a1, r6_mul(a2, a2fac)), f1), f2);
        const float shc = r6_mul(r6_mul(three, r6_mul(a2, a2fac)), r6_add(g1, g2));
        const float ri1 = r6_div(p5, smc);
        const float ri2 = r6_mul(rf1, smc);
        const float ri3 = r6_sub(r6_mul(r6_mul(four, rf2), smc), r6_mul(two, ri2));
        const float ri4 = mynn_ipow<2>(ri2);
        const float rf = mynn_min(r6_mul(ri1, r6_sub(r6_add(ri, ri2), mynn_sqrt(r6_add(r6_sub(mynn_ipow<2>(ri), r6_mul(ri3, ri)), ri4)))), rfc);
        const float shk = r6_div(r6_mul(shc, r6_sub(rfc, rf)), r6_sub(one, rf));
        const float smk = r6_mul(r6_div(r6_mul(smc, r6_sub(rf1, rf)), r6_sub(rf2, rf)), shk);
        sh(k) = shk;
        sm(k) = smk;
    }
    mynn_lane_sync();
}

// f_mym_length with the frozen el computed primal-only. qkw/qtw (live: mym_turbulence uses them) stay on the carrier,
// computed by the same statements as every bl_mynn_mixlength branch of the generated routine (0: qtw = qkw).
__device__ void m6_length(int kts, int kte, MR xland, MynnArray<MR, 1> dz, MR dx, MynnArray<MR, 1> zw, MR rmol, MR rmolh,
                          MR flt, MR fltv, MR flq, MynnArray<MR, 1> vt, MynnArray<MR, 1> vq, MynnArray<MR, 1> u1,
                          MynnArray<MR, 1> v1, MynnArray<MR, 1> qke, MynnArray<MR, 1> qpe, MynnArray<MR, 1> dtv,
                          MynnArray<MR, 1> el, MR pblh, MynnArray<MR, 1> theta, MynnArray<MR, 1> qkw, MynnArray<MR, 1> qtw,
                          MR psig_bl, MynnArray<MR, 1> cldfra_bl1, int bl_mynn_mixlength, MynnArray<MR, 1> edmf_w1,
                          MynnArray<MR, 1> edmf_a1, MynnArray<MR, 1> edmf_w_dd1, MynnArray<MR, 1> edmf_a_dd1) {
    if constexpr (m6_is_float<MR>::v) {
        f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u1, v1, qke, qpe, dtv, el, pblh, theta,
                     qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
        return;
    } else {
    if (bl_mynn_mixlength < 0 || bl_mynn_mixlength > 2) {   // no branch in the generated switch: nothing is written
        f_mym_length(kts, kte, xland, dz, dx, zw, rmol, rmolh, flt, fltv, flq, vt, vq, u1, v1, qke, qpe, dtv, el, pblh, theta,
                     qkw, qtw, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, edmf_w_dd1, edmf_a_dd1);
        return;
    }
    MynnFrame mynn_frame(arena);
    // 1) primal copies of every array input, taken on entry (qkw/qtw are pure outputs of the generated routine).
    MynnArray<float, 1> f_dz = m6_prim(dz), f_zw = m6_prim(zw), f_vt = m6_prim(vt), f_vq = m6_prim(vq);
    MynnArray<float, 1> f_u1 = m6_prim(u1), f_v1 = m6_prim(v1), f_qke = m6_prim(qke), f_qpe = m6_prim(qpe);
    MynnArray<float, 1> f_dtv = m6_prim(dtv), f_theta = m6_prim(theta), f_cf = m6_prim(cldfra_bl1);
    MynnArray<float, 1> f_ew = m6_prim(edmf_w1), f_ea = m6_prim(edmf_a1), f_ewd = m6_prim(edmf_w_dd1), f_ead = m6_prim(edmf_a_dd1);
    MynnArray<float, 1> f_el = m6_prim(el), f_qkw = m6_scratch(qkw), f_qtw = m6_scratch(qtw);
    // 2) frozen part: the generated routine on the primal column (el only is kept; f_qkw/f_qtw are recomputed copies).
    {
        ::MynnColumn<1, float, double> pc{arena};
        pc.lane_sub = lane_sub; pc.lane_mask = lane_mask; pc.active_i = active_i; pc.active_j = active_j;
        pc.f_mym_length(kts, kte, m6v(xland), f_dz, m6v(dx), f_zw, m6v(rmol), m6v(rmolh), m6v(flt), m6v(fltv), m6v(flq),
                        f_vt, f_vq, f_u1, f_v1, f_qke, f_qpe, f_dtv, f_el, m6v(pblh), f_theta, f_qkw, f_qtw, m6v(psig_bl),
                        f_cf, bl_mynn_mixlength, f_ew, f_ea, f_ewd, f_ead);
    }
    // 3) live part on the carrier: qkw/qtw, the generated statements of each branch.
    dz.rebase({kts}); qke.rebase({kts}); qpe.rebase({kts}); qkw.rebase({kts}); qtw.rebase({kts}); el.rebase({kts});
    f_el.rebase({kts});
    MR afk, abk;
    qkw(kts) = mynn_sqrt(mynn_max((MR)(qke(kts)), (MR)(qkemin)));
    if (bl_mynn_mixlength != 0) qtw(kts) = mynn_sqrt(mynn_max((MR)(r6_add(qke(kts), qpe(kts))), (MR)(qkemin)));
    { const int lo = kts + 1, n = mynn_trip(lo, kte, 1); mynn_lane_sync();
      for (int it = lane_sub; it < n; it += lane_width) { const int k = lo + it;
        afk = r6_div(dz(k), r6_add(dz(k), dz(((k) - (1)))));
        abk = r6_sub(one, afk);
        qkw(k) = mynn_sqrt(mynn_max((MR)(r6_add(r6_mul(qke(k), abk), r6_mul(qke(((k) - (1))), afk))), (MR)(qkemin)));
        if (bl_mynn_mixlength != 0)
            qtw(k) = mynn_sqrt(mynn_max((MR)(r6_add(r6_mul(r6_add(qke(k), qpe(k)), abk), r6_mul(r6_add(qke(((k) - (1))), qpe(((k) - (1)))), afk))), (MR)(qkemin)));
      } mynn_lane_sync(); }
    if (bl_mynn_mixlength == 0) {
        mynn_lane_sync(); for (int e = lane_sub; e < (qtw).size(); e += lane_width) { qtw.linear(e) = qkw.linear(e); } mynn_lane_sync();
    }
    // 4) el back as passive carriers (the caller freezes it anyway).
    mynn_lane_sync(); for (int e = lane_sub; e < (el).size(); e += lane_width) { el.linear(e) = f_el.linear(e); } mynn_lane_sync();
    }
}
