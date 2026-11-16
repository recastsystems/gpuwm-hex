// SPDX-License-Identifier: Apache-2.0
// x20 MYNN stage M8 (f_mym_predict): hand adjoint as one external tape record (gen_whole.py --fused m8).
//
// Record (taped forward and count pass): the primal is the captured forward's own f_mym_predict run on float copies
// (MynnColumn<1,float,double>, bitwise the forward by construction). Instead of the nodes the operator-overloading
// rules push for the tridiagonal coefficients and the two Thomas solves, the tape gets:
//   6 data nodes per level + 2: the stage's true inputs (primal float and tape id of dz, rho, el, dfq, s_aw, pdk, pdq,
//     pdt, pdc, qke, qsq per level; ust, pmz, delt), two per node (ids in left/right, both floats packed in weight);
//   1 marker node (left = -2, right = n, weight = bl_mynn_edmf), bar preset to 1.0 so the sweep stops at it;
//   one leaf per output word (qke, qsq per level; tsq, cov per level below kte; pdk/pdt/pdq/pdc at kts).
// Reverse (mw_ext_rev, called by MynnAdContext::reverse at the marker): recomputes every forward intermediate in
// float exactly as the forward does (rhoz, rhoinv, kqdz/kmdz with their mass-flux floors, dtz, qkw, bp, rp, the
// tridiagonal coefficients, the Thomas cp/dp and x), then runs the transposed Thomas solve and the coefficient chain
// in double. Derivative weights follow mw_ad_rules.inc (r6_div as mw_quot, sqrt as d/(2s), ipow<3> as 2v^2+v*v;
// mynn_max/min take the executed arm, ties to the second argument). Scratch: a fixed layout at the top of the
// column's arena tile (free after the forward), lane-interleaved like the forward's arrays.
//
// Supported shape (mw_m8_ok): bl_mynn_edmf 0 or 1, bl_mynn_edmf_tke 0 (onoff 0), tke_budget 0, 2.5 < closure < 2.7
// (prognostic qke and qsq, algebraic tsq and cov: the hrrrv5 / RRFS deck), kts 1, delt not active. Anything else
// takes the generated taped routine unchanged (f_mym_predict_taped).
#ifndef MW_M8_PART2
#ifndef MW_M8_PART1_DONE
#define MW_M8_PART1_DONE

template<class T> struct mw_m8_is_tape { static constexpr bool v = false; };
template<> struct mw_m8_is_tape<MynnTapeF> { static constexpr bool v = true; };

template<class S> __device__ __forceinline__ bool mw_m8_ok(int kts, int kte, S closure, S delt, int edmf, int edmf_tke, int budget) {
    return kts == 1 && kte >= 3 && (edmf == 0 || edmf == 1) && edmf_tke == 0 && budget == 0 && closure.v > 2.5f &&
           closure.v < 2.7f && delt.d.id < 0;
}

__device__ __forceinline__ double mw_m8_pk(float a, float b) { return __hiloint2double(__float_as_int(a), __float_as_int(b)); }

// Float copy of a carrier array (arena, lockstep like every automatic array). Absent arrays stay absent.
template<int TILE, class A> __device__ __forceinline__ MynnArray<float, 1> mw_m8_f32(MynnArena& arena, const A& a) {
    MynnArray<float, 1> f;
    f.lo[0] = a.lo[0]; f.hi[0] = a.hi[0]; f.stride[0] = 0; f.data = nullptr;
    if (!a.data) return f;
    f = mynn_array_any<float, 1, TILE>(arena, {a.lo[0]}, {a.hi[0]});
    for (int k = a.lo[0]; k <= a.hi[0]; ++k) f(k) = a(k).v;
    return f;
}

#define MW_M8_ND(n) (6 * (n) + 2)
#define MW_M8_NOUT(n) (4 * (n) + 2)

// Data nodes and the marker. Returns the marker's index in the column tape (or -1 on overflow, failure slot 0 set).
template<class A, class S> __device__ __forceinline__ int mw_m8_record_in(int kte, int edmf, S delt, S ust, S pmz, A dz, A rho,
        A el, A dfq, A saw, A pdk, A pdq, A pdt, A pdc, A qke, A qsq) {
    MynnAdContext* c = mw_ctxp[threadIdx.x & 31];
    const int n = kte, nd = MW_M8_ND(n);
    const int base = c->reserve(nd + 1);
    if (base < 0) return -1;
    if (c->count_only) return base + nd;
    for (int k = 1; k <= n; ++k) {
        const MynnTapeF z(0.f);
        MynnAdNode* o = c->nodes + base + 6 * (k - 1);
        MW_BT* ob = c->bars + base + 6 * (k - 1);
        auto put = [&](int t, const MynnTapeF& x, const MynnTapeF& y) { o[t] = {x.d.id, y.d.id, mw_m8_pk(x.v, y.v)}; ob[t] = 0.0; };
        put(0, dz(k), rho(k));
        put(1, el(k), dfq(k));
        put(2, saw(k), k >= 2 ? pdk(k) : z);
        put(3, k >= 2 ? pdq(k) : z, k >= 2 ? pdt(k) : z);
        put(4, k >= 2 ? pdc(k) : z, qke(k));
        put(5, qsq(k), z);
    }
    c->nodes[base + 6 * n] = {ust.d.id, pmz.d.id, mw_m8_pk(ust.v, pmz.v)};
    c->nodes[base + 6 * n + 1] = {delt.d.id, -1, mw_m8_pk(delt.v, 0.f)};
    c->bars[base + 6 * n] = 0.0; c->bars[base + 6 * n + 1] = 0.0;
    c->nodes[base + nd] = {-2, n, edmf == 1 ? 1.0 : 0.0};
    c->bars[base + nd] = 1.0;   // the sweep skips zero bars; the marker must be visited
    return base + nd;
}

// Output leaves right after the marker; the carriers take the forward's float outputs and the leaf ids.
template<class A, class F> __device__ __forceinline__ void mw_m8_record_out(int marker, int kte, A pdk, A pdt, A pdq, A pdc,
        A qke, A tsq, A qsq, A cov, F fpdk, F fpdt, F fpdq, F fpdc, F fqke, F ftsq, F fqsq, F fcov) {
    MynnAdContext* c = mw_ctxp[threadIdx.x & 31];
    const int n = kte, no = MW_M8_NOUT(n);
    int b = c->reserve(no);
    if (b >= 0 && marker >= 0 && b != marker + 1) { c->fail(2); b = -1; }
    if (b >= 0 && !c->count_only) for (int t = 0; t < no; ++t) { c->nodes[b + t] = {-1, -1, 0.0}; c->bars[b + t] = 0.0; }
    for (int k = 1; k <= n; ++k) qke(k) = MynnTapeF(fqke(k), b < 0 ? MynnLinear() : MynnLinear(b + k - 1, c));
    for (int k = 1; k <= n; ++k) qsq(k) = MynnTapeF(fqsq(k), b < 0 ? MynnLinear() : MynnLinear(b + n + k - 1, c));
    for (int k = 1; k < n; ++k) tsq(k) = MynnTapeF(ftsq(k), b < 0 ? MynnLinear() : MynnLinear(b + 2 * n + k - 1, c));
    for (int k = 1; k < n; ++k) cov(k) = MynnTapeF(fcov(k), b < 0 ? MynnLinear() : MynnLinear(b + 3 * n - 1 + k - 1, c));
    tsq(n) = tsq(n - 1); cov(n) = cov(n - 1);   // the forward copies the carrier: same word, same id
    pdk(1) = MynnTapeF(fpdk(1), b < 0 ? MynnLinear() : MynnLinear(b + 4 * n - 2, c));
    pdt(1) = MynnTapeF(fpdt(1), b < 0 ? MynnLinear() : MynnLinear(b + 4 * n - 1, c));
    pdq(1) = MynnTapeF(fpdq(1), b < 0 ? MynnLinear() : MynnLinear(b + 4 * n, c));
    pdc(1) = MynnTapeF(fpdc(1), b < 0 ? MynnLinear() : MynnLinear(b + 4 * n + 1, c));
}

// ---------------------------------------------------------------------------------------------------- reverse ----
// The tape never records a node with an exact-zero weight (operator*(lin, 0.0), mw_quot and mynn_ad_multiply drop it), so a
// non-finite bar that meets a zero weight contributes nothing there. The hand reverse follows the same rule: x += w * b only
// when w != 0 (0 * inf would otherwise be NaN where the tape has no term at all).
__device__ __forceinline__ void mw_m8_acc(double& x, double w, double b) { if (w != 0.0) x += w * b; }
// Lane-interleaved scratch view (element q: one word per column of the warp tile, like the forward's arrays).
template<class T> struct MwM8V {
    T* p;
    __device__ __forceinline__ T& operator[](int q) const { return p[q * MW_TILE]; }
};

// Transposed Thomas solve of the float algorithm f_tridiag2 (rows 1..n). In: a,b,c,d (float coefficients), cp,dp,x
// (float, recomputed), xb (double output bar, consumed). Out: ab, bb, cb, db (double, overwritten).
__device__ __forceinline__ void mw_m8_thomas_rev(int n, MwM8V<float> a, MwM8V<float> b, MwM8V<float> c, MwM8V<float> d,
        MwM8V<float> cp, MwM8V<float> dp, MwM8V<float> x, MwM8V<double> xb, MwM8V<double> cpb, MwM8V<double> dpb,
        MwM8V<double> ab, MwM8V<double> bb, MwM8V<double> cb, MwM8V<double> db) {
    for (int k = 1; k <= n; ++k) { cpb[k] = 0.0; dpb[k] = 0.0; ab[k] = 0.0; bb[k] = 0.0; cb[k] = 0.0; db[k] = 0.0; }
    // back substitution x(k) = dp(k) - cp(k) * x(k+1), k = n-1..1, reversed
    for (int k = 1; k <= n - 1; ++k) {
        const double g = xb[k];
        if (g == 0.0) continue;
        dpb[k] += g;
        const double pb = -g;
        mw_m8_acc(cpb[k], (double)x[k + 1], pb);
        mw_m8_acc(xb[k + 1], (double)cp[k], pb);
    }
    dpb[n] += xb[n];
    for (int k = n; k >= 2; --k) {
        const float ak = a[k], cpm = cp[k - 1], dpm = dp[k - 1];
        const float m = r6_sub(b[k], r6_mul(cpm, ak));
        const float num = r6_sub(d[k], r6_mul(dpm, ak));
        const double inv = 1.0 / (double)m;
        double mb = 0.0;
        const double tq = dpb[k] * inv;             // dp(k) = num / m
        if (tq != 0.0) {
            const double ub = -tq;                  // num = d(k) - dp(k-1) * a(k)
            db[k] += tq;
            mw_m8_acc(dpb[k - 1], (double)ak, ub);
            mw_m8_acc(ab[k], (double)dpm, ub);
            mw_m8_acc(mb, (double)num / (double)m, -tq);
        }
        const double tc = cpb[k] * inv;             // cp(k) = c(k) / m
        if (tc != 0.0) {
            cb[k] += tc;
            mw_m8_acc(mb, (double)c[k] / (double)m, -tc);
        }
        if (mb != 0.0) {                            // m = b(k) - cp(k-1) * a(k)
            bb[k] += mb;
            const double tb = -mb;
            mw_m8_acc(cpb[k - 1], (double)ak, tb);
            mw_m8_acc(ab[k], (double)cpm, tb);
        }
    }
    const double inv1 = 1.0 / (double)b[1];
    const double tc1 = cpb[1] * inv1, td1 = dpb[1] * inv1;
    cb[1] += tc1; db[1] += td1;
    mw_m8_acc(bb[1], (double)c[1] / (double)b[1], -tc1); mw_m8_acc(bb[1], (double)d[1] / (double)b[1], -td1);
}

#endif  // part 1
#else   // part 2: after the MynnColumn body (its constants)
#ifdef MW_EXT_REV
__device__ __noinline__ void mw_ext_rev(MynnAdContext* ctx, int M) {
    using MC = MynnColumn<1, float, double>;
    const float qkemin = MC::qkemin, b1 = MC::b1, b2 = MC::b2, sqfac = MC::sqfac, karman = MC::karman, p5 = MC::p5,
                two = MC::two, one = MC::one, zero = MC::zero, three = MC::three;
    const int n = ctx->nodes[M].right;
    const int edmf = ctx->nodes[M].weight != 0.0;
    const int base = M - MW_M8_ND(n);
    const int L0 = M + 1;
    const MynnAdNode* dn = ctx->nodes + base;
    MW_BT* bars = ctx->bars;
    // data: field s (0 dz, 1 rho, 2 el, 3 dfq, 4 saw, 5 pdk, 6 pdq, 7 pdt, 8 pdc, 9 qke, 10 qsq) at level k
    auto F = [&](int s, int k) -> float { const double w = dn[6 * (k - 1) + (s >> 1)].weight;
        return __int_as_float((s & 1) ? __double2loint(w) : __double2hiint(w)); };
    auto I = [&](int s, int k) -> int { const MynnAdNode& q = dn[6 * (k - 1) + (s >> 1)]; return (s & 1) ? q.right : q.left; };
    const MynnAdNode sc0 = dn[6 * n], sc1 = dn[6 * n + 1];
    const float ust = __int_as_float(__double2hiint(sc0.weight)), pmz = __int_as_float(__double2loint(sc0.weight));
    const float delt = __int_as_float(__double2hiint(sc1.weight));
    const int iust = sc0.left, ipmz = sc0.right;
    // Every term of this reverse carries one of the output leaves' bars, so with all of them zero (the window's case under
    // freeze=closure: no cotangent reaches qke/qsq/tsq/cov/pdk(kts)) the stage adds nothing and costs one pass over the leaves.
    { bool any = false; for (int t = 0; t < MW_M8_NOUT(n) && !any; ++t) any = bars[L0 + t] != 0.0; if (!any) return; }

    // scratch at the top of the column's arena tile (free after the forward); fixed layout, so lanes never collide
    // Each allocation starts 8-byte aligned (MynnArena::allocate), so a float array takes (4*S + 7) & ~7 logical bytes.
    // The region must end inside this column's arena slot: past it is the next block's tile, whose forward may still run
    // (x20 M8 fix: the first cut left out that padding, overran by up to 48 B and the top levels of pdcb came back NaN
    // on pk120).
    const int NF = 16, ND = 22, S = n + 2;
    const unsigned long long need = (unsigned long long)NF * ((4ULL * S + 7ULL) & ~7ULL) + (unsigned long long)ND * S * 8ULL;
    if (!ctx->scr || need + 64ULL > ctx->scr_cap) { ctx->fail(3); return; }
    MynnArena ar{ctx->scr, (ctx->scr_cap - need - 64ULL) & ~7ULL, MW_TILE, ctx->scr_lane};
    auto vf = [&]() { MwM8V<float> v{static_cast<float*>(ar.allocate(4ULL * S, 4))}; return v; };
    auto vd = [&]() { MwM8V<double> v{static_cast<double*>(ar.allocate(8ULL * S, 8))}; return v; };
    MwM8V<float> dtz = vf(), rhoinv = vf(), rhoz = vf(), kq = vf(), km = vf(), qkw = vf(), A = vf(), B = vf(), C = vf(),
                 D = vf(), CP = vf(), DP = vf(), X = vf(), src = vf(), pdU = vf(), rz0 = vf();
    MwM8V<double> xb = vd(), cpb = vd(), dpb = vd(), ab = vd(), bb = vd(), cb = vd(), db = vd(), kqb = vd(), kmb = vd(),
                  dtzb = vd(), rhoinvb = vd(), rhozb = vd(), qkwb = vd(), elb = vd(), dzb = vd(), rhob = vd(), sawb = vd(),
                  dfqb = vd(), pdkb = vd(), pdqb = vd(), pdtb = vd(), pdcb = vd();
    if (ar.offset > ctx->scr_cap) { ctx->fail(3); return; }   // never reached with the sizes above; a refusal, not a write past the slot
    double qkeb_n = 0.0, ustb = 0.0, pmzb = 0.0;
    for (int k = 0; k <= n + 1; ++k) {
        kqb[k] = 0.0; kmb[k] = 0.0; dtzb[k] = 0.0; rhoinvb[k] = 0.0; rhozb[k] = 0.0; qkwb[k] = 0.0; elb[k] = 0.0;
        dzb[k] = 0.0; rhob[k] = 0.0; sawb[k] = 0.0; dfqb[k] = 0.0; pdkb[k] = 0.0; pdqb[k] = 0.0; pdtb[k] = 0.0; pdcb[k] = 0.0;
        src[k] = 0.f; rz0[k] = 0.f;
    }

    // ---- forward recompute of the shared coefficients (float, the forward's exact operation order)
    for (int k = 1; k <= n; ++k) {
        const float q = F(9, k);
        qkw[k] = mynn_sqrt(q > qkemin ? q : qkemin);
        dtz[k] = r6_div(delt, F(0, k));
    }
    rhoz[1] = F(1, 1);
    rhoinv[1] = r6_div(one, F(1, 1));
    kq[1] = r6_mul(rhoz[1], r6_mul(sqfac, F(3, 1)));
    km[1] = r6_mul(rhoz[1], F(3, 1));
    for (int k = 2; k <= n; ++k) {
        const float r = r6_div(r6_add(r6_mul(F(1, k), F(0, k - 1)), r6_mul(F(1, k - 1), F(0, k))), r6_add(F(0, k - 1), F(0, k)));
        rz0[k] = r;
        rhoz[k] = r > 1e-4f ? r : 1e-4f;
        const float rr = F(1, k);
        rhoinv[k] = r6_div(one, rr > 1e-4f ? rr : 1e-4f);
        kq[k] = r6_mul(rhoz[k], r6_mul(sqfac, F(3, k)));
        km[k] = r6_mul(rhoz[k], F(3, k));
    }
    rhoz[n + 1] = rhoz[n];
    kq[n + 1] = r6_mul(rhoz[n + 1], r6_mul(sqfac, F(3, n)));
    km[n + 1] = r6_mul(rhoz[n + 1], F(3, n));
    // mass-flux floors; source per level: bits 0-1 kqdz, bits 2-3 kmdz (0 own, 1 0.5*s_aw(k), 2 -0.5*(s_aw(k)-s_aw(k+1)))
    if (edmf) for (int k = 2; k <= n - 1; ++k) {
        const float s1 = r6_mul(p5, F(4, k));
        const float s2 = r6_neg(r6_mul(p5, r6_sub(F(4, k), F(4, k + 1))));
        int code = 0;
        float v = kq[k];
        if (!(v > s1)) { v = s1; code = 1; }
        if (!(v > s2)) { v = s2; code = 2; }
        kq[k] = v;
        float w = km[k]; int cm = 0;
        if (!(w > s1)) { w = s1; cm = 1; }
        if (!(w > s2)) { w = s2; cm = 2; }
        km[k] = w;
        src[k] = __int_as_float(code | (cm << 2));
    }
    const float vkz = r6_mul(r6_mul(karman, p5), F(0, 1));
    const float P3 = mynn_ipow<3>(ust);
    const float Aa = r6_mul(two, P3);
    const float Bb = r6_mul(Aa, pmz);
    const float pdk1 = r6_div(Bb, vkz);
    if (ust == 0.0f && iust >= 0) ctx->fail(1);     // r6_div(two, ust) of the dead phm branch, as the rules refuse

    // output bars
    for (int k = 0; k <= n + 1; ++k) xb[k] = 0.0;
    double pdkU1b = bars[L0 + 4 * n - 2], pdtU1b = bars[L0 + 4 * n - 1], pdqU1b = bars[L0 + 4 * n], pdcU1b = bars[L0 + 4 * n + 1];

    // ---- tsq, cov (algebraic for closure < 2.7 / < 3.0): b2l = b2*0.25*(el(k+1)+el(k)) / qkw(k); pdt/pdc(1) := (2)
    for (int k = 1; k <= n - 1; ++k) {
        const double tb = bars[L0 + 2 * n + k - 1], cvb = bars[L0 + 3 * n - 1 + k - 1];
        if (tb == 0.0 && cvb == 0.0) continue;
        if (qkw[k] <= zero) continue;
        const float num2 = r6_mul(r6_mul(b2, 0.25f), r6_add(F(2, k + 1), F(2, k)));
        const float b2l = r6_div(num2, qkw[k]);
        const float pt0 = k >= 2 ? F(7, k) : F(7, 2), pc0 = k >= 2 ? F(8, k) : F(8, 2);
        const float st = r6_add(F(7, k + 1), pt0), scv = r6_add(F(8, k + 1), pc0);
        const float prod = r6_mul(b2l, st);
        double b2lb = 0.0;
        if (tb != 0.0 && prod > zero && prod < three) {
            mw_m8_acc(b2lb, (double)st, tb);
            double sb = 0.0; mw_m8_acc(sb, (double)b2l, tb);
            pdtb[k + 1] += sb;
            if (k >= 2) pdtb[k] += sb; else pdtU1b += sb;
        }
        if (cvb != 0.0) {
            mw_m8_acc(b2lb, (double)scv, cvb);
            double sb = 0.0; mw_m8_acc(sb, (double)b2l, cvb);
            pdcb[k + 1] += sb;
            if (k >= 2) pdcb[k] += sb; else pdcU1b += sb;
        }
        if (b2lb != 0.0) {
            const double tq = b2lb * (1.0 / (double)qkw[k]);
            mw_m8_acc(qkwb[k], (double)num2 / (double)qkw[k], -tq);
            const double e = tq * (double)r6_mul(b2, 0.25f);
            elb[k + 1] += e; elb[k] += e;
        }
    }

    // ---- the two implicit systems: s = 0 qsq (kmdz, b2, pdq; bottom row a=-1, d=0), s = 1 qke (kqdz, b1, pdk; d=qke)
    for (int s = 0; s < 2; ++s) {
        MwM8V<float> K = s ? kq : km;
        MwM8V<double> Kb = s ? kqb : kmb;
        MwM8V<double> pb = s ? pdkb : pdqb;
        const float bl0 = s ? r6_mul(b1, p5) : r6_mul(b2, p5);
        const int fq = s ? 9 : 10, fp = s ? 5 : 6;
        pdU[1] = s ? r6_sub(pdk1, F(5, 2)) : F(6, 2);
        for (int k = 2; k <= n; ++k) pdU[k] = F(fp, k);
        for (int k = 1; k <= n - 1; ++k) {
            const float bl = r6_mul(bl0, r6_add(F(2, k + 1), F(2, k)));
            const float bp = r6_div(r6_mul(two, qkw[k]), bl);
            const float rp = r6_add(pdU[k + 1], pdU[k]);
            A[k] = r6_mul(r6_neg(r6_mul(dtz[k], K[k])), rhoinv[k]);
            B[k] = r6_add(r6_add(one, r6_mul(r6_mul(dtz[k], r6_add(K[k], K[k + 1])), rhoinv[k])), r6_mul(bp, delt));
            C[k] = r6_mul(r6_neg(r6_mul(dtz[k], K[k + 1])), rhoinv[k]);
            D[k] = r6_add(r6_mul(rp, delt), F(fq, k));
        }
        A[n] = s ? zero : r6_neg(one); B[n] = one; C[n] = zero; D[n] = s ? F(9, n) : zero;
        CP[1] = r6_div(C[1], B[1]); DP[1] = r6_div(D[1], B[1]);
        for (int k = 2; k <= n; ++k) {
            const float m = r6_sub(B[k], r6_mul(CP[k - 1], A[k]));
            CP[k] = r6_div(C[k], m);
            DP[k] = r6_div(r6_sub(D[k], r6_mul(DP[k - 1], A[k])), m);
        }
        X[n] = DP[n];
        for (int k = n - 1; k >= 1; --k) X[k] = r6_sub(DP[k], r6_mul(CP[k], X[k + 1]));
        bool any = false;
        for (int k = 1; k <= n; ++k) {
            const float x = X[k];
            const double ob = bars[L0 + (s ? 0 : n) + k - 1];
            // qke = min(max(x, qkemin), 150); qsq = min(max(x, 1e-17), 6e-6): the derivative passes on the x arm only
            const bool pass = s ? (x > qkemin && x < 150.f) : (x > 1e-17f && x < 6e-6f);
            xb[k] = pass ? ob : 0.0;
            any = any || xb[k] != 0.0;
        }
        if (!any) continue;
        mw_m8_thomas_rev(n, A, B, C, D, CP, DP, X, xb, cpb, dpb, ab, bb, cb, db);
        if (s) qkeb_n += db[n];                      // d(kte) = qke(kte)
        for (int k = 1; k <= n - 1; ++k) {
            const double abk = k >= 2 ? ab[k] : 0.0, bbk = bb[k], cbk = cb[k], dbk = db[k];
            if (abk != 0.0) {                        // a(k) = -(dtz*K(k)) * rhoinv
                const float t1 = r6_mul(dtz[k], K[k]);
                mw_m8_acc(rhoinvb[k], (double)r6_neg(t1), abk);
                const double t1b = -(abk * (double)rhoinv[k]);
                mw_m8_acc(dtzb[k], (double)K[k], t1b); mw_m8_acc(Kb[k], (double)dtz[k], t1b);
            }
            if (bbk != 0.0) {                        // b(k) = (1 + (dtz*(K(k)+K(k+1)))*rhoinv) + bp*delt
                const float Ssum = r6_add(K[k], K[k + 1]);
                const float u = r6_mul(dtz[k], Ssum);
                mw_m8_acc(rhoinvb[k], (double)u, bbk);
                const double ub = bbk * (double)rhoinv[k];
                mw_m8_acc(dtzb[k], (double)Ssum, ub);
                mw_m8_acc(Kb[k], (double)dtz[k], ub); mw_m8_acc(Kb[k + 1], (double)dtz[k], ub);
                const double bpb = bbk * (double)delt;   // bp = (2*qkw) / bl, bl = bl0*(el(k+1)+el(k))
                const float bl = r6_mul(bl0, r6_add(F(2, k + 1), F(2, k)));
                const float a2 = r6_mul(two, qkw[k]);
                const double tq = bpb * (1.0 / (double)bl);
                qkwb[k] += tq * 2.0;
                const double e = (((double)a2 / (double)bl) * (-tq)) * (double)bl0;
                elb[k + 1] += e; elb[k] += e;
            }
            if (cbk != 0.0) {                        // c(k) = -(dtz*K(k+1)) * rhoinv
                const float t2 = r6_mul(dtz[k], K[k + 1]);
                mw_m8_acc(rhoinvb[k], (double)r6_neg(t2), cbk);
                const double t2b = -(cbk * (double)rhoinv[k]);
                mw_m8_acc(dtzb[k], (double)K[k + 1], t2b); mw_m8_acc(Kb[k + 1], (double)dtz[k], t2b);
            }
            if (dbk != 0.0) {                        // d(k) = rp*delt + q(k), rp = pdU(k+1) + pdU(k)
                const double rpb = dbk * (double)delt;
                pb[k + 1] += rpb;
                if (k >= 2) pb[k] += rpb; else { if (s) pdkU1b += rpb; else pdqU1b += rpb; }
                const int iq = I(fq, k);
                if (iq >= 0) bars[iq] += dbk;
            }
        }
    }
    { const int iq = I(9, n); if (iq >= 0 && qkeb_n != 0.0) bars[iq] += qkeb_n; }

    // ---- pdk(kts) = pdk1 - pdk(kts+1); pdt/pdq/pdc(kts) = their (kts+1)
    pdkb[2] -= pdkU1b; pdtb[2] += pdtU1b; pdqb[2] += pdqU1b; pdcb[2] += pdcU1b;
    if (pdkU1b != 0.0) {   // pdk1 = ((2*ust^3)*pmz) / vkz, vkz = (karman*p5)*dz(kts)
        const double tq = pdkU1b * (1.0 / (double)vkz);
        double vkzb = 0.0; mw_m8_acc(vkzb, (double)Bb / (double)vkz, -tq);
        mw_m8_acc(dzb[1], (double)r6_mul(karman, p5), vkzb);
        mw_m8_acc(pmzb, (double)Aa, tq);
        double Ab = 0.0; mw_m8_acc(Ab, (double)pmz, tq);
        const double P3b = Ab * 2.0;
        const double v = (double)ust;
        mw_m8_acc(ustb, 2.0 * v * v + (double)r6_mul(ust, ust), P3b);
    }

    // ---- kqdz / kmdz: own (rhoz*df3q, rhoz*dfq) or the mass-flux floor; k(kte+1) = rhoz(kte)*df(kte)
    for (int k = 1; k <= n + 1; ++k) {
        const int kk = k <= n ? k : n;
        const int code = (edmf && k >= 2 && k <= n - 1) ? __float_as_int(src[k]) : 0;
        const double gq = kqb[k], gm = kmb[k];
        if (gq != 0.0) {
            const int cq = code & 3;
            if (cq == 0) { mw_m8_acc(rhozb[kk], (double)r6_mul(sqfac, F(3, kk)), gq); double t = 0.0; mw_m8_acc(t, (double)rhoz[kk], gq); mw_m8_acc(dfqb[kk], (double)sqfac, t); }
            else if (cq == 1) sawb[k] += gq * (double)p5;
            else { const double t = -(gq * (double)p5); sawb[k] += t; sawb[k + 1] -= t; }
        }
        if (gm != 0.0) {
            const int cm = (code >> 2) & 3;
            if (cm == 0) { mw_m8_acc(rhozb[kk], (double)F(3, kk), gm); mw_m8_acc(dfqb[kk], (double)rhoz[kk], gm); }
            else if (cm == 1) sawb[k] += gm * (double)p5;
            else { const double t = -(gm * (double)p5); sawb[k] += t; sawb[k + 1] -= t; }
        }
    }
    // ---- rhoz, rhoinv, dtz, qkw
    rhob[1] += rhozb[1];
    for (int k = 1; k <= n; ++k) {
        if (k >= 2 && rhozb[k] != 0.0 && rz0[k] > 1e-4f) {
            const float num = r6_add(r6_mul(F(1, k), F(0, k - 1)), r6_mul(F(1, k - 1), F(0, k)));
            const float den = r6_add(F(0, k - 1), F(0, k));
            const double tq = rhozb[k] * (1.0 / (double)den);
            const double denb = ((double)num / (double)den) * (-tq);
            mw_m8_acc(rhob[k], (double)F(0, k - 1), tq); mw_m8_acc(dzb[k - 1], (double)F(1, k), tq);
            mw_m8_acc(rhob[k - 1], (double)F(0, k), tq); mw_m8_acc(dzb[k], (double)F(1, k - 1), tq);
            dzb[k - 1] += denb; dzb[k] += denb;
        }
        const float r = F(1, k);
        if (rhoinvb[k] != 0.0 && (k == 1 || r > 1e-4f)) {
            const double tq = rhoinvb[k] * (1.0 / (double)r);
            rhob[k] += ((double)one / (double)r) * (-tq);
        }
        if (dtzb[k] != 0.0) {
            const float z = F(0, k);
            const double tq = dtzb[k] * (1.0 / (double)z);
            dzb[k] += ((double)delt / (double)z) * (-tq);
        }
        if (qkwb[k] != 0.0 && F(9, k) > qkemin) {
            const int iq = I(9, k);
            if (iq >= 0) bars[iq] += qkwb[k] * (1.0 / (2.0 * (double)qkw[k]));
        }
    }
    // ---- scatter to the input words
    for (int k = 1; k <= n; ++k) {
        int id;
        if ((id = I(0, k)) >= 0 && dzb[k] != 0.0) bars[id] += dzb[k];
        if ((id = I(1, k)) >= 0 && rhob[k] != 0.0) bars[id] += rhob[k];
        if ((id = I(2, k)) >= 0 && elb[k] != 0.0) bars[id] += elb[k];
        if ((id = I(3, k)) >= 0 && dfqb[k] != 0.0) bars[id] += dfqb[k];
        if ((id = I(4, k)) >= 0 && sawb[k] != 0.0) bars[id] += sawb[k];
        if (k >= 2) {
            if ((id = I(5, k)) >= 0 && pdkb[k] != 0.0) bars[id] += pdkb[k];
            if ((id = I(6, k)) >= 0 && pdqb[k] != 0.0) bars[id] += pdqb[k];
            if ((id = I(7, k)) >= 0 && pdtb[k] != 0.0) bars[id] += pdtb[k];
            if ((id = I(8, k)) >= 0 && pdcb[k] != 0.0) bars[id] += pdcb[k];
        }
    }
    if (iust >= 0 && ustb != 0.0) bars[iust] += ustb;
    if (ipmz >= 0 && pmzb != 0.0) bars[ipmz] += pmzb;
}
#endif
#endif  // part 2
