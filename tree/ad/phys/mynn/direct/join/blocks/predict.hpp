// Direct adaptation of repository mw_m8_predict.cuh hand reverse. No tape/arena.
#pragma once
#include "numeric.hpp"
namespace direct {
template<class P> inline void thomas_reverse(int n, const P* a, const P* b, const P* c, const P* d,
        const P* cp, const P* dp, const P* x, double* xb, double* cpb, double* dpb,
        double* ab, double* bb, double* cb, double* db) {
    for (int k = 1; k <= n; ++k) { cpb[k] = 0.0; dpb[k] = 0.0; ab[k] = 0.0; bb[k] = 0.0; cb[k] = 0.0; db[k] = 0.0; }
    // back substitution x(k) = dp(k) - cp(k) * x(k+1), k = n-1..1, reversed
    for (int k = 1; k <= n - 1; ++k) {
        const double g = xb[k];
        if (g == 0.0) continue;
        dpb[k] += g;
        const double pb = -g;
        acc(cpb[k], (double)x[k + 1], pb);
        acc(xb[k + 1], (double)cp[k], pb);
    }
    dpb[n] += xb[n];
    for (int k = n; k >= 2; --k) {
        const P ak = a[k], cpm = cp[k - 1], dpm = dp[k - 1];
        const P m = r6_sub(b[k], r6_mul(cpm, ak));
        const P num = r6_sub(d[k], r6_mul(dpm, ak));
        const double inv = 1.0 / (double)m;
        double mb = 0.0;
        const double tq = dpb[k] * inv;             // dp(k) = num / m
        if (tq != 0.0) {
            const double ub = -tq;                  // num = d(k) - dp(k-1) * a(k)
            db[k] += tq;
            acc(dpb[k - 1], (double)ak, ub);
            acc(ab[k], (double)dpm, ub);
            acc(mb, (double)num / (double)m, -tq);
        }
        const double tc = cpb[k] * inv;             // cp(k) = c(k) / m
        if (tc != 0.0) {
            cb[k] += tc;
            acc(mb, (double)c[k] / (double)m, -tc);
        }
        if (mb != 0.0) {                            // m = b(k) - cp(k-1) * a(k)
            bb[k] += mb;
            const double tb = -mb;
            acc(cpb[k - 1], (double)ak, tb);
            acc(ab[k], (double)cpm, tb);
        }
    }
    const double inv1 = 1.0 / (double)b[1];
    const double tc1 = cpb[1] * inv1, td1 = dpb[1] * inv1;
    cb[1] += tc1; db[1] += td1;
    acc(bb[1], (double)c[1] / (double)b[1], -tc1); acc(bb[1], (double)d[1] / (double)b[1], -td1);
}

// Input profile order: dz,rho,el,dfq,saw,pdk,pdq,pdt,pdc,qke,qsq; stride MAX=61.
// Additional input scalars ust,pmz follow 11*MAX. dt and edmf are constants.
// Seed layout: qke[n],qsq[n],tsq[n-1],cov[n-1],pdk1,pdt1,pdq1,pdc1.
// Caller folds top tsq/cov seeds into level n-1 and passes through p*d levels2..n.
// Additive bar output; no alias with input or seed. 3<=n<=59; production 2.5<closure<2.7.
template<class P> bool predict_reverse(int n,int edmf,P delt,const P*input,const double*seed,double*bar,int planted=0) {
 if(n<3||n>59||(edmf!=0&&edmf!=1))return false;
 constexpr P qkemin=P(1e-5f),b1=P(24.f),b2=P(15.f),sqfac=P(3.f),karman=P(.4f),p5=P(.5f),two=P(2),one=P(1),zero=P(0),three=P(3);
 auto F=[&](int s,int k){return input[s*MAX+k];};
 auto I=[](int s,int k){return s*MAX+k;};
 const int iust=11*MAX,ipmz=iust+1; const P ust=input[iust],pmz=input[ipmz];
 P dtz[MAX]{},rhoinv[MAX]{},rhoz[MAX]{},kq[MAX]{},km[MAX]{},qkw[MAX]{},A[MAX]{},B[MAX]{},C[MAX]{},D[MAX]{},CP[MAX]{},DP[MAX]{},X[MAX]{},src[MAX]{},pdU[MAX]{},rz0[MAX]{};
 double xb[MAX]{},cpb[MAX]{},dpb[MAX]{},ab[MAX]{},bb[MAX]{},cb[MAX]{},db[MAX]{},kqb[MAX]{},kmb[MAX]{},dtzb[MAX]{},rhoinvb[MAX]{},rhozb[MAX]{},qkwb[MAX]{},elb[MAX]{},dzb[MAX]{},rhob[MAX]{},sawb[MAX]{},dfqb[MAX]{},pdkb[MAX]{},pdqb[MAX]{},pdtb[MAX]{},pdcb[MAX]{};
 double qkeb_n=0,ustb=0,pmzb=0;
    // ---- forward recompute of the shared coefficients (float, the forward's exact operation order)
    for (int k = 1; k <= n; ++k) {
        const P q = F(9, k);
        qkw[k] = mynn_sqrt(q > qkemin ? q : qkemin);
        dtz[k] = r6_div(delt, F(0, k));
    }
    rhoz[1] = F(1, 1);
    rhoinv[1] = r6_div(one, F(1, 1));
    kq[1] = r6_mul(rhoz[1], r6_mul(sqfac, F(3, 1)));
    km[1] = r6_mul(rhoz[1], F(3, 1));
    for (int k = 2; k <= n; ++k) {
        const P r = r6_div(r6_add(r6_mul(F(1, k), F(0, k - 1)), r6_mul(F(1, k - 1), F(0, k))), r6_add(F(0, k - 1), F(0, k)));
        rz0[k] = r;
        rhoz[k] = r > 1e-4f ? r : 1e-4f;
        const P rr = F(1, k);
        rhoinv[k] = r6_div(one, rr > 1e-4f ? rr : 1e-4f);
        kq[k] = r6_mul(rhoz[k], r6_mul(sqfac, F(3, k)));
        km[k] = r6_mul(rhoz[k], F(3, k));
    }
    rhoz[n + 1] = rhoz[n];
    kq[n + 1] = r6_mul(rhoz[n + 1], r6_mul(sqfac, F(3, n)));
    km[n + 1] = r6_mul(rhoz[n + 1], F(3, n));
    // mass-flux floors; source per level: bits 0-1 kqdz, bits 2-3 kmdz (0 own, 1 0.5*s_aw(k), 2 -0.5*(s_aw(k)-s_aw(k+1)))
    if (edmf) for (int k = 2; k <= n - 1; ++k) {
        const P s1 = r6_mul(p5, F(4, k));
        const P s2 = r6_neg(r6_mul(p5, r6_sub(F(4, k), F(4, k + 1))));
        int code = 0;
        P v = kq[k];
        if (!(v > s1)) { v = s1; code = 1; }
        if (!(v > s2)) { v = s2; code = 2; }
        kq[k] = v;
        P w = km[k]; int cm = 0;
        if (!(w > s1)) { w = s1; cm = 1; }
        if (!(w > s2)) { w = s2; cm = 2; }
        km[k] = w;
        src[k] = P(code | (cm << 2));
    }
    const P vkz = r6_mul(r6_mul(karman, p5), F(0, 1));
    const P P3 = mynn_ipow<3>(ust);
    const P Aa = r6_mul(two, P3);
    const P Bb = r6_mul(Aa, pmz);
    const P pdk1 = r6_div(Bb, vkz);
    if (ust == 0) return false;     // r6_div(two, ust) of the dead phm branch, as the rules refuse

    // output bars
    for (int k = 0; k <= n + 1; ++k) xb[k] = 0.0;
    double pdkU1b = seed[4 * n - 2], pdtU1b = seed[4 * n - 1], pdqU1b = seed[4 * n], pdcU1b = seed[4 * n + 1];

    // ---- tsq, cov (algebraic for closure < 2.7 / < 3.0): b2l = b2*0.25*(el(k+1)+el(k)) / qkw(k); pdt/pdc(1) := (2)
    for (int k = 1; k <= n - 1; ++k) {
        const double tb = seed[2 * n + k - 1], cvb = seed[3 * n - 1 + k - 1];
        if (tb == 0.0 && cvb == 0.0) continue;
        if (qkw[k] <= zero) continue;
        const P num2 = r6_mul(r6_mul(b2, 0.25f), r6_add(F(2, k + 1), F(2, k)));
        const P b2l = r6_div(num2, qkw[k]);
        const P pt0 = k >= 2 ? F(7, k) : F(7, 2), pc0 = k >= 2 ? F(8, k) : F(8, 2);
        const P st = r6_add(F(7, k + 1), pt0), scv = r6_add(F(8, k + 1), pc0);
        const P prod = r6_mul(b2l, st);
        double b2lb = 0.0;
        if (tb != 0.0 && prod > zero && prod < three) {
            acc(b2lb, (double)st, tb);
            double sb = 0.0; acc(sb, (double)b2l, tb);
            pdtb[k + 1] += sb;
            if (k >= 2) pdtb[k] += sb; else pdtU1b += sb;
        }
        if (cvb != 0.0) {
            acc(b2lb, (double)scv, cvb);
            double sb = 0.0; acc(sb, (double)b2l, cvb);
            pdcb[k + 1] += sb;
            if (k >= 2) pdcb[k] += sb; else pdcU1b += sb;
        }
        if (b2lb != 0.0) {
            const double tq = b2lb * (1.0 / (double)qkw[k]);
            acc(qkwb[k], (double)num2 / (double)qkw[k], -tq);
            const double e = tq * (double)r6_mul(b2, 0.25f);
            elb[k + 1] += e; elb[k] += e;
        }
    }

    // ---- the two implicit systems: s = 0 qsq (kmdz, b2, pdq; bottom row a=-1, d=0), s = 1 qke (kqdz, b1, pdk; d=qke)
    for (int s = 0; s < 2; ++s) {
        P* K = s ? kq : km;
        double* Kb = s ? kqb : kmb;
        double* pb = s ? pdkb : pdqb;
        const P bl0 = s ? r6_mul(b1, p5) : r6_mul(b2, p5);
        const int fq = s ? 9 : 10, fp = s ? 5 : 6;
        pdU[1] = s ? r6_sub(pdk1, F(5, 2)) : F(6, 2);
        for (int k = 2; k <= n; ++k) pdU[k] = F(fp, k);
        for (int k = 1; k <= n - 1; ++k) {
            const P bl = r6_mul(bl0, r6_add(F(2, k + 1), F(2, k)));
            const P bp = r6_div(r6_mul(two, qkw[k]), bl);
            const P rp = r6_add(pdU[k + 1], pdU[k]);
            A[k] = r6_mul(r6_neg(r6_mul(dtz[k], K[k])), rhoinv[k]);
            B[k] = r6_add(r6_add(one, r6_mul(r6_mul(dtz[k], r6_add(K[k], K[k + 1])), rhoinv[k])), r6_mul(bp, delt));
            C[k] = r6_mul(r6_neg(r6_mul(dtz[k], K[k + 1])), rhoinv[k]);
            D[k] = r6_add(r6_mul(rp, delt), F(fq, k));
        }
        A[n] = s ? zero : r6_neg(one); B[n] = one; C[n] = zero; D[n] = s ? F(9, n) : zero;
        CP[1] = r6_div(C[1], B[1]); DP[1] = r6_div(D[1], B[1]);
        for (int k = 2; k <= n; ++k) {
            const P m = r6_sub(B[k], r6_mul(CP[k - 1], A[k]));
            CP[k] = r6_div(C[k], m);
            DP[k] = r6_div(r6_sub(D[k], r6_mul(DP[k - 1], A[k])), m);
        }
        X[n] = DP[n];
        for (int k = n - 1; k >= 1; --k) X[k] = r6_sub(DP[k], r6_mul(CP[k], X[k + 1]));
        bool any = false;
        for (int k = 1; k <= n; ++k) {
            const P x = X[k];
            const double ob = seed[(s ? 0 : n) + k - 1];
            // qke = min(max(x, qkemin), 150); qsq = min(max(x, 1e-17), 6e-6): the derivative passes on the x arm only
            const bool pass = s ? (x > qkemin && x < 150.f) : (x > 1e-17f && x < 6e-6f);
            xb[k] = pass ? ob : 0.0;
            any = any || xb[k] != 0.0;
        }
        if (!any) continue;
        thomas_reverse(n, A, B, C, D, CP, DP, X, xb, cpb, dpb, ab, bb, cb, db);
        if(planted && s) db[n/2]*=1.001;
        if (s) qkeb_n += db[n];                      // d(kte) = qke(kte)
        for (int k = 1; k <= n - 1; ++k) {
            const double abk = k >= 2 ? ab[k] : 0.0, bbk = bb[k], cbk = cb[k], dbk = db[k];
            if (abk != 0.0) {                        // a(k) = -(dtz*K(k)) * rhoinv
                const P t1 = r6_mul(dtz[k], K[k]);
                acc(rhoinvb[k], (double)r6_neg(t1), abk);
                const double t1b = -(abk * (double)rhoinv[k]);
                acc(dtzb[k], (double)K[k], t1b); acc(Kb[k], (double)dtz[k], t1b);
            }
            if (bbk != 0.0) {                        // b(k) = (1 + (dtz*(K(k)+K(k+1)))*rhoinv) + bp*delt
                const P Ssum = r6_add(K[k], K[k + 1]);
                const P u = r6_mul(dtz[k], Ssum);
                acc(rhoinvb[k], (double)u, bbk);
                const double ub = bbk * (double)rhoinv[k];
                acc(dtzb[k], (double)Ssum, ub);
                acc(Kb[k], (double)dtz[k], ub); acc(Kb[k + 1], (double)dtz[k], ub);
                const double bpb = bbk * (double)delt;   // bp = (2*qkw) / bl, bl = bl0*(el(k+1)+el(k))
                const P bl = r6_mul(bl0, r6_add(F(2, k + 1), F(2, k)));
                const P a2 = r6_mul(two, qkw[k]);
                const double tq = bpb * (1.0 / (double)bl);
                qkwb[k] += tq * 2.0;
                const double e = (((double)a2 / (double)bl) * (-tq)) * (double)bl0;
                elb[k + 1] += e; elb[k] += e;
            }
            if (cbk != 0.0) {                        // c(k) = -(dtz*K(k+1)) * rhoinv
                const P t2 = r6_mul(dtz[k], K[k + 1]);
                acc(rhoinvb[k], (double)r6_neg(t2), cbk);
                const double t2b = -(cbk * (double)rhoinv[k]);
                acc(dtzb[k], (double)K[k + 1], t2b); acc(Kb[k + 1], (double)dtz[k], t2b);
            }
            if (dbk != 0.0) {                        // d(k) = rp*delt + q(k), rp = pdU(k+1) + pdU(k)
                const double rpb = dbk * (double)delt;
                pb[k + 1] += rpb;
                if (k >= 2) pb[k] += rpb; else { if (s) pdkU1b += rpb; else pdqU1b += rpb; }
                const int iq = I(fq, k);
                if (iq >= 0) bar[iq] += dbk;
            }
        }
    }
    { const int iq = I(9, n); if (iq >= 0 && qkeb_n != 0.0) bar[iq] += qkeb_n; }

    // ---- pdk(kts) = pdk1 - pdk(kts+1); pdt/pdq/pdc(kts) = their (kts+1)
    pdkb[2] -= pdkU1b; pdtb[2] += pdtU1b; pdqb[2] += pdqU1b; pdcb[2] += pdcU1b;
    if (pdkU1b != 0.0) {   // pdk1 = ((2*ust^3)*pmz) / vkz, vkz = (karman*p5)*dz(kts)
        const double tq = pdkU1b * (1.0 / (double)vkz);
        double vkzb = 0.0; acc(vkzb, (double)Bb / (double)vkz, -tq);
        acc(dzb[1], (double)r6_mul(karman, p5), vkzb);
        acc(pmzb, (double)Aa, tq);
        double Ab = 0.0; acc(Ab, (double)pmz, tq);
        const double P3b = Ab * 2.0;
        const double v = (double)ust;
        acc(ustb, 2.0 * v * v + (double)r6_mul(ust, ust), P3b);
    }

    // ---- kqdz / kmdz: own (rhoz*df3q, rhoz*dfq) or the mass-flux floor; k(kte+1) = rhoz(kte)*df(kte)
    for (int k = 1; k <= n + 1; ++k) {
        const int kk = k <= n ? k : n;
        const int code = (edmf && k >= 2 && k <= n - 1) ? int(src[k]) : 0;
        const double gq = kqb[k], gm = kmb[k];
        if (gq != 0.0) {
            const int cq = code & 3;
            if (cq == 0) { acc(rhozb[kk], (double)r6_mul(sqfac, F(3, kk)), gq); double t = 0.0; acc(t, (double)rhoz[kk], gq); acc(dfqb[kk], (double)sqfac, t); }
            else if (cq == 1) sawb[k] += gq * (double)p5;
            else { const double t = -(gq * (double)p5); sawb[k] += t; sawb[k + 1] -= t; }
        }
        if (gm != 0.0) {
            const int cm = (code >> 2) & 3;
            if (cm == 0) { acc(rhozb[kk], (double)F(3, kk), gm); acc(dfqb[kk], (double)rhoz[kk], gm); }
            else if (cm == 1) sawb[k] += gm * (double)p5;
            else { const double t = -(gm * (double)p5); sawb[k] += t; sawb[k + 1] -= t; }
        }
    }
    // ---- rhoz, rhoinv, dtz, qkw
    rhob[1] += rhozb[1];
    for (int k = 1; k <= n; ++k) {
        if (k >= 2 && rhozb[k] != 0.0 && rz0[k] > 1e-4f) {
            const P num = r6_add(r6_mul(F(1, k), F(0, k - 1)), r6_mul(F(1, k - 1), F(0, k)));
            const P den = r6_add(F(0, k - 1), F(0, k));
            const double tq = rhozb[k] * (1.0 / (double)den);
            const double denb = ((double)num / (double)den) * (-tq);
            acc(rhob[k], (double)F(0, k - 1), tq); acc(dzb[k - 1], (double)F(1, k), tq);
            acc(rhob[k - 1], (double)F(0, k), tq); acc(dzb[k], (double)F(1, k - 1), tq);
            dzb[k - 1] += denb; dzb[k] += denb;
        }
        const P r = F(1, k);
        if (rhoinvb[k] != 0.0 && (k == 1 || r > 1e-4f)) {
            const double tq = rhoinvb[k] * (1.0 / (double)r);
            rhob[k] += ((double)one / (double)r) * (-tq);
        }
        if (dtzb[k] != 0.0) {
            const P z = F(0, k);
            const double tq = dtzb[k] * (1.0 / (double)z);
            dzb[k] += ((double)delt / (double)z) * (-tq);
        }
        if (qkwb[k] != 0.0 && F(9, k) > qkemin) {
            const int iq = I(9, k);
            if (iq >= 0) bar[iq] += qkwb[k] * (1.0 / (2.0 * (double)qkw[k]));
        }
    }
    // ---- scatter to the input words
    for (int k = 1; k <= n; ++k) {
        int id;
        if ((id = I(0, k)) >= 0 && dzb[k] != 0.0) bar[id] += dzb[k];
        if ((id = I(1, k)) >= 0 && rhob[k] != 0.0) bar[id] += rhob[k];
        if ((id = I(2, k)) >= 0 && elb[k] != 0.0) bar[id] += elb[k];
        if ((id = I(3, k)) >= 0 && dfqb[k] != 0.0) bar[id] += dfqb[k];
        if ((id = I(4, k)) >= 0 && sawb[k] != 0.0) bar[id] += sawb[k];
        if (k >= 2) {
            if ((id = I(5, k)) >= 0 && pdkb[k] != 0.0) bar[id] += pdkb[k];
            if ((id = I(6, k)) >= 0 && pdqb[k] != 0.0) bar[id] += pdqb[k];
            if ((id = I(7, k)) >= 0 && pdtb[k] != 0.0) bar[id] += pdtb[k];
            if ((id = I(8, k)) >= 0 && pdcb[k] != 0.0) bar[id] += pdcb[k];
        }
    }
    if (iust >= 0 && ustb != 0.0) bar[iust] += ustb;
    if (ipmz >= 0 && pmzb != 0.0) bar[ipmz] += pmzb;
 return true;
}
}
