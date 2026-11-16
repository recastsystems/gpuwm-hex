// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for rlw_setcoef, the RRTMG LW SETCOEF kernel:
// hexcore/mpasphys/rrtmg_lw_v842.py:766 (launch), CUDA source sha256 683d7a07 (capture object, kernel at line 711;
// MPAS module_ra_rrtmg_lw.F:3567-3932 setcoef). Written by hand from the forward CUDA source. It replaces P7's
// generic taped row for binding (bind/p7-rows.json, p7-launch harness: per-thread tape in global scratch and a 32 KB
// per-thread stack, 9-12 GB of VRAM reserved per context on a 188-SM card). Same maths, no tape: every layer of
// setcoef is a local map of that layer's inputs (the only loop-carried word is the integer laytrop count), so the
// adjoint recomputes each layer's f32 primal words and needs no saved state.
//
// Precision (the glue-row rule, as the live schemes): the f32 primal words the derivative uses are formed exactly
// as the forward forms them (__f*_rn, gfk_log copied unchanged from the forward source; built --fmad=false
// --ftz=false); __check writes every forward output from the same helpers and is graded bitwise against the
// captured post words. Local partials are formed in double from those f32 words and the f32 inputs; every written
// derivative word is rounded to f32 once.
//
// Frozen coefficients (executed branch; every integer the forward derives from the state keeps its primal value and
// carries no derivative):
//   Planck table indices indbound / indlev0 / indlay / indlev = clamp((int)(T - 159), 1, 180)   (src 769-789)
//   jp = clamp((int)(36 - 5 (ln p + 0.04)), 1, 58); jt, jt1 = clamp((int)(3 + (T - tref) / 15), 1, 4)   (src 870-890)
//   indfor = clamp((int)((332 - T) / 36), 1, 2); indself = clamp((int)((T - 188) / 7.2) - 7, 1, 9);
//   indminor = clamp((int)((T - 180.8) / 7.2), 1, 18)                                        (src 899, 905, 915, 943)
//   the lower/upper switch !(ln p <= 4.56) and the laytrop count                              (src 894-895)
// The fractions beside them (tbndfrac, t0frac, tlayfrac, tlevfrac, fp, ft, ft1, forfrac, selffrac, minorfrac) keep
// their exact local slope also where the index is clamped (the forward extrapolates linearly there).
// Limiter-like clamps: the purity guard p <= 0 -> 1e-6 (src 868) is a constant where it fires, so d ln p = 0 there;
// the exact-zero gas replacement col = 1e-32 coldry when 1e-20 wkl rounds to 0 (src 960-964) differentiates the
// replacement (d col = 1e-32 d coldry) and drops that gas's wkl tangent there.
// Inactive: ncol, nlayers, istart; the tables totplnk, totplk16, preflog, tref, chi_mls (constants); the integer
// outputs laytrop_v, jp, jt, jt1, indself, indfor, indminor. The rat_* outputs are ratios of chi_mls entries: zero
// derivative on the words the executed branch writes, identity (untouched) on the rest (selffrac likewise is written
// only in the lower set).
// Primal arguments are the pre-launch words (the forward never writes its inputs). TL: input tangents in, output
// tangents written on exactly the words the forward writes. AD: output cotangents consumed (read, then zeroed) on the
// written words, input cotangents accumulated (one f32 rounding per input word; one thread owns a column's words, so
// no atomics, deterministic). Refusal: a nonfinite read input, partial or written derivative word sets failure slot 0
// (source rrtmg_lw_v842.py:766).

#define RSC_NB 16
#define RSC_MXMOL 38
#define RSC_E20 1.e-20f
#define RSC_E32 1.e-32f
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))

// glibc 2.39 sysdeps/ieee754/flt-32/e_logf.c, copied unchanged from the forward source (683d7a07:258, gfk_log) with
// its e_logf_data.c tables (Arm optimized-routines, MIT; notice kept in the forward source and the repository NOTICE).
__device__ const double RSC_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double RSC_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define RSC_LOGF_LN2 0x1.62e42fefa39efp-1
#define RSC_LOGF_A0 (-0x1.00ea348b88334p-2)
#define RSC_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define RSC_LOGF_A2 (-0x1.ffffef20a4123p-2)
__device__ float rsc_log(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return __int_as_float(0xff800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(FMUL(x, 8388608.0f));   /* 0x1p23f */
        ix -= 23u << 23;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, RSC_LOGF_INVC[i]), 1.0);
    double y0 = DADD(RSC_LOGF_LOGC[i], DMUL((double)k, RSC_LOGF_LN2));
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(RSC_LOGF_A1, r), RSC_LOGF_A2);
    y = DADD(DMUL(RSC_LOGF_A0, r2), y);
    y = DADD(DMUL(y, r2), DADD(y0, r));
    return __double2float_rn(y);
}


// One f32 rounding of a derivative word; adding +0.0 maps a -0.0 partial sum to +0.0, so the zero direction gives
// bitwise zeros (x + 0.0 == x for every other value; __dadd_rn is never folded away).
__device__ __forceinline__ float rsc_f(double v) { return (float)__dadd_rn(v, 0.0); }

#define TOTPLNK(i, b) totplnk[(i - 1) + 181 * (b - 1)]
#define CHI(s, j) chi_mls[(s - 1) + 7 * (j - 1)]

// Planck index and fraction (src 769-789): ind = clamp((int)(t - 159), 1, 180), frac = (t - 159) - ind.
__device__ __forceinline__ int rsc_pind(float t, float *frac) {
    int ind = (int)FSUB(t, 159.0f);
    if (ind < 1) ind = 1; else if (ind > 180) ind = 180;
    *frac = FSUB(FSUB(t, 159.0f), (float)ind);
    return ind;
}
// Band-ib Planck base value and slope at table index ind, as the forward picks them (src 791-857).
// which: 0 = plankbnd (tbound), 1 = planklev(0) (tz(0)), 2 = planklay / planklev(lay).
__device__ __forceinline__ void rsc_ptab(const float *totplnk, const float *totplk16, int istart, int ib, int ind,
                                         int which, float *base, float *slope) {
    if (ib < 16 || istart != 16) { *slope = FSUB(TOTPLNK(ind + 1, ib), TOTPLNK(ind, ib)); *base = TOTPLNK(ind, ib); return; }
    if (which == 0) { *slope = FSUB(totplk16[ind], totplk16[ind - 1]); *base = totplk16[ind - 1]; return; }
    if (which == 1) { *slope = FSUB(TOTPLNK(ind + 1, ib), TOTPLNK(ind, ib)); *base = totplk16[ind - 1]; return; }  // src 823-826
    *slope = FSUB(totplk16[ind], totplk16[ind - 1]); *base = totplk16[ind - 1];
}

// One layer's f32 primal words, in the forward's order (src 867-974).
struct RscLay {
    float p, t, cd, w1, wb, plog, fp, ft, ft1, water, a, scalefac, onepw, ff0, sf0, r1, den, q;
    float col[7], colbrd, compfp, omft, omft1;
    bool guard, lower, zero[7];
    int jp, jt, jt1, indfor, indself, indminor;
    float forfrac, selffrac, minorfrac, ffac, sfac, fac00, fac01, fac10, fac11, scaleminorn2;
};
__device__ __forceinline__ void rsc_lay(RscLay &k, const float *pav, const float *tav, const float *cdry,
                                        const float *wklc, const float *wbr, const float *preflog, const float *tref,
                                        int lay, int nl) {
    const float stpfac = FDIV(296.0f, 1013.0f);
    k.p = pav[lay - 1]; k.t = tav[lay - 1]; k.cd = cdry[lay - 1]; k.w1 = wklc[0 * nl + (lay - 1)]; k.wb = wbr[lay - 1];
    float pavl = k.p; k.guard = !(pavl > 0.0f); if (k.guard) pavl = 1.0e-6f;
    k.plog = rsc_log(pavl);
    int jpl = (int)FSUB(36.0f, FMUL(5.0f, FADD(k.plog, 0.04f)));
    if (jpl < 1) jpl = 1; else if (jpl > 58) jpl = 58;
    k.jp = jpl;
    k.fp = FMUL(5.0f, FSUB(preflog[jpl - 1], k.plog));
    int jtl = (int)FADD(3.0f, FDIV(FSUB(k.t, tref[jpl - 1]), 15.0f));
    if (jtl < 1) jtl = 1; else if (jtl > 4) jtl = 4;
    k.jt = jtl;
    k.ft = FSUB(FDIV(FSUB(k.t, tref[jpl - 1]), 15.0f), (float)(jtl - 3));
    int jt1l = (int)FADD(3.0f, FDIV(FSUB(k.t, tref[jpl]), 15.0f));
    if (jt1l < 1) jt1l = 1; else if (jt1l > 4) jt1l = 4;
    k.jt1 = jt1l;
    k.ft1 = FSUB(FDIV(FSUB(k.t, tref[jpl]), 15.0f), (float)(jt1l - 3));
    k.water = FDIV(k.w1, k.cd);
    k.a = FMUL(k.p, stpfac);
    k.scalefac = FDIV(k.a, k.t);
    k.lower = !(k.plog <= 4.56f);
    k.onepw = FADD(1.0f, k.water);
    k.ff0 = FDIV(k.scalefac, k.onepw);
    if (k.lower) {
        float factor = FDIV(FSUB(332.0f, k.t), 36.0f);
        k.indfor = min(2, max(1, (int)factor));
        k.forfrac = FSUB(factor, (float)k.indfor);
        factor = FDIV(FSUB(k.t, 188.0f), 7.2f);
        k.indself = min(9, max(1, (int)factor - 7));
        k.selffrac = FSUB(factor, (float)(k.indself + 7));
    } else {
        float factor = FDIV(FSUB(k.t, 188.0f), 36.0f);
        k.indfor = 3;
        k.forfrac = FSUB(factor, 1.0f);
        k.indself = 0; k.selffrac = 0.0f;
    }
    k.sf0 = FMUL(k.water, k.ff0);
    k.r1 = FDIV(k.p, k.t);
    k.den = FADD(k.cd, k.w1);
    k.q = FDIV(k.wb, k.den);
    k.scaleminorn2 = FMUL(k.r1, k.q);
    float factor = FDIV(FSUB(k.t, 180.8f), 7.2f);
    k.indminor = min(18, max(1, (int)factor));
    k.minorfrac = FSUB(factor, (float)k.indminor);
    for (int s = 0; s < 7; ++s) {
        k.col[s] = FMUL(RSC_E20, wklc[s * nl + (lay - 1)]);
        k.zero[s] = (s >= 1 && s <= 5) && k.col[s] == 0.0f;       // co2, o3, n2o, co, ch4 (src 960-964)
        if (k.zero[s]) k.col[s] = FMUL(RSC_E32, k.cd);
    }
    k.colbrd = FMUL(RSC_E20, k.wb);
    k.compfp = FSUB(1.0f, k.fp);
    k.omft = FSUB(1.0f, k.ft); k.omft1 = FSUB(1.0f, k.ft1);
    k.fac10 = FMUL(k.compfp, k.ft); k.fac00 = FMUL(k.compfp, k.omft);
    k.fac11 = FMUL(k.fp, k.ft1);    k.fac01 = FMUL(k.fp, k.omft1);
    k.sfac = FMUL(k.col[0], k.sf0);
    k.ffac = FMUL(k.col[0], k.ff0);
}

#define RSC_PRIMAL_PARAMS int ncol, int nlayers, int istart, \
    const float *__restrict__ pavel, const float *__restrict__ tavel, const float *__restrict__ tz, \
    const float *__restrict__ tbound_v, const float *__restrict__ semiss, const float *__restrict__ coldry, \
    const float *__restrict__ wkl, const float *__restrict__ wbroad, const float *__restrict__ totplnk, \
    const float *__restrict__ totplk16, const float *__restrict__ preflog, const float *__restrict__ tref, \
    const float *__restrict__ chi_mls
#define RSC_COLUMN \
    int col = blockIdx.x * blockDim.x + threadIdx.x; \
    if (col >= ncol) return; \
    const int nl = nlayers; \
    const float *pav = pavel + (long long)col * nl, *tav = tavel + (long long)col * nl; \
    const float *tzc = tz + (long long)col * (nl + 1), *sem = semiss + (long long)col * RSC_NB; \
    const float *cdry = coldry + (long long)col * nl, *wklc = wkl + (long long)col * RSC_MXMOL * nl; \
    const float *wbr = wbroad + (long long)col * nl; \
    const float tbound = tbound_v[col]; (void)chi_mls;
#define LW(a) a[(long long)col * nl + (lay - 1)]
#define PLAY(a, lay, b) a[((long long)col * nl + (lay - 1)) * RSC_NB + (b - 1)]
#define PLEV(a, lev, b) a[((long long)col * (nl + 1) + (lev)) * RSC_NB + (b - 1)]

// Float output tangent / cotangent list, forward output order (the integer outputs are left out).
#define RSC_OUT_PARAMS(P) float *P##planklay, float *P##planklev, float *P##plankbnd, \
    float *P##colh2o, float *P##colco2, float *P##colo3, float *P##coln2o, float *P##colco, float *P##colch4, \
    float *P##colo2, float *P##colbrd, float *P##fac00, float *P##fac01, float *P##fac10, float *P##fac11, \
    float *P##rat_h2oco2, float *P##rat_h2oco2_1, float *P##rat_h2oo3, float *P##rat_h2oo3_1, \
    float *P##rat_h2on2o, float *P##rat_h2on2o_1, float *P##rat_h2och4, float *P##rat_h2och4_1, \
    float *P##rat_n2oco2, float *P##rat_n2oco2_1, float *P##rat_o3co2, float *P##rat_o3co2_1, \
    float *P##selffac, float *P##selffrac, float *P##forfac, float *P##forfrac, float *P##minorfrac, \
    float *P##scaleminor, float *P##scaleminorn2

// Harness only (not in launch_abi): every forward output word from the shared helpers, in the forward's own
// argument order, so the harness launches it exactly like the captured image.
// step6-radiation (2026-11-07): molecule-unit storage scale, shared with rlw_inatm (see the rationale there). The
// inputs coldry (the fs[coldry] slot), wkl rows 1..7 and wbroad are in molecules/cm2: their tangents arrive stored as
// t * 2^-72 and their cotangents are stored as b * 2^72. Exact (a power of two, in double, before the one f32 rounding).
#define RSC_MOL_T 0x1p72
#define RSC_MOL_B 0x1p72
extern "C" __global__ void rlw_setcoef__check(RSC_PRIMAL_PARAMS,
    int *laytrop_v, int *jp, int *jt, int *jt1,
    float *planklay, float *planklev, float *plankbnd,
    float *colh2o, float *colco2, float *colo3, float *coln2o, float *colco, float *colch4, float *colo2, float *colbrd,
    float *fac00, float *fac01, float *fac10, float *fac11,
    float *rat_h2oco2, float *rat_h2oco2_1, float *rat_h2oo3, float *rat_h2oo3_1, float *rat_h2on2o, float *rat_h2on2o_1,
    float *rat_h2och4, float *rat_h2och4_1, float *rat_n2oco2, float *rat_n2oco2_1, float *rat_o3co2, float *rat_o3co2_1,
    float *selffac, float *selffrac, int *indself, float *forfac, float *forfrac, int *indfor,
    float *minorfrac, float *scaleminor, float *scaleminorn2, int *indminor)
{
    RSC_COLUMN
    float tbf, t0f; const int ib0 = rsc_pind(tbound, &tbf), il0 = rsc_pind(tzc[0], &t0f);
    int laytrop = 0;
    for (int lay = 1; lay <= nl; ++lay) {
        float tlf, tvf; const int ily = rsc_pind(tav[lay - 1], &tlf), ilv = rsc_pind(tzc[lay], &tvf);
        for (int ib = 1; ib <= RSC_NB; ++ib) {
            float b, s;
            if (lay == 1) {
                rsc_ptab(totplnk, totplk16, istart, ib, ib0, 0, &b, &s);
                plankbnd[(long long)col * RSC_NB + (ib - 1)] = FMUL(sem[ib - 1], FADD(b, FMUL(tbf, s)));
                rsc_ptab(totplnk, totplk16, istart, ib, il0, 1, &b, &s);
                PLEV(planklev, 0, ib) = FADD(b, FMUL(t0f, s));
            }
            rsc_ptab(totplnk, totplk16, istart, ib, ily, 2, &b, &s); PLAY(planklay, lay, ib) = FADD(b, FMUL(tlf, s));
            rsc_ptab(totplnk, totplk16, istart, ib, ilv, 2, &b, &s); PLEV(planklev, lay, ib) = FADD(b, FMUL(tvf, s));
        }
        RscLay k; rsc_lay(k, pav, tav, cdry, wklc, wbr, preflog, tref, lay, nl);
        LW(jp) = k.jp; LW(jt) = k.jt; LW(jt1) = k.jt1;
        const int j = k.jp;
        if (k.lower) {
            laytrop = laytrop + 1;
            LW(indself) = k.indself; LW(selffrac) = k.selffrac;
            LW(rat_h2oo3) = FDIV(CHI(1, j), CHI(3, j)); LW(rat_h2oo3_1) = FDIV(CHI(1, j + 1), CHI(3, j + 1));
            LW(rat_h2on2o) = FDIV(CHI(1, j), CHI(4, j)); LW(rat_h2on2o_1) = FDIV(CHI(1, j + 1), CHI(4, j + 1));
            LW(rat_h2och4) = FDIV(CHI(1, j), CHI(6, j)); LW(rat_h2och4_1) = FDIV(CHI(1, j + 1), CHI(6, j + 1));
            LW(rat_n2oco2) = FDIV(CHI(4, j), CHI(2, j)); LW(rat_n2oco2_1) = FDIV(CHI(4, j + 1), CHI(2, j + 1));
        } else {
            LW(rat_o3co2) = FDIV(CHI(3, j), CHI(2, j)); LW(rat_o3co2_1) = FDIV(CHI(3, j + 1), CHI(2, j + 1));
        }
        LW(indfor) = k.indfor; LW(forfrac) = k.forfrac;
        LW(rat_h2oco2) = FDIV(CHI(1, j), CHI(2, j)); LW(rat_h2oco2_1) = FDIV(CHI(1, j + 1), CHI(2, j + 1));
        LW(scaleminor) = k.r1; LW(scaleminorn2) = k.scaleminorn2; LW(indminor) = k.indminor; LW(minorfrac) = k.minorfrac;
        LW(colh2o) = k.col[0]; LW(colco2) = k.col[1]; LW(colo3) = k.col[2]; LW(coln2o) = k.col[3];
        LW(colco) = k.col[4]; LW(colch4) = k.col[5]; LW(colo2) = k.col[6]; LW(colbrd) = k.colbrd;
        LW(fac10) = k.fac10; LW(fac00) = k.fac00; LW(fac11) = k.fac11; LW(fac01) = k.fac01;
        LW(selffac) = k.sfac; LW(forfac) = k.ffac;
    }
    laytrop_v[col] = laytrop;
}

// Local partials of one layer's float outputs with respect to the layer inputs (p, T, cd, w[0..6], wb), all double,
// from the f32 primal words of rsc_lay. Planck outputs are handled with their slopes separately.
struct RscJac {
    double dplog_dp;                                    // d ln p / d p (0 where the purity guard fired)
    double water_w1, water_cd;                          // water = w1 / cd
    double sf_p, sf_t;                                  // scalefac = (p stpfac) / T
    double ff0_p, ff0_t, ff0_w1, ff0_cd;                // forfac before the colh2o scaling
    double sf0_p, sf0_t, sf0_w1, sf0_cd;                // selffac before the colh2o scaling
    double r1_p, r1_t;                                  // scaleminor = p / T
    double q_wb, q_cd, q_w1;                            // q = wb / (cd + w1)
};
__device__ __forceinline__ void rsc_jac(const RscLay &k, RscJac &J) {
    const double stp = (double)FDIV(296.0f, 1013.0f);
    const double p = k.p, t = k.t, cd = k.cd, w1 = k.w1;
    J.dplog_dp = k.guard ? 0.0 : 1.0 / p;
    J.water_w1 = 1.0 / cd; J.water_cd = -w1 / (cd * cd);
    J.sf_p = stp / t; J.sf_t = -(double)k.a / (t * t);
    const double opw = k.onepw, sfv = k.scalefac;
    const double ff0_sf = 1.0 / opw, ff0_wat = -sfv / (opw * opw);
    J.ff0_p = ff0_sf * J.sf_p; J.ff0_t = ff0_sf * J.sf_t;
    J.ff0_w1 = ff0_wat * J.water_w1; J.ff0_cd = ff0_wat * J.water_cd;
    const double wat = k.water, ff0 = k.ff0;
    J.sf0_p = wat * J.ff0_p; J.sf0_t = wat * J.ff0_t;
    J.sf0_w1 = ff0 * J.water_w1 + wat * J.ff0_w1; J.sf0_cd = ff0 * J.water_cd + wat * J.ff0_cd;
    J.r1_p = 1.0 / t; J.r1_t = -p / (t * t);
    const double den = k.den, wb = k.wb;
    J.q_wb = 1.0 / den; J.q_cd = -wb / (den * den); J.q_w1 = J.q_cd;
}

extern "C" __global__ void rlw_setcoef__tl(RSC_PRIMAL_PARAMS,
    const float *d_pavel, const float *d_tavel, const float *d_tz, const float *d_tbound_v, const float *d_semiss,
    const float *d_coldry, const float *d_wkl, const float *d_wbroad, RSC_OUT_PARAMS(d_), int *failure)
{
    RSC_COLUMN
    const float *dpav = d_pavel + (long long)col * nl, *dtav = d_tavel + (long long)col * nl;
    const float *dtz = d_tz + (long long)col * (nl + 1), *dsem = d_semiss + (long long)col * RSC_NB;
    const float *dcdry = d_coldry + (long long)col * nl, *dwkl = d_wkl + (long long)col * RSC_MXMOL * nl;
    const float *dwbr = d_wbroad + (long long)col * nl;
    bool bad = !isfinite(tbound) || !isfinite(d_tbound_v[col]);
    float tbf, t0f; const int ib0 = rsc_pind(tbound, &tbf), il0 = rsc_pind(tzc[0], &t0f);
    const double dtb = (double)d_tbound_v[col], dtz0 = (double)dtz[0];
    bad |= !isfinite(tzc[0]) || !isfinite(dtz[0]);
    float o;
    for (int lay = 1; lay <= nl; ++lay) {
        float tlf, tvf; const int ily = rsc_pind(tav[lay - 1], &tlf), ilv = rsc_pind(tzc[lay], &tvf);
        const double dT = (double)dtav[lay - 1], dtzl = (double)dtz[lay];
        for (int ib = 1; ib <= RSC_NB; ++ib) {
            float b, s;
            if (lay == 1) {
                rsc_ptab(totplnk, totplk16, istart, ib, ib0, 0, &b, &s);
                const double inner = (double)FADD(b, FMUL(tbf, s)), sv = (double)sem[ib - 1];
                o = rsc_f((double)dsem[ib - 1] * inner + sv * (double)s * dtb);
                d_plankbnd[(long long)col * RSC_NB + (ib - 1)] = o; bad |= !isfinite(o) || !isfinite(sem[ib - 1]) || !isfinite(dsem[ib - 1]);
                rsc_ptab(totplnk, totplk16, istart, ib, il0, 1, &b, &s);
                o = rsc_f((double)s * dtz0); PLEV(d_planklev, 0, ib) = o; bad |= !isfinite(o);
            }
            rsc_ptab(totplnk, totplk16, istart, ib, ily, 2, &b, &s);
            o = rsc_f((double)s * dT); PLAY(d_planklay, lay, ib) = o; bad |= !isfinite(o);
            rsc_ptab(totplnk, totplk16, istart, ib, ilv, 2, &b, &s);
            o = rsc_f((double)s * dtzl); PLEV(d_planklev, lay, ib) = o; bad |= !isfinite(o);
        }
        RscLay k; rsc_lay(k, pav, tav, cdry, wklc, wbr, preflog, tref, lay, nl);
        RscJac J; rsc_jac(k, J);
        const double dp = (double)dpav[lay - 1], dcd = RSC_MOL_T * (double)dcdry[lay - 1], dwb = RSC_MOL_T * (double)dwbr[lay - 1];
        double dw[7];
        for (int s = 0; s < 7; ++s) dw[s] = RSC_MOL_T * (double)dwkl[s * nl + (lay - 1)];
        bad |= !isfinite(k.p) || !isfinite(k.t) || !isfinite(k.cd) || !isfinite(k.wb) || !isfinite(dp) || !isfinite(dT)
            || !isfinite(dcd) || !isfinite(dwb) || !isfinite(tzc[lay]) || !isfinite(dtzl);
        for (int s = 0; s < 7; ++s) bad |= !isfinite(wklc[s * nl + (lay - 1)]) || !isfinite(dw[s]);
        const double dplog = J.dplog_dp * dp;
        const double dfp = -5.0 * dplog, dft = dT / 15.0;
        const double dff0 = J.ff0_p * dp + J.ff0_t * dT + J.ff0_w1 * dw[0] + J.ff0_cd * dcd;
        const double dsf0 = J.sf0_p * dp + J.sf0_t * dT + J.sf0_w1 * dw[0] + J.sf0_cd * dcd;
        const double dr1 = J.r1_p * dp + J.r1_t * dT;
        const double dq = J.q_wb * dwb + J.q_cd * dcd + J.q_w1 * dw[0];
        if (k.lower) {
            o = rsc_f(-dT / 36.0); LW(d_forfrac) = o;
            o = rsc_f(dT / (double)7.2f); LW(d_selffrac) = o;          // (T - 188) / 7.2 - (indself + 7), lower set only
            LW(d_rat_h2oo3) = 0.0f; LW(d_rat_h2oo3_1) = 0.0f; LW(d_rat_h2on2o) = 0.0f; LW(d_rat_h2on2o_1) = 0.0f;
            LW(d_rat_h2och4) = 0.0f; LW(d_rat_h2och4_1) = 0.0f; LW(d_rat_n2oco2) = 0.0f; LW(d_rat_n2oco2_1) = 0.0f;
        } else {
            o = rsc_f(dT / 36.0); LW(d_forfrac) = o;
            LW(d_rat_o3co2) = 0.0f; LW(d_rat_o3co2_1) = 0.0f;
        }
        LW(d_rat_h2oco2) = 0.0f; LW(d_rat_h2oco2_1) = 0.0f;
        o = rsc_f(dr1); LW(d_scaleminor) = o; bad |= !isfinite(o);
        o = rsc_f(dr1 * (double)k.q + (double)k.r1 * dq); LW(d_scaleminorn2) = o; bad |= !isfinite(o);
        o = rsc_f(dT / (double)7.2f); LW(d_minorfrac) = o;
        double dcol[7];
        for (int s = 0; s < 7; ++s) dcol[s] = k.zero[s] ? (double)RSC_E32 * dcd : (double)RSC_E20 * dw[s];
        LW(d_colh2o) = rsc_f(dcol[0]); LW(d_colco2) = rsc_f(dcol[1]); LW(d_colo3) = rsc_f(dcol[2]);
        LW(d_coln2o) = rsc_f(dcol[3]); LW(d_colco) = rsc_f(dcol[4]); LW(d_colch4) = rsc_f(dcol[5]);
        LW(d_colo2) = rsc_f(dcol[6]);
        o = rsc_f((double)RSC_E20 * dwb); LW(d_colbrd) = o;
        const double fp = k.fp, ft = k.ft, ft1 = k.ft1, cfp = k.compfp, omft = k.omft, omft1 = k.omft1;
        o = rsc_f(-dfp * ft + cfp * dft); LW(d_fac10) = o; bad |= !isfinite(o);
        o = rsc_f(-dfp * omft - cfp * dft); LW(d_fac00) = o; bad |= !isfinite(o);
        o = rsc_f(dfp * ft1 + fp * dft); LW(d_fac11) = o; bad |= !isfinite(o);
        o = rsc_f(dfp * omft1 - fp * dft); LW(d_fac01) = o; bad |= !isfinite(o);
        const double c0 = k.col[0];
        o = rsc_f(dcol[0] * (double)k.sf0 + c0 * dsf0); LW(d_selffac) = o; bad |= !isfinite(o);
        o = rsc_f(dcol[0] * (double)k.ff0 + c0 * dff0); LW(d_forfac) = o; bad |= !isfinite(o);
        for (int s = 0; s < 7; ++s) bad |= !isfinite(rsc_f(dcol[s]));
    }
    if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rlw_setcoef__ad(RSC_PRIMAL_PARAMS,
    float *b_pavel, float *b_tavel, float *b_tz, float *b_tbound_v, float *b_semiss, float *b_coldry, float *b_wkl,
    float *b_wbroad, RSC_OUT_PARAMS(b_), int *failure)
{
    RSC_COLUMN
    float *bpav = b_pavel + (long long)col * nl, *btav = b_tavel + (long long)col * nl;
    float *btz = b_tz + (long long)col * (nl + 1), *bsem = b_semiss + (long long)col * RSC_NB;
    float *bcdry = b_coldry + (long long)col * nl, *bwkl = b_wkl + (long long)col * RSC_MXMOL * nl;
    float *bwbr = b_wbroad + (long long)col * nl;
    bool bad = !isfinite(tbound) || !isfinite(tzc[0]);
    float tbf, t0f; const int ib0 = rsc_pind(tbound, &tbf), il0 = rsc_pind(tzc[0], &t0f);
    double gtb = 0.0, gtz0 = 0.0;
    for (int lay = 1; lay <= nl; ++lay) {
        float tlf, tvf; const int ily = rsc_pind(tav[lay - 1], &tlf), ilv = rsc_pind(tzc[lay], &tvf);
        double gT = 0.0, gtzl = 0.0;
        for (int ib = 1; ib <= RSC_NB; ++ib) {
            float b, s;
            if (lay == 1) {
                rsc_ptab(totplnk, totplk16, istart, ib, ib0, 0, &b, &s);
                long long jb = (long long)col * RSC_NB + (ib - 1);
                const double yb = (double)b_plankbnd[jb]; b_plankbnd[jb] = 0.0f;
                const double inner = (double)FADD(b, FMUL(tbf, s)), sv = (double)sem[ib - 1];
                float t = rsc_f((double)bsem[ib - 1] + yb * inner); bsem[ib - 1] = t;
                bad |= !isfinite(t) || !isfinite(yb) || !isfinite(sem[ib - 1]);
                gtb += yb * sv * (double)s;
                rsc_ptab(totplnk, totplk16, istart, ib, il0, 1, &b, &s);
                const double y0 = (double)PLEV(b_planklev, 0, ib); PLEV(b_planklev, 0, ib) = 0.0f;
                gtz0 += y0 * (double)s; bad |= !isfinite(y0);
            }
            rsc_ptab(totplnk, totplk16, istart, ib, ily, 2, &b, &s);
            const double yl = (double)PLAY(b_planklay, lay, ib); PLAY(b_planklay, lay, ib) = 0.0f;
            gT += yl * (double)s;
            rsc_ptab(totplnk, totplk16, istart, ib, ilv, 2, &b, &s);
            const double yv = (double)PLEV(b_planklev, lay, ib); PLEV(b_planklev, lay, ib) = 0.0f;
            gtzl += yv * (double)s;
            bad |= !isfinite(yl) || !isfinite(yv);
        }
        RscLay k; rsc_lay(k, pav, tav, cdry, wklc, wbr, preflog, tref, lay, nl);
        RscJac J; rsc_jac(k, J);
        bad |= !isfinite(k.p) || !isfinite(k.t) || !isfinite(k.cd) || !isfinite(k.wb) || !isfinite(tzc[lay]);
        for (int s = 0; s < 7; ++s) bad |= !isfinite(wklc[s * nl + (lay - 1)]);
        // read and consume the written output cotangents of this layer
        const double yff = LW(b_forfrac), ysm = LW(b_scaleminor), ysm2 = LW(b_scaleminorn2), ymf = LW(b_minorfrac);
        LW(b_forfrac) = 0.0f; LW(b_scaleminor) = 0.0f; LW(b_scaleminorn2) = 0.0f; LW(b_minorfrac) = 0.0f;
        double ysf = 0.0;
        if (k.lower) {
            ysf = LW(b_selffrac); LW(b_selffrac) = 0.0f;
            LW(b_rat_h2oo3) = 0.0f; LW(b_rat_h2oo3_1) = 0.0f; LW(b_rat_h2on2o) = 0.0f; LW(b_rat_h2on2o_1) = 0.0f;
            LW(b_rat_h2och4) = 0.0f; LW(b_rat_h2och4_1) = 0.0f; LW(b_rat_n2oco2) = 0.0f; LW(b_rat_n2oco2_1) = 0.0f;
        } else {
            LW(b_rat_o3co2) = 0.0f; LW(b_rat_o3co2_1) = 0.0f;
        }
        LW(b_rat_h2oco2) = 0.0f; LW(b_rat_h2oco2_1) = 0.0f;
        double ycol[7];
        ycol[0] = LW(b_colh2o); ycol[1] = LW(b_colco2); ycol[2] = LW(b_colo3); ycol[3] = LW(b_coln2o);
        ycol[4] = LW(b_colco); ycol[5] = LW(b_colch4); ycol[6] = LW(b_colo2);
        LW(b_colh2o) = 0.0f; LW(b_colco2) = 0.0f; LW(b_colo3) = 0.0f; LW(b_coln2o) = 0.0f;
        LW(b_colco) = 0.0f; LW(b_colch4) = 0.0f; LW(b_colo2) = 0.0f;
        const double ycb = LW(b_colbrd); LW(b_colbrd) = 0.0f;
        const double y10 = LW(b_fac10), y00 = LW(b_fac00), y11 = LW(b_fac11), y01 = LW(b_fac01);
        LW(b_fac10) = 0.0f; LW(b_fac00) = 0.0f; LW(b_fac11) = 0.0f; LW(b_fac01) = 0.0f;
        const double yself = LW(b_selffac), yfor = LW(b_forfac);
        LW(b_selffac) = 0.0f; LW(b_forfac) = 0.0f;
        bad |= !isfinite(yff) || !isfinite(ysm) || !isfinite(ysm2) || !isfinite(ymf) || !isfinite(ysf) || !isfinite(ycb)
            || !isfinite(y10) || !isfinite(y00) || !isfinite(y11) || !isfinite(y01) || !isfinite(yself) || !isfinite(yfor);
        // transpose, intermediate by intermediate (the reverse of the TL above)
        const double fp = k.fp, ft = k.ft, ft1 = k.ft1, cfp = k.compfp, omft = k.omft, omft1 = k.omft1;
        double gfp = -ft * y10 - omft * y00 + ft1 * y11 + omft1 * y01;
        double gft = cfp * y10 - cfp * y00 + fp * y11 - fp * y01;    // dft1 = dft (both dT / 15)
        const double c0 = k.col[0];
        ycol[0] += (double)k.sf0 * yself + (double)k.ff0 * yfor;
        const double gsf0 = c0 * yself, gff0 = c0 * yfor;
        const double gr1 = ysm + (double)k.q * ysm2, gq = (double)k.r1 * ysm2;
        double gp = 0.0, gcd = 0.0, gwb = 0.0, gw[7] = {0, 0, 0, 0, 0, 0, 0};
        gT += gft / 15.0 + (k.lower ? -yff / 36.0 : yff / 36.0) + (k.lower ? ysf / (double)7.2f : 0.0) + ymf / (double)7.2f;
        const double gplog = -5.0 * gfp;
        gp += J.dplog_dp * gplog;
        gp += J.ff0_p * gff0 + J.sf0_p * gsf0 + J.r1_p * gr1;
        gT += J.ff0_t * gff0 + J.sf0_t * gsf0 + J.r1_t * gr1;
        gw[0] += J.ff0_w1 * gff0 + J.sf0_w1 * gsf0 + J.q_w1 * gq;
        gcd += J.ff0_cd * gff0 + J.sf0_cd * gsf0 + J.q_cd * gq;
        gwb += J.q_wb * gq + (double)RSC_E20 * ycb;
        for (int s = 0; s < 7; ++s) { if (k.zero[s]) gcd += (double)RSC_E32 * ycol[s]; else gw[s] += (double)RSC_E20 * ycol[s]; }
        float t;
        t = rsc_f((double)bpav[lay - 1] + gp); bpav[lay - 1] = t; bad |= !isfinite(t);
        t = rsc_f((double)btav[lay - 1] + gT); btav[lay - 1] = t; bad |= !isfinite(t);
        t = rsc_f((double)bcdry[lay - 1] + RSC_MOL_B * gcd); bcdry[lay - 1] = t; bad |= !isfinite(t);
        t = rsc_f((double)bwbr[lay - 1] + RSC_MOL_B * gwb); bwbr[lay - 1] = t; bad |= !isfinite(t);
        for (int s = 0; s < 7; ++s) { long long j = (long long)s * nl + (lay - 1); t = rsc_f((double)bwkl[j] + RSC_MOL_B * gw[s]); bwkl[j] = t; bad |= !isfinite(t); }
        t = rsc_f((double)btz[lay] + gtzl); btz[lay] = t; bad |= !isfinite(t);
    }
    float t = rsc_f((double)btz[0] + gtz0); btz[0] = t; bad |= !isfinite(t);
    t = rsc_f((double)b_tbound_v[col] + gtb); b_tbound_v[col] = t; bad |= !isfinite(t);
    if (bad) atomicExch(failure, 1);
}
