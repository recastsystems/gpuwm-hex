// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for rsw_spcvmc_gpt_b, the batched RRTMG SW
// two-stream solver (one thread per (column, g-point)): hexcore/mpasphys/rrtmg_sw_v842.py:2665 (pin 44588ef0), CUDA
// source sha256 3c499c7b (capture object; rsw_etbl :1409, rsw_reftra_layer :2569, rsw_vrtqdr :2691, rsw_spcvmc_body
// :2767, rsw_spcvmc_gpt_b :2984; MPAS module_ra_rrtmg_sw.F:2541-2821, 8046-8170, 8207-8785).
// Written by hand from the forward CUDA source. Status before: missing in the kernel set (an 11-02 multi-entry draft
// existed, but its launch ABI was marked incomplete and it never loaded in a window). Nothing in the forward changes;
// the forward's helper words are copied verbatim below so the entry is self-contained.
//
// Forward, per (col, g) with band = ngb[g]-16 and K = klev-1-J (inputs are stored bottom-up, the solver runs top-down):
//   layer J: Rayleigh + gas + aerosol optics, delta scaling, cloudy optics; two reftra calls (clear, cloudy); the McICA
//            mix ref = (1-cf)*clear + cf*cloudy (and refd, tra, trad); direct-beam factors dbtc, dbt; the unscaled
//            ("nodel") factors.  Cumulative products tdbt[J+1] = dbt[J]*tdbt[J] (both skies, both kinds).
//   vrtqdr (clear sky, then total): upward recurrence rup/rupd from the surface albedos, downward recurrence tdn/rdnd,
//            fluxes fu/fd at every level 0..klev.  zincflx = adjflux*sfluxzen*mu.
// Outputs: zincflx (ncol,112), zcd/zcu (clear), zfd/zfu (total), ztdbtc_nodel_out / ztdbt_nodel_out, all
// (ncol,klev+1,112), every word overwritten.  wk is the forward's private workspace (its own allocation, every word
// read in a launch is written earlier in the same launch): scratch, no derivative, not bound.  wkc is a null slot.
//
// Restructured passes (same statements, same operand order, same rounding, so every primal word the derivative uses
// equals the forward's; __check reproduces every output word and is compared bitwise with the capture):
//   TL  pass A (bottom-up): layer optics with tangents, upward recurrence; rup/rupd primal (f32) and tangent (f64) kept.
//       pass B (top-down):  layer optics with tangents again, tdbt and nodel products, downward recurrence, fluxes.
//   AD  P1 (top-down) primal tdbt, tdn, rdnd, nodel;  P2 (bottom-up) primal rup, rupd;
//       R1 (descending levels) reverse of fluxes, downward recurrence, tdbt and nodel products -> per-layer seeds;
//       R2 (ascending layers) reverse of the upward recurrence, then the layer Jacobian transpose -> input cotangents.
//   Every local Jacobian (layer: 11 inputs x 12 outputs; up/down step: 7 x 2; flux: 5 x 2) is evaluated with the SAME
//   dual-number code the TL runs, one unit direction per input, so TL and AD use identical coefficients.
//
// Precision (the glue-row rule): primal words in f32 with the forward's own add/sub/mul.rn, __fdiv_rn, __fsqrt_rn and
// NaN wrappers (built --fmad=false --ftz=false); derivative arithmetic in double from those f32 words; tangent and
// adjoint workspaces in double; one rounding to f32 per written derivative word.  +0.0 is added before each tangent
// store so TL(0) gives +0 bits.
// Frozen coefficients (exact executed-branch rule): the cloud test cf > 1e-12; reftra's g != 1, w > 0 && denom != 0 and
// wo >= 0.9999995 branch choice; the min(x, 500) clamps (clamped -> zero derivative); the ze2 == 1 reset (constants);
// the |zdenr| <= 1e-8 limiter (pref = eps constant, ptra = zem2 differentiated).  The polynomial branch x <= od_lo
// (1 - x + x^2/2) is differentiated exactly.
// Exponential table (rsw_etbl and reftra's inline lookups: nearest entry, no interpolation): the forward is a staircase
// of 1e-4 steps in x/(bpade+x) whose executed-branch derivative is zero; the entry differentiates the function the
// table tabulates instead: the table is max(exp(-tau), expeps = 1e-20) (checked on the captured words to 2.2e-5), so
// d = -exp(-x) dx in double from the f32 argument (P7's rule for exp, as the LW entries use) above the floor and 0 on
// the floor entries (a frozen limiter).  Table entries (tab) are inactive constants.
// Two-stream resonance (zrk*mu = 1, a removable 0/0 in pref/ptra): inside |1 - (zrk*mu)^2| < 1e-3 the derivative is the
// mean of the derivatives at mu*(1 +- 1e-3) on the float64 shadow (d_reftra); about 0.04% of the words of a launch.
// Shared inputs: band-indexed (palbd, palbp, adjflux, ptaua, pasya, pomga) and column-indexed (prmu0) cotangents are
// summed across the g-points of one column inside one block in a fixed order (shared memory, increasing g), so the
// reverse is deterministic with no atomics.  Geometry: one 128-thread block per column (thread = g, 112 active),
// grid-stride over columns; per-thread workspace interleaved across the launch's threads (RES = gridDim*blockDim).
// Refusal (failure slots): 0 nonfinite primal word (input or recomputed), 1 nonfinite derivative word, 2 table index
// outside 0..10000, 3 nlayers < 1 or ngb outside 16..29.

#define RSW_T_EXP_TBL 94302
#define RSW_C_OD_LO __uint_as_float(1031127695u)
#define RSW_C_BPADE __uint_as_float(1080440641u)
#define RSW_C_TBLINT __uint_as_float(1176256512u)
#define RSW_NGPT 112
#define RSW_NBND 14
#define RSW_EXP_N 10001
#define RSW_EXP_FLOOR 1.0e-20f     // expeps: table entries 9276..10000 (tau >= 46.1) hold this floor

typedef float real;

// ---- verbatim from capture source 3c499c7b (rrtmg_sw_v842.py pin 44588ef0, lines 955-1070) ----
static __device__ __forceinline__ real mpas_sw_nan_result(real r, real a, real b)
{
    if (!isnan(r)) return r;
    if (isnan(a)) return __uint_as_float(__float_as_uint(a) | 0x00400000u);
    if (isnan(b)) return __uint_as_float(__float_as_uint(b) | 0x00400000u);
    return __uint_as_float(0xffc00000u);
}
static __device__ __forceinline__ real mpas_sw_div(real a, real b)
{ return mpas_sw_nan_result(__fdiv_rn(a,b),a,b); }
static __device__ __forceinline__ real mpas_sw_sqrt(real a)
{ return mpas_sw_nan_result(__fsqrt_rn(a),a,0.0f); }
static __device__ __forceinline__ real rsw_rn_add(real a, real b)
{ real r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return mpas_sw_nan_result(r,a,b); }
static __device__ __forceinline__ real rsw_rn_sub(real a, real b)
{ real r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return mpas_sw_nan_result(r,a,b); }
static __device__ __forceinline__ real rsw_rn_mul(real a, real b)
{ real r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return mpas_sw_nan_result(r,a,b); }
#define AD(a, b) rsw_rn_add((a), (b))
#define SU(a, b) rsw_rn_sub((a), (b))
#define MU(a, b) rsw_rn_mul((a), (b))
#define DV(a, b) mpas_sw_div((a), (b))
// ---- end verbatim ----

// Dual number with a float64 shadow primal: v is the forward's f32 word (it alone decides every branch, clamp, limiter
// and table index, and __check proves it bitwise); w is the same statement evaluated in float64 from the same inputs
// along the same executed branches; d is the f64 derivative, formed with the w coefficients.  The shadow is needed
// because the two-stream formulas cancel catastrophically for thin layers (O(1) terms giving an O(tau) result): the
// partials of the f32 words there are rounding noise (measured on step 6: d ref/d taug 7.9e3 from the f32 words against
// -1.5e-3 from the float64 map, on a layer with tau = 6e-5).
struct D { float v; double w; double d; };
static __device__ __forceinline__ D dc(float v) { D r; r.v = v; r.w = (double)v; r.d = 0.0; return r; }
static __device__ __forceinline__ D dx(float v, double d) { D r; r.v = v; r.w = (double)v; r.d = d; return r; }
// Step Jacobians (vrtqdr steps and fluxes have no branch, so v is not read there): from a stored shadow word.
static __device__ __forceinline__ D dw(double w) { D r; r.v = (float)w; r.w = w; r.d = 0.0; return r; }
static __device__ __forceinline__ D dA(D a, D b) { D r; r.v = AD(a.v, b.v); r.w = a.w + b.w; r.d = a.d + b.d; return r; }
static __device__ __forceinline__ D dS(D a, D b) { D r; r.v = SU(a.v, b.v); r.w = a.w - b.w; r.d = a.d - b.d; return r; }
static __device__ __forceinline__ D dM(D a, D b) { D r; r.v = MU(a.v, b.v); r.w = a.w * b.w; r.d = a.d * b.w + a.w * b.d; return r; }
static __device__ __forceinline__ D dV(D a, D b) {
    D r; r.v = DV(a.v, b.v); r.w = a.w / b.w; r.d = (a.d - r.w * b.d) / b.w; return r; }
static __device__ __forceinline__ D dQ(D a) { D r; r.v = mpas_sw_sqrt(a.v); r.w = sqrt(a.w); r.d = a.d / (2.0 * r.w); return r; }

#define BAD_PRIMAL 1u
#define BAD_DERIV 2u
#define BAD_TABLE 4u
struct Ctx { const real* T; unsigned bad; };

// rsw_etbl (source :1409-1416) and reftra's inline lookups (source :2645-2660): nearest entry, or 1 - x + x^2/2.
static __device__ __forceinline__ D d_etbl(Ctx& c, D x) {
    if (x.v <= RSW_C_OD_LO) return dA(dS(dc(1.0f), x), dM(dM(dc(0.5f), x), x));
    real tblind = DV(x.v, AD(RSW_C_BPADE, x.v));
    int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
    if (!(itind >= 0 && itind < RSW_EXP_N)) { c.bad |= BAD_TABLE; itind = 0; }
    D r; r.v = c.T[itind];
    // the tabulated function max(exp(-x), 1e-20) (header): exp(-x) and -exp(-x) dx above the floor, constant on it.
    if (r.v == RSW_EXP_FLOOR) { r.w = 1.0e-20; r.d = 0.0; } else { r.w = exp(-x.w); r.d = -r.w * x.d; }
    return r;
}

struct RT { D ref, refd, tra, trad; };

#define RSW_RES_TAU 1.0e-3   // two-stream resonance band |1 - (zrk*mu)^2| < tau (header, hazard 1)
#define RSW_RES_S 1.0e-3     // relative shift of the float64 shadow of mu used to regularize the derivative there

// The non-conservative two-stream form of rsw_reftra_layer (source :2617-2683), statement for statement, from the
// gamma coefficients on.  zrpp_w returns the shadow of zrpp = 1 - (zrk*mu)^2.
static __device__ RT d_reftra_nc(Ctx& c, D zg, D prmuz, D zto1, D zw, double* zrpp_w) {
    const real eps = 1.0e-08f;
    RT r;
    D zg3 = dM(dc(3.0f), zg);
    D zgamma1 = dM(dS(dc(8.0f), dM(zw, dA(dc(5.0f), zg3))), dc(0.25f));
    D zgamma2 = dM(dM(dc(3.0f), dM(zw, dS(dc(1.0f), zg))), dc(0.25f));
    D zgamma3 = dM(dS(dc(2.0f), dM(zg3, prmuz)), dc(0.25f));
    D zgamma4 = dS(dc(1.0f), zgamma3);
    D za1 = dA(dM(zgamma1, zgamma4), dM(zgamma2, zgamma3));
    D za2 = dA(dM(zgamma1, zgamma3), dM(zgamma2, zgamma4));
    D zrk = dQ(dS(dM(zgamma1, zgamma1), dM(zgamma2, zgamma2)));
    D zrp = dM(zrk, prmuz);
    D zrp1 = dA(dc(1.0f), zrp);
    D zrm1 = dS(dc(1.0f), zrp);
    D zrk2 = dM(dc(2.0f), zrk);
    D zrpp = dS(dc(1.0f), dM(zrp, zrp));
    D zrkg = dA(zrk, zgamma1);
    D zr1 = dM(zrm1, dA(za2, dM(zrk, zgamma3)));
    D zr2 = dM(zrp1, dS(za2, dM(zrk, zgamma3)));
    D zr3 = dM(zrk2, dS(zgamma3, dM(za2, prmuz)));
    D zr4 = dM(zrpp, zrkg);
    D zr5 = dM(zrpp, dS(zrk, zgamma1));
    D zt1 = dM(zrp1, dA(za1, dM(zrk, zgamma4)));
    D zt2 = dM(zrm1, dS(za1, dM(zrk, zgamma4)));
    D zt3 = dM(zrk2, dA(zgamma4, dM(za1, prmuz)));
    D zt4 = zr4, zt5 = zr5;
    D zbeta = dV(dS(zgamma1, zrk), zrkg);
    D ze1 = dM(zrk, zto1);
    if (!(ze1.v < 500.0f)) ze1 = dc(500.0f);
    D ze2 = dV(zto1, prmuz);
    if (!(ze2.v < 500.0f)) ze2 = dc(500.0f);
    D zem1 = d_etbl(c, ze1); D zep1 = dV(dc(1.0f), zem1);
    D zem2 = d_etbl(c, ze2); D zep2 = dV(dc(1.0f), zem2);
    D zdenr = dA(dM(zr4, zep1), dM(zr5, zem1));
    D zdent = dA(dM(zt4, zep1), dM(zt5, zem1));
    if (zdenr.v >= -eps && zdenr.v <= eps) {
        r.ref = dc(eps);
        r.tra = zem2;
    } else {
        r.ref = dV(dM(zw, dS(dS(dM(zr1, zep1), dM(zr2, zem1)), dM(zr3, zem2))), zdenr);
        r.tra = dS(zem2, dV(dM(dM(zem2, zw), dS(dS(dM(zt1, zep1), dM(zt2, zem1)), dM(zt3, zep2))), zdent));
    }
    D zemm = dM(zem1, zem1);
    D zdend = dV(dc(1.0f), dM(dS(dc(1.0f), dM(zbeta, zemm)), zrkg));
    r.refd = dM(dM(zgamma2, dS(dc(1.0f), zemm)), zdend);
    r.trad = dM(dM(zrk2, zem1), zdend);
    *zrpp_w = zrpp.w;
    return r;
}

// rsw_reftra_layer (source :2569-2684), statement for statement.
static __device__ RT d_reftra(Ctx& c, bool lrtchk, D pgg, D prmuz, D ptau, D pw) {
    const real zwcrit = 0.9999995f;
    RT r;
    if (!lrtchk) { r.ref = dc(0.0f); r.tra = dc(1.0f); r.refd = dc(0.0f); r.trad = dc(1.0f); return r; }
    D zto1 = ptau, zw = pw, zg = pgg;
    real zwo = 0.0f, denom = 1.0f;                      // branch selector only (frozen)
    if (zg.v != 1.0f) {
        real q = DV(zg.v, SU(1.0f, zg.v));
        denom = SU(1.0f, MU(SU(1.0f, zw.v), MU(q, q)));
    }
    if (zw.v > 0.0f && denom != 0.0f) zwo = DV(zw.v, denom);
    if (zwo >= zwcrit) {
        D zg3 = dM(dc(3.0f), zg);
        D zgamma1 = dM(dS(dc(8.0f), dM(zw, dA(dc(5.0f), zg3))), dc(0.25f));
        D zgamma3 = dM(dS(dc(2.0f), dM(zg3, prmuz)), dc(0.25f));
        D za = dM(zgamma1, prmuz);
        D za1 = dS(za, zgamma3);
        D zgt = dM(zgamma1, zto1);
        D ze1 = dV(zto1, prmuz);
        if (!(ze1.v < 500.0f)) ze1 = dc(500.0f);
        D ze2 = d_etbl(c, ze1);
        r.ref = dV(dS(zgt, dM(za1, dS(dc(1.0f), ze2))), dA(dc(1.0f), zgt));
        r.tra = dS(dc(1.0f), r.ref);
        r.refd = dV(zgt, dA(dc(1.0f), zgt));
        r.trad = dS(dc(1.0f), r.refd);
        if (ze2.v == 1.0f) { r.ref = dc(0.0f); r.tra = dc(1.0f); r.refd = dc(0.0f); r.trad = dc(1.0f); }
    } else {
        double rpp;
        r = d_reftra_nc(c, zg, prmuz, zto1, zw, &rpp);
        // Two-stream resonance (hazard 1): zrpp = 1 - (zrk*mu)^2 sits in both the numerator and zdenr, a removable
        // 0/0. There even the float64 shadow's derivative is rounding noise (measured on step 1, column 471, g 0,
        // layer 8, |zrpp| = 3.4e-7: d ref/d ptaua -3.59 from the shadow against 0.148 from a 50-digit evaluation; the
        // f32 forward word itself is ref = -0.034 against 2.6e-6). Inside the band the derivative is the mean of
        // the derivatives at mu*(1 +- 1e-3) (shadow only: every branch, clamp and table index stays the base
        // point's); values v and w are the base point's. Bias O(1e-6) relative; outside the band nothing changes.
        if (fabs(rpp) < RSW_RES_TAU) {
            D mp = prmuz, mm = prmuz; mp.w *= 1.0 + RSW_RES_S; mm.w *= 1.0 - RSW_RES_S;
            double t;
            RT a = d_reftra_nc(c, zg, mp, zto1, zw, &t), b = d_reftra_nc(c, zg, mm, zto1, zw, &t);
            r.ref.d = 0.5 * (a.ref.d + b.ref.d); r.refd.d = 0.5 * (a.refd.d + b.refd.d);
            r.tra.d = 0.5 * (a.tra.d + b.tra.d); r.trad.d = 0.5 * (a.trad.d + b.trad.d);
        }
    }
    return r;
}

// One layer of rsw_spcvmc_body (source :2846-2917). Inputs (NIN = 11): 0 cf=pcldfmc 1 tcl=ptaucmc 2 acl=pasycmc
// 3 wcl=pomgcmc 4 tor=ptaormc 5 ta=ptaua 6 ga=pasya 7 wa=pomga 8 mu=prmu0 9 tg=taug 10 tr=taur.
// Outputs (NOUT = 12): 0-3 clear ref,refd,tra,trad; 4-7 total ref,refd,tra,trad; 8 dbtc; 9 dbt; 10 nodel factor clear
// (zdbtc_nodel); 11 nodel factor total (zdbt_nodel).  Sky s (0 clear, 1 total): reftra 4s..4s+3, dbt 8+s, nodel 10+s.
#define NIN 11
#define NOUT 12
struct LayOut { D o[NOUT]; };
static __device__ LayOut d_layer(Ctx& c, const D* x) {
    const real repclc = 1.0e-12f;
    D cf = x[0], tcl = x[1], acl = x[2], wcl = x[3], tor = x[4], ta = x[5], ga = x[6], wa = x[7], mu = x[8], tg = x[9], tr = x[10];
    bool lrtchkcld = (cf.v > repclc);
    LayOut L;
    D ztauc = dA(dA(tr, tg), ta);
    D zomcc = dA(dM(tr, dc(1.0f)), dM(ta, wa));
    D zgcc = dV(dM(dM(ga, wa), ta), zomcc);
    zomcc = dV(zomcc, ztauc);
    D zclear = dS(dc(1.0f), cf);
    D zcloud = cf;
    D ze1 = dV(ztauc, mu);
    D zdbtmc = d_etbl(c, ze1);
    L.o[10] = zdbtmc;
    D tauorig = dA(ztauc, tor);
    ze1 = dV(tauorig, mu);
    D zdbtmo = d_etbl(c, ze1);
    L.o[11] = dA(dM(zclear, zdbtmc), dM(zcloud, zdbtmo));
    D zf = dM(zgcc, zgcc);
    D zwf = dM(zomcc, zf);
    ztauc = dM(dS(dc(1.0f), zwf), ztauc);
    zomcc = dV(dS(zomcc, zwf), dS(dc(1.0f), zwf));
    zgcc = dV(dS(zgcc, zf), dS(dc(1.0f), zf));
    D ztauo = dA(ztauc, tcl);
    D zomco = dA(dM(ztauc, zomcc), dM(tcl, wcl));
    D zgco = dV(dA(dM(dM(tcl, wcl), acl), dM(dM(ztauc, zomcc), zgcc)), zomco);
    zomco = dV(zomco, ztauo);
    RT k = d_reftra(c, true, zgcc, mu, ztauc, zomcc);
    RT o = d_reftra(c, lrtchkcld, zgco, mu, ztauo, zomco);
    L.o[0] = k.ref; L.o[1] = k.refd; L.o[2] = k.tra; L.o[3] = k.trad;
    L.o[4] = dA(dM(zclear, k.ref), dM(zcloud, o.ref));
    L.o[5] = dA(dM(zclear, k.refd), dM(zcloud, o.refd));
    L.o[6] = dA(dM(zclear, k.tra), dM(zcloud, o.tra));
    L.o[7] = dA(dM(zclear, k.trad), dM(zcloud, o.trad));
    ze1 = dV(ztauc, mu);
    zdbtmc = d_etbl(c, ze1);
    L.o[8] = zdbtmc;
    ze1 = dV(ztauo, mu);
    zdbtmo = d_etbl(c, ze1);
    L.o[9] = dA(dM(zclear, zdbtmc), dM(zcloud, zdbtmo));
    return L;
}

// rsw_vrtqdr (source :2691-2759) steps.  The first upward statement (source :2694-2700) is the loop statement with
// rup[klev] = palbp = pref[klev], rupd[klev] = palbd = prefd[klev] (same words, same operand order).
struct D2 { D a, b; };
// Upward (one ikx): inputs rupA, rupdA (level below), ref, refd, tra, trad, dbt; outputs rup, rupd.
static __device__ __forceinline__ D2 d_up(D rupA, D rupdA, D ref, D refd, D tra, D trad, D dbt) {
    D zreflect = dV(dc(1.0f), dS(dc(1.0f), dM(rupdA, refd)));
    D2 r;
    r.a = dA(ref, dM(dM(trad, dA(dM(dS(tra, dbt), rupdA), dM(dbt, rupA))), zreflect));
    r.b = dA(refd, dM(dM(dM(trad, trad), rupdA), zreflect));
    return r;
}
// Downward (layer c >= 1): inputs tdn_c, rdnd_c, tdbt_c, ref, refd, tra, trad; outputs tdn, rdnd at c+1.
// Layer 0 (source :2721-2722): tdn[1] = tra[0], rdnd[1] = refd[0].
static __device__ __forceinline__ D2 d_down(D tdn, D rdnd, D tdbt, D ref, D refd, D tra, D trad) {
    D zreflect = dV(dc(1.0f), dS(dc(1.0f), dM(refd, rdnd)));
    D2 r;
    r.a = dA(dM(tdbt, tra), dM(dM(trad, dA(dS(tdn, tdbt), dM(dM(tdbt, ref), rdnd))), zreflect));
    r.b = dA(refd, dM(dM(dM(trad, trad), rdnd), zreflect));
    return r;
}
// Fluxes at one level (source :2737-2745): inputs tdbt, tdn, rdnd, rup, rupd; outputs a = fu, b = fd.
static __device__ __forceinline__ D2 d_flux(D tdbt, D tdn, D rdnd, D rup, D rupd) {
    D zreflect = dV(dc(1.0f), dS(dc(1.0f), dM(rdnd, rupd)));
    D2 r;
    r.a = dM(dA(dM(tdbt, rup), dM(dS(tdn, tdbt), rupd)), zreflect);
    r.b = dA(tdbt, dM(dA(dS(tdn, tdbt), dM(dM(tdbt, rup), rdnd)), zreflect));
    return r;
}

static __device__ __forceinline__ bool fin(float v) { return isfinite(v); }
static __device__ __forceinline__ float st(double d, unsigned& bad) { float t = (float)(d + 0.0); if (!isfinite(t)) bad |= BAD_DERIV; return t + 0.0f; }
static __device__ __forceinline__ void chk(const D& x, unsigned& bad) { if (!fin(x.v)) bad |= BAD_PRIMAL; if (!isfinite(x.d)) bad |= BAD_DERIV; }

// Per-(col, g) addressing of the forward's batched layout (source :2984-3035, interleaved).
struct Item { int col, g, band; size_t gl, bl, o1, ob; };
static __device__ __forceinline__ Item item(int col, int g, int nl, const int* ngb) {
    Item it; it.col = col; it.g = g; it.band = ngb[g] - 16;
    it.gl = (size_t)col * nl * RSW_NGPT + g;                       // + K*112: (col, K, g)
    it.bl = (size_t)col * nl * RSW_NBND + (size_t)it.band * nl;    // + K:     (col, band, K)
    it.o1 = (size_t)col * (nl + 1) * RSW_NGPT + g;                 // + j*112: (col, j, g)
    it.ob = (size_t)col * RSW_NBND + it.band;                      //          (col, band)
    return it;
}
static __device__ __forceinline__ size_t xidx(int k, const Item& it, int K) {
    if (k >= 5 && k <= 7) return it.bl + K;
    if (k == 8) return (size_t)it.col;
    return it.gl + (size_t)K * RSW_NGPT;
}

#define SPC_PRIMAL_ARGS int ncol, int nlayers, const real* __restrict__ tab, const int* __restrict__ ngb, \
    const real* __restrict__ palbd, const real* __restrict__ palbp, const real* __restrict__ pcldfmc, \
    const real* __restrict__ ptaucmc, const real* __restrict__ pasycmc, const real* __restrict__ pomgcmc, \
    const real* __restrict__ ptaormc, const real* __restrict__ ptaua, const real* __restrict__ pasya, \
    const real* __restrict__ pomga, const real* __restrict__ prmu0, const real* __restrict__ adjflux, \
    const real* __restrict__ sfluxzen, const real* __restrict__ taug, const real* __restrict__ taur
#define SPC_X_ARRAYS const real* xp[NIN] = {pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc, ptaua, pasya, pomga, prmu0, taug, taur}

static __device__ __forceinline__ void load_primal(const real* const* xp, const Item& it, int K, D* x, unsigned& bad) {
    for (int k = 0; k < NIN; ++k) { x[k] = dc(xp[k][xidx(k, it, K)]); if (!fin(x[k].v)) bad |= BAD_PRIMAL; }
}

// Interleaved workspace: entry (a, j) of thread t at ((a*n1 + j)*RES + t).
#define W(p, a, j) (p)[((size_t)(a) * n1 + (size_t)(j)) * RES + t]

// ---------------------------------------------------------------------------------------------------------------
// Harness only (not in launch_abi): the forward's output words from the restructured passes (the TL's primal path).
// Workspace: the TL's (4 f64 + 4 f32 arrays of n1 per thread).
extern "C" __global__ void rsw_spcvmc_gpt_b__check(SPC_PRIMAL_ARGS, real* zincflx, real* zcd, real* zcu, real* zfd,
    real* zfu, real* ztdbt_nodel_out, real* ztdbtc_nodel_out, double* ws, int* failure) {
    SPC_X_ARRAYS;
    const int nl = nlayers, klev = nlayers, n1 = nlayers + 1, g = threadIdx.x;
    const size_t RES = (size_t)gridDim.x * blockDim.x, t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= RSW_NGPT) return;
    if (nl < 1 || ngb[g] < 16 || ngb[g] > 29) { atomicExch(failure + 3, 1); return; }
    float* rp = (float*)(ws + 4 * (size_t)n1 * RES);
    for (int col = blockIdx.x; col < ncol; col += gridDim.x) {
        Item it = item(col, g, nl, ngb); Ctx c; c.T = tab + RSW_T_EXP_TBL; c.bad = 0;
        D x[NIN];
        D albp = dc(palbp[it.ob]), albd = dc(palbd[it.ob]);
        D ru[2] = {albp, albp}, rd[2] = {albd, albd};
        for (int s = 0; s < 2; ++s) { W(rp, 2 * s, klev) = ru[s].v; W(rp, 2 * s + 1, klev) = rd[s].v; }
        for (int b = klev - 1; b >= 0; --b) {
            load_primal(xp, it, klev - 1 - b, x, c.bad); LayOut L = d_layer(c, x);
            for (int s = 0; s < 2; ++s) {
                D2 u = d_up(ru[s], rd[s], L.o[4*s], L.o[4*s+1], L.o[4*s+2], L.o[4*s+3], L.o[8+s]);
                ru[s] = u.a; rd[s] = u.b; W(rp, 2 * s, b) = u.a.v; W(rp, 2 * s + 1, b) = u.b.v;
            }
        }
        real* fu[2] = {zcu, zfu}; real* fd[2] = {zcd, zfd}; real* nd[2] = {ztdbtc_nodel_out, ztdbt_nodel_out};
        D tdbt[2] = {dc(1.0f), dc(1.0f)}, tdn[2] = {dc(1.0f), dc(1.0f)}, rdnd[2] = {dc(0.0f), dc(0.0f)}, nod[2] = {dc(1.0f), dc(1.0f)};
        for (int s = 0; s < 2; ++s) {
            nd[s][it.o1] = 1.0f;
            D2 f = d_flux(tdbt[s], tdn[s], rdnd[s], dc(W(rp, 2 * s, 0)), dc(W(rp, 2 * s + 1, 0)));
            fu[s][it.o1] = f.a.v; fd[s][it.o1] = f.b.v;
        }
        for (int cc = 0; cc < klev; ++cc) {
            load_primal(xp, it, klev - 1 - cc, x, c.bad); LayOut L = d_layer(c, x);
            size_t o = it.o1 + (size_t)(cc + 1) * RSW_NGPT;
            for (int s = 0; s < 2; ++s) {
                D ref = L.o[4*s], refd = L.o[4*s+1], tra = L.o[4*s+2], trad = L.o[4*s+3];
                D2 dn; if (cc == 0) { dn.a = tra; dn.b = refd; } else dn = d_down(tdn[s], rdnd[s], tdbt[s], ref, refd, tra, trad);
                tdbt[s] = dM(L.o[8+s], tdbt[s]); tdn[s] = dn.a; rdnd[s] = dn.b;
                nod[s] = dM(L.o[10+s], nod[s]); nd[s][o] = nod[s].v;
                D2 f = d_flux(tdbt[s], tdn[s], rdnd[s], dc(W(rp, 2 * s, cc + 1)), dc(W(rp, 2 * s + 1, cc + 1)));
                fu[s][o] = f.a.v; fd[s][o] = f.b.v;
                if (!fin(f.a.v) || !fin(f.b.v)) c.bad |= BAD_PRIMAL;
            }
        }
        zincflx[(size_t)col * RSW_NGPT + g] = MU(MU(adjflux[it.ob], sfluxzen[(size_t)col * RSW_NGPT + g]), prmu0[col]);
        if (c.bad & BAD_PRIMAL) atomicExch(failure + 0, 1);
        if (c.bad & BAD_TABLE) atomicExch(failure + 2, 1);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tangent: outputs fully overwritten (io out).  Workspace per thread: 4 f64 arrays (d rup, d rupd per sky) then
// 4 f64 arrays (shadow rup, rupd per sky), n1 entries each:  bytes = RES * n1 * 64.
extern "C" __global__ void __launch_bounds__(128) rsw_spcvmc_gpt_b__tl(SPC_PRIMAL_ARGS,
    const real* d_palbd, const real* d_palbp, const real* d_pcldfmc, const real* d_ptaucmc, const real* d_pasycmc,
    const real* d_pomgcmc, const real* d_ptaormc, const real* d_ptaua, const real* d_pasya, const real* d_pomga,
    const real* d_prmu0, const real* d_adjflux, const real* d_sfluxzen, const real* d_taug, const real* d_taur,
    real* d_zincflx, real* d_zcd, real* d_zcu, real* d_zfd, real* d_zfu, real* d_ztdbt_nodel_out, real* d_ztdbtc_nodel_out,
    double* ws, int* failure) {
    SPC_X_ARRAYS;
    const real* dxp[NIN] = {d_pcldfmc, d_ptaucmc, d_pasycmc, d_pomgcmc, d_ptaormc, d_ptaua, d_pasya, d_pomga, d_prmu0, d_taug, d_taur};
    const int nl = nlayers, klev = nlayers, n1 = nlayers + 1, g = threadIdx.x;
    const size_t RES = (size_t)gridDim.x * blockDim.x, t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= RSW_NGPT) return;
    if (nl < 1 || ngb[g] < 16 || ngb[g] > 29) { atomicExch(failure + 3, 1); return; }
    double* tp = ws;                                   // d rup, d rupd (clear, total)
    double* rp = ws + 4 * (size_t)n1 * RES;            // shadow rup, rupd (clear, total)
    for (int col = blockIdx.x; col < ncol; col += gridDim.x) {
        Item it = item(col, g, nl, ngb); Ctx c; c.T = tab + RSW_T_EXP_TBL; c.bad = 0;
        D x[NIN];
        D albp = dx(palbp[it.ob], (double)d_palbp[it.ob]), albd = dx(palbd[it.ob], (double)d_palbd[it.ob]);
        D ru[2] = {albp, albp}, rd[2] = {albd, albd};
        for (int s = 0; s < 2; ++s) { W(rp, 2*s, klev) = ru[s].w; W(rp, 2*s+1, klev) = rd[s].w; W(tp, 2*s, klev) = ru[s].d; W(tp, 2*s+1, klev) = rd[s].d; }
        for (int b = klev - 1; b >= 0; --b) {                                 // pass A
            const int K = klev - 1 - b;
            for (int k = 0; k < NIN; ++k) { size_t i = xidx(k, it, K); x[k] = dx(xp[k][i], (double)dxp[k][i]);
                if (!fin(x[k].v)) c.bad |= BAD_PRIMAL; if (!isfinite(x[k].d)) c.bad |= BAD_DERIV; }
            LayOut L = d_layer(c, x);
            for (int s = 0; s < 2; ++s) {
                D2 u = d_up(ru[s], rd[s], L.o[4*s], L.o[4*s+1], L.o[4*s+2], L.o[4*s+3], L.o[8+s]);
                ru[s] = u.a; rd[s] = u.b;
                W(rp, 2*s, b) = u.a.w; W(rp, 2*s+1, b) = u.b.w; W(tp, 2*s, b) = u.a.d; W(tp, 2*s+1, b) = u.b.d;
            }
        }
        real* fu[2] = {d_zcu, d_zfu}; real* fd[2] = {d_zcd, d_zfd}; real* nd[2] = {d_ztdbtc_nodel_out, d_ztdbt_nodel_out};
        D tdbt[2] = {dc(1.0f), dc(1.0f)}, tdn[2] = {dc(1.0f), dc(1.0f)}, rdnd[2] = {dc(0.0f), dc(0.0f)}, nod[2] = {dc(1.0f), dc(1.0f)};
        for (int s = 0; s < 2; ++s) {                                         // pass B, level 0
            nd[s][it.o1] = 0.0f;
            D ru0 = dw(W(rp, 2*s, 0)), rd0 = dw(W(rp, 2*s+1, 0)); ru0.d = W(tp, 2*s, 0); rd0.d = W(tp, 2*s+1, 0);
            D2 f = d_flux(tdbt[s], tdn[s], rdnd[s], ru0, rd0);
            fu[s][it.o1] = st(f.a.d, c.bad); fd[s][it.o1] = st(f.b.d, c.bad); chk(f.a, c.bad); chk(f.b, c.bad);
        }
        for (int cc = 0; cc < klev; ++cc) {
            const int K = klev - 1 - cc;
            for (int k = 0; k < NIN; ++k) { size_t i = xidx(k, it, K); x[k] = dx(xp[k][i], (double)dxp[k][i]); }
            LayOut L = d_layer(c, x);
            size_t o = it.o1 + (size_t)(cc + 1) * RSW_NGPT;
            for (int s = 0; s < 2; ++s) {
                D ref = L.o[4*s], refd = L.o[4*s+1], tra = L.o[4*s+2], trad = L.o[4*s+3];
                D2 dn; if (cc == 0) { dn.a = tra; dn.b = refd; } else dn = d_down(tdn[s], rdnd[s], tdbt[s], ref, refd, tra, trad);
                tdbt[s] = dM(L.o[8+s], tdbt[s]); tdn[s] = dn.a; rdnd[s] = dn.b;
                nod[s] = dM(L.o[10+s], nod[s]); nd[s][o] = st(nod[s].d, c.bad); chk(nod[s], c.bad);
                D ru1 = dw(W(rp, 2*s, cc + 1)), rd1 = dw(W(rp, 2*s+1, cc + 1)); ru1.d = W(tp, 2*s, cc + 1); rd1.d = W(tp, 2*s+1, cc + 1);
                D2 f = d_flux(tdbt[s], tdn[s], rdnd[s], ru1, rd1);
                fu[s][o] = st(f.a.d, c.bad); fd[s][o] = st(f.b.d, c.bad); chk(f.a, c.bad); chk(f.b, c.bad);
            }
        }
        size_t ig = (size_t)col * RSW_NGPT + g;
        D a = dx(adjflux[it.ob], (double)d_adjflux[it.ob]), sf = dx(sfluxzen[ig], (double)d_sfluxzen[ig]), mu = dx(prmu0[col], (double)d_prmu0[col]);
        D z = dM(dM(a, sf), mu); d_zincflx[ig] = st(z.d, c.bad); chk(z, c.bad);
        if (c.bad & BAD_PRIMAL) atomicExch(failure + 0, 1);
        if (c.bad & BAD_DERIV) atomicExch(failure + 1, 1);
        if (c.bad & BAD_TABLE) atomicExch(failure + 2, 1);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Adjoint: output seeds consumed (io inout: zeroed, the forward overwrites every output word), input cotangents
// accumulated.  Workspace per thread: 16 f64 arrays (12 per-layer seeds, 4 flux seeds of rup/rupd) then 24 f64 shadow
// primal arrays (tdbt, tdn, rdnd, rup, rupd per sky, nodel per sky, the 12 layer outputs), n1 entries each:
// bytes = RES * n1 * 320.
#define S_REF(s, q) (4 * (s) + (q))     // 0..7: reftra seeds (q: 0 ref, 1 refd, 2 tra, 3 trad)
#define S_DBT(s) (8 + (s))
#define S_NOD(s) (10 + (s))
#define S_RUPF(s) (12 + 2 * (s))
#define S_RUPDF(s) (13 + 2 * (s))
#define P_TDBT(s) (0 + 5 * (s))
#define P_TDN(s) (1 + 5 * (s))
#define P_RDND(s) (2 + 5 * (s))
#define P_RUP(s) (3 + 5 * (s))
#define P_RUPD(s) (4 + 5 * (s))
#define P_NOD(s) (10 + (s))
#define P_O(q) (12 + (q))               // the 12 layer outputs of layer J (f32 primal, stored by P1)

extern "C" __global__ void __launch_bounds__(128) rsw_spcvmc_gpt_b__ad(SPC_PRIMAL_ARGS,
    real* b_palbd, real* b_palbp, real* b_pcldfmc, real* b_ptaucmc, real* b_pasycmc, real* b_pomgcmc, real* b_ptaormc,
    real* b_ptaua, real* b_pasya, real* b_pomga, real* b_prmu0, real* b_adjflux, real* b_sfluxzen, real* b_taug, real* b_taur,
    real* b_zincflx, real* b_zcd, real* b_zcu, real* b_zfd, real* b_zfu, real* b_ztdbt_nodel_out, real* b_ztdbtc_nodel_out,
    double* ws, int* failure) {
    SPC_X_ARRAYS;
    real* bxp[NIN] = {b_pcldfmc, b_ptaucmc, b_pasycmc, b_pomgcmc, b_ptaormc, nullptr, nullptr, nullptr, nullptr, b_taug, b_taur};
    __shared__ double sh[3][RSW_NGPT];
    __shared__ int sband[RSW_NGPT];
    __shared__ int sbad;
    const int nl = nlayers, klev = nlayers, n1 = nlayers + 1, g = threadIdx.x;
    const size_t RES = (size_t)gridDim.x * blockDim.x, t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const bool act = g < RSW_NGPT;
    if (threadIdx.x == 0) sbad = 0;
    __syncthreads();
    if (act) { sband[g] = ngb[g] - 16; if (nl < 1 || ngb[g] < 16 || ngb[g] > 29) atomicExch(&sbad, 1); }
    __syncthreads();
    if (sbad) { if (threadIdx.x == 0) atomicExch(failure + 3, 1); return; }    // block-uniform exit
    double* sp = ws;
    double* pp = ws + 16 * (size_t)n1 * RES;
    for (int col = blockIdx.x; col < ncol; col += gridDim.x) {
        Item it; if (act) it = item(col, g, nl, ngb); else { it.col = col; it.g = 0; it.band = 0; it.gl = it.bl = it.o1 = it.ob = 0; }
        Ctx c; c.T = tab + RSW_T_EXP_TBL; c.bad = 0;
        double bmu = 0.0, badj = 0.0;
        if (act) {
            D x[NIN];
            // P1: top-down primal tdbt, tdn, rdnd, nodel (levels 0..klev).
            D tdbt[2] = {dc(1.0f), dc(1.0f)}, tdn[2] = {dc(1.0f), dc(1.0f)}, rdnd[2] = {dc(0.0f), dc(0.0f)}, nod[2] = {dc(1.0f), dc(1.0f)};
            for (int s = 0; s < 2; ++s) { W(pp, P_TDBT(s), 0) = 1.0; W(pp, P_TDN(s), 0) = 1.0; W(pp, P_RDND(s), 0) = 0.0; W(pp, P_NOD(s), 0) = 1.0; }
            for (int cc = 0; cc < klev; ++cc) {
                load_primal(xp, it, klev - 1 - cc, x, c.bad); LayOut L = d_layer(c, x);
                for (int q = 0; q < NOUT; ++q) { W(pp, P_O(q), cc) = L.o[q].w; if (!fin(L.o[q].v) || !isfinite(L.o[q].w)) c.bad |= BAD_PRIMAL; }
                for (int s = 0; s < 2; ++s) {
                    D2 dn; if (cc == 0) { dn.a = L.o[4*s+2]; dn.b = L.o[4*s+1]; }
                    else dn = d_down(tdn[s], rdnd[s], tdbt[s], L.o[4*s], L.o[4*s+1], L.o[4*s+2], L.o[4*s+3]);
                    tdbt[s] = dM(L.o[8+s], tdbt[s]); tdn[s] = dn.a; rdnd[s] = dn.b; nod[s] = dM(L.o[10+s], nod[s]);
                    W(pp, P_TDBT(s), cc + 1) = tdbt[s].w; W(pp, P_TDN(s), cc + 1) = tdn[s].w; W(pp, P_RDND(s), cc + 1) = rdnd[s].w;
                    W(pp, P_NOD(s), cc + 1) = nod[s].w;
                }
            }
            // P2: bottom-up primal rup, rupd (levels klev..0).
            D albp = dc(palbp[it.ob]), albd = dc(palbd[it.ob]);
            D ru[2] = {albp, albp}, rd[2] = {albd, albd};
            for (int s = 0; s < 2; ++s) { W(pp, P_RUP(s), klev) = albp.w; W(pp, P_RUPD(s), klev) = albd.w; }
            for (int b = klev - 1; b >= 0; --b) {
                for (int s = 0; s < 2; ++s) {
                    D2 u = d_up(ru[s], rd[s], dw(W(pp, P_O(4*s), b)), dw(W(pp, P_O(4*s+1), b)), dw(W(pp, P_O(4*s+2), b)),
                                dw(W(pp, P_O(4*s+3), b)), dw(W(pp, P_O(8+s), b)));
                    ru[s] = u.a; rd[s] = u.b; W(pp, P_RUP(s), b) = u.a.w; W(pp, P_RUPD(s), b) = u.b.w;
                }
            }
            // R1: descending levels jk = klev..0.
            real* bfu[2] = {b_zcu, b_zfu}; real* bfd[2] = {b_zcd, b_zfd}; real* bnd[2] = {b_ztdbtc_nodel_out, b_ztdbt_nodel_out};
            double Btdbt[2] = {0, 0}, Btdn[2] = {0, 0}, Brdnd[2] = {0, 0}, Bnod[2] = {0, 0};
            for (int j = 0; j < n1; ++j) for (int a = 0; a < 16; ++a) W(sp, a, j) = 0.0;
            for (int jk = klev; jk >= 0; --jk) {
                size_t o = it.o1 + (size_t)jk * RSW_NGPT;
                for (int s = 0; s < 2; ++s) {
                    double yfu = (double)bfu[s][o], yfd = (double)bfd[s][o], yn = (double)bnd[s][o];
                    if (!isfinite(yfu) || !isfinite(yfd) || !isfinite(yn)) c.bad |= BAD_DERIV;
                    bfu[s][o] = 0.0f; bfd[s][o] = 0.0f; bnd[s][o] = 0.0f;
                    Bnod[s] += yn;
                    double pv[5] = {W(pp, P_TDBT(s), jk), W(pp, P_TDN(s), jk), W(pp, P_RDND(s), jk), W(pp, P_RUP(s), jk), W(pp, P_RUPD(s), jk)};
                    double acc[5];
                    for (int k = 0; k < 5; ++k) {
                        D v[5]; for (int q = 0; q < 5; ++q) v[q] = dw(pv[q]); v[k].d = 1.0;
                        D2 f = d_flux(v[0], v[1], v[2], v[3], v[4]);
                        acc[k] = f.a.d * yfu + f.b.d * yfd;
                    }
                    Btdbt[s] += acc[0]; Btdn[s] += acc[1]; Brdnd[s] += acc[2];
                    W(sp, S_RUPF(s), jk) = acc[3]; W(sp, S_RUPDF(s), jk) = acc[4];
                }
                if (jk == 0) break;                      // tdbt, tdn, rdnd and nodel at level 0 are constants
                const int cc = jk - 1;
                LayOut L; for (int q = 0; q < NOUT; ++q) L.o[q] = dw(W(pp, P_O(q), cc));
                for (int s = 0; s < 2; ++s) {
                    double nBtdn = 0.0, nBrdnd = 0.0, nBtdbt = 0.0;
                    if (cc == 0) {                       // tdn[1] = tra[0], rdnd[1] = refd[0]
                        W(sp, S_REF(s, 2), cc) += Btdn[s]; W(sp, S_REF(s, 1), cc) += Brdnd[s];
                    } else {
                        double pv[7] = {W(pp, P_TDN(s), cc), W(pp, P_RDND(s), cc), W(pp, P_TDBT(s), cc),
                                        L.o[4*s].w, L.o[4*s+1].w, L.o[4*s+2].w, L.o[4*s+3].w};
                        double acc[7];
                        for (int k = 0; k < 7; ++k) {
                            D v[7]; for (int q = 0; q < 7; ++q) v[q] = dw(pv[q]); v[k].d = 1.0;
                            D2 dn = d_down(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
                            acc[k] = dn.a.d * Btdn[s] + dn.b.d * Brdnd[s];
                        }
                        nBtdn = acc[0]; nBrdnd = acc[1]; nBtdbt = acc[2];
                        for (int q = 0; q < 4; ++q) W(sp, S_REF(s, q), cc) += acc[3 + q];
                    }
                    // tdbt[jk] = MU(dbt[cc], tdbt[cc])  (source :2913-2914)
                    W(sp, S_DBT(s), cc) += Btdbt[s] * W(pp, P_TDBT(s), cc);
                    nBtdbt += Btdbt[s] * L.o[8+s].w;
                    // nodel[jk] = MU(nodel factor[cc], nodel[cc])  (source :2864, :2871)
                    W(sp, S_NOD(s), cc) += Bnod[s] * W(pp, P_NOD(s), cc);
                    Bnod[s] = Bnod[s] * L.o[10+s].w;
                    Btdn[s] = nBtdn; Brdnd[s] = nBrdnd; Btdbt[s] = nBtdbt;
                }
            }
            // zincflx = MU(MU(adjflux, sfluxzen), mu)  (source :2800)
            size_t ig = (size_t)col * RSW_NGPT + g;
            double y = (double)b_zincflx[ig]; if (!isfinite(y)) c.bad |= BAD_DERIV; b_zincflx[ig] = 0.0f;
            float a = adjflux[it.ob], sf = sfluxzen[ig], mu = prmu0[col];
            bmu += y * ((double)a * (double)sf);
            badj += (y * (double)mu) * (double)sf;
            b_sfluxzen[ig] = st((double)b_sfluxzen[ig] + (y * (double)mu) * (double)a, c.bad);
        }
        // R2: ascending layers b = 0..klev-1, block-synchronous (aerosol cotangents reduced per layer).
        double BrupC[2] = {0, 0}, BrupdC[2] = {0, 0};
        for (int b = 0; b < klev; ++b) {
            const int K = klev - 1 - b;
            if (act) {
                float pv[NIN];
                for (int k = 0; k < NIN; ++k) pv[k] = xp[k][xidx(k, it, K)];
                LayOut L; for (int q = 0; q < NOUT; ++q) L.o[q] = dw(W(pp, P_O(q), b));
                double Y[NOUT];
                for (int q = 0; q < NOUT; ++q) Y[q] = W(sp, q, b);
                for (int s = 0; s < 2; ++s) {
                    double Br = W(sp, S_RUPF(s), b) + BrupC[s], Brd = W(sp, S_RUPDF(s), b) + BrupdC[s];
                    double uv[7] = {W(pp, P_RUP(s), b + 1), W(pp, P_RUPD(s), b + 1), L.o[4*s].w, L.o[4*s+1].w, L.o[4*s+2].w,
                                    L.o[4*s+3].w, L.o[8+s].w};
                    double acc[7];
                    for (int k = 0; k < 7; ++k) {
                        D v[7]; for (int q = 0; q < 7; ++q) v[q] = dw(uv[q]); v[k].d = 1.0;
                        D2 u = d_up(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
                        acc[k] = u.a.d * Br + u.b.d * Brd;
                    }
                    BrupC[s] = acc[0]; BrupdC[s] = acc[1];
                    for (int q = 0; q < 4; ++q) Y[S_REF(s, q)] += acc[2 + q];
                    Y[S_DBT(s)] += acc[6];
                }
                double baer[3] = {0, 0, 0};
                const bool clear_item = (pv[0] == 0.0f);  // cf == 0: columns 1-4 (cloud optics, ptaormc) are exactly 0
                for (int k = 0; k < NIN; ++k) {           // layer Jacobian transpose, one unit direction per input
                    if (clear_item && k >= 1 && k <= 4) continue;
                    D v[NIN]; for (int q = 0; q < NIN; ++q) v[q] = dc(pv[q]); v[k].d = 1.0;
                    LayOut J = d_layer(c, v);
                    double s_ = 0.0; for (int q = 0; q < NOUT; ++q) s_ += J.o[q].d * Y[q];
                    if (k >= 5 && k <= 7) baer[k - 5] = s_;
                    else if (k == 8) bmu += s_;
                    else { size_t i = xidx(k, it, K); bxp[k][i] = st((double)bxp[k][i] + s_, c.bad); }
                }
                for (int q = 0; q < 3; ++q) sh[q][g] = baer[q];
            }
            __syncthreads();
            if (threadIdx.x < 3 * RSW_NBND) {             // fixed-order band sums (increasing g)
                int q = threadIdx.x / RSW_NBND, bnd = threadIdx.x % RSW_NBND;
                double s_ = 0.0; for (int gg = 0; gg < RSW_NGPT; ++gg) if (sband[gg] == bnd) s_ += sh[q][gg];
                real* dst = q == 0 ? b_ptaua : (q == 1 ? b_pasya : b_pomga);
                size_t i = (size_t)col * nl * RSW_NBND + (size_t)bnd * nl + K;
                float v = (float)((double)dst[i] + s_); if (!isfinite(v)) atomicExch(failure + 1, 1); dst[i] = v;
            }
            __syncthreads();
        }
        if (act) {                                        // surface: rup[klev] = palbp, rupd[klev] = palbd (both skies)
            double balbp = 0.0, balbd = 0.0;
            for (int s = 0; s < 2; ++s) { balbp += BrupC[s] + W(sp, S_RUPF(s), klev); balbd += BrupdC[s] + W(sp, S_RUPDF(s), klev); }
            sh[0][g] = balbp; sh[1][g] = balbd; sh[2][g] = badj;
        }
        __syncthreads();
        if (threadIdx.x < 3 * RSW_NBND) {
            int q = threadIdx.x / RSW_NBND, bnd = threadIdx.x % RSW_NBND;
            double s_ = 0.0; for (int gg = 0; gg < RSW_NGPT; ++gg) if (sband[gg] == bnd) s_ += sh[q][gg];
            real* dst = q == 0 ? b_palbp : (q == 1 ? b_palbd : b_adjflux);
            size_t i = (size_t)col * RSW_NBND + bnd;
            float v = (float)((double)dst[i] + s_); if (!isfinite(v)) atomicExch(failure + 1, 1); dst[i] = v;
        }
        __syncthreads();
        if (act) sh[0][g] = bmu;
        __syncthreads();
        if (threadIdx.x == 0) {
            double s_ = 0.0; for (int gg = 0; gg < RSW_NGPT; ++gg) s_ += sh[0][gg];
            float v = (float)((double)b_prmu0[col] + s_); if (!isfinite(v)) atomicExch(failure + 1, 1); b_prmu0[col] = v;
        }
        __syncthreads();
        if (act) {
            if (c.bad & BAD_PRIMAL) atomicExch(failure + 0, 1);
            if (c.bad & BAD_DERIV) atomicExch(failure + 1, 1);
            if (c.bad & BAD_TABLE) atomicExch(failure + 2, 1);
        }
    }
}
