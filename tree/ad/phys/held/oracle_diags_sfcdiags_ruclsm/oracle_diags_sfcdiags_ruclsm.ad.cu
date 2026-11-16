// SPDX-License-Identifier: Apache-2.0
// Private autodiff program (ad/held-oracle_diags_sfcdiags_ruclsm). Whole-launch tangent-linear and adjoint entries of
// hex 16251589 oracle_diags_sfcdiags_ruclsm (tree/src/hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5322, device body
// diags_sfcdiags_ruclsm at :4761, saturation helpers diags_rslf :4880 and diags_rsif :4902; RRFS-MPAS 1d6e370e
// RUCLSM/MPAS/module_sf_sfcdiags_ruclsm.F:9-243), written by hand from the forward CUDA source.
//
// Forward per cell (flux = .true. is a constant of the source, L54, so the L72 and L113 arms are dead), Lnnn = Fortran line:
//   E = pow(1e5/psfc, rovcp), F = pow(1e-5*psfc, rovcp)                                   L66/L70/L74/L160
//   th2a = chs2 < 1e-5 ? t3*E : tsk*E - hfx/(rho*cp*chs2);  t2a = th2a*F                  L64-L70
//   qsat1 = (t3-273.15 <= 0 ? rsif : rslf)(p1, t3);  ql = min(qsat1, qv)                  L79-L87
//   qp = ql + qfx/(rho*cqs);  mr = qsfc/(1-qsfc)                                          L91-L92
//   q2a = cqs2 < 1e-5 ? ql : qp - qfx/(rho*cqs2)                                          L104-L109
//   dt = t3-tsk, dq = ql-mr, dz1 = dz/2;  fh = min(max(1-dt/10 | 1-dq/0.003, 0.01), 1)     L123-L138
//   fac = log(2.05/(0.05+fh)) / log((dz1+0.05)/(0.05+fh))                                 L130/L140
//   t2b = dt > 0 ? tsk + fac*(t3-tsk) : t2a;  q2b = dq > 0 ? mr + fac*(ql-mr) : q2a       L126-L146
//   t2 = clamp(t2b, min(tsk,t3), max(tsk,t3)); th2 = t2*E                                  L158-L160
//   q2c = clamp(q2b, min(mr,ql), max(mr,ql)); q2 = min((t2-273.15<=0 ? rsif : rslf)(psfc, t2), q2c)   L163-L174
// Outputs t2, th2, q2 are full overwrites (no output word is read before it is written). Inputs are read-only.
//
// Precision (the glue-row rule): the primal is recomputed in float32 with the forward's own explicitly rounded words
// (__f*_rn, and gfk_pow / gfk_log from the module's shared math, ruc_shared_math.cuh), so every branch and selection
// is the forward's; the local partials are formed in float64 from those float32 words; each written derivative word
// is rounded once to float32. Built with --fmad=false --ftz=false (as the live schemes).
//
// Branches and limiter-like selections (frozen-coefficient treatment, documented):
//   S1 chs2 < 1e-5 (L64), S2 cqs2 < 1e-5 (L104), S3 tempc <= 0 ice/liquid (L80, L167), S4 dt > 0 (L126), S5 dq > 0 (L137):
//      the branch is taken from the trajectory and its arm is differentiated exactly.
//   C1 ql = min(qsat1, qv) (L87), C2 the fh clamp to [0.01, 1] (L128, L138), C3 the t2 clamp to [min,max](tsk,t3)
//   (L158-159), C4 the q2 clamp to [min,max](mr,ql) (L163-164), C5 q2 = min(qsat2, q2c) (L174), C6 x = max(-80, t-273.16)
//   in rslf/rsif (L205, L239): every min/max is a selection; the selector is frozen from the trajectory (ruc_min/ruc_max
//   tie rules: min(a,b) = a<b ? a : b, max(a,b) = a>b ? a : b) and the tangent passes only through the selected operand;
//   a selected constant bound passes zero. This is the forward's exact derivative wherever it exists (away from ties).
// Inactive: cp, r_d, rovcp (per-cell physical constants of the pointer table), globalcells and the 18 int extents,
// the tables block (only tb.targetcell, an int, is read), the arena work (t2_alt/q2_alt: written then read within
// one thread of this launch), and chs (read only by the dead flux = .false. arm L72).
//
// Pointer table (ruc_layout.rs DIAGNOSTICS, 39 slots, no optional flags): 0 hfx 1 qfx 2 tsk 3 qsfc 4 cqs 5 cqs2 6 chs
// 7 chs2 | 8 t2 9 th2 10 q2 (outputs) | 11 t3d 12 qv3d 13 dz 14 rho3d 15 p3d (3-D, level 1 read) 16 psfc2d | 17 cp 18 r_d
// 19 rovcp | 20 globalcells | 21-38 ids..kte. The derivative table is the dispatcher's mirror of the same table
// (indirect.mirror): word k points at the shadow of forward pointer k, one shadow word per 4-byte word, so the
// forward's FArray indexing applies unchanged.
//
// Failure slots: 0 a nonfinite primal intermediate, Jacobian entry, incoming tangent or cotangent;
//                1 an output array overlaps an input array or another output (the in-place entries would be wrong).
// AD_GAIN_PROBE (diagnostic image only, never the bound image): the TL entry also reduces ||x|| over the active input
// words, ||TL x|| per output, branch counts and range margins, and compares its recomputed primal with the forward's
// written t2/th2/q2 words bit for bit; one ADGAIN line per TL launch.

#include "ruc_shared_math.cuh"
#ifndef AD_GAIN_PROBE
#define AD_GAIN_PROBE 0
#endif
#if AD_GAIN_PROBE
#include <cstdio>
#endif

namespace osd {
typedef unsigned long long u64;
enum { HFX = 0, QFX, TSK, QSFC, CQS, CQS2, CHS, CHS2, T2, TH2, Q2, T3D, QV3D, DZ, RHO3D, P3D, PSFC2D, CP, R_D, ROVCP,
       GLOBALCELLS, IDS, IDE, JDS, JDE, KDS, KDE, IMS, IME, JMS, JME, KMS, KME, ITS, ITE, JTS, JTE, KTS, KTE, NPTR };
// active inputs, Jacobian column order
enum { IHFX = 0, IQFX, ITSK, IQSFC, ICQS, ICQS2, ICHS2, IT3, IQV, IDZ, IRHO, IP1, IPSFC, NI };
__device__ const int SLOT[NI] = {HFX, QFX, TSK, QSFC, CQS, CQS2, CHS2, T3D, QV3D, DZ, RHO3D, P3D, PSFC2D};
__device__ const bool IS3D[NI] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0};

template<class T> __device__ __forceinline__ T rmax(T a, T b) { return a > b ? a : b; }   // ruc_max
template<class T> __device__ __forceinline__ T rmin(T a, T b) { return a < b ? a : b; }   // ruc_min

struct Ext { int ims, ime, jms, jme, kms, kme, its, ite, jts, jte; };
__device__ __forceinline__ Ext ext(const u64 *p, int c) {
  Ext e;
  e.ims = ((const int *)p[IMS])[c]; e.ime = ((const int *)p[IME])[c]; e.jms = ((const int *)p[JMS])[c]; e.jme = ((const int *)p[JME])[c];
  e.kms = ((const int *)p[KMS])[c]; e.kme = ((const int *)p[KME])[c]; e.its = ((const int *)p[ITS])[c]; e.ite = ((const int *)p[ITE])[c];
  e.jts = ((const int *)p[JTS])[c]; e.jte = ((const int *)p[JTE])[c];
  return e;
}
// word index of element (i,j) of a 2-D array / (i,1,j) of a 3-D array: the forward's FArray arithmetic (int)
__device__ __forceinline__ int w2(const Ext &e, int nc, int c, int i, int j) { return c + (i - e.ims) * nc + (j - e.jms) * (nc * (e.ime - e.ims + 1)); }
__device__ __forceinline__ int w3(const Ext &e, int nc, int c, int i, int j) {
  return c + (i - e.ims) * nc + (1 - e.kms) * (nc * (e.ime - e.ims + 1)) + (j - e.jms) * (nc * (e.ime - e.ims + 1) * (e.kme - e.kms + 1));
}

struct V { double g[NI]; };
__device__ __forceinline__ V vz() { V r; for (int k = 0; k < NI; ++k) r.g[k] = 0.0; return r; }
__device__ __forceinline__ V ve(int k) { V r = vz(); r.g[k] = 1.0; return r; }
__device__ __forceinline__ V operator+(const V &a, const V &b) { V r; for (int k = 0; k < NI; ++k) r.g[k] = a.g[k] + b.g[k]; return r; }
__device__ __forceinline__ V operator-(const V &a, const V &b) { V r; for (int k = 0; k < NI; ++k) r.g[k] = a.g[k] - b.g[k]; return r; }
__device__ __forceinline__ V operator*(double s, const V &a) { V r; for (int k = 0; k < NI; ++k) r.g[k] = s * a.g[k]; return r; }

// diags_rslf (L205-L208) / diags_rsif (L239-L241), the forward's words, plus d r/d p and d r/d t in float64.
__device__ float sat(bool ice, float p, float t, double &rp, double &rt, bool &xclamped, double &margin) {
  float c[9];
  if (ice) { c[0] = .609868993e03f; c[1] = .499320233e02f; c[2] = .184672631e01f; c[3] = .402737184e-1f; c[4] = .565392987e-3f;
             c[5] = .521693933e-5f; c[6] = .307839583e-7f; c[7] = .105785160e-9f; c[8] = .161444444e-12f; }
  else     { c[0] = .611583699e03f; c[1] = .444606896e02f; c[2] = .143177157e01f; c[3] = .264224321e-1f; c[4] = .299291081e-3f;
             c[5] = .203154182e-5f; c[6] = .702620698e-8f; c[7] = .379534310e-11f; c[8] = (-.321582393e-13f); }
  const float x0 = __fsub_rn(t, 273.16f);
  const float x = rmax<float>((-80.f), x0);                       // C6: selector frozen
  xclamped = ((-80.f) > x0);
  const float es = __fadd_rn(c[0], __fmul_rn(x, __fadd_rn(c[1], __fmul_rn(x, __fadd_rn(c[2], __fmul_rn(x, __fadd_rn(c[3], __fmul_rn(x,
                   __fadd_rn(c[4], __fmul_rn(x, __fadd_rn(c[5], __fmul_rn(x, __fadd_rn(c[6], __fmul_rn(x, __fadd_rn(c[7], __fmul_rn(x, c[8]))))))))))))))));
  const float den = __fsub_rn(p, es);
  const float r = __fdiv_rn(__fmul_rn(.622f, es), den);
  double d = 8.0 * (double)c[8];                                   // d es / d x, Horner on the same coefficients
  for (int k = 7; k >= 1; --k) d = d * (double)x + (double)k * (double)c[k];
  const double D = (double)den, A = (double).622f;
  rp = -A * (double)es / (D * D);
  const double res = A * ((double)den + (double)es) / (D * D);     // d r / d es
  rt = xclamped ? 0.0 : res * d;
  margin = D;
  return r;
}

struct Stats { unsigned int chs2_small, cqs2_small, ice1, ice2, dt_pos, dq_pos, ql_sat, fh_t_clamp, fh_q_clamp, t_clamp, q_clamp, q_sat2, x80;
               float min_den_sat, max_abs_fac, min_abs_L2, max_inv_rhocqs2; };

// The 13 active input words of element (i,j) of cell c (Jacobian column order) and the two constants, from a table.
__device__ __forceinline__ void load(const u64 *p, const Ext &e, int nc, int c, int i, int j, float v[NI], float &cp, float &rovcp) {
  const int a2 = w2(e, nc, c, i, j), a3 = w3(e, nc, c, i, j);
  for (int k = 0; k < NI; ++k) v[k] = ((const float *)p[SLOT[k]])[IS3D[k] ? a3 : a2];
  cp = ((const float *)p[CP])[c]; rovcp = ((const float *)p[ROVCP])[c];
}

// The Jacobian of (t2, th2, q2) with respect to the NI active inputs v at one element. Returns the recomputed float32
// outputs in o[3] (the forward's words when the branches are reproduced: checked bit for bit by the probe image).
__device__ bool jac(const float v[NI], float cp, float rovcp, V &Gt2, V &Gth2, V &Gq2, float o[3], Stats *st) {
  const float hfx = v[IHFX], qfx = v[IQFX], tsk = v[ITSK], qsfc = v[IQSFC], cqs = v[ICQS], cqs2 = v[ICQS2], chs2 = v[ICHS2],
              t3 = v[IT3], qv = v[IQV], dzr = v[IDZ], rho = v[IRHO], p1 = v[IP1], psfc = v[IPSFC];
  const V Ghfx = ve(IHFX), Gqfx = ve(IQFX), Gtsk = ve(ITSK), Gqsfc = ve(IQSFC), Gcqs = ve(ICQS), Gcqs2 = ve(ICQS2), Gchs2 = ve(ICHS2),
          Gt3 = ve(IT3), Gqv = ve(IQV), Gdz = ve(IDZ), Grho = ve(IRHO), Gp1 = ve(IP1), Gps = ve(IPSFC);
  // E and F (L66/L70/L74/L160): d pow(x, y)/dx = y pow / x
  const float b1 = __fdiv_rn(1.e5f, psfc);
  const float E = gfk_pow(b1, rovcp);
  const V GE = (-(double)rovcp * (double)E / (double)psfc) * Gps;
  const float b2 = __fmul_rn(1.e-5f, psfc);
  const float F = gfk_pow(b2, rovcp);
  const V GF = ((double)rovcp * (double)F / (double)b2 * (double)1.e-5f) * Gps;
  // S1 (L64-L68)
  float th2a; V Gth2a;
  if ((chs2 < 1.e-5f)) {
    th2a = __fmul_rn(t3, E); Gth2a = (double)E * Gt3 + (double)t3 * GE;
    if (st) st->chs2_small++;
  } else {
    const float A = __fmul_rn(tsk, E), B = __fmul_rn(rho, cp), C = __fmul_rn(B, chs2), Dh = __fdiv_rn(hfx, C);
    th2a = __fsub_rn(A, Dh);
    const V GA = (double)E * Gtsk + (double)tsk * GE;
    const V GC = ((double)cp * (double)chs2) * Grho + (double)B * Gchs2;
    const V GD = (1.0 / (double)C) * Ghfx - ((double)Dh / (double)C) * GC;
    Gth2a = GA - GD;
  }
  const float t2a = __fmul_rn(th2a, F);                                       // L70
  const V Gt2a = (double)F * Gth2a + (double)th2a * GF;
  // L77-L87: qsat at level 1, C1
  double sp, stt, mg; bool xc;
  const float tempc = __fsub_rn(t3, 273.15f);
  const bool ice1 = (tempc <= 0.0f);
  const float qsat1 = sat(ice1, p1, t3, sp, stt, xc, mg);
  if (st) { st->ice1 += ice1; st->x80 += xc; st->min_den_sat = fminf(st->min_den_sat, (float)mg); }
  const V Gqs1 = sp * Gp1 + stt * Gt3;
  float ql = qv; V Gql = Gqv;
  if (qsat1 < ql) { ql = qsat1; Gql = Gqs1; if (st) st->ql_sat++; }        // ruc_min(qsat, qlev1)
  // L91-L92
  const float R = __fmul_rn(rho, cqs), Q = __fdiv_rn(qfx, R), qp = __fadd_rn(ql, Q);
  const V GR = (double)cqs * Grho + (double)rho * Gcqs;
  const V Gqp = Gql + ((1.0 / (double)R) * Gqfx - ((double)Q / (double)R) * GR);
  const float om = __fsub_rn(1.f, qsfc), mr = __fdiv_rn(qsfc, om);
  const V Gmr = (1.0 / ((double)om * (double)om)) * Gqsfc;
  // S2 (L104-L109)
  float q2a; V Gq2a;
  if ((cqs2 < 1.e-5f)) { q2a = ql; Gq2a = Gql; if (st) st->cqs2_small++; }
  else {
    const float R2 = __fmul_rn(rho, cqs2), Q2 = __fdiv_rn(qfx, R2);
    q2a = __fsub_rn(qp, Q2);
    const V GR2 = (double)cqs2 * Grho + (double)rho * Gcqs2;
    Gq2a = Gqp - ((1.0 / (double)R2) * Gqfx - ((double)Q2 / (double)R2) * GR2);
    if (st) st->max_inv_rhocqs2 = fmaxf(st->max_inv_rhocqs2, fabsf(__fdiv_rn(1.f, R2)));
  }
  // L123-L125
  const float dt = __fsub_rn(t3, tsk), dq = __fsub_rn(ql, mr), dz1 = __fmul_rn(0.5f, dzr);
  const V Gdt = Gt3 - Gtsk, Gdq = Gql - Gmr, Gdz1 = 0.5 * Gdz;
  float t2b; V Gt2b;
  if ((dt > 0.f)) {                                                            // S4, L126-L132
    const float u = __fsub_rn(1.f, __fdiv_rn(dt, 10.f));
    V Gfh = (-1.0 / (double)10.f) * Gdt;
    const float f1 = rmax<float>(u, 0.01f); if (!(u > 0.01f)) Gfh = vz();      // C2
    const float fh = rmin<float>(f1, 1.0f); if (!(f1 < 1.0f)) Gfh = vz();
    if (st && !((u > 0.01f) && (f1 < 1.0f))) st->fh_t_clamp++;
    const float d1 = __fadd_rn(0.05f, fh), dzp = __fadd_rn(dz1, .05f);
    const float L1 = gfk_log(__fdiv_rn(__fadd_rn(2.f, .05f), d1)), L2 = gfk_log(__fdiv_rn(dzp, d1));
    const float fac = __fdiv_rn(L1, L2);
    const V GL1 = (-1.0 / (double)d1) * Gfh;
    const V GL2 = (1.0 / (double)dzp) * Gdz1 - (1.0 / (double)d1) * Gfh;
    const V Gfac = (1.0 / (double)L2) * GL1 - ((double)fac / (double)L2) * GL2;
    t2b = __fadd_rn(tsk, __fmul_rn(fac, __fsub_rn(t3, tsk)));
    Gt2b = Gtsk + ((double)dt * Gfac + (double)fac * Gdt);
    if (st) { st->dt_pos++; st->max_abs_fac = fmaxf(st->max_abs_fac, fabsf(fac)); st->min_abs_L2 = fminf(st->min_abs_L2, fabsf(L2)); }
  } else { t2b = t2a; Gt2b = Gt2a; }
  float q2b; V Gq2b;
  if ((dq > 0.f)) {                                                            // S5, L137-L142
    const float u = __fsub_rn(1.f, __fdiv_rn(dq, 0.003f));
    V Gfh = (-1.0 / (double)0.003f) * Gdq;
    const float f1 = rmax<float>(u, 0.01f); if (!(u > 0.01f)) Gfh = vz();      // C2
    const float fh = rmin<float>(f1, 1.0f); if (!(f1 < 1.0f)) Gfh = vz();
    if (st && !((u > 0.01f) && (f1 < 1.0f))) st->fh_q_clamp++;
    const float d1 = __fadd_rn(0.05f, fh), dzp = __fadd_rn(dz1, .05f);
    const float L1 = gfk_log(__fdiv_rn(__fadd_rn(2.f, .05f), d1)), L2 = gfk_log(__fdiv_rn(dzp, d1));
    const float fac = __fdiv_rn(L1, L2);
    const V GL1 = (-1.0 / (double)d1) * Gfh;
    const V GL2 = (1.0 / (double)dzp) * Gdz1 - (1.0 / (double)d1) * Gfh;
    const V Gfac = (1.0 / (double)L2) * GL1 - ((double)fac / (double)L2) * GL2;
    q2b = __fadd_rn(mr, __fmul_rn(fac, __fsub_rn(ql, mr)));
    Gq2b = Gmr + ((double)dq * Gfac + (double)fac * Gdq);
    if (st) { st->dq_pos++; st->max_abs_fac = fmaxf(st->max_abs_fac, fabsf(fac)); st->min_abs_L2 = fminf(st->min_abs_L2, fabsf(L2)); }
  } else { q2b = q2a; Gq2b = Gq2a; }
  // C3 (L158-L160)
  const float lo = rmin<float>(tsk, t3); const V Glo = (tsk < t3) ? Gtsk : Gt3;
  const float x = rmax<float>(lo, t2b);  const V Gx = (lo > t2b) ? Glo : Gt2b;
  const float hi = rmax<float>(tsk, t3); const V Ghi = (tsk > t3) ? Gtsk : Gt3;
  const float t2 = rmin<float>(hi, x);   Gt2 = (hi < x) ? Ghi : Gx;
  if (st && t2 != t2b) st->t_clamp++;
  const float th2 = __fmul_rn(t2, E);
  Gth2 = (double)E * Gt2 + (double)t2 * GE;
  // C4 (L163-L164)
  const float qlo = rmin<float>(mr, ql); const V Gqlo = (mr < ql) ? Gmr : Gql;
  const float qx = rmax<float>(qlo, q2b); const V Gqx = (qlo > q2b) ? Gqlo : Gq2b;
  const float qhi = rmax<float>(mr, ql); const V Gqhi = (mr > ql) ? Gmr : Gql;
  const float q2c = rmin<float>(qhi, qx); const V Gq2c = (qhi < qx) ? Gqhi : Gqx;
  if (st && q2c != q2b) st->q_clamp++;
  // L166-L174, C5
  const float tempc2 = __fsub_rn(t2, 273.15f);
  const bool ice2 = (tempc2 <= 0.0f);
  double sp2, st2, mg2; bool xc2;
  const float qsat2 = sat(ice2, psfc, t2, sp2, st2, xc2, mg2);
  if (st) { st->ice2 += ice2; st->x80 += xc2; st->min_den_sat = fminf(st->min_den_sat, (float)mg2); }
  float q2 = q2c; Gq2 = Gq2c;
  if (qsat2 < q2c) { q2 = qsat2; Gq2 = sp2 * Gps + st2 * Gt2; if (st) st->q_sat2++; }
  o[0] = t2; o[1] = th2; o[2] = q2;
  bool ok = isfinite(t2) && isfinite(th2) && isfinite(q2);
  for (int k = 0; k < NI; ++k) ok = ok && isfinite(Gt2.g[k]) && isfinite(Gth2.g[k]) && isfinite(Gq2.g[k]);
  return ok;
}

// [lo, hi) byte ranges the launch reads (inputs, level 1 of the 3-D arrays) and writes (outputs).
__device__ bool overlaps(const u64 *p, const Ext &e, int nc) {
  const long long s2 = (long long)nc * (e.ime - e.ims + 1), nj = e.jme - e.jms + 1, s3 = s2 * (e.kme - e.kms + 1);
  u64 lo[NI + 3], hi[NI + 3];
  for (int k = 0; k < NI; ++k) {
    const u64 b = p[SLOT[k]];
    if (IS3D[k]) { lo[k] = b + 4ull * (u64)((1 - e.kms) * s2); hi[k] = lo[k] + 4ull * (u64)((nj - 1) * s3 + s2); }
    else { lo[k] = b; hi[k] = b + 4ull * (u64)(s2 * nj); }
  }
  const int outs[3] = {T2, TH2, Q2};
  for (int o = 0; o < 3; ++o) { lo[NI + o] = p[outs[o]]; hi[NI + o] = p[outs[o]] + 4ull * (u64)(s2 * nj); }
  // outputs: disjoint from everything; inputs: disjoint or the very same view (then one thread owns each word, and
  // the adjoint's two accumulations into it are sequential in that thread)
  for (int o = 0; o < NI + 3; ++o)
    for (int k = o + 1; k < NI + 3; ++k) {
      if (!(lo[o] < hi[k] && lo[k] < hi[o])) continue;
      if (o < NI && k < NI && lo[o] == lo[k] && hi[o] == hi[k] && IS3D[o] == IS3D[k]) continue;
      return true;
    }
  return false;
}

#if AD_GAIN_PROBE
__device__ double g_x2, g_t2[3], g_col[NI * 3];   // g_col[3k+o]: sum over cells of (d out_o / d in_k)^2
__device__ float g_cmax[NI];                       // max over cells and outputs of |d out / d in_k|
// central-difference Taylor check on the real inputs: direction d_k = x_k |v_k| (relative, from the dot's tangent x),
// h = 1e-2, 1e-3, 1e-4. Each cell is classed by whether any branch or selection (the 13 Stats flags) differs between
// x+hd, x and x-hd: "flip" cells straddle a kink of the forward, where no derivative describes the step; "smooth"
// cells do not. Sums per h (index 3h+o, o = t2, th2, q2): g_fd* = sum (FD - J d_eff)^2, g_jd* = sum (J d_eff)^2.
// g_cnt[2h] = flip cells, g_cnt[2h+1] = smooth cells whose error exceeds 1e-2 |J d| + 16 ulp(out)/h (none if the TL is right);
// g_flag[13h+f] = flip cells by flag.
__device__ double g_fds[9], g_jds[9], g_fdf[9], g_jdf[9];
__device__ unsigned int g_cnt[6], g_flag[39];
__device__ unsigned int g_n[13], g_mis[3], g_cells, g_blocks;
__device__ float g_f[4] = {3.4e38f, 0.f, 3.4e38f, 0.f};
#endif
}  // namespace osd

extern "C" __global__ void oracle_diags_sfcdiags_ruclsm__tl(const unsigned long long *ptrs, const unsigned long long *dptrs,
                                                           int nc, int *failures) {
  using namespace osd;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
#if AD_GAIN_PROBE
  Stats S = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.4e38f, 0.f, 3.4e38f, 0.f};
  double x2 = 0.0, t2s[3] = {0.0, 0.0, 0.0};
  unsigned int mis[3] = {0, 0, 0}, cells = 0;
  Stats *sp = &S;
#else
  Stats *sp = nullptr;
#endif
  if (c < nc) {
    const Ext e = ext(ptrs, c);
    if (c == 0 && overlaps(ptrs, e, nc)) atomicOr(&failures[1], 1);
    for (int j = e.jts; j <= e.jte; ++j) for (int i = e.its; i <= e.ite; ++i) {
      V Gt2, Gth2, Gq2; float o[3], v[NI], cp, rovcp;
      load(ptrs, e, nc, c, i, j, v, cp, rovcp);
      bool ok = jac(v, cp, rovcp, Gt2, Gth2, Gq2, o, sp);
      const int a2 = w2(e, nc, c, i, j), a3 = w3(e, nc, c, i, j);
      double x[NI];
      // every incoming tangent is read before any output word is written
      for (int k = 0; k < NI; ++k) { x[k] = (double)((const float *)dptrs[SLOT[k]])[IS3D[k] ? a3 : a2]; ok = ok && isfinite(x[k]); }
      double y0 = 0.0, y1 = 0.0, y2 = 0.0;
      for (int k = 0; k < NI; ++k) { y0 += Gt2.g[k] * x[k]; y1 += Gth2.g[k] * x[k]; y2 += Gq2.g[k] * x[k]; }
      if (!ok) atomicOr(&failures[0], 1);
      ((float *)dptrs[T2])[a2] = __double2float_rn(y0);
      ((float *)dptrs[TH2])[a2] = __double2float_rn(y1);
      ((float *)dptrs[Q2])[a2] = __double2float_rn(y2);
#if AD_GAIN_PROBE
      for (int k = 0; k < NI; ++k) x2 += x[k] * x[k];
      const float r0 = __double2float_rn(y0), r1 = __double2float_rn(y1), r2 = __double2float_rn(y2);
      t2s[0] += (double)r0 * r0; t2s[1] += (double)r1 * r1; t2s[2] += (double)r2 * r2;
      const float *P = (const float *)ptrs[T2], *PT = (const float *)ptrs[TH2], *PQ = (const float *)ptrs[Q2];
      mis[0] += __float_as_uint(P[a2]) != __float_as_uint(o[0]);
      mis[1] += __float_as_uint(PT[a2]) != __float_as_uint(o[1]);
      mis[2] += __float_as_uint(PQ[a2]) != __float_as_uint(o[2]);
      cells += 1;
      bool dir = false; for (int k = 0; k < NI; ++k) dir = dir || x[k] != 0.0;
      if (dir) {
        const double H[3] = {1e-2, 1e-3, 1e-4};
        Stats z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.4e38f, 0.f, 3.4e38f, 0.f};
        Stats s0 = z; { V a, b, cc; float oo[3]; jac(v, cp, rovcp, a, b, cc, oo, &s0); }
        const unsigned int *f0 = &s0.chs2_small;
        for (int h = 0; h < 3; ++h) {
          float vp[NI], vm[NI], op[3], om[3]; double de[NI];
          for (int k = 0; k < NI; ++k) {
            const double dk = x[k] * fabs((double)v[k]);
            vp[k] = __double2float_rn((double)v[k] + H[h] * dk); vm[k] = __double2float_rn((double)v[k] - H[h] * dk);
            de[k] = ((double)vp[k] - (double)vm[k]) / (2.0 * H[h]);
          }
          Stats sP = z, sM = z;
          { V a, b, cc; jac(vp, cp, rovcp, a, b, cc, op, &sP); jac(vm, cp, rovcp, a, b, cc, om, &sM); }
          const unsigned int *fp = &sP.chs2_small, *fm = &sM.chs2_small;
          bool flip = false;
          for (int f = 0; f < 13; ++f) if (fp[f] != f0[f] || fm[f] != f0[f]) { flip = true; atomicAdd(&g_flag[13 * h + f], 1u); }
          bool bad = false;
          for (int q = 0; q < 3; ++q) {
            const V &G = q == 0 ? Gt2 : (q == 1 ? Gth2 : Gq2);
            double jd = 0.0; for (int k = 0; k < NI; ++k) jd += G.g[k] * de[k];
            const double fd = ((double)op[q] - (double)om[q]) / (2.0 * H[h]);
            const double e2 = (fd - jd) * (fd - jd);
            if (flip) { atomicAdd(&g_fdf[3 * h + q], e2); atomicAdd(&g_jdf[3 * h + q], jd * jd); }
            else      { atomicAdd(&g_fds[3 * h + q], e2); atomicAdd(&g_jds[3 * h + q], jd * jd); }
            bad = bad || fabs(fd - jd) > 1e-2 * fabs(jd) + 16.0 * 1.1920929e-7 * fabs((double)o[q]) / H[h];
          }
          if (flip) atomicAdd(&g_cnt[2 * h], 1u); else if (bad) atomicAdd(&g_cnt[2 * h + 1], 1u);
        }
      }
      for (int k = 0; k < NI; ++k) {
        atomicAdd(&g_col[3 * k], Gt2.g[k] * Gt2.g[k]); atomicAdd(&g_col[3 * k + 1], Gth2.g[k] * Gth2.g[k]); atomicAdd(&g_col[3 * k + 2], Gq2.g[k] * Gq2.g[k]);
        const float m = fmaxf(fabsf((float)Gt2.g[k]), fmaxf(fabsf((float)Gth2.g[k]), fabsf((float)Gq2.g[k])));
        atomicMax((int *)&g_cmax[k], __float_as_int(m));
      }
#endif
    }
  }
#if AD_GAIN_PROBE
  if (x2 != 0.0) atomicAdd(&g_x2, x2);
  for (int k = 0; k < 3; ++k) { if (t2s[k] != 0.0) atomicAdd(&g_t2[k], t2s[k]); if (mis[k]) atomicAdd(&g_mis[k], mis[k]); }
  if (cells) {
    const unsigned int n[13] = {S.chs2_small, S.cqs2_small, S.ice1, S.ice2, S.dt_pos, S.dq_pos, S.ql_sat, S.fh_t_clamp, S.fh_q_clamp,
                                S.t_clamp, S.q_clamp, S.q_sat2, S.x80};
    for (int k = 0; k < 13; ++k) if (n[k]) atomicAdd(&g_n[k], n[k]);
    atomicAdd(&g_cells, cells);
    // nonnegative floats order as their int bits
    atomicMin((int *)&g_f[0], __float_as_int(fmaxf(S.min_den_sat, 0.f))); atomicMax((int *)&g_f[1], __float_as_int(S.max_abs_fac));
    atomicMin((int *)&g_f[2], __float_as_int(S.min_abs_L2)); atomicMax((int *)&g_f[3], __float_as_int(S.max_inv_rhocqs2));
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int done = atomicAdd(&g_blocks, 1u) + 1u;
    if (done == gridDim.x * gridDim.y * gridDim.z) {
      __threadfence();
      printf("ADGAIN {\"x2\":%.17g,\"t2\":[%.17g,%.17g,%.17g],\"mismatch_t2_th2_q2\":[%u,%u,%u],\"cells\":%u,"
             "\"n\":{\"chs2_small\":%u,\"cqs2_small\":%u,\"ice1\":%u,\"ice2\":%u,\"dt_pos\":%u,\"dq_pos\":%u,\"ql_sat\":%u,"
             "\"fh_t_clamp\":%u,\"fh_q_clamp\":%u,\"t_clamp\":%u,\"q_clamp\":%u,\"q_sat2\":%u,\"x_m80\":%u},"
             "\"min_p_minus_es\":%.9g,\"max_abs_fac\":%.9g,\"min_abs_log_dz\":%.9g,\"max_inv_rho_cqs2\":%.9g}\n",
             g_x2, g_t2[0], g_t2[1], g_t2[2], g_mis[0], g_mis[1], g_mis[2], g_cells,
             g_n[0], g_n[1], g_n[2], g_n[3], g_n[4], g_n[5], g_n[6], g_n[7], g_n[8], g_n[9], g_n[10], g_n[11], g_n[12],
             g_f[0], g_f[1], g_f[2], g_f[3]);
      printf("ADGAINCOL {\"order\":\"hfx qfx tsk qsfc cqs cqs2 chs2 t3d qv3d dz rho3d p3d psfc2d\",\"sumsq_t2_th2_q2\":[");
      for (int k = 0; k < NI; ++k) printf("%s[%.9g,%.9g,%.9g]", k ? "," : "", g_col[3 * k], g_col[3 * k + 1], g_col[3 * k + 2]);
      printf("],\"max_abs\":[");
      for (int k = 0; k < NI; ++k) printf("%s%.9g", k ? "," : "", g_cmax[k]);
      printf("]}\n");
      printf("ADTAYLOR {\"h\":[1e-2,1e-3,1e-4],\"smooth_err_sumsq\":[");
      for (int k = 0; k < 9; ++k) printf("%s%.9g", k ? "," : "", g_fds[k]);
      printf("],\"smooth_jd_sumsq\":["); for (int k = 0; k < 9; ++k) printf("%s%.9g", k ? "," : "", g_jds[k]);
      printf("],\"flip_err_sumsq\":["); for (int k = 0; k < 9; ++k) printf("%s%.9g", k ? "," : "", g_fdf[k]);
      printf("],\"flip_jd_sumsq\":["); for (int k = 0; k < 9; ++k) printf("%s%.9g", k ? "," : "", g_jdf[k]);
      printf("],\"flip_cells_smooth_bad_cells\":[%u,%u,%u,%u,%u,%u],\"flip_by_flag\":[", g_cnt[0], g_cnt[1], g_cnt[2], g_cnt[3], g_cnt[4], g_cnt[5]);
      for (int k = 0; k < 39; ++k) printf("%s%u", k ? "," : "", g_flag[k]);
      printf("],\"flags\":\"chs2_small cqs2_small ice1 ice2 dt_pos dq_pos ql_sat fh_t_clamp fh_q_clamp t_clamp q_clamp q_sat2 x_m80\"}\n");
      for (int k = 0; k < 9; ++k) { g_fds[k] = 0.0; g_jds[k] = 0.0; g_fdf[k] = 0.0; g_jdf[k] = 0.0; }
      for (int k = 0; k < 6; ++k) g_cnt[k] = 0u;
      for (int k = 0; k < 39; ++k) g_flag[k] = 0u;
      for (int k = 0; k < NI * 3; ++k) g_col[k] = 0.0;
      for (int k = 0; k < NI; ++k) g_cmax[k] = 0.f;
      g_x2 = 0.0; for (int k = 0; k < 3; ++k) { g_t2[k] = 0.0; g_mis[k] = 0u; }
      for (int k = 0; k < 13; ++k) g_n[k] = 0u;
      g_f[0] = 3.4e38f; g_f[1] = 0.f; g_f[2] = 3.4e38f; g_f[3] = 0.f; g_cells = 0u; g_blocks = 0u;
    }
  }
#endif
}

extern "C" __global__ void oracle_diags_sfcdiags_ruclsm__ad(const unsigned long long *ptrs, const unsigned long long *bptrs,
                                                           int nc, int *failures) {
  using namespace osd;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nc) return;
  const Ext e = ext(ptrs, c);
  if (c == 0 && overlaps(ptrs, e, nc)) atomicOr(&failures[1], 1);
  for (int j = e.jte; j >= e.jts; --j) for (int i = e.ite; i >= e.its; --i) {
    V Gt2, Gth2, Gq2; float o[3], v[NI], cp, rovcp;
    load(ptrs, e, nc, c, i, j, v, cp, rovcp);
    bool ok = jac(v, cp, rovcp, Gt2, Gth2, Gq2, o, nullptr);
    const int a2 = w2(e, nc, c, i, j), a3 = w3(e, nc, c, i, j);
    float *bt2 = (float *)bptrs[T2], *bth2 = (float *)bptrs[TH2], *bq2 = (float *)bptrs[Q2];
    // the forward overwrote every output word: consume its cotangent and clear it
    const double y0 = bt2[a2], y1 = bth2[a2], y2 = bq2[a2];
    bt2[a2] = 0.f; bth2[a2] = 0.f; bq2[a2] = 0.f;
    ok = ok && isfinite(y0) && isfinite(y1) && isfinite(y2);
    if (!ok) atomicOr(&failures[0], 1);
    for (int k = 0; k < NI; ++k) {
      float *b = (float *)bptrs[SLOT[k]] + (IS3D[k] ? a3 : a2);
      *b = __double2float_rn((double)*b + (Gt2.g[k] * y0 + Gth2.g[k] * y1 + Gq2.g[k] * y2));
    }
  }
}
