// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for rsw_sfluxzen_b, the batched RRTMG SW
// solar-source kernel (sfluxzen per g-point): hexcore/mpasphys/rrtmg_sw_v842.py:1910 (launch), CUDA source sha256
// 3c499c7b (capture object; body rsw_sfluxzen_body at line 2087, rsw_spec at line 1580, batched wrapper at line 2231;
// MPAS module_ra_rrtmg_sw.F:3201-4666). Written by hand from the forward CUDA source. Status before: missing.
//
// Forward (one thread per (col, iw), iw = 0..111; all f32 round-to-nearest, no FMA):
//   jb = ngb[iw], lsf = laysolfr[col, jb-16], L = lsf-1
//   constant bands 16, 20, 23, 25, 26, 27, 29: v = tab word (27: scalekur * tab word); 16/27/29 give 0 when lsf <= laytrop
//   spectral bands 17, 18, 19, 21, 22, 24, 28 (17 and 28 give 0 when lsf <= laytrop):
//     sc = A + r*B ; sp = min(A/sc, oneminus) ; sm = mult*sp ; js = 1 + (int)sm ; fs = sm - trunc(sm)
//     v  = a + fs*(b - a),  a = sf[ig + rows*(js-1)], b = sf[ig + rows*js]   (sf = the band's sfluxref table)
//   with (A, B, r, mult) = 17 (h2o, co2, strrat17, 4), 18 (h2o, ch4, strrat18, 8), 19 (h2o, co2, strrat19, 8),
//   21 (h2o, co2, strrat21, 8), 22 (h2o, o2, 1.6*strrat22, 8), 24 (h2o, o2, strrat24, 8), 28 (o3, o2, strrat28, 4).
//   sfluxzen[col, iw] = v on every path (full overwrite).
// So the map reads colh2o, colco2, colch4, colo2, colo3 at the band's solar-source layer only. fac00..fac11, jp, jt
// and jt1 are passed to the forward's rsw_spec call, but sfluxzen does not depend on them (the f000..f111 weights are
// dead in this kernel), and tab is a constant table.
//
// Linearisation (frozen-coefficient treatment, documented):
//   * js (the integer bin of sm) and the branch lsf <= laytrop are frozen from the trajectory. v is continuous across
//     a bin edge (a piecewise-linear interpolation: at fs -> 1 in bin js, v -> sf[js] = the next bin's a), so freezing
//     js gives the one-sided slope of the bin the trajectory is in; it is exact everywhere except on the edge itself.
//   * The clamp sp = min(A/sc, oneminus) is a limiter: when the trajectory is clamped (sp >= oneminus, i.e. B == 0 or
//     A/(A+rB) rounds to 1 in f32) the coefficient is frozen at its clamped value and the tangent of sp is 0.
//   * Otherwise dv = (b - a) * mult * dsp, dsp = r*(B*dA - A*dB)/(A + r*B)^2.
// Precision (the glue-row rule): the f32 primal words the derivative needs (sp, its clamp flag, js, a, b) are formed
// exactly as the forward forms them (__f*_rn, built --fmad=false --ftz=false; the harness-only __check reproduces
// every forward output word bitwise from the same helper). Partials are formed in double from those f32 words; every
// written derivative word is rounded to f32 once.
// In-place forms: TL d_sfluxzen is out (full overwrite). AD b_sfluxzen is inout (seed consumed: zeroed, since the
// forward overwrites every word); b_col* are accumulated.
// Determinism: TL is one thread per (col, iw) as in the forward. AD runs one thread per column (the first ncol threads
// of the forward geometry) walking iw = 0..111 in order. Every g-point of a band, and every band whose solar-source
// layer and gas coincide, adds to the same (gas, layer) word, so contributions are summed in double into a per-column
// list of at most 7 x 2 (gas, layer) slots in first-touch order and each word is written once: no atomics.
// Refusal: any nonfinite input, partial or written word, sc == 0 (A = B = 0: the forward divides 0/0), or a bin
// outside 1..mult (a negative column amount) sets failure slot 0 (source rrtmg_sw_v842.py:1910).

#define RSZ_NGPT 112
#define RSZ_NBND 14
// Constants copied bit-exact from the capture source (3c499c7b lines 5-104).
#define RSZ_T16 4998
#define RSZ_T17 26292
#define RSZ_T18 33016
#define RSZ_T19 39752
#define RSZ_T20 42964
#define RSZ_T21 60724
#define RSZ_T22 62480
#define RSZ_T23 63278
#define RSZ_T24 69962
#define RSZ_T25 70520
#define RSZ_T26 70544
#define RSZ_T27 72956
#define RSZ_T28 83532
#define RSZ_T29 87330
#define RSZ_K17 __uint_as_float(1052422714u)
#define RSZ_K18 __uint_as_float(1109120490u)
#define RSZ_K19 __uint_as_float(1085261081u)
#define RSZ_K21 __uint_as_float(999588355u)
#define RSZ_K22 __uint_as_float(1018824225u)
#define RSZ_K24 __uint_as_float(1040146053u)
#define RSZ_K27_SCALEKUR __uint_as_float(1065661914u)
#define RSZ_K28 __uint_as_float(892538338u)
#define RSZ_ONEMINUS __uint_as_float(1065353199u)

// gas ids: 0 h2o, 1 co2, 2 ch4, 3 o2, 4 o3
#define RSZ_COLS const float *colh2o, const float *colco2, const float *colch4, const float *colo2, const float *colo3
__device__ __forceinline__ const float *rsz_gas(int g, RSZ_COLS) {
  switch (g) { case 0: return colh2o; case 1: return colco2; case 2: return colch4; case 3: return colo2; default: return colo3; }
}

// One g-point: the forward word v and, for a live spectral point, the layer, the two gases and dv/dA, dv/dB.
struct RszPt { float v; int L, gA, gB; double cA, cB; bool live, bad; };

__device__ __forceinline__ RszPt rsz_pt(int iw, int laytrop, const float *tab, const int *ngb, const int *lsfc,
                                        long long o, RSZ_COLS) {
  RszPt q; q.v = 0.0f; q.L = -1; q.gA = q.gB = -1; q.cA = q.cB = 0.0; q.live = false; q.bad = false;
  int jb = ngb[iw];
  int lsf = lsfc[jb - 16];
  int L = lsf - 1;
  int gA = 0, gB = 0, rows = 0, ig = 0, toff = 0; float r = 0.0f, mult = 0.0f;
  switch (jb) {
    case 16: if (lsf <= laytrop) return q; q.v = tab[RSZ_T16 + iw]; return q;
    case 17: if (lsf <= laytrop) return q; gA = 0; gB = 1; r = RSZ_K17; mult = 4.0f; rows = 12; ig = iw - 6; toff = RSZ_T17; break;
    case 18: gA = 0; gB = 2; r = RSZ_K18; mult = 8.0f; rows = 8; ig = iw - 18; toff = RSZ_T18; break;
    case 19: gA = 0; gB = 1; r = RSZ_K19; mult = 8.0f; rows = 8; ig = iw - 26; toff = RSZ_T19; break;
    case 20: q.v = tab[RSZ_T20 + iw - 34]; return q;
    case 21: gA = 0; gB = 1; r = RSZ_K21; mult = 8.0f; rows = 10; ig = iw - 44; toff = RSZ_T21; break;
    case 22: gA = 0; gB = 3; r = __fmul_rn(1.6f, RSZ_K22); mult = 8.0f; rows = 2; ig = iw - 54; toff = RSZ_T22; break;
    case 23: q.v = tab[RSZ_T23 + iw - 56]; return q;
    case 24: gA = 0; gB = 3; r = RSZ_K24; mult = 8.0f; rows = 8; ig = iw - 66; toff = RSZ_T24; break;
    case 25: q.v = tab[RSZ_T25 + iw - 74]; return q;
    case 26: q.v = tab[RSZ_T26 + iw - 80]; return q;
    case 27: if (lsf <= laytrop) return q; q.v = __fmul_rn(RSZ_K27_SCALEKUR, tab[RSZ_T27 + iw - 86]); return q;
    case 28: if (lsf <= laytrop) return q; gA = 4; gB = 3; r = RSZ_K28; mult = 4.0f; rows = 6; ig = iw - 94; toff = RSZ_T28; break;
    default: if (lsf <= laytrop) return q; q.v = tab[RSZ_T29 + iw - 100]; return q;   // 29
  }
  float A = rsz_gas(gA, colh2o, colco2, colch4, colo2, colo3)[o + L];
  float B = rsz_gas(gB, colh2o, colco2, colch4, colo2, colo3)[o + L];
  float sc = __fadd_rn(A, __fmul_rn(r, B));                    // forward rsw_spec, capture source 1584
  float sp = __fdiv_rn(A, sc);                                  // 1585
  bool clamp = sp >= RSZ_ONEMINUS;                              // 1586 (limiter)
  if (clamp) sp = RSZ_ONEMINUS;
  float sm = __fmul_rn(mult, sp);                               // 1587
  int js = 1 + (int)sm;                                         // 1588 (frozen bin)
  float fs = __fsub_rn(sm, truncf(sm));                         // 1589
  bool inbin = js >= 1 && js <= (int)mult;
  if (!inbin) js = 1;                                           // refused below; keeps the table read in range
  const float *sf = tab + toff;
  float a = sf[ig + rows * (js - 1)], b = sf[ig + rows * js];   // body 2119, 2128, 2137, 2149, 2159, 2171, 2191
  q.v = __fadd_rn(a, __fmul_rn(fs, __fsub_rn(b, a)));
  double sd = (double)A + (double)r * (double)B;
  double k = clamp ? 0.0 : ((double)b - (double)a) * (double)mult * (double)r / (sd * sd);
  q.cA = k * (double)B;
  q.cB = -k * (double)A;
  q.L = L; q.gA = gA; q.gB = gB; q.live = true;
  q.bad = !isfinite(A) || !isfinite(B) || sd == 0.0 || !isfinite(sd) || !isfinite(q.cA) || !isfinite(q.cB) || !isfinite(q.v)
          || !inbin;
  return q;
}

// Harness only (not in launch_abi): the forward output words from the shared helper, out of place.
extern "C" __global__ void rsw_sfluxzen_b__check(int ncol, int nl, const int *laytrop, const float *tab, const int *ngb,
    const int *laysolfr, RSZ_COLS, float *sfluxzen_o, int *failure) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)ncol * RSZ_NGPT) return;
  int col = (int)(idx / RSZ_NGPT), iw = (int)(idx % RSZ_NGPT);
  RszPt q = rsz_pt(iw, laytrop[col], tab, ngb, laysolfr + (long long)col * RSZ_NBND, (long long)col * nl,
                   colh2o, colco2, colch4, colo2, colo3);
  sfluxzen_o[idx] = q.v;
  if (q.bad) atomicExch(failure, 1);
}

extern "C" __global__ void rsw_sfluxzen_b__tl(int ncol, int nl, const int *laytrop, const float *tab, const int *ngb,
    const int *laysolfr, RSZ_COLS,
    const float *d_colh2o, const float *d_colco2, const float *d_colch4, const float *d_colo2, const float *d_colo3,
    float *d_sfluxzen, int *failure) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)ncol * RSZ_NGPT) return;
  int col = (int)(idx / RSZ_NGPT), iw = (int)(idx % RSZ_NGPT);
  long long o = (long long)col * nl;
  RszPt q = rsz_pt(iw, laytrop[col], tab, ngb, laysolfr + (long long)col * RSZ_NBND, o, colh2o, colco2, colch4, colo2, colo3);
  bool bad = q.bad;
  double dv = 0.0;
  if (q.live) {
    double dA = (double)rsz_gas(q.gA, d_colh2o, d_colco2, d_colch4, d_colo2, d_colo3)[o + q.L];
    double dB = (double)rsz_gas(q.gB, d_colh2o, d_colco2, d_colch4, d_colo2, d_colo3)[o + q.L];
    dv = q.cA * dA + q.cB * dB;
    bad |= !isfinite(dA) || !isfinite(dB);
  }
  float t = (float)dv;
  d_sfluxzen[idx] = t;
  if (bad || !isfinite(t)) atomicExch(failure, 1);
}

extern "C" __global__ void rsw_sfluxzen_b__ad(int ncol, int nl, const int *laytrop, const float *tab, const int *ngb,
    const int *laysolfr, RSZ_COLS,
    float *b_colh2o, float *b_colco2, float *b_colch4, float *b_colo2, float *b_colo3,
    float *b_sfluxzen, int *failure) {
  long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // first ncol threads of the forward geometry
  if (col >= ncol) return;
  long long o = col * nl, os = col * RSZ_NGPT;
  const int *lsfc = laysolfr + col * RSZ_NBND;
  int lt = laytrop[col];
  int sg[RSZ_NBND], sl[RSZ_NBND]; double sa[RSZ_NBND]; int ns = 0;   // 7 spectral bands x 2 gases
  bool bad = false;
  for (int iw = 0; iw < RSZ_NGPT; ++iw) {
    RszPt q = rsz_pt(iw, lt, tab, ngb, lsfc, o, colh2o, colco2, colch4, colo2, colo3);
    double y = (double)b_sfluxzen[os + iw];
    bad |= q.bad || !isfinite(y);
    b_sfluxzen[os + iw] = 0.0f;                                    // forward overwrites every word: seed consumed
    if (!q.live) continue;
    for (int s = 0; s < 2; ++s) {
      int g = s ? q.gB : q.gA; double c = (s ? q.cB : q.cA) * y;
      int k = 0;
      while (k < ns && !(sg[k] == g && sl[k] == q.L)) ++k;
      if (k == ns) {
        if (ns == RSZ_NBND) { bad = true; continue; }              // unreachable with the 7 spectral bands
        sg[ns] = g; sl[ns] = q.L; sa[ns] = 0.0; ++ns;
      }
      sa[k] += c;
    }
  }
  float *bc[5] = {b_colh2o, b_colco2, b_colch4, b_colo2, b_colo3};
  for (int k = 0; k < ns; ++k) {
    float *w = bc[sg[k]] + o + sl[k];
    float t = (float)((double)*w + sa[k]);
    *w = t;
    bad |= !isfinite(t);
  }
  if (bad) atomicExch(failure, 1);
}
