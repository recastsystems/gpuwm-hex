// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for rlw_inatm, the RRTMG LW INATM
// derived-quantity kernel: hexcore/mpasphys/rrtmg_lw_v842.py:728 (launch), CUDA source sha256 683d7a07
// (capture object, kernel at line 1644; MPAS module_ra_rrtmg_lw.F inatm). Written by hand from the forward
// CUDA source; replaces P7's generic taped route for binding (no per-thread tape, no 32 KB stack).
//
// Forward (one thread per column, serial layers l = 1..nl, all f32 __f*_rn):
//   amm   = (1-w1)*amd + w1*amw,  w1 = wkl(1,l) (h2o vmr, pre-launch)
//   cd    = (p(l-1)-p(l))*1e3*avogad / (1e2*grav*amm*(1+w1))           -> coldry(l)
//   S     = sum_{m=2..7} wkl(m,l)                                       (pre-launch vmr)
//   wbrodl(l) = cd*(1-S);  wkl(m,l) <- cd*wkl(m,l) (m=1..7);  wx(i,l) <- cd*wx(i,l)*1e-20 (i=1..4)
//   amttl += cd + wkl_out(1,l);  wvttl += wkl_out(1,l)
//   pwvcm = (amw*wvttl/(amd*amttl)) * 1e3*p(0) / (1e2*grav)
// Rows 8..38 of wkl are not touched (identity); coldry, wbrodl, pwvcm_v are full overwrites.
//
// Precision (the glue-row rule): the f32 primal words the derivative needs (cd, S, amttl, wvttl) are formed
// exactly as the forward forms them (__f*_rn, built --fmad=false --ftz=false; __check reproduces the forward
// output words bitwise from the same helper). Local partials are formed in double from those f32 words and the
// f32 inputs, and every written derivative word is rounded to f32 once.
// Limiters: none. The kernel has no clamp, max/min or state branch (only the fixed column bound), so no
// frozen-coefficient treatment is needed; the linearisation is the exact local derivative of the real map.
// Primal arguments are the pre-launch words (the window binds primal without `post` to the pre-launch copy).
// In-place forms: TL d_wkl / d_wx are inout (each layer reads its incoming tangents before writing); AD b_wkl /
// b_wx are inout (consumed in place: post seed in, pre cotangent out), b_coldry / b_wbrodl / b_pwvcm_v are
// consumed (zeroed), b_plev is accumulated. One thread owns one column's words, so no atomics: deterministic.
// Refusal: any nonfinite input, partial or written word sets failure slot 0 (source rrtmg_lw_v842.py:728).

#define RLI_MXMOL 38
#define RLI_MAXXSEC 4
#define RLI_AMD 28.9660f
#define RLI_AMW 18.0160f
#define RLI_E20 1.e-20f
// step6-radiation (2026-11-07): molecule-unit storage scale. The words this kernel writes in molecules/cm2 (coldry,
// wbrodl, wkl rows 1..7 after the launch) keep their window shadows in units of 2^72 molecules/cm2: a tangent is
// stored as t * 2^-72 and a cotangent as b * 2^72 (exact: a power of two, applied in double before the one f32
// rounding). Their readers (rlw_setcoef, the fs[coldry] slot of rlw_taumol_batched, and the plan's plain copy into
// that slot) use the same unit. THE BREAKAGE THIS PREVENTS: at the window's y-scale 1e-20 the true cotangents of
// these words are 1e-45..1e-29, so 95-99 % of them were f32 subnormal (window_ad_census s6r-c1: rlw_setcoef b_coldry
// 125,406 of 159,935 nonzero words subnormal, b_wbroad 158,428 of 159,921) and the pressure and vmr gradients
// through the column amounts lost their bits; at x-scale 1 their tangents reach 1e25 per word (G2's 1e56).
#define RLI_MOL_T 0x1p-72
#define RLI_MOL_B 0x1p-72

__device__ __forceinline__ long long rli_wk(int col, int m, int l, int nl) { return ((long long)col * RLI_MXMOL + (m - 1)) * nl + (l - 1); }
__device__ __forceinline__ long long rli_wx(int col, int m, int l, int nl) { return ((long long)col * RLI_MAXXSEC + (m - 1)) * nl + (l - 1); }

// One layer's primal words (forward order, f32) and local partials of cd (double).
struct RliLay { float cd, summol; double A, B; bool bad; };
__device__ __forceinline__ RliLay rli_lay(const float *pz, const float *wkl, int col, int l, int nl, float grav, float avogad) {
  RliLay k;
  float p0 = pz[l - 1], p1 = pz[l];
  float w1 = wkl[rli_wk(col, 1, l, nl)];
  float amm = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, w1), RLI_AMD), __fmul_rn(w1, RLI_AMW));
  float num = __fmul_rn(__fmul_rn(__fsub_rn(p0, p1), 1.e3f), avogad);
  float den = __fmul_rn(__fmul_rn(__fmul_rn(1.e2f, grav), amm), __fadd_rn(1.0f, w1));
  k.cd = __fdiv_rn(num, den);
  float s = 0.0f;
  for (int m = 2; m <= 7; ++m) s = __fadd_rn(s, wkl[rli_wk(col, m, l, nl)]);
  k.summol = s;
  double w = (double)w1, amd = (double)RLI_AMD, amw = (double)RLI_AMW;
  double ammd = (1.0 - w) * amd + w * amw;
  double dend = 1.e2 * (double)grav * ammd * (1.0 + w);
  double numd = ((double)p0 - (double)p1) * 1.e3 * (double)avogad;
  k.A = 1.e3 * (double)avogad / dend;                                   // d cd / d p(l-1) = -d cd / d p(l)
  k.B = -(numd / dend) * ((amw - amd) / ammd + 1.0 / (1.0 + w));         // d cd / d w1
  k.bad = !isfinite(p0) || !isfinite(p1) || !isfinite(w1) || !isfinite(k.cd) || !isfinite(k.A) || !isfinite(k.B);
  return k;
}

// Harness only (not in launch_abi): the forward output words from the shared helper, out of place.
extern "C" __global__ void rlw_inatm__check(int ncol, int nl, const float *plev, float grav, float avogad,
    const float *wkl, const float *wx, float *wkl_o, float *wx_o, float *coldry, float *wbrodl, float *pwvcm_v, int *failure) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncol) return;
  const float *pz = plev + (long long)col * (nl + 1);
  float amttl = 0.0f, wvttl = 0.0f; bool bad = false;
  for (int l = 1; l <= nl; ++l) {
    RliLay k = rli_lay(pz, wkl, col, l, nl, grav, avogad); bad |= k.bad;
    coldry[(long long)col * nl + (l - 1)] = k.cd;
    wbrodl[(long long)col * nl + (l - 1)] = __fmul_rn(k.cd, __fsub_rn(1.0f, k.summol));
    for (int m = 1; m <= 7; ++m) wkl_o[rli_wk(col, m, l, nl)] = __fmul_rn(k.cd, wkl[rli_wk(col, m, l, nl)]);
    float o1 = wkl_o[rli_wk(col, 1, l, nl)];
    amttl = __fadd_rn(__fadd_rn(amttl, k.cd), o1);
    wvttl = __fadd_rn(wvttl, o1);
    for (int i = 1; i <= RLI_MAXXSEC; ++i) wx_o[rli_wx(col, i, l, nl)] = __fmul_rn(__fmul_rn(k.cd, wx[rli_wx(col, i, l, nl)]), RLI_E20);
  }
  float wvsh = __fdiv_rn(__fmul_rn(RLI_AMW, wvttl), __fmul_rn(RLI_AMD, amttl));
  pwvcm_v[col] = __fdiv_rn(__fmul_rn(wvsh, __fmul_rn(1.e3f, pz[0])), __fmul_rn(1.e2f, grav));
  if (bad || !isfinite(pwvcm_v[col])) atomicExch(failure, 1);
}

extern "C" __global__ void rlw_inatm__tl(int ncol, int nl, const float *plev, float grav, float avogad,
    const float *wkl, const float *wx,
    const float *d_plev, float *d_wkl, float *d_wx, float *d_coldry, float *d_wbrodl, float *d_pwvcm_v, int *failure) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncol) return;
  const float *pz = plev + (long long)col * (nl + 1);
  const float *dpz = d_plev + (long long)col * (nl + 1);
  float amttl = 0.0f, wvttl = 0.0f;
  double dam = 0.0, dwv = 0.0;
  bool bad = false;
  for (int l = 1; l <= nl; ++l) {
    RliLay k = rli_lay(pz, wkl, col, l, nl, grav, avogad); bad |= k.bad;
    double dw[8];
    for (int m = 1; m <= 7; ++m) dw[m] = (double)d_wkl[rli_wk(col, m, l, nl)];      // incoming, before any write
    double dcd = k.A * ((double)dpz[l - 1] - (double)dpz[l]) + k.B * dw[1];
    double cd = (double)k.cd, dS = 0.0;
    for (int m = 2; m <= 7; ++m) dS += dw[m];
    float t;
    t = (float)(RLI_MOL_T * dcd); d_coldry[(long long)col * nl + (l - 1)] = t; bad |= !isfinite(t);
    t = (float)(RLI_MOL_T * ((1.0 - (double)k.summol) * dcd - cd * dS)); d_wbrodl[(long long)col * nl + (l - 1)] = t; bad |= !isfinite(t);
    double dout1 = 0.0;
    for (int m = 1; m <= 7; ++m) {
      double o = (double)wkl[rli_wk(col, m, l, nl)] * dcd + cd * dw[m];
      if (m == 1) dout1 = o;
      t = (float)(RLI_MOL_T * o); d_wkl[rli_wk(col, m, l, nl)] = t; bad |= !isfinite(t);
    }
    float o1 = __fmul_rn(k.cd, wkl[rli_wk(col, 1, l, nl)]);
    amttl = __fadd_rn(__fadd_rn(amttl, k.cd), o1);
    wvttl = __fadd_rn(wvttl, o1);
    dam += dcd + dout1;
    dwv += dout1;
    for (int i = 1; i <= RLI_MAXXSEC; ++i) {
      long long j = rli_wx(col, i, l, nl);
      double din = (double)d_wx[j];
      t = (float)((double)RLI_E20 * ((double)wx[j] * dcd + cd * din)); d_wx[j] = t; bad |= !isfinite(t);
    }
  }
  double am = (double)amttl, wv = (double)wvttl, p0 = (double)pz[0];
  double K = (double)RLI_AMW * 1.e3 / ((double)RLI_AMD * 1.e2 * (double)grav);
  double dpw = K * (p0 / am * dwv - wv * p0 / (am * am) * dam + wv / am * (double)dpz[0]);
  float t = (float)dpw; d_pwvcm_v[col] = t;
  if (bad || !isfinite(t)) atomicExch(failure, 1);
}

extern "C" __global__ void rlw_inatm__ad(int ncol, int nl, const float *plev, float grav, float avogad,
    const float *wkl, const float *wx,
    float *b_plev, float *b_wkl, float *b_wx, float *b_coldry, float *b_wbrodl, float *b_pwvcm_v, int *failure) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= ncol) return;
  const float *pz = plev + (long long)col * (nl + 1);
  float *bpz = b_plev + (long long)col * (nl + 1);
  bool bad = false;
  // Forward sweep for the primal totals amttl / wvttl (f32, forward order).
  float amttl = 0.0f, wvttl = 0.0f;
  for (int l = 1; l <= nl; ++l) {
    RliLay k = rli_lay(pz, wkl, col, l, nl, grav, avogad); bad |= k.bad;
    float o1 = __fmul_rn(k.cd, wkl[rli_wk(col, 1, l, nl)]);
    amttl = __fadd_rn(__fadd_rn(amttl, k.cd), o1);
    wvttl = __fadd_rn(wvttl, o1);
  }
  double am = (double)amttl, wv = (double)wvttl, p0 = (double)pz[0];
  double K = (double)RLI_AMW * 1.e3 / ((double)RLI_AMD * 1.e2 * (double)grav);
  double ypw = (double)b_pwvcm_v[col];
  double bwv = K * p0 / am * ypw;                       // cotangent of wvttl
  double bam = -K * wv * p0 / (am * am) * ypw;          // cotangent of amttl
  double bp0 = K * wv / am * ypw;                       // direct p(0) term of pwvcm
  b_pwvcm_v[col] = 0.0f;
  bad |= !isfinite(ypw);
  double carry = 0.0;                                   // contribution to p(l-1) from layer l-1 (as its p(l))
  for (int l = 1; l <= nl; ++l) {
    RliLay k = rli_lay(pz, wkl, col, l, nl, grav, avogad);
    double cd = (double)k.cd;
    long long jc = (long long)col * nl + (l - 1);
    double ycd = RLI_MOL_B * (double)b_coldry[jc], ywb = RLI_MOL_B * (double)b_wbrodl[jc];
    double yw[8], w[8];
    for (int m = 1; m <= 7; ++m) { yw[m] = RLI_MOL_B * (double)b_wkl[rli_wk(col, m, l, nl)]; w[m] = (double)wkl[rli_wk(col, m, l, nl)]; }
    double eff1 = yw[1] + bam + bwv;                    // wkl_out(1,l) feeds amttl and wvttl
    double g = ycd + (1.0 - (double)k.summol) * ywb + bam + w[1] * eff1;   // total cotangent of cd
    for (int m = 2; m <= 7; ++m) g += w[m] * yw[m];
    double yx[RLI_MAXXSEC + 1];
    for (int i = 1; i <= RLI_MAXXSEC; ++i) { long long j = rli_wx(col, i, l, nl); yx[i] = (double)b_wx[j]; g += (double)RLI_E20 * (double)wx[j] * yx[i]; }
    float t;
    t = (float)(cd * eff1 + k.B * g); b_wkl[rli_wk(col, 1, l, nl)] = t; bad |= !isfinite(t);
    for (int m = 2; m <= 7; ++m) { t = (float)(cd * yw[m] - cd * ywb); b_wkl[rli_wk(col, m, l, nl)] = t; bad |= !isfinite(t); }
    for (int i = 1; i <= RLI_MAXXSEC; ++i) { t = (float)((double)RLI_E20 * cd * yx[i]); b_wx[rli_wx(col, i, l, nl)] = t; bad |= !isfinite(t); }
    b_coldry[jc] = 0.0f;
    b_wbrodl[jc] = 0.0f;
    double ag = k.A * g;
    double v = carry + ag + (l == 1 ? bp0 : 0.0);
    t = (float)((double)bpz[l - 1] + v); bpz[l - 1] = t; bad |= !isfinite(t) || !isfinite(ycd) || !isfinite(ywb);
    carry = -ag;
  }
  float t = (float)((double)bpz[nl] + carry); bpz[nl] = t;
  if (bad || !isfinite(t)) atomicExch(failure, 1);
}
