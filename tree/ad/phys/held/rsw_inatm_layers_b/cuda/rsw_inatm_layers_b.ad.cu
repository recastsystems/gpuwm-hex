// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for rsw_inatm_layers_b, the batched RRTMG SW
// INATM layer kernel: hexcore/mpasphys/rrtmg_sw_v842.py:2909 (launch), CUDA source sha256 3c499c7b (capture object;
// body rsw_inatm_layers_body at line 3181, batched wrapper at line 3230; MPAS module_ra_rrtmg_sw.F:9652-10005).
// Written by hand from the forward CUDA source. Status before: missing (no derivative of any kind).
//
// Forward (one thread per (col, L), L = 0..nl-1, all f32 round-to-nearest, no FMA):
//   w1     = h2ovmr(L)
//   pdp(L) = p(L) - p(L+1)
//   amm    = (1-w1)*amd + w1*amw
//   cd     = ((p(L)-p(L+1))*1e3*avogad) / (((1e2*grav)*amm)*(1+w1))      -> coldry(L)
//   wkl(m,L) <- cd * v_m, v = (h2o, co2, o3, n2o, wkl(4,L) pre-launch [CO, not set here], ch4, o2), m = 0..6
//   wkl rows 7..37 are not touched (identity).
// So the map reads plev, the six vmr inputs and wkl row 4; it writes pdp, coldry and wkl rows 0..6.
//
// Precision (the glue-row rule): the f32 primal word cd the derivative needs is formed exactly as the forward forms it
// (__f*_rn, built --fmad=false --ftz=false; __check reproduces every forward output word bitwise from the same
// helper). Local partials are formed in double from that f32 word and the f32 inputs; every written derivative word
// is rounded to f32 once.
// Limiters: none. The forward has no clamp, min/max or state branch (only the launch-index guard), so no frozen-
// coefficient treatment is needed; the linearisation is the exact local derivative of the real map.
// Primal arguments are the pre-launch words (the window binds primal without `post` to the pre-launch copy), which
// is what wkl row 4 needs.
// In-place forms: TL d_wkl is inout (row 4's incoming tangent is read before the layer's rows are written; rows
// 0..3, 5, 6 are overwritten, rows 7..37 untouched). AD b_wkl is inout (consumed in place: post seed in, pre
// cotangent out: rows 0..3, 5, 6 -> 0, row 4 -> cd*seed, rows 7..37 untouched); b_pdp and b_coldry are consumed
// (zeroed); b_plev and the six vmr cotangents are accumulated.
// Determinism: TL is one thread per (col, L) as in the forward, each thread owning that layer's words. AD runs one
// thread per column (the first ncol threads of the forward geometry) marching the layers in order, so the two
// layers that touch each interface word p(L) are summed in a fixed order with no atomics.
// Refusal: any nonfinite input, partial or written word sets failure slot 0 (source rrtmg_sw_v842.py:2909).

#define RSI_MXMOL 38
#define RSI_NMOL 7
#define RSI_AMD 28.9660f
#define RSI_AMW 18.0160f
// step6-radiation (2026-11-07): molecule-unit storage scale, shared with rsw_setcoef_b and the LW pair (rationale in
// rlw_inatm.ad.cu). The outputs coldry and wkl rows 0..6 are molecules/cm2: their tangents are stored as t * 2^-72 and
// their cotangents arrive stored as b * 2^72. Exact (power of two, in double, before the one f32 rounding).
// Measured before the fix (window_ad_census s6r-c1, y-scale 1e-20): rsw_setcoef_b's b_coldry 73,015 of 76,839
// nonzero words subnormal, b_wkl 67,245 of 386,860.
#define RSI_MOL_T 0x1p-72
#define RSI_MOL_B 0x1p-72

// One layer's primal word cd (forward order, f32) and the local partials of cd (double).
struct RsiLay { float cd; double A, B; bool bad; };
__device__ __forceinline__ RsiLay rsi_lay(const float *pz, const float *h2o, int L, float grav, float avogad) {
  RsiLay k;
  float p0 = pz[L], p1 = pz[L + 1];
  float w1 = h2o[L];
  float amm = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, w1), RSI_AMD), __fmul_rn(w1, RSI_AMW));
  float num = __fmul_rn(__fmul_rn(__fsub_rn(p0, p1), 1.0e3f), avogad);
  float den = __fmul_rn(__fmul_rn(__fmul_rn(1.0e2f, grav), amm), __fadd_rn(1.0f, w1));
  k.cd = __fdiv_rn(num, den);
  double w = (double)w1, amd = (double)RSI_AMD, amw = (double)RSI_AMW;
  double ammd = (1.0 - w) * amd + w * amw;
  double dend = 1.0e2 * (double)grav * ammd * (1.0 + w);
  double numd = ((double)p0 - (double)p1) * 1.0e3 * (double)avogad;
  k.A = 1.0e3 * (double)avogad / dend;                                   // d cd / d p(L) = -d cd / d p(L+1)
  k.B = -(numd / dend) * ((amw - amd) / ammd + 1.0 / (1.0 + w));          // d cd / d w1
  k.bad = !isfinite(p0) || !isfinite(p1) || !isfinite(w1) || !isfinite(k.cd) || !isfinite(k.A) || !isfinite(k.B);
  return k;
}

// The six vmr inputs in wkl row order (row 4, CO, is not an input: it is wkl's own pre-launch word).
#define RSI_VMR_ARGS const float *h2o, const float *co2, const float *o3, const float *n2o, const float *ch4, const float *o2
__device__ __forceinline__ float rsi_v(int m, long long j, RSI_VMR_ARGS, const float *wkl, long long jw) {
  switch (m) {
    case 0: return h2o[j];
    case 1: return co2[j];
    case 2: return o3[j];
    case 3: return n2o[j];
    case 4: return wkl[jw + 4];
    case 5: return ch4[j];
    default: return o2[j];
  }
}

// Harness only (not in launch_abi): the forward output words from the shared helper, out of place.
extern "C" __global__ void rsw_inatm_layers_b__check(int ncol, int nl, float grav, float avogad, const float *plev,
    RSI_VMR_ARGS, const float *wkl, float *pdp, float *coldry, float *wkl_o, int *failure) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)ncol * nl) return;
  int col = (int)(idx / nl), L = (int)(idx % nl);
  const float *pz = plev + (long long)col * (nl + 1);
  long long o = (long long)col * nl, j = o + L, jw = (long long)col * RSI_MXMOL * nl + (long long)RSI_MXMOL * L;
  RsiLay k = rsi_lay(pz, h2o + o, L, grav, avogad);
  pdp[j] = __fsub_rn(pz[L], pz[L + 1]);
  coldry[j] = k.cd;
  for (int m = 0; m < RSI_NMOL; ++m) wkl_o[jw + m] = __fmul_rn(k.cd, rsi_v(m, j, h2o, co2, o3, n2o, ch4, o2, wkl, jw));
  if (k.bad) atomicExch(failure, 1);
}

extern "C" __global__ void rsw_inatm_layers_b__tl(int ncol, int nl, float grav, float avogad, const float *plev,
    RSI_VMR_ARGS, const float *wkl,
    const float *d_plev, const float *d_h2o, const float *d_co2, const float *d_o3, const float *d_n2o,
    const float *d_ch4, const float *d_o2, float *d_pdp, float *d_coldry, float *d_wkl, int *failure) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)ncol * nl) return;
  int col = (int)(idx / nl), L = (int)(idx % nl);
  const float *pz = plev + (long long)col * (nl + 1);
  const float *dpz = d_plev + (long long)col * (nl + 1);
  long long o = (long long)col * nl, j = o + L, jw = (long long)col * RSI_MXMOL * nl + (long long)RSI_MXMOL * L;
  RsiLay k = rsi_lay(pz, h2o + o, L, grav, avogad);
  bool bad = k.bad;
  double dv[RSI_NMOL];
  dv[0] = (double)d_h2o[j]; dv[1] = (double)d_co2[j]; dv[2] = (double)d_o3[j]; dv[3] = (double)d_n2o[j];
  dv[4] = (double)d_wkl[jw + 4];                                           // incoming, before any write
  dv[5] = (double)d_ch4[j]; dv[6] = (double)d_o2[j];
  double dp = (double)dpz[L] - (double)dpz[L + 1];
  double dcd = k.A * dp + k.B * dv[0];
  double cd = (double)k.cd;
  float t;
  t = (float)dp; d_pdp[j] = t; bad |= !isfinite(t);
  t = (float)(RSI_MOL_T * dcd); d_coldry[j] = t; bad |= !isfinite(t);
  for (int m = 0; m < RSI_NMOL; ++m) {
    double v = (double)rsi_v(m, j, h2o, co2, o3, n2o, ch4, o2, wkl, jw);
    t = (float)(RSI_MOL_T * (v * dcd + cd * dv[m])); d_wkl[jw + m] = t; bad |= !isfinite(t) || !isfinite(v);
  }
  if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rsw_inatm_layers_b__ad(int ncol, int nl, float grav, float avogad, const float *plev,
    RSI_VMR_ARGS, const float *wkl,
    float *b_plev, float *b_h2o, float *b_co2, float *b_o3, float *b_n2o, float *b_ch4, float *b_o2,
    float *b_pdp, float *b_coldry, float *b_wkl, int *failure) {
  long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // first ncol threads of the forward geometry
  if (col >= ncol) return;
  const float *pz = plev + col * (nl + 1);
  float *bpz = b_plev + col * (nl + 1);
  long long o = col * nl;
  bool bad = false;
  double carry = 0.0;                                   // layer L-1's contribution to p(L) (as its upper interface)
  for (int L = 0; L < nl; ++L) {
    long long j = o + L, jw = col * RSI_MXMOL * nl + (long long)RSI_MXMOL * L;
    RsiLay k = rsi_lay(pz, h2o + o, L, grav, avogad); bad |= k.bad;
    double cd = (double)k.cd;
    double ycd = RSI_MOL_B * (double)b_coldry[j], ypd = (double)b_pdp[j];
    double y[RSI_NMOL];
    double g = ycd;                                     // total cotangent of cd
    for (int m = 0; m < RSI_NMOL; ++m) {
      y[m] = RSI_MOL_B * (double)b_wkl[jw + m];
      g += (double)rsi_v(m, j, h2o, co2, o3, n2o, ch4, o2, wkl, jw) * y[m];
      bad |= !isfinite(y[m]);
    }
    bad |= !isfinite(ycd) || !isfinite(ypd);
    float t;
    t = (float)((double)b_h2o[j] + cd * y[0] + k.B * g); b_h2o[j] = t; bad |= !isfinite(t);
    t = (float)((double)b_co2[j] + cd * y[1]); b_co2[j] = t; bad |= !isfinite(t);
    t = (float)((double)b_o3[j]  + cd * y[2]); b_o3[j]  = t; bad |= !isfinite(t);
    t = (float)((double)b_n2o[j] + cd * y[3]); b_n2o[j] = t; bad |= !isfinite(t);
    t = (float)((double)b_ch4[j] + cd * y[5]); b_ch4[j] = t; bad |= !isfinite(t);
    t = (float)((double)b_o2[j]  + cd * y[6]); b_o2[j]  = t; bad |= !isfinite(t);
    for (int m = 0; m < RSI_NMOL; ++m) b_wkl[jw + m] = 0.0f;           // overwritten rows: no pre-launch cotangent
    t = (float)(cd * y[4]); b_wkl[jw + 4] = t; bad |= !isfinite(t);     // row 4 (CO) is read: cd * seed
    b_pdp[j] = 0.0f;
    b_coldry[j] = 0.0f;
    double q = k.A * g + ypd;                           // d/d p(L) of this layer; d/d p(L+1) is -q
    t = (float)((double)bpz[L] + (q + carry)); bpz[L] = t; bad |= !isfinite(t);
    carry = -q;
  }
  float t = (float)((double)bpz[nl] + carry); bpz[nl] = t;
  if (bad || !isfinite(t)) atomicExch(failure, 1);
}
