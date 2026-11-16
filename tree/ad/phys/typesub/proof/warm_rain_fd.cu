// SPDX-License-Identifier: Apache-2.0
// P2 independent check (private autodiff program): finite-difference referee for the generated
// TEMPO warm_rain tangent.  Built in one translation unit with the generated file plus `tshp`,
// a copy of the generated tangent namespace with TF = TD = double (the same statements and the
// same float-rounded table constants, evaluated in double), so centred differences resolve the
// derivative of the real function to ~1e-9 instead of float noise.  tsfd writes warm_rain_hp.gen.cu.
#include "warm_rain_hp.gen.cu"

struct WrIn {
  float rhof, rc, nc, mvd_c, rr, nr, mvd_r;
  double ilamc, pnr_rcr, prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw;
  int l_qc, l_qr;
};
static_assert(sizeof(WrIn) == 96, "call layout");

__device__ __forceinline__ TempoCtx fd_ctx(float inv) {
  TempoCtx c;
  c.global_dt = 0.0f; c.global_inverse_dt = inv;
  c.aerosolaware_flag = 0; c.hailaware_flag = 0; c.semi_sedi_flag = 0; c.refl10cm_from_melting_flag = 0;
  c.turn_off_micro_flag = 0; c.cloud_condensation_flag = 0; c.refl10cm_flag = 0; c.re_cloud_flag = 0;
  c.re_ice_flag = 0; c.re_snow_flag = 0; c.max_hail_diameter_flag = 0; c.first_call_main = 0; c.ml_for_nc_flag = 0;
  return c;
}
__device__ __forceinline__ double in_k(const WrIn& a, int k) {
  switch (k) {
    case 0: return a.rhof; case 1: return a.rc; case 2: return a.nc; case 3: return a.mvd_c; case 4: return a.rr;
    case 5: return a.nr; case 6: return a.mvd_r; case 7: return a.ilamc; case 8: return a.pnr_rcr; case 9: return a.prr_wau;
    case 10: return a.pnr_wau; case 11: return a.pnc_wau; case 12: return a.prr_rcw; default: return a.pnc_rcw;
  }
}

// Unit-tangent Jacobian column: thread (i, k) -> J[(i*14+k)*6 + j] = d out_j / d in_k.
extern "C" __global__ void wr_jtl(int n, const WrIn* x, const TempoBlob* L, float inv, double* J) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)n * 14) return;
  const int i = (int)(t / 14), kk = (int)(t % 14);
  using tsad_tl::TF; using tsad_tl::TD;
  const WrIn a = x[i];
#define D(q) (kk == (q) ? 1.0 : 0.0)
  TF rhof[2], rc[2], nc[2], mvd_c[2], rr[2], nr[2], mvd_r[2]; TD ilamc[2]; bool lqc[2], lqr[2];
  rhof[1] = TF(a.rhof, D(0), 0); rc[1] = TF(a.rc, D(1), 0); nc[1] = TF(a.nc, D(2), 0); mvd_c[1] = TF(a.mvd_c, D(3), 0);
  rr[1] = TF(a.rr, D(4), 0); nr[1] = TF(a.nr, D(5), 0); mvd_r[1] = TF(a.mvd_r, D(6), 0); ilamc[1] = TD(a.ilamc, D(7), 0);
  lqc[1] = a.l_qc != 0; lqr[1] = a.l_qr != 0;
  tsad_tl::TempoTend T;
  T.pnr_rcr[1] = TD(a.pnr_rcr, D(8), 0); T.prr_wau[1] = TD(a.prr_wau, D(9), 0); T.pnr_wau[1] = TD(a.pnr_wau, D(10), 0);
  T.pnc_wau[1] = TD(a.pnc_wau, D(11), 0); T.prr_rcw[1] = TD(a.prr_rcw, D(12), 0); T.pnc_rcw[1] = TD(a.pnc_rcw, D(13), 0);
#undef D
  TempoParams P; P.L = L;
  tsad_tl::warm_rain(P, fd_ctx(inv), 1, rhof, lqc, rc, nc, ilamc, mvd_c, lqr, rr, nr, mvd_r, T);
  double* o = J + t * 6;
  o[0] = T.pnr_rcr[1].d; o[1] = T.prr_wau[1].d; o[2] = T.pnr_wau[1].d; o[3] = T.pnc_wau[1].d; o[4] = T.prr_rcw[1].d; o[5] = T.pnc_rcw[1].d;
}

__device__ void hp_eval(const WrIn& a, int kk, double dlt, const TempoBlob* L, float inv, double* y) {
  double v[14];
  for (int q = 0; q < 14; ++q) v[q] = in_k(a, q) + (q == kk ? dlt : 0.0);
  double rhof[2], rc[2], nc[2], mvd_c[2], rr[2], nr[2], mvd_r[2], ilamc[2]; bool lqc[2], lqr[2];
  rhof[1] = v[0]; rc[1] = v[1]; nc[1] = v[2]; mvd_c[1] = v[3]; rr[1] = v[4]; nr[1] = v[5]; mvd_r[1] = v[6]; ilamc[1] = v[7];
  lqc[1] = a.l_qc != 0; lqr[1] = a.l_qr != 0;
  tshp::TempoTend T;
  T.pnr_rcr[1] = v[8]; T.prr_wau[1] = v[9]; T.pnr_wau[1] = v[10]; T.pnc_wau[1] = v[11]; T.prr_rcw[1] = v[12]; T.pnc_rcw[1] = v[13];
  TempoParams P; P.L = L;
  tshp::warm_rain(P, fd_ctx(inv), 1, rhof, lqc, rc, nc, ilamc, mvd_c, lqr, rr, nr, mvd_r, T);
  y[0] = T.pnr_rcr[1]; y[1] = T.prr_wau[1]; y[2] = T.pnr_wau[1]; y[3] = T.pnc_wau[1]; y[4] = T.prr_rcw[1]; y[5] = T.pnc_rcw[1];
}

// Centred differences of the double-promoted copy at steps h and h/4 (h = rel*|x_k|, or rel*scale[k]
// when x_k == 0): F1/F2[(i*14+k)*6+j]; Y0[i*6+j] = promoted primal at the call.  rel is read from
// memory because the launcher's argument type carries no f64.
extern "C" __global__ void wr_fdhp_m(int n, const WrIn* x, const TempoBlob* L, float inv, const double* rel, const double* scale,
                                     double* F1, double* F2, double* Y0) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)n * 14) return;
  const int i = (int)(t / 14), kk = (int)(t % 14);
  const WrIn a = x[i];
  const double xv = in_k(a, kk);
  const double h = rel[0] * (xv != 0.0 ? fabs(xv) : scale[kk]);
  double yp[6], ym[6], yq[6], yr[6];
  hp_eval(a, kk, h, L, inv, yp); hp_eval(a, kk, -h, L, inv, ym);
  hp_eval(a, kk, 0.25 * h, L, inv, yq); hp_eval(a, kk, -0.25 * h, L, inv, yr);
  for (int j = 0; j < 6; ++j) { F1[t * 6 + j] = (yp[j] - ym[j]) / (2.0 * h); F2[t * 6 + j] = (yq[j] - yr[j]) / (0.5 * h); }
  if (kk == 0) { double y0[6]; hp_eval(a, -1, 0.0, L, inv, y0); for (int j = 0; j < 6; ++j) Y0[i * 6 + j] = y0[j]; }
}

// Original captured function (warp shape, call at level 1+lane), primal only.
extern "C" __global__ void wr_orig(int n, const WrIn* x, const TempoBlob* L, float inv, ::TempoTend* Tb, double* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int K = 1 + (int)(threadIdx.x & 31u);
  float rhof[34], rc[34], nc[34], mvd_c[34], rr[34], nr[34], mvd_r[34];
  double ilamc[34]; bool lqc[34], lqr[34];
  const WrIn a = x[i];
  rhof[K] = a.rhof; rc[K] = a.rc; nc[K] = a.nc; mvd_c[K] = a.mvd_c; rr[K] = a.rr; nr[K] = a.nr; mvd_r[K] = a.mvd_r;
  ilamc[K] = a.ilamc; lqc[K] = a.l_qc != 0; lqr[K] = a.l_qr != 0;
  ::TempoTend& T = Tb[i];
  T.pnr_rcr[K] = a.pnr_rcr; T.prr_wau[K] = a.prr_wau; T.pnr_wau[K] = a.pnr_wau;
  T.pnc_wau[K] = a.pnc_wau; T.prr_rcw[K] = a.prr_rcw; T.pnc_rcw[K] = a.pnc_rcw;
  TempoParams P; P.L = L;
  ::warm_rain(P, fd_ctx(inv), K, rhof, lqc, rc, nc, ilamc, mvd_c, lqr, rr, nr, mvd_r, T);
  y[i * 6 + 0] = T.pnr_rcr[K]; y[i * 6 + 1] = T.prr_wau[K]; y[i * 6 + 2] = T.pnr_wau[K];
  y[i * 6 + 3] = T.pnc_wau[K]; y[i * 6 + 4] = T.prr_rcw[K]; y[i * 6 + 5] = T.pnc_rcw[K];
}
extern "C" __global__ void fd_sizes(unsigned long long* o) { o[0] = sizeof(::TempoTend); }
