// SPDX-License-Identifier: Apache-2.0
// P2 proof entries for TEMPO warm_rain (private autodiff program).
// Built in one translation unit with the generated file, so the original warm_rain (captured
// text, captured options) and the two generated copies see the same headers and tables.
#include "warm_rain.gen.cu"

// Per-level call record, layout of the 11-02 warm-rain lane's captured calls (96 bytes).
struct WrIn {
  float rhof, rc, nc, mvd_c, rr, nr, mvd_r;
  double ilamc, pnr_rcr, prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw;
  int l_qc, l_qr;
};
// Direction / input cotangent in the hand pair's shadow layout (mvd_r's shadow is double).
struct WrGrad {
  float rhof, rc, nc, mvd_c, rr, nr;
  double mvd_r;
  double ilamc, pnr_rcr, prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw;
  int l_qc, l_qr;
};
// Generated adjoint result, all double.
struct WrBarD { double rhof, rc, nc, mvd_c, rr, nr, mvd_r, ilamc, t[6]; };
struct Out6 { double v[6]; };
static_assert(sizeof(WrIn) == 96 && sizeof(WrGrad) == 96, "call layouts");

__device__ __forceinline__ TempoCtx wr_ctx(float inv) {
  TempoCtx c;
  c.global_dt = 0.0f; c.global_inverse_dt = inv;
  c.aerosolaware_flag = 0; c.hailaware_flag = 0; c.semi_sedi_flag = 0; c.refl10cm_from_melting_flag = 0;
  c.turn_off_micro_flag = 0; c.cloud_condensation_flag = 0; c.refl10cm_flag = 0; c.re_cloud_flag = 0;
  c.re_ice_flag = 0; c.re_snow_flag = 0; c.max_hail_diameter_flag = 0; c.first_call_main = 0; c.ml_for_nc_flag = 0;
  return c;
}

#define WR_TEND(T, K, F) F(T.pnr_rcr[K], 0) F(T.prr_wau[K], 1) F(T.pnr_wau[K], 2) F(T.pnc_wau[K], 3) F(T.prr_rcw[K], 4) F(T.pnc_rcw[K], 5)

// Original, captured warp shape: lane L of the warp owns level 1+L, so the call sits there.
extern "C" __global__ void wr_orig(int n, const WrIn* x, const TempoBlob* L, float inv, ::TempoTend* Tb, Out6* y) {
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
  ::warm_rain(P, wr_ctx(inv), K, rhof, lqc, rc, nc, ilamc, mvd_c, lqr, rr, nr, mvd_r, T);
#define WR_OUT(e, j) y[i].v[j] = e;
  WR_TEND(T, K, WR_OUT)
#undef WR_OUT
}

// Generated tangent: one thread per call, serial shape, level 1.
extern "C" __global__ void wr_tl(int n, const WrIn* x, const WrGrad* dx, const TempoBlob* L, float inv,
                                 tsad_tl::TempoTend* Tb, Out6* y, Out6* dy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  using tsad_tl::TF; using tsad_tl::TD;
  const WrIn a = x[i]; const WrGrad g = dx[i];
  TF rhof[2], rc[2], nc[2], mvd_c[2], rr[2], nr[2], mvd_r[2]; TD ilamc[2]; bool lqc[2], lqr[2];
  rhof[1] = TF(a.rhof, (double)g.rhof, 0); rc[1] = TF(a.rc, (double)g.rc, 0); nc[1] = TF(a.nc, (double)g.nc, 0);
  mvd_c[1] = TF(a.mvd_c, (double)g.mvd_c, 0); rr[1] = TF(a.rr, (double)g.rr, 0); nr[1] = TF(a.nr, (double)g.nr, 0);
  mvd_r[1] = TF(a.mvd_r, g.mvd_r, 0); ilamc[1] = TD(a.ilamc, g.ilamc, 0);
  lqc[1] = a.l_qc != 0; lqr[1] = a.l_qr != 0;
  tsad_tl::TempoTend& T = Tb[i];
  T.pnr_rcr[1] = TD(a.pnr_rcr, g.pnr_rcr, 0); T.prr_wau[1] = TD(a.prr_wau, g.prr_wau, 0);
  T.pnr_wau[1] = TD(a.pnr_wau, g.pnr_wau, 0); T.pnc_wau[1] = TD(a.pnc_wau, g.pnc_wau, 0);
  T.prr_rcw[1] = TD(a.prr_rcw, g.prr_rcw, 0); T.pnc_rcw[1] = TD(a.pnc_rcw, g.pnc_rcw, 0);
  TempoParams P; P.L = L;
  tsad_tl::warm_rain(P, wr_ctx(inv), 1, rhof, lqc, rc, nc, ilamc, mvd_c, lqr, rr, nr, mvd_r, T);
#define WR_OUT(e, j) y[i].v[j] = e.v; dy[i].v[j] = e.d;
  WR_TEND(T, 1, WR_OUT)
#undef WR_OUT
}

// Generated adjoint over one tile of calls [first, first+m): record, seed, reverse, gather.
// count_only != 0 is the sizing pass (no tape stored, used[] gets the node count).
extern "C" __global__ void wr_ad(int m, int first, int count_only, const WrIn* x, const Out6* by,
                                 const TempoBlob* L, float inv, tsad_tp::TempoTend* Tb, Out6* y, WrBarD* bx) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= m) return;
  const int i = first + s;
  tsad::g_tsad_arena.used[s] = 0; tsad::g_tsad_arena.overflow[s] = 0;
  using tsad_tp::TF; using tsad_tp::TD;
  const WrIn a = x[i];
  TF rhof[2], rc[2], nc[2], mvd_c[2], rr[2], nr[2], mvd_r[2]; TD ilamc[2]; bool lqc[2], lqr[2];
  int leaf[14];
#define LF(v, val, j) v = TF(val, tsad::leaf_node(), 0); leaf[j] = v.d;
#define LD(v, val, j) v = TD(val, tsad::leaf_node(), 0); leaf[j] = v.d;
  LF(rhof[1], a.rhof, 0) LF(rc[1], a.rc, 1) LF(nc[1], a.nc, 2) LF(mvd_c[1], a.mvd_c, 3) LF(rr[1], a.rr, 4)
  LF(nr[1], a.nr, 5) LF(mvd_r[1], a.mvd_r, 6) LD(ilamc[1], a.ilamc, 7)
  lqc[1] = a.l_qc != 0; lqr[1] = a.l_qr != 0;
  tsad_tp::TempoTend& T = Tb[s];
  LD(T.pnr_rcr[1], a.pnr_rcr, 8) LD(T.prr_wau[1], a.prr_wau, 9) LD(T.pnr_wau[1], a.pnr_wau, 10)
  LD(T.pnc_wau[1], a.pnc_wau, 11) LD(T.prr_rcw[1], a.prr_rcw, 12) LD(T.pnc_rcw[1], a.pnc_rcw, 13)
#undef LF
#undef LD
  TempoParams P; P.L = L;
  tsad_tp::warm_rain(P, wr_ctx(inv), 1, rhof, lqc, rc, nc, ilamc, mvd_c, lqr, rr, nr, mvd_r, T);
#define WR_OUT(e, j) y[i].v[j] = e.v;
  WR_TEND(T, 1, WR_OUT)
#undef WR_OUT
  if (count_only) return;
  const Out6 b = by[i];
#define WR_SEED(e, j) tsad::seed_bar(e.d, b.v[j]);
  WR_TEND(T, 1, WR_SEED)
#undef WR_SEED
  tsad::reverse_slice();
  WrBarD r;
  r.rhof = tsad::read_bar(leaf[0]); r.rc = tsad::read_bar(leaf[1]); r.nc = tsad::read_bar(leaf[2]);
  r.mvd_c = tsad::read_bar(leaf[3]); r.rr = tsad::read_bar(leaf[4]); r.nr = tsad::read_bar(leaf[5]);
  r.mvd_r = tsad::read_bar(leaf[6]); r.ilamc = tsad::read_bar(leaf[7]);
  for (int j = 0; j < 6; ++j) r.t[j] = tsad::read_bar(leaf[8 + j]);
  bx[i] = r;
}

// Host sets the arena through this entry (one thread), so no symbol lookup is needed.
extern "C" __global__ void tsad_set_arena(tsad::Node* nodes, double* bars, int* used, int* overflow, int capacity) {
  tsad::g_tsad_arena.nodes = nodes; tsad::g_tsad_arena.bars = bars; tsad::g_tsad_arena.used = used;
  tsad::g_tsad_arena.overflow = overflow; tsad::g_tsad_arena.capacity = capacity;
}
extern "C" __global__ void wr_sizes(unsigned long long* o) {
  o[0] = sizeof(::TempoTend); o[1] = sizeof(tsad_tl::TempoTend); o[2] = sizeof(tsad_tp::TempoTend);
  o[3] = sizeof(tsad::Node); o[4] = sizeof(WrBarD);
}
