// SPDX-License-Identifier: Apache-2.0
// Whole-launch derivative entries for tempo_warp_steps (hrrrv5 TEMPO) in the
// forecast's own warp shape: one 32-lane warp per column, level k on lane
// (k-1)%32, the driver statement by statement as tempo_warp_steps runs it
// (tempo_pack, the substeps of tempo_main_column, tempo_step_diags on the
// lead lane, the unpack/repack), on active types.
//
// A launch is split into segments of substeps [s0, s1) (per-substep
// checkpoints): the record entry starts from the launch inputs (s0 == 0) or
// from the state at the start of substep s0 read from a checkpoint, and ends
// with the launch outputs (s1 == steps) or the state at the start of substep
// s1. The state slot layout is the input layout (nin = 18*nz + 10):
// rows 0..16 of the column state (row 0 holds theta*exner after the pack),
// row 17 exner, then the 10 accumulators. The launch outputs keep the serial
// build's layout (nout = 18*nz + 10: f[0], f[4..16], s[0..9], profiles 0..3).
// Leaves: iidx[q] is the leaf node of input slot q (0: constant), oidx[q]
// the node of output slot q. The clock-cost buffer and the column order are
// not touched: they schedule columns and carry no derivative.
// Checkpoints (ckin/ckout) are [substep][slot][ckstride], this tile at column ckoff; the
// checkpoint pass (s0 = 0, s1 = steps, ckout set) writes the state at the start of every
// substep 0 < sub < steps, and ckm marks which slots are active there (a slot that is still a
// constant, e.g. an absent carrier never written, stays a constant in the segment, so the
// segment tape is the same graph as the whole-launch tape).
#define AD_NF_IN 18
#define AD_NF_OUT 14
#ifndef AD_PRIM
#define AD_PRIM float
#endif
typedef AD_PRIM ad_prim;
struct AdState { AR<ad_prim> f[18][TEMPO_NZP]; };

// F08: the PBL subgrid radii. The runner points g_tempo_bl at its snapshot of the launch's bl ([2][nz][columns]:
// qc_bl, cldfrac_bl) on a launch that takes tempo_warp_steps' `ml_bl && has_bl && re` path, and at 0 otherwise.
__device__ const float* g_tempo_bl = nullptr;
extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_adw_record(
    int nz, int columns, int steps, float dt, float** f, float** s, const TempoBlob* blob,
    int aero, int hail, int melt, int first, int presence, int radii,
    const int* cols, int ntile, int s0, int s1, const ad_prim* ckin, ad_prim* ckout, unsigned char* ckm, long long ckstride, int ckoff,
    AdState* states, adt::TempoWork* work, ad_prim* yout, int* oidx, int* iidx, const double* xpert) {
  ad_begin();
  const int lane = ad_lane();
  const int w = (int)(ad_tid() >> 5);
  if (w >= ntile) return; // whole warp
  const int c = cols[w];
  AdState& st = states[w];
  adt::TempoWork& W = work[w];
  TempoParams P = {blob};
  const long long nin = (long long)AD_NF_IN * nz + 10;
  // Leaves in slot order, each lane its own levels.
  TEMPO_LEVELS(k, nz) {
    const long long x = (long long)(k - 1) * columns + c;
    for (int j = 0; j < AD_NF_IN; ++j) {
      const long long q = (long long)j * nz + k - 1;
      const float raw = s0 == 0 ? (f[j] ? f[j][x] : 0.0f) : (float)ckin[q * ckstride + ckoff + w];
      const double p = xpert ? xpert[q * ntile + w] : 0.0;
      if (s0 == 0 || ckm[(long long)s0 * nin * ckstride + q * ckstride + ckoff + w])
        st.f[j][k] = ad_mk(xpert ? (ad_prim)((double)raw + p) : (ad_prim)raw, ad_leaf());
      else
        st.f[j][k] = ad_mk((ad_prim)raw, 0); // a constant at this substep boundary (never written yet): stays one
      iidx[q * ntile + w] = st.f[j][k].i;
    }
  }
  AR<ad_prim> sa[10];
  if (TEMPO_LEAD) {
    for (int j = 0; j < 10; ++j) {
      const long long q = (long long)AD_NF_IN * nz + j;
      const float raw = s0 == 0 ? (s[j] ? s[j][c] : 0.0f) : (float)ckin[q * ckstride + ckoff + w];
      if (s0 == 0 || ckm[(long long)s0 * nin * ckstride + q * ckstride + ckoff + w])
        sa[j] = ad_mk((ad_prim)((double)raw + (xpert ? xpert[q * ntile + w] : 0.0)), ad_leaf());
      else
        sa[j] = ad_mk((ad_prim)raw, 0);
      iidx[q * ntile + w] = sa[j].i;
    }
  }
  // tempo_pack: theta*exner into slot 0 (exner stays in row 17), absent carriers zero.
  if (s0 == 0) {
    TEMPO_LEVELS(k, nz) {
      st.f[0][k] = st.f[0][k] * st.f[17][k];
      for (int j = 1; j < 17; ++j) if (!f[j]) st.f[j][k] = AR<ad_prim>(0.0f);
    }
  }
  for (int sub = s0; sub < s1; ++sub) {
    if (ckout && sub > s0) {
      ad_prim* ck = ckout + (long long)sub * nin * ckstride + ckoff;
      unsigned char* cm = ckm + (long long)sub * nin * ckstride + ckoff;
      TEMPO_LEVELS(k, nz) for (int j = 0; j < AD_NF_IN; ++j) {
        const long long o = ((long long)j * nz + k - 1) * ckstride + w;
        ck[o] = st.f[j][k].v; cm[o] = st.f[j][k].i != 0;
      }
      if (TEMPO_LEAD) for (int j = 0; j < 10; ++j) {
        const long long o = ((long long)AD_NF_IN * nz + j) * ckstride + w;
        ck[o] = sa[j].v; cm[o] = sa[j].i != 0;
      }
    }
    ad_mark(sub - s0);
    // tempo_columns ran on freshly zeroed work every substep; tempo_main_column clears every
    // TempoTend element before use, so the forward keeps it. The first substep of a segment
    // clears it as well (its words then hold node ids of another tape): same primal words.
    unsigned long long* w8 = reinterpret_cast<unsigned long long*>(&W);
    const int tb = (int)((char*)&W.T - (char*)&W) / 8;
    const int te = tb + (int)(sizeof(adt::TempoTend) / 8);
    for (int i = lane; i < (int)(sizeof(adt::TempoWork) / 8); i += 32)
      if (sub == s0 || i < tb || i >= te) w8[i] = 0ULL;
    ad_sync();
    const int last = (sub + 1 == steps);
    const int re = last && radii;
    TempoCtx ctx = {dt, 1.0f / dt, aero, hail, 0, melt, 0, 1, last, re, re, re, last, first && sub == 0 && c == 0};
    adt::tempo_main_column(P, ctx, nz, W, st.f[4], st.f[5], st.f[7], st.f[6], st.f[8], st.f[9],
                           (presence & 16) ? st.f[16] : nullptr, st.f[10], st.f[11],
                           (presence & 1) ? st.f[12] : nullptr, (presence & 8) ? st.f[15] : nullptr,
                           (presence & 2) ? st.f[13] : nullptr, (presence & 4) ? st.f[14] : nullptr,
                           st.f[0], st.f[1], st.f[2], st.f[3]);
    ad_sync();
    // F08: tempo_warp_steps' PBL subgrid radii, the forward's own call and arguments (b = bl[x], b + nz =
    // bl[nz * columns + x], nt 1, c0 0). bl enters as constants (no input slots): its slope is held.
    if (re && g_tempo_bl) {
      AR<float> qb[TEMPO_NZP], fb[TEMPO_NZP];
      TEMPO_LEVELS(k, nz) {
        const long long x = (long long)(k - 1) * columns + c;
        qb[k - 1] = ad_mk((ad_prim)g_tempo_bl[x], 0);
        fb[k - 1] = ad_mk((ad_prim)g_tempo_bl[(long long)nz * columns + x], 0);
      }
      adt::tempo_subgrid_cloud_radii(P, ctx, nz, W, st.f[5], st.f[6], st.f[7], st.f[8],
                                     (presence & 1) ? st.f[12] : nullptr, st.f[2], qb, fb, 1, 0);
      ad_sync();
    }
    // tempo_step_diags, in its order of operations (lead lane).
    if (TEMPO_LEAD) {
      AR<ad_prim> maxhail = W.max_hail_diameter[1];
      for (int k = 2; k <= nz; ++k) maxhail = tmax(maxhail, W.max_hail_diameter[k]);
      const AR<ad_prim> r = W.rain_precip, i = W.ice_liquid_equiv_precip,
                      sn = W.snow_liquid_equiv_precip, g = W.graupel_liquid_equiv_precip;
      sa[2] = (sa[2] + i) + sn; sa[3] = (sa[3] + i) + sn;
      sa[4] += g; sa[5] += g; sa[0] = (((sa[0] + r) + i) + sn) + g; sa[1] = (((sa[1] + r) + i) + sn) + g;
      sa[6] = W.frozen_fraction; sa[7] += W.frz_rain_precip;
      sa[8] = W.max_hail_diameter[1]; sa[9] = maxhail;
    }
    // tempo_pack's unpack then the next pack: theta = T / exner, T = theta * exner.
    TEMPO_LEVELS(k, nz) {
      const AR<ad_prim> theta = st.f[0][k] / st.f[17][k];
      if (sub + 1 < steps) st.f[0][k] = theta * st.f[17][k];
      else {
        const long long q = (long long)(k - 1);
        yout[(0 * nz + q) * ntile + w] = theta.v; oidx[(0 * nz + q) * ntile + w] = theta.i;
        for (int j = 4; j < 17; ++j) {
          const long long o = ((long long)(j - 3) * nz + q) * ntile + w;
          yout[o] = st.f[j][k].v; oidx[o] = f[j] ? st.f[j][k].i : 0;
        }
        const long long pb = (long long)AD_NF_OUT * nz + 10;
        yout[(pb + q) * ntile + w] = W.refl10cm[k].v; oidx[(pb + q) * ntile + w] = W.refl10cm[k].i;
        const AR<ad_prim> rr[3] = {W.re_cloud[k], W.re_ice[k], W.re_snow[k]};
        for (int p = 0; p < 3; ++p) {
          const long long o = (pb + (long long)(p + 1) * nz + q) * ntile + w;
          yout[o] = radii ? rr[p].v : (ad_prim)0; oidx[o] = radii ? rr[p].i : 0;
        }
      }
    }
    ad_sync();
  }
  ad_mark(s1 - s0);
  if (s1 < steps) {
    TEMPO_LEVELS(k, nz) for (int j = 0; j < AD_NF_IN; ++j) {
      const long long o = ((long long)j * nz + k - 1) * ntile + w;
      yout[o] = st.f[j][k].v; oidx[o] = st.f[j][k].i;
    }
    if (TEMPO_LEAD) for (int j = 0; j < 10; ++j) {
      const long long o = ((long long)AD_NF_IN * nz + j) * ntile + w;
      yout[o] = sa[j].v; oidx[o] = sa[j].i;
    }
  } else if (TEMPO_LEAD) {
    for (int j = 0; j < 10; ++j) {
      const long long o = ((long long)AD_NF_OUT * nz + j) * ntile + w;
      yout[o] = sa[j].v; oidx[o] = s[j] ? sa[j].i : 0;
    }
  }
  ad_end();
}

// Tangent: forward sweep of one segment's tape, phase by phase. tv has the tape's layout.
extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_adw_tl(int ntile, int nin, int nout, const double* xdot, double* tv,
                                                                     const int* iidx, const int* oidx, double* ydot) {
  const int lane = ad_lane();
  const long long w = ad_tid() >> 5;
  if (w >= ntile) return;
  const long long B = g_ad.base[w];
  const int* cn = g_ad.count + AD_NCNT * (w * 32 + lane);
  const int N = cn[0], P = cn[4];
  double* t = tv + B;
  for (int q = lane; q < nin; q += 32) { const int id = iidx[(long long)q * ntile + w]; if (id) t[id] = xdot[(long long)q * ntile + w]; }
  __syncwarp();
  const int* ph = g_ad.phase + w * g_ad.pcap * 32;
  int lo = 1; // phases start on a common row (ad_sync)
  for (int p = 0; p <= P; ++p) {
    const int hi = p < P ? ph[p * 32 + lane] : N;
    const int top = (int)__reduce_max_sync(0xffffffffu, (unsigned)hi);
    for (int n = lo; n < top; ++n) {
      if (n >= hi) continue;
      const long long o = B + ((long long)n << 5) + lane;
      const int a = g_ad.a[o];
      if (a < 0) continue;
      const int b = g_ad.b[o];
      double v = g_ad.pa[o] * t[a];
      if (b) v += g_ad.pb[o] * t[b];
      tv[o] = v;
    }
    lo = top;
    __syncwarp();
  }
  for (int q = lane; q < nout; q += 32) { const int id = oidx[(long long)q * ntile + w]; ydot[(long long)q * ntile + w] = id ? t[id] : 0.0; }
}

// Adjoint: reverse sweep of one segment's tape, phase by phase in reverse. Contributions to
// another lane's node wait in the lane's outbox [obcap][32] until the phase ends, then each lane
// adds the ones addressed to it in fixed lane order. fault[w] bit 1: outbox full, bit 2: a
// cross-lane edge inside one phase (the phase rule), bit 4: a cross-lane edge in phase 0.
extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_adw_adj(int ntile, int nin, int nout, const double* ybar, double* av,
    const int* iidx, const int* oidx, double* xbar, int* obid, double* obv, int obcap, int* fault) {
  __shared__ int obn[AD_BLOCK];
  const int lane = ad_lane();
  const long long w = ad_tid() >> 5;
  if (w >= ntile) return;
  const long long B = g_ad.base[w];
  const int* cn = g_ad.count + AD_NCNT * (w * 32 + lane);
  const int N = cn[0], P = cn[4];
  double* g = av + B;
  for (int n = 0; n < N; ++n) g[((long long)n << 5) + lane] = 0.0;
  __syncwarp();
  for (int q = 0; q < nout; ++q) { const int id = oidx[(long long)q * ntile + w]; if (id && (id & 31) == lane) g[id] += ybar[(long long)q * ntile + w]; }
  __syncwarp();
  const int* ph = g_ad.phase + w * g_ad.pcap * 32;
  int* bid = obid + w * (long long)obcap * 32;
  double* bv = obv + w * (long long)obcap * 32;
  const int wb = threadIdx.x & ~31;
  int flt = 0;
  for (int p = P; p >= 0; --p) {
    // phase p: rows [lo, top) of the warp, this lane's nodes [lo, hi)
    const int lo = p > 0 ? (int)__reduce_max_sync(0xffffffffu, (unsigned)ph[(p - 1) * 32 + lane]) : 1;
    const int hi = p < P ? ph[p * 32 + lane] : N;
    const int top = (int)__reduce_max_sync(0xffffffffu, (unsigned)hi);
    int k = 0;
    for (int n = top - 1; n >= lo; --n) {
      if (n >= hi) continue;
      const long long off = ((long long)n << 5) + lane;
      const double gg = g[off];
      if (gg == 0.0) continue;
      const int a = g_ad.a[B + off];
      if (a < 0) continue;
      const int b = g_ad.b[B + off];
      const int tg[2] = {a, b};
      const double v[2] = {g_ad.pa[B + off] * gg, b ? g_ad.pb[B + off] * gg : 0.0};
      for (int e = 0; e < 2; ++e) {
        const int id = tg[e];
        if (!id) continue;
        if ((id & 31) == lane) { g[id] += v[e]; continue; }
        if (p == 0 || (id >> 5) >= lo) flt |= p == 0 ? 4 : 2;
        if (k < obcap) { bid[(long long)k * 32 + lane] = id; bv[(long long)k * 32 + lane] = v[e]; } else flt |= 1;
        ++k;
      }
    }
    obn[threadIdx.x] = k < obcap ? k : obcap;
    __syncwarp();
    for (int l = 0; l < 32; ++l) {
      const int m = obn[wb + l];
      for (int e = 0; e < m; ++e) { const int id = bid[(long long)e * 32 + l]; if ((id & 31) == lane) g[id] += bv[(long long)e * 32 + l]; }
    }
    __syncwarp();
  }
  for (int q = lane; q < nin; q += 32) { const int id = iidx[(long long)q * ntile + w]; xbar[(long long)q * ntile + w] = id ? g[id] : 0.0; }
  flt = (int)__reduce_or_sync(0xffffffffu, (unsigned)flt);
  if (lane == 0) fault[w] = flt;
}

extern "C" __global__ void tempo_adw_sizes(unsigned long long* out) {
  out[0] = sizeof(adt::TempoWork); out[1] = sizeof(AdState); out[2] = sizeof(ad_prim); out[3] = sizeof(AdTape);
}
