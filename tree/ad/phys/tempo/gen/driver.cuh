// SPDX-License-Identifier: Apache-2.0
// Whole-launch derivative entries for tempo_warp_steps (hrrrv5 TEMPO).
// Serial shape, one thread per column. The record kernel replays the
// forward's driver statement by statement (tempo_pack, substeps of
// tempo_main_column, tempo_step_diags, the unpack/repack) on active types,
// writes the primal outputs to yout (never in place) and the tape node of
// each output to oidx. The clock-cost buffer and the column order are not
// touched: they schedule columns and carry no derivative.
//
// Input slots  (nin  = 18*nz + 10): f[j][k] for j = 0..17, then s[0..9].
// Output slots (nout = 18*nz + 10): f[0], f[4..16], then s[0..9], then
//                                   profiles[0..3] (refl10cm, re_cloud, re_ice, re_snow).
// Leaf node of input slot q is q + 1.
#define AD_NF_IN 18
#define AD_NF_OUT 14
#ifndef AD_PRIM
#define AD_PRIM float
#endif
typedef AD_PRIM ad_prim;
struct AdState { AR<ad_prim> f[18][TEMPO_NZP]; };

extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_ad_record(
    int nz, int columns, int steps, float dt, float** f, float** s, const TempoBlob* blob,
    int aero, int hail, int melt, int first, int presence, int radii,
    const int* cols, int ntile, AdState* states, adt::TempoWork* work, ad_prim* yout, int* oidx, const double* xpert) {
  ad_begin();
  const int t = (int)ad_tid();
  if (t >= ntile) return;
  const int c = cols[t];
  AdState& st = states[t];
  adt::TempoWork& W = work[t];
  TempoParams P = {blob};
  // Leaves in slot order.
  for (int j = 0; j < AD_NF_IN; ++j)
    for (int k = 1; k <= nz; ++k) {
      const long long x = (long long)(k - 1) * columns + c;
      const double p = xpert ? xpert[((long long)j * nz + k - 1) * ntile + t] : 0.0;
      st.f[j][k] = ad_mk(xpert ? (ad_prim)((double)(f[j] ? f[j][x] : 0.0f) + p) : (ad_prim)(f[j] ? f[j][x] : 0.0f), ad_leaf());
    }
  AR<ad_prim> sa[10];
  for (int j = 0; j < 10; ++j) sa[j] = ad_mk((ad_prim)((double)(s[j] ? s[j][c] : 0.0f) + (xpert ? xpert[((long long)AD_NF_IN * nz + j) * ntile + t] : 0.0)), ad_leaf());
  // tempo_pack: theta*exner into slot 0 (exner stays in row 17), absent carriers zero.
  AR<ad_prim> ex[TEMPO_NZP];
  for (int k = 1; k <= nz; ++k) {
    ex[k] = st.f[17][k];
    st.f[0][k] = st.f[0][k] * ex[k];
    for (int j = 1; j < 17; ++j) if (!f[j]) st.f[j][k] = AR<ad_prim>(0.0f);
  }
  for (int sub = 0; sub < steps; ++sub) {
    unsigned long long* w8 = reinterpret_cast<unsigned long long*>(&W);
    const int tb = (int)((char*)&W.T - (char*)&W) / 8;
    const int te = tb + (int)(sizeof(adt::TempoTend) / 8);
    for (int i = 0; i < (int)(sizeof(adt::TempoWork) / 8); ++i)
      if (i < tb || i >= te) w8[i] = 0ULL;
    const int last = (sub + 1 == steps);
    const int re = last && radii;
    TempoCtx ctx = {dt, 1.0f / dt, aero, hail, 0, melt, 0, 1, last, re, re, re, last, first && sub == 0 && c == 0};
    adt::tempo_main_column(P, ctx, nz, W, st.f[4], st.f[5], st.f[7], st.f[6], st.f[8], st.f[9],
                           (presence & 16) ? st.f[16] : nullptr, st.f[10], st.f[11],
                           (presence & 1) ? st.f[12] : nullptr, (presence & 8) ? st.f[15] : nullptr,
                           (presence & 2) ? st.f[13] : nullptr, (presence & 4) ? st.f[14] : nullptr,
                           st.f[0], st.f[1], st.f[2], st.f[3]);
    // tempo_step_diags, in its order of operations.
    {
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
    for (int k = 1; k <= nz; ++k) {
      const AR<ad_prim> theta = st.f[0][k] / ex[k];
      if (sub + 1 < steps) st.f[0][k] = theta * ex[k];
      else {
        const long long q = (long long)(k - 1);
        yout[(0 * nz + q) * ntile + t] = theta.v; oidx[(0 * nz + q) * ntile + t] = theta.i;
        for (int j = 4; j < 17; ++j) {
          const long long o = ((long long)(j - 3) * nz + q) * ntile + t;
          yout[o] = st.f[j][k].v; oidx[o] = f[j] ? st.f[j][k].i : 0;
        }
        const long long pb = (long long)AD_NF_OUT * nz + 10;
        yout[(pb + q) * ntile + t] = W.refl10cm[k].v; oidx[(pb + q) * ntile + t] = W.refl10cm[k].i;
        const AR<ad_prim> rr[3] = {W.re_cloud[k], W.re_ice[k], W.re_snow[k]};
        for (int p = 0; p < 3; ++p) {
          const long long o = (pb + (long long)(p + 1) * nz + q) * ntile + t;
          yout[o] = radii ? rr[p].v : (ad_prim)0; oidx[o] = radii ? rr[p].i : 0;
        }
      }
    }
  }
  for (int j = 0; j < 10; ++j) {
    const long long o = ((long long)AD_NF_OUT * nz + j) * ntile + t;
    yout[o] = sa[j].v; oidx[o] = s[j] ? sa[j].i : 0;
  }
  ad_end();
}

// Tangent: forward sweep of the tape. xdot [nin][ntile], tv [cap][ntile] scratch, ydot [nout][ntile].
extern "C" __global__ void tempo_ad_tl(int ntile, int nin, int nout, const double* xdot, double* tv,
                                       const int* oidx, double* ydot) {
  const long long t = ad_tid();
  if (t >= ntile) return;
  const long long T = g_ad.T;
  const int N = g_ad.count[4 * t];
  tv[t] = 0.0;
  for (int q = 0; q < nin; ++q) tv[(long long)(q + 1) * T + t] = xdot[(long long)q * ntile + t];
  for (int n = nin + 1; n < N; ++n) {
    const long long o = (long long)n * T + t;
    const int a = g_ad.a[o], b = g_ad.b[o];
    double v = g_ad.pa[o] * tv[(long long)a * T + t];
    if (b) v += g_ad.pb[o] * tv[(long long)b * T + t];
    tv[o] = v;
  }
  for (int q = 0; q < nout; ++q) {
    const int n = oidx[(long long)q * ntile + t];
    ydot[(long long)q * ntile + t] = n ? tv[(long long)n * T + t] : 0.0;
  }
}

// Adjoint: reverse sweep. ybar [nout][ntile], av [cap][ntile] scratch, xbar [nin][ntile].
extern "C" __global__ void tempo_ad_adj(int ntile, int nin, int nout, const double* ybar, double* av,
                                        const int* oidx, double* xbar) {
  const long long t = ad_tid();
  if (t >= ntile) return;
  const long long T = g_ad.T;
  const int N = g_ad.count[4 * t];
  for (int n = 0; n < N; ++n) av[(long long)n * T + t] = 0.0;
  for (int q = 0; q < nout; ++q) {
    const int n = oidx[(long long)q * ntile + t];
    if (n) av[(long long)n * T + t] += ybar[(long long)q * ntile + t];
  }
  for (int n = N - 1; n > nin; --n) {
    const long long o = (long long)n * T + t;
    const double g = av[o];
    if (g == 0.0) continue;
    const int a = g_ad.a[o], b = g_ad.b[o];
    av[(long long)a * T + t] += g_ad.pa[o] * g;
    if (b) av[(long long)b * T + t] += g_ad.pb[o] * g;
  }
  for (int q = 0; q < nin; ++q) xbar[(long long)q * ntile + t] = av[(long long)(q + 1) * T + t];
}

extern "C" __global__ void tempo_ad_sizes(unsigned long long* out) {
  out[0] = sizeof(adt::TempoWork); out[1] = sizeof(AdState); out[2] = sizeof(ad_prim);
}

// ---------------------------------------------------------------------------
// tempo_precip, one column per thread. Inputs (nin = 3*nz + 23): f[3], f[4],
// f[18] by level, s[0..9], d[0..12]. Outputs (23): s[0..9], d[0..12]; the
// bucket counter (integer) goes to cout for the identity check only.
extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_ad_precip_record(
    float** f, float** s, float** d, const int* counter, int nz, int nc, int finish, int accrue, float bucket,
    const int* cols, int ntile, AR<ad_prim>* buf, ad_prim* yout, int* oidx, int* cout, const double* xpert) {
  ad_begin();
  const int t = (int)ad_tid();
  if (t >= ntile) return;
  const int c = cols[t];
  const int nin = 3 * nz + 23;
  AR<ad_prim>* b = buf + (long long)t * nin;
  int q = 0;
  auto leaf = [&](float v) { const double p = xpert ? xpert[(long long)q * ntile + t] : 0.0;
                             b[q] = ad_mk(xpert ? (ad_prim)((double)v + p) : (ad_prim)v, ad_leaf()); ++q; };
  const int fj[3] = {3, 4, 18};
  for (int i = 0; i < 3; ++i) for (int k = 0; k < nz; ++k) leaf(f[fj[i]] ? f[fj[i]][(long long)k * nc + c] : 0.0f);
  for (int j = 0; j < 10; ++j) leaf(s[j] ? s[j][c] : 0.0f);
  for (int j = 0; j < 13; ++j) leaf(d[j] ? d[j][c] : 0.0f);
  AR<ad_prim>* fp[20] = {nullptr}; AR<ad_prim>* sp[12] = {nullptr}; AR<ad_prim>* dp[13] = {nullptr};
  fp[3] = b; fp[4] = b + nz; fp[18] = b + 2 * nz;
  for (int j = 0; j < 10; ++j) sp[j] = b + 3 * nz + j;
  for (int j = 0; j < 13; ++j) dp[j] = b + 3 * nz + 10 + j;
  int cnt = counter[c];
  adt::tempo_precip(fp, sp, dp, &cnt, nz, 1, finish, accrue, bucket);
  for (int j = 0; j < 23; ++j) {
    const AR<ad_prim>& v = j < 10 ? *sp[j] : *dp[j - 10];
    yout[(long long)j * ntile + t] = v.v; oidx[(long long)j * ntile + t] = (j < 10 ? (s[j] != nullptr) : (d[j - 10] != nullptr)) ? v.i : 0;
  }
  cout[t] = cnt;
  ad_end();
}

// tempo_radar, one column per thread. Inputs (nin = 7*nz + 3): f[0], f[1], f[3],
// f[4], f[17], f[19], profile[0] by level, radar[0..2]. Outputs (nz + 3):
// radar[0..2], rh by level.
extern "C" __global__ void __launch_bounds__(AD_BLOCK) tempo_ad_radar_record(
    float** f, float** profile, float** radar, int nz, int nc, int hourly, int full, const int* hourly_word,
    const int* cols, int ntile, AR<ad_prim>* buf, ad_prim* yout, int* oidx, int* unused, const double* xpert) {
  ad_begin();
  const int t = (int)ad_tid();
  if (t >= ntile) return;
  const int c = cols[t];
  const int nin = 7 * nz + 3;
  AR<ad_prim>* b = buf + (long long)t * (nin + nz);
  int q = 0;
  auto leaf = [&](float v) { const double p = xpert ? xpert[(long long)q * ntile + t] : 0.0;
                             b[q] = ad_mk(xpert ? (ad_prim)((double)v + p) : (ad_prim)v, ad_leaf()); ++q; };
  const int fj[6] = {0, 1, 3, 4, 17, 19};
  for (int i = 0; i < 6; ++i) for (int k = 0; k < nz; ++k) leaf(f[fj[i]] ? f[fj[i]][(long long)k * nc + c] : 0.0f);
  for (int k = 0; k < nz; ++k) leaf(profile[0] ? profile[0][(long long)k * nc + c] : 0.0f);
  for (int j = 0; j < 3; ++j) leaf(radar[j] ? radar[j][c] : 0.0f);
  AR<ad_prim>* fp[20] = {nullptr}; AR<ad_prim>* pp[4] = {nullptr}; AR<ad_prim>* rp[3] = {nullptr};
  for (int i = 0; i < 6; ++i) fp[fj[i]] = b + i * nz;
  pp[0] = b + 6 * nz;
  for (int j = 0; j < 3; ++j) rp[j] = b + 7 * nz + j;
  AR<ad_prim>* rh = b + nin;
  for (int k = 0; k < nz; ++k) rh[k] = AR<ad_prim>(0.0f);
  adt::tempo_radar(fp, pp, rp, rh, nz, 1, hourly, full, hourly_word);
  for (int j = 0; j < 3; ++j) { yout[(long long)j * ntile + t] = rp[j]->v; oidx[(long long)j * ntile + t] = rp[j]->i; }
  for (int k = 0; k < nz; ++k) { yout[(long long)(3 + k) * ntile + t] = rh[k].v; oidx[(long long)(3 + k) * ntile + t] = rh[k].i; }
  ad_end();
}
