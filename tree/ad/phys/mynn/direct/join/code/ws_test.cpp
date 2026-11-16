// CPU test that the joined backward's tendency and prediction stages read no word of
// the thread's workspace that they did not write. On the card the stages' large work arrays are carved from a per-thread
// workspace (card/arena_rewrite.py, -DJW_ARENA) that is not cleared and is kept from launch to launch; an array declared
// without `{}` starts from what the last launch left. THE BREAKAGE (bl-fast-live, JOIN_MIX at 949ca6b8e): the bars of the
// diffusivities (DFH of the heat solves, DF of the six solves) were added into such an array and read back by
// mix_reverse() without ever being cleared, so the tape-free route's output bars were wrong from its first launch and
// grew every step; and after the merge of the plume momentum its bars (FU, SUB, DET) were sent to the sink and the words
// pmom_reverse() read were never written at all.
// The test: the same frame and the same seeded bars through the heat solves, the six solves and the prediction three
// times, with the workspace filled before every stage call with bytes 0x00, 0xFF (every double a NaN) and 0x5A. The bar
// ledger must come out the same bytes.
// build (from join/, the arena form the card ships):
//   JW_MIN_WORDS=150 JW_OBJECTS=0 python3 card/arena_rewrite.py blocks B
//   g++ -std=c++17 -O1 -DJW_ARENA [-DJOIN_MIX] [-DJOIN_PMOM] -IB code/ws_test.cpp -o t && ./t        (code/ws_test.sh runs all four)
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <memory>
#include "join.hpp"
using namespace direct;
using namespace direct::join;

struct Rng { std::mt19937_64 g{20261114}; double u(double a, double b) { return a + (b - a) * (double(g() >> 11) / 9007199254740992.0); } };

static void fill_frame(ColumnFrame<float>& F, Rng& r, int n) {
  std::memset(static_cast<void*>(&F), 0, sizeof F); F.n = n; F.dt = 60.f; F.dheat = 1; F.ktop = -1;
  float dz[MAX + 1]{}, zw[MAX + 2]{}, rho[MAX + 1]{}, tk[MAX + 1]{}, u[MAX + 1]{}, dfm[MAX + 1]{}, dfh[MAX + 1]{}, el[MAX + 1]{}, qkw[MAX + 1]{};
  for (int k = 1; k <= n; ++k) { dz[k] = float(r.u(20, 300)); zw[k + 1] = zw[k] + dz[k]; rho[k] = float(1.15 * std::exp(-zw[k] / 9000.0)); tk[k] = float(295.0 - 0.006 * zw[k]);
                                 u[k] = float(r.u(-15, 15)); el[k] = float(r.u(5, 200)); qkw[k] = float(r.u(0.05, 2.0)); }
  float* P = F.prod; P[5 * MAX + 1] = dz[1];
  for (int k = 2; k <= n; ++k) {     // the production cut's words mix_dfs() reads, and the diffusivities re-formed from them
    const float sm = float(r.u(0.05, 1.0)), sh = float(r.u(0.05, 1.5));
    P[k] = qkw[k]; P[MAX + k] = float(r.u(1e-6, 1e-2)); P[2 * MAX + k] = float(r.u(-1e-3, 1e-3)); P[5 * MAX + k] = dz[k]; P[6 * MAX + k] = el[k]; P[7 * MAX + k] = sm; P[8 * MAX + k] = sh;
    const float dzk = 0.5f * (dz[k] + dz[k - 1]); dfm[k] = el[k] * qkw[k] * sm / dzk; dfh[k] = el[k] * qkw[k] * sh / dzk;
  }
#ifdef JOIN_MIX
  F.mx.on = 1; F.mx.t = 1e-4;
  for (int k = 2; k <= n; ++k) { F.mx.qtw[k] = qkw[k]; for (int i = 0; i < 20; ++i) F.mx.jac[20 * k + i] = r.u(-1, 1); }
#endif
#ifdef JOIN_PMOM
  F.pm.on = 1; F.pm.started = 0x15; F.pm.adj_set = 1; F.pm.dz1 = dz[1]; F.pm.dz2 = dz[2]; F.pm.psig = 0.8f; F.pm.adj = 0.9f;
  for (int k = 1; k <= n; ++k) { F.pm.pass[k] = k < n / 2 ? 0x15 : 0; F.pm.cs[k] = float(r.u(0, 0.2)); F.pm.dzi[k] = dz[k]; F.pm.rhoi[k] = rho[k]; F.pm.rho[k] = rho[k]; F.pm.ea[k] = float(r.u(0.8, 1));
    for (int ip = 0; ip < NUPM; ++ip) { F.pm.om[ip * MAX + k] = float(r.u(0.5, 1)); F.pm.en[ip * MAX + k] = float(r.u(0, 0.5)); F.pm.c1[ip * MAX + k] = float(r.u(0, 0.1)); F.pm.dc[ip * MAX + k] = r.u(0, 1e-3); } }
#endif
  float* H = F.heat;
  for (int k = 1; k <= n; ++k) {
    H[DZ * MAX + k] = dz[k]; H[RHO * MAX + k] = rho[k]; H[TK * MAX + k] = tk[k]; H[QV * MAX + k] = float(r.u(1e-4, 1e-2)); H[EX * MAX + k] = float(std::pow(std::exp(-zw[k] / 8000.0), 0.2854));
    H[TH * MAX + k] = tk[k] / H[EX * MAX + k]; H[THL * MAX + k] = H[TH * MAX + k]; H[SQV * MAX + k] = H[QV * MAX + k]; H[SQC * MAX + k] = float(r.u(0, 1e-4)); H[SQI * MAX + k] = float(r.u(0, 1e-5));
    H[DFH * MAX + k] = dfh[k];
  }
  H[NHF * MAX] = 1.0e5f; H[NHF * MAX + 1] = 0.05f; H[NHF * MAX + 2] = 0.f; H[NHF * MAX + 3] = 2e-5f;
  for (int j = 0; j < 6; ++j) { float* S = F.solve[j];
    for (int k = 1; k <= n; ++k) { S[solves::DZ * MAX + k] = dz[k]; S[solves::RHO * MAX + k] = rho[k]; S[solves::DF * MAX + k] = j < 2 ? dfm[k] : dfh[k];
      S[solves::Q * MAX + k] = j < 2 ? u[k] : j == 2 ? 2e8f : j == 3 ? 1e3f : j == 4 ? 3e9f : 1e6f; S[solves::ZW * MAX + k] = zw[k]; S[solves::TK * MAX + k] = tk[k]; S[solves::QV * MAX + k] = H[QV * MAX + k];
      S[solves::COLD * MAX + k] = H[SQC * MAX + k]; S[solves::CNEW * MAX + k] = H[SQC * MAX + k]; }
    S[solves::PS] = 1.0e5f; S[solves::UST] = 0.35f; S[solves::WSPD] = 6.f; S[solves::OCE] = 0.f; S[solves::PBLH] = 900.f; S[solves::LAND] = 1.f; }
  float* Q = F.pred;      // f_mym_predict: dz, rho, el, dfq, s_aw, pdk, pdq, pdt, pdc, qke, qsq; ust, pmz
  for (int k = 1; k <= n; ++k) { Q[k] = dz[k]; Q[MAX + k] = rho[k]; Q[2 * MAX + k] = el[k]; Q[3 * MAX + k] = dfm[k]; Q[5 * MAX + k] = float(r.u(0, 1e-3)); Q[6 * MAX + k] = float(r.u(0, 1e-9));
                                 Q[7 * MAX + k] = float(r.u(0, 1e-6)); Q[8 * MAX + k] = float(r.u(-1e-9, 1e-9)); Q[9 * MAX + k] = qkw[k] * qkw[k]; Q[10 * MAX + k] = float(r.u(1e-12, 1e-8)); }
  Q[11 * MAX] = 0.35f; Q[11 * MAX + 1] = 1.f;
}

static void seed(ColumnBars& B, Rng& r, int n) {
  std::memset(static_cast<void*>(&B), 0, sizeof B);
  double* d[] = {B.du, B.dv, B.dth, B.dqv, B.dqc, B.dqi, B.dqs, B.dqnc, B.dqni, B.dqnwfa, B.dqnifa, B.qke1, B.qsq1, B.tsq1, B.cov1};
  for (double* a : d) for (int k = 1; k <= n; ++k) a[k] = r.u(-1, 1);
}

int main() {
  Rng r; auto F = std::make_unique<ColumnFrame<float>>(); auto B0 = std::make_unique<ColumnBars>(); auto B = std::make_unique<ColumnBars>(); auto R = std::make_unique<ColumnBars>();
  const unsigned char fills[3] = {0x00, 0xFF, 0x5A}; int bad = 0, ran = 0, pred_ran = 0; long long worst_words = 0;
  for (int t = 0; t < 200; ++t) {
    const int n = 10 + int(r.u(0, 49.99)); fill_frame(*F, r, n); seed(*B0, r, n);
    for (int f = 0; f < 3; ++f) {
      // the workspace is filled again before EACH stage call, so a stage that reads a word an earlier stage left there
      // (not only a word nobody wrote) is seen too. The calls and the clears are rev_tend()'s own, in its order.
      auto refill = [&] { std::memset(jw::base(), fills[f], 1 << 20); *jw::slot() = jw::base(); };
      *jw::over() = 0;
      std::memcpy(static_cast<void*>(B.get()), B0.get(), sizeof(ColumnBars));
      refill(); bool okt = rev_tend_heat(*F, *B);
      for (int j = 0; j < 6 && okt; ++j) { refill(); okt = rev_tend_solve(*F, *B, j); }
      { double* d[11] = {B->dth, B->dqv, B->dqc, B->dqi, B->dqs, B->du, B->dv, B->dqnc, B->dqni, B->dqnwfa, B->dqnifa}; for (double* a : d) clr(a, n); }
      refill(); const bool okp = okt && rev_pred(*F, *B);
      if (!okt) { std::printf("frame %d: the tendency stage refused a synthetic frame\n", t); return 2; }
      if (*jw::over()) { std::printf("frame %d: workspace overflow\n", t); return 2; }
      if (f == 0) { std::memcpy(static_cast<void*>(R.get()), B.get(), sizeof(ColumnBars)); ++ran; pred_ran += okp; continue; }
      const unsigned long long* a = reinterpret_cast<const unsigned long long*>(B.get()); const unsigned long long* b = reinterpret_cast<const unsigned long long*>(R.get());
      long long nd = 0; for (size_t i = 0; i < sizeof(ColumnBars) / 8; ++i) nd += a[i] != b[i];
      if (nd) { ++bad; if (nd > worst_words) worst_words = nd; }
    }
  }
  std::printf("%d synthetic frames (the prediction stage ran on %d), each with the workspace pre-filled 0x00, 0xFF and 0x5A: "
              "%d of %d filled runs differ from the zero-filled ledger (most words differing in one run: %lld of %zu)\n",
              ran, pred_ran, bad, 2 * ran, worst_words, sizeof(ColumnBars) / 8);
  std::printf("%s\n", bad == 0 && pred_ran > 0 ? "PASS" : "FAIL");
  return bad == 0 && pred_ran > 0 ? 0 : 1;
}
