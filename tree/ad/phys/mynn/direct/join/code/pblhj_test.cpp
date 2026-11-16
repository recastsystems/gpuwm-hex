// CPU test of the returned boundary-layer height in the joined backward (join.hpp rev_pblh, -DJOIN_PBLH):
// a unit bar of the height through rev_pblh and rev_driver against central differences of a double transcription of the
// driver statements (the zw sum, thv) and of f_get_pblh, over every driver input word the height can read
// (th, sqv, sqc, sqi, dz at every level, ust). Columns where a difference crosses a branch are skipped. The differences
// carry the float rounding of thv (3e-5 K against level differences of 0.01 K and less), so that bar is loose (5e-3);
// the tight bar (1e-6) is on the glue: the same bars against the mirror's six partials taken through the driver's
// statements by hand (thv = th ((1 + p608 sqv) - (sqc + sqi)), zw the sum of dz); the mirror itself is code/pblh_test.cpp's.
// Also: the plant (F.plant == 90, x1.001) must be seen, and a frame whose pblh1 word is not the mirror's is refused.
//   g++ -std=c++17 -O1 -ffp-contract=off -DJOIN_PBLH [-DJOIN_PMOM -DJOIN_MIX -DJOIN_CLOSURE -DJOIN_PHEAT] -Iblocks code/pblhj_test.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <memory>
#include "join.hpp"
using namespace direct;
using namespace direct::join;
static const int N = 45;
// the generated statements of f_get_pblh in double (code/pblh_test.cpp's transcription)
static double native(const double* thv, const double* zw, const double* dz, double ust, double landsea, int kts, int kte, int* sig) {
  int k = kts + 1, kthv = 1, kc = 0, fb = 0; double minthv = 9.e9;
  while (k <= kte && zw[k] <= 200.0) { if (minthv > thv[k]) { minthv = thv[k]; kthv = k; } k = k + 1; }
  const double delt = (landsea - 1.5 >= 0) ? 0.75 : 1.25;
  double pblh = 0; int capped = 0, floor6 = 0;
  for (k = kts + 1; k <= kte - 1; ++k) {
    if (thv[k] >= minthv + delt) {
      double den = thv[k] - thv[k - 1]; floor6 = !(den > (double)1e-6f); if (floor6) den = (double)1e-6f;
      double fr = (thv[k] - (minthv + delt)) / den; capped = !(fr < 1.0); if (capped) fr = 1.0;
      pblh = zw[k] - dz[k - 1] * fr; kc = k;
    }
    if (k == kte - 1) { pblh = zw[kts + 1]; fb = 1; }
    if (pblh > 0) break;
  }
  double wt = 0.5 * std::tanh((pblh - 100.0) / 100.0) + 0.5;
  const double cp = 400.0 * (1.0 - wt) + 700.0 * wt;
  double ps = ust * cp; int b1 = ps > 10.0; if (!b1) ps = 10.0;
  int b2 = ps > pblh - 200.0; if (!b2) ps = pblh - 200.0;
  int b3 = ps < pblh + 200.0; if (!b3) ps = pblh + 200.0;
  wt = 0.5 * std::tanh((pblh - 200.0) / 170.0) + 0.5;
  pblh = ps * (1.0 - wt) + pblh * wt;
  if (sig) { sig[0] = kthv; sig[1] = kc; sig[2] = fb; sig[3] = capped; sig[4] = floor6; sig[5] = b1; sig[6] = b2; sig[7] = b3; }
  return pblh;
}
struct In { double dz[MAX + 1], sqv[MAX + 1], sqc[MAX + 1], sqi[MAX + 1], th[MAX + 1], ust, land; };
static double height(const In& c, int* sig) {
  const double p608 = double(ess_const::p608);
  double zw[MAX + 2], thv[MAX + 1]; zw[1] = 0.0;
  for (int k = 1; k <= N; ++k) { zw[k + 1] = zw[k] + c.dz[k]; thv[k] = c.th[k] * ((1.0 + p608 * c.sqv[k]) - (c.sqc[k] + c.sqi[k])); }
  return native(thv, zw, c.dz, c.ust, c.land, 1, N, sig);
}
int main(int argc, char** argv) {
  const int ncol = argc > 1 ? atoi(argv[1]) : 400; std::mt19937 g(20261115u);
  std::uniform_real_distribution<float> U(0.0f, 1.0f);
  auto Fp = std::make_unique<ColumnFrame<float>>(); auto Bp = std::make_unique<ColumnBars>();
  ColumnFrame<float>& F = *Fp; ColumnBars& B = *Bp;
  int used = 0, skipped = 0, refused = 0, refuse_seen = 0; double worst = 0.0, plant_seen = 0.0, gworst = 0.0; const char* wname = "";
  for (int ic = 0; ic < ncol; ++ic) {
    std::memset(static_cast<void*>(&F), 0, sizeof F); F.n = N; F.dt = 15.f; F.dheat = 1; F.ktop = -1;
    float* x = F.drv; float z = 0.0f;
    const float base = 285.0f + 25.0f * U(g), zi = 60.0f + 2500.0f * U(g) * U(g); const int kind = ic % 3;
    for (int k = 1; k <= N; ++k) {
      const float d = 15.0f + 12.0f * k * (0.5f + U(g)), zm = z + 0.5f * d; z += d;
      float t = base + (kind == 1 ? 0.004f * zm : 0.0f) + (zm > zi ? 0.006f * (zm - zi) + 1.0f * U(g) : 0.0f) + 0.3f * (U(g) - 0.5f);
      if (kind == 2 && zm < 150.0f) t += 2.0f * (150.0f - zm) / 150.0f;
      x[k] = d; x[MAX + k] = 0.002f + 0.012f * U(g) * std::exp(-zm / 2500.0f); x[2 * MAX + k] = U(g) < 0.2f ? 3e-4f * U(g) : 0.0f;
      x[3 * MAX + k] = U(g) < 0.1f ? 1e-4f * U(g) : 0.0f; x[4 * MAX + k] = 0.0f; x[5 * MAX + k] = t;
      x[6 * MAX + k] = std::pow(std::exp(-zm / 8000.0f), 0.2854f); x[7 * MAX + k] = 1.15f * std::exp(-zm / 9000.0f);
    }
    x[8 * MAX] = base + 1.0f; x[8 * MAX + 1] = 2e-5f; x[8 * MAX + 2] = 0.05f; x[8 * MAX + 3] = 0.02f + 0.9f * U(g) * U(g); x[8 * MAX + 4] = U(g) < 0.8f ? 1.0f : 2.0f;
    In c; for (int k = 1; k <= N; ++k) { c.dz[k] = x[k]; c.sqv[k] = x[MAX + k]; c.sqc[k] = x[2 * MAX + k]; c.sqi[k] = x[3 * MAX + k]; c.th[k] = x[5 * MAX + k]; }
    c.ust = x[8 * MAX + 3]; c.land = x[8 * MAX + 4];
    int s0[8]; const double h0 = height(c, s0);
    double ex[5][MAX + 1] = {{0}}, eu = 0.0;
    { // the forward's own height word: the float statements (the mirror, checked bit for bit by code/pblh_test.cpp)
      float zw[MAX + 2], thv[MAX + 1]; zw[1] = 0.0f;
      for (int k = 1; k <= N; ++k) { zw[k + 1] = r6_add(zw[k], x[k]); float a[6], o[7]; for (int q = 0; q < 6; ++q) a[q] = x[(q + 1) * MAX + k]; driver_thermo<float, false>(a, o, nullptr, nullptr); thv[k] = o[6]; }
      const pblhjac::Out o = pblhjac::get_pblh(1, N, [&](int k) { return thv[k]; }, [&](int k) { return zw[k]; }, [&](int k) { return x[k]; }, x[8 * MAX + 3], x[8 * MAX + 4]);
      F.ccfg.pblh1 = o.pblh;
      double tb[MAX + 2] = {0}, zb[MAX + 3] = {0}, db[MAX + 2] = {0}; pblhjac::reverse(o, 1.0, tb, zb, db, eu);
      const double p608 = double(ess_const::p608); double run = 0.0;
      for (int k = N; k >= 1; --k) { run += zb[k + 1]; ex[0][k] = db[k] + run; const double th = x[5 * MAX + k];
        ex[1][k] = tb[k] * th * p608; ex[2][k] = -tb[k] * th; ex[3][k] = -tb[k] * th;
        ex[4][k] = tb[k] * ((1.0 + p608 * double(x[MAX + k])) - (double(x[2 * MAX + k]) + double(x[3 * MAX + k]))); }
      if (std::fabs(double(o.pblh) - h0) > 1e-4 * std::fabs(h0) || (s0[1] && o.kc != s0[1])) { ++skipped; continue; }   // float and double statements at different branches
    }
    auto run = [&](int plant, double out[5][MAX + 1], double& ust) -> bool {
      std::memset(static_cast<void*>(&B), 0, sizeof B); B.pblh = 1.0; F.plant = plant;
      if (!rev_pblh(F, B)) return false;
      if (!rev_driver(F, B)) return false;
      for (int k = 1; k <= N; ++k) { out[0][k] = B.dz[k]; out[1][k] = B.sqv[k]; out[2][k] = B.sqc[k]; out[3][k] = B.sqi[k]; out[4][k] = B.th[k]; }
      ust = B.ust; return true;
    };
    double an[5][MAX + 1], au;
    if (!run(0, an, au)) { ++refused; continue; }        // the float mirror and the double transcription took different branches
    { double ge[6] = {0}, gs[6] = {0};
      for (int f = 0; f < 5; ++f) for (int k = 1; k <= N; ++k) { ge[f] = std::fmax(ge[f], std::fabs(an[f][k] - ex[f][k])); gs[f] = std::fmax(gs[f], std::fabs(ex[f][k])); }
      ge[5] = std::fabs(au - eu); gs[5] = std::fabs(eu);
      for (int f = 0; f < 6; ++f) if (gs[f] > 0) gworst = std::fmax(gworst, ge[f] / gs[f]); }
    bool ok = true; double err[6] = {0}, sc[6] = {0};
    auto fdq = [&](double* w, double h, double a, int f) {
      const double w0 = *w; int sp[8], sm[8];
      *w = w0 + h; const double fp = height(c, sp); *w = w0 - h; const double fm = height(c, sm); *w = w0;
      if (std::memcmp(sp, s0, sizeof s0) || std::memcmp(sm, s0, sizeof s0)) { ok = false; return; }
      const double fd = (fp - fm) / (2.0 * h); err[f] = std::fmax(err[f], std::fabs(fd - a)); sc[f] = std::fmax(sc[f], std::fabs(fd));
    };
    for (int k = 1; k <= N && ok; ++k) { fdq(&c.dz[k], 1e-3, an[0][k], 0); fdq(&c.sqv[k], 1e-8, an[1][k], 1); fdq(&c.sqc[k], 1e-8, an[2][k], 2);
                                         fdq(&c.sqi[k], 1e-8, an[3][k], 3); fdq(&c.th[k], 1e-5, an[4][k], 4); }
    if (ok) fdq(&c.ust, 1e-6, au, 5);
    if (!ok) { ++skipped; continue; }
    ++used; static const char* nm[6] = {"dz", "sqv", "sqc", "sqi", "th", "ust"};
    for (int f = 0; f < 6; ++f) { const double r = err[f] / std::fmax(sc[f], 1e-30); if (sc[f] > 0 && r > worst) { worst = r; wname = nm[f]; } }
    double pl[5][MAX + 1], pu; if (run(90, pl, pu)) for (int k = 1; k <= N; ++k) plant_seen = std::fmax(plant_seen, std::fabs(pl[4][k] - an[4][k]) / std::fmax(sc[4], 1e-30));
    F.plant = 0; F.ccfg.pblh1 = float(h0) * 1.01f; std::memset(static_cast<void*>(&B), 0, sizeof B); B.pblh = 1.0; if (!rev_pblh(F, B)) ++refuse_seen;
  }
  printf("columns %d: checked %d (skipped at a branch %d, refused %d); glue: worst |bar - partials by hand| / largest of its field %.3e; worst |bar - difference| / largest slope of its field %.3e (%s); plant x1.001 seen at %.3e; wrong height refused %d of %d\n",
         ncol, used, skipped, refused, gworst, worst, wname, plant_seen, refuse_seen, used);
  const bool pass = used > ncol / 2 && gworst < 1e-6 && worst < 5e-3 && plant_seen > 1e-5 && refuse_seen == used && refused < ncol / 20;
  printf("%s\n", pass ? "PASS" : "FAIL"); return pass ? 0 : 1;
}
