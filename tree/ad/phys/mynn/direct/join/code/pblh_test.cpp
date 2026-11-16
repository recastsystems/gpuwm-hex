// CPU test of blocks/pblhjac.hpp (the mirror of f_get_pblh and its transpose).
//  1. value: the mirror against a plain transcription of the generated statements, bit for bit;
//  2. derivative: the transposed Jacobian scattered to every input against central differences of a double transcription,
//     over every thv, zw, dz word and ust (so a dependency the six words miss shows), on columns whose branches hold;
//  3. plant: one partial x1.001 must be seen.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include "pblhjac.hpp"
static const int N = 59;
struct Col { float thv[N + 2], zw[N + 2], dz[N + 2], ust, landsea; };
// the generated statements of f_get_pblh, real type T (float: the forward's words; double: the difference quotients)
template<class T> static T native(const T* thv, const T* zw, const T* dz, T ust, T landsea, int kts, int kte, int* sig) {
  int k = kts + 1, kthv = 1, kc = 0, fb = 0; T minthv = (T)9.e9f;
  while (k <= kte && zw[k] <= (T)200) { if (minthv > thv[k]) { minthv = thv[k]; kthv = k; } k = k + 1; }
  const T delt = (landsea - (T)1.5f >= 0) ? (T)0.75f : (T)1.25f;
  T pblh = 0; int capped = 0, floor6 = 0;
  for (k = kts + 1; k <= kte - 1; ++k) {
    if (thv[k] >= minthv + delt) {
      T den = thv[k] - thv[k - 1]; floor6 = !(den > (T)1e-6f); if (floor6) den = (T)1e-6f;
      T fr = (thv[k] - (minthv + delt)) / den; capped = !(fr < (T)1); if (capped) fr = (T)1;
      pblh = zw[k] - dz[k - 1] * fr; kc = k;
    }
    if (k == kte - 1) { pblh = zw[kts + 1]; fb = 1; }
    if (pblh > 0) break;
  }
  T wt = (T)0.5f * std::tanh((pblh - (T)100) / (T)100) + (T)0.5f;
  const T cp = (T)400 * ((T)1 - wt) + (T)700 * wt;
  T ps = ust * cp; int b1 = ps > (T)10; if (!b1) ps = (T)10;
  int b2 = ps > pblh - (T)200; if (!b2) ps = pblh - (T)200;
  int b3 = ps < pblh + (T)200; if (!b3) ps = pblh + (T)200;
  wt = (T)0.5f * std::tanh((pblh - (T)200) / (T)170) + (T)0.5f;
  pblh = ps * ((T)1 - wt) + pblh * wt;
  if (sig) { sig[0] = kthv; sig[1] = kc; sig[2] = fb; sig[3] = capped; sig[4] = floor6; sig[5] = b1; sig[6] = b2; sig[7] = b3; }
  return pblh;
}
static void make(std::mt19937& g, Col& c, int kind) {
  std::uniform_real_distribution<float> U(0.0f, 1.0f);
  float z = 0.0f;
  for (int k = 1; k <= N + 1; ++k) { c.zw[k] = z; const float d = 15.0f + 12.0f * k * (0.5f + U(g)); if (k <= N) c.dz[k] = d; z += d; }
  const float base = 285.0f + 25.0f * U(g), zi = 60.0f + 2500.0f * U(g) * U(g);
  for (int k = 1; k <= N; ++k) {
    const float zm = c.zw[k] + 0.5f * c.dz[k];
    float t = base + (kind == 1 ? 0.004f * zm : 0.0f) + (zm > zi ? 0.006f * (zm - zi) + 1.0f * U(g) : 0.0f) + 0.3f * (U(g) - 0.5f);
    if (kind == 2 && zm < 150.0f) t += 2.0f * (150.0f - zm) / 150.0f;                 // warm near the ground
    c.thv[k] = t;
  }
  if (kind == 3) for (int k = 1; k <= N; ++k) c.thv[k] = base;                          // no crossing: the kte-1 fallback
  c.ust = 0.02f + 0.9f * U(g) * U(g); c.landsea = U(g) < 0.8f ? 1.0f : 2.0f;
}
int main(int argc, char** argv) {
  const int ncol = argc > 1 ? atoi(argv[1]) : 4000; std::mt19937 g(20261115u);
  int vbad = 0, used = 0, skipped = 0, fb = 0, cap = 0, alias = 0, stab[3] = {0, 0, 0}; double worst = 0.0, plant_seen = 0.0;
  for (int ic = 0; ic < ncol; ++ic) {
    Col c; make(g, c, ic % 40 == 39 ? 3 : ic % 3);
    auto TH = [&](int k) { return c.thv[k]; }; auto ZW = [&](int k) { return c.zw[k]; }; auto DZ = [&](int k) { return c.dz[k]; };
    const pblhjac::Out o = pblhjac::get_pblh(1, N, TH, ZW, DZ, c.ust, c.landsea);
    int sg[8]; const float pv = native<float>(c.thv, c.zw, c.dz, c.ust, c.landsea, 1, N, sg);
    if (std::memcmp(&pv, &o.pblh, 4) != 0) { ++vbad; if (vbad < 4) printf("value differs col %d: %.9g mirror %.9g\n", ic, pv, o.pblh); continue; }
    fb += sg[2]; cap += sg[3]; if (o.kmin && o.kmin == o.kc - 1) ++alias; stab[0] += !sg[5]; stab[1] += !sg[6]; stab[2] += !sg[7];
    // gradient by the transpose
    double gth[N + 2] = {0}, gzw[N + 3] = {0}, gdz[N + 2] = {0}, gu = 0.0; pblhjac::reverse(o, 1.0, gth, gzw, gdz, gu);
    // central differences of the double transcription, every input
    double th[N + 2], zw[N + 3], dz[N + 2]; for (int k = 1; k <= N; ++k) { th[k] = c.thv[k]; dz[k] = c.dz[k]; } for (int k = 1; k <= N + 1; ++k) zw[k] = c.zw[k];
    int s0[8]; native<double>(th, zw, dz, (double)c.ust, (double)c.landsea, 1, N, s0);
    bool ok = true; double err = 0.0, scale = 0.0;
    auto fdq = [&](double* x, double h, double an) {
      const double x0 = *x; int sp[8], sm[8];
      *x = x0 + h; const double fp = native<double>(th, zw, dz, (double)c.ust, (double)c.landsea, 1, N, sp);
      *x = x0 - h; const double fm = native<double>(th, zw, dz, (double)c.ust, (double)c.landsea, 1, N, sm); *x = x0;
      if (std::memcmp(sp, s0, sizeof s0) || std::memcmp(sm, s0, sizeof s0)) { ok = false; return; }
      const double fd = (fp - fm) / (2.0 * h); err = std::fmax(err, std::fabs(fd - an)); scale = std::fmax(scale, std::fabs(fd));
    };
    for (int k = 1; k <= N && ok; ++k) fdq(&th[k], 1e-5, gth[k]);
    for (int k = 1; k <= N + 1 && ok; ++k) fdq(&zw[k], 1e-3, gzw[k]);
    for (int k = 1; k <= N && ok; ++k) fdq(&dz[k], 1e-3, gdz[k]);
    if (ok) { double u = c.ust; const double h = 1e-6; int sp[8], sm[8];
      const double fp = native<double>(th, zw, dz, u + h, (double)c.landsea, 1, N, sp), fm = native<double>(th, zw, dz, u - h, (double)c.landsea, 1, N, sm);
      if (std::memcmp(sp, s0, sizeof s0) || std::memcmp(sm, s0, sizeof s0)) ok = false;
      else { const double fd = (fp - fm) / (2.0 * h); err = std::fmax(err, std::fabs(fd - gu)); scale = std::fmax(scale, std::fabs(fd)); } }
    if (!ok) { ++skipped; continue; }
    ++used; const double rel = err / std::fmax(scale, 1e-30); if (rel > worst) worst = rel;
    // plant: the THK partial x1.001
    pblhjac::Out p = o; p.d[pblhjac::THK] *= 1.001; double pth[N + 2] = {0}, pzw[N + 3] = {0}, pdz[N + 2] = {0}, pu = 0.0; pblhjac::reverse(p, 1.0, pth, pzw, pdz, pu);
    plant_seen = std::fmax(plant_seen, std::fabs(pth[o.kc] - gth[o.kc]) / std::fmax(scale, 1e-30));
  }
  printf("columns %d: value differs %d; derivative checked %d (skipped at a branch %d); worst |transpose - difference| / largest slope %.3e; plant x1.001 seen at %.3e\n",
         ncol, vbad, used, skipped, worst, plant_seen);
  printf("covered: fallback %d, fraction capped at 1 %d, minimum level = kc-1 %d, stable height at its floor 10 m %d, at pblh-200 %d, at pblh+200 %d\n", fb, cap, alias, stab[0], stab[1], stab[2]);
  const bool pass = vbad == 0 && used > ncol / 2 && worst < 2e-5 && plant_seen > 1e-5;
  printf("%s\n", pass ? "PASS" : "FAIL"); return pass ? 0 : 1;
}
