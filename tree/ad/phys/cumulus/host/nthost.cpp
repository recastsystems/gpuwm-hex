// SPDX-License-Identifier: Apache-2.0
// Processor proofs of the New Tiedtke derivative entries on captured launches.
//   nthost BUNDLE fwd           the forward (host form) on the pre-state against the captured post-state, every word
//   nthost BUNDLE dot [SEED]    tangent and taped reverse on the pre-state:
//                               (1) primal words of both entries against the captured post-state (bitwise);
//                               (2) tape census (entries per column);
//                               (3) the transpose check <J dx, y> = <dx, J^T y> per column, whole and with the
//                                   identity part of the in/out arrays removed (K = J - I), and in total.
// Bundle (mkbundle.py): "NTB1", kernel name[64], grid.x, block.x, nslot; per slot: kind (0 scalar, 1 pointer),
// dtype (0 f32, 1 i32), nwords, raw scalar word; then for pointers pre words and post words.
#include "hostshim.h"
nt_dim3 threadIdx, blockIdx, blockDim, gridDim;
#include "nt-forward.cu"
#include "nt_ad.cuh"
namespace tl {
typedef AF R;
#include "nt_R.inc"
}
namespace adr {
typedef AR R;
#include "nt_R.inc"
}
#define NT_DT double
#include "nt_entries.inc"
#include <vector>
#include <string>
#include <random>

struct Slot { int kind, dtype; long long n; unsigned raw; std::vector<unsigned> pre, post, work; void *ext = nullptr; };
struct Slots {
  unsigned gx = 0, bx = 0; std::vector<Slot> s;
  int i(int k) { int v; std::memcpy(&v, &s[k].raw, 4); return v; }
  float f(int k) { float v; std::memcpy(&v, &s[k].raw, 4); return v; }
  void *p(int k) { return s[k].ext ? s[k].ext : (void *)s[k].work.data(); }
};
struct Deriv {
  int W = 0, nf = 0; long long rows[64]; int written[64], slot[64]; const char *fname[64];
  size_t tl_ws = 0, ad_ws0 = 0, ad_ws_per_cap = 0;
  double *tin[64], *tout[64], *bin[64], *acc[64];
  int cap = 0; unsigned char *ws = nullptr; int fail[4] = {0, 0, 0, 0};
};
#include "nt_host.inc"
// Library form (the processor replica of hex's pipeline, host/replica.py): run kernel NAME in mode 1 (forward) on
// caller-owned arrays: ptrs[k] for pointer slots (null for scalars), raw[k] the scalar word.
extern "C" int nt_call(const char *name, int nslots, void **ptrs, const unsigned *raw, unsigned gx, unsigned bx) {
  int (*fn)(Slots &, int, Deriv &) = nullptr;
  for (auto &h : HK) if (std::strcmp(name, h.name) == 0) fn = h.fn;
  if (!fn) return -1;
  Slots S; S.gx = gx; S.bx = bx; S.s.resize(nslots);
  for (int k = 0; k < nslots; ++k) { S.s[k].kind = ptrs[k] ? 1 : 0; S.s[k].ext = ptrs[k]; S.s[k].raw = raw[k]; }
  Deriv D; fn(S, 0, D); fn(S, 1, D); return 0;
}

static Slots load(const char *path, std::string &name) {
  FILE *f = std::fopen(path, "rb"); if (!f) { std::perror(path); std::exit(2); }
  char magic[4], nm[64]; unsigned gx, bx; int ns;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "NTB1", 4)) { std::fprintf(stderr, "bad bundle\n"); std::exit(2); }
  std::fread(nm, 1, 64, f); std::fread(&gx, 4, 1, f); std::fread(&bx, 4, 1, f); std::fread(&ns, 4, 1, f);
  name = std::string(nm, strnlen(nm, 64));
  Slots S; S.gx = gx; S.bx = bx; S.s.resize(ns);
  for (auto &x : S.s) {
    std::fread(&x.kind, 4, 1, f); std::fread(&x.dtype, 4, 1, f); std::fread(&x.n, 8, 1, f); std::fread(&x.raw, 4, 1, f);
    if (x.kind == 1) {
      x.pre.resize(x.n); x.post.resize(x.n);
      std::fread(x.pre.data(), 4, x.n, f); std::fread(x.post.data(), 4, x.n, f);
    }
  }
  std::fclose(f); return S;
}
static void reset(Slots &S) { for (auto &x : S.s) if (x.kind == 1) x.work = x.pre; }

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: nthost BUNDLE fwd|dot [SEED]\n"); return 2; }
  std::string name; Slots S = load(argv[1], name);
  int (*fn)(Slots &, int, Deriv &) = nullptr;
  for (auto &h : HK) if (name == h.name) fn = h.fn;
  if (!fn) { std::fprintf(stderr, "no dispatcher for %s\n", name.c_str()); return 2; }
  const std::string mode = argv[2];
  Deriv D; reset(S); fn(S, 0, D);
  if (mode == "fwd") {
    fn(S, 1, D);
    long long diff = 0, words = 0;
    for (size_t k = 0; k < S.s.size(); ++k) {
      auto &x = S.s[k]; if (x.kind != 1) continue;
      long long d = 0; for (long long w = 0; w < x.n; ++w) d += x.work[w] != x.post[w];
      words += x.n; diff += d;
      if (d) std::printf("  slot %zu differs in %lld of %lld words\n", k, d, x.n);
    }
    std::printf("FWD %s: %lld of %lld words differ (W %d)\n", name.c_str(), diff, words, D.W);
    return diff ? 1 : 0;
  }
  if (mode == "fd") {
    // tangent against centred differences of the forward itself, along a relative direction dx_w = u_w |x_w|, on the
    // columns whose integer outputs (cloud type, levels, labels, flags) are the base run's in both perturbed runs.
    const unsigned seed = argc > 3 ? (unsigned)std::atoi(argv[3]) : 1u;
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(-1.0, 1.0);
    const int W = D.W, nf = D.nf;
    std::vector<std::vector<double>> dx(nf), jdx(nf), zero(nf), dum(nf);
    for (int j = 0; j < nf; ++j) {
      const long long n = D.rows[j] * W; auto &x = S.s[D.slot[j]];
      dx[j].resize(n); jdx[j].assign(n, 0.0);
      for (long long w = 0; w < n; ++w) { float v; std::memcpy(&v, &x.pre[w], 4); dx[j][w] = std::isfinite(v) ? U(rng) * std::fabs((double)v) : 0.0; }
      D.tin[j] = dx[j].data(); D.tout[j] = jdx[j].data();
    }
    std::vector<unsigned char> wsT(D.tl_ws + 64); D.ws = wsT.data(); reset(S); fn(S, 2, D);
    auto run = [&](double e, std::vector<std::vector<float>> &outF, std::vector<std::vector<unsigned>> &outI) {
      reset(S);
      for (int j = 0; j < nf; ++j) { auto &x = S.s[D.slot[j]]; const long long n = D.rows[j] * W;
        for (long long w = 0; w < n; ++w) { float v; std::memcpy(&v, &x.pre[w], 4); float p = (float)((double)v + e * dx[j][w]); std::memcpy(&x.work[w], &p, 4); } }
      fn(S, 1, D);
      outF.assign(nf, {}); for (int j = 0; j < nf; ++j) if (D.written[j]) { auto &x = S.s[D.slot[j]]; outF[j].resize(x.n); std::memcpy(outF[j].data(), x.work.data(), 4 * x.n); }
      outI.clear(); for (auto &x : S.s) if (x.kind == 1 && x.dtype == 1) outI.push_back(x.work);
    };
    std::vector<std::vector<float>> b0F; std::vector<std::vector<unsigned>> b0I; run(0.0, b0F, b0I);
    for (double e : {1e-2, 1e-3, 1e-4, 1e-5}) {
      std::vector<std::vector<float>> pF, mF; std::vector<std::vector<unsigned>> pI, mI; run(e, pF, pI); run(-e, mF, mI);
      std::vector<double> rel; int stable = 0, live = 0, wcol = -1; double wrel = -1;
      for (int c = 0; c < W; ++c) {
        bool same = true;   // integer words of this column (every int pointer slot, column layout r*W + c where it has W columns)
        for (size_t q = 0; q < b0I.size() && same; ++q) { const size_t n = b0I[q].size(); if (n % W) continue;
          for (size_t w = c; w < n; w += W) if (pI[q][w] != b0I[q][w] || mI[q][w] != b0I[q][w]) { same = false; break; } }
        if (!same) continue; ++stable;
        double num = 0, den = 0;
        for (int j = 0; j < nf; ++j) if (D.written[j]) for (long long r = 0; r < D.rows[j]; ++r) { const size_t w = (size_t)r * W + c;
          // the change the launch makes, not the pass-through: (out - in) of each perturbed run, in the words it used
          float v; std::memcpy(&v, &S.s[D.slot[j]].pre[w], 4);
          const float pin = (float)((double)v + e * dx[j][w]), min_ = (float)((double)v - e * dx[j][w]);
          const double fdK = (((double)pF[j][w] - pin) - ((double)mF[j][w] - min_)) / (2 * e), tK = jdx[j][w] - dx[j][w];
          if (!std::isfinite(fdK) || !std::isfinite(tK)) continue;
          if (getenv("NT_FD_COL") && atoi(getenv("NT_FD_COL")) == c && std::fabs(fdK - tK) > 1e-3 * (std::fabs(tK) + std::fabs(fdK)) && std::fabs(fdK - tK) > 0)
            std::printf("  col %d %s[%lld]: FD %.6e TL %.6e (in %.6e, dx %.3e)\n", c, D.fname[j], r, fdK, tK, (double)v, dx[j][w]);
          num += (fdK - tK) * (fdK - tK); den += tK * tK; }
        if (den > 0) { ++live; const double rr = std::sqrt(num / den); rel.push_back(rr); if (rr > wrel) { wrel = rr; wcol = c; } }
      }
      std::sort(rel.begin(), rel.end());
      auto q = [&](double f) { return rel.empty() ? 0.0 : rel[(size_t)(f * (rel.size() - 1))]; };
      std::printf("FD %s seed %u eps %.0e: integer-stable columns %d of %d, with a change %d; |FD - TL| / |TL - I| median %.2e p90 %.2e worst %.2e (col %d)\n",
                  name.c_str(), seed, e, stable, W, live, q(0.5), q(0.9), q(1.0), wcol);
    }
    return 0;
  }
  const unsigned seed = argc > 3 ? (unsigned)std::atoi(argv[3]) : 1u;
  std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int W = D.W, nf = D.nf;
  std::vector<std::vector<double>> dx(nf), jdx(nf), y(nf), jty(nf);
  for (int j = 0; j < nf; ++j) {
    const long long n = D.rows[j] * W; auto &x = S.s[D.slot[j]];
    double ss = 0; long long m = 0;
    for (long long w = 0; w < n; ++w) { float v; std::memcpy(&v, &x.pre[w], 4); if (std::isfinite(v)) { ss += (double)v * v; ++m; } }
    const double sc = (m && ss > 0) ? std::sqrt(ss / m) : 1.0;
    dx[j].resize(n); for (auto &v : dx[j]) v = U(rng) * sc;
    jdx[j].assign(n, 0.0); jty[j].assign(n, 0.0); y[j].assign(n, 0.0);
    D.tin[j] = dx[j].data(); D.tout[j] = jdx[j].data(); D.acc[j] = jty[j].data(); D.bin[j] = y[j].data();
  }
  // ---- tangent
  std::vector<unsigned char> wsT(D.tl_ws + 64); D.ws = wsT.data(); reset(S); fn(S, 2, D);
  const int tlfail = D.fail[0];
  // primal words of the tangent entry's shadows against the captured post-state
  long long pdiffT = 0, pwords = 0; { size_t off = 0;
    for (int j = 0; j < nf; ++j) { const long long n = D.rows[j] * W; const AF *sh = (const AF *)(D.ws + off);
      auto &x = S.s[D.slot[j]];
      if (D.written[j]) for (long long w = 0; w < n; ++w) { unsigned u; std::memcpy(&u, &sh[w].v, 4); pdiffT += u != x.post[w]; ++pwords; }
      off += 16 * n; } }
  // ---- adjoint seeds on the written arrays, scaled per array by its tangent's size
  for (int j = 0; j < nf; ++j) if (D.written[j]) {
    double ss = 0; for (double v : jdx[j]) ss += v * v;
    const double sc = ss > 0 ? 1.0 / std::sqrt(ss / jdx[j].size()) : 1.0;
    for (auto &v : y[j]) v = U(rng) * sc;
  }
  // ---- tape census (capacity 0: every entry counted, none stored)
  std::vector<unsigned char> ws0(D.ad_ws0 + 64); D.ws = ws0.data(); D.cap = 0; D.fail[0] = 0; reset(S); fn(S, 3, D);
  long long nslotw = 0; for (int j = 0; j < nf; ++j) nslotw += D.rows[j];
  const int *cnt = (const int *)(ws0.data() + 16 * (size_t)nslotw * W);
  long long tmax = 0, tsum = 0; int tcol = 0;
  for (int t = 0; t < W; ++t) { tmax = std::max<long long>(tmax, cnt[t]); tsum += cnt[t]; tcol += cnt[t] > 0; }
  // ---- taped reverse
  for (int j = 0; j < nf; ++j) std::fill(jty[j].begin(), jty[j].end(), 0.0);
  D.cap = (int)tmax + 1; std::vector<unsigned char> wsA(D.ad_ws0 + D.ad_ws_per_cap * D.cap + 64);
  D.ws = wsA.data(); D.fail[0] = 0; reset(S); fn(S, 3, D);
  const int adfail = D.fail[0];
  long long pdiffA = 0; { size_t off = 0;
    for (int j = 0; j < nf; ++j) { const long long n = D.rows[j] * W; const AR *sh = (const AR *)(D.ws + off);
      auto &x = S.s[D.slot[j]];
      if (D.written[j]) for (long long w = 0; w < n; ++w) { unsigned u; std::memcpy(&u, &sh[w].v, 4); pdiffA += u != x.post[w]; }
      off += 8 * n; } }
  // ---- transpose check per column
  double Lt = 0, Rt = 0, At = 0, LKt = 0, RKt = 0, AKt = 0, worst = 0, worstK = 0; int wc = -1, wck = -1, over = 0, overK = 0, live = 0;
  for (int c = 0; c < W; ++c) {
    double L = 0, R = 0, A = 0, LK = 0, RK = 0, AK = 0;
    for (int j = 0; j < nf; ++j) for (long long r = 0; r < D.rows[j]; ++r) {
      const size_t w = (size_t)r * W + c;
      const double a = jdx[j][w] * y[j][w], b = dx[j][w] * jty[j][w];
      L += a; R += b; A += std::fabs(a) + std::fabs(b);
      if (D.written[j]) {   // K = J - I on the in/out arrays: remove the pass-through term dx . y
        const double id = dx[j][w] * y[j][w];
        LK += a - id; RK += b - id; AK += std::fabs(a - id) + std::fabs(b - id);
      } else { RK += b; AK += std::fabs(b); }
    }
    Lt += L; Rt += R; At += A; LKt += LK; RKt += RK; AKt += AK;
    const double rel = A > 0 ? std::fabs(L - R) / A : 0.0, relK = AK > 0 ? std::fabs(LK - RK) / AK : 0.0;
    if (rel > worst) { worst = rel; wc = c; }
    if (relK > worstK) { worstK = relK; wck = c; }
    over += rel > 1e-10; overK += relK > 1e-10; live += AK > 0;
  }
  std::printf("DOT %s seed %u W %d: primal words differ TL %lld AD %lld of %lld; tape max %lld mean %.1f over %d columns;"
              " column worst rel %.3e (col %d), K-only worst %.3e (col %d), columns over 1e-10: %d / K %d, K-live columns %d;"
              " total rel %.3e, K total rel %.3e; failure TL %d AD %d\n",
              name.c_str(), seed, W, pdiffT, pdiffA, pwords, tmax, tcol ? (double)tsum / tcol : 0.0, tcol,
              worst, wc, worstK, wck, over, overK, live, At > 0 ? std::fabs(Lt - Rt) / At : 0.0,
              AKt > 0 ? std::fabs(LKt - RKt) / AKt : 0.0, tlfail, adfail);
  return (pdiffT || pdiffA || overK || tlfail || adfail) ? 1 : 0;
}
