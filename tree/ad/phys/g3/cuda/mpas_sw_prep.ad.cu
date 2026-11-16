// SPDX-License-Identifier: Apache-2.0
// g3-radland: mpas_sw_prep (rrtmg_sw_v842.py:3174, captured .cu 3c499c7b; MPAS module_ra_rrtmg_sw.F:10197-10550), the
// RRTMG SW sounding preparation, one thread per column (the forward geometry), so every input and output word belongs to
// one thread and the AD needs no atomics. Every output word is written as a sum of float64 coefficients times input
// words; the coefficients come from the forward's own float32 words, recomputed in its order (FDIV/FMUL/FSUB, fmaxf,
// mpas_ice_radius, mpas_ozone_interp's `upper` walk). Executed-branch (frozen) treatment of every clamp:
//   fmaxf(qv, 1e-12), fmaxf(q, 0): slope 1 where the variable wins (strictly), else 0;
//   cloud radius fmaxf(2.5, re*1e6) / ice fmaxf(5, ..) / snow fmaxf(10, ..): slope 1e6 where the field wins; the
//   rel <= 2.5 && cf > 0 land/sea defaults are constants; the ice default mpas_ice_radius(t, 75) has slope
//   RETAB[idx+1] - RETAB[idx] in t (frozen table index), 0 where fmaxf(5, .) picks 5;
//   snow above 130 um: res = 130 (0), snowfactor = (130/res)^2 (slope -2*130^2/res^3 d res);
//   denom = fmaxf(.01, cf): slope 1 where cf > .01;
//   ozone: the forward's `upper` (searched from the previous level's) frozen; p < pin[0] and p > pin[top] branches.
// Inactive: pin (climatology pressure levels). Active: fields, surf (lat/lon for coszr), pw, tw, oz.
// Refusals: slot 0 a non-hrrrv5 branch (reqc/reqi/reqs == 0, o3input != 2: the diagnosed-radius and diagnosed-ozone
// paths); slot 1 an output aliasing an input.
#include "sw.cu"
#include "g3_common.cuh"
struct G3Prep {
    // mode 0: TL (out = sum co * in); mode 1: AD (in += co * out, out consumed).
    int mode;
    const float *fi, *sf, *pw, *tw, *oz;  // derivative bases of the inputs (tangent or adjoint), oz already at the column
    float *lay, *il, *hgt, *cos;
    float* in[12]; double co[12]; int n;
    __device__ void begin() { n = 0; }
    __device__ void t(const float* p, double c) { in[n] = (float*)p; co[n] = c; ++n; }
    __device__ void end(float* out) {
        if (mode == 0) { double s = 0.0; for (int i = 0; i < n; ++i) s += co[i] * G3D(*in[i]); *out = g3_st(s); }
        else { double g = *out; *out = 0.0f; for (int i = 0; i < n; ++i) g3_acc(in[i], co[i] * g); }
    }
};
__device__ void g3_prep_col(G3Prep& D, int c, int nc, int nz, int icloud, int o3input, int reqc, int reqi, int reqs, int nozn,
    float xtime, float gmt, float radt, float degrad, float declin, const float* fields, const float* surf, const float* pw,
    const float* tw, const float* pin, const float* oz, int* failure) {
    int nl = nz + 1, nl1 = nl + 1;
    if (!reqc || !reqi || !reqs || o3input != 2) g3_fail(failure, 0);
#define FI(f, k) fields[((size_t)(f) * nc + c) * nz + (k)]
#define FIx(f, k) (D.fi + ((size_t)(f) * nc + c) * nz + (k))
#define SFx(f) (D.sf + (size_t)(f) * nc + c)
#define Lx(f, k) (D.lay + ((size_t)(f) * nc + c) * nl + (k))
#define ILx(f, k) (D.il + ((size_t)(f) * nc + c) * nl1 + (k))
    const size_t cw = (size_t)c * (nz + 1);
    // coszr
    {
        float xt24 = fmodf(FADD(xtime, FMUL(radt, .5f)), 1440.f);
        float local = FADD(FADD(gmt, FDIV(xt24, 60.f)), FDIV(surf[(size_t)1 * nc + c], 15.f));
        float angle = FMUL(FMUL(15.f, FSUB(local, 12.f)), degrad);
        float lat = FMUL(surf[c], degrad);
        double la = lat, an = angle, dc = declin, dg = degrad;
        D.begin(); D.t(SFx(0), dg * (cos(la) * sin(dc) - sin(la) * cos(dc) * cos(an))); D.t(SFx(1), -dg * cos(la) * cos(dc) * sin(an));
        D.end(D.cos + c);
    }
    // interfaces
    for (int k = 0; k <= nz; k++) { D.begin(); D.t(D.pw + cw + k, 0.01); D.end(ILx(0, k)); D.begin(); D.t(D.tw + cw + k, 1.0); D.end(ILx(1, k)); }
    D.begin(); D.end(ILx(0, nl));
    D.begin(); D.t(D.tw + cw + nz, 1.0); D.end(ILx(1, nl));
    const double K = 100.0 * 1000.0 / 9.80616;
    for (int k = 0; k < nl; k++) {
        // p, t
        D.begin(); if (k < nz) D.t(FIx(0, k), 0.01); else D.t(D.pw + cw + nz, 0.005); D.end(Lx(0, k));
        D.begin(); if (k < nz) D.t(FIx(1, k), 1.0); else D.t(D.tw + cw + nz, 1.0); D.end(Lx(1, k));
        { int kq = k < nz ? k : nz - 1; D.begin(); if (FI(4, kq) > 1.e-12f) D.t(FIx(4, kq), 1.607793); D.end(Lx(2, k)); }
        for (int f = 4; f <= 7; ++f) { D.begin(); D.end(Lx(f, k)); }
        // hgt[k] = sum_{j<k} dz_j + .5 dz_{min(k, nz-1)}
        D.begin();
        { float* hk = D.hgt + (size_t)c * nl + k;
          // more than 12 terms: accumulate in chunks (TL: running sum; AD: direct distribution)
          if (D.mode == 0) { double s = 0.0; for (int j = 0; j < k; ++j) s += G3D(*FIx(2, j)); s += 0.5 * G3D(*FIx(2, k < nz ? k : nz - 1)); *hk = g3_st(s); }
          else { double g = *hk; *hk = 0.0f; for (int j = 0; j < k; ++j) g3_acc((float*)FIx(2, j), g); g3_acc((float*)FIx(2, k < nz ? k : nz - 1), 0.5 * g); } }
        if (k == nz) { for (int f = 8; f <= 14; ++f) { D.begin(); D.end(Lx(f, k)); } continue; }
        float t = FI(1, k);
        float cf = icloud > 0 ? FI(8, k) : 0.f;
        float qc = icloud > 0 ? fmaxf(FI(5, k), 0.f) : 0.f;
        float qi = icloud > 0 ? fmaxf(FI(6, k), 0.f) : 0.f;
        float qs = icloud > 0 ? fmaxf(FI(7, k), 0.f) : 0.f;
        bool aqc = icloud > 0 && FI(5, k) > 0.f, aqi = icloud > 0 && FI(6, k) > 0.f, aqs = icloud > 0 && FI(7, k) > 0.f;
        float pdel = FSUB(FDIV(pw[cw + k], 100.f), FDIV(pw[cw + k + 1], 100.f));
        D.begin(); if (icloud > 0) D.t(FIx(8, k), 1.0); D.end(Lx(8, k));
        // radii (hrrrv5: reqc = reqi = reqs = 1)
        float xc = FMUL(FI(9, k), 1.e6f), rel = fmaxf(2.5f, xc);
        if (rel <= 2.5f && cf > 0.f) { if (FSUB(surf[(size_t)4 * nc + c], 1.5f) > 0.f) rel = 10.5f; else if (FSUB(surf[(size_t)4 * nc + c], 1.5f) < 0.f) rel = 7.5f; }
        D.begin(); if (xc > 2.5f) D.t(FIx(9, k), 1.e6); D.end(Lx(13, k));
        float xi = FMUL(FI(10, k), 1.e6f);
        D.begin();
        if (xi > 5.f) D.t(FIx(10, k), 1.e6);
        else if (cf > 0.f) {
            float r = mpas_ice_radius(t, 75);
            if (r > 5.f) { int idx = max(1, min((int)FSUB(t, 179.f), 75)) - 1; D.t(FIx(1, k), G3D(MPAS_RETAB[idx + 1]) - G3D(MPAS_RETAB[idx])); }
        }
        D.end(Lx(12, k));
        float xs = FMUL(FI(11, k), 1.e6f), res = fmaxf(10.f, xs);
        bool ares = xs > 10.f, big = res > 130.f;
        float snowfactor = 1.f; double dsf = 0.0;   // d snowfactor / d FI11 (only when big)
        if (big) { snowfactor = FMUL(FDIV(130.f, res), FDIV(130.f, res)); dsf = -2.0 * 130.0 * 130.0 / (G3D(res) * res * res) * 1.e6; }
        D.begin(); if (ares && !big) D.t(FIx(11, k), 1.e6); D.end(Lx(14, k));
        float denom = fmaxf(.01f, cf); bool aden = cf > .01f;
        double dn = denom;
        // water paths: L10 = K qc pdel / denom, L9 = K qi pdel / denom, L11 = sf K qs pdel / denom
        float L10 = FDIV(FMUL(FDIV(FMUL(FMUL(qc, pdel), 100.f), 9.80616f), 1000.f), denom);
        float L9 = FDIV(FMUL(FDIV(FMUL(FMUL(qi, pdel), 100.f), 9.80616f), 1000.f), denom);
        float sn = FMUL(FDIV(FMUL(FMUL(qs, pdel), 100.f), 9.80616f), 1000.f);
        float L11 = FDIV(FMUL(snowfactor, sn), denom);
        const float* p0 = D.pw + cw + k; const float* p1 = D.pw + cw + k + 1;
        D.begin(); if (aqc) D.t(FIx(5, k), K * pdel / dn);
        if (qc != 0.f) { D.t(p0, K * qc / dn * 0.01); D.t(p1, -K * qc / dn * 0.01); }
        if (aden) D.t(FIx(8, k), -G3D(L10) / dn); D.end(Lx(10, k));
        D.begin(); if (aqi) D.t(FIx(6, k), K * pdel / dn);
        if (qi != 0.f) { D.t(p0, K * qi / dn * 0.01); D.t(p1, -K * qi / dn * 0.01); }
        if (aden) D.t(FIx(8, k), -G3D(L9) / dn); D.end(Lx(9, k));
        D.begin(); if (aqs) D.t(FIx(7, k), G3D(snowfactor) * K * pdel / dn);
        if (qs != 0.f) { D.t(p0, G3D(snowfactor) * K * qs / dn * 0.01); D.t(p1, -G3D(snowfactor) * K * qs / dn * 0.01); }
        if (big && ares) D.t(FIx(11, k), dsf * G3D(sn) / dn);
        if (aden) D.t(FIx(8, k), -G3D(L11) / dn); D.end(Lx(11, k));
    }
    // ozone (o3input == 2): the forward's walk, k from the top down, `upper` carried
    const float* o3 = nozn < 2 ? oz + (size_t)c * 3 + 1 : oz + (size_t)c * nozn;
    const float* do3 = nozn < 2 ? D.oz + (size_t)c * 3 + 1 : D.oz + (size_t)c * nozn;
    int upper = 0;
    for (int k = nl - 1; k >= 0; k--) {
        float p = k < nz ? FDIV(FI(0, k), 100.f) : FMUL(.5f, FDIV(pw[cw + nz], 100.f));
        double dpdx = k < nz ? 0.01 : 0.005; const float* dpx = k < nz ? FIx(0, k) : D.pw + cw + nz;
        D.begin();
        if (p < pin[0]) { D.t(do3, G3D(p) / G3D(pin[0])); D.t(dpx, G3D(o3[0]) / G3D(pin[0]) * dpdx); }
        else if (p > pin[nozn - 1]) { D.t(do3 + nozn - 1, 1.0); }
        else {
            for (int j = upper; j < nozn - 1; j++) if (pin[j] < p && p <= pin[j + 1]) { upper = j; break; }
            float dpu = FSUB(p, pin[upper]), dpl = FSUB(pin[upper + 1], p), den = FADD(dpl, dpu);
            D.t(do3 + upper, G3D(dpl) / G3D(den)); D.t(do3 + upper + 1, G3D(dpu) / G3D(den));
            D.t(dpx, (G3D(o3[upper + 1]) - G3D(o3[upper])) / G3D(den) * dpdx);
        }
        D.end(Lx(3, k));
    }
#undef FI
#undef FIx
#undef SFx
#undef Lx
#undef ILx
}
#define G3_PREP_PARAMS int nc, int nz, int icloud, int o3input, int reqc, int reqi, int reqs, int nozn, float xtime, float gmt, \
    float radt, float degrad, float declin, const float* __restrict__ fields, const float* __restrict__ surf, \
    const float* __restrict__ pw, const float* __restrict__ tw, const float* __restrict__ pin, const float* __restrict__ oz
#define G3_PREP_CALL nc, nz, icloud, o3input, reqc, reqi, reqs, nozn, xtime, gmt, radt, degrad, declin, fields, surf, pw, tw, pin, oz, failure
extern "C" __global__ void mpas_sw_prep__tl(G3_PREP_PARAMS, const float* d_fields, const float* d_surf, const float* d_pw,
    const float* d_tw, const float* d_oz, float* d_layers, float* d_interfaces, float* d_hgt, float* d_coszr, int* failure) {
    int c = blockDim.x * blockIdx.x + threadIdx.x; if (c >= nc) return;
    const void* in[5] = {d_fields, d_surf, d_pw, d_tw, d_oz}; const void* out[4] = {d_layers, d_interfaces, d_hgt, d_coszr};
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 4; ++b) if (in[a] == out[b]) { g3_fail(failure, 1); return; }
    G3Prep D; D.mode = 0; D.fi = d_fields; D.sf = d_surf; D.pw = d_pw; D.tw = d_tw; D.oz = d_oz;
    D.lay = d_layers; D.il = d_interfaces; D.hgt = d_hgt; D.cos = d_coszr;
    g3_prep_col(D, c, G3_PREP_CALL);
}
extern "C" __global__ void mpas_sw_prep__ad(G3_PREP_PARAMS, float* b_fields, float* b_surf, float* b_pw,
    float* b_tw, float* b_oz, float* b_layers, float* b_interfaces, float* b_hgt, float* b_coszr, int* failure) {
    int c = blockDim.x * blockIdx.x + threadIdx.x; if (c >= nc) return;
    const void* in[5] = {b_fields, b_surf, b_pw, b_tw, b_oz}; const void* out[4] = {b_layers, b_interfaces, b_hgt, b_coszr};
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 4; ++b) if (in[a] == out[b]) { g3_fail(failure, 1); return; }
    G3Prep D; D.mode = 1; D.fi = b_fields; D.sf = b_surf; D.pw = b_pw; D.tw = b_tw; D.oz = b_oz;
    D.lay = b_layers; D.il = b_interfaces; D.hgt = b_hgt; D.cos = b_coszr;
    g3_prep_col(D, c, G3_PREP_CALL);
}
