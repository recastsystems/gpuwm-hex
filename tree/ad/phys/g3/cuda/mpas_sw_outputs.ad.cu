// SPDX-License-Identifier: Apache-2.0
// g3-radland: mpas_sw_outputs (rrtmg_sw_v842.py:3414, captured .cu 3c499c7b; module_ra_rrtmg_sw.F:10600-10635,
// mpas_atmphys_driver_radiation_sw.F:714-794). One thread per column (the forward geometry), so every input word is
// read by one thread and the AD needs no atomics. S0 = cosine always; night columns (cosine <= 0, frozen branch)
// return with S1..S13 and the tendency untouched. Day: S1..S11, S13 are flux words and differences, S12 = S11/cosine,
// tendency[k] = heat[k]/86400/exner[k]. Partials float64 from the float32 words read. Refusal slot 0: aliasing.
#include "sw.cu"
#include "g3_common.cuh"
#define G3V(a, k) ((size_t)c * nlev + (k))
#define G3S(i) ((size_t)(i) * nc + c)
extern "C" __global__ void mpas_sw_outputs__tl(int nc, int nz, const float* __restrict__ cosine, const float* __restrict__ exner,
    const float* __restrict__ direct, const float* __restrict__ heat,
    const float* d_cos, const float* d_ex, const float* d_up, const float* d_dn, const float* d_upc, const float* d_dnc,
    const float* d_dir, const float* d_heat, float* d_s, float* d_t, int* failure) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    const void* in[8] = {d_cos, d_ex, d_up, d_dn, d_upc, d_dnc, d_dir, d_heat};
    for (int k = 0; k < 8; ++k) if (in[k] == d_s || in[k] == d_t) g3_fail(failure, 0);
    int nlev = nz + 2;
    d_s[G3S(0)] = d_cos[c];
    float cs = cosine[c];
    if (cs <= 0.f) return;
    const size_t b0 = G3V(x, 0), bt = G3V(x, nlev - 1);
    d_s[G3S(1)] = g3_st(G3D(d_dn[b0]) - G3D(d_up[b0]));
    d_s[G3S(2)] = g3_st((G3D(d_dn[bt]) - G3D(d_up[bt])) - (G3D(d_dnc[bt]) - G3D(d_upc[bt])));
    d_s[G3S(3)] = d_up[bt]; d_s[G3S(4)] = d_upc[bt]; d_s[G3S(5)] = d_dn[bt]; d_s[G3S(6)] = d_dnc[bt];
    d_s[G3S(7)] = d_up[b0]; d_s[G3S(8)] = d_upc[b0]; d_s[G3S(9)] = d_dn[b0]; d_s[G3S(10)] = d_dnc[b0];
    d_s[G3S(11)] = d_dir[b0];
    double s11 = direct[b0], co = cs;
    d_s[G3S(12)] = g3_st(G3D(d_dir[b0]) / co - s11 / (co * co) * G3D(d_cos[c]));
    d_s[G3S(13)] = g3_st(G3D(d_dn[b0]) - G3D(d_dir[b0]));
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)c * (nz + 1) + k, e = (size_t)c * nz + k;
        double ex = exner[e], hv = G3D(heat[h]) / 86400.0;
        d_t[e] = g3_st(G3D(d_heat[h]) / 86400.0 / ex - hv / (ex * ex) * G3D(d_ex[e]));
    }
}
extern "C" __global__ void mpas_sw_outputs__ad(int nc, int nz, const float* __restrict__ cosine, const float* __restrict__ exner,
    const float* __restrict__ direct, const float* __restrict__ heat,
    float* b_cos, float* b_ex, float* b_up, float* b_dn, float* b_upc, float* b_dnc, float* b_dir, float* b_heat,
    float* b_s, float* b_t, int* failure) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    const void* in[8] = {b_cos, b_ex, b_up, b_dn, b_upc, b_dnc, b_dir, b_heat};
    for (int k = 0; k < 8; ++k) if (in[k] == b_s || in[k] == b_t) { g3_fail(failure, 0); return; }
    int nlev = nz + 2;
    double gcos = b_s[G3S(0)]; b_s[G3S(0)] = 0.0f;
    float cs = cosine[c];
    if (cs <= 0.f) { g3_acc(b_cos + c, gcos); return; }
    double g[14];
    for (int i = 1; i < 14; ++i) { g[i] = b_s[G3S(i)]; b_s[G3S(i)] = 0.0f; }
    const size_t b0 = G3V(x, 0), bt = G3V(x, nlev - 1);
    double s11 = direct[b0], co = cs;
    gcos += -s11 / (co * co) * g[12];
    double gdn0 = g[1] + g[9] + g[13], gup0 = -g[1] + g[7], gupc0 = g[8], gdnc0 = g[10];
    double gdir0 = g[11] + g[12] / co - g[13];
    double gdnt = g[2] + g[5], gupt = -g[2] + g[3], gdnct = -g[2] + g[6], gupct = g[2] + g[4];
    g3_acc(b_cos + c, gcos);
    if (bt == b0) { gdn0 += gdnt; gup0 += gupt; gdnc0 += gdnct; gupc0 += gupct; }
    else { g3_acc(b_dn + bt, gdnt); g3_acc(b_up + bt, gupt); g3_acc(b_dnc + bt, gdnct); g3_acc(b_upc + bt, gupct); }
    g3_acc(b_dn + b0, gdn0); g3_acc(b_up + b0, gup0); g3_acc(b_upc + b0, gupc0); g3_acc(b_dnc + b0, gdnc0);
    g3_acc(b_dir + b0, gdir0);
    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)c * (nz + 1) + k, e = (size_t)c * nz + k;
        double ex = exner[e], hv = G3D(heat[h]) / 86400.0, gt = b_t[e];
        g3_acc(b_heat + h, gt / 86400.0 / ex);
        g3_acc(b_ex + e, -hv / (ex * ex) * gt);
        b_t[e] = 0.0f;
    }
}
