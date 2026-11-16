// SPDX-License-Identifier: Apache-2.0
// g3-radland: cloud_from (radiation_rrfs.py:57, captured .cu 4a88150e). qv/qc/qi/qs = copies of v/c/ice/snow; cf from
// the MYNN BL cloud fraction (frozen branch: mynn && step > 0 && bl > .001 -> d cf = d bl, else 0; the step-0
// incidence 0/1 is a constant); dx = len / density^.25 * .001 per column (d dx = -.25 dx / density d density);
// xland = land. Refusals: slot 0 gfl != 0 (GF convective fraction: cu pointer plus the min(1, frac + cu) clamp; not
// hrrrv5); slot 1 an output aliasing an input (the AD would zero a live cotangent).
#include "cloud.cu"
#include "g3_common.cuh"
__device__ __forceinline__ bool g3_cf_live(int mynn, int step, float blv) { return mynn && step > 0 && blv > .001f; }
extern "C" __global__ void cloud_from__tl(int nc, int nz, float len, int mynn, int step, int gfl,
    const float* __restrict__ density, const float* __restrict__ bl,
    const float* d_density, const float* d_land, const float* d_v, const float* d_c, const float* d_ice, const float* d_snow,
    const float* d_bl, float* d_dx, float* d_xland, float* d_qv, float* d_qc, float* d_qi, float* d_qs, float* d_cf, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nc * nz) return;
    if (gfl) g3_fail(failure, 0);
    d_qv[i] = d_v[i]; d_qc[i] = d_c[i]; d_qi[i] = d_ice[i]; d_qs[i] = d_snow[i];
    d_cf[i] = g3_cf_live(mynn, step, bl[i]) ? d_bl[i] : 0.0f;
    if (i % nz == 0) {
        int col = i / nz; double den = density[col];
        double dxv = G3D(len) / pow(den, 0.25) * 1e-3;
        d_dx[col] = g3_st(-0.25 * dxv / den * G3D(d_density[col]));
        d_xland[col] = d_land[col];
    }
}
extern "C" __global__ void cloud_from__ad(int nc, int nz, float len, int mynn, int step, int gfl,
    const float* __restrict__ density, const float* __restrict__ bl,
    float* b_density, float* b_land, float* b_v, float* b_c, float* b_ice, float* b_snow, float* b_bl,
    float* b_dx, float* b_xland, float* b_qv, float* b_qc, float* b_qi, float* b_qs, float* b_cf, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nc * nz) return;
    if (gfl) g3_fail(failure, 0);
    float* in[5] = {b_v, b_c, b_ice, b_snow, b_bl};
    float* out[5] = {b_qv, b_qc, b_qi, b_qs, b_cf};
    for (int a = 0; a < 5; ++a) for (int o = 0; o < 5; ++o) if (in[a] == out[o]) { g3_fail(failure, 1); return; }
    if (b_dx == b_density || b_xland == b_land) { g3_fail(failure, 1); return; }
    g3_acc(b_v + i, b_qv[i]); b_qv[i] = 0.0f;
    g3_acc(b_c + i, b_qc[i]); b_qc[i] = 0.0f;
    g3_acc(b_ice + i, b_qi[i]); b_qi[i] = 0.0f;
    g3_acc(b_snow + i, b_qs[i]); b_qs[i] = 0.0f;
    if (g3_cf_live(mynn, step, bl[i])) g3_acc(b_bl + i, b_cf[i]);
    b_cf[i] = 0.0f;
    if (i % nz == 0) {
        int col = i / nz; double den = density[col];
        double dxv = G3D(len) / pow(den, 0.25) * 1e-3;
        g3_acc(b_density + col, -0.25 * dxv / den * G3D(b_dx[col])); b_dx[col] = 0.0f;
        g3_acc(b_land + col, b_xland[col]); b_xland[col] = 0.0f;
    }
}
