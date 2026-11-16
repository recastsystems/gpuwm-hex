// SPDX-License-Identifier: Apache-2.0
// g3-radland: solar (radiation_rrfs.py:43; captured .cu 8a580fc9, the RRFS 44c0849 constants patched in at load).
// coszr = sin(xl) sin(dc) + cos(xl) cos(dc) cos(h), xl = lat*degrad, h = 15 (gmt + xt24/60 + lon/15 - 12) degrad.
// Active: lat, lon (degrees, from the `degrees` launch). Time scalars are host constants. The primal angles are the
// forward's float32 words (its own rrfs_eot and FADD/FDIV order); the partials are float64 trig of those words.
// eot: the hrrrv5 slot passes NULL (captured as a null pointer scalar). A non-null eot is refused (slot 0), as is
// coszr aliasing lat or lon (slot 1).
#include "solar.cu"
#include "g3_common.cuh"
__device__ __forceinline__ void g3_solar_coef(float julian, float xtime, float gmt, float radt, float degrad, float declin,
                                              float lat, float lon, double& a, double& b) {
    float eq = rrfs_eot(julian);
    float xt24 = FADD(fmodf(FADD(xtime, FMUL(radt, .5f)), 1440.f), eq);
    float local = FADD(FADD(gmt, FDIV(xt24, 60.f)), FDIV(lon, 15.f));
    float hrang = FMUL(FMUL(15.f, FSUB(local, 12.f)), degrad);
    float xxlat = FMUL(lat, degrad);
    double xl = xxlat, h = hrang, dc = declin, dg = degrad;
    a = dg * (cos(xl) * sin(dc) - sin(xl) * cos(dc) * cos(h));   // d coszr / d lat
    b = -dg * cos(xl) * cos(dc) * sin(h);                        // d coszr / d lon (d local/d lon = 1/15, times 15 degrad)
}
extern "C" __global__ void solar__tl(int n, float julian, float xtime, float gmt, float radt, float degrad, float declin,
    const float* __restrict__ lat, const float* __restrict__ lon, unsigned long long eot,
    const float* d_lat, const float* d_lon, float* d_coszr, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (eot) g3_fail(failure, 0);
    if (d_coszr == d_lat || d_coszr == d_lon) g3_fail(failure, 1);
    double a, b; g3_solar_coef(julian, xtime, gmt, radt, degrad, declin, lat[i], lon[i], a, b);
    d_coszr[i] = g3_st(a * G3D(d_lat[i]) + b * G3D(d_lon[i]));
}
extern "C" __global__ void solar__ad(int n, float julian, float xtime, float gmt, float radt, float degrad, float declin,
    const float* __restrict__ lat, const float* __restrict__ lon, unsigned long long eot,
    float* b_lat, float* b_lon, float* b_coszr, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (eot) g3_fail(failure, 0);
    if (b_coszr == b_lat || b_coszr == b_lon) { g3_fail(failure, 1); return; }
    double a, b; g3_solar_coef(julian, xtime, gmt, radt, degrad, declin, lat[i], lon[i], a, b);
    double g = b_coszr[i];
    g3_acc(b_lat + i, a * g); g3_acc(b_lon + i, b * g); b_coszr[i] = 0.0f;
}
