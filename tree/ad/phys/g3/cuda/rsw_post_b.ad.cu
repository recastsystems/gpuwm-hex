// SPDX-License-Identifier: Apache-2.0
// g3-radland: rsw_post_b (rrtmg_sw_v842.py:2973, captured .cu 3c499c7b; MPAS module_ra_rrtmg_sw.F:9494-9517).
// Per (col, layer i < nl-1): swhr[i] = ((fd-fu)[i+1] - (fd-fu)[i]) * heatfac/pdp[i], swhrc the same with the clear-sky
// fluxes; the top layer i = nl-1 is set to 0. The primal net fluxes and zdpgcp are the forward's float32 words (its
// SU/DV); partials in float64: d z = -heatfac/pdp^2 d pdp.
// TL in the forward geometry (one thread per (col, layer)). AD in its own geometry, one thread per column walking
// the layers in order, so every level cotangent (read by two layers) is summed in one thread, in a fixed order:
// deterministic, no atomics. Refusal slot 0: an output aliasing an input.
#include "sw.cu"
#include "g3_common.cuh"
__device__ __forceinline__ bool g3_post_alias(const void* const* in, int nin, const void* a, const void* b) {
    for (int k = 0; k < nin; ++k) if (in[k] == a || in[k] == b) return true;
    return a == b;
}
extern "C" __global__ void rsw_post_b__tl(int ncol, int nlayers, float heatfac, const float* __restrict__ pdp,
    const float* __restrict__ fu, const float* __restrict__ fd, const float* __restrict__ cu, const float* __restrict__ cd,
    const float* d_pdp, const float* d_fu, const float* d_fd, const float* d_cu, const float* d_cd,
    float* d_swhr, float* d_swhrc, int* failure) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    const void* in[5] = {d_pdp, d_fu, d_fd, d_cu, d_cd};
    if (g3_post_alias(in, 5, d_swhr, d_swhrc)) g3_fail(failure, 0);
    int col = (int)(idx / nlayers), i = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers, o1 = (size_t)col * (nlayers + 1);
    if (i == nlayers - 1) { d_swhr[o + i] = 0.0f; d_swhrc[o + i] = 0.0f; return; }
    const size_t a = o1 + i, b = o1 + i + 1;
    float nc0 = SU(cd[a], cu[a]), nc1 = SU(cd[b], cu[b]), nf0 = SU(fd[a], fu[a]), nf1 = SU(fd[b], fu[b]);
    float z = DV(heatfac, pdp[o + i]);
    double p = pdp[o + i], dz = -G3D(heatfac) / (p * p) * G3D(d_pdp[o + i]);
    double dnc = (G3D(d_cd[b]) - G3D(d_cu[b])) - (G3D(d_cd[a]) - G3D(d_cu[a]));
    double dnf = (G3D(d_fd[b]) - G3D(d_fu[b])) - (G3D(d_fd[a]) - G3D(d_fu[a]));
    d_swhrc[o + i] = g3_st(dnc * G3D(z) + G3D(SU(nc1, nc0)) * dz);
    d_swhr[o + i] = g3_st(dnf * G3D(z) + G3D(SU(nf1, nf0)) * dz);
}
extern "C" __global__ void rsw_post_b__ad(int ncol, int nlayers, float heatfac, const float* __restrict__ pdp,
    const float* __restrict__ fu, const float* __restrict__ fd, const float* __restrict__ cu, const float* __restrict__ cd,
    float* b_pdp, float* b_fu, float* b_fd, float* b_cu, float* b_cd, float* b_swhr, float* b_swhrc, int* failure) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; if (col >= ncol) return;
    const void* in[5] = {b_pdp, b_fu, b_fd, b_cu, b_cd};
    if (g3_post_alias(in, 5, b_swhr, b_swhrc)) { g3_fail(failure, 0); return; }
    size_t o = (size_t)col * nlayers, o1 = (size_t)col * (nlayers + 1);
    // carry: the level-(i+1) parts of layer i's cotangent (d net1 = + g z), added when level i+1 is written.
    double carry_f = 0.0, carry_c = 0.0;
    for (int j = 0; j <= nlayers; ++j) {
        double lev_f = carry_f, lev_c = carry_c;   // cotangent of net flux at level j from layer j-1
        carry_f = carry_c = 0.0;
        if (j < nlayers - 1) {
            const size_t a = o1 + j, b = o1 + j + 1;
            float nc0 = SU(cd[a], cu[a]), nc1 = SU(cd[b], cu[b]), nf0 = SU(fd[a], fu[a]), nf1 = SU(fd[b], fu[b]);
            float z = DV(heatfac, pdp[o + j]);
            double g = b_swhr[o + j], gc = b_swhrc[o + j], p = pdp[o + j];
            double kz = -G3D(heatfac) / (p * p);
            g3_acc(b_pdp + o + j, (g * G3D(SU(nf1, nf0)) + gc * G3D(SU(nc1, nc0))) * kz);
            lev_f -= g * G3D(z); lev_c -= gc * G3D(z);          // net0 enters with -z
            carry_f = g * G3D(z); carry_c = gc * G3D(z);        // net1 enters with +z
        }
        if (j < nlayers) { b_swhr[o + j] = 0.0f; b_swhrc[o + j] = 0.0f; }
        // net = fd - fu (and cd - cu) at level j
        g3_acc(b_fd + o1 + j, lev_f); g3_acc(b_fu + o1 + j, -lev_f);
        g3_acc(b_cd + o1 + j, lev_c); g3_acc(b_cu + o1 + j, -lev_c);
    }
}
