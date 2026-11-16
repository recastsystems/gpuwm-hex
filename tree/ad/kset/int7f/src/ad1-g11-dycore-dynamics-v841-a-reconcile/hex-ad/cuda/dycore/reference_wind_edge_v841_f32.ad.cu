// SPDX-License-Identifier: Apache-2.0
// Executed-arm tangent and transpose for the fixed-geometry wind mapping.
#define AD_B78 -1
#define AD_B83 -1
#define AD_B93 -1
#define AD_B111 3
#define AD_B116 4
#define AD_B123 5
#define AD_B142 1
#define AD_B145 2
#define AD_B42 6
#define AD_B45 7
#define AD_B57 8
#define AD_B62 9
#include "dynamics_v841_ad.cuh"

__device__ __forceinline__ AdJet ref_eval(float u, float v, float angle,
                                         float du, float dv, unsigned *branch_records=nullptr) {
    const AdJet a = ad_branch_mul({u, du}, ad_const(cosf(angle)), branch_records);
    const AdJet b = ad_branch_mul({v, dv}, ad_const(sinf(angle)), branch_records);
    return ad_branch_sub(a, b, branch_records);
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void reference_wind_edge_v841_f32__tl(
    const int nlev, const int nedges, const float *angle_edge,
    const float *u_init, const float *v_init, const float *reference_u,
    const float *d_u_init, const float *d_v_init, float *d_reference_u,
    AdFailureRecord *failures, unsigned *branch_records)
{
    (void)reference_u;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nlev * nedges; i += stride) {
        const int e = i % nedges, k = i / nedges;
        AdCheck check{failures, i, 1ull};
        const float angle = ad_read(angle_edge, e, 3, check);
        const float u = ad_read(u_init, k, 4, check), du = ad_read(d_u_init, k, 4, check);
        const float v = ad_read(v_init, k, 5, check), dv = ad_read(d_v_init, k, 5, check);
        const AdJet out = ref_eval(u, v, angle, du, dv, branch_records);
        if (!isfinite(out.v) || !isfinite(out.d)) ad_fail(check, 6, i);
        if (!check.failed) d_reference_u[i] = out.d;
    }
}
#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void reference_wind_edge_v841_f32__ad(
    const int nlev, const int nedges, const float *angle_edge,
    const float *u_init, const float *v_init, const float *reference_u,
    const float *b_reference_u, float *b_reference_u_consumed, float *b_u_init, float *b_v_init,
    AdFailureRecord *failures, unsigned *branch_records)
{
    (void)reference_u;
    const int stride = blockDim.x * gridDim.x;
    // The input seed is an immutable, disjoint scheduler snapshot.
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nlev * nedges; i += stride) b_reference_u_consumed[i] = 0.0f;
    if (branch_records) for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<nlev*nedges; i+=stride)
        (void)ref_eval(u_init[i/nedges],v_init[i/nedges],angle_edge[i%nedges],0.0f,0.0f,branch_records);
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < nlev; k += stride) {
        AdCheck check{failures, k, 1ull};
        const float u = ad_read(u_init, k, 4, check);
        const float v = ad_read(v_init, k, 5, check);
        double su = 0.0, sv = 0.0;
        for (int e = 0; e < nedges; ++e) {
            const int i = k * nedges + e;
            const float angle = ad_read(angle_edge, e, 3, check);
            const float seed = ad_read(b_reference_u, i, 6, check);
            const AdJet ju = ref_eval(u, v, angle, 1.0f, 0.0f);
            const AdJet jv = ref_eval(u, v, angle, 0.0f, 1.0f);
            if (!isfinite(ju.v) || !isfinite(ju.d) || !isfinite(jv.d)) ad_fail(check, 6, i);
            // One owner per level, traversing edges in ascending order.
            su += mpas_f32_to_f64_ieee(mpas_mul(seed, ju.d));
            sv += mpas_f32_to_f64_ieee(mpas_mul(seed, jv.d));
        }
        ad_accumulate(b_u_init, k, su, 4, check);
        ad_accumulate(b_v_init, k, sv, 5, check);
    }
}

#endif
