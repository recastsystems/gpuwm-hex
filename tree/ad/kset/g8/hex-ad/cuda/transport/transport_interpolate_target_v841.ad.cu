// SPDX-License-Identifier: Apache-2.0
#include "ad_transport_common.cuh"
#include "ad_interpolate_branches.cuh"

#ifdef AD_BUILD_TL
extern "C" __global__ void transport_interpolate_target_v841__tl(
    int nvalues, float weight, const float *rho_old, const float *rho_new,
    const float *target, const float *d_weight, const float *d_rho_old,
    const float *d_rho_new, float *d_target, unsigned int *branch_record, AdTransportFailure *failure)
{
    (void)target;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const bool outside = i >= nvalues;
    if (branch_record) ad_ib_record(branch_record,0,outside);
    if (outside) return;
    if (branch_record) ad_ib_interpolate(weight,rho_old[i],rho_new[i],branch_record);
    if (!isfinite(weight) || !isfinite(*d_weight)) { ad_transport_fail(failure, i, 1, 0, 1); return; }
    if (!isfinite(rho_old[i]) || !isfinite(d_rho_old[i])) { ad_transport_fail(failure, i, 2, i, 1); return; }
    if (!isfinite(rho_new[i]) || !isfinite(d_rho_new[i])) { ad_transport_fail(failure, i, 3, i, 1); return; }
    // Decode subnormal inputs before arithmetic and round once on output.
    // Promoting the close density difference avoids cancellation.
    const double old_weight = mpas_f32_to_f64_ieee(mpas_sub(1.0f, weight));
    const double w = mpas_f32_to_f64_ieee(weight);
    const double delta = mpas_f32_to_f64_ieee(rho_new[i])
                       - mpas_f32_to_f64_ieee(rho_old[i]);
    d_target[i] = mpas_f64_to_f32_ieee(
        old_weight * mpas_f32_to_f64_ieee(d_rho_old[i])
        + w * mpas_f32_to_f64_ieee(d_rho_new[i])
        + mpas_f32_to_f64_ieee(*d_weight) * delta);
    if (!isfinite(d_target[i])) ad_transport_fail(failure, i, 4, i, 2);
}
#endif

#ifdef AD_BUILD_AD
#include "g8_reduce.cuh"
// One word of the reverse, exactly as the one-thread-per-word body was; returns
// early on a refusal (the caller still reduces every word, as before).
__device__ __forceinline__ void g8_interp_word(
    int i, float weight, const float *rho_old, const float *rho_new,
    float *b_rho_old, float *b_rho_new, float *b_target_live, const float *b_target,
    unsigned int *branch_record, AdTransportFailure *failure)
{
    if (branch_record) ad_ib_interpolate(weight,rho_old[i],rho_new[i],branch_record);
    if (!isfinite(weight)) { ad_transport_fail(failure, i, 1, 0, 1); return; }
    if (!isfinite(rho_old[i]) || !isfinite(b_rho_old[i])) { ad_transport_fail(failure, i, 2, i, 1); return; }
    if (!isfinite(rho_new[i]) || !isfinite(b_rho_new[i])) { ad_transport_fail(failure, i, 3, i, 1); return; }
    if (!isfinite(b_target[i])) { ad_transport_fail(failure, i, 4, i, 1); return; }
    const double old_weight = mpas_f32_to_f64_ieee(mpas_sub(1.0f, weight));
    const double w = mpas_f32_to_f64_ieee(weight);
    const double seed = mpas_f32_to_f64_ieee(b_target[i]);
    b_rho_old[i] = mpas_f64_to_f32_ieee(
        mpas_f32_to_f64_ieee(b_rho_old[i]) + old_weight * seed);
    b_rho_new[i] = mpas_f64_to_f32_ieee(
        mpas_f32_to_f64_ieee(b_rho_new[i]) + w * seed);
    if (!isfinite(b_rho_old[i])) { ad_transport_fail(failure, i, 2, i, 2); return; }
    if (!isfinite(b_rho_new[i])) { ad_transport_fail(failure, i, 3, i, 2); return; }
    // The weight reduction reads the immutable seed snapshot.
    b_target_live[i] = 0.0f;
}

// G8: the weight cotangent sum_j b_target[j]*(rho_new[j]-rho_old[j]) was one
// GPU thread walking all nvalues words (151,158 on pkg18). It is now a
// fixed-order two-level f64 reduction (g8_reduce.cuh): this entry writes one
// partial per block of G8_CHUNK words; __ad_weight adds the total into b_weight.
extern "C" __global__ void transport_interpolate_target_v841__ad(
    int nvalues, float weight, const float *rho_old, const float *rho_new,
    const float *target, float *b_weight, float *b_rho_old, float *b_rho_new,
    float *b_target_live, const float *b_target, unsigned int *branch_record,
    double *part, AdTransportFailure *failure)
{
    (void)target; (void)b_weight;
    if (blockDim.x != G8_THREADS) { if (threadIdx.x == 0) ad_transport_fail(failure, 0, 1, 0, 2); return; }
    double t = 0.0;
    for (int m = 0; m < G8_PER_THREAD; ++m) {
        const long long j = (long long)blockIdx.x * G8_CHUNK + (long long)m * G8_THREADS + threadIdx.x;
        const bool outside = j >= nvalues;
        if (branch_record) ad_ib_record(branch_record,0,outside);
        if (outside) continue;
        const int i = (int)j;
        t += mpas_f32_to_f64_ieee(b_target[i])
           * (mpas_f32_to_f64_ieee(rho_new[i])
              - mpas_f32_to_f64_ieee(rho_old[i]));
        g8_interp_word(i, weight, rho_old, rho_new, b_rho_old, b_rho_new, b_target_live, b_target, branch_record, failure);
    }
    const double s = g8_block_sum(t);
    if (threadIdx.x == 0) part[blockIdx.x] = s;
}

extern "C" __global__ void transport_interpolate_target_v841__ad_weight(
    int nvalues, float weight, float *b_weight, const double *part, int nb, AdTransportFailure *failure)
{
    (void)nvalues;
    if (blockDim.x != G8_THREADS || gridDim.x != 1) { if (threadIdx.x == 0) ad_transport_fail(failure, 0, 1, 0, 2); return; }
    const double sum = g8_partials_sum(part, 0, nb);
    if (threadIdx.x == 0) {
        if (!isfinite(weight)) { ad_transport_fail(failure, 0, 1, 0, 1); return; }
        *b_weight = mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(*b_weight) + sum);
        if (!isfinite(*b_weight)) { ad_transport_fail(failure, 0, 1, 0, 2); return; }
    }
}
#endif
