// SPDX-License-Identifier: Apache-2.0
#include "dycore_math.cuh"
extern "C" __global__ void add_inplace_f32__tl(
    int nlevels, int nowners, const float *increment, const float *target,
    const float *d_increment, const float *d_target, float *d_output,
    int *failures)
{
    AD_LOOP(nlevels * nowners) {
        const float p = mpas_add(target[i], increment[i]);
        const float d = d_target[i] + d_increment[i];
        if (!isfinite(target[i]) || !isfinite(increment[i]) || !isfinite(p) ||
            !isfinite(d_target[i]) || !isfinite(d_increment[i]) || !isfinite(d))
            ad_fail(failures, i, 2, i, 1);
        d_output[i] = d;
    }
}
extern "C" __global__ void add_inplace_f32__ad(
    int nlevels, int nowners, const float *increment, const float *target,
    float *b_output, float *b_increment, float *b_target,
    int *failures)
{
    AD_LOOP(nlevels * nowners) {
        const float p = mpas_add(target[i], increment[i]);
        const float bi = b_increment[i] + b_output[i];
        // When target's pre- and post-state share their derivative buffer,
        // that buffer already contains the target contribution.  It remains
        // live as the pre-state cotangent after this reverse operation.
        const bool shared_target = b_target == b_output;
        const float bt = shared_target ? b_output[i] : b_target[i] + b_output[i];
        if (!isfinite(target[i]) || !isfinite(increment[i]) || !isfinite(p) ||
            !isfinite(b_output[i]) || !isfinite(bi) || !isfinite(bt))
            ad_fail(failures, i, 2, i, 1);
        b_increment[i] = bi;
        b_target[i] = bt;
        if (!shared_target) b_output[i] = 0.0f;
    }
}
