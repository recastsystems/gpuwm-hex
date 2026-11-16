// SPDX-License-Identifier: MIT
#ifndef HRRRV5_GLUE_MEMBER_SCALAR_SUM_CUH
#define HRRRV5_GLUE_MEMBER_SCALAR_SUM_CUH

// One thread reduces the already computed, per-element contributions in a
// fixed order.  The accumulator starts with the incoming scalar cotangent.
// Accumulating in double avoids introducing a cancellation-dependent error
// from a float reduction tree, while the exposed scalar adjoint stays float.
#if !defined(AD_MODE_TL)
extern "C" __global__ void hrrrv5_glue_scalar_sum(
    int n, const float *contributions, float *adjoint, int *failure) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double sum = (double)*adjoint;
    if (n < 0 || !isfinite(sum)) { *failure = 1; return; }
    for (int i = 0; i < n; ++i) {
        double value = (double)contributions[i];
        if (!isfinite(value)) { *failure = 1; return; }
        sum += value;
    }
    float result = (float)sum;
    if (!isfinite(result)) { *failure = 1; return; }
    *adjoint = result;
}
#endif
#endif
