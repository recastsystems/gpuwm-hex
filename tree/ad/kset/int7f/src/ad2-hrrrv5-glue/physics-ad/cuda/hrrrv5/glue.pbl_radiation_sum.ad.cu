// SPDX-License-Identifier: Apache-2.0
// Derivative of rrfs_pbl_radiation_sum, driver_rrfs.py:1312-1316 at pin 16251589:
//   out[i] = __fadd_rn(lw[i], sw[i])   (affine in lw and sw)
// __aug repeats the forward exactly (identity proof); __tl writes only the output tangent;
// __ad accumulates both input cotangents and consumes the write-only output cotangent.
// Derivative arithmetic is float64 internally, rounded once (COMMON-AD 39.4).
// Failure slot 0: non-finite primal, tangent, or adjoint value.

extern "C" __global__ void rrfs_pbl_radiation_sum__aug(
    int n, const float* lw, const float* sw, float* out, int* failure) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        const float value = __fadd_rn(lw[i], sw[i]);
        out[i] = value;
        if (!isfinite(lw[i]) || !isfinite(sw[i]) || !isfinite(value)) atomicExch(failure, 1);
    }
}

extern "C" __global__ void rrfs_pbl_radiation_sum__tl(
    int n, const float* lw, const float* sw,
    const float* dlw, const float* dsw, float* dout, int* failure) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)((double)dlw[i] + (double)dsw[i]);
    dout[i] = v;
    if (!isfinite(lw[i]) || !isfinite(sw[i]) || !isfinite(dlw[i]) ||
        !isfinite(dsw[i]) || !isfinite(v)) atomicExch(failure, 1);
}

extern "C" __global__ void rrfs_pbl_radiation_sum__ad(
    int n, const float* lw, const float* sw,
    float* blw, float* bsw, float* bout, int* failure) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double b = (double)bout[i];
    if (!isfinite(lw[i]) || !isfinite(sw[i]) || !isfinite(blw[i]) ||
        !isfinite(bsw[i]) || !isfinite(bout[i])) atomicExch(failure, 1);
    float a = (float)((double)blw[i] + b);
    float c = (float)((double)bsw[i] + b);
    blw[i] = a;
    bsw[i] = c;
    bout[i] = 0.0f;
    if (!isfinite(a) || !isfinite(c)) atomicExch(failure, 1);
}
