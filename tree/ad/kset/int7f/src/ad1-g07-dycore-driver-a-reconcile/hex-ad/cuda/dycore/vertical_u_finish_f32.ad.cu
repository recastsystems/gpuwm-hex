// SPDX-License-Identifier: Apache-2.0
// Geometry rdzw is fixed. One thread owns each input cotangent; no atomics.
using AdFail = int;
__device__ __forceinline__ bool finite_f32(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ void refuse(AdFail *f, int slot, unsigned long long field,
                                      unsigned long long element, unsigned long long source) { atomicExch(&f[slot], 1); }
// All pointers are disjoint. fail has one zero-initialized int32 source slot.
extern "C" __global__ void vertical_u_finish_f32__tl(
    int nlev, int nedges, const float *rdzw, const float *flux,
    const float *d_flux, float *d_result, AdFail *fail) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<nlev*nedges; i+=blockDim.x*gridDim.x) {
        const int k=i/nedges;
        float p=flux[i+nedges]-flux[i], y=-rdzw[k]*p;
        float v=-rdzw[k]*(d_flux[i+nedges]-d_flux[i]);
        if (!finite_f32(rdzw[k]) || !finite_f32(flux[i]) || !finite_f32(flux[i+nedges]) ||
            !finite_f32(p) || !finite_f32(y) || !finite_f32(d_flux[i]) ||
            !finite_f32(d_flux[i+nedges]) || !finite_f32(v)) refuse(fail,0,2,i,1014);
        d_result[i]=v;
    }
}
// fail: output/primal checks [0,nlev*nedges), then one per flux input.
extern "C" __global__ void vertical_u_finish_f32__ad(
    int nlev, int nedges, const float *rdzw, const float *flux,
    const float *b_result, float *b_result_consumed, float *b_flux, AdFail *fail) {
    const int count=(nlev+1)*nedges, nout=nlev*nedges;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=blockDim.x*gridDim.x) {
        // b_result is the immutable pre-adjoint snapshot; this is the disjoint incoming seed.
        if (i < nout) b_result_consumed[i] = 0.0f;
        const int k=i/nedges;
        if (i<nout) {
            float p=flux[i+nedges]-flux[i], y=-rdzw[k]*p;
            if (!finite_f32(rdzw[k]) || !finite_f32(flux[i]) || !finite_f32(flux[i+nedges]) ||
                !finite_f32(p) || !finite_f32(y) || !finite_f32(b_result[i]))
                refuse(fail,1,3,i,1014);
        }
        double v=(double)b_flux[i];
        if (k<nlev) v += rdzw[k]*b_result[i];
        if (k>0) v -= rdzw[k-1]*b_result[i-nedges];
        if (!finite_f32(b_flux[i]) || !finite_f32(v)) refuse(fail,2,2,i,1014);
        b_flux[i]=v;
    }
}
