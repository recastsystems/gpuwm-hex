// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 5
#define AD_TRACE_ADD_CANCEL 6
#define AD_TRACE_MUL_NONZERO 7
#define AD_TRACE_DIV_UNDERFLOW 8
#include "branch_trace.cuh"

__device__ __forceinline__ void trace_forward(int i,int acoustic_steps,const float *a,const float *b,const float *c,uint32_t *r) {
 ad_branch(r,0,true);
 float u=ad_add(a[i],b[i],r,2); float f=ad_div(1.0f,(float)acoustic_steps,r,4);
 float q=ad_mul(c[i],f,r,3);float v=ad_add(a[i],q,r,2);
 ad_branch(r,1,acoustic_steps>0 && isfinite(a[i]) && isfinite(b[i]) && isfinite(c[i]) && isfinite(u) && isfinite(v));
}

// The original forward arguments precede seed and result arguments.
extern "C" __global__ void recover_edges_f32__tl(
    int nlev, int nedges, int acoustic_steps,
    const float *ru_saved, const float *ru_p, const float *ru_avg_p,
    float *ru, float *flux_u,
    const float *d_ru_saved, const float *d_ru_p, const float *d_ru_avg_p,
    float *d_ru, float *d_flux_u, int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<nlev*nedges;tr+=blockDim.x*gridDim.x) trace_forward(tr,acoustic_steps,ru_saved,ru_p,ru_avg_p,branch_record);
    const int n = nlev * nedges;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        if (acoustic_steps <= 0) { ad_refuse(failure, i, 4, 0); continue; }
        else if (!isfinite(ru_saved[i])) { ad_refuse(failure, i, 4, 1); continue; }
        else if (!isfinite(ru_p[i])) { ad_refuse(failure, i, 4, 2); continue; }
        else if (!isfinite(ru_avg_p[i])) { ad_refuse(failure, i, 4, 3); continue; }
        else if (!isfinite(d_ru_saved[i])) { ad_refuse(failure, i, 4, 4); continue; }
        else if (!isfinite(d_ru_p[i])) { ad_refuse(failure, i, 4, 5); continue; }
        else if (!isfinite(d_ru_avg_p[i])) { ad_refuse(failure, i, 4, 6); continue; }
        if (!isfinite(ru[i])) { ad_refuse(failure, i, 7, 7); continue; }
        else if (!isfinite(flux_u[i])) { ad_refuse(failure, i, 7, 8); continue; }
        const float factor = 1.0f / (float)acoustic_steps;
        const float u = d_ru_saved[i] + d_ru_p[i];
        const float f = d_ru_saved[i] + d_ru_avg_p[i] * factor;
        if (!isfinite(u)) { ad_refuse(failure, i, 7, 9); continue; }
        else if (!isfinite(f)) { ad_refuse(failure, i, 7, 10); continue; }
        d_ru[i] = u;
        d_flux_u[i] = f;
    }
}

extern "C" __global__ void recover_edges_f32__ad(
    int nlev, int nedges, int acoustic_steps,
    const float *ru_saved, const float *ru_p, const float *ru_avg_p,
    float *ru, float *flux_u,
    float *b_ru, float *b_flux_u,
    float *b_ru_saved, float *b_ru_p, float *b_ru_avg_p,
    int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<nlev*nedges;tr+=blockDim.x*gridDim.x) trace_forward(tr,acoustic_steps,ru_saved,ru_p,ru_avg_p,branch_record);
    const int n = nlev * nedges;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        if (acoustic_steps <= 0) { ad_refuse(failure, i, 4, 0); continue; }
        else if (!isfinite(ru_saved[i])) { ad_refuse(failure, i, 4, 1); continue; }
        else if (!isfinite(ru_p[i])) { ad_refuse(failure, i, 4, 2); continue; }
        else if (!isfinite(ru_avg_p[i])) { ad_refuse(failure, i, 4, 3); continue; }
        else if (!isfinite(b_ru[i])) { ad_refuse(failure, i, 4, 4); continue; }
        else if (!isfinite(b_flux_u[i])) { ad_refuse(failure, i, 4, 5); continue; }
        if (!isfinite(ru[i])) { ad_refuse(failure, i, 7, 6); continue; }
        else if (!isfinite(flux_u[i])) { ad_refuse(failure, i, 7, 7); continue; }
        const float factor = 1.0f / (float)acoustic_steps;
        const float acc_saved = (float)((double)b_ru_saved[i] + (double)b_ru[i] + (double)b_flux_u[i]);
        const float acc_p = b_ru_p[i] + b_ru[i];
        const float acc_avg = b_ru_avg_p[i] + factor * b_flux_u[i];
        if (!isfinite(acc_saved)) { ad_refuse(failure, i, 4, 8); continue; }
        else if (!isfinite(acc_p)) { ad_refuse(failure, i, 4, 9); continue; }
        else if (!isfinite(acc_avg)) { ad_refuse(failure, i, 4, 10); continue; }
        b_ru_saved[i] = acc_saved; b_ru_p[i] = acc_p; b_ru_avg_p[i] = acc_avg;
        b_ru[i] = 0.0f; b_flux_u[i] = 0.0f;
    }
}
