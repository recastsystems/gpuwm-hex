// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 5
#define AD_TRACE_ADD_CANCEL 6
#define AD_TRACE_MUL_NONZERO 7
#define AD_TRACE_DIV_UNDERFLOW 8
#include "branch_trace.cuh"

__device__ __forceinline__ void trace_forward(int i,int nlev,int ncells,int acoustic_steps,const float *a,const float *b,const float *c,uint32_t *r) {
 int k=i/ncells; bool boundary=k==0||k==nlev;ad_branch(r,0,boundary);
 float u=boundary?0.0f:ad_add(a[i],b[i],r,2);float f=ad_div(1.0f,(float)acoustic_steps,r,4);
 float q=ad_mul(c[i],f,r,3);float v=ad_add(a[i],q,r,2);
 ad_branch(r,1,acoustic_steps>0 && isfinite(a[i]) && isfinite(c[i]) && (boundary||isfinite(b[i])) && isfinite(u) && isfinite(v));
}

extern "C" __global__ void recover_interfaces_f32__tl(
    int nlev, int ncells, int acoustic_steps,
    const float *rw_saved, const float *rw_p, const float *ww_avg_p,
    float *rw, float *flux_w,
    const float *d_rw_saved, const float *d_rw_p, const float *d_ww_avg_p,
    float *d_rw, float *d_flux_w, int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<(nlev+1)*ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,acoustic_steps,rw_saved,rw_p,ww_avg_p,branch_record);
    const int n = (nlev + 1) * ncells;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const int k = i / ncells;
        if (acoustic_steps <= 0) { ad_refuse(failure, i, 4, 0); continue; }
        else if (!isfinite(rw_saved[i])) { ad_refuse(failure, i, 4, 1); continue; }
        else if (!isfinite(ww_avg_p[i])) { ad_refuse(failure, i, 4, 2); continue; }
        else if (!isfinite(d_rw_saved[i])) { ad_refuse(failure, i, 4, 3); continue; }
        else if (!isfinite(d_ww_avg_p[i])) { ad_refuse(failure, i, 4, 4); continue; }
        else if ((k > 0) && (k < nlev) && (!isfinite(rw_p[i]))) { ad_refuse(failure, i, 4, 5); continue; }
        else if ((k > 0) && (k < nlev) && (!isfinite(d_rw_p[i]))) { ad_refuse(failure, i, 4, 6); continue; }
        if (!isfinite(rw[i])) { ad_refuse(failure, i, 7, 7); continue; }
        else if (!isfinite(flux_w[i])) { ad_refuse(failure, i, 7, 8); continue; }
        const float factor = 1.0f / (float)acoustic_steps;
        const float w = (k == 0 || k == nlev) ? 0.0f : d_rw_saved[i] + d_rw_p[i];
        const float f = d_rw_saved[i] + d_ww_avg_p[i] * factor;
        if (!isfinite(w)) { ad_refuse(failure, i, 7, 9); continue; }
        else if (!isfinite(f)) { ad_refuse(failure, i, 7, 10); continue; }
        d_rw[i] = w;
        d_flux_w[i] = f;
    }
}

extern "C" __global__ void recover_interfaces_f32__ad(
    int nlev, int ncells, int acoustic_steps,
    const float *rw_saved, const float *rw_p, const float *ww_avg_p,
    float *rw, float *flux_w,
    float *b_rw, float *b_flux_w,
    float *b_rw_saved, float *b_rw_p, float *b_ww_avg_p,
    int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<(nlev+1)*ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,acoustic_steps,rw_saved,rw_p,ww_avg_p,branch_record);
    const int n = (nlev + 1) * ncells;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const int k = i / ncells;
        if (acoustic_steps <= 0) { ad_refuse(failure, i, 4, 0); continue; }
        else if (!isfinite(rw_saved[i])) { ad_refuse(failure, i, 4, 1); continue; }
        else if (!isfinite(ww_avg_p[i])) { ad_refuse(failure, i, 4, 2); continue; }
        else if (!isfinite(b_flux_w[i])) { ad_refuse(failure, i, 4, 3); continue; }
        else if ((k > 0) && (k < nlev) && (!isfinite(rw_p[i]))) { ad_refuse(failure, i, 4, 4); continue; }
        else if ((k > 0) && (k < nlev) && (!isfinite(b_rw[i]))) { ad_refuse(failure, i, 4, 5); continue; }
        if (!isfinite(rw[i])) { ad_refuse(failure, i, 7, 6); continue; }
        else if (!isfinite(flux_w[i])) { ad_refuse(failure, i, 7, 7); continue; }
        const float interior = (k == 0 || k == nlev) ? 0.0f : b_rw[i];
        const bool active_p = k > 0 && k < nlev;
        const float acc_saved = (float)((double)b_rw_saved[i] + (double)interior + (double)b_flux_w[i]);
        const float acc_p = active_p ? b_rw_p[i] + interior : 0.0f;
        const float acc_avg = b_ww_avg_p[i] + b_flux_w[i] * (1.0f / (float)acoustic_steps);
        if (!isfinite(acc_saved)) { ad_refuse(failure, i, 4, 8); continue; }
        else if (active_p && !isfinite(acc_p)) { ad_refuse(failure, i, 4, 9); continue; }
        else if (!isfinite(acc_avg)) { ad_refuse(failure, i, 4, 10); continue; }
        b_rw_saved[i] = acc_saved; if (active_p) b_rw_p[i] = acc_p; b_ww_avg_p[i] = acc_avg;
        b_rw[i] = 0.0f; b_flux_w[i] = 0.0f;
    }
}
