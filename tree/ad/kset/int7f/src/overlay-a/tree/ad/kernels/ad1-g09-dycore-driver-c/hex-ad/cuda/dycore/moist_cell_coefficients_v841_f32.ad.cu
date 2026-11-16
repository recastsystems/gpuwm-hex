// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 7
#define AD_TRACE_ADD_CANCEL 8
#define AD_TRACE_MUL_NONZERO 9
#define AD_TRACE_DIV_UNDERFLOW 10
#include "branch_trace.cuh"

__device__ __forceinline__ void trace_forward(int cell,int nlev,int ncells,int nmass,const float *x,uint32_t *r) {
 float prev=0.0f;ad_branch(r,0,true);
 for(int k=0;k<nlev;k++){
  float sum=0.0f;
  for(int s=0;s<nmass;s++){float v=x[(s*nlev+k)*ncells+cell];bool f=isfinite(v);ad_branch(r,18,f);ad_branch(r,1,!f);sum=ad_add(sum,v,r,4);}
  bool fs=isfinite(sum);ad_branch(r,20,fs);ad_branch(r,19,!fs);ad_branch(r,2,k>0);
  if(k>0){float t=ad_mul(0.5f,ad_add(sum,prev,r,4),r,5);float den=ad_add(1.0f,t,r,4);float v=ad_div(1.0f,den,r,6);bool ft=isfinite(t),fv=true;ad_branch(r,25,ft);if(ft){fv=isfinite(v);ad_branch(r,26,fv);}ad_branch(r,3,!ft||!fv);}
  prev=sum;
 }
}

extern "C" __global__ void moist_cell_coefficients_v841_f32__tl(
    int nlev, int ncells, int nmass, const float *scalars,
    float *qtot, float *cqw, int *invalid,
    const float *d_scalars, float *d_qtot, float *d_cqw,
    int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,nmass,scalars,branch_record);
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;
    if (nmass < 0) { ad_refuse(failure, cell, 4, 0); return; }
    else if (*invalid != 0) { ad_refuse(failure, cell, 4, 1); return; }
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        float dt = 0.0f;
        for (int s = 0; s < nmass; ++s) {
            const int j = (s * nlev + k) * ncells + cell;
            if (!isfinite(scalars[j])) { ad_refuse(failure, cell, 4, 2); return; }
            else if (!isfinite(d_scalars[j])) { ad_refuse(failure, cell, 4, 3); return; }
            dt += d_scalars[j];
        }
        if (!isfinite(qtot[i])) { ad_refuse(failure, cell, 5, 4); return; }
        else if (!isfinite(dt)) { ad_refuse(failure, cell, 5, 5); return; }
        d_qtot[i] = dt;
    }
    // The forward does not write level zero: leave that tangent untouched.
    for (int k = 1; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float value = -0.5f * cqw[i] * cqw[i] * (d_qtot[i] + d_qtot[i - ncells]);
        if (!isfinite(cqw[i])) { ad_refuse(failure, cell, 6, 6); return; }
        else if (!isfinite(value)) { ad_refuse(failure, cell, 6, 7); return; }
        d_cqw[i] = value;
    }
}

extern "C" __global__ void moist_cell_coefficients_v841_f32__ad(
    int nlev, int ncells, int nmass, const float *scalars,
    float *qtot, float *cqw, int *invalid,
    float *b_qtot, float *b_cqw, float *b_cqw_prior, float *b_scalars,
    int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,nmass,scalars,branch_record);
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;
    if (nmass < 0) { ad_refuse(failure, cell, 4, 0); return; }
    else if (*invalid != 0) { ad_refuse(failure, cell, 4, 1); return; }
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        double b = (double)b_qtot[i];
        if (!isfinite(qtot[i])) { ad_refuse(failure, cell, 5, 2); return; }
        else if (!isfinite(b)) { ad_refuse(failure, cell, 5, 3); return; }
        if (k > 0) {
            if (!isfinite(cqw[i])) { ad_refuse(failure, cell, 6, 4); return; }
            else if (!isfinite(b_cqw[i])) { ad_refuse(failure, cell, 6, 5); return; }
            b -= 0.5 * (double)cqw[i] * cqw[i] * b_cqw[i];
        }
        if (k + 1 < nlev) {
            const int next = i + ncells;
            if (!isfinite(cqw[next])) { ad_refuse(failure, cell, 6, 6); return; }
            else if (!isfinite(b_cqw[next])) { ad_refuse(failure, cell, 6, 7); return; }
            b -= 0.5 * (double)cqw[next] * cqw[next] * b_cqw[next];
        }
        for (int s = 0; s < nmass; ++s) {
            const int j = (s * nlev + k) * ncells + cell;
            if (!isfinite(scalars[j])) { ad_refuse(failure, cell, 4, 8); return; }
            const float acc = (float)((double)b_scalars[j] + b);
            if (!isfinite(acc)) { ad_refuse(failure, cell, 4, 9); return; }
            b_scalars[j] = acc;
        }
        // The runner may supply distinct input/output cotangents, or recycle
        // the output seed as the prior-state cotangent.  In the latter case,
        // ascending k has already consumed this seed both here and at k-1.
        if (k == 0 && !isfinite(b_cqw[i])) { ad_refuse(failure, cell, 6, 10); return; }
        if (b_cqw_prior == b_cqw) {
            if (k > 0) b_cqw_prior[i] = 0.0f;
        } else if (k == 0) {
            const float acc = b_cqw_prior[i] + b_cqw[i];
            if (!isfinite(acc)) { ad_refuse(failure, cell, 6, 11); return; }
            b_cqw_prior[i] = acc;
        }
        // qtot is fully overwritten; cqw level zero is not overwritten.
        // In ascending order, cqw[k] has been used by both k-1 and k here.
        b_qtot[i] = 0.0f;
        if (k > 0) b_cqw[i] = 0.0f;
    }
}
