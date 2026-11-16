// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 5
#define AD_TRACE_ADD_CANCEL 6
#define AD_TRACE_SUB_SUBNORMAL 7
#define AD_TRACE_SUB_CANCEL 8
#define AD_TRACE_MUL_NONZERO 9
#include "branch_trace.cuh"

__device__ __forceinline__ void trace_forward(int cell,int nlev,int ncells,float gravity,const float *rho,const float *d,const float *q,uint32_t *r) {
 ad_branch(r,0,true);
 for(int k=0;k<nlev;k++){int i=k*ncells+cell;float x=ad_mul(rho[i],q[i],r,4);float y=ad_mul(d[i],ad_add(1.0f,q[i],r,2),r,4);
  float v=ad_mul(ad_add(0.0f,gravity,r,3,true),ad_add(x,y,r,2),r,4);
  bool a=isfinite(rho[i]),b=true,c=true,e=true;ad_branch(r,18,a);if(a){b=isfinite(d[i]);ad_branch(r,20,b);if(b){c=isfinite(q[i]);ad_branch(r,22,c);if(c){e=isfinite(v);ad_branch(r,24,e);}}}ad_branch(r,1,!a||!b||!c||!e);}
}

extern "C" __global__ void moist_dpdz_v841_f32__tl(
    int nlev, int ncells, float gravity,
    const float *rho_base, const float *density_perturbation,
    const float *qtot, float *dpdz, int *invalid,
    const float *d_gravity_ptr, const float *d_rho_base, const float *d_density_perturbation,
    const float *d_qtot, float *d_dpdz, int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,gravity,rho_base,density_perturbation,qtot,branch_record);
    const float d_gravity=*d_gravity_ptr;
    (void)dpdz;
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;
    if (*invalid) { ad_refuse(failure, cell, 3, 0); return; }
    else if (!isfinite(gravity)) { ad_refuse(failure, cell, 3, 1); return; }
    else if (!isfinite(d_gravity)) { ad_refuse(failure, cell, 3, 2); return; }
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float r = rho_base[i], d = density_perturbation[i], q = qtot[i];
        const float x = r*q + d*(1.0f+q);
        // Form the derivative in double precision before rounding the stored
        // f32 tangent. This avoids cancellation when different active fields
        // contribute nearly equal and opposite directional terms.
        const double dx = (double)d_rho_base[i]*q + (double)r*d_qtot[i] +
                          (double)d_density_perturbation[i]*(1.0f+q) + (double)d*d_qtot[i];
        const float out = (float)(-(double)d_gravity*x - (double)gravity*dx);
        if (!isfinite(r)) { ad_refuse(failure, cell, 4, 3); return; }
        else if (!isfinite(d)) { ad_refuse(failure, cell, 4, 4); return; }
        else if (!isfinite(q)) { ad_refuse(failure, cell, 4, 5); return; }
        else if (!isfinite(x)) { ad_refuse(failure, cell, 4, 6); return; }
        else if (!isfinite(out)) { ad_refuse(failure, cell, 4, 7); return; }
        else if (!isfinite(-gravity*x)) { ad_refuse(failure, cell, 4, 8); return; }
        d_dpdz[i] = out;
    }
}

// Block zero reduces the shared scalar parameter in a fixed tree order while
// all blocks independently update their disjoint cell fields.
extern "C" __global__ void moist_dpdz_v841_f32__ad(
    int nlev, int ncells, float gravity,
    const float *rho_base, const float *density_perturbation,
    const float *qtot, float *dpdz, int *invalid,
    float *b_dpdz, float *b_gravity, float *b_rho_base,
    float *b_density_perturbation, float *b_qtot, int *failure, uint32_t *branch_record,
    unsigned int *completed)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,gravity,rho_base,density_perturbation,qtot,branch_record);
    (void)dpdz;
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < ncells) {
        if (*invalid) { ad_refuse(failure, cell, 3, 0); }
        else if (!isfinite(gravity)) { ad_refuse(failure, cell, 3, 1); }
        else for (int k = 0; k < nlev; ++k) {
            const int i = k * ncells + cell;
            const float r = rho_base[i], d = density_perturbation[i], q = qtot[i], b = b_dpdz[i];
            const float x = r*q + d*(1.0f+q);
            if (!isfinite(r)) { ad_refuse(failure, cell, 4, 2); break; }
            else if (!isfinite(d)) { ad_refuse(failure, cell, 4, 3); break; }
            else if (!isfinite(q)) { ad_refuse(failure, cell, 4, 4); break; }
            else if (!isfinite(b)) { ad_refuse(failure, cell, 4, 5); break; }
            else if (!isfinite(x)) { ad_refuse(failure, cell, 4, 6); break; }
            else if (!isfinite(-gravity*x)) { ad_refuse(failure, cell, 4, 7); break; }
            const float acc_r = (float)((double)b_rho_base[i] - (double)gravity*q*b);
            const float acc_d = (float)((double)b_density_perturbation[i] - (double)gravity*(1.0f+q)*b);
            const float acc_q = (float)((double)b_qtot[i] - (double)gravity*((double)r+d)*b);
            if (!isfinite(acc_r)) { ad_refuse(failure, cell, 4, 10); break; }
            else if (!isfinite(acc_d)) { ad_refuse(failure, cell, 4, 11); break; }
            else if (!isfinite(acc_q)) { ad_refuse(failure, cell, 4, 12); break; }
            b_rho_base[i] = acc_r; b_density_perturbation[i] = acc_d; b_qtot[i] = acc_q;
        }
    }
    if (blockIdx.x == 0) {
        __shared__ double sum[1024];
        const int t = threadIdx.x, n = nlev * ncells;
        double value = 0.0;
        for (int i = t; i < n; i += blockDim.x) {
            const float x = rho_base[i]*qtot[i] + density_perturbation[i]*(1.0f+qtot[i]);
            value -= (double)x*b_dpdz[i];
        }
        sum[t] = value;
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if ((t % (stride*2)) == 0 && t + stride < blockDim.x) sum[t] += sum[t + stride];
            __syncthreads();
        }
        if (t == 0) {
            const float result = (float)((double)*b_gravity + sum[0]);
            if (!isfinite(sum[0])) { ad_refuse(failure, ncells, 3, 8); }
            else if (!isfinite(result)) { ad_refuse(failure, ncells, 3, 9); }
            else *b_gravity = result;
        }
    }
    if (ad_last_block(completed))
        for (int i = threadIdx.x; i < nlev*ncells; i += blockDim.x)
            b_dpdz[i] = 0.0f;
}
