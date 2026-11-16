// SPDX-License-Identifier: Apache-2.0
#include "derivative_support.cuh"

#if !defined(AD_ONLY_AD)
extern "C" __global__ void add_lagged_diabatic_theta_v842_f32__tl(
    int nlev, int ncells, const float *rho, const float *diabatic_theta,
    const float *tend_rt, const float *d_rho, const float *d_diabatic_theta,
    const float *d_tend_rt_in, float *d_tend_rt_out, int *failure, uint32_t *branch_record)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const uint64_t i = (uint64_t)k * ncells + cell;
        const float p = ad_mul(rho[i], diabatic_theta[i]);
        const float primal = ad_forward_add(tend_rt[i],p,false);
        ad_record_mul(branch_record,3,5,rho[i],diabatic_theta[i]);
        ad_record_add(branch_record,0,5,tend_rt[i],p,false);
        ad_count(branch_record,9,!isfinite(rho[i]) || !isfinite(diabatic_theta[i]) || !isfinite(tend_rt[i]) || !isfinite(p) || !isfinite(primal));
        float pr, pd, pt, pp;
        ad_mul_partials(rho[i],diabatic_theta[i],pr,pd);
        ad_add_partials(tend_rt[i],p,false,pt,pp);
        const float dp = ad_add(ad_mul(d_rho[i],pr),ad_mul(pd,d_diabatic_theta[i]));
        const float value = ad_add(ad_mul(pt,d_tend_rt_in[i]),ad_mul(pp,dp));
        if (!isfinite(rho[i]) || !isfinite(d_rho[i])) ad_refuse(failure,i,1,1632);
        else if (!isfinite(diabatic_theta[i]) || !isfinite(d_diabatic_theta[i])) ad_refuse(failure,i,2,1632);
        else if (!isfinite(tend_rt[i]) || !isfinite(d_tend_rt_in[i]) || !isfinite(p) || !isfinite(primal) || !isfinite(value)) ad_refuse(failure,i,3,1632);
        else d_tend_rt_out[i] = value;
    }
}

#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void add_lagged_diabatic_theta_v842_f32__ad(
    int nlev, int ncells, const float *rho, const float *diabatic_theta,
    const float *tend_rt, float *b_rho, float *b_diabatic_theta,
    float *b_tend_rt_in, const float *b_tend_rt_out, int *failure, uint32_t *branch_record)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const uint64_t i = (uint64_t)k * ncells + cell;
        const float bar = b_tend_rt_out[i];
        const float p = ad_mul(rho[i], diabatic_theta[i]);
        const float primal = ad_forward_add(tend_rt[i],p,false);
        ad_record_mul(branch_record,3,5,rho[i],diabatic_theta[i]);
        ad_record_add(branch_record,0,5,tend_rt[i],p,false);
        ad_count(branch_record,9,!isfinite(rho[i]) || !isfinite(diabatic_theta[i]) || !isfinite(tend_rt[i]) || !isfinite(p) || !isfinite(primal));
        float pr, pd, pt, pp;
        ad_mul_partials(rho[i],diabatic_theta[i],pr,pd);
        ad_add_partials(tend_rt[i],p,false,pt,pp);
        const float bp = ad_mul(bar,pp);
        const float r = ad_add(b_rho[i],ad_mul(bp,pr));
        const float d = ad_add(b_diabatic_theta[i],ad_mul(bp,pd));
        const float t = ad_add(b_tend_rt_in[i],ad_mul(bar,pt));
        if (!isfinite(rho[i]) || !isfinite(b_rho[i]) || !isfinite(r)) ad_refuse(failure,i,1,1632);
        else if (!isfinite(diabatic_theta[i]) || !isfinite(b_diabatic_theta[i]) || !isfinite(d)) ad_refuse(failure,i,2,1632);
        else if (!isfinite(tend_rt[i]) || !isfinite(bar) || !isfinite(b_tend_rt_in[i]) || !isfinite(t) || !isfinite(p) || !isfinite(primal)) ad_refuse(failure,i,3,1632);
        else { b_rho[i] = r; b_diabatic_theta[i] = d; b_tend_rt_in[i] = t; }
    }
}

#endif
