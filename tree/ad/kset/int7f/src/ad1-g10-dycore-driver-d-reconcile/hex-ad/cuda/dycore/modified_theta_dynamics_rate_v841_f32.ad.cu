// SPDX-License-Identifier: Apache-2.0
#include "derivative_support.cuh"
#if !defined(AD_ONLY_AD)
extern "C" __global__ void modified_theta_dynamics_rate_v841_f32__tl(
    int nlev, int ncells, const float *tend_rtheta, const float *tend_rho,
    const float *rho, const float *theta_m, const float *result, const int *invalid,
    const float *d_tend_rtheta, const float *d_tend_rho, const float *d_rho,
    const float *d_theta_m, float *d_result, int *failure, uint32_t *branch_record)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const uint64_t i = (uint64_t)k * ncells + cell;
        const float p = ad_mul(tend_rho[i], theta_m[i]);
        const float num = ad_forward_add(tend_rtheta[i],p,true);
        const float y = ad_div(num, rho[i]);
        ad_record_mul(branch_record,3,7,tend_rho[i],theta_m[i]);
        ad_record_add(branch_record,0,7,tend_rtheta[i],p,true);
        ad_record_div(branch_record,5,7,num,rho[i]);
        ad_count(branch_record,11,!isfinite(tend_rtheta[i]));
        ad_count(branch_record,12,!isfinite(tend_rho[i]) || !isfinite(theta_m[i]));
        ad_count(branch_record,13,*invalid || !isfinite(rho[i]) || ad_flush(rho[i])<=0.0f || !isfinite(p) || !isfinite(num) || !isfinite(y));
        ad_count(branch_record,14,!isfinite(rho[i]) || rho[i]<=0.0f || !isfinite(y));
        ad_count(branch_record,15,!isfinite(rho[i]));
        ad_count(branch_record,16,!isfinite(rho[i]) || rho[i]<=0.0f || !isfinite(y));
        if (isfinite(rho[i]) && rho[i]>0.0f) ad_count(branch_record,17,!isfinite(y));
        float pb, pt, pa, pp;
        ad_mul_partials(tend_rho[i],theta_m[i],pb,pt);
        ad_add_partials(tend_rtheta[i],p,true,pa,pp);
        const float dp = ad_add(ad_mul(d_tend_rho[i],pb),ad_mul(pt,d_theta_m[i]));
        const float dn = ad_sub(ad_mul(pa,d_tend_rtheta[i]),ad_mul(pp,dp));
        const float dy = ad_div(ad_sub(dn,ad_mul(y,d_rho[i])),rho[i]);
        if (*invalid || !isfinite(rho[i]) || ad_flush(rho[i]) <= 0.0f || !isfinite(d_rho[i])) ad_refuse(failure,i,3,1662);
        else if (!isfinite(tend_rtheta[i]) || !isfinite(d_tend_rtheta[i])) ad_refuse(failure,i,1,1660);
        else if (!isfinite(tend_rho[i]) || !isfinite(d_tend_rho[i])) ad_refuse(failure,i,2,1661);
        else if (!isfinite(theta_m[i]) || !isfinite(d_theta_m[i])) ad_refuse(failure,i,4,1661);
        else if (!isfinite(p) || !isfinite(num) || !isfinite(y) || !isfinite(dp) || !isfinite(dy)) ad_refuse(failure,i,5,1662);
        else d_result[i] = dy;
    }
    (void)result;
}
#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void modified_theta_dynamics_rate_v841_f32__ad(
    int nlev, int ncells, const float *tend_rtheta, const float *tend_rho,
    const float *rho, const float *theta_m, const float *result, const int *invalid,
    float *b_tend_rtheta, float *b_tend_rho, float *b_rho, float *b_theta_m,
    float *b_result, int *failure, uint32_t *branch_record)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const uint64_t i = (uint64_t)k * ncells + cell;
        const float p = ad_mul(tend_rho[i], theta_m[i]);
        const float num = ad_forward_add(tend_rtheta[i],p,true);
        const float y = ad_div(num, rho[i]);
        ad_record_mul(branch_record,3,7,tend_rho[i],theta_m[i]);
        ad_record_add(branch_record,0,7,tend_rtheta[i],p,true);
        ad_record_div(branch_record,5,7,num,rho[i]);
        ad_count(branch_record,11,!isfinite(tend_rtheta[i]));
        ad_count(branch_record,12,!isfinite(tend_rho[i]) || !isfinite(theta_m[i]));
        ad_count(branch_record,13,*invalid || !isfinite(rho[i]) || ad_flush(rho[i])<=0.0f || !isfinite(p) || !isfinite(num) || !isfinite(y));
        ad_count(branch_record,14,!isfinite(rho[i]) || rho[i]<=0.0f || !isfinite(y));
        ad_count(branch_record,15,!isfinite(rho[i]));
        ad_count(branch_record,16,!isfinite(rho[i]) || rho[i]<=0.0f || !isfinite(y));
        if (isfinite(rho[i]) && rho[i]>0.0f) ad_count(branch_record,17,!isfinite(y));
        const float by = b_result[i];
        const float bn = ad_div(by, rho[i]);
        float pb, pt, pa, pp;
        ad_mul_partials(tend_rho[i],theta_m[i],pb,pt);
        ad_add_partials(tend_rtheta[i],p,true,pa,pp);
        const float bp = ad_mul(bn,pp);
        const float ba = ad_add(b_tend_rtheta[i],ad_mul(bn,pa));
        const float bt = ad_sub(b_tend_rho[i],ad_mul(bp,pb));
        const float bm = ad_sub(b_theta_m[i],ad_mul(bp,pt));
        const float br = ad_sub(b_rho[i], ad_mul(bn, y));
        if (*invalid || !isfinite(rho[i]) || ad_flush(rho[i]) <= 0.0f || !isfinite(b_rho[i]) || !isfinite(br)) ad_refuse(failure,i,3,1662);
        else if (!isfinite(tend_rtheta[i]) || !isfinite(b_tend_rtheta[i]) || !isfinite(ba)) ad_refuse(failure,i,1,1660);
        else if (!isfinite(tend_rho[i]) || !isfinite(b_tend_rho[i]) || !isfinite(bt)) ad_refuse(failure,i,2,1661);
        else if (!isfinite(theta_m[i]) || !isfinite(b_theta_m[i]) || !isfinite(bm)) ad_refuse(failure,i,4,1661);
        else if (!isfinite(by) || !isfinite(p) || !isfinite(num) || !isfinite(y) || !isfinite(bn)) ad_refuse(failure,i,5,1662);
        else { b_tend_rtheta[i]=ba; b_tend_rho[i]=bt; b_rho[i]=br; b_theta_m[i]=bm; b_result[i]=0.0f; }
    }
    (void)result;
}

#endif
