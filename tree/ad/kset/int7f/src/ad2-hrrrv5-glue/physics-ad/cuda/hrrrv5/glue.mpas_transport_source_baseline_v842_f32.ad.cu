// SPDX-License-Identifier: MIT
#define AD_FAILURE_PAIRS(X) X(1847,6,0) X(1848,5,1) X(1848,7,2) X(1848,8,3)
#include "glue.member_g28.cuh"
#include "glue.member_scalar_sum.cuh"

__device__ __forceinline__ void baseline_record(int q, int plane, int ncells,
        int nsolve, int nspecies, float dt, const float *before,
        const float *source, const float *rho, unsigned *r) {
    bool outside=q >= nspecies * plane;
    ad_branch(r,0,outside);
    if (outside) return;
    bool solve=q % ncells < nsolve;
    ad_branch(r,1,solve);
    if (solve) {
        float den=rho[q % plane];
        ad_branch(r,2,!isfinite(den) || den == 0.0f);
        float prod=ad_record_mul(dt,source[q],r,6,10,11);
        float quotient=ad_record_div(prod,den,r,8,10,11);
        (void)ad_record_addsub(before[q],quotient,false,r,3,10,11);
    }
}

#if !defined(AD_MODE_AD)
extern "C" __global__ void mpas_transport_source_baseline_v842_f32__tl(
    int nlev, int ncells, int nsolve, int nspecies, float dt,
    const float *before, const float *source, const float *rho_old,
    const float *baseline, const float *d_dt, const float *d_before,
    const float *d_source, const double *d_rho_old, double *d_baseline,
    int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = nlev * ncells;
    if (branch_record) baseline_record(q,plane,ncells,nsolve,nspecies,dt,before,source,rho_old,branch_record);
    if (q >= nspecies * plane) return;
    if (!isfinite(before[q])) { ad_refuse(failure,failure_detail,q,6,1847); return; }
    if (q % ncells >= nsolve) { d_baseline[q] = d_before[q]; return; }
    float r = rho_old[q % plane], s = source[q];
    if (!isfinite(dt)) { ad_refuse(failure,failure_detail,0,5,1848); return; }
    if (!isfinite(s)) { ad_refuse(failure,failure_detail,q,7,1848); return; }
    if (!isfinite(r) || r == 0.0f) { ad_refuse(failure,failure_detail,q%plane,8,1848); return; }
    float u = dt * s, v = u / r;
    if (!isfinite(u) || !isfinite(v) || !isfinite(before[q] + v)) {
        ad_refuse(failure,failure_detail,q%plane,8,1848); return;
    }
    double du = (double)(*d_dt) * s + (double)dt * d_source[q];
    d_baseline[q] = (double)d_before[q] + (du - (double)v * d_rho_old[q % plane]) / r;
    (void)baseline;
}
#endif

#if !defined(AD_MODE_TL)
// One thread owns all species for a rho cell, so the shared rho cotangent is
// gathered in fixed species order without floating-point atomics.
extern "C" __global__ void mpas_transport_source_baseline_v842_f32__ad(
    int nlev, int ncells, int nsolve, int nspecies, float dt,
    const float *before, const float *source, const float *rho_old,
    const float *baseline, float *b_before, float *b_source,
    double *b_rho_old, double *seed_baseline, float *b_dt_work,
    int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = nlev * ncells;
    if (branch_record && i < plane) {
        for (int s=0;s<nspecies;++s)
            baseline_record(s*plane+i,plane,ncells,nsolve,nspecies,dt,before,source,rho_old,branch_record);
        if (i==0) {
            int total=nspecies*plane, padded=((total+127)/128)*128;
            for (int q=total;q<padded;++q)
                baseline_record(q,plane,ncells,nsolve,nspecies,dt,before,source,rho_old,branch_record);
        }
    }
    if (i >= plane) return;
    double bd = 0.0, br = 0.0;
    bool solve = i % ncells < nsolve;
    float r = solve ? rho_old[i] : 1.0f;
    if (solve && !isfinite(dt)) { ad_refuse(failure,failure_detail,0,5,1848); return; }
    if (solve && (!isfinite(r) || r == 0.0f)) { ad_refuse(failure,failure_detail,i,8,1848); return; }
    for (int s = 0; s < nspecies; ++s) {
        int q = s * plane + i;
        if (!isfinite(before[q])) { ad_refuse(failure,failure_detail,q,6,1847); return; }
        if (solve && !isfinite(source[q])) { ad_refuse(failure,failure_detail,q,7,1848); return; }
        double g = seed_baseline[q];
        if (solve) {
            float u = dt * source[q], v = u / r;
            if (!isfinite(u) || !isfinite(v) || !isfinite(before[q] + v)) {
                ad_refuse(failure,failure_detail,i,8,1848); return;
            }
            double bu = (double)g / r;
            b_source[q] = (float)((double)b_source[q] + bu * dt);
            br -= bu * v;
            bd += bu * source[q];
        }
        b_before[q] += g;
        seed_baseline[q] = 0.0;
    }
    if (solve) b_rho_old[i] += br;
    b_dt_work[i] = (float)bd;
    (void)baseline;
}
#endif
