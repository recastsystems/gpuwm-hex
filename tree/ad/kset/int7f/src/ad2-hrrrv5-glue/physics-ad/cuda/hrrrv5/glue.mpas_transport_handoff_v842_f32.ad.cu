// SPDX-License-Identifier: MIT
#define AD_FAILURE_PAIRS(X) X(1866,8,0) X(1866,12,1) X(1869,11,2) X(1871,9,3) X(1871,10,4) X(1871,14,5) X(1872,16,6) X(1879,13,7)
#include "glue.member_g28.cuh"
#include "glue.member_scalar_sum.cuh"

__device__ __forceinline__ bool handoff_clamps(float x) {
    unsigned int w = __float_as_uint(x), m = w & 0x7fffffffu;
    return (w & 0x80000000u) && m != 0u && m <= 0x7f800000u;
}

__device__ __forceinline__ void handoff_record(int i,int plane,int ncells,int nsolve,
        int nspecies,int iqv,int handoff,int monotonic,float dt,float rv,float rd,
        float rvord,const float *before,const float *after,const float *theta,
        const float *rth,unsigned *r) {
    bool outside=i>=plane; ad_branch(r,0,outside);
    if(outside || (handoff && (iqv<0 || iqv>=nspecies))) return;
    int qi=iqv*plane+i;
    ad_branch(r,1,handoff != 0);
    if(handoff) {
        ad_branch(r,2,monotonic != 0);
        float dq=0.0f;
        if(monotonic) {
            float num=ad_record_addsub(after[qi],before[qi],true,r,10,17,18);
            dq=ad_record_div(num,dt,r,15,17,18);
        }
        bool solve=i%ncells<nsolve; ad_branch(r,3,solve);
        if(solve) {
            float coeff=ad_record_addsub(1.0f,ad_record_mul(rvord,after[qi],r,13,17,18),false,r,7,17,18);
            ad_branch(r,4,!isfinite(coeff) || coeff==0.0f);
            float fac=ad_record_div(1.0f,coeff,r,15,17,18);
            float local=ad_record_mul(theta[i],fac,r,13,17,18);
            float product=ad_record_mul(local,rv,r,13,17,18);
            float ratio=ad_record_div(product,rd,r,15,17,18);
            float vapor=ad_record_mul(ratio,dq,r,13,17,18);
            float diff=ad_record_addsub(rth[i],vapor,true,r,10,17,18);
            (void)ad_record_mul(fac,diff,r,13,17,18);
        }
    }
    atomicAdd(r+2*5+1,(unsigned)nspecies);
    ad_branch(r,5,false);
    for (int s=0; s<nspecies; ++s) {
        unsigned word=__float_as_uint(after[s*plane+i]);
        unsigned mag=word & 0x7fffffffu;
        bool negative=(word & 0x80000000u)!=0u;
        if (negative) ad_branch(r,19,mag!=0u);
        if (negative && mag!=0u) ad_branch(r,20,mag<=0x7f800000u);
        ad_branch(r,6,negative && mag!=0u && mag<=0x7f800000u);
    }
}

#if !defined(AD_MODE_AD)
extern "C" __global__ void mpas_transport_handoff_v842_f32__tl(
    int nlev, int ncells, int nsolve, int nspecies, int iqv, int handoff,
    int monotonic, float dt, float rv, float rd, float rvord,
    const float *before, const float *after_before, const float *theta_m,
    const float *rqv_before, const float *rth_before,
    const float *d_dt, const float *d_rv, const float *d_rd, const float *d_rvord,
    const float *d_before, float *d_after, const float *d_theta_m,
    float *d_rqv, float *d_rth, int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, plane = nlev * ncells;
    if(branch_record) handoff_record(i,plane,ncells,nsolve,nspecies,iqv,handoff,monotonic,dt,rv,rd,rvord,before,after_before,theta_m,rth_before,branch_record);
    if (i >= plane) return;
    int qi = iqv * plane + i;
    for (int s = 0; s < nspecies; ++s) {
        if (!isfinite(after_before[s * plane + i])) {
            ad_refuse(failure,failure_detail,(unsigned long long)s*plane+i,13,1879); return;
        }
    }
    if (handoff) {
        float x = after_before[qi], dq = 0.0f;
        double ddq = 0.0;
        if (monotonic) {
            if (!isfinite(before[qi])) { ad_refuse(failure,failure_detail,qi,12,1866); return; }
            if (!isfinite(dt) || dt == 0.0f) { ad_refuse(failure,failure_detail,0,8,1866); return; }
            dq = (x - before[qi]) / dt;
            ddq = ((double)d_after[qi] - d_before[qi] - (double)dq * (*d_dt)) / dt;
        }
        if (!isfinite(dq)) { ad_refuse(failure,failure_detail,0,8,1866); return; }
        d_rqv[i] = (float)ddq;
        if (i % ncells < nsolve) {
            if (!isfinite(rv)) { ad_refuse(failure,failure_detail,0,9,1871); return; }
            if (!isfinite(rd) || rd == 0.0f) { ad_refuse(failure,failure_detail,0,10,1871); return; }
            if (!isfinite(rvord)) { ad_refuse(failure,failure_detail,0,11,1869); return; }
            if (!isfinite(theta_m[i])) { ad_refuse(failure,failure_detail,i,14,1871); return; }
            if (!isfinite(rth_before[i])) { ad_refuse(failure,failure_detail,i,16,1872); return; }
            float coeff = 1.0f + rvord * x;
            if (!isfinite(coeff) || coeff == 0.0f) { ad_refuse(failure,failure_detail,0,11,1869); return; }
            float fac = 1.0f / coeff;
            double dfac = -(double)fac * fac * ((double)(*d_rvord) * x + (double)rvord * d_after[qi]);
            float t = theta_m[i] * fac;
            double dtlocal = (double)d_theta_m[i] * fac + (double)theta_m[i] * dfac;
            float a = t * rv;
            double da = dtlocal * rv + (double)t * (*d_rv);
            float ratio = a / rd;
            double dratio = (da - (double)ratio * (*d_rd)) / rd;
            float v = ratio * dq;
            double dv = dratio * dq + (double)ratio * ddq;
            float diff = rth_before[i] - v;
            if (!isfinite(fac) || !isfinite(ratio) || !isfinite(diff) || !isfinite(fac * diff)) {
                ad_refuse(failure,failure_detail,i,16,1872); return;
            }
            d_rth[i] = (float)(dfac * diff + (double)fac * ((double)d_rth[i] - dv));
        }
    }
    for (int s = 0; s < nspecies; ++s) {
        int si = s * plane + i;
        if (handoff_clamps(after_before[si])) d_after[si] = 0.0f;
    }
    (void)rqv_before;
}
#endif

#if !defined(AD_MODE_TL)
extern "C" __global__ void mpas_transport_handoff_v842_f32__ad(
    int nlev, int ncells, int nsolve, int nspecies, int iqv, int handoff,
    int monotonic, float dt, float rv, float rd, float rvord,
    const float *before, const float *after_before, const float *theta_m,
    const float *rqv_before, const float *rth_before,
    float *b_before, float *seed_after, float *seed_rqv,
    float *seed_rth, float *b_after, float *b_theta_m,
    float *b_rqv, float *b_rth, float *b_dt_work, float *b_rv_work,
    float *b_rd_work, float *b_rvord_work, int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, plane = nlev * ncells;
    if(branch_record) handoff_record(i,plane,ncells,nsolve,nspecies,iqv,handoff,monotonic,dt,rv,rd,rvord,before,after_before,theta_m,rth_before,branch_record);
    if (i >= plane) return;
    int qi = iqv * plane + i;
    double bdt=0.0, brv=0.0, brd=0.0, brvord=0.0, bx=0.0;
    for (int s = 0; s < nspecies; ++s) {
        if (!isfinite(after_before[s * plane + i])) {
            ad_refuse(failure,failure_detail,(unsigned long long)s*plane+i,13,1879); return;
        }
    }
    if (handoff) {
        float x = after_before[qi], dq = 0.0f;
        if (monotonic) {
            if (!isfinite(before[qi])) { ad_refuse(failure,failure_detail,qi,12,1866); return; }
            if (!isfinite(dt) || dt == 0.0f) { ad_refuse(failure,failure_detail,0,8,1866); return; }
            dq = (x - before[qi]) / dt;
        }
        if (!isfinite(dq)) { ad_refuse(failure,failure_detail,0,8,1866); return; }
        double bdq = seed_rqv[i];
        if (i % ncells < nsolve) {
            if (!isfinite(rv)) { ad_refuse(failure,failure_detail,0,9,1871); return; }
            if (!isfinite(rd) || rd == 0.0f) { ad_refuse(failure,failure_detail,0,10,1871); return; }
            if (!isfinite(rvord)) { ad_refuse(failure,failure_detail,0,11,1869); return; }
            if (!isfinite(theta_m[i])) { ad_refuse(failure,failure_detail,i,14,1871); return; }
            if (!isfinite(rth_before[i])) { ad_refuse(failure,failure_detail,i,16,1872); return; }
            float coeff = 1.0f + rvord * x;
            if (!isfinite(coeff) || coeff == 0.0f) { ad_refuse(failure,failure_detail,0,11,1869); return; }
            float fac = 1.0f / coeff, t = theta_m[i] * fac;
            float a = t * rv, ratio = a / rd, diff = rth_before[i] - ratio * dq;
            if (!isfinite(fac) || !isfinite(ratio) || !isfinite(diff) || !isfinite(fac * diff)) {
                ad_refuse(failure,failure_detail,i,16,1872); return;
            }
            double g = seed_rth[i];
            double bfac = g * diff, bdiff = g * fac;
            // The runner may bind an in-place field's output seed and input
            // cotangent to the same storage.  Consume the seed before
            // replacing it with the contribution to the previous state.
            b_rth[i] = (float)((b_rth == seed_rth ? 0.0 : (double)b_rth[i]) + bdiff);
            double bv = -bdiff, bratio = bv * dq;
            bdq += bv * ratio;
            double ba = bratio / rd;
            brd -= ba * ratio;
            double bt = ba * rv;
            brv += ba * t;
            b_theta_m[i] = (float)((double)b_theta_m[i] + bt * fac);
            bfac += bt * theta_m[i];
            double bc = -(bfac * fac) * fac;
            brvord += bc * x;
            bx += bc * rvord;
        } else if (b_rth != seed_rth) b_rth[i] += seed_rth[i];
        if (monotonic) {
            double bn = bdq / dt;
            bx += bn;
            b_before[qi] = (float)((double)b_before[qi] - bn);
            bdt -= bn * dq;
        }
    } else {
        if (b_rqv != seed_rqv) b_rqv[i] += seed_rqv[i];
        if (b_rth != seed_rth) b_rth[i] += seed_rth[i];
    }
    for (int s = 0; s < nspecies; ++s) {
        int si = s * plane + i;
        double sum = b_after == seed_after ? 0.0 : (double)b_after[si];
        if (!handoff_clamps(after_before[si])) sum += seed_after[si];
        if (handoff && si == qi) sum += bx;
        b_after[si] = (float)sum;
        if (b_after != seed_after && handoff_clamps(after_before[si])) seed_after[si] = 0.0f;
    }
    if (handoff) seed_rqv[i] = 0.0f;
    if (handoff && i % ncells < nsolve && b_rth != seed_rth) seed_rth[i] = 0.0f;
    b_dt_work[i] = (float)bdt; b_rv_work[i] = (float)brv;
    b_rd_work[i] = (float)brd; b_rvord_work[i] = (float)brvord;
    (void)rqv_before;
}
#endif
