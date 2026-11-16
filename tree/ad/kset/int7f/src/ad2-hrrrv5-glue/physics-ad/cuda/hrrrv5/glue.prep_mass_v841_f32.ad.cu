// SPDX-License-Identifier: Apache-2.0
// The forward runs first and its input arrays are retained. Compile with
// --fmad=false and the forward route's actual FTZ mode, as specified in the
// row spec. One thread owns one column and every vertical interface it updates.
// Phase 1 reduces per-column scalar cotangents in increasing cell order, with
// no unordered atomic operations.

#include "glue.member_g26.cuh"

#define MASS_FORWARD_ARGS \
    int nlev, int ncells, int nspecies, float rvord, \
    const float *rho_zz, const float *theta_m, const float *scalars, \
    const float *exner, const float *pressure_base, const float *pressure_p, \
    const float *zgrid, const float *zz, float *const *qout, float *zz_p, \
    float *rho_dry, float *rho_p, float *th_p, float *t_p, float *pi_p, \
    float *pres_p, float *zmid_p, float *dz_p, int *invalid

__device__ __forceinline__ float mass_q(float a) {
    return (__float_as_uint(a) & 0x80000000u) ? 0.0f : a;
}
__device__ __forceinline__ bool mass_q_active(float a) {
    return (__float_as_uint(a) & 0x80000000u) == 0u;
}

#if !defined(AD_AD_ONLY)
extern "C" __global__ void prep_mass_v841_f32__tl(
    MASS_FORWARD_ARGS,
    const float *d_rvord, const float *d_rho_zz, const float *d_theta_m,
    const float *d_scalars, const float *d_exner,
    const float *d_pressure_base, const float *d_pressure_p,
    const float *d_zgrid, const float *d_zz, float *const *d_qout,
    float *d_zz_p, float *d_rho_dry, float *d_rho_p, float *d_th_p,
    float *d_t_p, float *d_pi_p, float *d_pres_p, float *d_zmid_p,
    float *d_dz_p, unsigned int *branch_record)
{
    (void)qout; (void)zz_p; (void)rho_dry; (void)rho_p; (void)th_p;
    (void)t_p; (void)pi_p; (void)pres_p; (void)zmid_p; (void)dz_p;
    (void)pressure_base; (void)pressure_p; (void)zgrid; (void)invalid;
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    PrepBranchRecorder br{branch_record, 6};
    br.count(0, cell >= ncells);
    if (cell >= ncells) { br.flush(20); return; }
    br.count_n(1, true, (unsigned)nlev); br.count(1, false);
    br.count_n(2, true, (unsigned)(nlev * nspecies));
    br.count_n(2, false, (unsigned)nlev);
    int plane = nlev * ncells;
    for (int k = 0; k < nlev; ++k) {
        int i = k * ncells + cell;
        for (int s = 0; s < nspecies; ++s) {
            int si = s * plane + i;
            float raw=scalars[si];
            br.count(3, !isfinite(raw));
            br.count(4, (__float_as_uint(raw) & 0x80000000u) != 0u);
            d_qout[s][i] = mass_q_active(raw) ? d_scalars[si] : 0.0f;
        }
        float q = mass_q(scalars[i]);
        // Recompute the pinned helper calls only for branch recording.
        float rr=br.mul(zz[i],rho_zz[i]);
        (void)br.mul(rr,br.add(1.0f,q));
        float rd=br.add(1.0f,br.mul(rvord,q));
        float rt=br.div(theta_m[i],rd);
        (void)br.mul(rt,exner[i]);
        (void)br.add(pressure_p[i],pressure_base[i]);
        (void)br.mul(0.5f,br.add(zgrid[i+ncells],zgrid[i]));
        (void)br.sub(zgrid[i+ncells],zgrid[i]);
        br.count(5, !isfinite(rho_zz[i]) || rho_zz[i] <= 0.0f
          || !isfinite(theta_m[i]) || theta_m[i] <= 0.0f
          || !isfinite(zz[i]) || !isfinite(exner[i])
          || !isfinite(pressure_p[i]) || !isfinite(pressure_base[i]));
        // The ledger records every short-circuit operand independently.
        bool pass=isfinite(rho_zz[i]);
        atomicAdd(branch_record+2*20+(pass?1:0),1u);
        if(pass) { bool z=rho_zz[i]<=0.0f; atomicAdd(branch_record+2*21+(z?1:0),1u); pass=!z; }
        if(pass) { bool f=isfinite(theta_m[i]); atomicAdd(branch_record+2*22+(f?1:0),1u); pass=f; }
        if(pass) { bool z=theta_m[i]<=0.0f; atomicAdd(branch_record+2*23+(z?1:0),1u); pass=!z; }
        if(pass) { bool f=isfinite(zz[i]); atomicAdd(branch_record+2*24+(f?1:0),1u); pass=f; }
        if(pass) { bool f=isfinite(exner[i]); atomicAdd(branch_record+2*25+(f?1:0),1u); pass=f; }
        if(pass) { bool f=isfinite(pressure_p[i]); atomicAdd(branch_record+2*26+(f?1:0),1u); pass=f; }
        if(pass) { bool f=isfinite(pressure_base[i]); atomicAdd(branch_record+2*27+(f?1:0),1u); }

        float dq = mass_q_active(scalars[i]) ? d_scalars[i] : 0.0f;
        float dry = zz[i] * rho_zz[i];
        float ddry = d_zz[i] * rho_zz[i] + zz[i] * d_rho_zz[i];
        float den = 1.0f + rvord * q;
        float th = theta_m[i] / den;
        float dden = *d_rvord * q + rvord * dq;
        float dth = (d_theta_m[i] - th * dden) / den;
        d_zz_p[i] = d_zz[i];
        d_rho_dry[i] = ddry;
        d_rho_p[i] = ddry * (1.0f + q) + dry * dq;
        d_th_p[i] = dth;
        d_t_p[i] = dth * exner[i] + th * d_exner[i];
        d_pi_p[i] = d_exner[i];
        d_pres_p[i] = d_pressure_p[i] + d_pressure_base[i];
        d_zmid_p[i] = 0.5f * (d_zgrid[i + ncells] + d_zgrid[i]);
        d_dz_p[i] = d_zgrid[i + ncells] - d_zgrid[i];
    }
    br.flush(20);
}
#endif

#if !defined(AD_TL_ONLY)
extern "C" __global__ void prep_mass_v841_f32__ad(
    MASS_FORWARD_ARGS,
    float *b_rvord, float *b_rho_zz, float *b_theta_m, float *b_scalars,
    float *b_exner, float *b_pressure_base, float *b_pressure_p,
    float *b_zgrid, float *b_zz, float *const *b_qout,
    float *b_zz_p, float *b_rho_dry, float *b_rho_p,
    float *b_th_p, float *b_t_p, float *b_pi_p,
    float *b_pres_p, float *b_zmid_p, float *b_dz_p,
    double *scalar_workspace, int phase)
{
    (void)qout; (void)zz_p; (void)rho_dry; (void)rho_p; (void)th_p;
    (void)t_p; (void)pi_p; (void)pres_p; (void)zmid_p; (void)dz_p;
    (void)pressure_base; (void)pressure_p; (void)zgrid; (void)invalid;
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (phase == 1) {
        if (cell == 0) {
            double sum = 0.0;
            for (int c = 0; c < ncells; ++c) sum += scalar_workspace[c];
            *b_rvord = (float)((double)*b_rvord + sum);
        }
        return;
    }
    if (cell >= ncells) return;
    int plane = nlev * ncells;
    double bparam = 0.0;
    for (int k = 0; k < nlev; ++k) {
        int i = k * ncells + cell;
        float q = mass_q(scalars[i]);
        float dry = zz[i] * rho_zz[i];
        float den = 1.0f + rvord * q;
        float th = theta_m[i] / den;
        double bth = (double)b_th_p[i] + (double)b_t_p[i] * exner[i];
        double bden = -bth * th / den;
        double bdry = (double)b_rho_dry[i] + (double)b_rho_p[i] * (1.0f + q);
        double bq = (double)b_rho_p[i] * dry + bden * rvord;
        b_rho_zz[i] = (float)((double)b_rho_zz[i] + bdry * zz[i]);
        b_zz[i] = (float)((double)b_zz[i] + b_zz_p[i] + bdry * rho_zz[i]);
        b_theta_m[i] = (float)((double)b_theta_m[i] + bth / den);
        b_exner[i] = (float)((double)b_exner[i] + b_pi_p[i] + (double)b_t_p[i] * th);
        b_pressure_p[i] += b_pres_p[i];
        b_pressure_base[i] += b_pres_p[i];
        bparam += bden * q;
        for (int s = 0; s < nspecies; ++s) {
            int si = s * plane + i;
            if (mass_q_active(scalars[si]))
                b_scalars[si] = (float)((double)b_scalars[si] + b_qout[s][i] + (s == 0 ? bq : 0.0));
        }
    }
    for (int k = 0; k <= nlev; ++k) {
        int i = k * ncells + cell;
        double bz = b_zgrid[i];
        if (k < nlev) bz += 0.5 * b_zmid_p[i] - b_dz_p[i];
        if (k > 0) bz += 0.5 * b_zmid_p[i-ncells] + b_dz_p[i-ncells];
        b_zgrid[i] = (float)bz;
    }
    // All contributions from this column have been gathered.  Consume only
    // the output slots written by the primal; other columns are untouched.
    for (int k = 0; k < nlev; ++k) {
        int i = k * ncells + cell;
        for (int s = 0; s < nspecies; ++s) b_qout[s][i] = 0.0f;
        b_zz_p[i] = 0.0f; b_rho_dry[i] = 0.0f; b_rho_p[i] = 0.0f;
        b_th_p[i] = 0.0f; b_t_p[i] = 0.0f; b_pi_p[i] = 0.0f;
        b_pres_p[i] = 0.0f; b_zmid_p[i] = 0.0f; b_dz_p[i] = 0.0f;
    }
    scalar_workspace[cell] = bparam;
}
#endif
