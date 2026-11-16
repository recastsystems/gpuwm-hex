// SPDX-License-Identifier: Apache-2.0
#include "ad_transport_common.cuh"
#include "ad_mpas_f32.cuh"

// Read the sign from the binary32 representation so that a subnormal keeps
// the same winner as the forward's guarded min/max. Zero chooses the constant
// left operand for both operations. Nonfinite inputs are refused separately.
__device__ __forceinline__ bool ad_pos(float x) {
    unsigned u = __float_as_uint(x); return (u & 0x7fffffffu) && !(u >> 31);
}
__device__ __forceinline__ bool ad_neg(float x) {
    unsigned u = __float_as_uint(x); return (u & 0x7fffffffu) && (u >> 31);
}
__device__ __forceinline__ float ad_plus(float x) { return mpas_max(0.0f, x); }
__device__ __forceinline__ float ad_minus(float x) { return mpas_min(0.0f, x); }
__device__ __forceinline__ int ad_q(int t, int k, int c, int nl, int nc) {
    return ((t * nl + k) * nc + c);
}
__device__ __forceinline__ int ad_qi(int t, int k, int c, int nl, int nc) {
    return ((t * (nl + 1) + k) * nc + c);
}
__device__ __forceinline__ float ad_fhat(float v, float a, float b) {
    return mpas_add(mpas_mul(ad_plus(v), a), mpas_mul(ad_minus(v), b));
}
__device__ __forceinline__ double ad_flux_tl(
    float dt, float ddt, float v, float dv, float a, float da, float b, float db)
{
    const double aa = (ad_pos(v) ? mpas_f32_to_f64_ieee(dv) : 0.0) * mpas_f32_to_f64_ieee(a)
                    + mpas_f32_to_f64_ieee(ad_plus(v)) * mpas_f32_to_f64_ieee(da);
    const double bb = (ad_neg(v) ? mpas_f32_to_f64_ieee(dv) : 0.0) * mpas_f32_to_f64_ieee(b)
                    + mpas_f32_to_f64_ieee(ad_minus(v)) * mpas_f32_to_f64_ieee(db);
    return mpas_f32_to_f64_ieee(ddt) * mpas_f32_to_f64_ieee(ad_fhat(v, a, b)) + mpas_f32_to_f64_ieee(dt) * (aa + bb);
}

#include "ad_vertical_branches.cuh"

// Recompute a residual tangent in f64 so that each public f32 output is
// rounded only once, including when the residual feeds a scale output.
__device__ __forceinline__ double ad_residual_tl(
    int t, int k, int c, int nl, int nc, float dt, float ddt,
    const float *source, const float *velocity, const float *high,
    const float *d_source, const float *d_velocity, const float *d_high)
{
    const int qi = ad_qi(t,k,c,nl,nc);
    double value = mpas_f32_to_f64_ieee(ddt) * mpas_f32_to_f64_ieee(high[qi]) + mpas_f32_to_f64_ieee(dt) * mpas_f32_to_f64_ieee(d_high[qi]);
    if (k > 0 && k < nl) {
        const int vi = k * nc + c;
        const int a = ad_q(t,k-1,c,nl,nc), b = ad_q(t,k,c,nl,nc);
        value -= ad_flux_tl(dt,ddt,velocity[vi],d_velocity[vi],source[a],d_source[a],source[b],d_source[b]);
    }
    return value;
}

// Cotangent of one residual after reversing both neighboring scale outputs.
__device__ __forceinline__ double ad_bar_residual(
    int t, int k, int c, int nl, int nc, const float *rdzw,
    const float *residual, const float *b_residual,
    const float *b_scale_in, const float *b_scale_out)
{
    const int qi = ad_qi(t,k,c,nl,nc);
    const float r = residual[qi];
    double b = mpas_f32_to_f64_ieee(b_residual[qi]);
    if (k > 0) {
        const int q = ad_q(t,k-1,c,nl,nc);
        const double g = (ad_neg(r) ? mpas_f32_to_f64_ieee(b_scale_in[q]) : 0.0)
                       + (ad_pos(r) ? mpas_f32_to_f64_ieee(b_scale_out[q]) : 0.0);
        b -= mpas_f32_to_f64_ieee(rdzw[k-1]) * g;
    }
    if (k < nl) {
        const int q = ad_q(t,k,c,nl,nc);
        const double g = (ad_pos(r) ? mpas_f32_to_f64_ieee(b_scale_in[q]) : 0.0)
                       + (ad_neg(r) ? mpas_f32_to_f64_ieee(b_scale_out[q]) : 0.0);
        b += mpas_f32_to_f64_ieee(rdzw[k]) * g;
    }
    return b;
}
__device__ __forceinline__ double ad_bar_flux(
    int t, int k, int c, int nl, int nc, const float *rdzw,
    const float *residual, const float *b_mass, const float *b_residual,
    const float *b_scale_in, const float *b_scale_out)
{
    const double br = ad_bar_residual(t,k,c,nl,nc,rdzw,residual,b_residual,b_scale_in,b_scale_out);
    return -br + mpas_f32_to_f64_ieee(rdzw[k]) * mpas_f32_to_f64_ieee(b_mass[ad_q(t,k,c,nl,nc)])
               - mpas_f32_to_f64_ieee(rdzw[k-1]) * mpas_f32_to_f64_ieee(b_mass[ad_q(t,k-1,c,nl,nc)]);
}

#ifdef AD_BUILD_TL
extern "C" __global__ void fct_vertical_low_order_v841__tl(
    int ntracers, int nlev, int ncells, float dt,
    const float *source_old, const float *rho_old, const float *velocity,
    const float *rdzw, const float *high_vertical, const float *mass,
    const float *vertical_residual, const float *scale_in, const float *scale_out,
    const float *d_dt, const float *d_source, const float *d_rho, const float *d_velocity,
    const float *d_rdzw, const float *d_high, float *d_mass, float *d_residual,
    float *d_scale_in, float *d_scale_out, unsigned int *branch_record, AdTransportFailure *failure)
{
    (void)mass; (void)scale_in; (void)scale_out;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const bool outside = i >= ntracers * ncells;
    if (branch_record) ad_ib_record(branch_record,0,outside);
    if (outside) return;
    const int c = i % ncells, t = i / ncells;
    if (branch_record) ad_vr_trace(t,c,nlev,ncells,dt,source_old,rho_old,velocity,rdzw,high_vertical,vertical_residual,branch_record);
    if (!isfinite(dt) || !isfinite(*d_dt)) {
        ad_transport_fail(failure, i, 1, 0, 1); return;
    }
    for (int k = 0; k < nlev; ++k) {
        const int q = ad_q(t,k,c,nlev,ncells), cc = k * ncells + c;
        if (!isfinite(source_old[q]) || !isfinite(d_source[q]) ||
            !isfinite(rho_old[cc]) || !isfinite(d_rho[cc]) ||
            !isfinite(rdzw[k]) || !isfinite(d_rdzw[k])) {
            ad_transport_fail(failure, i, 2, q, 2); return;
        }
    }
    for (int k = 0; k <= nlev; ++k) {
        const int qi = ad_qi(t,k,c,nlev,ncells);
        if (!isfinite(high_vertical[qi]) || !isfinite(d_high[qi]) ||
            !isfinite(vertical_residual[qi])) {
            ad_transport_fail(failure, i, 6, qi, 3); return;
        }
    }
    for (int k = 1; k < nlev; ++k) {
        const int vi = k * ncells + c;
        if (!isfinite(velocity[vi]) || !isfinite(d_velocity[vi])) {
            ad_transport_fail(failure, i, 4, vi, 4); return;
        }
    }
    for (int k = 0; k <= nlev; ++k) {
        const int qi = ad_qi(t,k,c,nlev,ncells);
        const double dr = ad_residual_tl(t,k,c,nlev,ncells,dt,*d_dt,
                                        source_old,velocity,high_vertical,d_source,d_velocity,d_high);
        d_residual[qi] = mpas_f64_to_f32_ieee(dr);
        if (!isfinite(d_residual[qi])) { ad_transport_fail(failure,i,6,qi,5); return; }
    }
    for (int k = 0; k < nlev; ++k) {
        const int q = ad_q(t,k,c,nlev,ncells), cc = k * ncells + c;
        const int lo = ad_qi(t,k,c,nlev,ncells), hi = ad_qi(t,k+1,c,nlev,ncells);
        double dm = mpas_f32_to_f64_ieee(d_source[q])*mpas_f32_to_f64_ieee(rho_old[cc]) + mpas_f32_to_f64_ieee(source_old[q])*mpas_f32_to_f64_ieee(d_rho[cc]);
        if (k + 1 < nlev) {
            const int b = ad_q(t,k+1,c,nlev,ncells), vi = (k+1)*ncells+c;
            const float v = velocity[vi], flux = mpas_mul(dt,ad_fhat(v,source_old[q],source_old[b]));
            const double df = ad_flux_tl(dt,*d_dt,v,d_velocity[vi],source_old[q],d_source[q],source_old[b],d_source[b]);
            dm -= df*mpas_f32_to_f64_ieee(rdzw[k]) + mpas_f32_to_f64_ieee(flux)*mpas_f32_to_f64_ieee(d_rdzw[k]);
        }
        if (k > 0) {
            const int a = ad_q(t,k-1,c,nlev,ncells), vi = k*ncells+c;
            const float v = velocity[vi], flux = mpas_mul(dt,ad_fhat(v,source_old[a],source_old[q]));
            const double df = ad_flux_tl(dt,*d_dt,v,d_velocity[vi],source_old[a],d_source[a],source_old[q],d_source[q]);
            dm += df*mpas_f32_to_f64_ieee(rdzw[k]) + mpas_f32_to_f64_ieee(flux)*mpas_f32_to_f64_ieee(d_rdzw[k]);
        }
        d_mass[q] = mpas_f64_to_f32_ieee(dm);
        const float l = vertical_residual[lo], u = vertical_residual[hi];
        const double dl = ad_residual_tl(t,k,c,nlev,ncells,dt,*d_dt,source_old,velocity,high_vertical,d_source,d_velocity,d_high);
        const double du = ad_residual_tl(t,k+1,c,nlev,ncells,dt,*d_dt,source_old,velocity,high_vertical,d_source,d_velocity,d_high);
        const double di = (ad_neg(u) ? du : 0.0) - (ad_pos(l) ? dl : 0.0);
        const double dout = (ad_pos(u) ? du : 0.0) - (ad_neg(l) ? dl : 0.0);
        d_scale_in[q] = mpas_f64_to_f32_ieee((-mpas_f32_to_f64_ieee(d_rdzw[k])*(mpas_f32_to_f64_ieee(ad_minus(u))-mpas_f32_to_f64_ieee(ad_plus(l))) - mpas_f32_to_f64_ieee(rdzw[k])*di));
        d_scale_out[q] = mpas_f64_to_f32_ieee((-mpas_f32_to_f64_ieee(d_rdzw[k])*(mpas_f32_to_f64_ieee(ad_plus(u))-mpas_f32_to_f64_ieee(ad_minus(l))) - mpas_f32_to_f64_ieee(rdzw[k])*dout));
        if (!isfinite(d_mass[q]) || !isfinite(d_scale_in[q]) || !isfinite(d_scale_out[q])) {
            ad_transport_fail(failure, i, 7, q, 5); return;
        }
    }
}
#endif

#ifdef AD_BUILD_AD
extern "C" __global__ void fct_vertical_low_order_v841__ad(
    int ntracers, int nlev, int ncells, float dt,
    const float *source_old, const float *rho_old, const float *velocity,
    const float *rdzw, const float *high_vertical, const float *mass,
    const float *vertical_residual, const float *scale_in, const float *scale_out,
    float *b_dt, float *b_source, float *b_rho, float *b_velocity,
    float *b_rdzw, float *b_high,
    float *b_mass_live, float *b_residual_live, float *b_scale_in_live, float *b_scale_out_live,
    const float *b_mass, const float *b_residual, const float *b_scale_in, const float *b_scale_out,
    unsigned int *branch_record, AdTransportFailure *failure)
{
    (void)mass; (void)scale_in; (void)scale_out;
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (branch_record) ad_ib_record(branch_record,0,c >= ntracers * ncells);
    const unsigned long long launched = (unsigned long long)gridDim.x * blockDim.x;
    if (ntracers <= 0 || ncells <= 0 || c >= ncells) return;
    const int owners = launched < (unsigned long long)ncells ? (int)launched : ncells;
    if (branch_record) for (int t=0;t<ntracers && (unsigned long long)t*ncells+c<launched;++t) ad_vr_trace(t,c,nlev,ncells,dt,source_old,rho_old,velocity,rdzw,high_vertical,vertical_residual,branch_record);
    if (!isfinite(dt)) { ad_transport_fail(failure,c,1,0,1); return; }
    for (int t=0;t<ntracers && (unsigned long long)t*ncells+c<launched;++t) {
        for (int k=0;k<nlev;++k) {
            int q=ad_q(t,k,c,nlev,ncells),cc=k*ncells+c;
            if (!isfinite(source_old[q]) || !isfinite(rho_old[cc]) || !isfinite(rdzw[k]) ||
                !isfinite(b_mass[q]) || !isfinite(b_scale_in[q]) || !isfinite(b_scale_out[q]) || !isfinite(b_source[q])) {
                ad_transport_fail(failure,c,2,q,2); return;
            }
        }
        for (int k=0;k<=nlev;++k) {
            int qi=ad_qi(t,k,c,nlev,ncells);
            if (!isfinite(high_vertical[qi]) || !isfinite(vertical_residual[qi]) || !isfinite(b_residual[qi]) || !isfinite(b_high[qi])) {
                ad_transport_fail(failure,c,6,qi,3); return;
            }
        }
    }
    for (int k=1;k<nlev;++k) {
        int vi=k*ncells+c;
        if (!isfinite(velocity[vi]) || !isfinite(b_velocity[vi])) { ad_transport_fail(failure,c,4,vi,4); return; }
    }
    // This thread owns one cell. Combine every contribution to a source,
    // density, velocity, or high-flux cotangent in f64 and round once.
    for (int k=0;k<nlev;++k) {
        int cc=k*ncells+c; double density_bar=mpas_f32_to_f64_ieee(b_rho[cc]);
        for (int t=0;t<ntracers && (unsigned long long)t*ncells+c<launched;++t) {
            int q=ad_q(t,k,c,nlev,ncells);
            double g=mpas_f32_to_f64_ieee(b_source[q])+mpas_f32_to_f64_ieee(rho_old[cc])*mpas_f32_to_f64_ieee(b_mass[q]);
            if (k+1<nlev) {
                float v=velocity[(k+1)*ncells+c];
                double bf=ad_bar_flux(t,k+1,c,nlev,ncells,rdzw,vertical_residual,b_mass,b_residual,b_scale_in,b_scale_out);
                g+=mpas_f32_to_f64_ieee(dt)*mpas_f32_to_f64_ieee(ad_plus(v))*bf;
            }
            if (k>0) {
                float v=velocity[k*ncells+c];
                double bf=ad_bar_flux(t,k,c,nlev,ncells,rdzw,vertical_residual,b_mass,b_residual,b_scale_in,b_scale_out);
                g+=mpas_f32_to_f64_ieee(dt)*mpas_f32_to_f64_ieee(ad_minus(v))*bf;
            }
            b_source[q]=mpas_f64_to_f32_ieee(g);
            density_bar+=mpas_f32_to_f64_ieee(source_old[q])*mpas_f32_to_f64_ieee(b_mass[q]);
            if(!isfinite(b_source[q])) { ad_transport_fail(failure,c,2,q,5); return; }
        }
        b_rho[cc]=mpas_f64_to_f32_ieee(density_bar);
        if(!isfinite(b_rho[cc])) { ad_transport_fail(failure,c,3,cc,5); return; }
    }
    for (int k=0;k<=nlev;++k) {
        int vi=k*ncells+c; double velocity_bar=(k>0&&k<nlev)?mpas_f32_to_f64_ieee(b_velocity[vi]):0.0;
        for (int t=0;t<ntracers && (unsigned long long)t*ncells+c<launched;++t) {
            int qi=ad_qi(t,k,c,nlev,ncells);
            double br=ad_bar_residual(t,k,c,nlev,ncells,rdzw,vertical_residual,b_residual,b_scale_in,b_scale_out);
            b_high[qi]=mpas_f64_to_f32_ieee((mpas_f32_to_f64_ieee(b_high[qi])+mpas_f32_to_f64_ieee(dt)*br));
            if(!isfinite(b_high[qi])) { ad_transport_fail(failure,c,6,qi,5); return; }
            if(k>0&&k<nlev) {
                float v=velocity[vi];int a=ad_q(t,k-1,c,nlev,ncells),b=ad_q(t,k,c,nlev,ncells);
                double bf=ad_bar_flux(t,k,c,nlev,ncells,rdzw,vertical_residual,b_mass,b_residual,b_scale_in,b_scale_out);
                double upwind=(ad_pos(v)?mpas_f32_to_f64_ieee(source_old[a]):0.0)+(ad_neg(v)?mpas_f32_to_f64_ieee(source_old[b]):0.0);
                velocity_bar+=mpas_f32_to_f64_ieee(dt)*upwind*bf;
            }
        }
        if(k>0&&k<nlev) {
            b_velocity[vi]=mpas_f64_to_f32_ieee(velocity_bar);
            if(!isfinite(b_velocity[vi])) { ad_transport_fail(failure,c,4,vi,5); return; }
        }
    }
    // One owner per level, in fixed cell/tracer order.
    for (int k=c;k<nlev;k+=owners) {
        double sum=mpas_f32_to_f64_ieee(b_rdzw[k]);
        for (int cell=0;cell<ncells;++cell) for(int t=0;t<ntracers;++t) {
            if ((unsigned long long)t*ncells+cell >= launched) continue;
            int q=ad_q(t,k,cell,nlev,ncells);
            float l=vertical_residual[ad_qi(t,k,cell,nlev,ncells)];
            float u=vertical_residual[ad_qi(t,k+1,cell,nlev,ncells)];
            double g=-mpas_f32_to_f64_ieee(b_scale_in[q])*(mpas_f32_to_f64_ieee(ad_minus(u))-mpas_f32_to_f64_ieee(ad_plus(l)))
                     -mpas_f32_to_f64_ieee(b_scale_out[q])*(mpas_f32_to_f64_ieee(ad_plus(u))-mpas_f32_to_f64_ieee(ad_minus(l)));
            if(k+1<nlev){float v=velocity[(k+1)*ncells+cell];float f=mpas_mul(dt,ad_fhat(v,source_old[q],source_old[ad_q(t,k+1,cell,nlev,ncells)]));g-=mpas_f32_to_f64_ieee(f)*mpas_f32_to_f64_ieee(b_mass[q]);}
            if(k>0){float v=velocity[k*ncells+cell];float f=mpas_mul(dt,ad_fhat(v,source_old[ad_q(t,k-1,cell,nlev,ncells)],source_old[q]));g+=mpas_f32_to_f64_ieee(f)*mpas_f32_to_f64_ieee(b_mass[q]);}
            sum+=g;
        }
        b_rdzw[k]=mpas_f64_to_f32_ieee(sum);
        if(!isfinite(b_rdzw[k])) { ad_transport_fail(failure,c,5,k,5); return; }
    }
    if(c==0) {
        double sum=mpas_f32_to_f64_ieee(*b_dt);
        for(int cell=0;cell<ncells;++cell) for(int t=0;t<ntracers;++t) for(int k=0;k<=nlev;++k) {
            if ((unsigned long long)t*ncells+cell >= launched) continue;
            int qi=ad_qi(t,k,cell,nlev,ncells);
            double br=ad_bar_residual(t,k,cell,nlev,ncells,rdzw,vertical_residual,b_residual,b_scale_in,b_scale_out);
            sum+=br*mpas_f32_to_f64_ieee(high_vertical[qi]);
            if(k>0&&k<nlev) {
                float v=velocity[k*ncells+cell];
                float h=ad_fhat(v,source_old[ad_q(t,k-1,cell,nlev,ncells)],source_old[ad_q(t,k,cell,nlev,ncells)]);
                double bf=ad_bar_flux(t,k,cell,nlev,ncells,rdzw,vertical_residual,b_mass,b_residual,b_scale_in,b_scale_out);
                sum+=bf*mpas_f32_to_f64_ieee(h);
            }
        }
        *b_dt=mpas_f64_to_f32_ieee(sum);
        if(!isfinite(*b_dt)) { ad_transport_fail(failure,0,1,0,5); return; }
    }
    // Global reductions read disjoint scheduler-copied snapshots, so
    // each owning cell can consume its live output seeds without a race.
    for (int t=0;t<ntracers && (unsigned long long)t*ncells+c<launched;++t) {
        for (int k=0;k<nlev;++k) {
            const int q=ad_q(t,k,c,nlev,ncells);
            b_mass_live[q]=0.0f; b_scale_in_live[q]=0.0f; b_scale_out_live[q]=0.0f;
        }
        for (int k=0;k<=nlev;++k) b_residual_live[ad_qi(t,k,c,nlev,ncells)]=0.0f;
    }
}
#endif
