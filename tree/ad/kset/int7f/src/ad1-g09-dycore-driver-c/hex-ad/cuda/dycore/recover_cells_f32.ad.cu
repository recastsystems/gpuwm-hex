// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 19
#define AD_TRACE_ADD_CANCEL 20
#define AD_TRACE_SUB_SUBNORMAL 21
#define AD_TRACE_SUB_CANCEL 22
#define AD_TRACE_MUL_NONZERO 23
#define AD_TRACE_DIV_UNDERFLOW 24
#include "branch_trace.cuh"

enum RecFailureSlot {
    REC_RPS = 0,
    REC_RTPS = 1,
    REC_RPP = 2,
    REC_RTPP = 3,
    REC_RB = 4,
    REC_RTB = 5,
    REC_DP = 6,
    REC_RHO = 7,
    REC_RHO_ZERO = 8,
    REC_DT = 9,
    REC_DIAB = 10,
    REC_TP = 11,
    REC_RT = 12,
    REC_THETA = 13,
    REC_RGAS = 14,
    REC_CP = 15,
    REC_P0 = 16,
    REC_P0_ZERO = 17,
    REC_CP_RGAS_ZERO = 18,
    REC_Z = 19,
    REC_EB = 20,
    REC_ARG = 21,
    REC_POW_DOMAIN = 22,
    REC_POWER = 23,
    REC_EX = 24,
    REC_POW_DX = 25,
    REC_POW_DP = 26,
    REC_ES = 27,
    REC_PS = 28,
    REC_PRESSURE = 29,
    REC_SEED_RHO = 30,
    REC_SEED_RT = 31,
    REC_SEED_DP = 32,
    REC_SEED_TP = 33,
    REC_SEED_THETA = 34,
    REC_SEED_EX = 35,
    REC_SEED_PRESSURE = 36,
    REC_BAR_RPS = 37,
    REC_BAR_RTPS = 38,
    REC_BAR_RPP = 39,
    REC_BAR_RTPP = 40,
    REC_BAR_DIAB = 41,
    REC_BAR_RB = 42,
    REC_BAR_RTB = 43,
    REC_BAR_EB = 44,
    REC_BAR_Z = 45,
    REC_BAR_ES = 46,
    REC_BAR_PS = 47,
    REC_BAR_RGAS = 48,
    REC_BAR_CP = 49,
    REC_BAR_P0 = 50,
    REC_BAR_DT = 51,
    REC_TAN_DP = 52,
    REC_TAN_RHO = 53,
    REC_TAN_TP = 54,
    REC_TAN_RT = 55,
    REC_TAN_THETA = 56,
    REC_TAN_EX = 57,
    REC_TAN_PRESSURE = 58,
    REC_ACC_RPS = 59,
    REC_ACC_RTPS = 60,
    REC_ACC_RPP = 61,
    REC_ACC_RTPP = 62,
    REC_ACC_DIAB = 63,
    REC_ACC_RB = 64,
    REC_ACC_RTB = 65,
    REC_ACC_EB = 66,
    REC_ACC_Z = 67,
    REC_ACC_ES = 68,
    REC_ACC_PS = 69,
    REC_SCALAR_RGAS = 70,
    REC_SCALAR_CP = 71,
    REC_SCALAR_P0 = 72,
    REC_SCALAR_DT = 73,
};
__device__ __forceinline__ bool rec_bad(int &slot, int cause) { slot = cause; return false; }

struct RecInput {
    float rps, rtps, rpp, rtpp, diab, rb, rtb, eb, z, es, ps;
};
struct RecState {
    float dp, tp, rho, rt, theta, ex, pressure, argument, power, pow_dx, pow_dp;
};
struct RecSeeds { float rho, rt, dp, tp, theta, ex, pressure; };
struct RecDoubleInput {
    double rps, rtps, rpp, rtpp, diab, rb, rtb, eb, z, es, ps;
};
struct RecBars {
    RecDoubleInput x;
    double rgas, cp, p0, dt;
};

__device__ __forceinline__ RecInput rec_load(int i,
    const float *rps, const float *rtps, const float *rpp, const float *rtpp,
    const float *diab, const float *rb, const float *rtb, const float *eb,
    const float *z, const float *es, const float *ps)
{
    RecInput a = {rps[i],rtps[i],rpp[i],rtpp[i],diab[i],rb[i],rtb[i],eb[i],z[i],es[i],ps[i]};
    return a;
}

// Recompute the exact branch's primal intermediates. The build uses
// --ftz=false and --fmad=false: these IEEE basic operations match the pinned
// guarded mpas_add/sub/mul/div on finite inputs, including subnormals.
__device__ __forceinline__ bool rec_state(const RecInput &x, int stage,
        int native_pow, int has_diabatic, float rgas, float cp, float p0,
        float dt, float executed_exner, RecState &s, int &bad)
{
    (void)native_pow;
    if (!isfinite(x.rps)) return rec_bad(bad, REC_RPS);
    if (!isfinite(x.rtps)) return rec_bad(bad, REC_RTPS);
    if (!isfinite(x.rpp)) return rec_bad(bad, REC_RPP);
    if (!isfinite(x.rtpp)) return rec_bad(bad, REC_RTPP);
    if (!isfinite(x.rb)) return rec_bad(bad, REC_RB);
    if (!isfinite(x.rtb)) return rec_bad(bad, REC_RTB);
    s.dp = x.rps+x.rpp;
    s.tp = x.rtps+x.rtpp;
    s.rho = s.dp+x.rb;
    if (!isfinite(s.dp)) return rec_bad(bad, REC_DP);
    if (!isfinite(s.rho)) return rec_bad(bad, REC_RHO);
    if (s.rho == 0.0f) return rec_bad(bad, REC_RHO_ZERO);
    if (stage == 3 && has_diabatic) {
        if (!isfinite(dt)) return rec_bad(bad, REC_DT);
        if (!isfinite(x.diab)) return rec_bad(bad, REC_DIAB);
        s.tp = s.tp - ((dt*s.rho)*x.diab);
    }
    s.rt = s.tp+x.rtb;
    s.theta = s.rt/s.rho;
    if (!isfinite(s.tp)) return rec_bad(bad, REC_TP);
    if (!isfinite(s.rt)) return rec_bad(bad, REC_RT);
    if (!isfinite(s.theta)) return rec_bad(bad, REC_THETA);
    s.argument = s.power = s.pow_dx = s.pow_dp = 0.0f;
    if (stage == 3) {
        if (!isfinite(rgas)) return rec_bad(bad, REC_RGAS);
        if (!isfinite(cp)) return rec_bad(bad, REC_CP);
        if (!isfinite(p0)) return rec_bad(bad, REC_P0);
        if (p0 == 0.0f) return rec_bad(bad, REC_P0_ZERO);
        if (cp == rgas) return rec_bad(bad, REC_CP_RGAS_ZERO);
        if (!isfinite(x.z)) return rec_bad(bad, REC_Z);
        if (!isfinite(x.eb)) return rec_bad(bad, REC_EB);
        s.argument = (x.z*(rgas/p0))*s.rt;
        s.power = rgas/(cp-rgas);
        if (!isfinite(s.argument)) return rec_bad(bad, REC_ARG);
        if (!(s.argument > 0.0f)) return rec_bad(bad, REC_POW_DOMAIN);
        if (!isfinite(s.power)) return rec_bad(bad, REC_POWER);
        if (!isfinite(executed_exner)) return rec_bad(bad, REC_EX);
        // Differentiate the executed forward output, without copying libm.
        const double value = (double)executed_exner;
        s.ex = executed_exner;
        s.pow_dx = (float)((double)s.power * (value / (double)s.argument));
        s.pow_dp = (float)(value * log((double)s.argument));
        if (!isfinite(s.pow_dx)) return rec_bad(bad, REC_POW_DX);
        if (!isfinite(s.pow_dp)) return rec_bad(bad, REC_POW_DP);
        s.pressure = (x.z*rgas)*((s.ex*s.tp)+(x.rtb*(s.ex-x.eb)));
    } else {
        s.ex = x.es; s.pressure = x.ps;
        if (!isfinite(s.ex)) return rec_bad(bad, REC_ES);
        if (!isfinite(s.pressure)) return rec_bad(bad, REC_PS);
    }
    if (!isfinite(s.pressure)) return rec_bad(bad, REC_PRESSURE);
    return true;
}

// Hand reverse for one element; reused by the scalar-parameter reduction.
__device__ __forceinline__ bool rec_reverse(const RecInput &x, const RecState &s,
        const RecSeeds &y, int stage, int has_diabatic,
        float rgas, float cp, float p0, float dt, RecBars &g, int &bad)
{
    if (!isfinite(y.rho)) return rec_bad(bad, REC_SEED_RHO);
    if (!isfinite(y.rt)) return rec_bad(bad, REC_SEED_RT);
    if (!isfinite(y.dp)) return rec_bad(bad, REC_SEED_DP);
    if (!isfinite(y.tp)) return rec_bad(bad, REC_SEED_TP);
    if (!isfinite(y.theta)) return rec_bad(bad, REC_SEED_THETA);
    if (!isfinite(y.ex)) return rec_bad(bad, REC_SEED_EX);
    if (!isfinite(y.pressure)) return rec_bad(bad, REC_SEED_PRESSURE);
    g = {};
    double brho = (double)y.rho - (double)y.theta*(s.rt/((double)s.rho*s.rho));
    double brt = (double)y.rt + (double)y.theta/s.rho;
    double bdp = y.dp, btp = y.tp;
    if (stage == 3) {
        const float scale = x.z*rgas;
        const float inner = s.ex*s.tp + x.rtb*(s.ex-x.eb);
        const double bscale = (double)y.pressure*inner, binner = (double)y.pressure*scale;
        g.x.z += bscale*rgas;
        g.rgas += bscale*x.z;
        const double bex = (double)y.ex + binner*(s.tp+x.rtb);
        btp += binner*s.ex;
        g.x.rtb += binner*(s.ex-x.eb);
        g.x.eb -= binner*x.rtb;
        const double barg = bex*s.pow_dx, bp = bex*s.pow_dp;
        const float a = rgas/p0;
        g.x.z += barg*a*s.rt;
        g.rgas += barg*x.z*s.rt/p0;
        g.p0 -= barg*x.z*s.rt*rgas/(p0*p0);
        brt += barg*(x.z*a);
        const float denom = cp-rgas;
        g.rgas += bp*(1.0f/denom + rgas/(denom*denom));
        g.cp -= bp*rgas/(denom*denom);
    } else {
        g.x.es += y.ex;
        g.x.ps += y.pressure;
    }
    btp += brt;
    g.x.rtb += brt;
    if (stage == 3 && has_diabatic) {
        g.dt -= btp*s.rho*x.diab;
        brho -= btp*dt*x.diab;
        g.x.diab -= btp*dt*s.rho;
    }
    g.x.rtps += btp;
    g.x.rtpp += btp;
    bdp += brho;
    g.x.rb += brho;
    g.x.rps += bdp;
    g.x.rpp += bdp;
    if (!isfinite(g.x.rps)) return rec_bad(bad, REC_BAR_RPS);
    if (!isfinite(g.x.rtps)) return rec_bad(bad, REC_BAR_RTPS);
    if (!isfinite(g.x.rpp)) return rec_bad(bad, REC_BAR_RPP);
    if (!isfinite(g.x.rtpp)) return rec_bad(bad, REC_BAR_RTPP);
    if (!isfinite(g.x.diab)) return rec_bad(bad, REC_BAR_DIAB);
    if (!isfinite(g.x.rb)) return rec_bad(bad, REC_BAR_RB);
    if (!isfinite(g.x.rtb)) return rec_bad(bad, REC_BAR_RTB);
    if (!isfinite(g.x.eb)) return rec_bad(bad, REC_BAR_EB);
    if (!isfinite(g.x.z)) return rec_bad(bad, REC_BAR_Z);
    if (!isfinite(g.x.es)) return rec_bad(bad, REC_BAR_ES);
    if (!isfinite(g.x.ps)) return rec_bad(bad, REC_BAR_PS);
    if (!isfinite(g.rgas)) return rec_bad(bad, REC_BAR_RGAS);
    if (!isfinite(g.cp)) return rec_bad(bad, REC_BAR_CP);
    if (!isfinite(g.p0)) return rec_bad(bad, REC_BAR_P0);
    if (!isfinite(g.dt)) return rec_bad(bad, REC_BAR_DT);
    return true;
}


// Re-evaluate the captured forward's control predicates and guarded
// arithmetic; no value from this diagnostic is written to the model state.
__device__ __forceinline__ void trace_forward(const RecInput &x,int stage,int native,int heat,float rgas,float cp,float p0,float dt,float executed_exner,uint32_t *r) {
 ad_branch(r,0,true);float dp=ad_add(x.rps,x.rpp,r,6);float tp=ad_add(x.rtps,x.rtpp,r,6);float rho=ad_add(dp,x.rb,r,6);
 bool heating=stage==3 && heat;ad_branch(r,1,heating);
 if(heating) tp=ad_add(tp,ad_mul(ad_mul(dt,rho,r,8),x.diab,r,8),r,7,true);
 float rt=ad_add(tp,x.rtb,r,6),th=ad_div(rt,rho,r,9),ex=x.es,pp=x.ps;
 ad_branch(r,2,stage==3);
 if(stage==3){
  float arg=ad_mul(ad_mul(x.z,ad_div(rgas,p0,r,9),r,8),rt,r,8);
  (void)ad_div(rgas,ad_add(cp,rgas,r,7,true),r,9);ad_branch(r,3,native!=0);
  if(native){ unsigned bits=__float_as_uint(arg);ad_branch(r,4,bits>=0x00800000u && bits<0x7f800000u); }
  // Library-internal lookup slots 10..18 require a forward trace. They are
  // intentionally unrecorded here; zero counters mean unavailable.
  ex=executed_exner;
  float a=ad_mul(ex,tp,r,8),b=ad_mul(x.rtb,ad_add(ex,x.eb,r,7,true),r,8);
  pp=ad_mul(ad_mul(x.z,rgas,r,8),ad_add(a,b,r,6),r,8);
 }
 ad_branch(r,5,isfinite(dp)&&isfinite(rho)&&rho!=0.0f&&isfinite(tp)&&isfinite(rt)&&isfinite(th)&&isfinite(ex)&&isfinite(pp));
}

#define REC_FORWARD_ARGS \
    int nlev, int ncells, int stage, int native_gnu_powf, \
    float rgas, float cp, float p0, \
    const float *rho_p_saved, const float *rtheta_p_saved, \
    const float *rho_pp, const float *rtheta_pp, \
    int has_diabatic, float dt, const float *diabatic_theta, \
    const float *rho_base, const float *rtheta_base, \
    const float *exner_base, const float *zz, \
    const float *exner_saved, const float *pressure_p_saved, \
    float *rho, float *rtheta, float *rho_p, float *rtheta_p, \
    float *theta, float *exner, float *pressure_p
#define REC_INPUT(i) rec_load(i,rho_p_saved,rtheta_p_saved,rho_pp,rtheta_pp,diabatic_theta, \
    rho_base,rtheta_base,exner_base,zz,exner_saved,pressure_p_saved)

extern "C" __global__ void recover_cells_f32__tl(
    REC_FORWARD_ARGS,
    const float *d_rgas_ptr, const float *d_cp_ptr, const float *d_p0_ptr, const float *d_dt_ptr,
    const float *d_rho_p_saved, const float *d_rtheta_p_saved,
    const float *d_rho_pp, const float *d_rtheta_pp, const float *d_diabatic_theta,
    const float *d_rho_base, const float *d_rtheta_base, const float *d_exner_base,
    const float *d_zz, const float *d_exner_saved, const float *d_pressure_p_saved,
    float *d_rho, float *d_rtheta, float *d_rho_p, float *d_rtheta_p,
    float *d_theta, float *d_exner, float *d_pressure_p,
    int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<nlev*ncells;tr+=blockDim.x*gridDim.x) trace_forward(REC_INPUT(tr),stage,native_gnu_powf,has_diabatic,rgas,cp,p0,dt,exner[tr],branch_record);
    const float d_rgas=*d_rgas_ptr, d_cp=*d_cp_ptr, d_p0=*d_p0_ptr, d_dt=*d_dt_ptr;
    (void)rho; (void)rtheta; (void)rho_p; (void)rtheta_p;
    (void)theta; (void)pressure_p;
    const int n = nlev*ncells;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        const RecInput x = REC_INPUT(i);
        const RecInput d = rec_load(i,d_rho_p_saved,d_rtheta_p_saved,d_rho_pp,d_rtheta_pp,
            d_diabatic_theta,d_rho_base,d_rtheta_base,d_exner_base,d_zz,d_exner_saved,d_pressure_p_saved);
        RecState s; int bad = 0;
        if (!rec_state(x,stage,native_gnu_powf,has_diabatic,rgas,cp,p0,dt,exner[i],s,bad)) {
            ad_refuse(failure, i, 8, bad); continue;
        }
        const float ddp = d.rps+d.rpp, drho = ddp+d.rb;
        float dtp = d.rtps+d.rtpp;
        if (stage == 3 && has_diabatic)
            dtp -= (d_dt*s.rho*x.diab + dt*drho*x.diab + dt*s.rho*d.diab);
        const float drt = dtp+d.rtb;
        const float dth = drt/s.rho - (s.rt*drho)/(s.rho*s.rho);
        float dex, dpressure;
        if (stage == 3) {
            const float a = rgas/p0;
            const float da = d_rgas/p0 - rgas*d_p0/(p0*p0);
            const float darg = (d.z*a+x.z*da)*s.rt + (x.z*a)*drt;
            const float den = cp-rgas;
            const float dpower = d_rgas/den - rgas*(d_cp-d_rgas)/(den*den);
            dex = s.pow_dx*darg+s.pow_dp*dpower;
            const float scale = x.z*rgas, dscale = d.z*rgas+x.z*d_rgas;
            const float inner = s.ex*s.tp+x.rtb*(s.ex-x.eb);
            const float dinner = dex*s.tp+s.ex*dtp+d.rtb*(s.ex-x.eb)+x.rtb*(dex-d.eb);
            dpressure = dscale*inner+scale*dinner;
        } else { dex = d.es; dpressure = d.ps; }
        if (!isfinite(ddp)) { ad_refuse(failure, i, 8, REC_TAN_DP); continue; }
        if (!isfinite(drho)) { ad_refuse(failure, i, 8, REC_TAN_RHO); continue; }
        if (!isfinite(dtp)) { ad_refuse(failure, i, 8, REC_TAN_TP); continue; }
        if (!isfinite(drt)) { ad_refuse(failure, i, 8, REC_TAN_RT); continue; }
        if (!isfinite(dth)) { ad_refuse(failure, i, 8, REC_TAN_THETA); continue; }
        if (!isfinite(dex)) { ad_refuse(failure, i, 8, REC_TAN_EX); continue; }
        if (!isfinite(dpressure)) { ad_refuse(failure, i, 8, REC_TAN_PRESSURE); continue; }
        d_rho_p[i]=ddp; d_rtheta_p[i]=dtp; d_rho[i]=drho; d_rtheta[i]=drt;
        d_theta[i]=dth; d_exner[i]=dex; d_pressure_p[i]=dpressure;
    }
}

extern "C" __global__ void recover_cells_f32__ad(
    REC_FORWARD_ARGS,
    float *b_rho, float *b_rtheta, float *b_rho_p,
    float *b_rtheta_p, float *b_theta, float *b_exner,
    float *b_pressure_p,
    float *b_rgas, float *b_cp, float *b_p0, float *b_dt,
    float *b_rho_p_saved, float *b_rtheta_p_saved, float *b_rho_pp,
    float *b_rtheta_pp, float *b_diabatic_theta, float *b_rho_base,
    float *b_rtheta_base, float *b_exner_base, float *b_zz,
    float *b_exner_saved, float *b_pressure_p_saved, int *failure, uint32_t *branch_record,
    unsigned int *completed)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<nlev*ncells;tr+=blockDim.x*gridDim.x) trace_forward(REC_INPUT(tr),stage,native_gnu_powf,has_diabatic,rgas,cp,p0,dt,exner[tr],branch_record);
    (void)rho; (void)rtheta; (void)rho_p; (void)rtheta_p;
    (void)theta; (void)pressure_p;
    const int n=nlev*ncells;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x) {
        const RecInput x=REC_INPUT(i);
        const RecSeeds y={b_rho[i],b_rtheta[i],b_rho_p[i],b_rtheta_p[i],b_theta[i],b_exner[i],b_pressure_p[i]};
        RecState s; RecBars g; int bad = 0;
        if (!rec_state(x,stage,native_gnu_powf,has_diabatic,rgas,cp,p0,dt,exner[i],s,bad) ||
            !rec_reverse(x,s,y,stage,has_diabatic,rgas,cp,p0,dt,g,bad)) {
            ad_refuse(failure, i, 8, bad); continue;
        }
#define REC_ACCUM(dst, value, slot) do { \
            const float v = (float)((double)(dst)[i] + (value)); \
            if (!isfinite(v)) ad_refuse(failure, i, 8, slot); else (dst)[i] = v; \
        } while (0)
        REC_ACCUM(b_rho_p_saved, g.x.rps, REC_ACC_RPS);
        REC_ACCUM(b_rtheta_p_saved, g.x.rtps, REC_ACC_RTPS);
        REC_ACCUM(b_rho_pp, g.x.rpp, REC_ACC_RPP);
        REC_ACCUM(b_rtheta_pp, g.x.rtpp, REC_ACC_RTPP);
        REC_ACCUM(b_diabatic_theta, g.x.diab, REC_ACC_DIAB);
        REC_ACCUM(b_rho_base, g.x.rb, REC_ACC_RB);
        REC_ACCUM(b_rtheta_base, g.x.rtb, REC_ACC_RTB);
        REC_ACCUM(b_exner_base, g.x.eb, REC_ACC_EB);
        REC_ACCUM(b_zz, g.x.z, REC_ACC_Z);
        REC_ACCUM(b_exner_saved, g.x.es, REC_ACC_ES);
        REC_ACCUM(b_pressure_p_saved, g.x.ps, REC_ACC_PS);
#undef REC_ACCUM
    }
    // Blocks own scalar reductions in a fixed tree, with grid-stride
    // assignment also handling captures launched with fewer than four blocks.
    for (int which=blockIdx.x; which<4; which+=gridDim.x) {
        __shared__ double sum[1024];
        const int t=threadIdx.x;
        double total=0.0;
        for (int i=t; i<n; i+=blockDim.x) {
            const RecInput x=REC_INPUT(i);
            const RecSeeds y={b_rho[i],b_rtheta[i],b_rho_p[i],b_rtheta_p[i],b_theta[i],b_exner[i],b_pressure_p[i]};
            RecState s; RecBars g; int bad = 0;
            if (rec_state(x,stage,native_gnu_powf,has_diabatic,rgas,cp,p0,dt,exner[i],s,bad) &&
                rec_reverse(x,s,y,stage,has_diabatic,rgas,cp,p0,dt,g,bad))
                total += which==0 ? g.rgas : which==1 ? g.cp : which==2 ? g.p0 : g.dt;
        }
        sum[t]=total;
        __syncthreads();
        for (int stride=1; stride<blockDim.x; stride*=2) {
            if (t%(2*stride)==0 && t+stride<blockDim.x) sum[t]+=sum[t+stride];
            __syncthreads();
        }
        if (t==0) {
            float *dest = which==0 ? b_rgas : which==1 ? b_cp : which==2 ? b_p0 : b_dt;
            const float value = (float)((double)*dest + sum[0]);
            if (!isfinite(sum[0]) || !isfinite(value)) ad_refuse(failure, n+which, 8, REC_SCALAR_RGAS+which);
            else *dest = value;
        }
        __syncthreads();
    }
    if (ad_last_block(completed))
        for (int i=threadIdx.x; i<n; i+=blockDim.x) {
            b_rho[i]=0.0f; b_rtheta[i]=0.0f; b_rho_p[i]=0.0f;
            b_rtheta_p[i]=0.0f; b_theta[i]=0.0f; b_exner[i]=0.0f;
            b_pressure_p[i]=0.0f;
        }
}
#undef REC_FORWARD_ARGS
#undef REC_INPUT
