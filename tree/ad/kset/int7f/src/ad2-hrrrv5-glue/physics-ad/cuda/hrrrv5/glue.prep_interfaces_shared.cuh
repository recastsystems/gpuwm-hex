// SPDX-License-Identifier: Apache-2.0
// A lane owns all levels of one column. Forward outputs named in the spec
// remain available until reverse; no augmented forward is needed.
#include "glue.member_g26.cuh"
#include "glue.gfk_log_pinned.cuh"

#ifdef AD_INTERFACE_CAPTURE_WEIGHTS
#define INTERFACE_TL prep_interfaces_weights_v841_f32__tl
#define INTERFACE_AD prep_interfaces_weights_v841_f32__ad
#define INTERFACE_EXTRA_ARGS , const float *pressure_perturbation, const float *pressure_base, float *weights_minus, float *weights_plus
#define INTERFACE_EXTRA_TL_ARGS , const float *d_pressure_perturbation, const float *d_pressure_base, float *d_weights_minus, float *d_weights_plus
#define INTERFACE_EXTRA_AD_ARGS , float *b_pressure_perturbation, float *b_pressure_base, float *b_weights_minus, float *b_weights_plus
#else
#define INTERFACE_TL prep_interfaces_v841_f32__tl
#define INTERFACE_AD prep_interfaces_v841_f32__ad
#define INTERFACE_EXTRA_ARGS
#define INTERFACE_EXTRA_TL_ARGS
#define INTERFACE_EXTRA_AD_ARGS
#endif

#define INTERFACE_ARGS \
    int nlev, int ncells, float gravity, const float *rho_zz, \
    const float *zz, const float *qv_p, const float *pres_p, \
    const float *t_p, const float *rho_p, const float *dz_p, \
    const float *zgrid, const float *w, float *w_p, float *z_p, \
    float *t2_p, float *pres2_p, float *pres_hyd_p, float *pres2_hyd_p, \
    float *pres_hydd_p, float *pres2_hydd_p, float *psfc_p, \
    float *psfc_hyd_p, float *psfc_hydd_p, float *znu_p, \
    float *znu_hyd_p, float *plrad, int *invalid INTERFACE_EXTRA_ARGS

__device__ __forceinline__ void interface_boundary_tl(
    int j0, int j1, int j2, int a, int b, bool logarithmic,
    const float *z, const float *dz, const float *t, const float *dt,
    const float *p, const float *dp, float primal_pressure,
    float *d_temperature, float *d_pressure)
{
    float z0=z[j0], z1=0.5f*(z[j0]+z[j1]), z2=0.5f*(z[j1]+z[j2]);
    float dz0=dz[j0], dz1=0.5f*(dz[j0]+dz[j1]), dz2=0.5f*(dz[j1]+dz[j2]);
    float den=z1-z2, w1=(z0-z2)/den, w2=1.0f-w1;
    float dw1=((dz0-dz2)-w1*(dz1-dz2))/den;
    *d_temperature = dw1*(t[a]-t[b]) + w1*dt[a] + w2*dt[b];
    if (logarithmic)
        *d_pressure = primal_pressure*(dw1*(gfk_log(p[a])-gfk_log(p[b]))
            + w1*dp[a]/p[a] + w2*dp[b]/p[b]);
    else
        *d_pressure = dw1*(p[a]-p[b]) + w1*dp[a] + w2*dp[b];
}

__device__ __forceinline__ void interface_boundary_ad(
    int j0, int j1, int j2, int a, int b, bool logarithmic,
    const float *z, double *bz, const float *t, double *bt,
    const float *p, double *bp, float primal_pressure,
    double b_temperature, double b_pressure)
{
    float z0=z[j0], z1=0.5f*(z[j0]+z[j1]), z2=0.5f*(z[j1]+z[j2]);
    float den=z1-z2, w1=(z0-z2)/den, w2=1.0f-w1;
    bt[a] += b_temperature*w1; bt[b] += b_temperature*w2;
    double bw = b_temperature*(t[a]-t[b]);
    if (logarithmic) {
        double q=b_pressure*primal_pressure;
        bp[a] += q*w1/p[a]; bp[b] += q*w2/p[b];
        bw += q*(gfk_log(p[a])-gfk_log(p[b]));
    } else {
        bp[a] += b_pressure*w1; bp[b] += b_pressure*w2;
        bw += b_pressure*(p[a]-p[b]);
    }
    double bn=bw/den, bd=-bw*w1/den;
    bz[j0] += bn+0.5*bd;
    bz[j1] += 0.5*bd-0.5*(bn+bd);
    bz[j2] -= 0.5*(bn+bd);
}

#if !defined(AD_AD_ONLY)
extern "C" __global__ void INTERFACE_TL(
    INTERFACE_ARGS, const float *d_gravity, const float *d_rho_zz,
    const float *d_zz, const float *d_qv_p, const float *d_pres_p,
    const float *d_t_p, const float *d_rho_p, const float *d_dz_p,
    const float *d_zgrid, const float *d_w, float *d_w_p, float *d_z_p,
    float *d_t2_p, float *d_pres2_p, float *d_pres_hyd_p,
    float *d_pres2_hyd_p, float *d_pres_hydd_p, float *d_pres2_hydd_p,
    float *d_psfc_p, float *d_psfc_hyd_p, float *d_psfc_hydd_p,
    float *d_znu_p, float *d_znu_hyd_p, float *d_plrad INTERFACE_EXTRA_TL_ARGS,
    int *failure, unsigned int *branch_record)
{
    (void)w; (void)w_p; (void)z_p; (void)t2_p; (void)pres2_hyd_p;
    (void)pres_hydd_p; (void)pres2_hydd_p; (void)psfc_hydd_p;
    (void)znu_p; (void)znu_hyd_p; (void)plrad; (void)invalid;
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    PrepBranchRecorder br{branch_record,11}; br.count(0,c>=ncells);
    if (c>=ncells) { br.flush(26); return; }
    br.count_n(1,true,(unsigned)(nlev+1));br.count(1,false);
    br.count_n(3,true,(unsigned)nlev);br.count(3,false);
    br.count_n(4,true,(unsigned)(nlev-1));br.count(4,false);
    br.count_n(8,true,(unsigned)nlev);br.count(8,false);
    br.count_n(10,true,(unsigned)nlev);br.count(10,false);
    br.count_n(25,true,(unsigned)nlev);br.count(25,false);
    // Recompute pinned helper calls solely to count their executed arms.
    int j=c+ncells;
    float tem1r=br.sub(zgrid[j],zgrid[c]);
    float tem2r=br.sub(zgrid[j+ncells],zgrid[j]);
    float rho1r=br.mul(br.mul(rho_zz[c],zz[c]),br.add(1.0f,qv_p[c]));
    float rho2r=br.mul(br.mul(rho_zz[j],zz[j]),br.add(1.0f,qv_p[j]));
    float denom=br.add(tem1r,tem2r);br.count(2,denom==0.0f);
    float rcorr=br.div(br.mul(br.mul(0.5f,br.sub(rho2r,rho1r)),tem1r),denom);
    float hydro=br.mul(br.mul(br.mul(0.5f,gravity),tem1r),br.sub(rho1r,rcorr));
#ifdef AD_INTERFACE_CAPTURE_WEIGHTS
    float surf=br.add(br.add(hydro,pressure_perturbation[c]),pressure_base[c]);
#else
    float surf=br.add(hydro,pres_p[c]);
#endif
    for(int k=0;k<nlev;++k) (void)br.div(pres_p[k*ncells+c],surf);
    for(int k=1;k<nlev;++k) {
        int i=k*ncells+c;
        float inv=br.div(1.0f,br.sub(zgrid[i+ncells],zgrid[i-ncells]));
        float fzm=br.mul(br.sub(zgrid[i],zgrid[i-ncells]),inv);
        float fzp=br.mul(br.sub(zgrid[i+ncells],zgrid[i]),inv);
        (void)br.add(br.mul(fzm,t_p[i]),br.mul(fzp,t_p[i-ncells]));
        (void)br.add(br.mul(fzm,pres_p[i]),br.mul(fzp,pres_p[i-ncells]));
    }
    int top=nlev*ncells+c;
    float rz1=br.mul(0.5f,br.add(zgrid[top],zgrid[top-ncells]));
    float rz2=br.mul(0.5f,br.add(zgrid[top-ncells],zgrid[top-2*ncells]));
    float rn=br.sub(zgrid[top],rz2),rd=br.sub(rz1,rz2);br.count(5,rd==0.0f);
    float rw1=br.div(rn,rd),rw2=br.sub(1.0f,rw1);
    (void)br.add(br.mul(rw1,t_p[top-ncells]),br.mul(rw2,t_p[top-2*ncells]));
    float pa=pres_p[top-ncells],pb=pres_p[top-2*ncells];
    bool log_domain_failure=!isfinite(pa)||pa<=0.0f||!isfinite(pb)||pb<=0.0f;
    br.count(6,log_domain_failure);
    if (log_domain_failure) {
        if (!isfinite(pa) || pa<=0.0f) atomicExch(failure+0,1); // :2012 gfk_log
        if (!isfinite(pb) || pb<=0.0f) atomicExch(failure+1,1); // :2013 gfk_log
        br.flush(26);
        return;
    }
    (void)br.add(br.mul(rw1,gfk_log(pa)),br.mul(rw2,gfk_log(pb)));
    rz1=br.mul(0.5f,br.add(zgrid[c],zgrid[ncells+c]));
    rz2=br.mul(0.5f,br.add(zgrid[ncells+c],zgrid[2*ncells+c]));
    rn=br.sub(zgrid[c],rz2);rd=br.sub(rz1,rz2);br.count(7,rd==0.0f);
    rw1=br.div(rn,rd);rw2=br.sub(1.0f,rw1);
    (void)br.add(br.mul(rw1,t_p[c]),br.mul(rw2,t_p[ncells+c]));
    (void)br.add(br.mul(rw1,pres_p[c]),br.mul(rw2,pres_p[ncells+c]));
    for(int k=nlev-1;k>=0;--k) {
        int i=k*ncells+c;float qden=br.add(1.0f,qv_p[i]);br.count(9,qden==0.0f);
        float rhoa=br.div(rho_p[i],qden);
        (void)br.add(pres2_hyd_p[i+ncells],br.mul(br.mul(gravity,rho_p[i]),dz_p[i]));
        (void)br.add(pres2_hydd_p[i+ncells],br.mul(br.mul(gravity,rhoa),dz_p[i]));
    }
    for(int k=nlev-1;k>=0;--k) {
        int i=k*ncells+c;
        (void)br.mul(0.5f,br.add(pres2_hyd_p[i+ncells],pres2_hyd_p[i]));
        (void)br.mul(0.5f,br.add(pres2_hydd_p[i+ncells],pres2_hydd_p[i]));
    }
    for(int k=nlev-1;k>=0;--k) (void)br.div(pres_hyd_p[k*ncells+c],psfc_hyd_p[c]);
    for (int k=0;k<=nlev;++k) {
        int i=k*ncells+c; d_w_p[i]=d_w[i]; d_z_p[i]=d_zgrid[i];
    }
    j=c+ncells;
    float tem1=zgrid[j]-zgrid[c], tem2=zgrid[j+ncells]-zgrid[j];
    float dt1=d_zgrid[j]-d_zgrid[c], dt2=d_zgrid[j+ncells]-d_zgrid[j];
    float r1=(rho_zz[c]*zz[c])*(1.0f+qv_p[c]);
    float r2=(rho_zz[j]*zz[j])*(1.0f+qv_p[j]);
    float dr1=(d_rho_zz[c]*zz[c]+rho_zz[c]*d_zz[c])*(1.0f+qv_p[c])
               +(rho_zz[c]*zz[c])*d_qv_p[c];
    float dr2=(d_rho_zz[j]*zz[j]+rho_zz[j]*d_zz[j])*(1.0f+qv_p[j])
               +(rho_zz[j]*zz[j])*d_qv_p[j];
    float den=tem1+tem2;
    float corr=(0.5f*(r2-r1)*tem1)/den;
    float dcorr=(0.5f*((dr2-dr1)*tem1+(r2-r1)*dt1)-corr*(dt1+dt2))/den;
    float dh=0.5f*((*d_gravity)*tem1*(r1-corr)
        +gravity*dt1*(r1-corr)+gravity*tem1*(dr1-dcorr));
#ifdef AD_INTERFACE_CAPTURE_WEIGHTS
    d_psfc_p[c]=dh+d_pressure_perturbation[c]+d_pressure_base[c];
#else
    d_psfc_p[c]=dh+d_pres_p[c];
#endif
    for (int k=0;k<nlev;++k) {
        int i=k*ncells+c;
        d_znu_p[i]=(d_pres_p[i]-(pres_p[i]/psfc_p[c])*d_psfc_p[c])/psfc_p[c];
    }
    for (int k=1;k<nlev;++k) {
        int i=k*ncells+c;
        float a=zgrid[i]-zgrid[i-ncells], b=zgrid[i+ncells]-zgrid[i];
        float inv=1.0f/(zgrid[i+ncells]-zgrid[i-ncells]);
        float fm=a*inv, fp=b*inv;
        // Use wider intermediates for the two nearly cancelling height terms.
        // The primal coefficients remain those of the original f32 kernel.
        double da=(double)d_zgrid[i]-d_zgrid[i-ncells];
        double db=(double)d_zgrid[i+ncells]-d_zgrid[i];
        double dinv=-(double)inv*inv*((double)d_zgrid[i+ncells]-d_zgrid[i-ncells]);
        double dfm=da*inv+(double)a*dinv, dfp=db*inv+(double)b*dinv;
#ifdef AD_INTERFACE_CAPTURE_WEIGHTS
        d_weights_minus[i]=(float)dfm; d_weights_plus[i]=(float)dfp;
#endif
        d_t2_p[i]=(float)(dfm*t_p[i]+(double)fm*d_t_p[i]+dfp*t_p[i-ncells]+(double)fp*d_t_p[i-ncells]);
        d_pres2_p[i]=(float)(dfm*pres_p[i]+(double)fm*d_pres_p[i]+dfp*pres_p[i-ncells]+(double)fp*d_pres_p[i-ncells]);
    }
    top=nlev*ncells+c;
    interface_boundary_tl(top,top-ncells,top-2*ncells,top-ncells,top-2*ncells,true,
        zgrid,d_zgrid,t_p,d_t_p,pres_p,d_pres_p,pres2_p[top],&d_t2_p[top],&d_pres2_p[top]);
    interface_boundary_tl(c,c+ncells,c+2*ncells,c,c+ncells,false,
        zgrid,d_zgrid,t_p,d_t_p,pres_p,d_pres_p,pres2_p[c],&d_t2_p[c],&d_pres2_p[c]);
    d_pres2_hyd_p[top]=d_pres2_p[top]; d_pres2_hydd_p[top]=d_pres2_p[top];
    for (int k=nlev-1;k>=0;--k) {
        int i=k*ncells+c;
        float qden=1.0f+qv_p[i], ra=rho_p[i]/qden;
        float dra=(d_rho_p[i]-ra*d_qv_p[i])/qden;
        d_pres2_hyd_p[i]=d_pres2_hyd_p[i+ncells]
            +((*d_gravity)*rho_p[i]+gravity*d_rho_p[i])*dz_p[i]+gravity*rho_p[i]*d_dz_p[i];
        d_pres2_hydd_p[i]=d_pres2_hydd_p[i+ncells]
            +((*d_gravity)*ra+gravity*dra)*dz_p[i]+gravity*ra*d_dz_p[i];
    }
    for (int k=nlev-1;k>=0;--k) {
        int i=k*ncells+c;
        d_pres_hyd_p[i]=0.5f*(d_pres2_hyd_p[i+ncells]+d_pres2_hyd_p[i]);
        d_pres_hydd_p[i]=0.5f*(d_pres2_hydd_p[i+ncells]+d_pres2_hydd_p[i]);
    }
    d_psfc_hyd_p[c]=d_pres2_hyd_p[c]; d_psfc_hydd_p[c]=d_pres2_hydd_p[c];
    for (int k=0;k<nlev;++k) {
        int i=k*ncells+c;
        d_znu_hyd_p[i]=(d_pres_hyd_p[i]-(pres_hyd_p[i]/psfc_hyd_p[c])*d_psfc_hyd_p[c])/psfc_hyd_p[c];
    }
    d_plrad[c]=d_pres2_p[top];
    br.flush(26);
}
#endif

#if !defined(AD_TL_ONLY)
extern "C" __global__ void INTERFACE_AD(
    INTERFACE_ARGS, float *b_gravity, float *b_rho_zz, float *b_zz,
    float *b_qv_p, float *b_pres_p, float *b_t_p, float *b_rho_p,
    float *b_dz_p, float *b_zgrid, float *b_w,
    float *b_w_p, float *b_z_p, float *b_t2_p,
    float *b_pres2_p, float *b_pres_hyd_p,
    float *b_pres2_hyd_p, float *b_pres_hydd_p,
    float *b_pres2_hydd_p, float *b_psfc_p,
    float *b_psfc_hyd_p, float *b_psfc_hydd_p,
    float *b_znu_p, float *b_znu_hyd_p, float *b_plrad INTERFACE_EXTRA_AD_ARGS,
    int *failure, double *scalar_workspace, int phase)
{
    (void)w; (void)w_p; (void)z_p; (void)t2_p; (void)pres2_hyd_p;
    (void)pres_hydd_p; (void)pres2_hydd_p; (void)psfc_hydd_p;
    (void)znu_p; (void)znu_hyd_p; (void)plrad; (void)invalid;
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if (phase==1) {
        if (c==0) {
            double sum=0.0;
            for(int j=0;j<ncells;++j) sum+=scalar_workspace[j];
            *b_gravity=(float)((double)*b_gravity+sum);
        }
        return;
    }
    if(c>=ncells) return;
    int top=nlev*ncells+c;
    float pa=pres_p[top-ncells], pb=pres_p[top-2*ncells];
    if (!isfinite(pa) || pa<=0.0f || !isfinite(pb) || pb<=0.0f) {
        if (!isfinite(pa) || pa<=0.0f) atomicExch(failure+0,1); // :2012 gfk_log
        if (!isfinite(pb) || pb<=0.0f) atomicExch(failure+1,1); // :2013 gfk_log
        scalar_workspace[c]=0.0;
        return;
    }
    // A column owns every element it touches. Preserve existing cotangents in
    // workspace and round each scattered stencil contribution only once.
    double *bz=scalar_workspace+ncells;
    double *bt=bz+(nlev+1)*ncells;
    double *bp=bt+nlev*ncells;
    double *bq=bp+nlev*ncells;
    double bg=0.0;
    for(int k=0;k<=nlev;++k) {
        int i=k*ncells+c;
        b_w[i]+=b_w_p[i]; bz[i]=(double)b_zgrid[i]+b_z_p[i];
    }
    for(int k=0;k<nlev;++k) {
        int i=k*ncells+c; bt[i]=b_t_p[i]; bp[i]=b_pres_p[i]; bq[i]=b_qv_p[i];
    }
    double bsurf=b_psfc_p[c], bsurfhyd=b_psfc_hyd_p[c];
    for(int k=0;k<nlev;++k) {
        int i=k*ncells+c;
        bp[i]+=(double)b_znu_p[i]/psfc_p[c];
        bsurf-=(double)b_znu_p[i]*(pres_p[i]/psfc_p[c])/psfc_p[c];
        bsurfhyd-=(double)b_znu_hyd_p[i]*(pres_hyd_p[i]/psfc_hyd_p[c])/psfc_hyd_p[c];
    }
    // Reverse the two top-down pressure recurrences in bottom-up order.
    double carry=bsurfhyd, carrydry=b_psfc_hydd_p[c];
    double previous=0.0, previousdry=0.0;
    for(int k=0;k<nlev;++k) {
        int i=k*ncells+c;
        double mass=(double)b_pres_hyd_p[i]+(double)b_znu_hyd_p[i]/psfc_hyd_p[c];
        double massdry=b_pres_hydd_p[i];
        double g=carry+b_pres2_hyd_p[i]+0.5*(mass+previous);
        double gd=carrydry+b_pres2_hydd_p[i]+0.5*(massdry+previousdry);
        float qden=1.0f+qv_p[i], ra=rho_p[i]/qden;
        b_rho_p[i]=(float)((double)b_rho_p[i]+g*gravity*dz_p[i]+gd*gravity*dz_p[i]/qden);
        bq[i]-=gd*gravity*dz_p[i]*ra/qden;
        b_dz_p[i]=(float)((double)b_dz_p[i]+g*gravity*rho_p[i]+gd*gravity*ra);
        bg+=g*rho_p[i]*dz_p[i]+gd*ra*dz_p[i];
        carry=g; carrydry=gd; previous=mass; previousdry=massdry;
    }
    double btop=(double)b_pres2_p[top]+b_plrad[c]+carry+b_pres2_hyd_p[top]+0.5*previous
              +carrydry+b_pres2_hydd_p[top]+0.5*previousdry;
    interface_boundary_ad(top,top-ncells,top-2*ncells,top-ncells,top-2*ncells,true,
        zgrid,bz,t_p,bt,pres_p,bp,pres2_p[top],b_t2_p[top],btop);
    interface_boundary_ad(c,c+ncells,c+2*ncells,c,c+ncells,false,
        zgrid,bz,t_p,bt,pres_p,bp,pres2_p[c],b_t2_p[c],b_pres2_p[c]);
    for(int k=1;k<nlev;++k) {
        int i=k*ncells+c;
        float a=zgrid[i]-zgrid[i-ncells], b=zgrid[i+ncells]-zgrid[i];
        float inv=1.0f/(zgrid[i+ncells]-zgrid[i-ncells]);
        float fm=a*inv, fp=b*inv;
        double gt=b_t2_p[i], gp=b_pres2_p[i];
        bt[i]+=gt*fm; bt[i-ncells]+=gt*fp;
        bp[i]+=gp*fm; bp[i-ncells]+=gp*fp;
        double bm=gt*t_p[i]+gp*pres_p[i];
        double bpw=gt*t_p[i-ncells]+gp*pres_p[i-ncells];
#ifdef AD_INTERFACE_CAPTURE_WEIGHTS
        bm+=(double)b_weights_minus[i]; bpw+=(double)b_weights_plus[i];
        b_weights_minus[i]=0.0f; b_weights_plus[i]=0.0f;
#endif
        double bd=-(bm*a+bpw*b)*(double)inv*inv;
        bz[i-ncells]+=-bm*inv-bd;
        bz[i]+=(bm-bpw)*inv;
        bz[i+ncells]+=bpw*inv+bd;
    }
    int j=c+ncells;
    float tem1=zgrid[j]-zgrid[c], tem2=zgrid[j+ncells]-zgrid[j];
    float r1=(rho_zz[c]*zz[c])*(1.0f+qv_p[c]);
    float r2=(rho_zz[j]*zz[j])*(1.0f+qv_p[j]);
    float den=tem1+tem2, corr=(0.5f*(r2-r1)*tem1)/den;
#ifdef AD_INTERFACE_CAPTURE_WEIGHTS
    b_pressure_perturbation[c]=(float)((double)b_pressure_perturbation[c]+bsurf);
    b_pressure_base[c]=(float)((double)b_pressure_base[c]+bsurf);
#else
    bp[c]+=bsurf;
#endif
    bg+=bsurf*0.5*tem1*(r1-corr);
    double btem1=bsurf*(0.5f*gravity)*(r1-corr);
    double br1=bsurf*((0.5f*gravity)*tem1);
    double bcorr=-br1;
    double bn=bcorr/den, bd=-bcorr*corr/den;
    double br2=bn*0.5*tem1;
    br1-=br2;
    btem1+=bn*0.5*(r2-r1)+bd;
    double btem2=bd;
    bz[c]-=btem1; bz[j]+=btem1-btem2; bz[j+ncells]+=btem2;
    b_rho_zz[c]=(float)((double)b_rho_zz[c]+br1*zz[c]*(1.0f+qv_p[c]));
    b_zz[c]=(float)((double)b_zz[c]+br1*rho_zz[c]*(1.0f+qv_p[c]));
    bq[c]+=br1*(rho_zz[c]*zz[c]);
    b_rho_zz[j]=(float)((double)b_rho_zz[j]+br2*zz[j]*(1.0f+qv_p[j]));
    b_zz[j]=(float)((double)b_zz[j]+br2*rho_zz[j]*(1.0f+qv_p[j]));
    bq[j]+=br2*(rho_zz[j]*zz[j]);
    for(int k=0;k<=nlev;++k) { int i=k*ncells+c; b_zgrid[i]=(float)bz[i]; }
    for(int k=0;k<nlev;++k) {
        int i=k*ncells+c; b_t_p[i]=(float)bt[i]; b_pres_p[i]=(float)bp[i]; b_qv_p[i]=(float)bq[i];
    }
    for(int k=0;k<=nlev;++k) {
        int i=k*ncells+c;
        b_w_p[i]=0.0f; b_z_p[i]=0.0f; b_t2_p[i]=0.0f; b_pres2_p[i]=0.0f;
        b_pres2_hyd_p[i]=0.0f; b_pres2_hydd_p[i]=0.0f;
    }
    for(int k=0;k<nlev;++k) {
        int i=k*ncells+c;
        b_pres_hyd_p[i]=0.0f; b_pres_hydd_p[i]=0.0f;
        b_znu_p[i]=0.0f; b_znu_hyd_p[i]=0.0f;
    }
    b_psfc_p[c]=0.0f; b_psfc_hyd_p[c]=0.0f;
    b_psfc_hydd_p[c]=0.0f; b_plrad[c]=0.0f;
    scalar_workspace[c]=bg;
}
#endif
