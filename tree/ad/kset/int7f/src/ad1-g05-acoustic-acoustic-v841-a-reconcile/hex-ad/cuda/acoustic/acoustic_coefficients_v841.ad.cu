// SPDX-License-Identifier: Apache-2.0
// Hand transcription of coefficient construction and its factorization;
// reverse replays each operation and unwinds the executed recurrence.
#include "acoustic_coefficients_tape.cuh"
#define CF_C(k,c,n) ((k)*(n)+(c))
struct CoeffFields {
    AcousticField zz,cqw,exner,theta,rho,rbase,ebase,rp,q,rdzw,fzm,fzp,rdzu,etp,ewp;
    AcousticField wr,wz,tz,wt,rz,a,b,c,alpha,gamma;
};
template<class O>
__device__ void coefficients(O &o,int nlev,int ncells,int cell,float dts,float gravity,float rgas,float cp,
                              typename O::S *ddts,typename O::S *dg,typename O::S *drg,typename O::S *dcp,CoeffFields f,typename O::V *work) {
    typedef typename O::V V;
    int stride=nlev+1;
    V *wr=work,*wz=wr+stride,*tz=wz+stride,*wt=tz+stride,*a=wt+stride,*b=a+stride,*c=b+stride,*alpha=c+stride,*gamma=alpha+stride;
    V one=o.constant(1.0f),half=o.constant(0.5f),zero=o.constant(0.0f);
    V dt=o.scalar(dts,ddts,1),g=o.scalar(gravity,dg,2),rg=o.scalar(rgas,drg,3),cpv=o.scalar(cp,dcp,4);
    V rcv=o.div(rg,o.sub(cpv,rg)),csq=o.mul(cpv,rcv),dts2=o.mul(dt,dt);
    for(int k=0;k<nlev;++k) {
        int i=CF_C(k,cell,ncells);
        wr[k]=zero;wz[k]=zero;tz[k]=zero;
        V value=o.mul(half,rcv);
        value=o.mul(value,o.input(f.zz,i));value=o.mul(value,g);value=o.mul(value,o.input(f.rho,i));
        value=o.div(value,o.add(one,o.input(f.q,i)));value=o.mul(value,o.input(f.exner,i));
        value=o.div(value,o.mul(o.add(o.input(f.rbase,i),o.input(f.rp,i)),o.input(f.ebase,i)));
        wt[k]=value;a[k]=zero;b[k]=zero;c[k]=zero;alpha[k]=zero;gamma[k]=zero;
    }
    tz[nlev]=zero;if(nlev>0)b[0]=one;
    for(int k=1;k<nlev;++k) {
        int i=CF_C(k,cell,ncells),im=CF_C(k-1,cell,ncells);
        V iz=o.add(o.mul(o.input(f.fzm,k),o.input(f.zz,i)),o.mul(o.input(f.fzp,k),o.input(f.zz,im)));
        wr[k]=o.mul(o.mul(half,g),iz);
        V z=o.mul(csq,iz);z=o.mul(z,o.input(f.rdzu,k));z=o.mul(z,o.input(f.cqw,i));
        z=o.mul(z,o.add(o.mul(o.input(f.fzm,k),o.input(f.exner,i)),o.mul(o.input(f.fzp,k),o.input(f.exner,im))));
        wz[k]=z;tz[k]=o.add(o.mul(o.input(f.fzm,k),o.input(f.theta,i)),o.mul(o.input(f.fzp,k),o.input(f.theta,im)));
    }
    for(int k=1;k<nlev;++k) {
        int i=CF_C(k,cell,ncells),im=CF_C(k-1,cell,ncells);
        V av=o.sub(zero,o.mul(o.mul(o.mul(wz[k],tz[k-1]),o.input(f.rdzw,k-1)),o.input(f.zz,im)));
        av=o.add(av,o.mul(wr[k],o.input(f.rz,k-1)));
        av=o.sub(av,o.mul(o.mul(wt[k-1],tz[k-1]),o.input(f.rdzw,k-1)));
        av=o.mul(av,o.input(f.etp,k-1));a[k]=o.mul(av,o.input(f.ewp,k-1));
        V um=o.mul(o.mul(o.input(f.etp,k),o.input(f.rdzw,k)),o.input(f.zz,i));
        V lm=o.mul(o.mul(o.input(f.etp,k-1),o.input(f.rdzw,k-1)),o.input(f.zz,im));
        V bv=o.mul(o.mul(wz[k],tz[k]),o.add(um,lm));
        V uh=o.mul(o.mul(o.input(f.etp,k),wt[k]),o.input(f.rdzw,k));
        V lh=o.mul(o.mul(o.input(f.etp,k-1),wt[k-1]),o.input(f.rdzw,k-1));
        bv=o.sub(bv,o.mul(tz[k],o.sub(uh,lh)));
        V ur=o.mul(o.input(f.etp,k),o.input(f.rz,k));
        V lr=o.mul(o.input(f.etp,k-1),o.input(f.rz,k-1));
        bv=o.add(bv,o.mul(wr[k],o.sub(ur,lr)));b[k]=o.mul(bv,o.input(f.ewp,k));
        V cv=o.sub(zero,o.mul(o.mul(o.mul(wz[k],tz[k+1]),o.input(f.rdzw,k)),o.input(f.zz,i)));
        cv=o.sub(cv,o.mul(wr[k],o.input(f.rz,k)));
        cv=o.add(cv,o.mul(o.mul(wt[k],tz[k+1]),o.input(f.rdzw,k)));
        cv=o.mul(cv,o.input(f.etp,k));c[k]=o.mul(cv,o.input(f.ewp,k+1));
    }
    if(nlev>0)c[nlev-1]=zero;
    for(int k=1;k<nlev;++k) {
        V den=o.add(one,o.mul(dts2,o.sub(b[k],o.mul(a[k],gamma[k-1]))));
        bool finite=isfinite(den.v);
        bool invalid=!(finite&&den.v!=0.0f);
        o.record(4,invalid);o.record(19,finite);o.record(20,invalid);
        if(invalid) acoustic_fail(o.failure,0,k,7);
        alpha[k]=o.div(one,den);gamma[k]=o.mul(o.mul(dts2,c[k]),alpha[k]);
    }
    for(int k=0;k<nlev;++k) {
        int i=CF_C(k,cell,ncells);
        o.output(f.wr,i,wr[k]);o.output(f.wz,i,wz[k]);o.output(f.tz,i,tz[k]);o.output(f.wt,i,wt[k]);
        o.output(f.a,i,a[k]);o.output(f.b,i,b[k]);o.output(f.c,i,c[k]);o.output(f.alpha,i,alpha[k]);o.output(f.gamma,i,gamma[k]);
    }
    o.output(f.tz,CF_C(nlev,cell,ncells),tz[nlev]);o.finish();
}
#define CF_PRIMAL_ARGS \
    const int nlev,const int ncells,const float dts,const float gravity,const float rgas,const float cp, \
    const float *zz,const float *cqw,const float *exner,const float *theta,const float *rho_base, \
    const float *rtheta_base,const float *exner_base,const float *rtheta_p,const float *qtot, \
    const float *rdzw,const float *fzm,const float *fzp,const float *rdzu,const float *etp,const float *ewp, \
    float *cofwr,float *cofwz,float *coftz,float *cofwt,const float *cofrz, \
    float *a_tri,float *b_tri,float *c_tri,float *alpha_tri,float *gamma_tri,int *singular
#define CF_SHADOW_ARGS \
    AcousticShadow *d_zz,AcousticShadow *d_cqw,AcousticShadow *d_exner,AcousticShadow *d_theta,AcousticShadow *d_rho,AcousticShadow *d_rbase, \
    AcousticShadow *d_ebase,AcousticShadow *d_rp,AcousticShadow *d_q,AcousticShadow *d_rdzw,AcousticShadow *d_fzm,AcousticShadow *d_fzp,AcousticShadow *d_rdzu, \
    AcousticShadow *d_etp,AcousticShadow *d_ewp,AcousticShadow *d_wr,AcousticShadow *d_wz,AcousticShadow *d_tz,AcousticShadow *d_wt,AcousticShadow *d_rz, \
    AcousticShadow *d_a,AcousticShadow *d_b,AcousticShadow *d_c,AcousticShadow *d_alpha,AcousticShadow *d_gamma
#define CF_FIELDS CoeffFields f={ \
    {zz,d_zz,5},{cqw,d_cqw,6},{exner,d_exner,7},{theta,d_theta,8},{rho_base,d_rho,9}, \
    {rtheta_base,d_rbase,10},{exner_base,d_ebase,11},{rtheta_p,d_rp,12},{qtot,d_q,13}, \
    {rdzw,d_rdzw,14},{fzm,d_fzm,15},{fzp,d_fzp,16},{rdzu,d_rdzu,17},{etp,d_etp,18},{ewp,d_ewp,19}, \
    {cofwr,d_wr,20},{cofwz,d_wz,21},{coftz,d_tz,22},{cofwt,d_wt,23},{cofrz,d_rz,24}, \
    {a_tri,d_a,25},{b_tri,d_b,26},{c_tri,d_c,27},{alpha_tri,d_alpha,28},{gamma_tri,d_gamma,29} }
extern "C" __global__ void acoustic_coefficients_v841__tl(CF_PRIMAL_ARGS,
    const AcousticShadow *d_dts,const AcousticShadow *d_g,const AcousticShadow *d_rg,const AcousticShadow *d_cp,CF_SHADOW_ARGS,
    AcousticDual *work,AcousticFailure *failures,unsigned *branch_record) {
    (void)singular;int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=ncells)return;
    if(nlev<0){acoustic_fail(failures,0,0,5);return;}
    CF_FIELDS;unsigned branch_local[42]={0};unsigned *branch_counter=branch_record?branch_local:nullptr;
    AcousticTangent o{failures,branch_counter,5};AcousticShadow ddts=*d_dts,dg=*d_g,drg=*d_rg,dcp=*d_cp;
    coefficients(o,nlev,ncells,cell,dts,gravity,rgas,cp,&ddts,&dg,&drg,&dcp,f,work+(unsigned long long)cell*9*(nlev+1));
    if(branch_record)for(int i=0;i<42;++i)if(branch_local[i]) atomicAdd(branch_record+i,branch_local[i]);
}
// One block owns the complete reverse launch. Each thread replays distinct
// columns into private tape and private shared-field contributions. After the
// barrier, each vertical element is reduced in fixed ascending column order.
extern "C" __global__ void acoustic_coefficients_v841__ad(CF_PRIMAL_ARGS,
    AcousticShadow *b_dts,AcousticShadow *b_g,AcousticShadow *b_rg,AcousticShadow *b_cp,CF_SHADOW_ARGS,
    AcousticValue *work,AcousticNode *scratch,double *contrib,AcousticFailure *failures) {
    (void)singular;
    if(gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||nlev<0){
        if(!blockIdx.x&&!blockIdx.y&&!blockIdx.z&&!threadIdx.x&&!threadIdx.y&&!threadIdx.z)
            acoustic_fail(failures,0,0,6);
        return;
    }
    const int tid=threadIdx.x, s=nlev+1, stride=7*s+4, capacity=512*s+256;
    for(int cell=tid;cell<ncells;cell+=128) {
        double *partial=contrib+(unsigned long long)cell*stride;
        for(int q=0;q<stride;++q)partial[q]=0.0f;
        CF_FIELDS;
        f.rdzw.accum=partial;f.fzm.accum=partial+s;f.fzp.accum=partial+2*s;
        f.rdzu.accum=partial+3*s;f.etp.accum=partial+4*s;f.ewp.accum=partial+5*s;f.rz.accum=partial+6*s;
        AcousticReverse o{scratch+(unsigned long long)tid*capacity,0,capacity,failures};o.reset();
        coefficients(o,nlev,ncells,cell,dts,gravity,rgas,cp,
            partial+7*s,partial+7*s+1,partial+7*s+2,partial+7*s+3,
            f,work+(unsigned long long)tid*9*s);
    }
    __syncthreads();
    AcousticShadow *dst[7]={d_rdzw,d_fzm,d_fzp,d_rdzu,d_etp,d_ewp,d_rz};
    for(int q=tid;q<stride;q+=128) {
        AcousticShadow *target;int plane=q/s,k=q%s;
        if(q<7*s) { if(k==nlev && plane!=5)continue;target=dst[plane]+k; }
        else {AcousticShadow *scalars[4]={b_dts,b_g,b_rg,b_cp};target=scalars[q-7*s];}
        double sum=acoustic_shadow_decode(*target);
        for(int cell=0;cell<ncells;++cell)sum+=contrib[(unsigned long long)cell*stride+q];
        if(!isfinite(sum))acoustic_fail(failures,0,q,3);
        *target=acoustic_shadow_encode(sum);
    }
}
