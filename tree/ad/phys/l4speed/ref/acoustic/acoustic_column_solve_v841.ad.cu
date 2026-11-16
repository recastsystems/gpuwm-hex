// SPDX-License-Identifier: Apache-2.0
// Hand transcription of all four loops and the two tridiagonal sweeps.
// Reverse replays the same rounded primal operations and unwinds their tape.
#include "acoustic_coefficients_tape.cuh"
#define CS_C(k,c,n) ((k)*(n)+(c))
struct ColumnFields {
    AcousticField zz,rhoz,fzm,fzp,rdzw,dss,w,rw,rws,tend,rs,ts;
    AcousticField cofwr,cofwz,coftz,cofwt,cofrz,a,alpha,gamma,etp,etm,ewp,ewm;
    AcousticField rwp,rhop,rtp,ww;
};
template<class O>
__device__ void column_solve(O &o,int nlev,int ncells,int cell,float dts,typename O::S *ddts,
                             ColumnFields f,typename O::V *work) {
    typedef typename O::V V;
    V *u=work,*avg=work+(nlev+1);
    V dt=o.scalar(dts,ddts,1),dts2=o.mul(dt,dt);
    for(int k=0;k<=nlev;++k) {
        int i=CS_C(k,cell,ncells);u[k]=o.input(f.rwp,i);avg[k]=o.input(f.ww,i);
    }
    for(int k=1;k<nlev;++k) {
        int i=CS_C(k,cell,ncells),im=CS_C(k-1,cell,ncells);
        avg[k]=o.add(avg[k],o.mul(o.input(f.ewm,k),u[k]));
        V ti=o.sub(o.mul(o.mul(o.input(f.etp,k),o.input(f.zz,i)),o.input(f.ts,i)),
                   o.mul(o.mul(o.input(f.etp,k-1),o.input(f.zz,im)),o.input(f.ts,im)));
        V te=o.sub(o.mul(o.mul(o.input(f.etm,k),o.input(f.zz,i)),o.input(f.rtp,i)),
                   o.mul(o.mul(o.input(f.etm,k-1),o.input(f.zz,im)),o.input(f.rtp,im)));
        V di=o.add(o.mul(o.input(f.etp,k),o.input(f.rs,i)),o.mul(o.input(f.etp,k-1),o.input(f.rs,im)));
        V de=o.add(o.mul(o.input(f.etm,k),o.input(f.rhop,i)),o.mul(o.input(f.etm,k-1),o.input(f.rhop,im)));
        V hu=o.add(o.mul(o.input(f.etp,k),o.input(f.ts,i)),o.mul(o.input(f.etm,k),o.input(f.rtp,i)));
        V hl=o.add(o.mul(o.input(f.etp,k-1),o.input(f.ts,im)),o.mul(o.input(f.etm,k-1),o.input(f.rtp,im)));
        V inner=o.sub(o.input(f.tend,i),o.mul(o.input(f.cofwz,i),o.add(ti,te)));
        inner=o.sub(inner,o.mul(o.input(f.cofwr,i),o.add(di,de)));
        inner=o.add(inner,o.mul(o.input(f.cofwt,i),hu));
        inner=o.add(inner,o.mul(o.input(f.cofwt,im),hl));
        u[k]=o.add(u[k],o.mul(dt,inner));
    }
    for(int k=1;k<nlev;++k) {
        int i=CS_C(k,cell,ncells);
        u[k]=o.mul(o.sub(u[k],o.mul(o.mul(dts2,o.input(f.a,i)),u[k-1])),o.input(f.alpha,i));
    }
    for(int k=nlev-1;k>=0;--k) {
        int i=CS_C(k,cell,ncells);
        u[k]=o.sub(u[k],o.mul(o.input(f.gamma,i),u[k+1]));
    }
    for(int k=1;k<nlev;++k) {
        int i=CS_C(k,cell,ncells),im=CS_C(k-1,cell,ncells);
        V delta=o.sub(o.input(f.rws,i),o.input(f.rw,i));
        V zi=o.add(o.mul(o.input(f.fzm,k),o.input(f.zz,i)),o.mul(o.input(f.fzp,k),o.input(f.zz,im)));
        V rhoi=o.add(o.mul(o.input(f.fzm,k),o.input(f.rhoz,i)),o.mul(o.input(f.fzp,k),o.input(f.rhoz,im)));
        V value=o.add(u[k],delta);
        value=o.sub(value,o.mul(o.mul(o.mul(o.mul(dt,o.input(f.dss,i)),zi),rhoi),o.input(f.w,i)));
        value=o.div(value,o.add(o.constant(1.0f),o.mul(dt,o.input(f.dss,i))));
        u[k]=o.sub(value,delta);
        avg[k]=o.add(avg[k],o.mul(o.input(f.ewp,k),u[k]));
    }
    for(int k=0;k<nlev;++k) {
        int i=CS_C(k,cell,ncells),ip=CS_C(k+1,cell,ncells);
        V density=o.sub(o.input(f.rs,i),o.mul(o.mul(dt,o.input(f.cofrz,k)),
                    o.sub(o.mul(o.input(f.ewp,k+1),u[k+1]),o.mul(o.input(f.ewp,k),u[k]))));
        V heat=o.sub(o.input(f.ts,i),o.mul(o.mul(dt,o.input(f.rdzw,k)),
                 o.sub(o.mul(o.mul(o.input(f.ewp,k+1),o.input(f.coftz,ip)),u[k+1]),
                       o.mul(o.mul(o.input(f.ewp,k),o.input(f.coftz,i)),u[k]))));
        o.output(f.rhop,i,density);o.output(f.rtp,i,heat);
    }
    for(int k=0;k<=nlev;++k) {
        int i=CS_C(k,cell,ncells);o.output(f.rwp,i,u[k]);o.output(f.ww,i,avg[k]);
    }
    o.finish();
}
#define CS_PRIMAL_ARGS \
    const int nlev,const int ncells,const float dts,const float *zz,const float *rho_zz, \
    const float *fzm,const float *fzp,const float *rdzw,const float *dss,const float *w, \
    const float *rw,const float *rw_save,const float *tend_rw,const float *rs,const float *ts, \
    const float *cofwr,const float *cofwz,const float *coftz,const float *cofwt,const float *cofrz, \
    const float *a_tri,const float *alpha_tri,const float *gamma_tri,const float *etp,const float *etm, \
    const float *ewp,const float *ewm,float *rw_p,float *rho_pp,float *rtheta_pp,float *ww_avg
#define CS_SHADOW_ARGS \
    float *d_zz,float *d_rhoz,float *d_fzm,float *d_fzp,float *d_rdzw,float *d_dss,float *d_w, \
    float *d_rw,float *d_rws,float *d_tend,float *d_rs,float *d_ts,float *d_cofwr,float *d_cofwz, \
    float *d_coftz,float *d_cofwt,float *d_cofrz,float *d_a,float *d_alpha,float *d_gamma, \
    float *d_etp,float *d_etm,float *d_ewp,float *d_ewm,float *d_rwp,float *d_rhop,float *d_rtp,float *d_ww
#define CS_FIELDS ColumnFields f={ \
    {zz,d_zz,2},{rho_zz,d_rhoz,3},{fzm,d_fzm,4},{fzp,d_fzp,5},{rdzw,d_rdzw,6},{dss,d_dss,7}, \
    {w,d_w,8},{rw,d_rw,9},{rw_save,d_rws,10},{tend_rw,d_tend,11},{rs,d_rs,12},{ts,d_ts,13}, \
    {cofwr,d_cofwr,14},{cofwz,d_cofwz,15},{coftz,d_coftz,16},{cofwt,d_cofwt,17},{cofrz,d_cofrz,18}, \
    {a_tri,d_a,19},{alpha_tri,d_alpha,20},{gamma_tri,d_gamma,21},{etp,d_etp,22},{etm,d_etm,23}, \
    {ewp,d_ewp,24},{ewm,d_ewm,25},{rw_p,d_rwp,26},{rho_pp,d_rhop,27},{rtheta_pp,d_rtp,28},{ww_avg,d_ww,29} }
extern "C" __global__ void acoustic_column_solve_v841__tl(CS_PRIMAL_ARGS,const float *d_dts,
                            CS_SHADOW_ARGS,AcousticDual *work,AcousticFailure *failures,unsigned *branch_record) {
    int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=ncells)return;
    if(nlev<0){acoustic_fail(failures,0,0,5);return;}
    CS_FIELDS;unsigned branch_local[40]={0};unsigned *branch_counter=branch_record?branch_local:nullptr;
    AcousticTangent o{failures,branch_counter,6};float ddts=*d_dts;
    column_solve(o,nlev,ncells,cell,dts,&ddts,f,work+(unsigned long long)cell*2*(nlev+1));
    if(branch_record)for(int i=0;i<40;++i)if(branch_local[i]) atomicAdd(branch_record+i,branch_local[i]);
}
// Private per-thread tapes and per-column shared-field contributions; the
// barrier and ascending-column gather make accumulation deterministic.
extern "C" __global__ void acoustic_column_solve_v841__ad(CS_PRIMAL_ARGS,float *b_dts,
              CS_SHADOW_ARGS,float *seed_rwp,float *seed_rhop,
              float *seed_rtp,float *seed_ww,
              AcousticValue *work,AcousticNode *scratch,
              double *contrib,AcousticFailure *failures) {
    if(gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||nlev<0){
        if(!blockIdx.x&&!blockIdx.y&&!blockIdx.z&&!threadIdx.x&&!threadIdx.y&&!threadIdx.z)
            acoustic_fail(failures,0,0,6);
        return;
    }
    const int tid=threadIdx.x,s=nlev+1,stride=8*s+1,capacity=512*s+256;
    for(int cell=tid;cell<ncells;cell+=128) {
        double *partial=contrib+(unsigned long long)cell*stride;
        for(int q=0;q<stride;++q)partial[q]=0.0f;
        CS_FIELDS;
        // A null seed selects the tape's in-place path: it reads and clears
        // the output cotangent before adding the input cotangent in finish().
        f.rwp.seed=seed_rwp==d_rwp?nullptr:seed_rwp;
        f.rhop.seed=seed_rhop==d_rhop?nullptr:seed_rhop;
        f.rtp.seed=seed_rtp==d_rtp?nullptr:seed_rtp;
        f.ww.seed=seed_ww==d_ww?nullptr:seed_ww;
        f.fzm.accum=partial;f.fzp.accum=partial+s;f.rdzw.accum=partial+2*s;f.cofrz.accum=partial+3*s;
        f.etp.accum=partial+4*s;f.etm.accum=partial+5*s;f.ewp.accum=partial+6*s;f.ewm.accum=partial+7*s;
        AcousticReverse o{scratch+(unsigned long long)tid*capacity,0,capacity,failures};o.reset();
        column_solve(o,nlev,ncells,cell,dts,partial+8*s,f,work+(unsigned long long)tid*2*s);
        // The forward writes these exact sets, not the top boundary or
        // ww_avg's lower boundary. Leave incoming pass-through seeds intact.
        for(int k=0;k<nlev;++k){
            int i=CS_C(k,cell,ncells);
            seed_rwp[i]=0.0f;seed_rhop[i]=0.0f;seed_rtp[i]=0.0f;
            if(k>0)seed_ww[i]=0.0f;
        }
    }
    __syncthreads();
    float *dst[8]={d_fzm,d_fzp,d_rdzw,d_cofrz,d_etp,d_etm,d_ewp,d_ewm};
    for(int q=tid;q<stride;q+=128) {
        float *target;int plane=q/s,k=q%s;
        if(q<8*s) {if(k==nlev && plane<6)continue;target=dst[plane]+k;}else target=b_dts;
        double sum=mpas_f32_to_f64_ieee(*target);
        for(int cell=0;cell<ncells;++cell)sum+=contrib[(unsigned long long)cell*stride+q];
        if(!isfinite(sum))acoustic_fail(failures,0,q,3);
        *target=mpas_f64_to_f32_ieee(sum);
    }
}
