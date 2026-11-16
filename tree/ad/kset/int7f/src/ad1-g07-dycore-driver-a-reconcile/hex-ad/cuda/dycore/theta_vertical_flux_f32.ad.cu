// SPDX-License-Identifier: Apache-2.0
// Independent input owners gather vertical neighbors in deterministic order.
using AdFail = int;
// Integer branch counts are warp-aggregated and do not affect field reductions.
__device__ __forceinline__ void record_sign(unsigned int *record,int slot,unsigned int sign) {
 unsigned int live=__activemask(),negative=__ballot_sync(live,sign!=0);
 if ((threadIdx.x&31)==(__ffs((int)live)-1)) {
  atomicAdd(&record[2*slot],(unsigned int)__popc(live&~negative));
  atomicAdd(&record[2*slot+1],(unsigned int)__popc(negative));
 }
}

__device__ __forceinline__ bool finite_f32(float x) { return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u; }
__device__ __forceinline__ void refuse(AdFail *f,int slot,unsigned long long field,unsigned long long element) { atomicExch(&f[slot], 1); }
struct Terms {float value,dm2,dm1,di,dp1,ds0,ds1,drv,drs,dc;bool good;};
__device__ __forceinline__ Terms terms(int nlev,int ncells,int k,int c,float coef,
 const float *theta,const float *saved,const float *rw,const float *rs,
 const float *fzm,const float *fzp) {
    Terms t={0,0,0,0,0,0,0,0,0,0,true};
    if(k<=0 || k>=nlev) return t;
    int i=k*ncells+c;float qi=theta[i],qm1=theta[i-ncells];
    t.good=finite_f32(qi)&&finite_f32(qm1)&&finite_f32(coef);
    if(k==1 || k==nlev-1) {
        float v=(k==1)?rw[i]:rs[i];
        float s=fzm[k]*qi+fzp[k]*qm1;
        t.value=v*s;t.di=v*fzm[k];t.dm1=v*fzp[k];
        if(k==1)t.drv=s;else t.drs=s;
        t.good=t.good&&finite_f32(v)&&finite_f32(fzm[k])&&finite_f32(fzp[k])&&finite_f32(s);
    } else {
        float qm2=theta[i-2*ncells],qp1=theta[i+ncells],v=rw[i];
        float a=7.0f*(qi+qm1)-(qp1+qm2);
        float b=(qp1-qm2)-3.0f*(qi-qm1);
        float av=__uint_as_float(__float_as_uint(v)&0x7fffffffu);
        float sign=(__float_as_uint(v)&0x80000000u)?-1.0f:1.0f;
        t.value=v*a/12.0f+(coef*av)*b/12.0f;
        t.drv=(a+coef*sign*b)/12.0f;t.dc=av*b/12.0f;
        t.dm2=(-v-coef*av)/12.0f;t.dm1=(7.0f*v+3.0f*coef*av)/12.0f;
        t.di=(7.0f*v-3.0f*coef*av)/12.0f;t.dp1=(-v+coef*av)/12.0f;
        t.good=t.good&&finite_f32(qm2)&&finite_f32(qp1)&&finite_f32(v)&&finite_f32(a)&&finite_f32(b);
    }
    if(k<nlev-1) {
        float si=saved[i],sm=saved[i-ncells],v=rw[i],sv=rs[i];
        float q=fzm[k]*si+fzp[k]*sm, diff=sv-v;
        t.value+=diff*q;t.drv-=q;t.drs+=q;t.ds0=diff*fzm[k];t.ds1=diff*fzp[k];
        t.good=t.good&&finite_f32(si)&&finite_f32(sm)&&finite_f32(v)&&finite_f32(sv)&&
          finite_f32(fzm[k])&&finite_f32(fzp[k])&&finite_f32(q)&&finite_f32(diff);
    }
    t.good=t.good&&finite_f32(t.value)&&finite_f32(t.dm2)&&finite_f32(t.dm1)&&finite_f32(t.di)&&
      finite_f32(t.dp1)&&finite_f32(t.ds0)&&finite_f32(t.ds1)&&finite_f32(t.drv)&&finite_f32(t.drs)&&finite_f32(t.dc);
    return t;
}
extern "C" __global__ void theta_vertical_flux_f32__tl(int nlev,int ncells,float coefficient,
 const float *theta,const float *theta_saved,const float *rw,const float *rw_saved,
 const float *fzm,const float *fzp,const float *d_coefficient,const float *d_theta,
 const float *d_theta_saved,const float *d_rw,const float *d_rw_saved,float *d_flux,AdFail *fail, unsigned int *branch_record) {
    int nout=(nlev+1)*ncells;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nout;i+=blockDim.x*gridDim.x) {
        int k=i/ncells,c=i%ncells;float v=0;
        Terms t=terms(nlev,ncells,k,c,coefficient,theta,theta_saved,rw,rw_saved,fzm,fzp);
        if(k>1 && k<nlev-1) record_sign(branch_record,3,__float_as_uint(rw[i])>>31);
        if(k>0 && k<nlev) {
            v=t.di*d_theta[i]+t.dm1*d_theta[i-ncells]+t.drv*d_rw[i]+t.drs*d_rw_saved[i]+t.dc*(*d_coefficient);
            if(k!=1 && k!=nlev-1) v+=t.dm2*d_theta[i-2*ncells]+t.dp1*d_theta[i+ncells];
            if(k<nlev-1) v+=t.ds0*d_theta_saved[i]+t.ds1*d_theta_saved[i-ncells];
            if(!t.good||!finite_f32(*d_coefficient)||!finite_f32(d_theta[i])||!finite_f32(d_theta[i-ncells])||
               !finite_f32(d_rw[i])||!finite_f32(d_rw_saved[i])||!finite_f32(v) ||
               (k!=1 && k!=nlev-1 && (!finite_f32(d_theta[i-2*ncells])||!finite_f32(d_theta[i+ncells]))) ||
               (k<nlev-1 && (!finite_f32(d_theta_saved[i])||!finite_f32(d_theta_saved[i-ncells])))) refuse(fail,0,1,i);
        }
        d_flux[i]=v;
    }
}
extern "C" __global__ void theta_vertical_flux_f32__ad(int nlev,int ncells,float coefficient,
 const float *theta,const float *theta_saved,const float *rw,const float *rw_saved,
 const float *fzm,const float *fzp,const float *b_flux, float *b_flux_consumed,float *b_coefficient,float *b_theta,
 float *b_theta_saved,float *b_rw,float *b_rw_saved,AdFail *fail, unsigned int *branch_record) {
    int nout=(nlev+1)*ncells,nstate=nlev*ncells;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nout;i+=blockDim.x*gridDim.x) {
        // b_flux is the immutable pre-adjoint snapshot; this is the disjoint incoming seed.
        if (i < nout) b_flux_consumed[i] = 0.0f;
        int k=i/ncells,c=i%ncells;
        Terms t=terms(nlev,ncells,k,c,coefficient,theta,theta_saved,rw,rw_saved,fzm,fzp);
        if(k>1 && k<nlev-1) record_sign(branch_record,3,__float_as_uint(rw[i])>>31);
        if(!t.good||!finite_f32(b_flux[i])) refuse(fail,1,7,i);
        double r=(double)b_rw[i],s=(double)b_rw_saved[i];
        if(k>0&&k<nlev) {r+=t.drv*b_flux[i];s+=t.drs*b_flux[i];}
        if(!finite_f32(b_rw[i])||!finite_f32(b_rw_saved[i])||!finite_f32(r)||!finite_f32(s)||(b_rw==b_rw_saved&&!finite_f32(r+s-(double)b_rw[i]))) refuse(fail,2,3,i);
        if(b_rw==b_rw_saved) {b_rw[i]=r+s-(double)b_rw[i];} else {b_rw[i]=r;b_rw_saved[i]=s;}
        if(i<nstate) {
            double a=(double)b_theta[i],b=(double)b_theta_saved[i];
            for(int q=k-1;q<=k+2;++q) if(q>0&&q<nlev) {
                Terms z=terms(nlev,ncells,q,c,coefficient,theta,theta_saved,rw,rw_saved,fzm,fzp);
                float w=k==q?z.di:k==q-1?z.dm1:k==q-2?z.dm2:z.dp1;
                a+=w*b_flux[q*ncells+c];
                if(k==q) b+=z.ds0*b_flux[q*ncells+c];
                if(k==q-1) b+=z.ds1*b_flux[q*ncells+c];
            }
            if(!finite_f32(b_theta[i])||!finite_f32(b_theta_saved[i])||!finite_f32(a)||!finite_f32(b)||(b_theta==b_theta_saved&&!finite_f32(a+b-(double)b_theta[i]))) refuse(fail,3,1,i);
            if(b_theta==b_theta_saved) {b_theta[i]=a+b-(double)b_theta[i];} else {b_theta[i]=a;b_theta_saved[i]=b;}
        }
    }
    // Fixed-order block reduction, with f64 partial sums and one f32 output rounding.
    if(blockIdx.x==0) {
        __shared__ double reduction[1024];double sum=0.0;
        for(int j=threadIdx.x;j<nout;j+=blockDim.x) {int q=j/ncells,cell=j%ncells;
            Terms z=terms(nlev,ncells,q,cell,coefficient,theta,theta_saved,rw,rw_saved,fzm,fzp);
            sum+=(double)(z.dc*b_flux[j]);
        }
        int lane=threadIdx.x;reduction[lane]=sum;__syncthreads();
        for(int stride=1;stride<blockDim.x;stride*=2) {if(lane%(2*stride)==0 && lane+stride<blockDim.x)reduction[lane]+=reduction[lane+stride];__syncthreads();}
        if(lane==0) {float b=(float)((double)*b_coefficient+reduction[0]);
            if(!finite_f32(*b_coefficient)||!finite_f32(b)) refuse(fail,4,8,0);
            *b_coefficient=b;
        }
    }
}
