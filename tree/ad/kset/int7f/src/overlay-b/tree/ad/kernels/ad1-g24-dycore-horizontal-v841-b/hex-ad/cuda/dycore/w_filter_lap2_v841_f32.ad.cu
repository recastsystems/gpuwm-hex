// SPDX-License-Identifier: Apache-2.0
// rho, velocity, kdiff and del2 are active, so this is not a fixed linear map.
#include "mpas_f32.cuh"
typedef int Lap2Failure;
// One zero-initialized int32 flag per refusal source; integer atomics are deterministic.
__device__ __forceinline__ void l2_fail(Lap2Failure *f,unsigned long long slot,unsigned long long field,unsigned long long index,unsigned long long source){(void)slot;(void)field;(void)index;atomicExch(&f[source-1],1);}
struct Lap2Term {int e,c0,c1;float factor,rs,wd,flux,ks;double br,bw,bk,bd;bool valid;};
__device__ __forceinline__ Lap2Term l2_term(int4 g,int owner,int k,const float*w,const float*rho,const float*kd,const float*d2,int nc,int ne,float bl,float bt){
 Lap2Term t={};t.e=g.x;t.c0=g.y;t.c1=g.z;t.factor=__int_as_float(g.w);
 if(t.e<0||t.e>=ne||t.c0<0||t.c0>=nc||t.c1<0||t.c1>=nc||t.c0==t.c1||(owner!=t.c0&&owner!=t.c1))return t;
 int e=t.e,a=t.c0,b=t.c1;float r0=rho[(k-1)*ne+e],r1=rho[k*ne+e],x=w[k*nc+a],y=w[k*nc+b];
 float k0=kd[(k-1)*nc+a],k1=kd[(k-1)*nc+b],k2=kd[k*nc+a],k3=kd[k*nc+b],del=d2[e];
 if(!isfinite(t.factor)||!isfinite(r0)||!isfinite(r1)||!isfinite(x)||!isfinite(y)||!isfinite(k0)||!isfinite(k1)||!isfinite(k2)||!isfinite(k3)||!isfinite(del)||!isfinite(bl)||!isfinite(bt))return t;
 t.rs=mpas_add(r1,r0);t.wd=mpas_sub(y,x);t.ks=mpas_add(mpas_add(mpas_add(k2,k3),k0),k1);
 float fr=mpas_mul(t.factor,t.rs);t.flux=mpas_mul(fr,t.wd);
 double q=mpas_f32_to_f64_ieee(bt)*mpas_f32_to_f64_ieee(t.ks)*0.25;double bf=mpas_f32_to_f64_ieee(bl)+q*mpas_f32_to_f64_ieee(del);
 t.br=bf*mpas_f32_to_f64_ieee(t.wd)*mpas_f32_to_f64_ieee(t.factor);t.bw=bf*mpas_f32_to_f64_ieee(fr);t.bk=mpas_f32_to_f64_ieee(bt)*mpas_f32_to_f64_ieee(mpas_mul(mpas_mul(t.flux,del),0.25f));t.bd=q*mpas_f32_to_f64_ieee(t.flux);
 t.valid=isfinite(t.flux)&&isfinite(t.ks)&&isfinite(t.br)&&isfinite(t.bw)&&isfinite(t.bk)&&isfinite(t.bd);return t;
}
extern "C" __global__ void w_filter_lap2_v841_f32__tl(const float*w,const float*rho,const float*kd,const float*d2,const int*eoc,const int*neoc,const int*coe,const float*dc,const float*dv,const int4*factors,const int nlev,const int nc,const int ne,const int max_edges,float*lap,float*tend,const float*dw,const float*drho,const float*dkd,const float*dd2,float*dlap,float*dtend,Lap2Failure*failure){
 (void)eoc;(void)coe;(void)dc;(void)dv;(void)lap;(void)tend;int stride=blockDim.x*gridDim.x;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nlev*nc;i+=stride){int c=i%nc,k=i/nc,count=neoc[c];if(k==0){dlap[i]=0;dtend[i]=0;continue;}if(count<0||count>max_edges){l2_fail(failure,i, 6, c, 1);continue;}float a=0,b=0;
  for(int j=0;j<count;++j){int4 g=factors[j*nc+c];Lap2Term t=l2_term(g,c,k,w,rho,kd,d2,nc,ne,0.0f,0.0f);if(!t.valid){l2_fail(failure,i, 10, i, 2);continue;}
   int e=t.e,c0=t.c0,c1=t.c1;float dr=mpas_add(drho[k*ne+e],drho[(k-1)*ne+e]);float dwd=mpas_sub(dw[k*nc+c1],dw[k*nc+c0]);
   float df=mpas_add(mpas_mul(mpas_mul(t.factor,dr),t.wd),mpas_mul(mpas_mul(t.factor,t.rs),dwd));
   float dk=mpas_add(mpas_add(mpas_add(dkd[k*nc+c0],dkd[k*nc+c1]),dkd[(k-1)*nc+c0]),dkd[(k-1)*nc+c1]);
   float z=mpas_mul(mpas_mul(d2[e],0.25f),t.ks);float q=mpas_mul(mpas_mul(t.flux,0.25f),t.ks);float h=mpas_mul(mpas_mul(t.flux,d2[e]),0.25f);
   a=mpas_add(a,df);b=mpas_add(b,mpas_add(mpas_add(mpas_mul(z,df),mpas_mul(q,dd2[e])),mpas_mul(h,dk)));
  }if(!isfinite(a)||!isfinite(b))l2_fail(failure,i, 15, i, 3);dlap[i]=a;dtend[i]=b;
 }
}
extern "C" __global__ void w_filter_lap2_v841_f32__ad(const float*w,const float*rho,const float*kd,const float*d2,const int*eoc,const int*neoc,const int*coe,const float*dc,const float*dv,const int4*factors,const int nlev,const int nc,const int ne,const int max_edges,float*lap,float*tend,const float*blap_snapshot,const float*btend_snapshot,float*blap,float*btend,float*bw,float*brho,float*bkd,float*bd2,float*btend_in,const int tendency_elems,Lap2Failure*failure){
 (void)dc;(void)dv;(void)lap;(void)tend;int first=blockDim.x*blockIdx.x+threadIdx.x,stride=blockDim.x*gridDim.x;unsigned long long nw=(unsigned long long)nlev*nc,nr=(unsigned long long)nlev*ne;
 // The consumed seed b_tendency is cleared over its whole extent, tail included: the composed sweeps bind it to the
 // same shadow the separate b_tendency_in is added back into, so a tail left in place was counted twice (the U6 dry
 // per-launch dot FAIL and the hrrrv5 stage-A FAIL rows, 2026-11-05). With separate buffers the clear is harmless.
 // One owner per input cell and level, gathering the adjacent output cells.
 for(int i=first;i<nlev*nc;i+=stride){int c=i%nc,k=i/nc,count=neoc[c];double a=mpas_f32_to_f64_ieee(bw[i]),b=mpas_f32_to_f64_ieee(bkd[i]);unsigned long long slot=i;
  if(count<0||count>max_edges||!isfinite(a)||!isfinite(b)){l2_fail(failure,slot, 1, i, 4);continue;}
  for(int j=0;j<count;++j){int e=eoc[c*max_edges+j];if(e<0||e>=ne){l2_fail(failure,slot, 5, c, 5);continue;}int c0=coe[2*e],c1=coe[2*e+1];if(c0<0||c0>=nc||c1<0||c1>=nc||(c!=c0&&c!=c1)){l2_fail(failure,slot, 7, e, 6);continue;}
   for(int side=0;side<2;++side){int owner=side?c1:c0,n=neoc[owner];if(n<0||n>max_edges){l2_fail(failure,slot, 6, owner, 7);continue;}
    for(int p=0;p<n;++p){int4 g=factors[p*nc+owner];if(g.x!=e)continue;
     if(k>0){Lap2Term t=l2_term(g,owner,k,w,rho,kd,d2,nc,ne,blap_snapshot[k*nc+owner],btend_snapshot[k*nc+owner]);if(!t.valid||(c!=t.c0&&c!=t.c1)){l2_fail(failure,slot, 10, i, 8);continue;}a+=c==t.c1?t.bw:-t.bw;b+=t.bk;}
     if(k+1<nlev){Lap2Term t=l2_term(g,owner,k+1,w,rho,kd,d2,nc,ne,blap_snapshot[(k+1)*nc+owner],btend_snapshot[(k+1)*nc+owner]);if(!t.valid||(c!=t.c0&&c!=t.c1)){l2_fail(failure,slot, 10, i, 9);continue;}b+=t.bk;}
    }
   }
  }
  float out_a=mpas_f64_to_f32_ieee(a),out_b=mpas_f64_to_f32_ieee(b);if(!isfinite(out_a)||!isfinite(out_b))l2_fail(failure,slot, 1, i, 10);bw[i]=out_a;bkd[i]=out_b;
 }
 // One owner per input edge and level; each rho level affects two forward levels.
 for(int i=first;i<nlev*ne;i+=stride){int e=i%ne,k=i/ne,c0=coe[2*e],c1=coe[2*e+1];double b=mpas_f32_to_f64_ieee(brho[i]);unsigned long long slot=nw+i;if(c0<0||c0>=nc||c1<0||c1>=nc||!isfinite(b)){l2_fail(failure,slot, 2, i, 11);continue;}
  for(int side=0;side<2;++side){int owner=side?c1:c0,n=neoc[owner];if(n<0||n>max_edges){l2_fail(failure,slot, 6, owner, 12);continue;}for(int p=0;p<n;++p){int4 g=factors[p*nc+owner];if(g.x!=e)continue;
   if(k>0){Lap2Term t=l2_term(g,owner,k,w,rho,kd,d2,nc,ne,blap_snapshot[k*nc+owner],btend_snapshot[k*nc+owner]);if(t.valid)b+=t.br;else l2_fail(failure,slot, 2, i, 13);}
   if(k+1<nlev){Lap2Term t=l2_term(g,owner,k+1,w,rho,kd,d2,nc,ne,blap_snapshot[(k+1)*nc+owner],btend_snapshot[(k+1)*nc+owner]);if(t.valid)b+=t.br;else l2_fail(failure,slot, 2, i, 14);}
  }}float out=mpas_f64_to_f32_ieee(b);if(!isfinite(out))l2_fail(failure,slot, 2, i, 15);brho[i]=out;
 }
 for(int e=first;e<ne;e+=stride){unsigned long long slot=nw+nr+e;int c0=coe[2*e],c1=coe[2*e+1];double b=mpas_f32_to_f64_ieee(bd2[e]);if(c0<0||c0>=nc||c1<0||c1>=nc||!isfinite(b)){l2_fail(failure,slot, 4, e, 16);continue;}
  for(int side=0;side<2;++side){int owner=side?c1:c0,n=neoc[owner];if(n<0||n>max_edges){l2_fail(failure,slot, 6, owner, 17);continue;}for(int p=0;p<n;++p){int4 g=factors[p*nc+owner];if(g.x!=e)continue;for(int k=1;k<nlev;++k){Lap2Term t=l2_term(g,owner,k,w,rho,kd,d2,nc,ne,blap_snapshot[k*nc+owner],btend_snapshot[k*nc+owner]);if(t.valid)b+=t.bd;else l2_fail(failure,slot, 4, e, 18);}}}float out=mpas_f64_to_f32_ieee(b);if(!isfinite(out))l2_fail(failure,slot, 4, e, 19);bd2[e]=out;
 }
 for(int i=first;i<tendency_elems;i+=stride){unsigned long long slot=nw+nr+ne+i;float seed=btend_snapshot[i],b=btend_in[i];if(!isfinite(seed)||!isfinite(b)){l2_fail(failure,slot, 16, i, 20);continue;}btend[i]=0;if(i>=nlev*nc)btend_in[i]=mpas_add(b,seed);}
 for(int i=first;i<nlev*nc;i+=stride)blap[i]=0;
}
