// SPDX-License-Identifier: Apache-2.0
// Hand rule: del4 and h4 are active, so the full map is not fixed-linear.
#include "mpas_f32.cuh"
typedef int WFailure;
// One zero-initialized int32 flag per refusal source; integer atomics are deterministic.
__device__ __forceinline__ void wf_fail(WFailure *f,unsigned long long slot,unsigned long long field,unsigned long long index,unsigned long long source){(void)slot;(void)field;(void)index;atomicExch(&f[source-1],1);}
struct WTerm {float fac, bd, bh; bool valid;};
__device__ __forceinline__ WTerm wf_term(int edge,int cell,int c0,int c1,int k,
 const float *w,const float *d4,const float *dc,const float *dv,const float *area,
 float h4,int ncells,float seed){
 WTerm r={0,0,0,false};
 float ar=area[cell],dist=dc[edge],dvx=dv[edge],x=d4[edge];
 float a=w[k*ncells+c0],b=w[k*ncells+c1];
 if(!isfinite(ar)||ar==0.0f||!isfinite(dist)||dist==0.0f||!isfinite(dvx)||!isfinite(x)||!isfinite(h4)||!isfinite(a)||!isfinite(b)||!isfinite(seed))return r;
 float ia=mpas_div(1.0f,ar),id=mpas_div(1.0f,dist),sgn=cell==c0?1.0f:-1.0f;
 float h=mpas_mul(h4,ia);float z=mpas_mul(x,h);z=mpas_mul(z,dvx);z=mpas_mul(z,sgn);r.fac=mpas_mul(z,id);
 float bar=mpas_mul(-seed,mpas_sub(b,a));bar=mpas_mul(bar,id);bar=mpas_mul(bar,sgn);bar=mpas_mul(bar,dvx);
 r.bd=mpas_mul(bar,h);r.bh=mpas_mul(mpas_mul(bar,x),ia);
 r.valid=isfinite(r.fac)&&isfinite(r.bd)&&isfinite(r.bh);return r;
}
extern "C" __global__ void w_filter_lap4_v841_f32__tl(
 const float *w,const float *d4,const int *eoc,const int *neoc,const int *coe,
 const float *dc,const float *dv,const float *area,const float h4,const int nlev,
 const int ncells,const int max_edges,float *tendency,
 const float *d_w,const float *d_d4,const float *d_h4,float *d_tendency,
 const int nedges,WFailure *failure){
 (void)tendency;int stride=gridDim.x*blockDim.x;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<(nlev-1)*ncells;i+=stride){
  int c=i%ncells,k=1+i/ncells,tid=k*ncells+c,count=neoc[c];float sum=d_tendency[tid];
  if(count<0||count>max_edges||!isfinite(sum)||!isfinite(*d_h4)){wf_fail(failure,i, 13, tid, 1);continue;}
  for(int slot=0;slot<count;++slot){
   int e=eoc[c*max_edges+slot];if(e<0||e>=nedges){wf_fail(failure,i, 3, c, 2);break;}
   int c0=coe[2*e],c1=coe[2*e+1];if(c0<0||c0>=ncells||c1<0||c1>=ncells||(c!=c0&&c!=c1)){wf_fail(failure,i, 5, e, 3);break;}
   WTerm t=wf_term(e,c,c0,c1,k,w,d4,dc,dv,area,h4,ncells,1.0f);
   float u=d_w[k*ncells+c0],v=d_w[k*ncells+c1],dd=d_d4[e];
   if(!t.valid||!isfinite(u)||!isfinite(v)||!isfinite(dd)){wf_fail(failure,i, 1, tid, 4);break;}
   // bd and bh are derivatives of the complete negative flux when seed=1.
   float change=mpas_add(mpas_mul(t.bd,dd),mpas_mul(t.bh,*d_h4));
   change=mpas_sub(change,mpas_mul(t.fac,mpas_sub(v,u)));sum=mpas_add(sum,change);
  }
  if(!isfinite(sum))wf_fail(failure,i, 13, tid, 5);d_tendency[tid]=sum;
 }
}
extern "C" __global__ void w_filter_lap4_v841_f32__ad(
 const float *w,const float *d4,const int *eoc,const int *neoc,const int *coe,
 const float *dc,const float *dv,const float *area,const float h4,const int nlev,
 const int ncells,const int max_edges,float *tendency,
 const float *b_tendency_snapshot,float *b_tendency,float *b_w,float *b_d4,
 float *b_h4,float *b_tendency_in,const int nedges,const int tendency_elems,WFailure *failure){
 (void)tendency;int first=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 unsigned long long n=(unsigned long long)nlev*ncells;
 // The consumed seed b_tendency is cleared over its whole extent (level 0 and the tail too): the composed sweeps add
 // b_tendency_in back into the same shadow, so rows left in place were counted twice (2026-11-05 stage-A FAIL rows).
 // A frozen copy of the incoming cotangent prevents clearing/gather races.
 for(int i=first;i<nlev*ncells;i+=stride){int c=i%ncells,k=i/ncells;if(k==0)continue;
  float seed=b_tendency_snapshot[i];double b=mpas_f32_to_f64_ieee(b_w[i]);int count=neoc[c];
  if(!isfinite(seed)||!isfinite(b)||count<0||count>max_edges){wf_fail(failure,i, 1, i, 6);continue;}
  for(int slot=0;slot<count;++slot){int e=eoc[c*max_edges+slot];if(e<0||e>=nedges){wf_fail(failure,i, 3, c, 7);break;}
   int c0=coe[2*e],c1=coe[2*e+1];if(c0<0||c0>=ncells||c1<0||c1>=ncells||c0==c1||(c!=c0&&c!=c1)){wf_fail(failure,i, 5, e, 8);break;}
   for(int side=0;side<2;++side){int owner=side?c1:c0,nn=neoc[owner];if(nn<0||nn>max_edges){wf_fail(failure,i, 4, owner, 9);continue;}
    for(int j=0;j<nn;++j)if(eoc[owner*max_edges+j]==e){float outseed=b_tendency_snapshot[k*ncells+owner];WTerm t=wf_term(e,owner,c0,c1,k,w,d4,dc,dv,area,h4,ncells,outseed);
     if(!t.valid){wf_fail(failure,i, 1, i, 10);continue;}float v=mpas_mul(outseed,t.fac);double v64=mpas_f32_to_f64_ieee(v);b+=c==c0?v64:-v64;
    }
   }
  }
  float result=mpas_f64_to_f32_ieee(b);if(!isfinite(result))wf_fail(failure,i, 1, i, 11);b_w[i]=result;
 }
 for(int e=first;e<nedges;e+=stride){unsigned long long s=n+e;double b=mpas_f32_to_f64_ieee(b_d4[e]);int c0=coe[2*e],c1=coe[2*e+1];
  if(c0<0||c0>=ncells||c1<0||c1>=ncells||c0==c1||!isfinite(b)){wf_fail(failure,s, 5, e, 12);continue;}
  for(int side=0;side<2;++side){int owner=side?c1:c0,count=neoc[owner];if(count<0||count>max_edges){wf_fail(failure,s, 4, owner, 13);continue;}
   for(int j=0;j<count;++j)if(eoc[owner*max_edges+j]==e)for(int k=1;k<nlev;++k){float seed=b_tendency_snapshot[k*ncells+owner];WTerm t=wf_term(e,owner,c0,c1,k,w,d4,dc,dv,area,h4,ncells,seed);if(!t.valid){wf_fail(failure,s, 2, e, 14);continue;}b+=mpas_f32_to_f64_ieee(t.bd);}
  }
  float result=mpas_f64_to_f32_ieee(b);if(!isfinite(result))wf_fail(failure,s, 2, e, 15);b_d4[e]=result;
 }
 for(int i=first;i<tendency_elems;i+=stride){int k=i/ncells;unsigned long long s=n+nedges+i;
  float seed=b_tendency_snapshot[i];float prior=b_tendency_in[i];if(!isfinite(seed)||!isfinite(prior)){wf_fail(failure,s, 13, i, 16);continue;}
  b_tendency_in[i]=mpas_add(prior,seed);b_tendency[i]=0.0f;(void)k;
 }
 // Fixed-order block reduction; double accumulation limits cancellation in h4's cotangent.
 // Every thread of block zero reaches the barrier after its independent gathers.
 __shared__ double partial_h4[1024];
 if(blockIdx.x==0){unsigned long long s=n+nedges+tendency_elems+threadIdx.x;double local=0.0;
  if(threadIdx.x==0&&!isfinite(*b_h4))wf_fail(failure,s, 9, 0, 17);
  for(int i=threadIdx.x;i<(nlev-1)*ncells;i+=blockDim.x){int k=1+i/ncells,c=i%ncells,count=neoc[c];if(count<0||count>max_edges){wf_fail(failure,s, 4, c, 18);continue;}
   for(int j=0;j<count;++j){int e=eoc[c*max_edges+j];if(e<0||e>=nedges){wf_fail(failure,s, 3, c, 19);continue;}int c0=coe[2*e],c1=coe[2*e+1];if(c0<0||c0>=ncells||c1<0||c1>=ncells||(c!=c0&&c!=c1)){wf_fail(failure,s, 5, e, 20);continue;}
    WTerm t=wf_term(e,c,c0,c1,k,w,d4,dc,dv,area,h4,ncells,b_tendency_snapshot[k*ncells+c]);if(!t.valid){wf_fail(failure,s, 9, 0, 21);continue;}local+=mpas_f32_to_f64_ieee(t.bh);
   }
  }
  partial_h4[threadIdx.x]=local;__syncthreads();
  if(threadIdx.x==0){double b=mpas_f32_to_f64_ieee(*b_h4);for(int t=0;t<blockDim.x;++t)b+=partial_h4[t];float result=mpas_f64_to_f32_ieee(b);if(!isfinite(result))wf_fail(failure,s, 9, 0, 22);*b_h4=result;}
 }
}
