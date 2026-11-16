// SPDX-License-Identifier: Apache-2.0
// Derivative of recovery.py:176. Original forward remains unchanged.
#include "recovery_math.cuh"
#include "recovery_branch_record.cuh"
template<class T> __device__ __forceinline__ T tw_add(T a,T b){return a+b;} template<class T> __device__ __forceinline__ T tw_sub(T a,T b){return a-b;}
template<> __device__ __forceinline__ float tw_add(float a,float b){return mpas_add(a,b);} template<> __device__ __forceinline__ float tw_sub(float a,float b){return mpas_sub(a,b);}
template<class T> __device__ __forceinline__ T tw_mul(T a,T b){return a*b;}
template<> __device__ __forceinline__ float tw_mul(float a,float b){return mpas_mul(a,b);}
template<class T> __device__ __forceinline__ T tw_div(T a,T b){return a/b;}
template<> __device__ __forceinline__ float tw_div(float a,float b){return mpas_div(a,b);}
template<class T> __device__ __forceinline__ T tw_sign(T x){return copysign(T(1),x);}
template<> __device__ __forceinline__ float tw_sign(float x){return mpas_copysign(1.0f,x);}
template<class T> __device__ __forceinline__ bool tw_nonzero(T x){return x!=T(0);}
template<> __device__ __forceinline__ bool tw_nonzero(float x){return mpas_f32_to_f64_ieee(x)!=0.0;}
__device__ __forceinline__ void tw_bad(int *f,int slot,unsigned long long field,unsigned long long source){(void)slot; (void)field; atomicExch(f + source - 1, 1);}
template<class T> struct Terrain {
 const T *rho,*ru,*rw,*zz,*fzm,*fzp;const int *eoc,*neoc;const T *sgn,*zb,*zb3;
 int nlev,ncells,nedges,max_edges;T cf1,cf2,cf3;
};
template<class T> __device__ __forceinline__ T tw_flux(const Terrain<T>& a,int l,int edge) {
 if(l==0)return tw_add<T>(tw_add<T>(tw_mul<T>(a.cf1,a.ru[edge]),tw_mul<T>(a.cf2,a.ru[a.nedges+edge])),tw_mul<T>(a.cf3,a.ru[2*a.nedges+edge]));
 return tw_add<T>(tw_mul<T>(a.fzm[l],a.ru[l*a.nedges+edge]),tw_mul<T>(a.fzp[l],a.ru[(l-1)*a.nedges+edge]));
}
template<class T> struct TVals {T metric,first,value,density,w;};
template<class T> __device__ __forceinline__ bool tw_values(const Terrain<T>& a,int l,int c,TVals<T>& v) {
 if(l<0||l>=a.nlev || c<0||c>=a.ncells || a.nlev<3 || a.max_edges<=0)return false;
 int slots=a.neoc[c];if(slots<0||slots>a.max_edges)return false;
 if(!isfinite(a.cf1)||!isfinite(a.cf2)||!isfinite(a.cf3))return false;
 if(l==0){
  v.metric=T(1);v.first=T(0);v.value=T(0);
  v.density=tw_add<T>(tw_add<T>(tw_mul<T>(a.cf1,a.rho[c]),tw_mul<T>(a.cf2,a.rho[a.ncells+c])),tw_mul<T>(a.cf3,a.rho[2*a.ncells+c]));
  for(int k=0;k<3;++k)if(!isfinite(a.rho[k*a.ncells+c])||!(recovery_wide<T>(a.rho[k*a.ncells+c])>0.0))return false;
 }else{
  int i=l*a.ncells+c,j=i-a.ncells;
  if(!isfinite(a.rho[i])||!isfinite(a.rho[j])||!(recovery_wide<T>(a.rho[i])>0.0)||!(recovery_wide<T>(a.rho[j])>0.0)||!isfinite(a.rw[i])||!isfinite(a.zz[i])||!isfinite(a.zz[j])||!isfinite(a.fzm[l])||!isfinite(a.fzp[l]))return false;
  v.metric=tw_add<T>(tw_mul<T>(a.fzm[l],a.zz[i]),tw_mul<T>(a.fzp[l],a.zz[j]));
  v.density=tw_add<T>(tw_mul<T>(a.fzm[l],a.rho[i]),tw_mul<T>(a.fzp[l],a.rho[j]));
  if(!isfinite(v.metric)||!tw_nonzero<T>(v.metric))return false;
  v.first=tw_div<T>(a.rw[i],v.metric);v.value=v.first;
 }
 if(!isfinite(v.density)||!(recovery_wide<T>(v.density)>0.0)||!isfinite(v.first))return false;
 for(int s=0;s<slots;++s){
  int e=a.eoc[c*a.max_edges+s],mi=(l*a.ncells+c)*a.max_edges+s,si=c*a.max_edges+s;
  if(e<0||e>=a.nedges||!isfinite(a.sgn[si])||!isfinite(a.zb[mi])||!isfinite(a.zb3[mi]))return false;
  if(l==0){for(int k=0;k<3;++k)if(!isfinite(a.ru[k*a.nedges+e]))return false;}
  else if(!isfinite(a.ru[l*a.nedges+e])||!isfinite(a.ru[(l-1)*a.nedges+e]))return false;
  T flux=tw_flux<T>(a,l,e),g=tw_add<T>(a.zb[mi],tw_mul<T>(tw_sign<T>(flux),a.zb3[mi]));
  v.value=tw_add<T>(v.value,tw_mul<T>(tw_mul<T>(a.sgn[si],g),flux));
 }
 v.w=tw_div<T>(v.value,v.density);return isfinite(v.value)&&isfinite(v.w);
}
template<class T> __device__ __forceinline__ void tw_bars(const TVals<T>& v,T by,T &bv,T &bd,T &bm,T &brw){
 bv=tw_div<T>(by,v.density);bd=recovery_neg<T>(tw_div<T>(tw_mul<T>(by,v.w),v.density));
 bm=recovery_neg<T>(tw_div<T>(tw_mul<T>(bv,v.first),v.metric));brw=tw_div<T>(bv,v.metric);
}
template<class T> __device__ __forceinline__ T tw_bflux(const Terrain<T>& a,int l,int c,int s,T bv){
 int edge=a.eoc[c*a.max_edges+s],mi=(l*a.ncells+c)*a.max_edges+s,si=c*a.max_edges+s;
 T f=tw_flux<T>(a,l,edge),g=tw_add<T>(a.zb[mi],tw_mul<T>(tw_sign<T>(f),a.zb3[mi]));
 return tw_mul<T>(tw_mul<T>(bv,a.sgn[si]),g);
}
template<class T> __device__ __forceinline__ bool tw_contrib(const Terrain<T>& a,const T *bw,int l,int c,TVals<T>& v,T& bv,T& bd,T& bm,T& brw){
 if(!tw_values<T>(a,l,c,v)||!isfinite(bw[l*a.ncells+c]))return false;
 tw_bars<T>(v,bw[l*a.ncells+c],bv,bd,bm,brw);
 return isfinite(bv)&&isfinite(bd)&&isfinite(bm)&&isfinite(brw);
}
template<class T> __device__ __forceinline__ void tw_record(RecoveryBranchTrace<T>& t,const Terrain<T>& a,int l,int c){
 T value=T(0);int ns=a.max_edges,nc=a.ncells,ne=a.nedges,i=l*nc+c;int sign_slot=(sizeof(T)==4?15:4)+(l==0?0:1);
 if(l!=0){int j=i-nc;T metric=t.add(t.mul(a.fzm[l],a.zz[i]),t.mul(a.fzp[l],a.zz[j]));value=t.div(a.rw[i],metric);}
 for(int s=0;s<a.neoc[c];++s){int e=a.eoc[c*ns+s],mi=i*ns+s;T f;
  if(l==0)f=t.add(t.add(t.mul(a.cf1,a.ru[e]),t.mul(a.cf2,a.ru[ne+e])),t.mul(a.cf3,a.ru[2*ne+e]));
  else f=t.add(t.mul(a.fzm[l],a.ru[l*ne+e]),t.mul(a.fzp[l],a.ru[(l-1)*ne+e]));
  T sign=t.sign(f,sign_slot),m=t.add(a.zb[mi],t.mul(sign,a.zb3[mi]));
  value=t.add(value,t.mul(t.mul(a.sgn[c*ns+s],m),f));
 }
 T density;
 if(l==0)density=t.add(t.add(t.mul(a.cf1,a.rho[c]),t.mul(a.cf2,a.rho[nc+c])),t.mul(a.cf3,a.rho[2*nc+c]));
 else {int j=i-nc;density=t.add(t.mul(a.fzm[l],a.rho[i]),t.mul(a.fzp[l],a.rho[j]));}
 (void)t.div(value,density);
}
#define TERRAIN_ARGS(T) const T *rho,const T *ru,const T *rw,const T *zz,const T *fzm,const T *fzp,const int *eoc,const int *neoc,const T *sgn,const T *zb,const T *zb3,T *w,int nlev,int ncells,int nedges,int max_edges,T cf1,T cf2,T cf3
#define TERRAIN_INIT(T) Terrain<T> a{rho,ru,rw,zz,fzm,fzp,eoc,neoc,sgn,zb,zb3,nlev,ncells,nedges,max_edges,cf1,cf2,cf3};(void)w
#define DEFINE_TERRAIN(T,SUFFIX) \
extern "C" __global__ void recover_terrain_w_##SUFFIX##__tl( TERRAIN_ARGS(T), \
 const T *drho,const T *dru,const T *drw,const T *dzz,const T *dfzm,const T *dfzp,const T *dsgn,const T *dzb,const T *dzb3,const T *dcf1_in,const T *dcf2_in,const T *dcf3_in,T *dw,int *fail,unsigned int *branch_record) { \
 TERRAIN_INIT(T);RecoveryBranchTrace<T> trace;int n=(nlev+1)*ncells;T dcf1=*dcf1_in,dcf2=*dcf2_in,dcf3=*dcf3_in; \
 for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){int l=i/ncells,c=i%ncells;if(l==nlev){dw[i]=T(0);continue;}TVals<T> v; \
  if(!tw_values<T>(a,l,c,v)){tw_bad(fail,i,1,1);dw[i]=T(0);continue;} \
  tw_record<T>(trace,a,l,c); T dv=T(0),dd=T(0); \
  if(l==0){T cf[3]={cf1,cf2,cf3},dcf[3]={dcf1,dcf2,dcf3}; \
   for(int k=0;k<3;++k)dd=tw_add<T>(dd,tw_add<T>(tw_mul<T>(dcf[k],rho[k*ncells+c]),tw_mul<T>(cf[k],drho[k*ncells+c]))); \
  }else{int j=i-ncells; \
   T dm=tw_add<T>(tw_add<T>(tw_mul<T>(dfzm[l],zz[i]),tw_mul<T>(fzm[l],dzz[i])),tw_add<T>(tw_mul<T>(dfzp[l],zz[j]),tw_mul<T>(fzp[l],dzz[j]))); \
   dd=tw_add<T>(tw_add<T>(tw_mul<T>(dfzm[l],rho[i]),tw_mul<T>(fzm[l],drho[i])),tw_add<T>(tw_mul<T>(dfzp[l],rho[j]),tw_mul<T>(fzp[l],drho[j]))); \
   dv=tw_div<T>(tw_sub<T>(drw[i],tw_mul<T>(v.first,dm)),v.metric); \
  } \
  for(int s=0;s<neoc[c];++s){int e=eoc[c*max_edges+s],mi=i*max_edges+s,si=c*max_edges+s;T f=tw_flux<T>(a,l,e),df=T(0); \
   if(l==0){T cf[3]={cf1,cf2,cf3},dcf[3]={dcf1,dcf2,dcf3};for(int k=0;k<3;++k)df=tw_add<T>(df,tw_add<T>(tw_mul<T>(dcf[k],ru[k*nedges+e]),tw_mul<T>(cf[k],dru[k*nedges+e])));} \
   else df=tw_add<T>(tw_add<T>(tw_mul<T>(dfzm[l],ru[l*nedges+e]),tw_mul<T>(fzm[l],dru[l*nedges+e])),tw_add<T>(tw_mul<T>(dfzp[l],ru[(l-1)*nedges+e]),tw_mul<T>(fzp[l],dru[(l-1)*nedges+e]))); \
   T sign=tw_sign<T>(f),g=tw_add<T>(zb[mi],tw_mul<T>(sign,zb3[mi])),dg=tw_add<T>(dzb[mi],tw_mul<T>(sign,dzb3[mi])); \
   dv=tw_add<T>(dv,tw_add<T>(tw_mul<T>(tw_mul<T>(dsgn[si],g),f),tw_mul<T>(sgn[si],tw_add<T>(tw_mul<T>(dg,f),tw_mul<T>(g,df))))); \
  } \
  T dy=tw_div<T>(tw_sub<T>(dv,tw_mul<T>(v.w,dd)),v.density);if(!isfinite(dy)){tw_bad(fail,i,12,2);dw[i]=T(0);}else dw[i]=dy; \
 } trace.publish(branch_record,sizeof(T)==4?17:6); \
} \
extern "C" __global__ void recover_terrain_w_##SUFFIX##__ad( TERRAIN_ARGS(T),const int *coe, \
 T *seed_bw,T *brho,T *bru,T *brw,T *bzz,T *bfzm,T *bfzp,T *bsgn,T *bzb,T *bzb3,T *bcf1,T *bcf2,T *bcf3,int *fail,unsigned int *branch_record,const T *bw) { \
 TERRAIN_INIT(T);RecoveryBranchTrace<T> trace;int n=(nlev+1)*ncells, nc=nlev*ncells,ne=nlev*nedges,ns=ncells*max_edges,nm=nlev*ns; \
 for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){int l=i/ncells,c=i%ncells;if(l==nlev)continue; \
  TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,l,c,v,bv,bd,bm,bri)){tw_bad(fail,i,1,3);continue;} \
  tw_record<T>(trace,a,l,c); if(l>0){T result=tw_add<T>(brw[i],bri);if(!isfinite(bri)||!isfinite(brw[i])||!isfinite(result)){tw_bad(fail,i,3,4);continue;}brw[i]=result;} \
 } trace.publish(branch_record,sizeof(T)==4?17:6); \
 for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<nc;i+=gridDim.x*blockDim.x){int k=i/ncells,c=i%ncells;double sr=0.0,sz=0.0; \
  if(k<3){TVals<T> v;T bv,bd,bm,bri;if(tw_contrib<T>(a,bw,0,c,v,bv,bd,bm,bri)){T cf[3]={cf1,cf2,cf3};sr+=recovery_wide<T>(tw_mul<T>(bd,cf[k]));}} \
  for(int t=0;t<2;++t){int l=k+t;if(l<=0||l>=nlev)continue;TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,l,c,v,bv,bd,bm,bri))continue;T f=t==0?fzm[l]:fzp[l];sr+=recovery_wide<T>(tw_mul<T>(bd,f));sz+=recovery_wide<T>(tw_mul<T>(bm,f));} \
  T rr=recovery_narrow<T>(recovery_wide<T>(brho[i])+sr),rz=recovery_narrow<T>(recovery_wide<T>(bzz[i])+sz); \
  if(!isfinite(sr)||!isfinite(sz)||!isfinite(brho[i])||!isfinite(bzz[i])||!isfinite(rr)||!isfinite(rz)){tw_bad(fail,n+i,1,5);continue;}brho[i]=rr;bzz[i]=rz; \
 } \
 for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<ne;i+=gridDim.x*blockDim.x){int k=i/nedges,e=i%nedges;double sum=0.0; \
  int c0=coe[2*e],c1=coe[2*e+1];for(int side=0;side<2;++side){int c=side==0?c0:c1;if(side==1&&c==c0)continue;if(c<0||c>=ncells){tw_bad(fail,n+nc+i,7,6);continue;} \
   int count=neoc[c];if(count<0||count>max_edges){tw_bad(fail,n+nc+i,8,7);continue;}int slot=-1;for(int s=0;s<count;++s)if(eoc[c*max_edges+s]==e){slot=s;break;} \
   if(slot<0){tw_bad(fail,n+nc+i,7,8);continue;} \
   if(k<3){TVals<T> v;T bv,bd,bm,bri;if(tw_contrib<T>(a,bw,0,c,v,bv,bd,bm,bri)){T cf[3]={cf1,cf2,cf3};sum+=recovery_wide<T>(tw_mul<T>(tw_bflux<T>(a,0,c,slot,bv),cf[k]));}} \
   for(int t=0;t<2;++t){int l=k+t;if(l<=0||l>=nlev)continue;TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,l,c,v,bv,bd,bm,bri))continue;T f=t==0?fzm[l]:fzp[l];sum+=recovery_wide<T>(tw_mul<T>(tw_bflux<T>(a,l,c,slot,bv),f));} \
  } \
  T result=recovery_narrow<T>(recovery_wide<T>(bru[i])+sum);if(!isfinite(sum)||!isfinite(bru[i])||!isfinite(result)){tw_bad(fail,n+nc+i,2,9);continue;}bru[i]=result; \
 } \
 for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<nm;i+=gridDim.x*blockDim.x){int l=i/ns,c=(i%ns)/max_edges,s=i%max_edges;if(s>=neoc[c])continue;TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,l,c,v,bv,bd,bm,bri))continue; \
  int e=eoc[c*max_edges+s];T f=tw_flux<T>(a,l,e),zbar=tw_mul<T>(tw_mul<T>(bv,sgn[c*max_edges+s]),f);T rz=tw_add<T>(bzb[i],zbar),rz3=tw_add<T>(bzb3[i],tw_mul<T>(zbar,tw_sign<T>(f))); \
  if(!isfinite(zbar)||!isfinite(bzb[i])||!isfinite(bzb3[i])||!isfinite(rz)||!isfinite(rz3)){tw_bad(fail,n+nc+ne+i,10,10);continue;}bzb[i]=rz;bzb3[i]=rz3; \
 } \
 for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<ns;i+=gridDim.x*blockDim.x){int c=i/max_edges,s=i%max_edges;if(s>=neoc[c])continue;double sum=0.0; \
  for(int l=0;l<nlev;++l){TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,l,c,v,bv,bd,bm,bri))continue;int e=eoc[c*max_edges+s],mi=(l*ncells+c)*max_edges+s;T f=tw_flux<T>(a,l,e),g=tw_add<T>(zb[mi],tw_mul<T>(tw_sign<T>(f),zb3[mi]));sum+=recovery_wide<T>(tw_mul<T>(tw_mul<T>(bv,g),f));}T result=recovery_narrow<T>(recovery_wide<T>(bsgn[i])+sum); \
  if(!isfinite(sum)||!isfinite(bsgn[i])||!isfinite(result)){tw_bad(fail,n+nc+ne+nm+i,9,11);continue;}bsgn[i]=result; \
 } \
 for(int l=blockDim.x*blockIdx.x+threadIdx.x;l<nlev;l+=gridDim.x*blockDim.x){if(l==0)continue;double sm=0.0,sp=0.0; \
  for(int c=0;c<ncells;++c){TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,l,c,v,bv,bd,bm,bri))continue;int i=l*ncells+c,j=i-ncells;sm+=recovery_wide<T>(tw_mul<T>(bd,rho[i]))+recovery_wide<T>(tw_mul<T>(bm,zz[i]));sp+=recovery_wide<T>(tw_mul<T>(bd,rho[j]))+recovery_wide<T>(tw_mul<T>(bm,zz[j])); \
   for(int s=0;s<neoc[c];++s){int e=eoc[c*max_edges+s];T bf=tw_bflux<T>(a,l,c,s,bv);sm+=recovery_wide<T>(tw_mul<T>(bf,ru[l*nedges+e]));sp+=recovery_wide<T>(tw_mul<T>(bf,ru[(l-1)*nedges+e]));} \
  }T rm=recovery_narrow<T>(recovery_wide<T>(bfzm[l])+sm),rp=recovery_narrow<T>(recovery_wide<T>(bfzp[l])+sp); \
  if(!isfinite(sm)||!isfinite(sp)||!isfinite(bfzm[l])||!isfinite(bfzp[l])||!isfinite(rm)||!isfinite(rp)){tw_bad(fail,n+nc+ne+nm+ns+l,5,12);continue;}bfzm[l]=rm;bfzp[l]=rp; \
 } \
 for(int k=blockDim.x*blockIdx.x+threadIdx.x;k<3;k+=gridDim.x*blockDim.x){double sum=0.0; \
  for(int c=0;c<ncells;++c){TVals<T> v;T bv,bd,bm,bri;if(!tw_contrib<T>(a,bw,0,c,v,bv,bd,bm,bri))continue;sum+=recovery_wide<T>(tw_mul<T>(bd,rho[k*ncells+c]));for(int s=0;s<neoc[c];++s){int e=eoc[c*max_edges+s];sum+=recovery_wide<T>(tw_mul<T>(tw_bflux<T>(a,0,c,s,bv),ru[k*nedges+e]));}}T *dest=k==0?bcf1:(k==1?bcf2:bcf3);T result=recovery_narrow<T>(recovery_wide<T>(*dest)+sum); \
  if(!isfinite(sum)||!isfinite(*dest)||!isfinite(result)){tw_bad(fail,n+nc+ne+nm+ns+nlev+k,17+k,13);continue;}*dest=result; \
 } for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<(nlev+1)*ncells;i+=gridDim.x*blockDim.x){seed_bw[i]=T(0);} \
}
DEFINE_TERRAIN(float,f32)
DEFINE_TERRAIN(double,f64)
