// SPDX-License-Identifier: Apache-2.0
// Replay forward primal branch predicates; integer counters are deterministic.
#pragma once
#include "hex_mpas_branch_ad.cuh"
// Each arithmetic call site has its own predicate count. A loop executes a
// site once for each valid kite; integer atomics are order independent.
__device__ __forceinline__ void rrfs_pvc_record(int i,const float *p,const int *voc,const int *count,const int *cov,const float *area,const float *ia,int nc,int nv,int me,int vd,unsigned *r){
 int c=i%nc,k=i/nc;float value=0.f;
 for(int j=0;j<count[c];j++){
  int v=voc[c*me+j],z=-1;for(int t=0;t<vd;t++)if(cov[v*vd+t]==c){z=t;break;}
  if(z>=0){
   bool h0=false,h1=false,h2=false;
   float x=ad_trace_mul(area[v*vd+z],p[k*nv+v],h0);ad_count_branch(r,2,h0);
   x=ad_trace_mul(x,ia[c],h1);ad_count_branch(r,3,h1);
   value=ad_trace_add(value,x,h2);ad_count_branch(r,4,h2);
  }
 }
 (void)value;
}
__device__ __forceinline__ void rrfs_apvm_record(int i,const float *u,const float *v,const float *pv,const float *pc,const int *voe,const int *coe,const float *idc,const float *idv,float scale,int nc,int ne,int nv,const float *pe,int alias,unsigned *r){
 int e=i%ne,k=i/ne,c0=coe[2*e],c1=coe[2*e+1],v0=voe[2*e],v1=voe[2*e+1];
 ad_count_branch(r,0,alias!=0);
 float gt=ad_site_sub(pv[k*nv+v1],pv[k*nv+v0],r,1);gt=ad_site_mul(gt,idv[e],r,2);
 float gn=ad_site_sub(pc[k*nc+c1],pc[k*nc+c0],r,3);gn=ad_site_mul(gn,idc[e],r,4);
 float a=ad_site_mul(v[i],gt,r,5),b=ad_site_mul(u[i],gn,r,6);
 float prod=ad_site_add(a,b,r,7),scaled=ad_site_mul(scale,prod,r,8);
 float y=ad_site_sub(pe[i],scaled,r,9);(void)y;
}
__device__ __forceinline__ void rrfs_w2_record(int i,const float *w,const float *rho,const float *kd,const float *del2,const int *eoc,const int *cnt,const int *coe,const float *idc,const float *dv,const float *ia,int nc,int ne,int me,unsigned *r){
 int c=i%nc,k=i/nc;ad_count_branch(r,0,k==0);if(k==0)return;float lap=0.f,tend=0.f;
 for(int j=0;j<cnt[c];j++){
  int e=eoc[c*me+j],a=coe[2*e],b=coe[2*e+1];ad_count_branch(r,1,a==c);
  float f=ad_site_mul(.5f,ia[c],r,2);f=ad_site_mul(f,a==c?1.f:-1.f,r,3);f=ad_site_mul(f,dv[e],r,4);f=ad_site_mul(f,idc[e],r,5);
  float rs=ad_site_add(rho[k*ne+e],rho[(k-1)*ne+e],r,6),jump=ad_site_sub(w[k*nc+b],w[k*nc+a],r,7);
  float flux=ad_site_mul(ad_site_mul(f,rs,r,8),jump,r,9);lap=ad_site_add(lap,flux,r,10);
  float sk=ad_site_add(kd[k*nc+a],kd[k*nc+b],r,11);sk=ad_site_add(sk,kd[(k-1)*nc+a],r,12);sk=ad_site_add(sk,kd[(k-1)*nc+b],r,13);
  float z=ad_site_mul(ad_site_mul(ad_site_mul(flux,del2[e],r,14),.25f,r,15),sk,r,16);tend=ad_site_add(tend,z,r,17);
 }
 (void)lap;(void)tend;
}
__device__ __forceinline__ void rrfs_w4_record(int q,const float *w,const float *del4,const int *eoc,const int *cnt,const int *coe,const float *idc,const float *dv,const float *ia,float h4,int nc,int me,const float *tend,unsigned *r){
 int c=q%nc,k=1+q/nc;float val=tend[k*nc+c];ad_count_branch(r,0,h4>0.f);ad_count_branch(r,1,k>0);
 for(int j=0;j<cnt[c];j++){
  int e=eoc[c*me+j],a=coe[2*e],b=coe[2*e+1];ad_count_branch(r,2,a==c);
  float f=ad_site_mul(h4,ia[c],r,3);f=ad_site_mul(del4[e],f,r,4);f=ad_site_mul(f,dv[e],r,5);f=ad_site_mul(f,a==c?1.f:-1.f,r,6);f=ad_site_mul(f,idc[e],r,7);
  float jump=ad_site_sub(w[k*nc+b],w[k*nc+a],r,8);val=ad_site_sub(val,ad_site_mul(f,jump,r,9),r,10);
 }(void)val;
}
__device__ __forceinline__ void rrfs_mom4_record(int i,const float *rho,const float *div,const float *vort,const float *del4,const int *coe,const int *voe,const float *idc,const float *idv,float h4,float df,int nc,int ne,int nv,const float *tend,unsigned *r){
 int e=i%ne,k=i/ne,a=coe[2*e],b=coe[2*e+1],v0=voe[2*e],v1=voe[2*e+1];float idve=fminf(idv[e],4.f*idc[e]);ad_count_branch(r,0,h4>0.f);ad_count_branch(r,1,idv[e]>=4.f*idc[e]);
 float scale=ad_site_mul(del4[e],h4,r,2),dc=ad_site_mul(ad_site_mul(scale,df,r,3),idc[e],r,4),dv=ad_site_mul(scale,idve,r,5);
 float x=ad_site_mul(ad_site_sub(div[k*nc+b],div[k*nc+a],r,6),dc,r,7),y=ad_site_mul(ad_site_sub(vort[k*nv+v1],vort[k*nv+v0],r,8),dv,r,9);
 float d=ad_site_sub(x,y,r,10),z=ad_site_sub(tend[i],ad_site_mul(rho[i],d,r,11),r,12);(void)z;
}
__device__ __forceinline__ void rrfs_seed_record(int i,const float *rho,const float *div,const float *vort,const float *kd,const float *del2,const int *coe,const int *voe,const float *idc,const float *idv,int nc,int ne,int nv,const float *tend,unsigned *r){
 int e=i%ne,k=i/ne,a=coe[2*e],b=coe[2*e+1],v0=voe[2*e],v1=voe[2*e+1];float cap=ad_site_mul(4.f,idc[e],r,1),iv=ad_site_min(idv[e],cap,r,2);ad_count_branch(r,0,mpas_f32_to_f64_ieee(cap)<mpas_f32_to_f64_ieee(idv[e]));
 float x=ad_site_mul(ad_site_sub(div[k*nc+b],div[k*nc+a],r,3),idc[e],r,4),y=ad_site_mul(ad_site_sub(vort[k*nv+v1],vort[k*nv+v0],r,5),iv,r,6),lap=ad_site_sub(x,y,r,7);
 float ke=ad_site_mul(.5f,ad_site_add(kd[k*nc+a],kd[k*nc+b],r,8),r,9),z=ad_site_mul(ad_site_mul(ad_site_mul(rho[i],ke,r,10),lap,r,11),del2[e],r,12);
 z=ad_site_add(tend[i],z,r,13);(void)z;
}
