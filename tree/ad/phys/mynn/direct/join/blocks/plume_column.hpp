#pragma once
#include "plume.hpp"
// Cloud tail F90:7543--7713, fixed owner order; n<=59, ktop is zero-based.
// Per-level input: area,plth,plqt,plqc,w,dz,tk,p,qt,ex,thl,cldfra,qcbl,vt,vq.
// Last input is landsea. Per-level outputs: cldfra,qcbl,vt,vq.
namespace direct { namespace plume {
template<class P> inline void cloud_reverse(int n,int ktop,const P*x,const double*s,double*b,P*y=nullptr,bool planted=false){
 P qi[60]{},mf=0;double qb[60]{},mb=0;int base=-1;
 qi[0]=x[8];
 for(int k=0;k<n-1;++k){const P*a=x+15*k,*z=a+15;qi[k+1]=P(P(P(a[8]*z[5])+P(z[8]*a[5]))/P(z[5]+a[5]));}
 for(int k=1;k<=n-3 && k<=ktop;++k)if(x[15*k+3]>P(0)){base=k;mf=P(x[15*k]*x[15*k+4]);break;}
 for(int k=n-1;k>=0;--k){const P*a=x+15*k;
  if(k<1 || k>n-3 || k>ktop || !(a[3]>P(0))){for(int j=0;j<4;++j){b[15*k+11+j]+=s[4*k+j];if(y)y[4*k+j]=a[11+j];}continue;}
  P v[19]={a[0],a[1],a[2],a[3],a[4],qi[k],qi[k+1],a[5],a[6],a[7],a[8],a[9],a[10],x[15*n],mf,a[11],a[12],a[13],a[14]};double vb[19]{};
  blend_reverse(v,s+4*k,vb,y?y+4*k:nullptr,planted);
  for(int j=0;j<5;++j)b[15*k+j]+=vb[j];for(int j=5;j<11;++j)b[15*k+j]+=vb[j+2];for(int j=11;j<15;++j)b[15*k+j]+=vb[j+4];
  qb[k]+=vb[5];qb[k+1]+=vb[6];mb+=vb[14];b[15*n]+=vb[13];
 }
 if(base>=0){acc(b[15*base],double(x[15*base+4]),mb);acc(b[15*base+4],double(x[15*base]),mb);}
 for(int k=n-2;k>=0;--k){P v[4]={x[15*k+8],x[15*(k+1)+8],x[15*k+5],x[15*(k+1)+5]};double vb[4]{};interp_reverse(v,qb+k+1,vb);b[15*k+8]+=vb[0];b[15*(k+1)+8]+=vb[1];b[15*k+5]+=vb[2];b[15*(k+1)+5]+=vb[3];}
 b[8]+=qb[0];
}
}}
