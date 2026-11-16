#pragma once
#include "ess_scalar.hpp"
namespace direct {
// Profile order dz,zw,u,v,thl,thlv,th,p,exner,qke,cldfra,edmf_a,edmf_w,qw,thv,vt,vq;
// scalar tail pblh,xland. Held pblh/edmf_*; thv/vt/vq unused for ESS1, thvp=0.
constexpr int ENI=17*MAX+2,ENO=5*MAX;
template<class P>void ess_forward(int n,const P*x,P*y){
 P eth[MAX]{},cl,ci[3]={x[2*MAX+1],x[3*MAX+1],x[17*MAX+1]};
 ess_clam<P,false>(ci,&cl,nullptr,nullptr);
 for(int k=1;k<=n;++k){P a[3]={x[6*MAX+k],x[8*MAX+k],x[7*MAX+k]};ess_thetae<P,false>(a,&eth[k],nullptr,nullptr);}
 for(int i=0;i<ENO;++i)y[i]=0;
 for(int k=2;k<=n;++k){P a[25]={x[k],x[k-1],x[2*MAX+k],x[2*MAX+k-1],x[3*MAX+k],x[3*MAX+k-1],x[4*MAX+k],x[4*MAX+k-1],x[13*MAX+k],x[13*MAX+k-1],x[5*MAX+k],x[5*MAX+k-1],eth[k],eth[k-1],x[9*MAX+k],x[9*MAX+k-1],x[10*MAX+k],x[10*MAX+k-1],x[MAX+k],cl,x[17*MAX],x[11*MAX+k],x[11*MAX+k-1],x[12*MAX+k],x[12*MAX+k-1]},o[5];ess_edge<P,false>(a,o,nullptr,nullptr);for(int j=0;j<5;++j)y[j*MAX+k]=o[j];}
}
template<class P>void ess_reverse(int n,const P*x,const double*s,double*b,int planted=0){
 P eth[MAX]{},cl,ci[3]={x[2*MAX+1],x[3*MAX+1],x[17*MAX+1]};double cb=0;
 ess_clam<P,false>(ci,&cl,nullptr,nullptr);
 for(int k=1;k<=n;++k){P a[3]={x[6*MAX+k],x[8*MAX+k],x[7*MAX+k]};ess_thetae<P,false>(a,&eth[k],nullptr,nullptr);}
 // Fixed descending edge owner order; no atomics. Endpoint seeds at k=1 are constant outputs.
 for(int k=n;k>=2;--k){P a[25]={x[k],x[k-1],x[2*MAX+k],x[2*MAX+k-1],x[3*MAX+k],x[3*MAX+k-1],x[4*MAX+k],x[4*MAX+k-1],x[13*MAX+k],x[13*MAX+k-1],x[5*MAX+k],x[5*MAX+k-1],eth[k],eth[k-1],x[9*MAX+k],x[9*MAX+k-1],x[10*MAX+k],x[10*MAX+k-1],x[MAX+k],cl,x[17*MAX],x[11*MAX+k],x[11*MAX+k-1],x[12*MAX+k],x[12*MAX+k-1]};double z[5],ab[25]{};for(int j=0;j<5;++j)z[j]=s[j*MAX+k];ess_edge<P,true>(a,nullptr,z,ab,planted&&k==n/2);
 const int rows[9]={0,2,3,4,13,5,-1,9,10};for(int j=0;j<9;++j){if(rows[j]<0){for(int side=1;side>=0;--side){int l=k-side;P ta[3]={x[6*MAX+l],x[8*MAX+l],x[7*MAX+l]};double tb[3]={b[6*MAX+l],b[8*MAX+l],b[7*MAX+l]};ess_thetae<P,true>(ta,nullptr,&ab[12+side],tb);b[6*MAX+l]=tb[0];b[8*MAX+l]=tb[1];b[7*MAX+l]=tb[2];}}else{b[rows[j]*MAX+k]+=ab[2*j];b[rows[j]*MAX+k-1]+=ab[2*j+1];}}b[MAX+k]+=ab[18];cb+=ab[19];}
 double cib[3]{};ess_clam<P,true>(ci,nullptr,&cb,cib);b[2*MAX+1]+=cib[0];b[3*MAX+1]+=cib[1];
}
}
