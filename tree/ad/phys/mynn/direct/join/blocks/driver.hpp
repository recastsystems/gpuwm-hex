#pragma once
#include "ess_scalar.hpp"
namespace direct {
// Input dz,sqv,sqc,sqi,sqs,th,ex,rho; scalar tail ts,qfx,hfx,ust,xland.
// Output zw,qv,qc,qi,qs,sqw,thl,thv; tail flqv,flq,th_sfc,flt,fltv,rmol,rmolh,pmz,phh.
// zw has n+1 entries. exnerg is dead; f_get_pblh/f_scale_aware outputs are frozen outside this cut.
constexpr int DNI=8*MAX+5,DNO=8*MAX+9;
template<class P>void driver_forward(int n,const P*x,P*y){
 for(int i=0;i<DNO;++i)y[i]=0;
 for(int k=1;k<=n;++k){y[k+1]=r6_add(y[k],x[k]);P a[6],o[7];for(int j=0;j<6;++j)a[j]=x[(j+1)*MAX+k];driver_thermo<P,false>(a,o,nullptr,nullptr);for(int j=0;j<7;++j)y[(j+1)*MAX+k]=o[j];}
 P a[9]={x[MAX+1],x[7*MAX+1],x[6*MAX+1],x[8*MAX],x[8*MAX+1],x[8*MAX+2],x[8*MAX+3],x[1],x[8*MAX+4]};driver_surface<P,false>(a,y+8*MAX,nullptr,nullptr);
}
template<class P>void driver_reverse(int n,const P*x,const double*s,double*b,int planted=0){
 double zb=0;for(int k=n;k>=1;--k){zb+=s[k+1];b[k]+=zb;P a[6];double z[7],ab[6]{};for(int j=0;j<6;++j)a[j]=x[(j+1)*MAX+k];for(int j=0;j<7;++j)z[j]=s[(j+1)*MAX+k];driver_thermo<P,true>(a,nullptr,z,ab,planted&&k==n/2);for(int j=0;j<6;++j)b[(j+1)*MAX+k]+=ab[j];}
 P a[9]={x[MAX+1],x[7*MAX+1],x[6*MAX+1],x[8*MAX],x[8*MAX+1],x[8*MAX+2],x[8*MAX+3],x[1],x[8*MAX+4]};double ab[9]{};driver_surface<P,true>(a,nullptr,s+8*MAX,ab);const int at[9]={MAX+1,7*MAX+1,6*MAX+1,8*MAX,8*MAX+1,8*MAX+2,8*MAX+3,1,8*MAX+4};for(int j=0;j<9;++j)b[at[j]]+=ab[j];
}
}
