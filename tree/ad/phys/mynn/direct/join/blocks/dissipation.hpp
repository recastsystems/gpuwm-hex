#pragma once
#include "ess_scalar.hpp"
namespace direct {
// Input qke after prediction, frozen el, pressure. Top and disabled heating are constant zero.
constexpr int QNI=3*MAX,QNO=MAX;
template<class P>void dissipation_forward(int n,int enabled,const P*x,P*y){for(int k=0;k<MAX;++k)y[k]=0;if(enabled)for(int k=1;k<n;++k){P a[4]={x[k],x[MAX+k],x[MAX+k+1],x[2*MAX+k]};dissipation_level<P,false>(a,y+k,nullptr,nullptr);}}
template<class P>void dissipation_reverse(int n,int enabled,const P*x,const double*s,double*b,int planted=0){if(enabled)for(int k=n-1;k>=1;--k){P a[4]={x[k],x[MAX+k],x[MAX+k+1],x[2*MAX+k]};double ab[4]{};dissipation_level<P,true>(a,nullptr,s+k,ab,planted&&k==n/2);b[k]+=ab[0];b[2*MAX+k]+=ab[3];}}
}
