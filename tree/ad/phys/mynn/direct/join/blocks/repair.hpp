#pragma once
#include "numeric.hpp"
namespace direct {
// Profile order dp,exner,qv,qc,qi,qs,th,dqv,dqc,dqi,dqs,dth; all indices 1..n.
// Fixed block state: 7 profiles, 4 scalars and one Boolean. Input and output must not alias state.
template<class P> struct RepairState {P cv[MAX],ci[MAX],cs[MAX],v1[MAX],dv[MAX],vp[MAX],donor[MAX];P sum,aa,num,den;bool redistribute;};
template<class P>void repair_forward(int n,P dt,const P*in,P*out,RepairState<P>&s){
 auto X=[&](int f,int k)->P&{return out[f*MAX+k];};
 for(int i=0;i<12*MAX;++i)out[i]=in[i];
 constexpr P zero=P(0),qmin=P(1e-20f),lv=P(2.5e6f/1004.5f),ls=P(2.85e6f/1004.5f);
 for(int k=n;k>=1;--k){
  P c=mx(r6_sub(zero,X(3,k)),zero),i=mx(r6_sub(zero,X(4,k)),zero),z=mx(r6_sub(zero,X(5,k)),zero);
  s.cv[k]=c;s.ci[k]=i;s.cs[k]=z;
  X(8,k)=r6_add(X(8,k),r6_div(c,dt));X(9,k)=r6_add(X(9,k),r6_div(i,dt));X(10,k)=r6_add(X(10,k),r6_div(z,dt));
  X(7,k)=r6_sub(X(7,k),r6_div(r6_add(r6_add(c,i),z),dt));
  X(11,k)=r6_add(r6_add(X(11,k),r6_mul(r6_div(lv,X(1,k)),r6_div(c,dt))),r6_mul(r6_div(ls,X(1,k)),r6_div(r6_add(i,z),dt)));
  X(3,k)=r6_add(X(3,k),c);X(4,k)=r6_add(X(4,k),i);X(5,k)=r6_add(X(5,k),z);
  X(2,k)=r6_sub(r6_sub(r6_sub(X(2,k),c),i),z);s.v1[k]=X(2,k);
  X(6,k)=r6_add(r6_add(X(6,k),r6_mul(r6_div(lv,X(1,k)),c)),r6_mul(r6_div(ls,X(1,k)),r6_add(i,z)));
  P d=mx(r6_sub(qmin,X(2,k)),zero);s.dv[k]=d;
  X(7,k)=r6_add(X(7,k),r6_div(d,dt));X(2,k)=r6_add(X(2,k),d);s.vp[k]=X(2,k);
  if(k!=1){X(2,k-1)=r6_sub(X(2,k-1),r6_div(r6_mul(d,X(0,k)),X(0,k-1)));X(7,k-1)=r6_sub(X(7,k-1),r6_div(r6_div(r6_mul(d,X(0,k)),X(0,k-1)),dt));}
  X(2,k)=mx(X(2,k),qmin);X(3,k)=mx(zero,X(3,k));X(4,k)=mx(zero,X(4,k));X(5,k)=mx(zero,X(5,k));
 }
 s.redistribute=false;s.sum=zero;s.aa=zero;s.num=zero;s.den=qmin;
 for(int k=1;k<=n;++k)s.donor[k]=X(2,k);
 if(s.dv[1]>qmin){
  for(int k=1;k<=n;++k)if(X(2,k)>r6_mul(P(2),qmin))s.sum=r6_add(s.sum,r6_mul(X(2,k),X(0,k)));
  s.num=r6_mul(s.dv[1],X(0,1));s.den=mx(s.sum,qmin);s.aa=r6_div(s.num,s.den);
  if(s.aa<P(.5)){s.redistribute=true;for(int k=1;k<=n;++k)if(X(2,k)>r6_mul(P(2),qmin)){P d=r6_mul(s.aa,X(2,k));X(2,k)=r6_sub(X(2,k),d);X(7,k)=r6_sub(X(7,k),r6_div(d,dt));}}
 }
}
// Native quotient local rule uses unrounded double a/b in the denominator weight.
template<class P>inline void quotient_reverse(P a,P b,double g,double&ab,double&bb){double t=g*(1.0/double(b));ab+=t;acc(bb,double(a)/double(b),-t);}
template<class P>void repair_reverse(int n,P dt,const P*in,const RepairState<P>&s,const double*seed,double*bar,int planted=0,bool compact=false){
 double work[12*MAX];for(int j=0;j<12*MAX;++j)work[j]=seed[j];
 auto X=[&](int f,int k){return in[(compact&&f>=3?f-1:f)*MAX+k];};auto B=[&](int f,int k)->double&{return work[f*MAX+k];};
 constexpr P qmin=P(1e-20f),lv=P(2.5e6f/1004.5f),ls=P(2.85e6f/1004.5f);
 double bottom=0;
 if(s.redistribute){
  double aa_bar=0;
  for(int k=n;k>=1;--k)if(s.donor[k]>r6_mul(P(2),qmin)){
   double db=-B(2,k)-B(7,k)/double(dt);
   acc(aa_bar,double(s.donor[k]),db);acc(B(2,k),double(s.aa),db);
  }
  double sb=0,nb=0;quotient_reverse(s.num,s.den,aa_bar,nb,sb);
  acc(bottom,double(X(0,1)),nb);acc(B(0,1),double(s.dv[1]),nb);
  if(!(s.sum>qmin))sb=0;
  for(int k=n;k>=1;--k)if(s.donor[k]>r6_mul(P(2),qmin)){acc(B(2,k),double(X(0,k)),sb);acc(B(0,k),double(s.donor[k]),sb);}
 }
 for(int k=1;k<=n;++k){
  // Final max: qv chooses constant on equality; qc/qi/qs choose their second (state) operand.
  if(!(s.vp[k]>qmin))B(2,k)=0;
  if(P(0)>r6_add(X(3,k),s.cv[k]))B(3,k)=0;
  if(P(0)>r6_add(X(4,k),s.ci[k]))B(4,k)=0;
  if(P(0)>r6_add(X(5,k),s.cs[k]))B(5,k)=0;
  double dvb=k==1?bottom:0;
  if(k!=1){
   // The source evaluates the same ratio twice; combine only its double cotangents.
   double ratio_bar=-B(2,k-1)-B(7,k-1)/double(dt),numbar=0;
   P num=r6_mul(s.dv[k],X(0,k));quotient_reverse(num,X(0,k-1),ratio_bar,numbar,B(0,k-1));
   acc(dvb,double(X(0,k)),numbar);acc(B(0,k),double(s.dv[k]),numbar);
  }
  dvb+=B(2,k)+B(7,k)/double(dt);
  if(r6_sub(qmin,s.v1[k])>P(0))B(2,k)-=dvb;
  double cb=B(3,k)-B(2,k),ib=B(4,k)-B(2,k),zb=B(5,k)-B(2,k);
  const P l=r6_div(lv,X(1,k)),h=r6_div(ls,X(1,k));
  double lb=0,hb=0;
  acc(cb,double(l),B(6,k));acc(ib,double(h),B(6,k));acc(zb,double(h),B(6,k));
  acc(lb,double(s.cv[k]),B(6,k));acc(hb,double(r6_add(s.ci[k],s.cs[k])),B(6,k));
  // dth update uses the rounded divisions cv/dt and (ci+cs)/dt.
  acc(lb,double(r6_div(s.cv[k],dt)),B(11,k));acc(hb,double(r6_div(r6_add(s.ci[k],s.cs[k]),dt)),B(11,k));
  cb+=double(l)*B(11,k)/double(dt);ib+=double(h)*B(11,k)/double(dt);zb+=double(h)*B(11,k)/double(dt);
  double discard=0;quotient_reverse(lv,X(1,k),lb,discard,B(1,k));discard=0;quotient_reverse(ls,X(1,k),hb,discard,B(1,k));
  cb+=(B(8,k)-B(7,k))/double(dt);ib+=(B(9,k)-B(7,k))/double(dt);zb+=(B(10,k)-B(7,k))/double(dt);
  if(planted&&k==n/2)cb*=1.001;
  if(r6_sub(P(0),X(3,k))>P(0))B(3,k)-=cb;
  if(r6_sub(P(0),X(4,k))>P(0))B(4,k)-=ib;
  if(r6_sub(P(0),X(5,k))>P(0))B(5,k)-=zb;
 }
 for(int j=0;j<12*MAX;++j)bar[j]+=work[j];
}
}
