#pragma once
#include "heat.hpp"
namespace direct { namespace solves {
// One cut, one serial owner. j=0 U,1 V,2 NC,3 NI,4 NWFA,5 NIFA.
// All primals P (production float), all cotangents double. 2<=n<=59.
// Input layout uses stride MAX; PS,UST,WSPD,OCE,PBLH,LAND follow profiles.
enum Field {DZ,RHO,DF,SA,SD,Q,FU,FD,SUB,DET,COLD,CNEW,ZW,TK,QV,NF};
enum Scalar {PS=NF*MAX,UST,WSPD,OCE,PBLH,LAND,NI};
constexpr int NO=MAX;
template<class P>struct State {P t[MAX],r[MAX],rz[MAX],h[MAX],q[MAX],cp[MAX],dp[MAX],x[MAX];P rs,uw;};
template<class P>void assembly(int n,int j,P dt,const P*in,const State<P>&s,P*a,P*b,P*c,P*d){
 auto X=[&](int f,int k){return in[f*MAX+k];};bool mom=j<2;
 for(int k=1;k<n;++k){P t=s.t[k],r=s.r[k],h=r6_mul(r6_mul(P(.5),t),r),tr=r6_mul(t,r);
  a[k]=r6_mul(r6_neg(r6_mul(t,s.h[k])),r);
  P sum=k==1?(mom?r6_add(s.h[2],(j==0?r6_mul(r6_mul(s.rs,in[UST]),s.uw):r6_div(r6_mul(s.rs,mynn_ipow<2>(in[UST])),in[WSPD]))):r6_add(s.h[2],s.h[1])):r6_add(s.h[k],s.h[k+1]);
  b[k]=r6_add(P(1),r6_mul(r6_mul(t,sum),r));c[k]=r6_mul(r6_neg(r6_mul(t,s.h[k+1])),r);
  if(k==1){b[k]=r6_sub(b[k],r6_mul(h,X(SA,k+1)));b[k]=r6_sub(b[k],r6_mul(h,X(SD,k+1)));}
  else {a[k]=r6_add(a[k],r6_mul(h,X(SA,k)));a[k]=r6_add(a[k],r6_mul(h,X(SD,k)));b[k]=r6_add(b[k],r6_mul(h,r6_sub(X(SA,k),X(SA,k+1))));b[k]=r6_add(b[k],r6_mul(h,r6_sub(X(SD,k),X(SD,k+1))));}
  c[k]=r6_sub(c[k],r6_mul(h,X(SA,k+1)));c[k]=r6_sub(c[k],r6_mul(h,X(SD,k+1)));
  d[k]=s.q[k];if(mom&&k==1)d[k]=r6_add(d[k],r6_div(r6_mul(r6_mul(t,in[OCE]),mynn_ipow<2>(in[UST])),in[WSPD]));
  if(k==1){d[k]=r6_sub(d[k],r6_mul(tr,X(FU,k+1)));d[k]=r6_add(d[k],r6_mul(tr,X(FD,k+1)));}
  else {d[k]=r6_add(d[k],r6_mul(tr,r6_sub(X(FU,k),X(FU,k+1))));d[k]=r6_sub(d[k],r6_mul(tr,r6_sub(X(FD,k),X(FD,k+1))));}
  if(mom){d[k]=r6_add(d[k],r6_mul(X(SUB,k),dt));d[k]=r6_add(d[k],r6_mul(X(DET,k),dt));}
 }
 a[n]=0;b[n]=1;c[n]=0;d[n]=s.q[n];
}
template<class P>void aerosol(int j,int k,const P*in,P&lo,P&hi,P&e,P&wt,P&cap){
 P height=j==4?P(2000.f):P(10000.f),low=j==4?P(1e6f):P(0),upper=j==4?P(800e10f):P(500e6f);
 e=std::exp(r6_neg(r6_div(in[ZW*MAX+k],height)));wt=P(0);cap=upper;
 if(j==4&&r6_sub(in[LAND],P(1.5f))>=P(0)){wt=mn(r6_div(mx(r6_sub(in[ZW*MAX+k],P(100)),P(0)),mx(in[PBLH],P(200))),P(1));cap=r6_add(r6_mul(r6_sub(P(1),wt),P(6e8f)),r6_mul(wt,upper));}
 lo=r6_mul(low,e);hi=r6_mul(cap,e);
}
template<class P>void forward(int n,int j,P dt,const P*in,P*out,State<P>&s){
 auto X=[&](int f,int k){return in[f*MAX+k];};
 s.uw=P(0);s.rs=P(0);if(j<2){if(j==0&&in[WSPD]>P(1e-6f))s.uw=r6_div(in[UST],in[WSPD]);s.rs=r6_div(in[PS],r6_mul(P(287.f),r6_add(X(TK,1),r6_mul(P(461.6f/287.f-1.f),X(QV,1)))));}
 for(int k=1;k<=n;++k){s.t[k]=r6_div(dt,X(DZ,k));s.r[k]=r6_div(P(1),k==1?X(RHO,k):mx(X(RHO,k),P(1e-4f)));s.rz[k]=k==1?X(RHO,1):mx(r6_div(r6_add(r6_mul(X(RHO,k),X(DZ,k-1)),r6_mul(X(RHO,k-1),X(DZ,k))),r6_add(X(DZ,k-1),X(DZ,k))),P(1e-4f));s.h[k]=r6_mul(s.rz[k],X(DF,k));s.q[k]=j<2?X(Q,k):mx(P(0),X(Q,k));if((j==2||j==3)&&X(COLD,k)>P(1e-12f))s.q[k]=mx(s.q[k],j==2?P(100.f):P(1e-6f));}
 for(int k=2;k<n;++k){s.h[k]=mx(s.h[k],r6_mul(P(.5),r6_add(X(SA,k),X(SD,k))));s.h[k]=mx(s.h[k],r6_sub(r6_neg(r6_mul(P(.5),r6_sub(X(SA,k),X(SA,k+1)))),r6_mul(P(.5),r6_sub(X(SD,k),X(SD,k+1)))));}
 P a[MAX],b[MAX],c[MAX],d[MAX];assembly(n,j,dt,in,s,a,b,c,d);thomas_forward(n,a,b,c,d,s.cp,s.dp,s.x);
 if(!out)return;
 for(int k=1;k<=n;++k){P q=s.x[k];if((j==2||j==3)&&X(CNEW,k)>P(1e-12f))q=mx(q,j==2?P(100.f):P(1e-6f));if(j>=4){P lo,hi,e,wt,cap;aerosol(j,k,in,lo,hi,e,wt,cap);q=mn(mx(lo,mx(P(0),q)),hi);}out[k]=r6_div(r6_sub(q,X(Q,k)),dt);}
}
template<class P>void reverse(int n,int j,P dt,const P*in,const State<P>&s,const double*seed,double*bar,int planted=0){
 auto X=[&](int f,int k){return in[f*MAX+k];};auto B=[&](int f,int k)->double&{return bar[f*MAX+k];};bool mom=j<2;double Z=0;(void)Z;
 double xb[MAX],cpb[MAX],dpb[MAX],ab[MAX],bb[MAX],cb[MAX],db[MAX],tb[MAX]{},rb[MAX]{},hb[MAX]{},rzb[MAX]{};double rsb=0,uwb=0;
 for(int k=1;k<=n;++k){double q=seed[k]*(1.0/double(dt));B(Q,k)-=q;
  if(j>=4){P lo,hi,e,wt,cap;aerosol(j,k,in,lo,hi,e,wt,cap);P floor=mx(P(0),s.x[k]),m=mx(lo,floor);double lb=0,ub=0;
   if(!(m<hi)){ub=q;q=0;}else if(lo>floor){lb=q;q=0;}
   if(P(0)>s.x[k])q=0;
   double eb=double(j==4?P(1e6f):P(0))*lb+double(cap)*ub,capb=double(e)*ub;
   double exponentb=double(e)*eb;B(ZW,k)-=exponentb/double(j==4?P(2000.f):P(10000.f));
   if(j==4&&r6_sub(in[LAND],P(1.5f))>=P(0)){
    double wtb=-double(P(6e8f))*capb+double(P(800e10f))*capb;P num=mx(r6_sub(X(ZW,k),P(100)),P(0)),den=mx(in[PBLH],P(200));
    if(r6_div(num,den)<P(1)){double nb=0,denb=0;quotient_reverse(num,den,wtb,nb,denb);if(r6_sub(X(ZW,k),P(100))>P(0))B(ZW,k)+=nb;if(in[PBLH]>P(200))DIRECT_FROZEN_BAR(bar[PBLH])+=denb;}
   }
  }else if((j==2||j==3)&&X(CNEW,k)>P(1e-12f)&&!(s.x[k]>(j==2?P(100.f):P(1e-6f))))q=0;
  xb[k]=q;
 }
 P a[MAX],b[MAX],c[MAX],d[MAX];assembly(n,j,dt,in,s,a,b,c,d);thomas_reverse(n,a,b,c,d,s.cp,s.dp,s.x,xb,cpb,dpb,ab,bb,cb,db);
 for(int k=n;k>=1;--k){double q=db[k];if(planted&&k==n/2)q*=1.001;
  if(mom || (!(P(0)>X(Q,k))&&(!((j==2||j==3)&&X(COLD,k)>P(1e-12f)) || mx(P(0),X(Q,k))>(j==2?P(100.f):P(1e-6f)))))B(Q,k)+=q;
  if(k==n)continue;
  P t=s.t[k],r=s.r[k],h=r6_mul(r6_mul(P(.5),t),r),tr=r6_mul(t,r);double hbar=0,trbar=0,junk=0;
  if(k>1){double neg=0;prod2(r6_neg(r6_mul(t,s.h[k])),r,ab[k],neg,rb[k]);prod2(t,s.h[k],-neg,tb[k],hb[k]);prod2(h,X(SA,k),ab[k],hbar,DIRECT_FROZEN_BAR(B(SA,k)));prod2(h,X(SD,k),ab[k],hbar,DIRECT_FROZEN_BAR(B(SD,k)));}
  P sm=k==1?(mom?r6_add(s.h[2],(j==0?r6_mul(r6_mul(s.rs,in[UST]),s.uw):r6_div(r6_mul(s.rs,mynn_ipow<2>(in[UST])),in[WSPD]))):r6_add(s.h[2],s.h[1])):r6_add(s.h[k],s.h[k+1]);double smb=0;prod3(t,sm,r,bb[k],tb[k],smb,rb[k]);hb[k+1]+=smb;
  if(k==1&&mom){if(j==0)prod3(s.rs,in[UST],s.uw,smb,rsb,bar[UST],uwb);else{P pow=mynn_ipow<2>(in[UST]),num=r6_mul(s.rs,pow);double nb=0,pb=0;quotient_reverse(num,in[WSPD],smb,nb,bar[WSPD]);prod2(s.rs,pow,nb,rsb,pb);bar[UST]+=2.0*double(in[UST])*pb;}}else hb[k]+=smb;
  if(k==1){prod2(h,X(SA,k+1),-bb[k],hbar,DIRECT_FROZEN_BAR(B(SA,k+1)));prod2(h,X(SD,k+1),-bb[k],hbar,DIRECT_FROZEN_BAR(B(SD,k+1)));}
  else for(int f=SA;f<=SD;++f){double z=0;prod2(h,r6_sub(X(f,k),X(f,k+1)),bb[k],hbar,z);DIRECT_FROZEN_BAR(B(f,k))+=z;DIRECT_FROZEN_BAR(B(f,k+1))-=z;}
  double neg=0;prod2(r6_neg(r6_mul(t,s.h[k+1])),r,cb[k],neg,rb[k]);prod2(t,s.h[k+1],-neg,tb[k],hb[k+1]);prod2(h,X(SA,k+1),-cb[k],hbar,DIRECT_FROZEN_BAR(B(SA,k+1)));prod2(h,X(SD,k+1),-cb[k],hbar,DIRECT_FROZEN_BAR(B(SD,k+1)));prod3(P(.5),t,r,hbar,junk,tb[k],rb[k]);
  if(mom){DIRECT_PMOM_BAR(B(SUB,k))+=double(dt)*q;DIRECT_PMOM_BAR(B(DET,k))+=double(dt)*q;if(k==1){P pow=mynn_ipow<2>(in[UST]),tu=r6_mul(t,in[OCE]),num=r6_mul(tu,pow);double nb=0,tub=0,pb=0;quotient_reverse(num,in[WSPD],q,nb,bar[WSPD]);prod2(tu,pow,nb,tub,pb);prod2(t,in[OCE],tub,tb[k],bar[OCE]);bar[UST]+=2.0*double(in[UST])*pb;}}
  if(k==1){prod2(tr,X(FU,k+1),-q,trbar,DIRECT_PMOM_BAR(B(FU,k+1)));prod2(tr,X(FD,k+1),q,trbar,DIRECT_FROZEN_BAR(B(FD,k+1)));}
  else{double ub=0,dbb=0;prod2(tr,r6_sub(X(FU,k),X(FU,k+1)),q,trbar,ub);prod2(tr,r6_sub(X(FD,k),X(FD,k+1)),-q,trbar,dbb);DIRECT_PMOM_BAR(B(FU,k))+=ub;DIRECT_PMOM_BAR(B(FU,k+1))-=ub;DIRECT_FROZEN_BAR(B(FD,k))+=dbb;DIRECT_FROZEN_BAR(B(FD,k+1))-=dbb;}
  prod2(t,r,trbar,tb[k],rb[k]);
 }
 if(mom){if(in[WSPD]>P(1e-6f))quotient_reverse(in[UST],in[WSPD],uwb,bar[UST],bar[WSPD]);P therm=r6_add(X(TK,1),r6_mul(P(461.6f/287.f-1.f),X(QV,1))),den=r6_mul(P(287.f),therm);double denb=0;quotient_reverse(in[PS],den,rsb,bar[PS],denb);double thermb=double(P(287.f))*denb;B(TK,1)+=thermb;B(QV,1)+=double(P(461.6f/287.f-1.f))*thermb;}
 for(int k=n;k>=1;--k){double kb=hb[k];if(k>=2&&k<n){P own=r6_mul(s.rz[k],X(DF,k)),f1=r6_mul(P(.5),r6_add(X(SA,k),X(SD,k))),f2=r6_sub(r6_neg(r6_mul(P(.5),r6_sub(X(SA,k),X(SA,k+1)))),r6_mul(P(.5),r6_sub(X(SD,k),X(SD,k+1))));if(!(mx(own,f1)>f2)){double z=-.5*kb;DIRECT_FROZEN_BAR(B(SA,k))+=z;DIRECT_FROZEN_BAR(B(SA,k+1))-=z;DIRECT_FROZEN_BAR(B(SD,k))+=z;DIRECT_FROZEN_BAR(B(SD,k+1))-=z;kb=0;}else if(!(own>f1)){DIRECT_FROZEN_BAR(B(SA,k))+=.5*kb;DIRECT_FROZEN_BAR(B(SD,k))+=.5*kb;kb=0;}}prod2(s.rz[k],X(DF,k),kb,rzb[k],DIRECT_MIX_BAR(B(DF,k)));}
 for(int k=n;k>=1;--k){double discard=0;quotient_reverse(dt,X(DZ,k),tb[k],discard,B(DZ,k));if(k==1||X(RHO,k)>P(1e-4f))quotient_reverse(P(1),X(RHO,k),rb[k],discard,B(RHO,k));if(k==1)B(RHO,k)+=rzb[k];else{P num=r6_add(r6_mul(X(RHO,k),X(DZ,k-1)),r6_mul(X(RHO,k-1),X(DZ,k))),den=r6_add(X(DZ,k-1),X(DZ,k));if(r6_div(num,den)>P(1e-4f)){double nb=0,denb=0;quotient_reverse(num,den,rzb[k],nb,denb);prod2(X(RHO,k),X(DZ,k-1),nb,B(RHO,k),B(DZ,k-1));prod2(X(RHO,k-1),X(DZ,k),nb,B(RHO,k-1),B(DZ,k));B(DZ,k-1)+=denb;B(DZ,k)+=denb;}}}
}
template<class P>bool backward(int n,int j,P dt,const P*in,const double*seed,double*bar){if(n<2||n>=MAX-1||j<0||j>5)return false;State<P>s{};forward(n,j,dt,in,(P*)nullptr,s);reverse(n,j,dt,in,s,seed,bar);return true;}
}}
