#pragma once
#include "predict.hpp"
#include "repair.hpp"
namespace direct {
// Production heat/moisture cut: edmf=1, mixqt=0, cloudmix=1, flag_qc=flag_qi=true.
// No number, aerosol or momentum outputs; those do not feed these outputs in this call.
enum HF{DZ,RHO,TK,QV,EX,TH,THL,SQV,SQC,SQI,SQS,TCD,QCD,HEAT,SUBH,DETH,SUBV,DETV,DETC,DFH,SA,SD,FAH,FDH,FAC,FDC,FAV,FDV,FDI,NHF};
constexpr int HNI=NHF*MAX+4; // scalars ps,flt,flqc,flqv
constexpr int HNO=6*MAX; // thl_out,dth,dqv,dqc,dqi,dqs
// Reused shared transport coefficients; twelve solve profiles are rounded cp,dp,x for four systems.
template<class P>struct HeatState{P t[MAX],r[MAX],rz[MAX],kh[MAX],cp[4][MAX],dp[4][MAX],sol[4][MAX];P repair_in[5*MAX];RepairState<P>repair;P rhosfc,qvflux;};
template<class P>void thomas_forward(int n,const P*a,const P*b,const P*c,const P*d,P*cp,P*dp,P*x){cp[1]=r6_div(c[1],b[1]);dp[1]=r6_div(d[1],b[1]);for(int k=2;k<=n;++k){P m=r6_sub(b[k],r6_mul(cp[k-1],a[k]));cp[k]=r6_div(c[k],m);dp[k]=r6_div(r6_sub(d[k],r6_mul(dp[k-1],a[k])),m);}x[n]=dp[n];for(int k=n-1;k>=1;--k)x[k]=r6_sub(dp[k],r6_mul(cp[k],x[k+1]));}
template<class P>void heat_assembly(int n,P dt,const P*in,const HeatState<P>&s,int j,P*a,P*b,P*c,P*d){
 auto X=[&](int f,int k){return in[f*MAX+k];};int field[4]={THL,SQC,SQV,SQI},fu[4]={FAH,FAC,FAV,FDI},fd[4]={FDH,FDC,FDV,FDI};
 for(int k=1;k<n;++k){P t=s.t[k],r=s.r[k],h=r6_mul(r6_mul(P(.5),t),r),tr=r6_mul(t,r);bool ice=j==3;
  a[k]=r6_mul(r6_neg(r6_mul(t,s.kh[k])),r);b[k]=r6_add(P(1),r6_mul(r6_mul(t,r6_add(k==1?s.kh[k+1]:s.kh[k],k==1?s.kh[k]:s.kh[k+1])),r));c[k]=r6_mul(r6_neg(r6_mul(t,s.kh[k+1])),r);
  if(k==1){if(!ice){b[k]=r6_sub(b[k],r6_mul(h,X(SA,k+1)));c[k]=r6_sub(c[k],r6_mul(h,X(SA,k+1)));}b[k]=r6_sub(b[k],r6_mul(h,X(SD,k+1)));c[k]=r6_sub(c[k],r6_mul(h,X(SD,k+1)));}
  else{if(!ice)a[k]=r6_add(a[k],r6_mul(h,X(SA,k)));a[k]=r6_add(a[k],r6_mul(h,X(SD,k)));if(!ice)b[k]=r6_add(b[k],r6_mul(h,r6_sub(X(SA,k),X(SA,k+1))));b[k]=r6_add(b[k],r6_mul(h,r6_sub(X(SD,k),X(SD,k+1))));if(!ice)c[k]=r6_sub(c[k],r6_mul(h,X(SA,k+1)));c[k]=r6_sub(c[k],r6_mul(h,X(SD,k+1)));}
  d[k]=X(field[j],k);
  if(!ice){
   if(k==1){P flux=j==0?in[NHF*MAX+1]:j==1?in[NHF*MAX+2]:s.qvflux;d[k]=r6_add(d[k],r6_mul(r6_mul(r6_mul(t,s.rhosfc),flux),r));}
   d[k]=r6_add(d[k],r6_mul(X(j==0?TCD:QCD,k),dt));
   if(k==1){d[k]=r6_sub(d[k],r6_mul(tr,X(fu[j],k+1)));d[k]=r6_add(d[k],r6_mul(tr,X(fd[j],k+1)));}
   else{d[k]=r6_add(d[k],r6_mul(tr,r6_sub(X(fu[j],k),X(fu[j],k+1))));d[k]=r6_sub(d[k],r6_mul(tr,r6_sub(X(fd[j],k),X(fd[j],k+1))));}
   if(j==0)d[k]=r6_add(d[k],r6_mul(X(HEAT,k),dt));
   if(j!=1)d[k]=r6_add(d[k],r6_mul(X(j==0?SUBH:SUBV,k),dt));
   d[k]=r6_add(d[k],r6_mul(X(j==0?DETH:j==1?DETC:DETV,k),dt));
  }else{if(k==1)d[k]=r6_add(d[k],r6_mul(tr,X(FDI,k+1)));else d[k]=r6_sub(d[k],r6_mul(tr,r6_sub(X(FDI,k),X(FDI,k+1))));}
 }
 a[n]=0;b[n]=1;c[n]=0;d[n]=X(field[j],n);
}
template<class P>void heat_forward(int n,P dt,const P*in,P*out,HeatState<P>&s,P*repair_capture=nullptr){
 auto X=[&](int f,int k){return in[f*MAX+k];};constexpr P rd=P(287.f),pc=P(461.6f/287.f-1.f),g=P(9.80616f),third=P(.3333333333333333f),lv=P(2.5e6f/1004.5f),ls=P(2.85e6f/1004.5f);
 s.rhosfc=r6_div(in[NHF*MAX],r6_mul(rd,r6_add(X(TK,1),r6_mul(pc,X(QV,1)))));
 for(int k=1;k<=n;++k){s.t[k]=r6_div(dt,X(DZ,k));s.r[k]=r6_div(P(1),k==1?X(RHO,k):mx(X(RHO,k),P(1e-4f)));s.rz[k]=k==1?X(RHO,1):mx(r6_div(r6_add(r6_mul(X(RHO,k),X(DZ,k-1)),r6_mul(X(RHO,k-1),X(DZ,k))),r6_add(X(DZ,k-1),X(DZ,k))),P(1e-4f));s.kh[k]=r6_mul(s.rz[k],X(DFH,k));}
 for(int k=2;k<n;++k){s.kh[k]=mx(s.kh[k],r6_mul(P(.5),r6_add(X(SA,k),X(SD,k))));s.kh[k]=mx(s.kh[k],r6_sub(r6_neg(r6_mul(P(.5),r6_sub(X(SA,k),X(SA,k+1)))),r6_mul(P(.5),r6_sub(X(SD,k),X(SD,k+1)))));}
 s.qvflux=in[NHF*MAX+3];if(s.qvflux<0)s.qvflux=mx(s.qvflux,r6_div(mn(r6_sub(r6_mul(P(.9f),X(SQV,1)),P(1e-8f)),P(0)),s.t[1]));
 P a[MAX],b[MAX],c[MAX],d[MAX];for(int j=0;j<4;++j){heat_assembly(n,dt,in,s,j,a,b,c,d);thomas_forward(n,a,b,c,d,s.cp[j],s.dp[j],s.sol[j]);}
 P repair_in[12*MAX]{},repair_out[12*MAX];
 for(int k=1;k<=n;++k){repair_in[k]=r6_mul(r6_mul(X(RHO,k),g),X(DZ,k));repair_in[MAX+k]=X(EX,k);repair_in[2*MAX+k]=s.sol[2][k];repair_in[3*MAX+k]=s.sol[1][k];repair_in[4*MAX+k]=s.sol[3][k];repair_in[5*MAX+k]=X(SQS,k);repair_in[6*MAX+k]=s.sol[0][k];repair_in[7*MAX+k]=r6_div(r6_sub(s.sol[2][k],X(SQV,k)),dt);repair_in[8*MAX+k]=r6_div(r6_sub(s.sol[1][k],X(SQC,k)),dt);repair_in[9*MAX+k]=r6_div(r6_sub(s.sol[3][k],X(SQI,k)),dt);}
 if(repair_in[1]<r6_mul(third,repair_in[2]))repair_in[1]=r6_mul(third,repair_in[2]);
 for(int f=0;f<6;++f)if(f!=2)for(int k=0;k<MAX;++k)s.repair_in[(f>=3?f-1:f)*MAX+k]=repair_in[f*MAX+k];
 if(repair_capture)for(int i=0;i<12*MAX;++i)repair_capture[i]=repair_in[i];
 repair_forward(n,dt,repair_in,repair_out,s.repair);
 if(!out)return; // Backward-only entry does not materialize discarded final outputs.
 for(int i=0;i<HNO;++i)out[i]=0;
 for(int k=1;k<=n;++k){out[k]=repair_out[6*MAX+k];out[MAX+k]=r6_div(r6_sub(r6_add(r6_add(out[k],r6_mul(r6_div(lv,X(EX,k)),X(SQC,k))),r6_mul(r6_div(ls,X(EX,k)),X(SQI,k))),X(TH,k)),dt);out[2*MAX+k]=repair_out[7*MAX+k];out[3*MAX+k]=repair_out[8*MAX+k];out[4*MAX+k]=repair_out[9*MAX+k];out[5*MAX+k]=repair_out[10*MAX+k];}
}
// Local, hand-written product transpose. Intermediates explicitly rounded as in source.
template<class P>inline void prod2(P a,P b,double g,double&ab,double&bb){acc(ab,double(b),g);acc(bb,double(a),g);}
template<class P>inline void prod3(P a,P b,P c,double g,double&ab,double&bb,double&cb){P abv=r6_mul(a,b);double t=0;prod2(abv,c,g,t,cb);prod2(a,b,t,ab,bb);}
template<class P>void heat_reverse(int n,P dt,const P*in,const HeatState<P>&s,const double*seed,double*bar,int planted=0){
 auto X=[&](int f,int k){return in[f*MAX+k];};auto B=[&](int f,int k)->double&{return bar[f*MAX+k];};
 constexpr P rd=P(287.f),pc=P(461.6f/287.f-1.f),g=P(9.80616f),third=P(.3333333333333333f),lv=P(2.5e6f/1004.5f),ls=P(2.85e6f/1004.5f);
 double rs[12*MAX]{},rb[12*MAX]{},solb[4][MAX],tb[MAX]{},irb[MAX]{},khb[MAX]{},rzb[MAX]{},rhosfcb=0,qvfluxb=0,Z=0;(void)Z;
 for(int k=1;k<=n;++k){double q=seed[MAX+k]/double(dt);rs[6*MAX+k]=seed[k]+q;B(TH,k)-=q;P l=r6_div(lv,X(EX,k)),h=r6_div(ls,X(EX,k));acc(B(SQC,k),double(l),q);acc(B(SQI,k),double(h),q);double lb=double(X(SQC,k))*q,hb=double(X(SQI,k))*q,discard=0;quotient_reverse(lv,X(EX,k),lb,discard,B(EX,k));quotient_reverse(ls,X(EX,k),hb,discard,B(EX,k));for(int j=0;j<4;++j)rs[(7+j)*MAX+k]=seed[(2+j)*MAX+k];}
 repair_reverse(n,dt,s.repair_in,s.repair,rs,rb,0,true);
 for(int k=1;k<=n;++k){B(EX,k)+=rb[MAX+k];DIRECT_FROZEN_BAR(B(SQS,k))+=rb[5*MAX+k];solb[0][k]=rb[6*MAX+k];for(int j=0;j<3;++j){int z[3]={2,1,3},f[3]={SQV,SQC,SQI};double q=rb[(7+j)*MAX+k]/double(dt);solb[z[j]][k]=rb[(2+j)*MAX+k]+q;B(f[j],k)-=q;}}
 P bottom=r6_mul(r6_mul(X(RHO,1),g),X(DZ,1));if(bottom<r6_mul(third,s.repair_in[2])){rb[2]+=double(third)*rb[1];rb[1]=0;}
 for(int k=1;k<=n;++k){double junk=0;prod3(X(RHO,k),g,X(DZ,k),rb[k],B(RHO,k),junk,B(DZ,k));}
 P a[MAX],b[MAX],c[MAX],d[MAX];double xb[MAX],cpb[MAX],dpb[MAX],ab[MAX],bb[MAX],cb[MAX],db[MAX];
 int field[4]={THL,SQC,SQV,SQI},fu[4]={FAH,FAC,FAV,FDI},fd[4]={FDH,FDC,FDV,FDI};
 for(int j=3;j>=0;--j){
  heat_assembly(n,dt,in,s,j,a,b,c,d);for(int k=1;k<=n;++k){xb[k]=solb[j][k];}
  thomas_reverse(n,a,b,c,d,s.cp[j],s.dp[j],s.sol[j],xb,cpb,dpb,ab,bb,cb,db);
  B(field[j],n)+=db[n];
  for(int k=n-1;k>=1;--k){P t=s.t[k],r=s.r[k],h=r6_mul(r6_mul(P(.5),t),r),tr=r6_mul(t,r);double hb=0,trb=0,junk=0;bool ice=j==3;
   // coefficient source expressions, including all matrix bars
   if(k>1){double neg=0;prod2(r6_neg(r6_mul(t,s.kh[k])),r,ab[k],neg,irb[k]);prod2(t,s.kh[k],-neg,tb[k],khb[k]);if(!ice)prod2(h,X(SA,k),ab[k],hb,DIRECT_FROZEN_BAR(B(SA,k)));prod2(h,X(SD,k),ab[k],hb,DIRECT_FROZEN_BAR(B(SD,k)));}
   P sm=r6_add(k==1?s.kh[k+1]:s.kh[k],k==1?s.kh[k]:s.kh[k+1]);double sm_b=0;prod3(t,sm,r,bb[k],tb[k],sm_b,irb[k]);khb[k]+=sm_b;khb[k+1]+=sm_b;
   if(k==1){if(!ice)prod2(h,X(SA,k+1),-bb[k],hb,DIRECT_FROZEN_BAR(B(SA,k+1)));prod2(h,X(SD,k+1),-bb[k],hb,DIRECT_FROZEN_BAR(B(SD,k+1)));}
   else{for(int f=SA;f<=SD;++f)if(f!=SA||!ice){double diffb=0;prod2(h,r6_sub(X(f,k),X(f,k+1)),bb[k],hb,diffb);DIRECT_FROZEN_BAR(B(f,k))+=diffb;DIRECT_FROZEN_BAR(B(f,k+1))-=diffb;}}
   double neg=0;prod2(r6_neg(r6_mul(t,s.kh[k+1])),r,cb[k],neg,irb[k]);prod2(t,s.kh[k+1],-neg,tb[k],khb[k+1]);if(!ice)prod2(h,X(SA,k+1),-cb[k],hb,DIRECT_FROZEN_BAR(B(SA,k+1)));prod2(h,X(SD,k+1),-cb[k],hb,DIRECT_FROZEN_BAR(B(SD,k+1)));
   prod3(P(.5),t,r,hb,junk,tb[k],irb[k]);
   double q=db[k];if(planted&&j==0&&k==n/2)q*=1.001;B(field[j],k)+=q;
   if(!ice){
    if(k==1){P flux=j==0?in[NHF*MAX+1]:j==1?in[NHF*MAX+2]:s.qvflux;double fb=0,abc=0;prod2(r6_mul(r6_mul(t,s.rhosfc),flux),r,q,abc,irb[k]);prod3(t,s.rhosfc,flux,abc,tb[k],rhosfcb,fb);if(j==2)qvfluxb+=fb;else bar[NHF*MAX+1+j]+=fb;}
    B(j==0?TCD:QCD,k)+=double(dt)*q;
#ifdef JOIN_PHEAT
    // the plume's heat (j=0) and vapour (j=2) terms (FAH/FAV, SUBH/SUBV, DETH/DETV) are read back by join.hpp
    // pheat_reverse / pmoist_reverse; the cloud water's (j=1: FAC, DETC) and the downdraft's stay held. Live bars as below.
    {const bool lv=j!=1;
    if(k==1){double fub=0;prod2(tr,X(fu[j],k+1),-q,trb,fub);if(lv)B(fu[j],k+1)+=fub;prod2(tr,X(fd[j],k+1),q,trb,DIRECT_FROZEN_BAR(B(fd[j],k+1)));}
    else{double ub=0,vb=0;prod2(tr,r6_sub(X(fu[j],k),X(fu[j],k+1)),q,trb,ub);prod2(tr,r6_sub(X(fd[j],k),X(fd[j],k+1)),-q,trb,vb);if(lv){B(fu[j],k)+=ub;B(fu[j],k+1)-=ub;}DIRECT_FROZEN_BAR(B(fd[j],k))+=vb;DIRECT_FROZEN_BAR(B(fd[j],k+1))-=vb;}
    if(j==0)B(HEAT,k)+=double(dt)*q;if(lv){B(j==0?SUBH:SUBV,k)+=double(dt)*q;B(j==0?DETH:DETV,k)+=double(dt)*q;}}
#else
    if(k==1){prod2(tr,X(fu[j],k+1),-q,trb,DIRECT_FROZEN_BAR(B(fu[j],k+1)));prod2(tr,X(fd[j],k+1),q,trb,DIRECT_FROZEN_BAR(B(fd[j],k+1)));}
    else{double ub=0,vb=0;prod2(tr,r6_sub(X(fu[j],k),X(fu[j],k+1)),q,trb,ub);prod2(tr,r6_sub(X(fd[j],k),X(fd[j],k+1)),-q,trb,vb);DIRECT_FROZEN_BAR(B(fu[j],k))+=ub;DIRECT_FROZEN_BAR(B(fu[j],k+1))-=ub;DIRECT_FROZEN_BAR(B(fd[j],k))+=vb;DIRECT_FROZEN_BAR(B(fd[j],k+1))-=vb;}
    if(j==0)B(HEAT,k)+=double(dt)*q;if(j!=1)DIRECT_FROZEN_BAR(B(j==0?SUBH:SUBV,k))+=double(dt)*q;DIRECT_FROZEN_BAR(B(j==0?DETH:j==1?DETC:DETV,k))+=double(dt)*q;
#endif
   }else{if(k==1)prod2(tr,X(FDI,k+1),q,trb,DIRECT_FROZEN_BAR(B(FDI,k+1)));else{double fb=0;prod2(tr,r6_sub(X(FDI,k),X(FDI,k+1)),-q,trb,fb);DIRECT_FROZEN_BAR(B(FDI,k))+=fb;DIRECT_FROZEN_BAR(B(FDI,k+1))-=fb;}}
   prod2(t,r,trb,tb[k],irb[k]);
  }
 }
 // surface vapor limiter and moist surface density
 P f=in[NHF*MAX+3];if(f<0){P v=r6_sub(r6_mul(P(.9f),X(SQV,1)),P(1e-8f)),vmin=mn(v,P(0)),lim=r6_div(vmin,s.t[1]);if(!(f>lim)){double vb=0;quotient_reverse(vmin,s.t[1],qvfluxb,vb,tb[1]);if(v<P(0))B(SQV,1)+=double(P(.9f))*vb;qvfluxb=0;}}bar[NHF*MAX+3]+=qvfluxb;
 P therm=r6_add(X(TK,1),r6_mul(pc,X(QV,1))),den=r6_mul(rd,therm);double denb=0;quotient_reverse(in[NHF*MAX],den,rhosfcb,bar[NHF*MAX],denb);double thermb=double(rd)*denb;B(TK,1)+=thermb;B(QV,1)+=double(pc)*thermb;
 // floor selection and shared interpolation. Rebuild choices; no stored branch bits.
 for(int k=n;k>=1;--k){double kb=khb[k];if(k>=2&&k<n){P own=r6_mul(s.rz[k],X(DFH,k)),f1=r6_mul(P(.5),r6_add(X(SA,k),X(SD,k))),f2=r6_sub(r6_neg(r6_mul(P(.5),r6_sub(X(SA,k),X(SA,k+1)))),r6_mul(P(.5),r6_sub(X(SD,k),X(SD,k+1))));if(!(mx(own,f1)>f2)){double z=-.5*kb;DIRECT_FROZEN_BAR(B(SA,k))+=z;DIRECT_FROZEN_BAR(B(SA,k+1))-=z;DIRECT_FROZEN_BAR(B(SD,k))+=z;DIRECT_FROZEN_BAR(B(SD,k+1))-=z;kb=0;}else if(!(own>f1)){DIRECT_FROZEN_BAR(B(SA,k))+=.5*kb;DIRECT_FROZEN_BAR(B(SD,k))+=.5*kb;kb=0;}}prod2(s.rz[k],X(DFH,k),kb,rzb[k],DIRECT_MIX_BAR(B(DFH,k)));}
 for(int k=n;k>=1;--k){double discard=0;quotient_reverse(dt,X(DZ,k),tb[k],discard,B(DZ,k));if(k==1||X(RHO,k)>P(1e-4f))quotient_reverse(P(1),X(RHO,k),irb[k],discard,B(RHO,k));if(k==1)B(RHO,k)+=rzb[k];else{P num=r6_add(r6_mul(X(RHO,k),X(DZ,k-1)),r6_mul(X(RHO,k-1),X(DZ,k))),dn=r6_add(X(DZ,k-1),X(DZ,k));if(r6_div(num,dn)>P(1e-4f)){double nb=0,denbar=0;quotient_reverse(num,dn,rzb[k],nb,denbar);prod2(X(RHO,k),X(DZ,k-1),nb,B(RHO,k),B(DZ,k-1));prod2(X(RHO,k-1),X(DZ,k),nb,B(RHO,k-1),B(DZ,k));B(DZ,k-1)+=denbar;B(DZ,k)+=denbar;}}}
}
}
