// SPDX-License-Identifier: Apache-2.0
// Adapted from 20a441f67 cuda/stages/m7_turbprod.cuh; see derivation.md.
#pragma once
#include "numeric.hpp"
namespace direct {
// Profile-major, stride MAX. qdiv remains double even with float operands.
// x: qkw gm gh dtl dqw dz el sm sh gamt gamq gamv tkeprod_dn tkeprod_up.
// y/seed: pdk pdt pdq pdc tcd qcd qshear qbuoy.
// Input bar: qkw gm gh dtl dqw dz. All are +=; no aliasing is supported.
constexpr int TPNI=14*MAX, TPNO=8*MAX, TPNB=6*MAX;
template<class P> struct M7Prim {P el,qkw,sm,sh,gm,gh,dtl,dqw;double qdiv;P gamt,gamq,gamv;};
struct M7Bar {double pdk,pdt,pdq,pdc,pre_t,pre_q,qshear,qbuoy;};
struct M7In {double qkw,gm,gh,dtl,dqw,sm,sh,qdiv;};   // sm, sh, qdiv: read only by the closure-live join (full/closure)
template<class P> inline M7Prim<P> m7_load(const P*x,const double*qdiv,int k){
 return {x[6*MAX+k],x[k],x[7*MAX+k],x[8*MAX+k],x[MAX+k],x[2*MAX+k],x[3*MAX+k],x[4*MAX+k],qdiv[k],x[9*MAX+k],x[10*MAX+k],x[11*MAX+k]};
}
template<class P> inline void m7_rev_level(const M7Prim<P>&p,const M7Bar&b,int budget,M7In&o,bool planted=false){
 const P elq=r6_mul(p.el,p.qkw),elh=P(double(elq)*p.qdiv);
 const P S=r6_add(r6_add(r6_mul(p.sm,p.gm),r6_mul(p.sh,p.gh)),p.gamv);
 const P A=r6_add(r6_mul(p.sh,p.dtl),p.gamt),B=r6_add(r6_mul(p.sh,p.dqw),p.gamq);
 const P elhA=r6_mul(elh,A),elhB=r6_mul(elh,B);
 double eqb=b.pdk*double(S);
 double gmb=b.pdk*double(elq)*double(p.sm),ghb=b.pdk*double(elq)*double(p.sh);
 const double c=b.pdc*.5;
 const double ab=b.pdt*double(p.dtl)+c*double(p.dqw),bb=b.pdq*double(p.dqw)+c*double(p.dtl);
 double dtb=b.pdt*double(elhA)+c*double(elhB),dqb=b.pdq*double(elhB)+c*double(elhA);
 const double hb=ab*double(A)+bb*double(B);
 dtb+=ab*double(elh)*double(p.sh);dqb+=bb*double(elh)*double(p.sh);
 eqb+=hb*p.qdiv;
 eqb+=b.pre_t*double(p.gamt)+b.pre_q*double(p.gamq);
 if(budget==1){
  const P eqsm=r6_mul(elq,p.sm),Q=r6_add(r6_mul(p.gh,p.sh),p.gamv);
  eqb+=b.qshear*double(p.sm)*double(p.gm);gmb+=b.qshear*double(eqsm);
  eqb+=b.qbuoy*double(Q);ghb+=b.qbuoy*double(elq)*double(p.sh);
 }
 // the bars of the stability functions and of qdiv (S = sm gm + sh gh; A = sh dtl; B = sh dqw; elh = elq qdiv)
 o.sm+=b.pdk*double(elq)*double(p.gm);
 o.sh+=b.pdk*double(elq)*double(p.gh)+ab*double(elh)*double(p.dtl)+bb*double(elh)*double(p.dqw);
 o.qdiv+=hb*double(elq);
 if(budget==1){o.sm+=b.qshear*double(elq)*double(p.gm);o.sh+=b.qbuoy*double(elq)*double(p.gh);}
 o.qkw+=eqb*double(p.el)*(planted?1.001:1.0);
 o.gm+=gmb;o.gh+=ghb;o.dtl+=dtb;o.dqw+=dqb;
}
template<class P> inline P m7_pre(const P*x,int gam,int k,int n){
 return k<=1||k>=n?P(0):r6_mul(r6_mul(x[6*MAX+k],x[k]),x[gam*MAX+k]);
}
// The native ADF division weight uses the unrounded quotient of its rounded
// numerator, not its float output. This corrects the older hand source.
template<class P> inline double m7_diff_rev_gather(int k,int n,double bm,double b,P dzm,P dz,P pre,P next,double&dzb){
 double pb=0;
 if(k<n){
  const P numerator=r6_sub(next,pre);
  const double inv=1.0/double(dz);
  if(k>1)pb-=inv*b;
  dzb+=(-(double(numerator)/double(dz))/double(dz))*b;
 }
 if(k>1&&k<n)pb+=(1.0/double(dzm))*bm;
 return pb;
}
template<class P> inline void turbprod_reverse(int n,P closure,int budget,const P*x,const double*qdiv,const double*seed,double*bar,bool planted=false,double*kb=nullptr){
 // Fixed one-column ownership; every level is visited in ascending order.
 for(int k=1;k<=n;++k){
  M7Bar b{seed[k],seed[MAX+k],seed[2*MAX+k],seed[3*MAX+k],0,0,0,0};
  if(budget==1){b.qshear=seed[6*MAX+k];b.qbuoy=seed[7*MAX+k];}
  if(closure>=P(3)){
   double dzb=0;const P dz=x[5*MAX+k],dzm=k>1?x[5*MAX+k-1]:P(1);
   b.pre_t=m7_diff_rev_gather(k,n,k>1?seed[4*MAX+k-1]:0,seed[4*MAX+k],dzm,dz,m7_pre(x,9,k,n),m7_pre(x,9,k+1,n),dzb);
   b.pre_q=m7_diff_rev_gather(k,n,k>1?seed[5*MAX+k-1]:0,seed[5*MAX+k],dzm,dz,m7_pre(x,10,k,n),m7_pre(x,10,k+1,n),dzb);
   bar[5*MAX+k]+=dzb;
  }
  if(k>1){M7In o{};m7_rev_level(m7_load(x,qdiv,k),b,budget,o,planted);bar[k]+=o.qkw;bar[MAX+k]+=o.gm;bar[2*MAX+k]+=o.gh;bar[3*MAX+k]+=o.dtl;bar[4*MAX+k]+=o.dqw;if(kb){kb[k]+=o.sm;kb[MAX+k]+=o.sh;kb[2*MAX+k]+=o.qdiv;}}
 }
}
template<class P> inline void turbprod_forward(int n,int budget,const P*x,const double*qdiv,P*y){
 for(int j=0;j<8;++j)for(int k=0;k<MAX;++k)y[j*MAX+k]=P(0);
 for(int k=2;k<=n;++k){
  const auto p=m7_load(x,qdiv,k);const P eq=r6_mul(p.el,p.qkw),h=P(double(eq)*p.qdiv);
  const P A=r6_add(r6_mul(p.sh,p.dtl),p.gamt),B=r6_add(r6_mul(p.sh,p.dqw),p.gamq);
  y[k]=r6_add(r6_add(r6_mul(eq,r6_add(r6_add(r6_mul(p.sm,p.gm),r6_mul(p.sh,p.gh)),p.gamv)),r6_mul(P(.5),x[12*MAX+k])),r6_mul(P(.5),x[13*MAX+k]));
  y[MAX+k]=r6_mul(r6_mul(h,A),p.dtl);y[2*MAX+k]=r6_mul(r6_mul(h,B),p.dqw);
  y[3*MAX+k]=r6_add(r6_mul(r6_mul(r6_mul(h,A),p.dqw),P(.5)),r6_mul(r6_mul(r6_mul(h,B),p.dtl),P(.5)));
  y[4*MAX+k]=r6_mul(eq,p.gamt);y[5*MAX+k]=r6_mul(eq,p.gamq);
  if(budget==1){y[6*MAX+k]=r6_mul(r6_mul(eq,p.sm),p.gm);y[7*MAX+k]=r6_add(r6_add(r6_mul(eq,r6_add(r6_mul(p.gh,p.sh),p.gamv)),r6_mul(P(.5),x[12*MAX+k])),r6_mul(P(.5),x[13*MAX+k]));}
 }
 y[4*MAX+n]=y[5*MAX+n]=P(0);
 for(int k=1;k<n;++k){y[4*MAX+k]=r6_div(r6_sub(y[4*MAX+k+1],y[4*MAX+k]),x[5*MAX+k]);y[5*MAX+k]=r6_div(r6_sub(y[5*MAX+k+1],y[5*MAX+k]),x[5*MAX+k]);}
}
} // namespace direct
