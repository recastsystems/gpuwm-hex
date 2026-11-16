#pragma once
#include "plume_column.hpp"
// The plume block of the tape-free MYNN backward, as production B9 differentiates it.
//
// Recipe (20a441f67, tree/ad/phys/mynn): gen_whole.py --freeze closure, composed as variant B
// (integrated/compose.sh: M8 M6 M2 M3 M10 M4 M9). Stage M4 (cuda/m4_plume.py) runs the whole updraft integration of
// f_dmp_mf (everything before the edmf_th1 loop) on frozen copies of its inputs, and gen_whole.py freezes every
// listed output of f_dmp_mf right after the call (FREEZE_AFTER['f_dmp_mf']). What stays live is the cloud-blend tail
// F90:7544-7713: it reads the frozen plume values (edmf_a1, edmf_th1, edmf_qt1, edmf_qc1, edmf_w1, mf_at_cb, ktop)
// and the LIVE environment (dz1, tk1, pres1, qt1, ex1, thl1 and the incoming cldfra_bl1, qc_bl1, vt1, vq1), and
// overwrites cldfra_bl1, qc_bl1, vt1, vq1 at the levels where the plume carries condensate.
// The only other live arithmetic in the M4 prefix (qc_bl1 into envm_sqc into det_sqc1, mynn-arena) ends in det_sqc1,
// which is frozen after the call, so it sends no bar out of f_dmp_mf.
// Hence the transpose: the plume values are recomputed primals with no bar; the environment bars are the tail's.
// edmf_th1 = edmf_thl1 + xlvcp/ex1*edmf_qc1 is formed in the live tail from ex1, but the blend never reads it (thp is
// assigned and unused at F90:7563), so it adds no ex1 bar; the test checks that its bar is exactly zero.
//
// Layout as plume_column.hpp: per level area,plth,plqt,plqc,w,dz,tk,p,qt,ex,thl,cldfra,qcbl,vt,vq; then landsea.
// n <= 59, ktop zero-based. Float primals, double bars; bars are ADDED to b (plume and landsea slots untouched).
// planted=true scales the tk1 bar of every blended level by 1.001 (a test-only error; keep false).
namespace direct { namespace plume {
template<class P> inline void tail_reverse(int n,int ktop,const P*x,const double*s,double*b,P*y=nullptr,bool planted=false){
 P qi[60]{};double qb[60]{};
 qi[0]=x[8];
 for(int k=0;k<n-1;++k){const P*a=x+15*k,*z=a+15;qi[k+1]=P(P(P(a[8]*z[5])+P(z[8]*a[5]))/P(z[5]+a[5]));}
 P mf=0;
 for(int k=1;k<=n-3 && k<=ktop;++k)if(x[15*k+3]>P(0)){mf=P(x[15*k]*x[15*k+4]);break;}
 for(int k=n-1;k>=0;--k){const P*a=x+15*k;
  if(k<1 || k>n-3 || k>ktop || !(a[3]>P(0))){for(int j=0;j<4;++j){b[15*k+11+j]+=s[4*k+j];if(y)y[4*k+j]=a[11+j];}continue;}
  P v[19]={a[0],a[1],a[2],a[3],a[4],qi[k],qi[k+1],a[5],a[6],a[7],a[8],a[9],a[10],x[15*n],mf,a[11],a[12],a[13],a[14]};double vb[19]{};
  blend_reverse(v,s+4*k,vb,y?y+4*k:nullptr,false);
  if(planted)vb[8]*=1.001;
  // live environment only: v[7..12] = dz,tk,p,qt,ex,thl -> slots 5..10; v[15..18] = cldfra,qcbl,vt,vq -> slots 11..14
  for(int j=5;j<11;++j)b[15*k+j]+=vb[j+2];for(int j=11;j<15;++j)b[15*k+j]+=vb[j+4];
  qb[k]+=vb[5];qb[k+1]+=vb[6];
 }
 for(int k=n-2;k>=0;--k){P v[4]={x[15*k+8],x[15*(k+1)+8],x[15*k+5],x[15*(k+1)+5]};double vb[4]{};interp_reverse(v,qb+k+1,vb);b[15*k+8]+=vb[0];b[15*(k+1)+8]+=vb[1];b[15*k+5]+=vb[2];b[15*(k+1)+5]+=vb[3];}
 b[8]+=qb[0];
}
}}
