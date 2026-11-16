#pragma once
#include "repair.hpp"
namespace direct {
// Production f_qc=f_qi=f_qs=true. Profile order qv,qc,qi,qs[,dqv,dqc,dqi,dqs].
// No snapshots: every rounded scalar is rebuilt once per level. Additive bar, no aliases.
template<class P>void pre_forward(int n,const P*x,P*y){for(int j=0;j<4;++j)for(int k=1;k<=n;++k)y[j*MAX+k]=r6_div(x[j*MAX+k],r6_add(P(1),x[k]));}
template<class P>void pre_reverse(int n,const P*x,const double*s,double*b,int planted=0){for(int k=1;k<=n;++k){P den=r6_add(P(1),x[k]);double db=0;for(int j=3;j>=0;--j)quotient_reverse(x[j*MAX+k],den,(s[j*MAX+k]*(planted&&j==0&&k==n/2?1.001:1.)),b[j*MAX+k],db);b[k]+=db;}}
template<class P>void post_forward(int n,P dt,const P*x,P*y){for(int k=1;k<=n;++k){P den=r6_add(P(1),x[k]),v=r6_add(r6_div(x[k],den),r6_mul(x[4*MAX+k],dt));y[k]=r6_div(r6_sub(r6_div(v,r6_sub(P(1),v)),x[k]),dt);for(int j=1;j<4;++j){P c=r6_add(r6_div(x[j*MAX+k],den),r6_mul(x[(4+j)*MAX+k],dt));y[j*MAX+k]=r6_div(r6_sub(r6_mul(c,r6_add(P(1),v)),x[j*MAX+k]),dt);}}}
template<class P>void post_reverse(int n,P dt,const P*x,const double*s,double*b,int planted=0){for(int k=1;k<=n;++k){P den=r6_add(P(1),x[k]),v=r6_add(r6_div(x[k],den),r6_mul(x[4*MAX+k],dt));double vb=0,denb=0;
 for(int j=3;j>=1;--j){P c=r6_add(r6_div(x[j*MAX+k],den),r6_mul(x[(4+j)*MAX+k],dt));double z=s[j*MAX+k]/double(dt);b[j*MAX+k]-=z;double cb=double(r6_add(P(1),v))*z;vb+=double(c)*z*(planted&&j==1&&k==n/2?1.001:1.);b[(4+j)*MAX+k]+=double(dt)*cb;quotient_reverse(x[j*MAX+k],den,cb,b[j*MAX+k],denb);}
 double z=s[k]/double(dt),dd=0;b[k]-=z;quotient_reverse(v,r6_sub(P(1),v),z,vb,dd);vb-=dd;b[4*MAX+k]+=double(dt)*vb;quotient_reverse(x[k],den,vb,b[k],denb);b[k]+=denb;
}}
}
