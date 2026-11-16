// SPDX-License-Identifier: Apache-2.0
#include "mpas_f32.cuh"
typedef int SeedFailure;
// One zero-initialized int32 flag per refusal source; integer atomics are deterministic.
__device__ __forceinline__ void ms_fail(SeedFailure *f,unsigned long long slot,unsigned long long field,unsigned long long index,unsigned long long source){(void)slot;(void)field;(void)index;atomicExch(&f[source-1],1);}
struct SeedTerm {float id,iv,lap,ke,prod,b_lap,b_ke,b_rho,b_del2;double b_lap64;bool valid;};
__device__ __forceinline__ SeedTerm ms_term(int edge,int k,const float*rho,const float*div,const float*vort,const float*kd,const float*d2,const int*coe,const int*voe,const float*dc,const float*dv,int nc,int ne,int nv,float bl,float bt){
 SeedTerm t={};int c0=coe[2*edge],c1=coe[2*edge+1],v0=voe[2*edge],v1=voe[2*edge+1];
 if(c0<0||c0>=nc||c1<0||c1>=nc||v0<0||v0>=nv||v1<0||v1>=nv||c0==c1||v0==v1)return t;
 int i=k*ne+edge;float dist=dc[edge],vv=dv[edge],r=rho[i],z=d2[edge],a=div[k*nc+c0],b=div[k*nc+c1],x=vort[k*nv+v0],y=vort[k*nv+v1],p=kd[k*nc+c0],q=kd[k*nc+c1];
 if(!isfinite(dist)||dist==0||!isfinite(vv)||vv==0||!isfinite(r)||!isfinite(z)||!isfinite(a)||!isfinite(b)||!isfinite(x)||!isfinite(y)||!isfinite(p)||!isfinite(q)||!isfinite(bl)||!isfinite(bt))return t;
 t.id=mpas_div(1.0f,dist);t.iv=mpas_min(mpas_div(1.0f,vv),mpas_mul(4.0f,t.id));
 t.lap=mpas_sub(mpas_mul(mpas_sub(b,a),t.id),mpas_mul(mpas_sub(y,x),t.iv));t.ke=mpas_mul(0.5f,mpas_add(p,q));
 float aa=mpas_mul(r,t.ke);t.prod=mpas_mul(aa,t.lap);float bprod=mpas_mul(bt,z),ba=mpas_mul(bprod,t.lap);
 float bterm=mpas_mul(bprod,aa);t.b_lap=mpas_add(bl,bterm);t.b_lap64=mpas_f32_to_f64_ieee(bl)+mpas_f32_to_f64_ieee(bt)*mpas_f32_to_f64_ieee(z)*mpas_f32_to_f64_ieee(aa);t.b_ke=mpas_mul(ba,r);t.b_rho=mpas_mul(ba,t.ke);t.b_del2=mpas_mul(bt,t.prod);
 t.valid=isfinite(t.id)&&isfinite(t.iv)&&isfinite(t.lap)&&isfinite(t.ke)&&isfinite(t.b_lap)&&isfinite(t.b_ke)&&isfinite(t.b_rho)&&isfinite(t.b_del2);return t;
}
extern "C" __global__ void momentum_filter_seed_v841_f32__tl(
 const float*rho,const float*div,const float*vort,const float*kd,const float*d2,const int*coe,const int*voe,const float*dc,const float*dv,
 const int nlev,const int nc,const int ne,const int nv,float*lap_out,float*tendency,
 const float*drho,const float*ddiv,const float*dvort,const float*dkd,const float*dd2,float*d_lap,float*d_tendency,SeedFailure*failure){
 (void)lap_out;(void)tendency;int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=ne)return;int c0=coe[2*e],c1=coe[2*e+1],v0=voe[2*e],v1=voe[2*e+1];
 for(int k=0;k<nlev;++k){int i=k*ne+e;SeedTerm t=ms_term(e,k,rho,div,vort,kd,d2,coe,voe,dc,dv,nc,ne,nv,0.0f,1.0f);
  if(!t.valid){ms_fail(failure,i, 1, i, 1);continue;}
  float dl=mpas_sub(mpas_mul(mpas_sub(ddiv[k*nc+c1],ddiv[k*nc+c0]),t.id),mpas_mul(mpas_sub(dvort[k*nv+v1],dvort[k*nv+v0]),t.iv));
  float dke=mpas_mul(0.5f,mpas_add(dkd[k*nc+c0],dkd[k*nc+c1]));
  float change=mpas_add(mpas_add(mpas_mul(t.b_rho,drho[i]),mpas_mul(t.b_ke,dke)),mpas_add(mpas_mul(t.b_lap,dl),mpas_mul(t.b_del2,dd2[e])));
  float res=mpas_add(d_tendency[i],change);if(!isfinite(dl)||!isfinite(res)){ms_fail(failure,i, 15, i, 2);continue;}d_lap[i]=dl;d_tendency[i]=res;
 }
}
extern "C" __global__ void momentum_filter_seed_v841_f32__ad(
 const float*rho,const float*div,const float*vort,const float*kd,const float*d2,const int*coe,const int*voe,const float*dc,const float*dv,
 const int nlev,const int nc,const int ne,const int nv,float*lap_out,float*tendency,
 const float*bl_snapshot,const float*bt_snapshot,float*bl_out,float*bt_out,float*brho,float*bdiv,float*bvort,float*bkd,float*bd2,float*bt_in,
 const int*edges_on_cell,const int*n_edges_on_cell,const int*edges_on_vertex,const int max_edges,const int vertex_degree,SeedFailure*failure){
 (void)lap_out;(void)tendency;int first=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;unsigned long long nedge=(unsigned long long)nlev*ne,ncell=(unsigned long long)nlev*nc,nvert=(unsigned long long)nlev*nv;
 for(int i=first;i<nlev*ne;i+=stride){int e=i%ne,k=i/ne;SeedTerm t=ms_term(e,k,rho,div,vort,kd,d2,coe,voe,dc,dv,nc,ne,nv,bl_snapshot[i],bt_snapshot[i]);
  if(!t.valid||!isfinite(brho[i])||!isfinite(bt_in[i])){ms_fail(failure,i, 1, i, 3);continue;}
  brho[i]=mpas_add(brho[i],t.b_rho);bt_in[i]=mpas_add(bt_in[i],bt_snapshot[i]);bl_out[i]=0.0f;bt_out[i]=0.0f;
 }
 for(int i=first;i<nlev*nc;i+=stride){int c=i%nc,k=i/nc,count=n_edges_on_cell[c];unsigned long long slot=nedge+i;double bd=mpas_f32_to_f64_ieee(bdiv[i]),bk=mpas_f32_to_f64_ieee(bkd[i]);
  if(count<0||count>max_edges||!isfinite(bd)||!isfinite(bk)){ms_fail(failure,slot, 2, i, 4);continue;}
  for(int j=0;j<count;++j){int e=edges_on_cell[c*max_edges+j];if(e<0||e>=ne){ms_fail(failure,slot, 2, i, 5);continue;}int c0=coe[2*e],c1=coe[2*e+1];if(c!=c0&&c!=c1){ms_fail(failure,slot, 2, i, 6);continue;}int ei=k*ne+e;SeedTerm t=ms_term(e,k,rho,div,vort,kd,d2,coe,voe,dc,dv,nc,ne,nv,bl_snapshot[ei],bt_snapshot[ei]);if(!t.valid){ms_fail(failure,slot, 2, i, 7);continue;}double v=t.b_lap64*mpas_f32_to_f64_ieee(t.id);bd+=c==c1?v:-v;bk+=mpas_f32_to_f64_ieee(mpas_mul(0.5f,t.b_ke));}
  float out=mpas_f64_to_f32_ieee(bd),out_k=mpas_f64_to_f32_ieee(bk);if(!isfinite(out)||!isfinite(out_k))ms_fail(failure,slot, 2, i, 8);bdiv[i]=out;bkd[i]=out_k;
 }
 for(int i=first;i<nlev*nv;i+=stride){int v=i%nv,k=i/nv;unsigned long long slot=nedge+ncell+i;double b=mpas_f32_to_f64_ieee(bvort[i]);if(!isfinite(b)){ms_fail(failure,slot, 3, i, 9);continue;}
  for(int j=0;j<vertex_degree;++j){int e=edges_on_vertex[v*vertex_degree+j];if(e<0)continue;if(e>=ne){ms_fail(failure,slot, 3, i, 10);continue;}int v0=voe[2*e],v1=voe[2*e+1];if(v!=v0&&v!=v1){ms_fail(failure,slot, 3, i, 11);continue;}int ei=k*ne+e;SeedTerm t=ms_term(e,k,rho,div,vort,kd,d2,coe,voe,dc,dv,nc,ne,nv,bl_snapshot[ei],bt_snapshot[ei]);if(!t.valid){ms_fail(failure,slot, 3, i, 12);continue;}double x=t.b_lap64*mpas_f32_to_f64_ieee(t.iv);b+=v==v0?x:-x;}
  float out=mpas_f64_to_f32_ieee(b);if(!isfinite(out))ms_fail(failure,slot, 3, i, 13);bvort[i]=out;
 }
 for(int e=first;e<ne;e+=stride){unsigned long long slot=nedge+ncell+nvert+e;double b=mpas_f32_to_f64_ieee(bd2[e]);if(!isfinite(b)){ms_fail(failure,slot, 5, e, 14);continue;}for(int k=0;k<nlev;++k){int i=k*ne+e;SeedTerm t=ms_term(e,k,rho,div,vort,kd,d2,coe,voe,dc,dv,nc,ne,nv,bl_snapshot[i],bt_snapshot[i]);if(!t.valid){ms_fail(failure,slot, 5, e, 15);continue;}b+=mpas_f32_to_f64_ieee(t.b_del2);}float out=mpas_f64_to_f32_ieee(b);if(!isfinite(out))ms_fail(failure,slot, 5, e, 16);bd2[e]=out;}
}
