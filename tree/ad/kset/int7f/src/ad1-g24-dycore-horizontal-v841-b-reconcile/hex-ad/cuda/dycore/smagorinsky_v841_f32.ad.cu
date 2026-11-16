// SPDX-License-Identifier: Apache-2.0
#include "mpas_f32.cuh"
typedef int SmagFailure;
// One zero-initialized int32 flag per refusal source; integer atomics are deterministic.
__device__ __forceinline__ void sm_fail(SmagFailure *f,unsigned long long slot,unsigned long long field,unsigned long long index,unsigned long long source){(void)slot;(void)field;(void)index;atomicExch(&f[source-1],1);}
struct SmagTerm {float bd,b12,bs,bc;bool valid,singular,smooth;};
__device__ __forceinline__ SmagTerm sm_term(int cell,int k,const float*u,const float*v,const int*eoc,const int*neoc,const float*c2,const float*s2,const float*cs,float scale,float ceiling,int nc,int ne,int maxe,float seed){
 SmagTerm r={};int count=neoc[cell];if(count<0||count>maxe||!isfinite(seed)||!isfinite(scale)||!isfinite(ceiling))return r;float dudx=0,dudy=0,dvdx=0,dvdy=0;
 for(int j=0;j<count;++j){int off=cell*maxe+j,e=eoc[off];if(e<0||e>=ne)return r;float a=u[k*ne+e],b=v[k*ne+e],x=c2[off],y=s2[off],z=cs[off];if(!isfinite(a)||!isfinite(b)||!isfinite(x)||!isfinite(y)||!isfinite(z))return r;dudx=mpas_sub(mpas_add(dudx,mpas_mul(x,a)),mpas_mul(z,b));dudy=mpas_sub(mpas_add(dudy,mpas_mul(z,a)),mpas_mul(y,b));dvdx=mpas_add(mpas_add(dvdx,mpas_mul(z,a)),mpas_mul(x,b));dvdy=mpas_add(mpas_add(dvdy,mpas_mul(y,a)),mpas_mul(z,b));}
 float diff=mpas_sub(mpas_mul(2.0f,dudx),mpas_mul(2.0f,dvdy)),d12=mpas_add(dudy,dvdx);float strain=mpas_sqrt(mpas_add(mpas_mul(0.25f,mpas_mul(diff,diff)),mpas_mul(d12,d12))),value=mpas_mul(scale,strain);if(!isfinite(strain)||!isfinite(value))return r;
 // Reproduce mpas_min's chosen output including the sign of zero. Exact ties
 // with identical bits choose its first argument, the declared tie convention.
 float selected=mpas_min(value,ceiling);bool smooth=__float_as_uint(selected)==__float_as_uint(value);
 r.smooth=smooth; r.singular=smooth && strain==0.0f;
 if(smooth){if(r.singular)return r;float bstrain=mpas_mul(seed,scale),bq=mpas_mul(bstrain,mpas_div(0.5f,strain));r.bd=mpas_mul(mpas_mul(bq,0.25f),mpas_mul(2.0f,diff));r.b12=mpas_mul(bq,mpas_mul(2.0f,d12));r.bs=mpas_mul(seed,strain);}
 else r.bc=seed;
 r.valid=isfinite(r.bd)&&isfinite(r.b12)&&isfinite(r.bs)&&isfinite(r.bc);return r;
}
__device__ __forceinline__ void sm_edge_derivatives(SmagTerm t,float c2,float s2,float cs,float*bu,float*bv){
 float b11=mpas_mul(2.0f,t.bd),b22=-b11;float ud=mpas_add(mpas_mul(b11,c2),mpas_mul(b22,s2));float vd=mpas_add(-mpas_mul(b11,cs),mpas_mul(b22,cs));
 *bu=mpas_add(ud,mpas_add(mpas_mul(t.b12,cs),mpas_mul(t.b12,cs)));*bv=mpas_add(vd,mpas_add(-mpas_mul(t.b12,s2),mpas_mul(t.b12,c2)));
}
__device__ __forceinline__ void sm_edge_ad64(SmagTerm t,float c2,float s2,float cs,double*bu,double*bv){
 float b11=mpas_mul(2.0f,t.bd),b22=-b11;
 *bu=mpas_f32_to_f64_ieee(mpas_mul(b11,c2))+mpas_f32_to_f64_ieee(mpas_mul(b22,s2))+mpas_f32_to_f64_ieee(mpas_mul(t.b12,cs))+mpas_f32_to_f64_ieee(mpas_mul(t.b12,cs));
 *bv=mpas_f32_to_f64_ieee(-mpas_mul(b11,cs))+mpas_f32_to_f64_ieee(mpas_mul(b22,cs))+mpas_f32_to_f64_ieee(-mpas_mul(t.b12,s2))+mpas_f32_to_f64_ieee(mpas_mul(t.b12,c2));
}
extern "C" __global__ void smagorinsky_v841_f32__tl(const float*u,const float*v,const int*eoc,const int*neoc,const float*c2,const float*s2,const float*cs,const float scale,const float ceiling,const int nlev,const int nc,const int ne,const int maxe,float*kd,const float*du,const float*dv,const float*dscale,const float*dceiling,float*dkd,SmagFailure*failure,unsigned int*branch_record){
 (void)kd;int stride=blockDim.x*gridDim.x;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nlev*nc;i+=stride){int c=i%nc,k=i/nc;SmagTerm t=sm_term(c,k,u,v,eoc,neoc,c2,s2,cs,scale,ceiling,nc,ne,maxe,1.0f);atomicAdd(&branch_record[t.singular?1:0],1u);atomicAdd(&branch_record[2+(t.smooth?0:1)],1u);if(!t.valid||!isfinite(*dscale)||!isfinite(*dceiling)){sm_fail(failure,i, 14, i, 1);continue;}float b=mpas_add(mpas_mul(t.bs,*dscale),mpas_mul(t.bc,*dceiling));
  for(int j=0;j<neoc[c];++j){int off=c*maxe+j,e=eoc[off];float bu,bv;sm_edge_derivatives(t,c2[off],s2[off],cs[off],&bu,&bv);float x=du[k*ne+e],y=dv[k*ne+e];if(!isfinite(x)||!isfinite(y)){sm_fail(failure,i, 1, i, 2);continue;}b=mpas_add(b,mpas_add(mpas_mul(bu,x),mpas_mul(bv,y)));}if(!isfinite(b))sm_fail(failure,i, 14, i, 3);dkd[i]=b;
 }
}
extern "C" __global__ void smagorinsky_v841_f32__ad(const float*u,const float*v,const int*eoc,const int*neoc,const float*c2,const float*s2,const float*cs,const float scale,const float ceiling,const int nlev,const int nc,const int ne,const int maxe,float*kd,const float*bkd_snapshot,float*bkd,float*bu,float*bv,float*bscale,float*bceiling,const int*coe,SmagFailure*failure,unsigned int*branch_record){
 (void)kd;int first=blockDim.x*blockIdx.x+threadIdx.x,stride=blockDim.x*gridDim.x;unsigned long long ncell=(unsigned long long)nlev*nc,nedge=(unsigned long long)nlev*ne;
 for(int i=first;i<nlev*ne;i+=stride){int e=i%ne,k=i/ne,c0=coe[2*e],c1=coe[2*e+1];unsigned long long slot=ncell+i;double x=mpas_f32_to_f64_ieee(bu[i]),y=mpas_f32_to_f64_ieee(bv[i]);if(c0<0||c0>=nc||c1<0||c1>=nc||c0==c1||!isfinite(x)||!isfinite(y)){sm_fail(failure,slot, 1, i, 4);continue;}
  for(int side=0;side<2;++side){int c=side?c1:c0;SmagTerm t=sm_term(c,k,u,v,eoc,neoc,c2,s2,cs,scale,ceiling,nc,ne,maxe,bkd_snapshot[k*nc+c]);if(!t.valid){sm_fail(failure,slot, 14, k*nc+c, 5);continue;}for(int j=0;j<neoc[c];++j){int off=c*maxe+j;if(eoc[off]==e){double a,b;sm_edge_ad64(t,c2[off],s2[off],cs[off],&a,&b);x+=a;y+=b;}}}float out_x=mpas_f64_to_f32_ieee(x),out_y=mpas_f64_to_f32_ieee(y);if(!isfinite(out_x)||!isfinite(out_y))sm_fail(failure,slot, 1, i, 6);bu[i]=out_x;bv[i]=out_y;
 }
 for(int i=first;i<nlev*nc;i+=stride){int c=i%nc,k=i/nc;SmagTerm t=sm_term(c,k,u,v,eoc,neoc,c2,s2,cs,scale,ceiling,nc,ne,maxe,1.0f);atomicAdd(&branch_record[t.singular?1:0],1u);atomicAdd(&branch_record[2+(t.smooth?0:1)],1u);if(!t.valid||!isfinite(bkd_snapshot[i]))sm_fail(failure,i, 14, i, 7);bkd[i]=0;}
 __shared__ double partial_scale[1024],partial_ceil[1024];if(blockIdx.x==0){unsigned long long slot=ncell+nedge+threadIdx.x;double a=0,b=0;for(int i=threadIdx.x;i<nlev*nc;i+=blockDim.x){int c=i%nc,k=i/nc;SmagTerm t=sm_term(c,k,u,v,eoc,neoc,c2,s2,cs,scale,ceiling,nc,ne,maxe,bkd_snapshot[i]);if(!t.valid){sm_fail(failure,slot, 14, i, 8);continue;}a+=mpas_f32_to_f64_ieee(t.bs);b+=mpas_f32_to_f64_ieee(t.bc);}partial_scale[threadIdx.x]=a;partial_ceil[threadIdx.x]=b;__syncthreads();if(threadIdx.x==0){a=mpas_f32_to_f64_ieee(*bscale);b=mpas_f32_to_f64_ieee(*bceiling);for(int j=0;j<blockDim.x;++j){a+=partial_scale[j];b+=partial_ceil[j];}float x=mpas_f64_to_f32_ieee(a),y=mpas_f64_to_f32_ieee(b);if(!isfinite(x)||!isfinite(y))sm_fail(failure,slot, 8, 0, 9);*bscale=x;*bceiling=y;}}
}
