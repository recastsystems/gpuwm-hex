// SPDX-License-Identifier: Apache-2.0
// Reverse gathers over sorted unique edge lists built from fixed topology.
using AdFail = int;
// Integer branch counts are warp-aggregated and do not affect field reductions.
__device__ __forceinline__ void record_sign(unsigned int *record,int slot,unsigned int sign) {
 unsigned int live=__activemask(),negative=__ballot_sync(live,sign!=0);
 if ((threadIdx.x&31)==(__ffs((int)live)-1)) {
  atomicAdd(&record[2*slot],(unsigned int)__popc(live&~negative));
  atomicAdd(&record[2*slot+1],(unsigned int)__popc(negative));
 }
}

__device__ __forceinline__ bool finite_f32(float x) { return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u; }
__device__ __forceinline__ void refuse(AdFail *f,int slot,unsigned long long field,unsigned long long element) { atomicExch(&f[slot], 1); }
struct Terms {float value,er,dr,drs,dc;bool good;};
__device__ __forceinline__ Terms terms(int ncells,int nedges,int width,int rk,int k,int e,float coefficient,
 const float *theta,const float *saved,const float *ru,const float *rus,const float *dv,
 const int *cells,const float *adv,const float *adv3,const int *nadv,const int *ac) {
 int i=k*nedges+e;float r=ru[i],s=(__float_as_uint(r)&0x80000000u)?-1.0f:1.0f;
 Terms t={0,0,0,0,0,true};t.good=finite_f32(r)&&finite_f32(coefficient);
 for(int j=0;j<nadv[e];++j) {
  int a=e*width+j,c=ac[a];float x=theta[k*ncells+c];
  float w=adv[a]+coefficient*s*adv3[a];t.er+=w*x;t.dc+=s*adv3[a]*x;
  t.good=t.good&&finite_f32(x)&&finite_f32(adv[a])&&finite_f32(adv3[a])&&finite_f32(w);
 }
 t.value=r*t.er;t.dr=t.er;t.dc*=r;
 if(rk>1) {
  int c0=cells[2*e],c1=cells[2*e+1];float a=saved[k*ncells+c0],b=saved[k*ncells+c1];
  float m=0.5f*dv[e]*(a+b);t.value+=m*(rus[i]-r);t.dr-=m;t.drs=m;
  t.good=t.good&&finite_f32(a)&&finite_f32(b)&&finite_f32(dv[e])&&finite_f32(rus[i])&&finite_f32(m);
 }
 t.good=t.good&&finite_f32(t.value)&&finite_f32(t.er)&&finite_f32(t.dc)&&finite_f32(t.dr)&&finite_f32(t.drs);
 return t;
}
extern "C" __global__ void theta_edge_flux_f32__tl(int nlev,int ncells,int nedges,int width,int rk_step,float coefficient,
 const float *theta,const float *theta_saved,const float *ru,const float *ru_saved,const float *dv_edge,
 const int *cells_on_edge,const float *adv,const float *adv3,const int *n_adv,const int *adv_cells,
 const float *d_coefficient,const float *d_theta,const float *d_theta_saved,const float *d_ru,const float *d_ru_saved,
 float *d_flux,AdFail *fail, unsigned int *branch_record) {
 int nout=nlev*nedges;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nout;i+=blockDim.x*gridDim.x) {
  int k=i/nedges,e=i%nedges;float sign=(__float_as_uint(ru[i])&0x80000000u)?-1.0f:1.0f;
  Terms t=terms(ncells,nedges,width,rk_step,k,e,coefficient,theta,theta_saved,ru,ru_saved,dv_edge,cells_on_edge,adv,adv3,n_adv,adv_cells);
        record_sign(branch_record,0,__float_as_uint(ru[i])>>31);
  float v=t.dr*d_ru[i]+t.drs*d_ru_saved[i]+t.dc*(*d_coefficient);
  bool good=t.good&&finite_f32(d_ru[i])&&finite_f32(d_ru_saved[i])&&finite_f32(*d_coefficient);
  for(int j=0;j<n_adv[e];++j) {int a=e*width+j,c=adv_cells[a];
   float w=adv[a]+coefficient*sign*adv3[a];v+=ru[i]*w*d_theta[k*ncells+c];good=good&&finite_f32(d_theta[k*ncells+c]);
  }
  if(rk_step>1) {int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
   v+=0.5f*dv_edge[e]*(ru_saved[i]-ru[i])*(d_theta_saved[k*ncells+c0]+d_theta_saved[k*ncells+c1]);
   good=good&&finite_f32(d_theta_saved[k*ncells+c0])&&finite_f32(d_theta_saved[k*ncells+c1]);
  }
  if(!good||!finite_f32(v))refuse(fail,0,1,i);d_flux[i]=v;
 }
}
// Inverse lists contain sorted unique edges for each cell; adv lists include every
// edge whose active adv_cells slots contain that cell, and incident lists use cells_on_edge.
extern "C" __global__ void theta_edge_flux_f32__ad(int nlev,int ncells,int nedges,int width,int rk_step,float coefficient,
 const float *theta,const float *theta_saved,const float *ru,const float *ru_saved,const float *dv_edge,
 const int *cells_on_edge,const float *adv,const float *adv3,const int *n_adv,const int *adv_cells,
 const int *adv_offsets,const int *adv_edges,const int *cell_offsets,const int *cell_edges,
 const float *b_flux, float *b_flux_consumed,float *b_coefficient,float *b_theta,float *b_theta_saved,float *b_ru,float *b_ru_saved,AdFail *fail, unsigned int *branch_record) {
 int nout=nlev*nedges,nstate=nlev*ncells,total=nout>nstate?nout:nstate;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=blockDim.x*gridDim.x) {
        // b_flux is the immutable pre-adjoint snapshot; this is the disjoint incoming seed.
        if (i < nout) b_flux_consumed[i] = 0.0f;
  if(i<nout) {int k=i/nedges,e=i%nedges;
   Terms t=terms(ncells,nedges,width,rk_step,k,e,coefficient,theta,theta_saved,ru,ru_saved,dv_edge,cells_on_edge,adv,adv3,n_adv,adv_cells);
        record_sign(branch_record,0,__float_as_uint(ru[i])>>31);
   if(!t.good||!finite_f32(b_flux[i]))refuse(fail,1,5,i);
   double a=(double)b_ru[i]+(double)(t.dr*b_flux[i]),b=(double)b_ru_saved[i]+(double)(t.drs*b_flux[i]);
   if(!finite_f32(b_ru[i])||!finite_f32(b_ru_saved[i])||!finite_f32(a)||!finite_f32(b)||(b_ru==b_ru_saved&&!finite_f32(a+b-(double)b_ru[i])))refuse(fail,2,3,i);
   if(b_ru==b_ru_saved) {b_ru[i]=a+b-(double)b_ru[i];} else {b_ru[i]=a;b_ru_saved[i]=b;}
  }
  if(i<nstate) {int k=i/ncells,c=i%ncells;double a=(double)b_theta[i],b=(double)b_theta_saved[i];
   for(int j=adv_offsets[c];j<adv_offsets[c+1];++j) {int e=adv_edges[j],z=k*nedges+e;
    float sign=(__float_as_uint(ru[z])&0x80000000u)?-1.0f:1.0f;
    for(int slot=0;slot<n_adv[e];++slot) {int q=e*width+slot;if(adv_cells[q]==c)
      a+=ru[z]*(adv[q]+coefficient*sign*adv3[q])*b_flux[z];}
   }
   if(rk_step>1) for(int j=cell_offsets[c];j<cell_offsets[c+1];++j) {int e=cell_edges[j],z=k*nedges+e;
    int count=(cells_on_edge[2*e]==c)+(cells_on_edge[2*e+1]==c);
    b+=0.5f*(float)count*dv_edge[e]*(ru_saved[z]-ru[z])*b_flux[z];
   }
   if(!finite_f32(b_theta[i])||!finite_f32(b_theta_saved[i])||!finite_f32(a)||!finite_f32(b)||(b_theta==b_theta_saved&&!finite_f32(a+b-(double)b_theta[i])))refuse(fail,3,1,i);
   if(b_theta==b_theta_saved) {b_theta[i]=a+b-(double)b_theta[i];} else {b_theta[i]=a;b_theta_saved[i]=b;}
  }
 }
 // Fixed-order block reduction, with f64 partial sums and one f32 output rounding.
 if(blockIdx.x==0) {
  __shared__ double reduction[1024];double sum=0.0;
  for(int z=threadIdx.x;z<nout;z+=blockDim.x) {int k=z/nedges,e=z%nedges;
   Terms t=terms(ncells,nedges,width,rk_step,k,e,coefficient,theta,theta_saved,ru,ru_saved,dv_edge,cells_on_edge,adv,adv3,n_adv,adv_cells);
   sum+=(double)(t.dc*b_flux[z]);
  }
  int lane=threadIdx.x;reduction[lane]=sum;__syncthreads();
  for(int stride=1;stride<blockDim.x;stride*=2) {if(lane%(2*stride)==0 && lane+stride<blockDim.x)reduction[lane]+=reduction[lane+stride];__syncthreads();}
  if(lane==0) {float b=(float)((double)*b_coefficient+reduction[0]);
   if(!finite_f32(*b_coefficient)||!finite_f32(b))refuse(fail,4,6,0);*b_coefficient=b;}
 }
}
