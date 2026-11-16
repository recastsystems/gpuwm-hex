// SPDX-License-Identifier: Apache-2.0
// Exact local tangent and deterministic cell gather for cuda_horizontal.py:141.
#include "edge_fields_from_saved_u_f32.branch.cuh"
#define AD_BRANCH_RECORD() do { \
  __shared__ unsigned int ad_local_counts[22]; \
  for(int ad_i=threadIdx.x;ad_i<22;ad_i+=blockDim.x) ad_local_counts[ad_i]=0; \
  __syncthreads(); \
  ad_branch_replay_edge_fields_from_saved_u_f32(rho,normal_velocity,dc_edge,dv_edge,cells_on_edge,nlev,ncells,nedges,rho_edge,ke_edge,ad_local_counts); \
  __syncthreads(); \
  for(int ad_i=threadIdx.x;ad_i<22;ad_i+=blockDim.x) if(ad_local_counts[ad_i]) atomicAdd(branch_record+ad_i,ad_local_counts[ad_i]); \
} while(0)
#if !defined(AD_BUILD_AD)
extern "C" __global__ void edge_fields_from_saved_u_f32__branch(
    const float* rho,
    const float* normal_velocity,
    const float* dc_edge,
    const float* dv_edge,
    const int* cells_on_edge,
    const int nlev,
    const int ncells,
    const int nedges,
    float* rho_edge,
    float* ke_edge
, unsigned int* branch_record) { AD_BRANCH_RECORD(); }
#endif

__device__ __forceinline__ void edge_bad(int32_t* f,int t,uint32_t code,uint64_t field,uint64_t i){ atomicExch(f + (code == 2u ? 0 : 10), 1); }
#if !defined(AD_BUILD_AD)
extern "C" __global__ void edge_fields_from_saved_u_f32__tl(
 const float* rho,const float* normal_velocity,const float* dc_edge,const float* dv_edge,
 const int* cells_on_edge,int nlev,int ncells,int nedges,float* rho_edge,float* ke_edge,
 const float* d_rho,const float* d_normal_velocity,float* d_rho_edge,float* d_ke_edge,
 int32_t* failure, unsigned int* branch_record) {
 AD_BRANCH_RECORD();
 (void)rho_edge;(void)ke_edge;
 for(int e=blockIdx.x*blockDim.x+threadIdx.x;e<nedges;e+=blockDim.x*gridDim.x){
  int a=cells_on_edge[2*e],b=cells_on_edge[2*e+1];
  for(int k=0;k<nlev;k++) {int t=k*nedges+e;
   if(a<0||a>=ncells||b<0||b>=ncells){edge_bad(failure,t,2,5,e);continue;}
   float ra=rho[k*ncells+a],rb=rho[k*ncells+b],u=normal_velocity[t],area=dc_edge[e]*dv_edge[e];
   float da=d_rho[k*ncells+a],db=d_rho[k*ncells+b],du=d_normal_velocity[t];
   float re=0.5f*(ra+rb),ke=area*(u*u),dr=0.5f*(da+db),dk=(area*(u+u))*du;
   bool bad=false;
   if(!(ra>0.0f)||!(rb>0.0f)||!isfinite(ra)||!isfinite(rb)){atomicExch(failure+1,1);bad=true;}
   if(!isfinite(u)){atomicExch(failure+2,1);bad=true;}
   if(!isfinite(area)){atomicExch(failure+3,1);bad=true;}
   if(!isfinite(re)){atomicExch(failure+4,1);bad=true;}
   if(!isfinite(ke)){atomicExch(failure+5,1);bad=true;}
   if(!isfinite(da)||!isfinite(db)){atomicExch(failure+6,1);bad=true;}
   if(!isfinite(du)){atomicExch(failure+7,1);bad=true;}
   if(!isfinite(dr)){atomicExch(failure+8,1);bad=true;}
   if(!isfinite(dk)){atomicExch(failure+9,1);bad=true;}
   if(bad)continue;
   d_rho_edge[t]=dr; d_ke_edge[t]=dk;
  }
 }
}
#endif
#if !defined(AD_BUILD_TL)
extern "C" __global__ void edge_fields_from_saved_u_f32__ad(
 const float* rho,const float* normal_velocity,const float* dc_edge,const float* dv_edge,
 const int* cells_on_edge,int nlev,int ncells,int nedges,float* rho_edge,float* ke_edge,
 const float* b_rho_edge,const float* b_ke_edge,const int* cell_offsets,const int* cell_edges,
 float* b_rho,float* b_normal_velocity,int32_t* failure, unsigned int* branch_record) {
 AD_BRANCH_RECORD();
 (void)rho_edge;(void)ke_edge;
 int stride=blockDim.x*gridDim.x;
 for(int owner=blockIdx.x*blockDim.x+threadIdx.x;owner<(ncells>nedges?ncells:nedges);owner+=stride) {
  for(int k=0;k<nlev;k++) {
   int slot=k*(ncells>nedges?ncells:nedges)+owner;bool bad=false;
   if(owner<nedges){int t=k*nedges+owner,a=cells_on_edge[2*owner],b=cells_on_edge[2*owner+1];
    if(a<0||a>=ncells||b<0||b>=ncells){atomicExch(failure+0,1);bad=true;}
    else {float ra=rho[k*ncells+a],rb=rho[k*ncells+b],u=normal_velocity[t],ar=dc_edge[owner]*dv_edge[owner];
     float re=0.5f*(ra+rb),ke=ar*(u*u),du=(ar*(u+u))*b_ke_edge[t];double accum=(double)b_normal_velocity[t]+du;
     if(!(ra>0)||!(rb>0)||!isfinite(ra)||!isfinite(rb)){atomicExch(failure+1,1);bad=true;}
     if(!isfinite(u)){atomicExch(failure+2,1);bad=true;}
     if(!isfinite(ar)){atomicExch(failure+3,1);bad=true;}
     if(!isfinite(re)){atomicExch(failure+4,1);bad=true;}
     if(!isfinite(ke)){atomicExch(failure+5,1);bad=true;}
     if(!isfinite(b_rho_edge[t])){atomicExch(failure+10,1);bad=true;}
     if(!isfinite(b_ke_edge[t])){atomicExch(failure+11,1);bad=true;}
     if(!isfinite((float)accum)){atomicExch(failure+12,1);bad=true;}
     if(!bad) b_normal_velocity[t]=accum;
    }
   }
   if(owner<ncells){int t=k*ncells+owner;double sum=b_rho[t];
    if(!isfinite((float)sum)){atomicExch(failure+13,1);bad=true;}
    if(!(rho[t]>0)||!isfinite(rho[t])){atomicExch(failure+1,1);bad=true;}
    for(int j=cell_offsets[owner];j<cell_offsets[owner+1];j++){int e=cell_edges[j];if(e<0||e>=nedges){atomicExch(failure+14,1);bad=true;continue;}float v=b_rho_edge[k*nedges+e];sum+=0.5f*v;if(!isfinite(v)){atomicExch(failure+10,1);bad=true;}if(!isfinite((float)sum)){atomicExch(failure+13,1);bad=true;}}
    if(!bad)b_rho[t]=sum;
   }
   (void)slot;
  }
 }
}
#endif
