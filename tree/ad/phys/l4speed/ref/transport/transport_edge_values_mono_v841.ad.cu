// SPDX-License-Identifier: Apache-2.0
// Hand rule for cuda_transport_v841.py:236, including both reconstruction orders.
#include "helpers.cuh"
#define HEX_AD_TRACE_BASE 6
#include "trace_helpers.cuh"
__device__ __forceinline__ bool fin(float x){return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u;}
struct D {float x,d;};
__device__ __forceinline__ D da(D a,D b,unsigned int *rec){return {ad_trace_add(a.x,b.x,rec),mpas_add(a.d,b.d)};}
__device__ __forceinline__ D dm(D a,D b,unsigned int *rec){return {ad_trace_mul(a.x,b.x,rec),mpas_add(mpas_mul(a.d,b.x),mpas_mul(a.x,b.d))};}
__device__ __forceinline__ float weight(float c,float s,float a,float t){return mpas_add(a,mpas_mul(mpas_mul(c,s),t));}
#if !defined(AD_AD_ONLY)
extern "C" __global__ void transport_edge_values_mono_v841__tl(
 int ntracers,int nlev,int ncells,int nedges,int width,float coefficient,const float *stage,const float *velocity,
 const float *adv,const float *adv3,const int *n_adv,const int *adv_cells,const float *edge_values,
 const float *d_coefficient,const float *d_stage,const float *d_velocity,const float *d_adv,const float *d_adv3,
 float *d_edge_values,Failure *failure,unsigned int *branch_record)
{
 (void)edge_values;int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=nedges)return;int count=n_adv[e];
 unsigned int branch_local[44]={0};
 if(count<0){ad_fail(failure,0);return;}if(count>width){ad_fail(failure,27);return;}for(int s=0;s<count;++s){int j=ADV(e,s,width),c=adv_cells[j];if(c<0){ad_fail(failure,1);return;}if(c>=ncells){ad_fail(failure,28);return;}}
 {if(!fin(coefficient))ad_fail(failure,2);if(!fin(*d_coefficient))ad_fail(failure,3);}
 for(int k=0;k<nlev;++k){int v=E2(k,e,nedges);float vel=velocity[v],sg=mpas_copysign(1.0f,vel);{if(!fin(vel))ad_fail(failure,4);if(!fin(d_velocity[v]))ad_fail(failure,5);}
  if(count==10)ad_branch(branch_local,2,(__float_as_uint(vel)&0x80000000u)!=0u);
  else for(int s=0;s<count;++s)ad_branch(branch_local,5,(__float_as_uint(vel)&0x80000000u)!=0u);
  for(int q=0;q<ntracers;++q){D value={0.0f,0.0f};
   for(int s=0;s<count;++s){int j=ADV(e,s,width),c=adv_cells[j],i=Q3(q,k,c,nlev,ncells);{if(!fin(adv[j]))ad_fail(failure,6);if(!fin(d_adv[j]))ad_fail(failure,7);if(!fin(adv3[j]))ad_fail(failure,8);if(!fin(d_adv3[j]))ad_fail(failure,9);if(!fin(stage[i]))ad_fail(failure,10);if(!fin(d_stage[i]))ad_fail(failure,11);}
    // The general forward computes its scalar weight once per level and
    // stencil slot, outside the tracer loop. Count that helper only once.
    unsigned int *weight_record=(count==10||q==0)?branch_local:nullptr;
    D w=da({adv[j],d_adv[j]},dm(dm({coefficient,*d_coefficient},{sg,0.0f},weight_record),{adv3[j],d_adv3[j]},weight_record),weight_record);D st={stage[i],d_stage[i]};
    if(count==10){D z=dm(w,st,branch_local);value=(s==0)?z:da(value,z,branch_local);}else {D sw=dm({vel,d_velocity[v]},w,weight_record);value=da(value,dm(sw,st,branch_local),branch_local);}
   }
   if(count==10)value=dm({vel,d_velocity[v]},value,branch_local);
   int out=QE3(q,k,e,nlev,nedges);{if(!fin(value.x))ad_fail(failure,12);if(!fin(value.d))ad_fail(failure,13);}d_edge_values[out]=value.d;
  }
 }
 ad_branch_flush(branch_record,branch_local,22);
}
#endif
#if !defined(AD_TL_ONLY)
extern "C" __global__ void transport_edge_values_mono_v841__ad(
 int ntracers,int nlev,int ncells,int nedges,int width,float coefficient,const float *stage,const float *velocity,
 const float *adv,const float *adv3,const int *n_adv,const int *adv_cells,float *b_edge_values,
 float *b_coefficient,float *b_stage,float *b_velocity,float *b_adv,float *b_adv3,Failure *failure)
{
 int thread=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 for(int e=thread;e<nedges;e+=stride){int count=n_adv[e];if(count<0){ad_fail(failure,14);continue;}if(count>width){ad_fail(failure,29);continue;}bool valid=true;for(int s=0;s<count;++s){int j=ADV(e,s,width),c=adv_cells[j];if(c<0){ad_fail(failure,15);valid=false;}if(c>=ncells){ad_fail(failure,30);valid=false;}}if(!valid)continue;
  if(!fin(coefficient))ad_fail(failure,16);
  for(int k=0;k<nlev;++k){int vi=E2(k,e,nedges);float vel=velocity[vi],sg=mpas_copysign(1.0f,vel);double bv=(double)b_velocity[vi];if(!fin(vel))ad_fail(failure,17);
   for(int q=0;q<ntracers;++q){int o=QE3(q,k,e,nlev,nedges);float bo=b_edge_values[o],v=0.0f;if(!fin(bo))ad_fail(failure,18);
    for(int s=0;s<count;++s){int j=ADV(e,s,width),i=Q3(q,k,adv_cells[j],nlev,ncells);float w=weight(coefficient,sg,adv[j],adv3[j]),x=stage[i];{if(!fin(adv[j]))ad_fail(failure,19);if(!fin(adv3[j]))ad_fail(failure,20);if(!fin(x))ad_fail(failure,21);}
     float part=mpas_mul(w,x);v=(count==10&&s==0)?part:mpas_add(v,part);
    }
    bv+=(double)mpas_mul(bo,v);
   }
   b_velocity[vi]=(float)bv;if(!fin(b_velocity[vi]))ad_fail(failure,24);
  }
  for(int s=0;s<count;++s){int j=ADV(e,s,width),c=adv_cells[j];double ba=(double)b_adv[j],bt=(double)b_adv3[j];
   for(int k=0;k<nlev;++k){float vel=velocity[E2(k,e,nedges)],sg=mpas_copysign(1.0f,vel);
    for(int q=0;q<ntracers;++q){float bo=b_edge_values[QE3(q,k,e,nlev,nedges)],x=stage[Q3(q,k,c,nlev,ncells)];float bw=(count==10)?mpas_mul(mpas_mul(bo,vel),x):mpas_mul(mpas_mul(bo,x),vel);
     ba+=(double)bw;bt+=(double)mpas_mul(mpas_mul(bw,coefficient),sg);
    }
   }
   b_adv[j]=(float)ba;b_adv3[j]=(float)bt;{if(!fin(b_adv[j]))ad_fail(failure,22);if(!fin(b_adv3[j]))ad_fail(failure,23);}
  }
 }
 for(int c=thread;c<ncells;c+=stride)for(int q=0;q<ntracers;++q)for(int base=0;base<nlev;base+=64){
  double bs[64];int levels=min(64,nlev-base);for(int l=0;l<levels;++l)bs[l]=(double)b_stage[Q3(q,base+l,c,nlev,ncells)];
  for(int e=0;e<nedges;++e){int count=n_adv[e];if(count<0||count>width)continue;
   for(int s=0;s<count;++s){int j=ADV(e,s,width);if(adv_cells[j]!=c)continue;
    for(int l=0;l<levels;++l){int k=base+l;float vel=velocity[E2(k,e,nedges)],sg=mpas_copysign(1.0f,vel),w=weight(coefficient,sg,adv[j],adv3[j]);
     int o=QE3(q,k,e,nlev,nedges);bs[l]+=(double)((count==10)?mpas_mul(mpas_mul(b_edge_values[o],vel),w):mpas_mul(b_edge_values[o],mpas_mul(vel,w)));
    }
   }
  }
  for(int l=0;l<levels;++l){int i=Q3(q,base+l,c,nlev,ncells);b_stage[i]=(float)bs[l];if(!fin(b_stage[i]))ad_fail(failure,25);}
 }
 if(thread==0){double sum=(double)*b_coefficient;for(int e=0;e<nedges;++e){int count=n_adv[e];if(count<0||count>width)continue;
  for(int k=0;k<nlev;++k){float vel=velocity[E2(k,e,nedges)],sg=mpas_copysign(1.0f,vel);for(int q=0;q<ntracers;++q){int o=QE3(q,k,e,nlev,nedges);float bo=b_edge_values[o];
   for(int s=0;s<count;++s){int j=ADV(e,s,width),c=adv_cells[j];if(c<0||c>=ncells)continue;float x=stage[Q3(q,k,c,nlev,ncells)],bw=(count==10)?mpas_mul(mpas_mul(bo,vel),x):mpas_mul(mpas_mul(bo,x),vel);sum+=(double)mpas_mul(mpas_mul(bw,sg),adv3[j]);}
  }}
 }*b_coefficient=(float)sum;if(!fin(*b_coefficient))ad_fail(failure,26);}
}
#endif

// Runs after all reverse readers on the same stream.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void transport_edge_values_mono_v841__ad_clear(int ntracers, int nlev, int nedges, float *b_edge_values)
{
 int c = blockIdx.x * blockDim.x + threadIdx.x;
 if (c < nedges) for (int q=0; q<ntracers; ++q) for (int k=0; k<nlev; ++k) { int i=QE3(q,k,c,nlev,nedges); b_edge_values[i] = 0.0f; }
}
#endif
