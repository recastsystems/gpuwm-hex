// SPDX-License-Identifier: Apache-2.0
// Tangent and deterministic gather adjoint for cuda_transport_v841.py:202.
#include "helpers.cuh"
#define HEX_AD_TRACE_BASE 3
#include "trace_helpers.cuh"
__device__ __forceinline__ bool finite_d(float a) { return (__float_as_uint(a)&0x7f800000u)!=0x7f800000u; }
// Accumulate derivatives in f64 to keep cancellation in long reductions from
// overwhelming the f32 dot test.  Primal values retain the original f32 rules.
struct Dual { float x; double d; };
__device__ __forceinline__ Dual dm(Dual a,Dual b,unsigned int *rec){return {ad_trace_mul(a.x,b.x,rec),a.d*(double)b.x+(double)a.x*b.d};}
__device__ __forceinline__ Dual da(Dual a,Dual b,unsigned int *rec){return {ad_trace_add(a.x,b.x,rec),a.d+b.d};}
__device__ __forceinline__ Dual ds(Dual a,Dual b,unsigned int *rec){return {ad_trace_sub(a.x,b.x,rec),a.d-b.d};}

#if !defined(AD_AD_ONLY)
extern "C" __global__ void transport_target_density_v841__tl(
 int nlev,int ncells,int nedges,int max_edges,int advance_density,float dt,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,
 const float *dv_edge,const float *inv_area_cell,const float *velocity,
 const float *vertical_velocity,const float *rdzw,const float *rho_old,
 const float *rho_new,const float *target_density,
 const float *d_dt,const float *d_acoustic_sign,const float *d_dv_edge,
 const float *d_inv_area_cell,const float *d_velocity,const float *d_vertical_velocity,
 const float *d_rdzw,const float *d_rho_old,const float *d_rho_new,
 float *d_target_density,Failure *failure,unsigned int *branch_record)
{
 unsigned int branch_local[38]={0};
 (void)target_density;
 int cell=blockIdx.x*blockDim.x+threadIdx.x;
 if(cell>=ncells)return;
 int count=0;
 if(advance_density){count=n_edges_on_cell[cell];
  if(count<0){ad_fail(failure,0);return;}if(count>max_edges){ad_fail(failure,46);return;}
  for(int s=0;s<count;++s){int e=edges_on_cell[CES(cell,s,max_edges)];if(e<0){ad_fail(failure,1);return;}if(e>=nedges){ad_fail(failure,47);return;}}
 }
 for(int k=0;k<nlev;++k){
  int i=C2(k,cell,ncells);
  if(!advance_density){
   {if(!finite_d(rho_new[i]))ad_fail(failure,2);if(!finite_d(d_rho_new[i]))ad_fail(failure,3);}
   d_target_density[i]=d_rho_new[i];continue;
  }
  {if(!finite_d(dt))ad_fail(failure,4);if(!finite_d(*d_dt))ad_fail(failure,5);}
  {if(!finite_d(inv_area_cell[cell]))ad_fail(failure,6);if(!finite_d(d_inv_area_cell[cell]))ad_fail(failure,7);}
  Dual h={0.0f,0.0f};
  for(int s=0;s<count;++s){int j=CES(cell,s,max_edges),e=edges_on_cell[j],v=E2(k,e,nedges);
   {if(!finite_d(acoustic_sign[j]))ad_fail(failure,8);if(!finite_d(d_acoustic_sign[j]))ad_fail(failure,9);}
   {if(!finite_d(dv_edge[e]))ad_fail(failure,10);if(!finite_d(d_dv_edge[e]))ad_fail(failure,11);}
   {if(!finite_d(velocity[v]))ad_fail(failure,12);if(!finite_d(d_velocity[v]))ad_fail(failure,13);}
   Dual a={-acoustic_sign[j],-d_acoustic_sign[j]},b={velocity[v],d_velocity[v]},c={dv_edge[e],d_dv_edge[e]},d={inv_area_cell[cell],d_inv_area_cell[cell]};
   h=da(h,dm(dm(dm(a,b,branch_local),c,branch_local),d,branch_local),branch_local);
  }
  int upper=C2(k+1,cell,ncells);
  {if(!finite_d(rdzw[k]))ad_fail(failure,14);if(!finite_d(d_rdzw[k]))ad_fail(failure,15);}
  {if(!finite_d(vertical_velocity[i]))ad_fail(failure,16);if(!finite_d(vertical_velocity[upper]))ad_fail(failure,17);if(!finite_d(d_vertical_velocity[i]))ad_fail(failure,18);if(!finite_d(d_vertical_velocity[upper]))ad_fail(failure,19);}
  {if(!finite_d(rho_old[i]))ad_fail(failure,20);if(!finite_d(d_rho_old[i]))ad_fail(failure,21);}
  Dual z={rdzw[k],d_rdzw[k]},w1={vertical_velocity[upper],d_vertical_velocity[upper]},w0={vertical_velocity[i],d_vertical_velocity[i]},t={dt,*d_dt},r={rho_old[i],d_rho_old[i]};
  Dual out=da(r,dm(t,ds(h,dm(z,ds(w1,w0,branch_local),branch_local),branch_local),branch_local),branch_local);
  float derivative=(float)out.d;
  {if(!finite_d(out.x))ad_fail(failure,22);if(!finite_d(derivative))ad_fail(failure,23);}
  d_target_density[i]=derivative;
 }
 ad_branch_flush(branch_record,branch_local,19);
}

#endif

#if !defined(AD_TL_ONLY)
#include "g8_reduce.cuh"
extern "C" __global__ void transport_target_density_v841__ad(
 int nlev,int ncells,int nedges,int max_edges,int advance_density,float dt,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,
 const float *dv_edge,const float *inv_area_cell,const float *velocity,
 const float *vertical_velocity,const float *rdzw,const float *rho_old,
 const float *rho_new,float *b_target_density,float *b_dt,
 float *b_acoustic_sign,float *b_dv_edge,float *b_inv_area_cell,float *b_velocity,
 float *b_vertical_velocity,float *b_rdzw,float *b_rho_old,float *b_rho_new,Failure *failure)
{
 int thread=blockIdx.x*blockDim.x+threadIdx.x, stride=blockDim.x*gridDim.x;
 for(int c=thread;c<ncells;c+=stride){
  int count=0;if(advance_density){count=n_edges_on_cell[c];if(count<0){ad_fail(failure,24);continue;}if(count>max_edges){ad_fail(failure,48);continue;}
   bool valid=true;for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,25);valid=false;}if(e>=nedges){ad_fail(failure,49);valid=false;}}if(!valid)continue;
  }
  double ba=0.0;
  for(int k=0;k<nlev;++k){int i=C2(k,c,ncells);float b=b_target_density[i];
   if(!finite_d(b))ad_fail(failure,26);
   if(!advance_density){if(!finite_d(rho_new[i]))ad_fail(failure,27);b_rho_new[i]=mpas_add(b_rho_new[i],b);if(!finite_d(b_rho_new[i]))ad_fail(failure,28);continue;}
   if(!finite_d(rho_old[i]))ad_fail(failure,29);
   double bh=(double)b*(double)dt; b_rho_old[i]=mpas_add(b_rho_old[i],b);
   int upper=C2(k+1,c,ncells);
   {if(!finite_d(dt))ad_fail(failure,30);if(!finite_d(rdzw[k]))ad_fail(failure,31);if(!finite_d(vertical_velocity[i]))ad_fail(failure,32);if(!finite_d(vertical_velocity[upper]))ad_fail(failure,33);if(!finite_d(inv_area_cell[c]))ad_fail(failure,34);}
   for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],v=E2(k,e,nedges);float sign=-acoustic_sign[j];
    {if(!finite_d(sign))ad_fail(failure,35);if(!finite_d(velocity[v]))ad_fail(failure,36);if(!finite_d(dv_edge[e]))ad_fail(failure,37);}
    float sv=mpas_mul(sign,velocity[v]), svd=mpas_mul(sv,dv_edge[e]);
    ba+=bh*(double)svd;
   }
   if(!finite_d(b_rho_old[i]))ad_fail(failure,38);
  }
  if(advance_density){
   for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];double bs=0.0;
    for(int k=0;k<nlev;++k){int i=C2(k,c,ncells),v=E2(k,e,nedges);bs+=(double)b_target_density[i]*(double)dt*(double)inv_area_cell[c]*(double)dv_edge[e]*(double)velocity[v];}
    b_acoustic_sign[j]=(float)((double)b_acoustic_sign[j]-bs);if(!finite_d(b_acoustic_sign[j]))ad_fail(failure,39);
   }
   for(int k=0;k<=nlev;++k){int i=C2(k,c,ncells);double sum=(double)b_vertical_velocity[i];
    if(k<nlev)sum+=(double)b_target_density[i]*(double)dt*(double)rdzw[k];
    if(k>0)sum-=(double)b_target_density[C2(k-1,c,ncells)]*(double)dt*(double)rdzw[k-1];
    b_vertical_velocity[i]=(float)sum;if(!finite_d(b_vertical_velocity[i]))ad_fail(failure,40);
   }
   b_inv_area_cell[c]=(float)((double)b_inv_area_cell[c]+ba);if(!finite_d(b_inv_area_cell[c]))ad_fail(failure,41);
  }
 }
 (void)nedges;
}

// G8 edge phase (was: every edge thread scanned every cell, O(nedges*ncells), and
// thread 0 alone reduced b_dt over ncells*nlev*edges). One thread per edge now
// walks only the cells that own the edge, from the edge_to_slots CSR (cell
// ascending, slot ascending: the order the old scan met them), so b_velocity and
// b_dv_edge are bitwise the old gather.
extern "C" __global__ void transport_target_density_v841__ad_edges(
 int nlev,int ncells,int nedges,int max_edges,int advance_density,float dt,
 const float *acoustic_sign,const float *dv_edge,const float *inv_area_cell,const float *velocity,
 const float *b_target_density,float *b_dv_edge,float *b_velocity,
 const int *edge_offsets,const int *edge_cell_slots,Failure *failure)
{
 if(!advance_density)return;
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=nedges)return;
 const int p0=edge_offsets[e],p1=edge_offsets[e+1];
 double bd=0.0;
 for(int base=0;base<nlev;base+=16){double bv[16]={0.0};int count_k=min(16,nlev-base);
  for(int p=p0;p<p1;++p){int j=edge_cell_slots[p],c=j/max_edges;float sign=-acoustic_sign[j],area=inv_area_cell[c];
   for(int q=0;q<count_k;++q){int k=base+q,v=E2(k,e,nedges);double bh=(double)b_target_density[C2(k,c,ncells)]*(double)dt;
    bv[q]+=bh*(double)area*(double)dv_edge[e]*(double)sign;
    bd+=bh*(double)area*(double)mpas_mul(sign,velocity[v]);
   }
  }
  for(int q=0;q<count_k;++q){int v=E2(base+q,e,nedges);b_velocity[v]=(float)((double)b_velocity[v]+bv[q]);if(!finite_d(b_velocity[v]))ad_fail(failure,42);}
 }
 b_dv_edge[e]=(float)((double)b_dv_edge[e]+bd);if(!finite_d(b_dv_edge[e]))ad_fail(failure,43);
}

// G8 b_rdzw: per level, a fixed-order two-level f64 sum over cells (was one thread per level walking all cells).
// grid = [ceil(ncells/G8_CHUNK), nlev], block = G8_THREADS.
extern "C" __global__ void transport_target_density_v841__ad_rdzw_part(
 int nlev,int ncells,int advance_density,float dt,const float *vertical_velocity,
 const float *b_target_density,double *part,Failure *failure)
{
 if(!advance_density)return;
 if(blockDim.x!=G8_THREADS){if(threadIdx.x==0)ad_fail(failure,44);return;}
 const int k=blockIdx.y;double t=0.0;
 for(int m=0;m<G8_PER_THREAD;++m){long long c=(long long)blockIdx.x*G8_CHUNK+(long long)m*G8_THREADS+threadIdx.x;if(c>=ncells||k>=nlev)continue;
  int i=C2(k,(int)c,ncells);float dw=mpas_sub(vertical_velocity[C2(k+1,(int)c,ncells)],vertical_velocity[i]);t-=(double)b_target_density[i]*(double)dt*(double)dw;}
 const double s=g8_block_sum(t);
 if(threadIdx.x==0)part[(long long)k*gridDim.x+blockIdx.x]=s;
}
extern "C" __global__ void transport_target_density_v841__ad_rdzw(
 int nlev,int advance_density,float *b_rdzw,const double *part,int nb,Failure *failure)
{
 if(!advance_density)return;
 if(blockDim.x!=G8_THREADS){if(threadIdx.x==0)ad_fail(failure,44);return;}
 const int k=blockIdx.x;if(k>=nlev)return;
 const double s=g8_partials_sum(part,k,nb);
 if(threadIdx.x==0){b_rdzw[k]=(float)((double)b_rdzw[k]+s);if(!finite_d(b_rdzw[k]))ad_fail(failure,44);}
}

// G8 b_dt: fixed-order two-level f64 sum over all ncells*nlev words (was thread 0 alone).
// grid = [ceil(ncells*nlev/G8_CHUNK)], block = G8_THREADS; word w = C2(k,c,ncells).
extern "C" __global__ void transport_target_density_v841__ad_dt_part(
 int nlev,int ncells,int nedges,int max_edges,int advance_density,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,
 const float *dv_edge,const float *inv_area_cell,const float *velocity,
 const float *vertical_velocity,const float *rdzw,const float *b_target_density,double *part,Failure *failure)
{
 if(!advance_density)return;
 if(blockDim.x!=G8_THREADS){if(threadIdx.x==0)ad_fail(failure,45);return;}
 const long long nw=(long long)ncells*nlev;double t=0.0;
 for(int m=0;m<G8_PER_THREAD;++m){long long w=(long long)blockIdx.x*G8_CHUNK+(long long)m*G8_THREADS+threadIdx.x;if(w>=nw)continue;
  int k=(int)(w/ncells),c=(int)(w-(long long)k*ncells);int count=n_edges_on_cell[c];if(count<0||count>max_edges)continue;
  int i=C2(k,c,ncells);float h=0.0f;
  for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0||e>=nedges)continue;float term=mpas_mul(-acoustic_sign[j],velocity[E2(k,e,nedges)]);term=mpas_mul(term,dv_edge[e]);term=mpas_mul(term,inv_area_cell[c]);h=mpas_add(h,term);}
  float z=mpas_mul(rdzw[k],mpas_sub(vertical_velocity[C2(k+1,c,ncells)],vertical_velocity[i]));t+=(double)b_target_density[i]*(double)mpas_sub(h,z);
 }
 const double s=g8_block_sum(t);
 if(threadIdx.x==0)part[blockIdx.x]=s;
}
extern "C" __global__ void transport_target_density_v841__ad_dt(
 int advance_density,float *b_dt,const double *part,int nb,Failure *failure)
{
 if(!advance_density)return;
 if(blockDim.x!=G8_THREADS||gridDim.x!=1){if(threadIdx.x==0)ad_fail(failure,45);return;}
 const double s=g8_partials_sum(part,0,nb);
 if(threadIdx.x==0){*b_dt=(float)((double)*b_dt+s);if(!finite_d(*b_dt))ad_fail(failure,45);}
}
#endif

// Runs after all reverse readers on the same stream.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void transport_target_density_v841__ad_clear(int nlev, int ncells, float *b_target_density)
{
 int c = blockIdx.x * blockDim.x + threadIdx.x;
 if (c < ncells) for (int k=0; k<nlev; ++k) { int i=C2(k,c,ncells); b_target_density[i] = 0.0f; }
}
#endif
