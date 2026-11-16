// SPDX-License-Identifier: Apache-2.0
// Hand rule for cuda_transport_v841.py:358, with deterministic gathers.
#include "helpers.cuh"
#define HEX_AD_TRACE_BASE 3
#include "trace_helpers.cuh"
__device__ __forceinline__ bool fin(float x){return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u;}
struct D {float x,d;};
__device__ __forceinline__ D addd(D a,D b,unsigned int *rec){return {ad_trace_add(a.x,b.x,rec),mpas_add(a.d,b.d)};}
__device__ __forceinline__ D subd(D a,D b,unsigned int *rec){return {ad_trace_sub(a.x,b.x,rec),mpas_sub(a.d,b.d)};}
__device__ __forceinline__ D muld(D a,D b,unsigned int *rec){return {ad_trace_mul(a.x,b.x,rec),mpas_add(mpas_mul(a.d,b.x),mpas_mul(a.x,b.d))};}
__device__ __forceinline__ D divd(D a,D b,unsigned int *rec){float x=ad_trace_div(a.x,b.x,rec);return {x,mpas_div(mpas_sub(a.d,mpas_mul(x,b.d)),b.x)};}
__device__ __forceinline__ float fvalue(int q,int k,int c,int nl,int nc,int ne,int me,
 const int *counts,const int *edges,const float *sign,const float *area,const float *rdzw,
 const float *hr,const float *vr,const float *mass){
 float v=mass[Q3(q,k,c,nl,nc)];int count=counts[c];if(count<0||count>me)return __uint_as_float(0x7fc00000u);
 for(int s=0;s<count;++s){int j=CES(c,s,me),e=edges[j];if(e<0||e>=ne)return __uint_as_float(0x7fc00000u);v=mpas_add(v,mpas_mul(mpas_mul(-sign[j],hr[QE3(q,k,e,nl,ne)]),area[c]));}
 return mpas_sub(v,mpas_mul(rdzw[k],mpas_sub(vr[QI3(q,k+1,c,nl,nc)],vr[QI3(q,k,c,nl,nc)])));
}
#if !defined(AD_AD_ONLY)
extern "C" __global__ void fct_finish_v841__tl(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,const int *n_edges_on_cell,const int *edges_on_cell,
 const float *acoustic_sign,const float *inv_area_cell,const float *rdzw,const float *horizontal_residual,
 const float *vertical_residual,const float *target_density,const float *mass,const float *output,
 const float *d_acoustic_sign,const float *d_inv_area_cell,const float *d_rdzw,const float *d_horizontal_residual,
 const float *d_vertical_residual,const float *d_target_density,const float *d_mass,float *d_output,Failure *failure,unsigned int *branch_record)
{
 // G8: one thread per (level, cell); every (tracer, level, cell) output is independent, so the
 // tangent is bitwise the old one-thread-per-cell loop (grid ceil(ncells*nlev/128) in the spec).
 (void)output;const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)ncells*nlev)return;
 const int kk=(int)(t/ncells),c=(int)(t-(long long)kk*ncells);
 unsigned int branch_local[38]={0};
 int count=n_edges_on_cell[c];if(count<0){ad_fail(failure,0);return;}if(count>max_edges){ad_fail(failure,31);return;}
 for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,1);return;}if(e>=nedges){ad_fail(failure,32);return;}}
 {if(!fin(inv_area_cell[c]))ad_fail(failure,2);if(!fin(d_inv_area_cell[c]))ad_fail(failure,3);}
 for(int q=0;q<ntracers;++q){const int k=kk;int i=Q3(q,k,c,nlev,ncells),r=C2(k,c,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);
  if(!fin(target_density[r])){ad_fail(failure,4);continue;}if(!ad_nonzero(target_density[r])){ad_fail(failure,33);continue;}if(!fin(d_target_density[r])){ad_fail(failure,34);continue;}
  {if(!fin(mass[i]))ad_fail(failure,5);if(!fin(d_mass[i]))ad_fail(failure,6);if(!fin(rdzw[k]))ad_fail(failure,7);if(!fin(d_rdzw[k]))ad_fail(failure,8);if(!fin(vertical_residual[lo]))ad_fail(failure,9);if(!fin(vertical_residual[hi]))ad_fail(failure,10);if(!fin(d_vertical_residual[lo]))ad_fail(failure,11);if(!fin(d_vertical_residual[hi]))ad_fail(failure,12);}
  D v={mass[i],d_mass[i]};
  for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],h=QE3(q,k,e,nlev,nedges);
   {if(!fin(acoustic_sign[j]))ad_fail(failure,13);if(!fin(d_acoustic_sign[j]))ad_fail(failure,14);if(!fin(horizontal_residual[h]))ad_fail(failure,15);if(!fin(d_horizontal_residual[h]))ad_fail(failure,16);}
   D a={-acoustic_sign[j],-d_acoustic_sign[j]},b={horizontal_residual[h],d_horizontal_residual[h]},area={inv_area_cell[c],d_inv_area_cell[c]};v=addd(v,muld(muld(a,b,branch_local),area,branch_local),branch_local);
  }
  D dz={rdzw[k],d_rdzw[k]},vhi={vertical_residual[hi],d_vertical_residual[hi]},vlo={vertical_residual[lo],d_vertical_residual[lo]},rho={target_density[r],d_target_density[r]};
  D z=divd(subd(v,muld(dz,subd(vhi,vlo,branch_local),branch_local),branch_local),rho,branch_local);
  ad_branch(branch_local,2,ad_pos(z.x));(void)ad_trace_max(0.0f,z.x,branch_local);
  {if(!fin(z.x))ad_fail(failure,17);if(!fin(z.d))ad_fail(failure,18);}d_output[i]=ad_pos(z.x)?z.d:0.0f;
 }
 ad_branch_flush(branch_record,branch_local,19);
}
#endif
#if !defined(AD_TL_ONLY)
#include "g8_reduce.cuh"
extern "C" __global__ void fct_finish_v841__ad(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,const int *n_edges_on_cell,const int *edges_on_cell,
 const float *acoustic_sign,const float *inv_area_cell,const float *rdzw,const float *horizontal_residual,
 const float *vertical_residual,const float *target_density,const float *mass,float *b_output,
 float *b_acoustic_sign,float *b_inv_area_cell,float *b_rdzw,float *b_horizontal_residual,
 float *b_vertical_residual,float *b_target_density,float *b_mass,Failure *failure)
{
 int thread=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 for(int c=thread;c<ncells;c+=stride){int count=n_edges_on_cell[c];if(count<0){ad_fail(failure,19);continue;}if(count>max_edges){ad_fail(failure,35);continue;}
  bool valid=true;for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,20);valid=false;}if(e>=nedges){ad_fail(failure,36);valid=false;}}if(!valid)continue;
  double ba=(double)b_inv_area_cell[c];
  for(int k=0;k<nlev;++k){int r=C2(k,c,ncells);float den=target_density[r];double brho=(double)b_target_density[r];
   if(!fin(den)){ad_fail(failure,21);continue;}if(!ad_nonzero(den)){ad_fail(failure,37);continue;}
   for(int q=0;q<ntracers;++q){int i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);
    float v=fvalue(q,k,c,nlev,ncells,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,horizontal_residual,vertical_residual,mass);
    float ratio=mpas_div(v,den),bo=b_output[i];if(!fin(v)){ad_fail(failure,22);continue;}if(!fin(ratio)){ad_fail(failure,38);continue;}if(!fin(bo)){ad_fail(failure,39);continue;}if(!ad_pos(ratio))continue;
    float bv=mpas_div(bo,den);brho-=(double)mpas_mul(bo,mpas_div(ratio,den));b_mass[i]=mpas_add(b_mass[i],bv);
    for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],h=QE3(q,k,e,nlev,nedges);
     ba+=(double)mpas_mul(bv,mpas_mul(-acoustic_sign[j],horizontal_residual[h]));
    }
    if(!fin(b_mass[i]))ad_fail(failure,24);
   }
   b_target_density[r]=(float)brho;if(!fin(b_target_density[r]))ad_fail(failure,27);
  }
  for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];double bs=(double)b_acoustic_sign[j];
   for(int k=0;k<nlev;++k)for(int q=0;q<ntracers;++q){int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),h=QE3(q,k,e,nlev,nedges);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
    float v=fvalue(q,k,c,nlev,ncells,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,horizontal_residual,vertical_residual,mass),ratio=mpas_div(v,den);if(!fin(ratio)||!ad_pos(ratio))continue;
    bs-=(double)mpas_mul(mpas_mul(mpas_div(b_output[i],den),inv_area_cell[c]),horizontal_residual[h]);
   }b_acoustic_sign[j]=(float)bs;if(!fin(b_acoustic_sign[j]))ad_fail(failure,23);
  }
  for(int q=0;q<ntracers;++q)for(int face=0;face<=nlev;++face){int p=QI3(q,face,c,nlev,ncells);double bv=(double)b_vertical_residual[p];
   for(int z=0;z<2;++z){int k=face-z;if(k<0||k>=nlev)continue;int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
    float v=fvalue(q,k,c,nlev,ncells,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,horizontal_residual,vertical_residual,mass),ratio=mpas_div(v,den);if(!fin(ratio)||!ad_pos(ratio))continue;
    float term=mpas_mul(mpas_div(b_output[i],den),rdzw[k]);bv+=(z==0?(double)term:-(double)term);
   }
   b_vertical_residual[p]=(float)bv;{if(!fin(b_vertical_residual[p]))ad_fail(failure,25);if(!fin(b_vertical_residual[p]))ad_fail(failure,26);}
  }
  b_inv_area_cell[c]=(float)ba;if(!fin(b_inv_area_cell[c]))ad_fail(failure,28);
 }
}

// G8 edge phase (was: each edge thread scanned every cell and slot, O(nedges*ncells*max_edges)).
// One thread per (tracer, level, edge) word walks only the edge's own cells, from the
// edge_to_slots CSR in the old scan's order: b_horizontal_residual is bitwise the old gather.
extern "C" __global__ void fct_finish_v841__ad_edges(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,const int *n_edges_on_cell,const int *edges_on_cell,
 const float *acoustic_sign,const float *inv_area_cell,const float *rdzw,const float *horizontal_residual,
 const float *vertical_residual,const float *target_density,const float *mass,const float *b_output,
 float *b_horizontal_residual,const int *edge_offsets,const int *edge_cell_slots,Failure *failure)
{
 const long long h=(long long)blockIdx.x*blockDim.x+threadIdx.x,nh=(long long)ntracers*nlev*nedges;if(h>=nh)return;
 const int e=(int)(h%nedges);const long long qk=h/nedges;const int k=(int)(qk%nlev),q=(int)(qk/nlev);
 double bh=(double)b_horizontal_residual[h];
 const int p1=edge_offsets[e+1];
 for(int p=edge_offsets[e];p<p1;++p){int j=edge_cell_slots[p],c=j/max_edges;
  int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
  float v=fvalue(q,k,c,nlev,ncells,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,horizontal_residual,vertical_residual,mass),ratio=mpas_div(v,den);if(!fin(ratio)||!ad_pos(ratio))continue;
  bh+=(double)mpas_mul(mpas_mul(mpas_div(b_output[i],den),-acoustic_sign[j]),inv_area_cell[c]);
 }
 b_horizontal_residual[h]=(float)bh;if(!fin(b_horizontal_residual[h]))ad_fail(failure,29);
}

// G8 b_rdzw (was one thread per level walking ncells*ntracers words): fixed-order two-level
// f64 sum per level. grid = [ceil(ncells*ntracers/G8_CHUNK), nlev], word m = c*ntracers + q.
extern "C" __global__ void fct_finish_v841__ad_rdzw_part(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,const int *n_edges_on_cell,const int *edges_on_cell,
 const float *acoustic_sign,const float *inv_area_cell,const float *rdzw,const float *horizontal_residual,
 const float *vertical_residual,const float *target_density,const float *mass,const float *b_output,
 double *part,Failure *failure)
{
 if(blockDim.x!=G8_THREADS){if(threadIdx.x==0)ad_fail(failure,30);return;}
 const int k=blockIdx.y;const long long nm=(long long)ncells*ntracers;double t=0.0;
 for(int m=0;m<G8_PER_THREAD;++m){long long w=(long long)blockIdx.x*G8_CHUNK+(long long)m*G8_THREADS+threadIdx.x;if(w>=nm||k>=nlev)continue;
  int c=(int)(w/ntracers),q=(int)(w-(long long)c*ntracers);
  int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
  float v=fvalue(q,k,c,nlev,ncells,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,horizontal_residual,vertical_residual,mass),ratio=mpas_div(v,den);if(!fin(ratio)||!ad_pos(ratio))continue;
  t-=(double)mpas_mul(mpas_div(b_output[i],den),mpas_sub(vertical_residual[hi],vertical_residual[lo]));
 }
 const double s=g8_block_sum(t);
 if(threadIdx.x==0)part[(long long)k*gridDim.x+blockIdx.x]=s;
}
extern "C" __global__ void fct_finish_v841__ad_rdzw(int nlev,float *b_rdzw,const double *part,int nb,Failure *failure)
{
 if(blockDim.x!=G8_THREADS){if(threadIdx.x==0)ad_fail(failure,30);return;}
 const int k=blockIdx.x;if(k>=nlev)return;
 const double s=g8_partials_sum(part,k,nb);
 if(threadIdx.x==0){b_rdzw[k]=(float)((double)b_rdzw[k]+s);if(!fin(b_rdzw[k]))ad_fail(failure,30);}
}
#endif

// Runs after all reverse readers on the same stream.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void fct_finish_v841__ad_clear(int ntracers, int nlev, int ncells, float *b_output)
{
 int c = blockIdx.x * blockDim.x + threadIdx.x;
 if (c < ncells) for (int q=0; q<ntracers; ++q) for (int k=0; k<nlev; ++k) { int i=Q3(q,k,c,nlev,ncells); b_output[i] = 0.0f; }
}
#endif
