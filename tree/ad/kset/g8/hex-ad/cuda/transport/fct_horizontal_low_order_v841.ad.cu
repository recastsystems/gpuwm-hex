// SPDX-License-Identifier: Apache-2.0
// Hand rule for cuda_transport_v841.py:321. See the spec for the two reverse phases.
#include "helpers.cuh"
#define HEX_AD_TRACE_BASE 4
#include "trace_helpers.cuh"
__device__ __forceinline__ bool fin(float x){return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u;}
struct D {float x,d;};
__device__ __forceinline__ D da(D a,D b,unsigned int *rec){return {ad_trace_add(a.x,b.x,rec),mpas_add(a.d,b.d)};}
__device__ __forceinline__ D ds(D a,D b,unsigned int *rec){return {ad_trace_sub(a.x,b.x,rec),mpas_sub(a.d,b.d)};}
__device__ __forceinline__ D dm(D a,D b,unsigned int *rec){return {ad_trace_mul(a.x,b.x,rec),mpas_add(mpas_mul(a.d,b.x),mpas_mul(a.x,b.d))};}
__device__ __forceinline__ D dplus(D a,unsigned int *rec){return {ad_trace_max(0.0f,a.x,rec),ad_pos(a.x)?a.d:0.0f};}
__device__ __forceinline__ D dminus(D a,unsigned int *rec){return {ad_trace_min(0.0f,a.x,rec),ad_neg(a.x)?a.d:0.0f};}
#if !defined(AD_AD_ONLY)
extern "C" __global__ void fct_horizontal_low_order_v841__tl(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,const int *n_edges_on_cell,const int *edges_on_cell,
 const float *acoustic_sign,const float *inv_area_cell,const float *upwind_flux,const float *horizontal_residual,
 const float *mass,const float *scale_in,const float *scale_out,
 const float *d_acoustic_sign,const float *d_inv_area_cell,const float *d_upwind_flux,const float *d_horizontal_residual,
 float *d_mass,float *d_scale_in,float *d_scale_out,Failure *failure,unsigned int *branch_record)
{
 // G8: one thread per (level, cell) word column; every (tracer, level, cell) output is independent,
 // so the tangent is bitwise the old one-thread-per-cell loop (grid ceil(ncells*nlev/128) in the spec).
 const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)ncells*nlev)return;
 const int kk=(int)(t/ncells),c=(int)(t-(long long)kk*ncells);int count=n_edges_on_cell[c];
 unsigned int branch_local[40]={0};
 if(count<0){ad_fail(failure,0);return;}if(count>max_edges){ad_fail(failure,39);return;}for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,1);return;}if(e>=nedges){ad_fail(failure,40);return;}}
 {if(!fin(inv_area_cell[c]))ad_fail(failure,2);if(!fin(d_inv_area_cell[c]))ad_fail(failure,3);}
 for(int q=0;q<ntracers;++q){const int k=kk;int i=Q3(q,k,c,nlev,ncells);
  {if(!fin(mass[i]))ad_fail(failure,4);if(!fin(scale_in[i]))ad_fail(failure,5);if(!fin(scale_out[i]))ad_fail(failure,6);if(!fin(d_mass[i]))ad_fail(failure,7);if(!fin(d_scale_in[i]))ad_fail(failure,8);if(!fin(d_scale_out[i]))ad_fail(failure,9);}
  D cm={mass[i],d_mass[i]},in={scale_in[i],d_scale_in[i]},out={scale_out[i],d_scale_out[i]},area={inv_area_cell[c],d_inv_area_cell[c]};
  for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],h=QE3(q,k,e,nlev,nedges);
   {if(!fin(acoustic_sign[j]))ad_fail(failure,10);if(!fin(d_acoustic_sign[j]))ad_fail(failure,11);if(!fin(upwind_flux[h]))ad_fail(failure,12);if(!fin(d_upwind_flux[h]))ad_fail(failure,13);if(!fin(horizontal_residual[h]))ad_fail(failure,14);if(!fin(d_horizontal_residual[h]))ad_fail(failure,15);}
   D sg={-acoustic_sign[j],-d_acoustic_sign[j]},u={upwind_flux[h],d_upwind_flux[h]},r={horizontal_residual[h],d_horizontal_residual[h]},negsg={-sg.x,-sg.d};
   cm=da(cm,dm(dm(sg,u,branch_local),area,branch_local),branch_local);D sr=dm(negsg,r,branch_local);ad_branch(branch_local,2,ad_pos(sr.x));ad_branch(branch_local,3,ad_neg(sr.x));out=ds(out,dm(dplus(sr,branch_local),area,branch_local),branch_local);in=ds(in,dm(dminus(sr,branch_local),area,branch_local),branch_local);
  }
  {if(!fin(cm.x))ad_fail(failure,16);if(!fin(cm.d))ad_fail(failure,17);if(!fin(in.x))ad_fail(failure,18);if(!fin(in.d))ad_fail(failure,19);if(!fin(out.x))ad_fail(failure,20);if(!fin(out.d))ad_fail(failure,21);}
  d_mass[i]=cm.d;d_scale_in[i]=in.d;d_scale_out[i]=out.d;
 }
 ad_branch_flush(branch_record,branch_local,20);
}
#endif
#if !defined(AD_TL_ONLY)
extern "C" __global__ void fct_horizontal_low_order_v841__ad(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,const int *n_edges_on_cell,const int *edges_on_cell,
 const float *acoustic_sign,const float *inv_area_cell,const float *upwind_flux,const float *horizontal_residual,
 float *b_mass_out,float *b_scale_in_out,float *b_scale_out_out,
 float *b_acoustic_sign,float *b_inv_area_cell,float *b_upwind_flux,float *b_horizontal_residual,
 float *b_mass_in,float *b_scale_in_in,float *b_scale_out_in,int phase,Failure *failure)
{
 int thread=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 if(phase==1){int c=thread;if(c<ncells)for(int q=0;q<ntracers;++q)for(int k=0;k<nlev;++k){int i=Q3(q,k,c,nlev,ncells);b_mass_out[i]=0.0f;b_scale_in_out[i]=0.0f;b_scale_out_out[i]=0.0f;}return;}
 if(phase!=0){if(thread==0)ad_fail(failure,22);return;}
 for(int c=thread;c<ncells;c+=stride){int count=n_edges_on_cell[c];if(count<0){ad_fail(failure,23);continue;}if(count>max_edges){ad_fail(failure,41);continue;}
  bool valid=true;for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,24);valid=false;}if(e>=nedges){ad_fail(failure,42);valid=false;}}if(!valid)continue;
  double ba=(double)b_inv_area_cell[c]; if(!fin(inv_area_cell[c]))ad_fail(failure,25);
  for(int q=0;q<ntracers;++q)for(int k=0;k<nlev;++k){int i=Q3(q,k,c,nlev,ncells);float bm=b_mass_out[i],bi=b_scale_in_out[i],bo=b_scale_out_out[i];
   {if(!fin(bm))ad_fail(failure,26);if(!fin(bi))ad_fail(failure,27);if(!fin(bo))ad_fail(failure,28);}
   if(b_mass_in!=b_mass_out)b_mass_in[i]=mpas_add(b_mass_in[i],bm);
   if(b_scale_in_in!=b_scale_in_out)b_scale_in_in[i]=mpas_add(b_scale_in_in[i],bi);
   if(b_scale_out_in!=b_scale_out_out)b_scale_out_in[i]=mpas_add(b_scale_out_in[i],bo);
   for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],h=QE3(q,k,e,nlev,nedges);float a=acoustic_sign[j],u=upwind_flux[h],r=horizontal_residual[h],area=inv_area_cell[c];
    {if(!fin(a))ad_fail(failure,29);if(!fin(u))ad_fail(failure,30);if(!fin(r))ad_fail(failure,31);}
    float sr=mpas_mul(-(-a),r),plus=mpas_max(0.0f,sr),minus=mpas_min(0.0f,sr);
    ba+=(double)mpas_mul(bm,mpas_mul(-a,u))-(double)mpas_mul(bo,plus)-(double)mpas_mul(bi,minus);
   }
   {if(!fin(b_mass_in[i]))ad_fail(failure,33);if(!fin(b_scale_in_in[i]))ad_fail(failure,34);if(!fin(b_scale_out_in[i]))ad_fail(failure,35);}
  }
  for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];float a=acoustic_sign[j],area=inv_area_cell[c];double bs=(double)b_acoustic_sign[j];
   for(int q=0;q<ntracers;++q)for(int k=0;k<nlev;++k){int i=Q3(q,k,c,nlev,ncells),h=QE3(q,k,e,nlev,nedges);float u=upwind_flux[h],r=horizontal_residual[h],sr=mpas_mul(a,r);
    float bsr=mpas_mul(-area,ad_pos(sr)?b_scale_out_out[i]:(ad_neg(sr)?b_scale_in_out[i]:0.0f));
    bs-=(double)mpas_mul(mpas_mul(b_mass_out[i],area),u);bs+=(double)mpas_mul(bsr,r);
   }
   b_acoustic_sign[j]=(float)bs;if(!fin(b_acoustic_sign[j]))ad_fail(failure,32);
  }
  b_inv_area_cell[c]=(float)ba;if(!fin(b_inv_area_cell[c]))ad_fail(failure,36);
 }
}

// G8 edge phase (was: each edge thread scanned every cell and slot, O(nedges*ncells*max_edges)
// per tracer and level block). One thread per (tracer, level, edge) word now walks only the
// cells owning the edge, from the edge_to_slots CSR (cell then slot ascending, the order the
// old scan met them), so b_upwind_flux and b_horizontal_residual are bitwise the old gather.
extern "C" __global__ void fct_horizontal_low_order_v841__ad_edges(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,
 const float *acoustic_sign,const float *inv_area_cell,const float *horizontal_residual,
 const float *b_mass_out,const float *b_scale_in_out,const float *b_scale_out_out,
 float *b_upwind_flux,float *b_horizontal_residual,
 const int *edge_offsets,const int *edge_cell_slots,Failure *failure)
{
 const long long h=(long long)blockIdx.x*blockDim.x+threadIdx.x,nh=(long long)ntracers*nlev*nedges;if(h>=nh)return;
 const int e=(int)(h%nedges);const long long qk=h/nedges;const int k=(int)(qk%nlev),q=(int)(qk/nlev);
 double bu=(double)b_upwind_flux[h],br=(double)b_horizontal_residual[h];
 const int p1=edge_offsets[e+1];
 for(int p=edge_offsets[e];p<p1;++p){int j=edge_cell_slots[p],c=j/max_edges;float a=acoustic_sign[j],area=inv_area_cell[c];
  int i=Q3(q,k,c,nlev,ncells);float sr=mpas_mul(a,horizontal_residual[h]);
  float bsr=mpas_mul(-area,ad_pos(sr)?b_scale_out_out[i]:(ad_neg(sr)?b_scale_in_out[i]:0.0f));
  bu+=(double)mpas_mul(mpas_mul(b_mass_out[i],-a),area);br+=(double)mpas_mul(bsr,a);
 }
 b_upwind_flux[h]=(float)bu;b_horizontal_residual[h]=(float)br;
 {if(!fin(b_upwind_flux[h]))ad_fail(failure,37);if(!fin(b_horizontal_residual[h]))ad_fail(failure,38);}
}
#endif
