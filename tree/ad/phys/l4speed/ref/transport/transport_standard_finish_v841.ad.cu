// Candidate: f64 derivative arithmetic, f32 derivative storage, f32 primal replay.
// SPDX-License-Identifier: Apache-2.0
// Hand rule for cuda_transport_v841.py:164, using deterministic ownership.
#include "helpers.cuh"
#define HEX_AD_TRACE_BASE 2
#include "trace_helpers.cuh"
__device__ __forceinline__ bool fin(double x){return isfinite(x);}
__device__ __forceinline__ bool fin(float x){return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u;}
struct D {float x;double d;};
__device__ __forceinline__ D da(D a,D b,unsigned int *rec){return {ad_trace_add(a.x,b.x,rec),(a.d+b.d)};}
__device__ __forceinline__ D ds(D a,D b,unsigned int *rec){return {ad_trace_sub(a.x,b.x,rec),(a.d-b.d)};}
__device__ __forceinline__ D dm(D a,D b,unsigned int *rec){return {ad_trace_mul(a.x,b.x,rec),(a.d*(double)b.x+(double)a.x*b.d)};}
__device__ __forceinline__ D dd(D a,D b,unsigned int *rec){float x=ad_trace_div(a.x,b.x,rec);return {x,((a.d-(double)x*b.d)/(double)b.x)};}
__device__ __forceinline__ float horizontal(int q,int k,int c,int nl,int ne,int me,const int *count,const int *edges,const float *sign,const float *vel,const float *ev){
 float h=0.0f;for(int s=0;s<count[c];++s){int j=CES(c,s,me),e=edges[j];h=mpas_add(h,mpas_mul(mpas_mul(-sign[j],vel[E2(k,e,ne)]),ev[QE3(q,k,e,nl,ne)]));}return h;
}
#if !defined(AD_AD_ONLY)
extern "C" __global__ void transport_standard_finish_v841__tl(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,float dt,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,const float *inv_area_cell,
 const float *velocity,const float *edge_values,const float *vertical_flux,const float *rdzw,
 const float *old,const float *rho_old,const float *target_density,const float *source,const float *output,
 const float *d_dt,const float *d_acoustic_sign,const float *d_inv_area_cell,const float *d_velocity,
 const float *d_edge_values,const float *d_vertical_flux,const float *d_rdzw,const float *d_old,
 const float *d_rho_old,const float *d_target_density,const float *d_source,float *d_output,Failure *failure,unsigned int *branch_record)
{
 unsigned int branch_local[36]={0};
 (void)output;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=ncells)return;int count=n_edges_on_cell[c];
 if(count<0){ad_fail(failure,0);return;}if(count>max_edges){ad_fail(failure,52);return;}for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,1);return;}if(e>=nedges){ad_fail(failure,53);return;}}
 {if(!fin(dt))ad_fail(failure,2);if(!fin(*d_dt))ad_fail(failure,3);if(!fin(inv_area_cell[c]))ad_fail(failure,4);if(!fin(d_inv_area_cell[c]))ad_fail(failure,5);}
 for(int q=0;q<ntracers;++q)for(int k=0;k<nlev;++k){int i=Q3(q,k,c,nlev,ncells),r=C2(k,c,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);
  if(!fin(target_density[r])){ad_fail(failure,6);continue;}if(!ad_nonzero(target_density[r])){ad_fail(failure,54);continue;}if(!fin(d_target_density[r])){ad_fail(failure,55);continue;}
  {if(!fin(old[i]))ad_fail(failure,7);if(!fin(d_old[i]))ad_fail(failure,8);if(!fin(rho_old[r]))ad_fail(failure,9);if(!fin(d_rho_old[r]))ad_fail(failure,10);if(!fin(source[i]))ad_fail(failure,11);if(!fin(d_source[i]))ad_fail(failure,12);if(!fin(rdzw[k]))ad_fail(failure,13);if(!fin(d_rdzw[k]))ad_fail(failure,14);if(!fin(vertical_flux[lo]))ad_fail(failure,15);if(!fin(vertical_flux[hi]))ad_fail(failure,16);if(!fin(d_vertical_flux[lo]))ad_fail(failure,17);if(!fin(d_vertical_flux[hi]))ad_fail(failure,18);}
  D h={0.0f,0.0f};for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],v=E2(k,e,nedges),p=QE3(q,k,e,nlev,nedges);
   {if(!fin(acoustic_sign[j]))ad_fail(failure,19);if(!fin(d_acoustic_sign[j]))ad_fail(failure,20);if(!fin(velocity[v]))ad_fail(failure,21);if(!fin(d_velocity[v]))ad_fail(failure,22);if(!fin(edge_values[p]))ad_fail(failure,23);if(!fin(d_edge_values[p]))ad_fail(failure,24);}
   D a={-acoustic_sign[j],-d_acoustic_sign[j]},b={velocity[v],d_velocity[v]},evalue={edge_values[p],d_edge_values[p]};h=da(h,dm(dm(a,b,branch_local),evalue,branch_local),branch_local);}
  D area={inv_area_cell[c],d_inv_area_cell[c]},src={source[i],d_source[i]},dz={rdzw[k],d_rdzw[k]},vhi={vertical_flux[hi],d_vertical_flux[hi]},vlo={vertical_flux[lo],d_vertical_flux[lo]},ov={old[i],d_old[i]},ro={rho_old[r],d_rho_old[r]},den={target_density[r],d_target_density[r]},step={dt,*d_dt};
  D num=da(dm(ov,ro,branch_local),dm(step,ds(da(dm(h,area,branch_local),src,branch_local),dm(dz,ds(vhi,vlo,branch_local),branch_local),branch_local),branch_local),branch_local);D inv=dd({1.0f,0.0f},den,branch_local),out=dm(num,inv,branch_local);
  {if(!fin(out.x))ad_fail(failure,25);if(!fin(out.d))ad_fail(failure,26);}d_output[i]=out.d;
 }
 ad_branch_flush(branch_record,branch_local,18);
}

#endif
#if !defined(AD_TL_ONLY)
extern "C" __global__ void transport_standard_finish_v841__ad(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,float dt,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,const float *inv_area_cell,
 const float *velocity,const float *edge_values,const float *vertical_flux,const float *rdzw,
 const float *old,const float *rho_old,const float *target_density,const float *source,float *b_output,
 float *b_dt,float *b_acoustic_sign,float *b_inv_area_cell,float *b_velocity,float *b_edge_values,
 float *b_vertical_flux,float *b_rdzw,float *b_old,float *b_rho_old,float *b_target_density,float *b_source,Failure *failure)
{
 int thread=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 for(int c=thread;c<ncells;c+=stride){int count=n_edges_on_cell[c];if(count<0){ad_fail(failure,27);continue;}if(count>max_edges){ad_fail(failure,56);continue;}
  bool valid=true;for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,28);valid=false;}if(e>=nedges){ad_fail(failure,57);valid=false;}}if(!valid)continue;
  double ba=(double)b_inv_area_cell[c];{if(!fin(dt))ad_fail(failure,29);if(!fin(inv_area_cell[c]))ad_fail(failure,30);}
  for(int k=0;k<nlev;++k){int r=C2(k,c,ncells);float den=target_density[r];double bro=(double)b_rho_old[r],bden=(double)b_target_density[r];
   if(!fin(den)){ad_fail(failure,31);continue;}if(!ad_nonzero(den)){ad_fail(failure,58);continue;}
   for(int q=0;q<ntracers;++q){int i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);double bo=b_output[i];
    {if(!fin(bo))ad_fail(failure,32);if(!fin(old[i]))ad_fail(failure,33);if(!fin(rho_old[r]))ad_fail(failure,34);if(!fin(source[i]))ad_fail(failure,35);if(!fin(rdzw[k]))ad_fail(failure,36);if(!fin(vertical_flux[lo]))ad_fail(failure,37);if(!fin(vertical_flux[hi]))ad_fail(failure,38);}
    float h=horizontal(q,k,c,nlev,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,velocity,edge_values);
    float z=mpas_mul(rdzw[k],mpas_sub(vertical_flux[hi],vertical_flux[lo]));float d=mpas_sub(mpas_add(mpas_mul(h,inv_area_cell[c]),source[i]),z);
    float num=mpas_add(mpas_mul(old[i],rho_old[r]),mpas_mul(dt,d)),inv=mpas_div(1.0f,den);double bn=bo*(double)inv,bt=bn*(double)dt,bh=bt*(double)inv_area_cell[c];
    {if(!fin(h))ad_fail(failure,39);if(!fin(num))ad_fail(failure,40);if(!fin(bn))ad_fail(failure,41);}
    bden-=((bo*(double)num)*(double)inv)/(double)den;bro+=bn*(double)old[i];b_old[i]+=bn*(double)rho_old[r];b_source[i]+=bt;
    ba+=bt*(double)h;
    for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j],v=E2(k,e,nedges),p=QE3(q,k,e,nlev,nedges);{if(!fin(acoustic_sign[j]))ad_fail(failure,42);if(!fin(velocity[v]))ad_fail(failure,43);if(!fin(edge_values[p]))ad_fail(failure,44);}}
   }
   b_rho_old[r]=(double)bro;b_target_density[r]=(double)bden;
   if(!fin(b_rho_old[r]))ad_fail(failure,45);
   if(!fin(b_target_density[r]))ad_fail(failure,46);
  }
  for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];double bs=(double)b_acoustic_sign[j];
   for(int k=0;k<nlev;++k)for(int q=0;q<ntracers;++q){int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),v=E2(k,e,nedges),p=QE3(q,k,e,nlev,nedges);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
    double bh=((b_output[i]*(double)mpas_div(1.0f,den))*(double)dt)*(double)inv_area_cell[c];bs-=(bh*(double)edge_values[p])*(double)velocity[v];
   }b_acoustic_sign[j]=(double)bs;
  }
  for(int q=0;q<ntracers;++q)for(int face=0;face<=nlev;++face){int p=QI3(q,face,c,nlev,ncells);double bv=(double)b_vertical_flux[p];
   for(int z=0;z<2;++z){int k=face-z;if(k<0||k>=nlev)continue;int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
    double term=((b_output[i]*(double)mpas_div(1.0f,den))*(double)dt)*(double)rdzw[k];bv+=(z==0?term:-term);
   }b_vertical_flux[p]=(double)bv;
  }
  b_inv_area_cell[c]=(double)ba;if(!fin(b_inv_area_cell[c]))ad_fail(failure,47);
 }
 for(int e=thread;e<nedges;e+=stride)for(int base=0;base<nlev;base+=64){int levels=min(64,nlev-base);double bv[64];
  for(int l=0;l<levels;++l)bv[l]=(double)b_velocity[E2(base+l,e,nedges)];
  for(int q=0;q<ntracers;++q){double be[64];for(int l=0;l<levels;++l)be[l]=(double)b_edge_values[QE3(q,base+l,e,nlev,nedges)];
   for(int c=0;c<ncells;++c){int count=n_edges_on_cell[c];if(count<0||count>max_edges)continue;
    for(int s=0;s<count;++s){int j=CES(c,s,max_edges);if(edges_on_cell[j]!=e)continue;
     for(int l=0;l<levels;++l){int k=base+l,r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),v=E2(k,e,nedges),p=QE3(q,k,e,nlev,nedges);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
      double bh=((b_output[i]*(double)mpas_div(1.0f,den))*(double)dt)*(double)inv_area_cell[c];float sg=-acoustic_sign[j];
      bv[l]+=(bh*(double)edge_values[p])*(double)sg;be[l]+=bh*(double)mpas_mul(sg,velocity[v]);
     }
    }
   }
   for(int l=0;l<levels;++l){int p=QE3(q,base+l,e,nlev,nedges);b_edge_values[p]=(double)be[l];if(!fin(b_edge_values[p]))ad_fail(failure,49);}
  }
  for(int l=0;l<levels;++l){int v=E2(base+l,e,nedges);b_velocity[v]=(double)bv[l];if(!fin(b_velocity[v]))ad_fail(failure,48);}
 }
 for(int k=thread;k<nlev;k+=stride){double sum=(double)b_rdzw[k];for(int c=0;c<ncells;++c)for(int q=0;q<ntracers;++q){int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;double bt=(b_output[i]*(double)mpas_div(1.0f,den))*(double)dt;sum-=bt*(double)mpas_sub(vertical_flux[hi],vertical_flux[lo]);}
  b_rdzw[k]=(double)sum;if(!fin(b_rdzw[k]))ad_fail(failure,50);
 }
 if(thread==0){double sum=(double)*b_dt;for(int c=0;c<ncells;++c){int count=n_edges_on_cell[c];if(count<0||count>max_edges)continue;bool valid=true;for(int s=0;s<count;++s){int e=edges_on_cell[CES(c,s,max_edges)];if(e<0||e>=nedges)valid=false;}if(!valid)continue;
  for(int q=0;q<ntracers;++q)for(int k=0;k<nlev;++k){int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
   float h=horizontal(q,k,c,nlev,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,velocity,edge_values),d=mpas_sub(mpas_add(mpas_mul(h,inv_area_cell[c]),source[i]),mpas_mul(rdzw[k],mpas_sub(vertical_flux[hi],vertical_flux[lo])));sum+=(b_output[i]*(double)mpas_div(1.0f,den))*(double)d;}}
  *b_dt=(double)sum;if(!fin(*b_dt))ad_fail(failure,51);
 }
}
#endif

// Runs after all reverse readers on the same stream.
#if !defined(AD_TL_ONLY)
extern "C" __global__ void transport_standard_finish_v841__ad_clear(int ntracers, int nlev, int ncells, float *b_output)
{
 int c = blockIdx.x * blockDim.x + threadIdx.x;
 if (c < ncells) for (int q=0; q<ntracers; ++q) for (int k=0; k<nlev; ++k) { int i=Q3(q,k,c,nlev,ncells); b_output[i] = 0.0f; }
}
#endif
