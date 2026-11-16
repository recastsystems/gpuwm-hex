// SPDX-License-Identifier: Apache-2.0
// L4 speed (ad/window-L4-speed): owner-computes reverse of transport_standard_finish_v841.
// The tangent, the old reverse and ad_clear are the unchanged U2 source (included below).
// The old reverse searched every cell for every edge (O(nedges*ncells) per tracer) and summed
// dt on one thread. Here:
//   l4_cell      one thread per cell: the old per-cell owner loop, unchanged (bitwise)
//   l4_inv_*     edge -> cell slots j = c*max_edges + s, ascending (l4_common.cuh)
//   l4_vel       one thread per (k,e): b_velocity over (q, c, s) in the old order (bitwise)
//   l4_ev        one thread per (q,k,e): b_edge_values over (c, s) in the old order (bitwise)
//   l4_rdzw      one thread per level: the old per-level owner loop, unchanged (bitwise)
//   l4_dt_*      per-cell binary64 partials, then one fixed-order block sum (binary64 reassociation only)
// No float atomics. Failure slots keep their U2 numbers.
#include "transport_standard_finish_v841.ad.cu"
#include "l4_common.cuh"

#define TSF(name) transport_standard_finish_v841__l4_##name

extern "C" __global__ void TSF(inv_count)(int ncells, int nedges, int max_edges, const int *n_edges_on_cell, const int *edges_on_cell, int *ws)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ncells) return;
    const int count = n_edges_on_cell[c]; if (count < 0 || count > max_edges) return;
    for (int s = 0; s < count; ++s) { const int e = edges_on_cell[CES(c, s, max_edges)]; if (e >= 0 && e < nedges) atomicAdd(&ws[e], 1); }
}
extern "C" __global__ void TSF(inv_scan)(int nedges, int *ws) { l4_scan_block(ws, nedges); }
extern "C" __global__ void TSF(inv_fill)(int ncells, int nedges, int max_edges, const int *n_edges_on_cell, const int *edges_on_cell, int *ws)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ncells) return;
    const int count = n_edges_on_cell[c]; if (count < 0 || count > max_edges) return;
    int *cur = l4_cursor(ws, nedges), *list = l4_list(ws, nedges);
    for (int s = 0; s < count; ++s) {
        const int j = CES(c, s, max_edges), e = edges_on_cell[j];
        if (e >= 0 && e < nedges) list[ws[e] + atomicAdd(&cur[e], 1)] = j;
    }
}
extern "C" __global__ void TSF(inv_sort)(int nedges, int *ws)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e < nedges) l4_sort_key(ws, nedges, e);
}

extern "C" __global__ void TSF(cell)(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,float dt,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,const float *inv_area_cell,
 const float *velocity,const float *edge_values,const float *vertical_flux,const float *rdzw,
 const float *old,const float *rho_old,const float *target_density,const float *source,const float *b_output,
 float *b_acoustic_sign,float *b_inv_area_cell,float *b_vertical_flux,float *b_old,float *b_rho_old,float *b_target_density,float *b_source,Failure *failure)
{
 const int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=ncells)return;
 {int count=n_edges_on_cell[c];if(count<0){ad_fail(failure,27);return;}if(count>max_edges){ad_fail(failure,56);return;}
  bool valid=true;for(int s=0;s<count;++s){int j=CES(c,s,max_edges),e=edges_on_cell[j];if(e<0){ad_fail(failure,28);valid=false;}if(e>=nedges){ad_fail(failure,57);valid=false;}}if(!valid)return;
  double ba=(double)b_inv_area_cell[c];{if(!fin(dt))ad_fail(failure,29);if(!fin(inv_area_cell[c]))ad_fail(failure,30);}
  for(int k=0;k<nlev;++k){int r=C2(k,c,ncells);float den=target_density[r];double bro=(double)b_rho_old[r],bden=(double)b_target_density[r];
   if(!fin(den)){ad_fail(failure,31);continue;}if(!ad_nonzero(den)){ad_fail(failure,58);continue;}
   for(int q=0;q<ntracers;++q){int i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);double bo=b_output[i];
    {if(!fin(bo))ad_fail(failure,32);if(!fin(old[i]))ad_fail(failure,33);if(!fin(rho_old[r]))ad_fail(failure,34);if(!fin(source[i]))ad_fail(failure,35);if(!fin(rdzw[k]))ad_fail(failure,36);if(!fin(vertical_flux[lo]))ad_fail(failure,37);if(!fin(vertical_flux[hi]))ad_fail(failure,38);}
    float h=horizontal(q,k,c,nlev,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,velocity,edge_values);
    float z=mpas_mul(rdzw[k],mpas_sub(vertical_flux[hi],vertical_flux[lo]));float d=mpas_sub(mpas_add(mpas_mul(h,inv_area_cell[c]),source[i]),z);
    float num=mpas_add(mpas_mul(old[i],rho_old[r]),mpas_mul(dt,d)),inv=mpas_div(1.0f,den);double bn=bo*(double)inv,bt=bn*(double)dt,bh=bt*(double)inv_area_cell[c];
    (void)bh;
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
}

extern "C" __global__ void TSF(vel)(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,float dt,const float *acoustic_sign,const float *inv_area_cell,
 const float *edge_values,const float *target_density,const float *b_output,float *b_velocity,const int *ws,Failure *failure)
{
 const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(long long)nlev*nedges)return;
 const int k=(int)(t/nedges),e=(int)(t%nedges);
 const int *off=ws,*list=ws+2*nedges+1;
 const int v=E2(k,e,nedges); double bv=(double)b_velocity[v];
 for(int q=0;q<ntracers;++q){const int p=QE3(q,k,e,nlev,nedges);
  for(int a=off[e];a<off[e+1];++a){const int j=list[a],c=j/max_edges;const int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
   double bh=((b_output[i]*(double)mpas_div(1.0f,den))*(double)dt)*(double)inv_area_cell[c];float sg=-acoustic_sign[j];
   bv+=(bh*(double)edge_values[p])*(double)sg;
  }
 }
 b_velocity[v]=(double)bv;if(!fin(b_velocity[v]))ad_fail(failure,48);
}

extern "C" __global__ void TSF(ev)(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,float dt,const float *acoustic_sign,const float *inv_area_cell,
 const float *velocity,const float *target_density,const float *b_output,float *b_edge_values,const int *ws,Failure *failure)
{
 const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(long long)ntracers*nlev*nedges)return;
 const int e=(int)(t%nedges),k=(int)((t/nedges)%nlev),q=(int)(t/((long long)nedges*nlev));
 const int *off=ws,*list=ws+2*nedges+1;
 const int v=E2(k,e,nedges),p=QE3(q,k,e,nlev,nedges); double be=(double)b_edge_values[p];
 for(int a=off[e];a<off[e+1];++a){const int j=list[a],c=j/max_edges;const int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
  double bh=((b_output[i]*(double)mpas_div(1.0f,den))*(double)dt)*(double)inv_area_cell[c];float sg=-acoustic_sign[j];
  be+=bh*(double)mpas_mul(sg,velocity[v]);
 }
 b_edge_values[p]=(double)be;if(!fin(b_edge_values[p]))ad_fail(failure,49);
}

extern "C" __global__ void TSF(rdzw)(
 int ntracers,int nlev,int ncells,float dt,const float *vertical_flux,const float *target_density,const float *b_output,float *b_rdzw,Failure *failure)
{
 const int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=nlev)return;
 double sum=(double)b_rdzw[k];for(int c=0;c<ncells;++c)for(int q=0;q<ntracers;++q){int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;double bt=(b_output[i]*(double)mpas_div(1.0f,den))*(double)dt;sum-=bt*(double)mpas_sub(vertical_flux[hi],vertical_flux[lo]);}
 b_rdzw[k]=(double)sum;if(!fin(b_rdzw[k]))ad_fail(failure,50);
}

extern "C" __global__ void TSF(dt_partial)(
 int ntracers,int nlev,int ncells,int nedges,int max_edges,
 const int *n_edges_on_cell,const int *edges_on_cell,const float *acoustic_sign,const float *inv_area_cell,
 const float *velocity,const float *edge_values,const float *vertical_flux,const float *rdzw,
 const float *target_density,const float *source,const float *b_output,double *partial)
{
 const int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=ncells)return;
 double sum=0.0;int count=n_edges_on_cell[c];
 bool valid=!(count<0||count>max_edges);
 if(valid)for(int s=0;s<count;++s){int e=edges_on_cell[CES(c,s,max_edges)];if(e<0||e>=nedges)valid=false;}
 if(valid)for(int q=0;q<ntracers;++q)for(int k=0;k<nlev;++k){int r=C2(k,c,ncells),i=Q3(q,k,c,nlev,ncells),lo=QI3(q,k,c,nlev,ncells),hi=QI3(q,k+1,c,nlev,ncells);float den=target_density[r];if(!fin(den)||!ad_nonzero(den))continue;
  float h=horizontal(q,k,c,nlev,nedges,max_edges,n_edges_on_cell,edges_on_cell,acoustic_sign,velocity,edge_values),d=mpas_sub(mpas_add(mpas_mul(h,inv_area_cell[c]),source[i]),mpas_mul(rdzw[k],mpas_sub(vertical_flux[hi],vertical_flux[lo])));sum+=(b_output[i]*(double)mpas_div(1.0f,den))*(double)d;}
 partial[c]=sum;
}
extern "C" __global__ void TSF(dt_reduce)(int ncells,const double *partial,float *b_dt,Failure *failure)
{
 const double total=l4_block_sum(partial,ncells);
 if(threadIdx.x==0){*b_dt=(float)((double)*b_dt+total);if(!fin(*b_dt))ad_fail(failure,51);}
}
