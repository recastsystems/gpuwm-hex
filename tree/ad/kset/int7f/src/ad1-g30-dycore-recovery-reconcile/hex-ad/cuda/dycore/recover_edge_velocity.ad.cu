// SPDX-License-Identifier: Apache-2.0
// Derivative of recovery.py:72. Original forward remains unchanged.
#include "recovery_math.cuh"
#include "recovery_branch_record.cuh"


template<class T> __device__ __forceinline__ T rec_add(T a, T b) { return a+b; } template<class T> __device__ __forceinline__ T rec_sub(T a,T b){return a-b;}
template<> __device__ __forceinline__ float rec_add(float a, float b) { return mpas_add(a,b); } template<> __device__ __forceinline__ float rec_sub(float a,float b){return mpas_sub(a,b);}
template<class T> __device__ __forceinline__ T rec_mul(T a, T b) { return a*b; }
template<> __device__ __forceinline__ float rec_mul(float a, float b) { return mpas_mul(a,b); }
template<class T> __device__ __forceinline__ T rec_div(T a, T b) { return a/b; }
template<> __device__ __forceinline__ float rec_div(float a, float b) { return mpas_div(a,b); }
template<class T> __device__ __forceinline__ bool nonzero(T x) { return x != T(0); }
template<> __device__ __forceinline__ bool nonzero(float x) { return mpas_f32_to_f64_ieee(x) != 0.0; }
template<class T> __device__ __forceinline__ bool valid_edge(
    T r0, T r1, T momentum, T &den, T &numerator) {
  den=rec_add<T>(r0,r1); numerator=rec_mul<T>(T(2),momentum);
  return isfinite(r0) && isfinite(r1) && recovery_wide<T>(r0)>0.0 && recovery_wide<T>(r1)>0.0
      && isfinite(momentum) && isfinite(den) && nonzero<T>(den) && isfinite(numerator)
      && isfinite(rec_div<T>(numerator,den));
}
__device__ __forceinline__ void bad(int *out, int slot, unsigned long long field, unsigned long long source) {
  (void)slot; (void)field; atomicExch(out + source - 1, 1);
}

#define DEFINE_EDGE(T, SUFFIX) \
extern "C" __global__ void recover_edge_velocity_##SUFFIX##__tl( \
    const T *rho, const T *ru, const int *cells_on_edge, T *u, \
    int nlev, int ncells, int nedges, const T *d_rho, const T *d_ru, \
    T *d_u, int *failure, unsigned int *branch_record) { \
  (void)u; RecoveryBranchTrace<T> trace; \
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<nlev*nedges;i+=gridDim.x*blockDim.x) { \
    int edge=i%nedges, level=i/nedges; int c0=cells_on_edge[2*edge], c1=cells_on_edge[2*edge+1]; \
    if (c0<0 || c0>=ncells || c1<0 || c1>=ncells) { bad(failure,i,3,1); d_u[i]=T(0); continue; } \
    int i0=level*ncells+c0, i1=level*ncells+c1; T den,num; \
    if (!valid_edge<T>(rho[i0],rho[i1],ru[i],den,num)) { bad(failure,i,1,2); d_u[i]=T(0); continue; } \
    T traced_den=trace.add(rho[i0],rho[i1]); T traced_num=trace.mul(T(2),ru[i]); (void)trace.div(traced_num,traced_den); \
    T dr=rec_add<T>(d_rho[i0],d_rho[i1]); \
    T du=rec_div<T>(rec_sub<T>(rec_mul<T>(T(2),d_ru[i]),rec_mul<T>(rec_div<T>(num,den),dr)),den); \
    if (!isfinite(du)) { bad(failure,i,4,3); d_u[i]=T(0); } else d_u[i]=du; \
  } trace.publish(branch_record,sizeof(T)==4?11:1); \
} \
extern "C" __global__ void recover_edge_velocity_##SUFFIX##__ad( \
    const T *rho, const T *ru, const int *cells_on_edge, T *u, \
    int nlev, int ncells, int nedges, const int *edges_on_cell, \
    const int *n_edges_on_cell, int max_edges, T *seed_b_u, T *b_rho, T *b_ru, \
    int *failure, unsigned int *branch_record,const T *b_u) { \
  (void)u; RecoveryBranchTrace<T> trace; \
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<nlev*nedges;i+=gridDim.x*blockDim.x) { \
    int edge=i%nedges, level=i/nedges; int c0=cells_on_edge[2*edge], c1=cells_on_edge[2*edge+1]; \
    if (c0<0 || c0>=ncells || c1<0 || c1>=ncells) { bad(failure,i,3,4); continue; } \
    T den,num; if (!valid_edge<T>(rho[level*ncells+c0],rho[level*ncells+c1],ru[i],den,num) || !isfinite(b_u[i])) { bad(failure,i,1,5); continue; } \
    T traced_den=trace.add(rho[level*ncells+c0],rho[level*ncells+c1]); T traced_num=trace.mul(T(2),ru[i]); (void)trace.div(traced_num,traced_den); \
    T v=rec_div<T>(rec_mul<T>(T(2),b_u[i]),den); \
    T result=rec_add<T>(b_ru[i],v); \
    if (!isfinite(v) || !isfinite(b_ru[i]) || !isfinite(result)) { bad(failure,i,2,6); continue; } b_ru[i]=result; \
  } trace.publish(branch_record,sizeof(T)==4?11:1); \
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<nlev*ncells;i+=gridDim.x*blockDim.x) { \
    int cell=i%ncells, level=i/ncells; double sum=0.0; \
    int slots=n_edges_on_cell[cell]; \
    if (slots<0 || slots>max_edges) { bad(failure,nlev*nedges+i,3,7); continue; } \
    for (int slot=0;slot<slots;++slot) { \
      int edge=edges_on_cell[cell*max_edges+slot]; \
      if (edge<0 || edge>=nedges) { bad(failure,nlev*nedges+i,3,8); continue; } \
      int c0=cells_on_edge[2*edge], c1=cells_on_edge[2*edge+1]; int count=(c0==cell)+(c1==cell); \
      if (!count || c0<0 || c0>=ncells || c1<0 || c1>=ncells) { bad(failure,nlev*nedges+i,3,9); continue; } \
      int e=level*nedges+edge; T den,num; \
      if (!valid_edge<T>(rho[level*ncells+c0],rho[level*ncells+c1],ru[e],den,num) || !isfinite(b_u[e])) continue; \
      T v=recovery_neg<T>(rec_div<T>(rec_mul<T>(b_u[e],rec_div<T>(num,den)),den)); \
      sum+=recovery_wide<T>(v); if(count==2) sum+=recovery_wide<T>(v); \
    } \
    T result=recovery_narrow<T>(recovery_wide<T>(b_rho[i])+sum); \
    if (!isfinite(sum) || !isfinite(b_rho[i]) || !isfinite(result)) { bad(failure,nlev*nedges+i,1,10); continue; } \
    b_rho[i]=result; \
  } for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<nlev*nedges;i+=gridDim.x*blockDim.x){seed_b_u[i]=T(0);} \
}

DEFINE_EDGE(float, f32)
DEFINE_EDGE(double, f64)
