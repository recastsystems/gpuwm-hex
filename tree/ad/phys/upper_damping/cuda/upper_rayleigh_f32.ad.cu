// SPDX-License-Identifier: Apache-2.0
// Tangent and adjoint of hexcore/cuda_upper_damping_v841.py upper_rayleigh_f32 (NOAA's Rayleigh
// damping of u on the top levels: tend_u -= (rho_edge*u)*c(k), c(k) = (k-(nl-levels)+1)/(levels*days*86400)).
// Linear in rho_edge and u: d tend = d tend_in - (d rho*u + rho*d u)*c. c is formed with the forward's operations in
// its order. Elementwise: each word owns its own rho and u words, so the reverse has no gather.
__device__ __forceinline__ bool ray_active(int index,int nl,int ne,int solved,int levels){
 const int k=index/ne; return index%ne<solved && k>=nl-levels;
}
__device__ __forceinline__ float ray_coef(int k,int nl,int levels,float days){
 const float inverse=1.0f/(float(levels)*(days*86400.0f));
 return float(k-(nl-levels)+1)*inverse;
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void upper_rayleigh_f32__tl(int nl,int ne,int solved,int levels,float days,
    const float *tendency,const float *rho,const float *velocity,
    const float *d_rho,const float *d_velocity,const float *d_tendency_in,float *d_tendency_out,int *failure){
 const int index=blockDim.x*blockIdx.x+threadIdx.x; if(index>=nl*ne)return;
 float d=d_tendency_in[index];
 if(ray_active(index,nl,ne,solved,levels)){
  const float c=ray_coef(index/ne,nl,levels,days);
  const float r=rho[index],u=velocity[index];
  if(!isfinite(r)||!isfinite(u)||!isfinite(d_rho[index])||!isfinite(d_velocity[index]))failure[0]=1;
  d=d-((d_rho[index]*u)+(r*d_velocity[index]))*c;
 }
 if(!isfinite(d))failure[0]=1;
 d_tendency_out[index]=d;
}
#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void upper_rayleigh_f32__ad(int nl,int ne,int solved,int levels,float days,
    const float *tendency,const float *rho,const float *velocity,
    float *b_rho,float *b_velocity,float *b_tendency_in,const float *b_tendency_out,int *failure){
 const int index=blockDim.x*blockIdx.x+threadIdx.x; if(index>=nl*ne)return;
 const float bar=b_tendency_out[index];
 const float t=b_tendency_in[index]+bar;
 if(!isfinite(t))failure[0]=1; else b_tendency_in[index]=t;
 if(!ray_active(index,nl,ne,solved,levels))return;
 const float c=ray_coef(index/ne,nl,levels,days);
 const float r=rho[index],u=velocity[index];
 const float s=-(bar*c);
 const float br=b_rho[index]+s*u, bu=b_velocity[index]+s*r;
 if(!isfinite(r)||!isfinite(u)||!isfinite(br)||!isfinite(bu))failure[0]=1;
 else{b_rho[index]=br;b_velocity[index]=bu;}
}
#endif
