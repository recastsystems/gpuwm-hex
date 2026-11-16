// SPDX-License-Identifier: Apache-2.0
// Tangent and adjoint of hexcore/cuda_upper_damping_v841.py upper_cam_f32 (NOAA's CAM filter on
// the top config_number_cam_damping_levels levels: kdiff = max(kdiff, floor(k)), floor a fixed per-level constant).
// The executed arm: the derivative passes where the viscosity exceeds the floor, and is 0 where the floor was taken.
// Read with values > floor, the arm is the same whether the bound primal is the launch's input or its output
// (input > floor <=> output > floor; output == floor wherever the floor was taken). An exact tie takes the floor arm.
// floor is formed with the forward's own operations in its order (--fmad=false, --ftz=false: the forward's
// mpas_mul/mpas_sub/mpas_div are IEEE round-to-nearest on these normal operands), so the comparison is the forward's.
__device__ __forceinline__ bool cam_active(int index,int nl,int nc,int solved,int levels){
 const int k=index/nc; return index%nc<solved && k>=nl-levels;
}
__device__ __forceinline__ float cam_floor(int k,int nl,int levels,float length,float coefficient){
 const float base=((4.0f*2.0833f)*length)*coefficient;
 return base*(1.0f-(float(nl-1-k)/float(levels)));
}
#if !defined(AD_ONLY_AD)
extern "C" __global__ void upper_cam_f32__tl(int nl,int nc,int solved,int levels,float length,float coefficient,
    const float *values,const float *d_values_in,float *d_values_out,int *failure){
 const int index=blockDim.x*blockIdx.x+threadIdx.x; if(index>=nl*nc)return;
 float d=d_values_in[index];
 if(cam_active(index,nl,nc,solved,levels)){
  const float v=values[index];
  if(!isfinite(v)){failure[0]=1;}
  if(!(v>cam_floor(index/nc,nl,levels,length,coefficient)))d=0.0f;
 }
 if(!isfinite(d))failure[0]=1;
 d_values_out[index]=d;
}
#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void upper_cam_f32__ad(int nl,int nc,int solved,int levels,float length,float coefficient,
    const float *values,float *b_values_in,const float *b_values_out,int *failure){
 const int index=blockDim.x*blockIdx.x+threadIdx.x; if(index>=nl*nc)return;
 float bar=b_values_out[index];
 if(cam_active(index,nl,nc,solved,levels)){
  const float v=values[index];
  if(!isfinite(v)){failure[0]=1;}
  if(!(v>cam_floor(index/nc,nl,levels,length,coefficient)))bar=0.0f;
 }
 const float t=b_values_in[index]+bar;
 if(!isfinite(t))failure[0]=1; else b_values_in[index]=t;
}
#endif
