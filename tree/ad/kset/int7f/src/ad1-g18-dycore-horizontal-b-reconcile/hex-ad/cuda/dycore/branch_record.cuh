// SPDX-License-Identifier: Apache-2.0
// Recompute the original FP32 helper predicates for section 33 branch records.
// This header is used only by derivative entries; it never changes the forward.
#ifndef HEX_AD_DYCORE_BRANCH_RECORD_CUH
#define HEX_AD_DYCORE_BRANCH_RECORD_CUH
struct AdBranchSites { int s42,s45,s57,s62,s78,s83,s93,s111,s116,s123,s142,s145,s162,s170,s183,s207,s208,h523,h531,h549,h554; };
struct AdBranchRecorder { unsigned int* counts; AdBranchSites sites; };
__device__ __forceinline__ void ad_br_mark(AdBranchRecorder r,int slot,bool yes) {
 if(r.counts && slot>=0) atomicAdd(&r.counts[2*slot+(yes?1:0)],1u);
}
__device__ __forceinline__ double ad_br_decode(AdBranchRecorder r,float v){
 unsigned b=__float_as_uint(v),m=b&0x7fffffffu;bool sub=m!=0u&&(m&0x7f800000u)==0u;ad_br_mark(r,r.sites.s42,sub);
 if(sub){double x=ldexp((double)(m&0x007fffffu),-149);bool sign=(b&0x80000000u)!=0u;ad_br_mark(r,r.sites.s45,sign);return sign?-x:x;}return (double)v;
}
__device__ __forceinline__ float ad_br_pack(AdBranchRecorder r,double v){
 double m=fabs(v);bool sub=m>0.0&&m<1.1754943508222875e-38;ad_br_mark(r,r.sites.s57,sub);
 if(sub){unsigned mant=(unsigned)rint(m*7.1362384635297994e+44);bool sign=__double_as_longlong(v)<0LL;ad_br_mark(r,r.sites.s62,sign);return __uint_as_float((sign?0x80000000u:0u)|mant);}return __double2float_rn(v);
}
__device__ __forceinline__ float ad_br_addsub(AdBranchRecorder r,float l,float q,bool sub){
 float ordinary=sub?l-q:l+q;unsigned om=mpas_f32_magnitude_bits(ordinary),oe=om>>23;
 ad_br_mark(r,sub?r.sites.s111:r.sites.s78,oe>24u);if(oe>24u)return ordinary;
 unsigned lm=mpas_f32_magnitude_bits(l),rm=mpas_f32_magnitude_bits(q),le=lm>>23,re=rm>>23;
 bool den=(le==0u&&lm!=0u)||(re==0u&&rm!=0u);ad_br_mark(r,sub?r.sites.s116:r.sites.s83,den);
 if(den){double a=ad_br_decode(r,l),b=ad_br_decode(r,q);return ad_br_pack(r,sub?a-b:a+b);}
 bool cancel=le<=23u&&re<=23u&&oe==0u&&lm!=rm&&lm!=0u&&rm!=0u;ad_br_mark(r,sub?r.sites.s123:r.sites.s93,cancel);
 if(cancel){double a=ad_br_decode(r,l),b=ad_br_decode(r,q);return ad_br_pack(r,sub?a-b:a+b);}return ordinary;
}
__device__ __forceinline__ float ad_br_add(AdBranchRecorder r,float l,float q){return ad_br_addsub(r,l,q,false);}
__device__ __forceinline__ float ad_br_sub(AdBranchRecorder r,float l,float q){return ad_br_addsub(r,l,q,true);}
__device__ __forceinline__ float ad_br_mul(AdBranchRecorder r,float l,float q){
 float o=l*q;bool env=(mpas_f32_magnitude_bits(o)&0x7f800000u)==0u;ad_br_mark(r,r.sites.s142,env);
 if(env){unsigned a=mpas_f32_magnitude_bits(l),b=mpas_f32_magnitude_bits(q);bool nz=a!=0u&&b!=0u;ad_br_mark(r,r.sites.s145,nz);if(nz){double x=ad_br_decode(r,l),y=ad_br_decode(r,q);return ad_br_pack(r,x*y);}}return o;
}
__device__ __forceinline__ float ad_br_div(AdBranchRecorder r,float l,float q){
 unsigned lm=mpas_f32_magnitude_bits(l),rm=mpas_f32_magnitude_bits(q),le=lm>>23,re=rm>>23;bool den=(le==0u&&lm!=0u)||(re==0u&&rm!=0u);ad_br_mark(r,r.sites.s162,den);
 if(den){double a=ad_br_decode(r,l),b=ad_br_decode(r,q);return ad_br_pack(r,a/b);}float o=l/q;
 bool env=lm!=0u&&rm!=0u&&le+126u<=re&&(mpas_f32_magnitude_bits(o)&0x7f800000u)==0u;ad_br_mark(r,r.sites.s170,env);
 if(env){double a=ad_br_decode(r,l),b=ad_br_decode(r,q);return ad_br_pack(r,a/b);}return o;
}
__device__ __forceinline__ float ad_br_sqrt(AdBranchRecorder r,float x){bool sub=mpas_is_subnormal_f32(x);ad_br_mark(r,r.sites.s183,sub);return sub?ad_br_pack(r,sqrt(ad_br_decode(r,x))):sqrtf(x);}
__device__ __forceinline__ bool ad_br_right_min(float a,float b){
 if(mpas_is_subnormal_f32(a)||mpas_is_subnormal_f32(b))return mpas_f32_to_f64_ieee(b)<mpas_f32_to_f64_ieee(a);
 if(b<a)return true;if(a<b)return false;return (__float_as_uint(a)&0x80000000u)==0u&&(__float_as_uint(b)&0x80000000u)!=0u;
}
__device__ __forceinline__ float ad_br_min(AdBranchRecorder r,float a,float b){
 ad_br_mark(r,r.sites.h549,ad_br_right_min(a,b));bool sub=mpas_is_subnormal_f32(a)||mpas_is_subnormal_f32(b);ad_br_mark(r,r.sites.s207,sub);
 if(sub){double right=ad_br_decode(r,b),left=ad_br_decode(r,a);bool choose=right<left;ad_br_mark(r,r.sites.s208,choose);return choose?b:a;}return fminf(a,b);
}
__device__ __forceinline__ float ad_br_fmin(AdBranchRecorder r,float a,float b){ad_br_mark(r,r.sites.h554,ad_br_right_min(a,b));return fminf(a,b);}
__device__ __forceinline__ int ad_br_lidx(int k,int i,int n){return k*n+i;}
__device__ __forceinline__ float ad_br_cell_sign(int c,int e,const int* p){return p[2*e]==c?1.0f:-1.0f;}
__device__ __forceinline__ float ad_br_vertex_sign(int v,int e,const int* p){return p[2*e+1]==v?1.0f:-1.0f;}
#endif
