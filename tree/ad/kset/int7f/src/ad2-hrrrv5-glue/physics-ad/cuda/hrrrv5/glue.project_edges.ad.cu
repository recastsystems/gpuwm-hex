// SPDX-License-Identifier: Apache-2.0
#include "glue.member_g27.cuh"
__device__ __forceinline__ float project_dot(const float *basis, const float *normal, int c, int e) {
    float x=mpas_mul(normal[3*e],basis[3*c]);
    x=mpas_add(x,mpas_mul(normal[3*e+1],basis[3*c+1]));
    return mpas_add(x,mpas_mul(normal[3*e+2],basis[3*c+2]));
}
__device__ __forceinline__ float project_sum(float u1,float v1,float u2,float v2,float e1,float n1,float e2,float n2) {
    float x=mpas_mul(mpas_mul(u1,0.5f),e1);
    x=mpas_add(x,mpas_mul(mpas_mul(v1,0.5f),n1));
    x=mpas_add(x,mpas_mul(mpas_mul(u2,0.5f),e2));
    return mpas_add(x,mpas_mul(mpas_mul(v2,0.5f),n2));
}

// Differentiate both velocity and floating geometric inputs. Integer endpoint
// indices and the forward's refusal decision are fixed on the executed path.
__device__ __forceinline__ double project_ddot(const float *basis,const float *dbasis,
    const float *normal,const float *dnormal,int c,int e) {
    double v=0.0;
    for(int t=0;t<3;++t)
        v+=(double)dnormal[3*e+t]*(double)basis[3*c+t]
          +(double)normal[3*e+t]*(double)dbasis[3*c+t];
    return v;
}
#if !defined(AD_BUILD_AD)
extern "C" __global__ void project_edges_v841_f32__tl(
    int nlev,int ncells,int nedges,const int *cells,const float *east,
    const float *north,const float *normal,const float *du,const float *dv,
    const float *rho_edge,float *out_ru,int *invalid,
    const float *d_east,const float *d_north,const float *d_normal,
    const float *d_du,const float *d_dv,const float *d_rho_edge,float *d_out_ru,
    int *fail) {
    (void)out_ru;(void)invalid;
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=nedges)return;
    int c1=cells[2*e],c2=cells[2*e+1];
    if(c1<0||c1>=ncells||c2<0||c2>=ncells){ad_fail(fail,1,e,1);return;}
    float e1=project_dot(east,normal,c1,e),n1=project_dot(north,normal,c1,e);
    float e2=project_dot(east,normal,c2,e),n2=project_dot(north,normal,c2,e);
    double de1=project_ddot(east,d_east,normal,d_normal,c1,e);
    double dn1=project_ddot(north,d_north,normal,d_normal,c1,e);
    double de2=project_ddot(east,d_east,normal,d_normal,c2,e);
    double dn2=project_ddot(north,d_north,normal,d_normal,c2,e);
    for(int k=0;k<nlev;++k){
        int i=k*nedges+e,a=k*ncells+c1,b=k*ncells+c2;
        float v=project_sum(du[a],dv[a],du[b],dv[b],e1,n1,e2,n2);
        float y=mpas_mul(v,rho_edge[i]);
        if(!isfinite(y)||!isfinite(rho_edge[i])){ad_fail(fail,2,i,2);continue;}
        double z=0.5*((double)d_du[a]*e1+(double)d_dv[a]*n1
                     +(double)d_du[b]*e2+(double)d_dv[b]*n2)
            +(double)mpas_mul(du[a],0.5f)*de1+(double)mpas_mul(dv[a],0.5f)*dn1
            +(double)mpas_mul(du[b],0.5f)*de2+(double)mpas_mul(dv[b],0.5f)*dn2;
        d_out_ru[i]=(float)(z*(double)rho_edge[i]+(double)v*(double)d_rho_edge[i]);
    }
}
#endif
#if !defined(AD_BUILD_TL)
extern "C" __global__ void project_edges_v841_f32__ad(
    int nlev,int ncells,int nedges,const int *cells,const float *east,
    const float *north,const float *normal,const float *du,const float *dv,
    const float *rho_edge,float *out_ru,int *invalid,
    float *b_east,float *b_north,float *b_normal,
    float *b_du,float *b_dv,float *b_rho_edge,float *b_out_ru,
    const float *out_ru_seed,int *fail) {
    (void)out_ru;(void)invalid;
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x<nedges){
        int c1=cells[2*x],c2=cells[2*x+1];
        if(c1<0||c1>=ncells||c2<0||c2>=ncells)ad_fail(fail,1,x,1);
        else {
            float e1=project_dot(east,normal,c1,x),n1=project_dot(north,normal,c1,x);
            float e2=project_dot(east,normal,c2,x),n2=project_dot(north,normal,c2,x);
            double gn[3]={(double)b_normal[3*x],(double)b_normal[3*x+1],(double)b_normal[3*x+2]};
            for(int k=0;k<nlev;++k){
                int i=k*nedges+x,a=k*ncells+c1,b=k*ncells+c2;
                float v=project_sum(du[a],dv[a],du[b],dv[b],e1,n1,e2,n2);
                float y=mpas_mul(v,rho_edge[i]);
                if(!isfinite(y)||!isfinite(rho_edge[i])){ad_fail(fail,2,i,2);continue;}
                double z=(double)out_ru_seed[i];b_out_ru[i]=0.0f;
                b_rho_edge[i]=(float)((double)b_rho_edge[i]+z*(double)v);
                double zr=z*(double)rho_edge[i];
                for(int t=0;t<3;++t)
                    gn[t]+=zr*((double)mpas_mul(du[a],0.5f)*east[3*c1+t]
                             +(double)mpas_mul(dv[a],0.5f)*north[3*c1+t]
                             +(double)mpas_mul(du[b],0.5f)*east[3*c2+t]
                             +(double)mpas_mul(dv[b],0.5f)*north[3*c2+t]);
            }
            for(int t=0;t<3;++t)b_normal[3*x+t]=(float)gn[t];
        }
    }
    // One owner per cell, ascending edge/slot/level order. The immutable seed
    // snapshot avoids races with the edge owners that consume b_out_ru.
    for(int cell=x;cell<ncells;cell+=gridDim.x*blockDim.x){
        double ge[3]={(double)b_east[3*cell],(double)b_east[3*cell+1],(double)b_east[3*cell+2]};
        double gn[3]={(double)b_north[3*cell],(double)b_north[3*cell+1],(double)b_north[3*cell+2]};
        for(int e=0;e<nedges;++e)for(int slot=0;slot<2;++slot){
            if(cells[2*e+slot]!=cell)continue;
            int c1=cells[2*e],c2=cells[2*e+1];
            if(c1<0||c1>=ncells||c2<0||c2>=ncells)continue;
            float ev=project_dot(east,normal,cell,e),nv=project_dot(north,normal,cell,e);
            for(int k=0;k<nlev;++k){int i=k*nedges+e,j=k*ncells+cell;
                double zr=(double)out_ru_seed[i]*(double)rho_edge[i];
                b_du[j]=(float)((double)b_du[j]+zr*0.5*(double)ev);
                b_dv[j]=(float)((double)b_dv[j]+zr*0.5*(double)nv);
                for(int t=0;t<3;++t){
                    ge[t]+=zr*(double)mpas_mul(du[j],0.5f)*(double)normal[3*e+t];
                    gn[t]+=zr*(double)mpas_mul(dv[j],0.5f)*(double)normal[3*e+t];
                }
            }
        }
        for(int t=0;t<3;++t){b_east[3*cell+t]=(float)ge[t];b_north[3*cell+t]=(float)gn[t];}
    }
}
#endif

#if defined(AD_BUILD_BRANCH)
__device__ __forceinline__ float project_branch_dot(const float *b,const float *normal,int c,int e,unsigned int *r) {
 float x=ad_br_mul(normal[3*e],b[3*c],r,3);
 x=ad_br_add(x,ad_br_mul(normal[3*e+1],b[3*c+1],r,3),r,3);
 return ad_br_add(x,ad_br_mul(normal[3*e+2],b[3*c+2],r,3),r,3);
}
extern "C" __global__ void project_edges_v841_f32__branch_record(
 int nlev,int ncells,int nedges,const int *cells,const float *east,const float *north,
 const float *normal,const float *du,const float *dv,const float *rho_edge,
 float *out_ru,int *invalid,unsigned int *record) {
 int e=blockIdx.x*blockDim.x+threadIdx.x;
 if(ad_branch(record,0,e>=nedges)) return;
 int c1=cells[2*e],c2=cells[2*e+1];
 if(c1<0||c1>=ncells||c2<0||c2>=ncells) return;
 float e1=project_branch_dot(east,normal,c1,e,record),n1=project_branch_dot(north,normal,c1,e,record);
 float e2=project_branch_dot(east,normal,c2,e,record),n2=project_branch_dot(north,normal,c2,e,record);
 for(int k=0;k<nlev;++k) {
  ad_branch(record,1,true);int i=k*nedges+e,a=k*ncells+c1,b=k*ncells+c2;
  float x=ad_br_mul(ad_br_mul(du[a],0.5f,record,3),e1,record,3);
  x=ad_br_add(x,ad_br_mul(ad_br_mul(dv[a],0.5f,record,3),n1,record,3),record,3);
  x=ad_br_add(x,ad_br_mul(ad_br_mul(du[b],0.5f,record,3),e2,record,3),record,3);
  x=ad_br_add(x,ad_br_mul(ad_br_mul(dv[b],0.5f,record,3),n2,record,3),record,3);
  x=ad_br_mul(x,rho_edge[i],record,3);
  bool bad_value=!isfinite(x);ad_branch(record,12,bad_value);
  bool bad_rho=false;if(!bad_value)bad_rho=ad_branch(record,13,!isfinite(rho_edge[i]));
  ad_branch(record,2,bad_value||bad_rho);
 }
 ad_branch(record,1,false);
}
#endif
