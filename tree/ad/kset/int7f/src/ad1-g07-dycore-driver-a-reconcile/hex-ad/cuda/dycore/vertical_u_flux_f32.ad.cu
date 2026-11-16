// SPDX-License-Identifier: Apache-2.0
// Reverse gathers through fixed, sorted cell-to-edge adjacency; no atomics.
using AdFail = int;
// Integer branch counts are warp-aggregated and do not affect field reductions.
__device__ __forceinline__ void record_sign(unsigned int *record,int slot,unsigned int sign) {
 unsigned int live=__activemask(),negative=__ballot_sync(live,sign!=0);
 if ((threadIdx.x&31)==(__ffs((int)live)-1)) {
  atomicAdd(&record[2*slot],(unsigned int)__popc(live&~negative));
  atomicAdd(&record[2*slot+1],(unsigned int)__popc(negative));
 }
}

__device__ __forceinline__ bool finite_f32(float x) { return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u; }
__device__ __forceinline__ void refuse(AdFail *f,int slot,unsigned long long field,unsigned long long element) { atomicExch(&f[slot], 1); }
struct FluxTerms { float value,dv,dm2,dm1,di,dp1; bool good; };
__device__ __forceinline__ FluxTerms terms(int nlev,int ncells,int nedges,int k,int e,
  const float *u,const float *rw,const int *cells,const float *fzm,const float *fzp) {
    FluxTerms t={0,0,0,0,0,0,true};
    if(k<=0 || k>=nlev) return t;
    int c0=cells[2*e],c1=cells[2*e+1];
    float r0=rw[k*ncells+c0],r1=rw[k*ncells+c1],v=0.5f*(r0+r1);
    t.good=finite_f32(r0)&&finite_f32(r1)&&finite_f32(v);
    float qi=u[k*nedges+e],qm1=u[(k-1)*nedges+e];
    t.good=t.good&&finite_f32(qi)&&finite_f32(qm1);
    if(k==1 || k==nlev-1) {
        float f=fzm[k]*qi+fzp[k]*qm1;
        t.value=v*f;t.dv=f;t.di=v*fzm[k];t.dm1=v*fzp[k];
        t.good=t.good&&finite_f32(fzm[k])&&finite_f32(fzp[k])&&finite_f32(f);
    } else {
        float qm2=u[(k-2)*nedges+e],qp1=u[(k+1)*nedges+e];
        float a=7.0f*(qi+qm1)-(qp1+qm2);
        float b=(qp1-qm2)-3.0f*(qi-qm1);
        float av=__uint_as_float(__float_as_uint(v)&0x7fffffffu);
        float sign=(__float_as_uint(v)&0x80000000u)?-1.0f:1.0f;
        t.value=v*a/12.0f+av*b/12.0f;
        t.dv=(a+sign*b)/12.0f;
        t.dm2=(-v-av)/12.0f;t.dm1=(7.0f*v+3.0f*av)/12.0f;
        t.di=(7.0f*v-3.0f*av)/12.0f;t.dp1=(-v+av)/12.0f;
        t.good=t.good&&finite_f32(qm2)&&finite_f32(qp1)&&finite_f32(a)&&finite_f32(b);
    }
    t.good=t.good&&finite_f32(t.value)&&finite_f32(t.dv)&&finite_f32(t.dm2)&&
           finite_f32(t.dm1)&&finite_f32(t.di)&&finite_f32(t.dp1);
    return t;
}
extern "C" __global__ void vertical_u_flux_f32__tl(int nlev,int ncells,int nedges,
 const float *u,const float *rw,const int *cells,const float *fzm,const float *fzp,
 const float *d_u,const float *d_rw,float *d_flux,AdFail *fail, unsigned int *branch_record) {
    int nout=(nlev+1)*nedges;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nout;i+=blockDim.x*gridDim.x) {
        int k=i/nedges,e=i%nedges;float v=0.0f;
        FluxTerms t=terms(nlev,ncells,nedges,k,e,u,rw,cells,fzm,fzp);
        if(k>1 && k<nlev-1) { int c0=cells[2*e],c1=cells[2*e+1]; float vel=0.5f*(rw[k*ncells+c0]+rw[k*ncells+c1]); record_sign(branch_record,2,__float_as_uint(vel)>>31); }
        if(k>0 && k<nlev) {
            int c0=cells[2*e],c1=cells[2*e+1];
            float dvel=0.5f*(d_rw[k*ncells+c0]+d_rw[k*ncells+c1]);
            v=t.dv*dvel+t.di*d_u[k*nedges+e]+t.dm1*d_u[(k-1)*nedges+e];
            if(k!=1 && k!=nlev-1) v+=t.dm2*d_u[(k-2)*nedges+e]+t.dp1*d_u[(k+1)*nedges+e];
            if(!t.good || !finite_f32(dvel) || !finite_f32(d_u[k*nedges+e]) ||
               !finite_f32(d_u[(k-1)*nedges+e]) || !finite_f32(v) ||
               (k!=1 && k!=nlev-1 && (!finite_f32(d_u[(k-2)*nedges+e]) || !finite_f32(d_u[(k+1)*nedges+e]))))
                refuse(fail,0,1,i);
        }
        d_flux[i]=v;
    }
}
// cell_edge_offsets/cell_edges: sorted unique incident edges per cell, fixed geometry.
extern "C" __global__ void vertical_u_flux_f32__ad(int nlev,int ncells,int nedges,
 const float *u,const float *rw,const int *cells,const float *fzm,const float *fzp,
 const int *cell_edge_offsets,const int *cell_edges,const float *b_flux, float *b_flux_consumed,
 float *b_u,float *b_rw,AdFail *fail, unsigned int *branch_record) {
    int nout=(nlev+1)*nedges,nu=nlev*nedges,nrw=(nlev+1)*ncells;
    int total=nout>nrw?nout:nrw;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=blockDim.x*gridDim.x) {
        // b_flux is the immutable pre-adjoint snapshot; this is the disjoint incoming seed.
        if (i < nout) b_flux_consumed[i] = 0.0f;
        int k=i/nedges,e=i%nedges;
        if(i<nout) {
            FluxTerms t=terms(nlev,ncells,nedges,k,e,u,rw,cells,fzm,fzp);
        if(k>1 && k<nlev-1) { int c0=cells[2*e],c1=cells[2*e+1]; float vel=0.5f*(rw[k*ncells+c0]+rw[k*ncells+c1]); record_sign(branch_record,2,__float_as_uint(vel)>>31); }
            if(!t.good || !finite_f32(b_flux[i])) refuse(fail,1,6,i);
        }
        if(i<nu) {
            int j=k;double v=(double)b_u[i];
            for(int q=j-1;q<=j+2;++q) if(q>0 && q<nlev) {
                FluxTerms z=terms(nlev,ncells,nedges,q,e,u,rw,cells,fzm,fzp);
                float weight=(j==q)?z.di:(j==q-1)?z.dm1:(j==q-2)?z.dm2:z.dp1;
                v+=weight*b_flux[q*nedges+e];
            }
            if(!finite_f32(b_u[i]) || !finite_f32(v)) refuse(fail,2,1,i);
            b_u[i]=v;
        }
        if(i<nrw) {
            int z=i/ncells,c=i%ncells;double v=(double)b_rw[i];
            if(z>0 && z<nlev) for(int s=cell_edge_offsets[c];s<cell_edge_offsets[c+1];++s) {
                int edge=cell_edges[s];
                FluxTerms q=terms(nlev,ncells,nedges,z,edge,u,rw,cells,fzm,fzp);
                int count=(cells[2*edge]==c)+(cells[2*edge+1]==c);
                v+=0.5f*(float)count*q.dv*b_flux[z*nedges+edge];
            }
            if(!finite_f32(b_rw[i]) || !finite_f32(v)) refuse(fail,3,2,i);
            b_rw[i]=v;
        }
    }
}
