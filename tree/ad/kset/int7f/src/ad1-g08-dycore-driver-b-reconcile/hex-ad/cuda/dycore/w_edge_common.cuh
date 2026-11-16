// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dycore_math.cuh"
struct WEdge {
    int nlev, ncells, nedges, width;
    float coefficient;
    const float *w, *ru, *fzm, *fzp, *adv, *adv3;
    const int *n_adv, *adv_cells;
};
__device__ __forceinline__ float we_r(const WEdge &p, int k, int e) {
    const int x = k*p.nedges+e;
    return mpas_add(mpas_mul(p.fzm[k],p.ru[x]),mpas_mul(p.fzp[k],p.ru[x-p.nedges]));
}
__device__ __forceinline__ float we_weight(const WEdge &p, int a, float sign) {
    return mpas_add(p.adv[a], mpas_mul(mpas_mul(p.coefficient,sign),p.adv3[a]));
}
__device__ __forceinline__ float we_sum(const WEdge &p, int k, int e, float sign) {
    float z=0.0f;
    for (int s=0;s<p.n_adv[e];++s) {
        const int a=e*p.width+s,c=p.adv_cells[a];
        z=mpas_add(z,mpas_mul(we_weight(p,a,sign),p.w[k*p.ncells+c]));
    }
    return z;
}
struct WEdgeTangent {
    float d_coefficient;
    const float *dw,*dru,*dfzm,*dfzp,*dadv,*dadv3;
};
__device__ __forceinline__ void we_tangent_element(const WEdge &p, const WEdgeTangent &d,
                                                    int k, int e, float *dflux,
                                                    unsigned *branch_record, int branch_slot, int *failures) {
    const int x=k*p.nedges+e;
    float result=0.0f;
    if (k>0 && k<p.nlev) {
        const float r=we_r(p,k,e), sign=mpas_copysign(1.0f,r);
        atomicAdd(&branch_record[2*branch_slot+(__float_as_uint(r)>>31)], 1u);
        const float dr=d.dfzm[k]*p.ru[x]+p.fzm[k]*d.dru[x]
                      +d.dfzp[k]*p.ru[x-p.nedges]+p.fzp[k]*d.dru[x-p.nedges];
        float w=0.0f,dw=0.0f;
        for (int s=0;s<p.n_adv[e];++s) {
            const int a=e*p.width+s, c=p.adv_cells[a], wc=k*p.ncells+c;
            const float weight=we_weight(p,a,sign);
            const float dweight=d.dadv[a]+sign*(d.d_coefficient*p.adv3[a]+p.coefficient*d.dadv3[a]);
            w=mpas_add(w,mpas_mul(weight,p.w[wc]));
            dw+=dweight*p.w[wc]+weight*d.dw[wc];
            if (!isfinite(weight)||!isfinite(p.w[wc])||!isfinite(dweight)||!isfinite(d.dw[wc]))
                ad_fail(failures,x,8,x,1);
        }
        result=dr*w+r*dw;
        if (!isfinite(p.coefficient)||!isfinite(d.d_coefficient)||!isfinite(r)||!isfinite(dr)||
            !isfinite(p.ru[x])||!isfinite(p.ru[x-p.nedges])||!isfinite(p.fzm[k])||!isfinite(p.fzp[k])||
            !isfinite(w)||!isfinite(result)||!isfinite(mpas_mul(r,w))) ad_fail(failures,x,8,x,1);
    }
    dflux[x]=result;
}
struct WEdgeAdjoint {
    const int *cell_offsets,*cell_edges,*cell_slots;
    const float *bflux;
    float *bw,*bru,*bfzm,*bfzp,*badv,*badv3,*bcoefficient;
};
// Each domain element has one owner. The reverse topology is sorted by
// (edge,slot), and levels and edges are visited in increasing order.
__device__ __forceinline__ void we_adjoint_element(const WEdge &p, const WEdgeAdjoint &b,
                                                    int id, int *failures) {
    const int nw=(p.nlev+1)*p.ncells, nr=p.nlev*p.nedges, nf=p.nlev, na=p.nedges*p.width;
    if (id<nw) {
        const int k=id/p.ncells,c=id%p.ncells;
        double v=(double)b.bw[id];
        if (k>0 && k<p.nlev) {
            for (int q=b.cell_offsets[c];q<b.cell_offsets[c+1];++q) {
                const int e=b.cell_edges[q],a=e*p.width+b.cell_slots[q],x=k*p.nedges+e;
                const float r=we_r(p,k,e),sign=mpas_copysign(1.0f,r);
                v+=(double)b.bflux[x]*r*we_weight(p,a,sign);
            }
        }
        if (!isfinite(v)||!isfinite((float)v)) ad_fail(failures,id,2,id,1);
        b.bw[id]=(float)v;
    } else if (id<nw+nr) {
        const int j=id-nw,k=j/p.nedges,e=j%p.nedges;
        double v=(double)b.bru[j];
        if (k>0) {
            const float r=we_r(p,k,e),sgn=mpas_copysign(1.0f,r);
            v+=(double)b.bflux[k*p.nedges+e]*we_sum(p,k,e,sgn)*p.fzm[k];
        }
        if (k+1<p.nlev) {
            const float r=we_r(p,k+1,e),sgn=mpas_copysign(1.0f,r);
            v+=(double)b.bflux[(k+1)*p.nedges+e]*we_sum(p,k+1,e,sgn)*p.fzp[k+1];
        }
        if (!isfinite(v)||!isfinite((float)v)) ad_fail(failures,id,3,j,1);
        b.bru[j]=(float)v;
    } else if (id<nw+nr+nf) {
        const int k=id-nw-nr;
        double m=(double)b.bfzm[k],z=(double)b.bfzp[k];
        if (k>0) for (int e=0;e<p.nedges;++e) {
            const int x=k*p.nedges+e;
            const float r=we_r(p,k,e),sgn=mpas_copysign(1.0f,r);
            const double g=(double)b.bflux[x]*we_sum(p,k,e,sgn);
            m+=g*p.ru[x]; z+=g*p.ru[x-p.nedges];
        }
        if (!isfinite(m)||!isfinite(z)||!isfinite((float)m)||!isfinite((float)z)) ad_fail(failures,id,4,k,1);
        b.bfzm[k]=(float)m; b.bfzp[k]=(float)z;
    } else if (id<nw+nr+nf+na) {
        const int a=id-nw-nr-nf,e=a/p.width,s=a%p.width;
        double v=(double)b.badv[a],v3=(double)b.badv3[a];
        if (s<p.n_adv[e]) for (int k=1;k<p.nlev;++k) {
            const int x=k*p.nedges+e,c=p.adv_cells[a];
            const float r=we_r(p,k,e),sgn=mpas_copysign(1.0f,r);
            const double g=(double)b.bflux[x]*r*p.w[k*p.ncells+c];
            v+=g; v3+=g*p.coefficient*sgn;
        }
        if (!isfinite(v)||!isfinite(v3)||!isfinite((float)v)||!isfinite((float)v3)) ad_fail(failures,id,6,a,1);
        b.badv[a]=(float)v; b.badv3[a]=(float)v3;
    }
}
// A single 128-thread block owns the scalar reduction. Threads visit disjoint
// (level, edge) pairs in a fixed order; the shared tree has a fixed order.
__device__ __forceinline__ void we_adjoint_scalar(const WEdge &p, const WEdgeAdjoint &b,
                                                  int *failures, int slot) {
    if (blockIdx.x != 0) return;
    __shared__ double sums[128];
    const int lane=threadIdx.x;
    if (blockDim.x != 128) { if(lane==0) ad_fail(failures,slot,1,0,2); return; }
    double sum=0.0;
    const int pairs=(p.nlev-1)*p.nedges;
    for(int j=lane;j<pairs;j+=128) {
        const int k=j/p.nedges+1,e=j%p.nedges,x=k*p.nedges+e;
        const float r=we_r(p,k,e),sgn=mpas_copysign(1.0f,r);
        for(int s=0;s<p.n_adv[e];++s) {
            const int a=e*p.width+s,c=p.adv_cells[a];
            sum+=(double)b.bflux[x]*r*sgn*p.adv3[a]*p.w[k*p.ncells+c];
        }
    }
    sums[lane]=sum; __syncthreads();
    for(int stride=64;stride>0;stride/=2) {
        if(lane<stride) sums[lane]+=sums[lane+stride];
        __syncthreads();
    }
    if(lane==0) {
        const float value=(float)((double)b.bcoefficient[0]+sums[0]);
        if(!isfinite(value)) ad_fail(failures,slot,1,0,1);
        b.bcoefficient[0]=value;
    }
}
