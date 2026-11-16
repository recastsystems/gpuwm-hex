// SPDX-License-Identifier: Apache-2.0
// Tangent and deterministic Euclidean transpose of cuda_horizontal.py:874-908.
#include "mpas_fp32_ad.cuh"
#include "ad_failure_slots.cuh"

__device__ __forceinline__ void refuse(int* f, int slot,
                                       unsigned int code, unsigned long long field,
                                       unsigned long long index) {
    /* The exact failed predicate recorded its own slot. */
}

struct DivLocal { float delta, den, ab, a, b, q; };
__device__ __forceinline__ DivLocal div_local(float t0, float t1, float p0, float p1,
                                              float o0, float o1, float mask, float coefficient) {
    DivLocal z;
    z.delta = mpas_sub(mpas_sub(p0, o0), mpas_sub(p1, o1));
    z.den = mpas_add(t0, t1);
    z.a = mpas_mul(coefficient, z.delta);
    z.b = mpas_sub(1.0f, mask);
    z.ab = mpas_mul(z.a, z.b);
    z.q = mpas_div(z.ab, z.den);
    return z;
}
__device__ __forceinline__ bool valid_div(float t0,float t1,float p0,float p1,float o0,float o1,
                                         float m,float c,DivLocal z,int* failure) {
    return ad_validity(failure,0,isfinite(t0))&&ad_validity(failure,1,isfinite(t1))&&ad_validity(failure,2,isfinite(p0))&&ad_validity(failure,3,isfinite(p1))&&ad_validity(failure,4,isfinite(o0))&&ad_validity(failure,5,isfinite(o1))
        && ad_validity(failure,6,isfinite(m))&&ad_validity(failure,7,isfinite(c))&&ad_validity(failure,8,isfinite(z.delta))&&ad_validity(failure,9,isfinite(z.den))&&ad_validity(failure,10,ad_f32_nonzero(z.den))
        && ad_validity(failure,11,isfinite(z.a))&&ad_validity(failure,12,isfinite(z.b))&&ad_validity(failure,13,isfinite(z.ab))&&ad_validity(failure,14,isfinite(z.q));
}

#if !defined(AD_ONLY_AD)
extern "C" __global__ void divergence_damping_f32__tl(
    const float* theta, const float* pp, const float* old, const int* cells_on_edge,
    const float* mask, float coefficient, int nsolve, int nlev, int ncells, int nedges,
    const float* dtheta, const float* dpp, const float* dold, const float* dmask,
    const float* dcoefficient, const float* dru_in, float* dru_out, int* failure) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= nedges) return;
    int c0=cells_on_edge[2*edge], c1=cells_on_edge[2*edge+1];
    bool update=(c0<nsolve || c1<nsolve);
    for(int k=0;k<nlev;++k) {
        int i=k*nedges+edge, a=k*ncells+c0, b=k*ncells+c1;
        float d=dru_in[i];
        if (!ad_validity(failure,15,isfinite(d))) { refuse(failure,0,1,6,i); dru_out[i]=0.0f; continue; }
        if (update) {
            DivLocal z=div_local(theta[a],theta[b],pp[a],pp[b],old[a],old[b],mask[edge],coefficient);
            if (!valid_div(theta[a],theta[b],pp[a],pp[b],old[a],old[b],mask[edge],coefficient,z,failure)
                || !ad_validity(failure,16,isfinite(dtheta[a])) || !ad_validity(failure,17,isfinite(dtheta[b])) || !ad_validity(failure,18,isfinite(dpp[a])) || !ad_validity(failure,19,isfinite(dpp[b]))
                || !ad_validity(failure,20,isfinite(dold[a])) || !ad_validity(failure,21,isfinite(dold[b])) || !ad_validity(failure,22,isfinite(dmask[edge])) || !ad_validity(failure,23,isfinite(*dcoefficient))) {
                refuse(failure,1,1u,1,i); dru_out[i]=0.0f; continue;
            }
            float ddelta=mpas_sub(mpas_sub(dpp[a],dold[a]),mpas_sub(dpp[b],dold[b]));
            float da=mpas_add(mpas_mul(*dcoefficient,z.delta),mpas_mul(coefficient,ddelta));
            float dab=mpas_sub(mpas_mul(da,z.b),mpas_mul(z.a,dmask[edge]));
            float dden=mpas_add(dtheta[a],dtheta[b]);
            float dq=mpas_div(mpas_sub(dab,mpas_mul(z.q,dden)),z.den);
            d=mpas_add(d,dq);
            if(!ad_validity(failure,24,isfinite(d))) { refuse(failure,3,3u,6,i); d=0.0f; }
        }
        dru_out[i]=d;
    }
}

#endif

// bru is a scheduler-owned immutable snapshot; the separate live seed
// is consumed only on the edges written by the forward.
// edges_on_cell and n_edges_on_cell are fixed reverse connectivity, validated by
// the caller as the inverse incidence of cells_on_edge; no atomics are used.
#if !defined(AD_ONLY_TL)
extern "C" __global__ void divergence_damping_f32__ad(
    const float* theta, const float* pp, const float* old, const int* cells_on_edge,
    const float* mask, float coefficient, int nsolve, int nlev, int ncells, int nedges,
    const float* bru, float* bru_in, const int* edges_on_cell, const int* n_edges_on_cell, int max_edges,
    float* btheta, float* bpp, float* bold, float* bmask, float* bcoefficient,
    int* failure, float* live_seed) {
    int owner=blockIdx.x*blockDim.x+threadIdx.x;
    int edge_slots=nlev*nedges, cell_slots=nlev*ncells;
    if(owner < nedges) {
        for(int k=0;k<nlev;++k) bru_in[k*nedges+owner]=bru[k*nedges+owner];
        int e=owner,c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
        if(c0<nsolve || c1<nsolve) for(int k=0;k<nlev;++k) live_seed[k*nedges+e]=0.0f;
        double bm=0.0;
        for(int k=0;k<nlev;++k) {
            int i=k*nedges+e,a=k*ncells+c0,b=k*ncells+c1;
            float w=bru[i];
            if(!ad_validity(failure,25,isfinite(w))) { refuse(failure,4,1u,6,i); continue; }
            if(c0>=nsolve && c1>=nsolve) continue;
            DivLocal z=div_local(theta[a],theta[b],pp[a],pp[b],old[a],old[b],mask[e],coefficient);
            if(!valid_div(theta[a],theta[b],pp[a],pp[b],old[a],old[b],mask[e],coefficient,z,failure)) {
                refuse(failure,5,1u,1,i); continue;
            }
            double bn=mpas_f32_to_f64_ieee(w)/mpas_f32_to_f64_ieee(z.den);
            bm-=bn*mpas_f32_to_f64_ieee(z.a);
        }
        float v=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(bmask[e])+bm);
        if(!ad_validity(failure,26,isfinite(v))) refuse(failure,7,3u,4,e); else bmask[e]=v;
    }
    if(owner < ncells) {
        int cell=owner;
        for(int k=0;k<nlev;++k) {
            int tid=k*ncells+cell;
            double bt=0.0,bp=0.0;
            for(int s=0;s<n_edges_on_cell[cell];++s) {
                int e=edges_on_cell[cell*max_edges+s];
                int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
                if(c0>=nsolve && c1>=nsolve) continue;
                int a=k*ncells+c0,b=k*ncells+c1;
                DivLocal z=div_local(theta[a],theta[b],pp[a],pp[b],old[a],old[b],mask[e],coefficient);
                float w=bru[k*nedges+e];
                double bd=-mpas_f32_to_f64_ieee(w)*mpas_f32_to_f64_ieee(z.q)/mpas_f32_to_f64_ieee(z.den);
                double bpedge=(mpas_f32_to_f64_ieee(w)/mpas_f32_to_f64_ieee(z.den))*mpas_f32_to_f64_ieee(z.b)*mpas_f32_to_f64_ieee(coefficient);
                if(cell==c0) { bt+=bd; bp+=bpedge; }
                if(cell==c1) { bt+=bd; bp-=bpedge; }
            }
            float x=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(btheta[tid])+bt), y=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(bpp[tid])+bp), q=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(bold[tid])-bp);
            if(!ad_validity(failure,27,isfinite(x))||!ad_validity(failure,28,isfinite(y))||!ad_validity(failure,29,isfinite(q))) refuse(failure,8,3u,1,tid);
            else { btheta[tid]=x; bpp[tid]=y; bold[tid]=q; }
        }
    }
    if(blockIdx.x==0) {
        __shared__ double bcparts[256];
        double bc=0.0;
        for(int e=threadIdx.x;e<nedges;e+=blockDim.x) {
            int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
            if(c0>=nsolve && c1>=nsolve) continue;
            for(int k=0;k<nlev;++k) {
                int a=k*ncells+c0,b=k*ncells+c1;
                DivLocal z=div_local(theta[a],theta[b],pp[a],pp[b],old[a],old[b],mask[e],coefficient);
                float w=bru[k*nedges+e];
                bc+=(mpas_f32_to_f64_ieee(w)/mpas_f32_to_f64_ieee(z.den))*mpas_f32_to_f64_ieee(z.b)*mpas_f32_to_f64_ieee(z.delta);
            }
        }
        bcparts[threadIdx.x]=bc;
        __syncthreads();
        for (int stride=128;stride>0;stride/=2) {
            if(threadIdx.x<stride) bcparts[threadIdx.x]+=bcparts[threadIdx.x+stride];
            __syncthreads();
        }
        if(threadIdx.x==0) {
            float x=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(*bcoefficient)+bcparts[0]);
            if(!ad_validity(failure,30,isfinite(x))) refuse(failure,9,3u,5,0); else *bcoefficient=x;
        }
    }
}

#endif
