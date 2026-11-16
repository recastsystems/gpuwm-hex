// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "horizontal_rrfs_common.cuh"
// The scheduler validates the reciprocal cell/edge topology once before use.
__device__ void mass_tl(const float *rho, const int *eoc, const int *nec,
    const int *coe, const float *dv, const float *ia, int nl, int nc, int ne,
    int me, const float *out, const float *drho, float *dout, HorizontalFailure *fail) {
    for (long long i=h_tid(); i<(long long)nl*nc; i+=h_stride()) {
        const int c=i%nc, k=i/nc, count=nec[c];
        float sum=0.0f;
        h_finite(out[i],fail,i,11,i,1);
        h_finite(ia[c],fail,i,6,c,1);
        for (int s=0;s<count;++s) {
            int e=eoc[c*me+s]; long long p=(long long)k*ne+e;
            h_finite(rho[p],fail,i,1,p,1);
            h_finite(drho[p],fail,i,1,p,1);
            h_finite(dv[e],fail,i,5,e,1);
            const float weight=h_mul(h_sign(c,e,coe),dv[e]);
            sum=h_add(sum,h_mul(weight,drho[p]));
        }
        const float result=h_mul(sum,ia[c]);
        h_finite(result,fail,i,11,i,1); dout[i]=result;
    }
}
__device__ void mass_ad(const float *rho, const int *eoc, const int *nec,
    const int *coe, const float *dv, const float *ia, int nl, int nc, int ne,
    int me, const float *out, float *bout, float *brho, HorizontalFailure *fail) {
    // A single block supplies a grid-wide barrier before consuming output seeds.
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1) {
        if (h_tid()==0) horizontal_fail(fail,0,11,0,2);
        return;
    }
    for (long long i=h_tid();i<(long long)nl*ne;i+=h_stride()) {
        int e=i%ne,k=i/ne; double acc=(double)brho[i];
        h_finite(rho[i],fail,i,1,i,1); h_finite(brho[i],fail,i,1,i,1);
        h_finite(dv[e],fail,i,5,e,1);
        int first=coe[2*e],second=coe[2*e+1];
        if(first>second) { int t=first; first=second; second=t; }
        for (int j=0;j<2;++j) {
            int c=j?second:first;
            if (c<0 || c>=nc || (j && second==first)) continue;
            long long o=(long long)k*nc+c;
            h_finite(out[o],fail,i,11,o,1);
            h_finite(bout[o],fail,i,11,o,1); h_finite(ia[c],fail,i,6,c,1);
            for (int s=0;s<nec[c];++s) if (eoc[c*me+s]==e) {
                float scaled_seed=h_mul(bout[o],ia[c]);
                float weight=h_mul(h_sign(c,e,coe),dv[e]);
                float t=h_mul(weight,scaled_seed);
                acc+=(double)t;
            }
        }
        const float result=(float)acc;
        h_finite(result,fail,i,1,i,1); brho[i]=result;
    }
    __syncthreads();
    for (long long i=threadIdx.x;i<(long long)nl*nc;i+=blockDim.x) bout[i]=0.0f;
}
