// SPDX-License-Identifier: Apache-2.0
#include "horizontal_rrfs_common.cuh"
struct PressureValues { float denom, normal, terrain, dp_sum, p_diff; };
__device__ PressureValues pressure_values(long long i, int k, int e, int nc,
    const float *p, const float *dp, const float *cqu, const float *zz,
    const float *zxu, const int *coe, const float *invdc,
    HorizontalFailure *failure, long long slot) {
    int c0=coe[2*e],c1=coe[2*e+1]; long long a=(long long)k*nc+c0,b=(long long)k*nc+c1;
    h_finite(p[a],failure,slot,1,a,1); h_finite(p[b],failure,slot,1,b,1);
    h_finite(dp[a],failure,slot,2,a,1); h_finite(dp[b],failure,slot,2,b,1);
    h_finite(cqu[i],failure,slot,3,i,1); h_finite(zz[a],failure,slot,4,a,1);
    h_finite(zz[b],failure,slot,4,b,1); h_finite(zxu[i],failure,slot,5,i,1);
    h_finite(invdc[e],failure,slot,7,e,1);
    PressureValues v;
    v.denom=h_mul(0.5f,h_add(zz[b],zz[a]));
    if(!isfinite(v.denom) || v.denom==0.0f) horizontal_fail(failure,slot,4,a,2);
    v.p_diff=h_sub(p[b],p[a]);
    v.normal=h_div(h_mul(v.p_diff,invdc[e]),v.denom);
    v.dp_sum=h_add(dp[a],dp[b]);
    v.terrain=h_mul(h_mul(0.5f,zxu[i]),v.dp_sum);
    h_finite(v.normal,failure,slot,1,a,2); h_finite(v.terrain,failure,slot,2,a,2);
    return v;
}
extern "C" __global__ void pressure_gradient_v841_f32__tl(
    const float *pressure_p, const float *dpdz, const float *cqu, const float *zz,
    const float *zxu, const int *cells_on_edge, const float *inv_dc_edge,
    int nlev, int ncells, int nedges, float *tendency,
    const float *d_pressure_p, const float *d_dpdz, const float *d_cqu,
    const float *d_zz, const float *d_zxu, float *d_tendency, HorizontalFailure *failure) {
    for(long long i=h_tid();i<(long long)nlev*nedges;i+=h_stride()) {
        int e=i%nedges,k=i/nedges, c0=cells_on_edge[2*e], c1=cells_on_edge[2*e+1];
        long long a=(long long)k*ncells+c0,b=(long long)k*ncells+c1;
        h_finite(tendency[i],failure,i,11,i,1);
        PressureValues v=pressure_values(i,k,e,ncells,pressure_p,dpdz,cqu,zz,zxu,cells_on_edge,inv_dc_edge,failure,i);
        h_finite(d_pressure_p[a],failure,i,1,a,1); h_finite(d_pressure_p[b],failure,i,1,b,1);
        h_finite(d_dpdz[a],failure,i,2,a,1); h_finite(d_dpdz[b],failure,i,2,b,1);
        h_finite(d_cqu[i],failure,i,3,i,1); h_finite(d_zz[a],failure,i,4,a,1);
        h_finite(d_zz[b],failure,i,4,b,1); h_finite(d_zxu[i],failure,i,5,i,1);
        float dd=h_mul(0.5f,h_add(d_zz[b],d_zz[a]));
        float dn=h_sub(h_div(h_mul(h_sub(d_pressure_p[b],d_pressure_p[a]),inv_dc_edge[e]),v.denom),h_div(h_mul(v.normal,dd),v.denom));
        float dt=h_add(h_mul(h_mul(0.5f,d_zxu[i]),v.dp_sum),h_mul(h_mul(0.5f,zxu[i]),h_add(d_dpdz[a],d_dpdz[b])));
        float result=h_sub(h_mul(-d_cqu[i],h_sub(v.normal,v.terrain)),h_mul(cqu[i],h_sub(dn,dt)));
        h_finite(result,failure,i,11,i,2); d_tendency[i]=result;
    }
}
extern "C" __global__ void pressure_gradient_v841_f32__ad(
    const float *pressure_p, const float *dpdz, const float *cqu, const float *zz,
    const float *zxu, const int *cells_on_edge, const float *inv_dc_edge,
    int nlev, int ncells, int nedges, float *tendency,
    const int *cell_edge_offsets, const int *cell_edges,
    float *b_tendency, float *b_pressure_p, float *b_dpdz,
    float *b_cqu, float *b_zz, float *b_zxu, HorizontalFailure *failure) {
    // One block supplies the barrier before consuming overwritten output seeds.
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 || blockDim.y != 1 || blockDim.z != 1) {
        if (h_tid()==0) horizontal_fail(failure,0,11,0,3);
        return;
    }
    long long en=(long long)nlev*nedges, cn=(long long)nlev*ncells;
    for(long long i=h_tid();i<en;i+=h_stride()) {
        int e=i%nedges,k=i/nedges;
        h_finite(tendency[i],failure,i,11,i,1);
        PressureValues v=pressure_values(i,k,e,ncells,pressure_p,dpdz,cqu,zz,zxu,cells_on_edge,inv_dc_edge,failure,i);
        float w=b_tendency[i], bc=b_cqu[i], bz=b_zxu[i];
        h_finite(w,failure,i,11,i,1); h_finite(bc,failure,i,3,i,1); h_finite(bz,failure,i,5,i,1);
        bc=h_add(bc,h_mul(-w,h_sub(v.normal,v.terrain)));
        bz=h_add(bz,h_mul(h_mul(h_mul(w,cqu[i]),0.5f),v.dp_sum));
        h_finite(bc,failure,i,3,i,2); h_finite(bz,failure,i,5,i,2);
        b_cqu[i]=bc; b_zxu[i]=bz;
    }
    for(long long i=h_tid();i<cn;i+=h_stride()) {
        long long slot=en+i; int c=i%ncells,k=i/ncells;
        double bp=(double)b_pressure_p[i], bd=(double)b_dpdz[i], bz=(double)b_zz[i];
        h_finite(pressure_p[i],failure,slot,1,i,1); h_finite(dpdz[i],failure,slot,2,i,1);
        h_finite(zz[i],failure,slot,4,i,1); h_finite(b_pressure_p[i],failure,slot,1,i,1);
        h_finite(b_dpdz[i],failure,slot,2,i,1); h_finite(b_zz[i],failure,slot,4,i,1);
        for(int s=cell_edge_offsets[c];s<cell_edge_offsets[c+1];++s) {
            int e=cell_edges[s]; long long p=(long long)k*nedges+e;
            int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
            PressureValues v=pressure_values(p,k,e,ncells,pressure_p,dpdz,cqu,zz,zxu,cells_on_edge,inv_dc_edge,failure,slot);
            float w=b_tendency[p], bn=h_mul(-w,cqu[p]);
            h_finite(w,failure,slot,11,p,1);
            float pbar=h_mul(h_div(bn,v.denom),inv_dc_edge[e]);
            if(c==c0) bp-=(double)pbar;
            if(c==c1) bp+=(double)pbar;
            if(c==c0 || c==c1) {
                // If a degenerate edge names the cell twice, both occurrences contribute.
                int copies=(c==c0)+(c==c1);
                for(int t=0;t<copies;++t) {
                    bd+=(double)h_mul(h_mul(h_mul(w,cqu[p]),0.5f),zxu[p]);
                    bz+=(double)h_mul(-0.5f,h_div(h_mul(bn,v.normal),v.denom));
                }
            }
        }
        const float bp_out=(float)bp, bd_out=(float)bd, bz_out=(float)bz;
        h_finite(bp_out,failure,slot,1,i,2); h_finite(bd_out,failure,slot,2,i,2);
        h_finite(bz_out,failure,slot,4,i,2);
        b_pressure_p[i]=bp_out; b_dpdz[i]=bd_out; b_zz[i]=bz_out;
    }
    __syncthreads();
    for (long long i=threadIdx.x;i<(long long)nlev*nedges;i+=blockDim.x) { b_tendency[i]=0.0f; }
}
