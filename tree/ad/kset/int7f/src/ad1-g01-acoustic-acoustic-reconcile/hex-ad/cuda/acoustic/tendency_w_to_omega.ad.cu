#if defined(AD_BUILD_BRANCH)
#include "tendency_w_to_omega.branch.cu"
#else
// SPDX-License-Identifier: Apache-2.0
// Derivatives of cuda_acoustic.py:115-150. Compile with --fmad=false and
// --ftz=false. The forward's mpas helpers preserve IEEE f32 subnormals.
typedef unsigned int AdFailureRecord;
__device__ __forceinline__ bool ad_ok(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ void ad_fail(AdFailureRecord *r, unsigned int slot, unsigned long long field, unsigned long long index) { (void)field; (void)index; atomicExch(r+slot,1u); }
__device__ __forceinline__ float ad_sign(float x) {
    return __uint_as_float(0x3f800000u | (__float_as_uint(x) & 0x80000000u));
}

#ifndef AD_ONLY_AD
extern "C" __global__ void tendency_w_to_omega__tl(
    int nlev, int ncells, int nedges, int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *edge_sign_on_cell, const float *u_tendency,
    const float *fzm, const float *fzp, const float *zz,
    const float *zb, const float *zb3, const float *omega_tendency,
    const float *d_edge_sign_on_cell, const float *d_u_tendency,
    const float *d_fzm, const float *d_fzp, const float *d_zz,
    const float *d_zb, const float *d_zb3, float *d_omega_tendency,
    AdFailureRecord *records)
{
    int total=(nlev-1)*ncells;
    for (int row=blockIdx.x*blockDim.x+threadIdx.x; row<total; row+=gridDim.x*blockDim.x) {
        int cell=row%ncells, k=1+row/ncells, i=k*ncells+cell, im=i-ncells;
        int count=n_edges_on_cell[cell];
        if (count<0 || count>max_edges) { ad_fail(records, 0u,10u,cell); continue; }
        float o=omega_tendency[i], d=d_omega_tendency[i];
        if (!ad_ok(o)||!ad_ok(d)) ad_fail(records, 1u,8u,i);
        for (int slot=0; slot<count; ++slot) {
            int cs=cell*max_edges+slot, edge=edges_on_cell[cs];
            if (edge<0 || edge>=nedges) continue;
            int ui=k*nedges+edge, umi=ui-nedges, zi=(i*max_edges)+slot;
            float ut=u_tendency[ui], up=u_tendency[umi], s=edge_sign_on_cell[cs];
            float q=fzm[k]*ut+fzp[k]*up;
            float flux=s*q;
            float g=zb[zi]+ad_sign(ut)*zb3[zi];
            float dq=d_fzm[k]*ut+fzm[k]*d_u_tendency[ui]
                    +d_fzp[k]*up+fzp[k]*d_u_tendency[umi];
            float df=d_edge_sign_on_cell[cs]*q+s*dq;
            float dg=d_zb[zi]+ad_sign(ut)*d_zb3[zi];
            o-=g*flux;
            d-=dg*flux+g*df;
            if (!ad_ok(ut)||!ad_ok(up)||!ad_ok(s)||!ad_ok(q)||!ad_ok(g)||!ad_ok(d))
                ad_fail(records, 2u,2u,ui);
        }
        float metric=fzm[k]*zz[i]+fzp[k]*zz[im];
        float dm=d_fzm[k]*zz[i]+fzm[k]*d_zz[i]+d_fzp[k]*zz[im]+fzp[k]*d_zz[im];
        float out=d*metric+o*dm;
        if (!ad_ok(metric)||!ad_ok(dm)||!ad_ok(out)) ad_fail(records, 3u,5u,i);
        d_omega_tendency[i]=out;
    }
}
#endif

#ifndef AD_ONLY_TL
extern "C" __global__ void tendency_w_to_omega__ad(
    int nlev, int ncells, int nedges, int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *edge_sign_on_cell, const float *u_tendency,
    const float *fzm, const float *fzp, const float *zz,
    const float *zb, const float *zb3, const float *omega_tendency,
    float *s_omega_tendency,
    float *b_edge_sign_on_cell, float *b_u_tendency,
    float *b_fzm, float *b_fzp, float *b_zz,
    float *b_zb, float *b_zb3, float *b_omega_tendency,
    const int *edge_offsets, const int *edge_cell_slots,
    double *work_bm, double *work_bq, double *work_bs,
    AdFailureRecord *records)
{
    // One block is required for the stage barrier. Its threads own disjoint
    // output rows, then disjoint input elements. The scheduler validates the
    // CSR transpose (ascending cell/slot order) before making it immutable.
    const int t=threadIdx.x, nth=blockDim.x;
    const int rows=(nlev-1)*ncells;
    for (int row=t; row<rows; row+=nth) {
        const int cell=row%ncells, k=1+row/ncells, i=k*ncells+cell, im=i-ncells;
        int count=n_edges_on_cell[cell];
        if (count<0 || count>max_edges) { ad_fail(records, 4u,10u,cell); count=0; }
        float o=omega_tendency[i];
        for (int slot=0; slot<max_edges; ++slot) {
            const int cs=cell*max_edges+slot, zi=i*max_edges+slot;
            work_bq[zi]=0.0f; work_bs[zi]=0.0f;
            if (slot>=count) continue;
            const int edge=edges_on_cell[cs];
            if (edge<0 || edge>=nedges) continue;
            float ut=u_tendency[k*nedges+edge];
            float q=fzm[k]*ut+fzp[k]*u_tendency[(k-1)*nedges+edge];
            float flux=edge_sign_on_cell[cs]*q;
            float g=zb[zi]+ad_sign(ut)*zb3[zi];
            o-=g*flux;
            if (!ad_ok(ut)||!ad_ok(q)||!ad_ok(flux)||!ad_ok(g)||!ad_ok(o))
                ad_fail(records, 5u,2u,k*nedges+edge);
        }
        const float metric=fzm[k]*zz[i]+fzp[k]*zz[im];
        const float bout=s_omega_tendency[i];
        const double bo=(double)bout*metric;
        work_bm[i]=(double)bout*o;
        b_omega_tendency[i]=(float)((double)b_omega_tendency[i]+bo);
        if (!ad_ok(metric)||!ad_ok(bout)||!ad_ok(bo)||!ad_ok(work_bm[i]))
            ad_fail(records, 6u,8u,i);
        for (int slot=0; slot<count; ++slot) {
            const int cs=cell*max_edges+slot, edge=edges_on_cell[cs], zi=i*max_edges+slot;
            if (edge<0 || edge>=nedges) continue;
            const float ut=u_tendency[k*nedges+edge];
            const float q=fzm[k]*ut+fzp[k]*u_tendency[(k-1)*nedges+edge];
            const float flux=edge_sign_on_cell[cs]*q;
            const float g=zb[zi]+ad_sign(ut)*zb3[zi];
            const double bg=-bo*flux, bf=-bo*g;
            const double bz=(double)b_zb[zi]+bg, bz3=(double)b_zb3[zi]+ad_sign(ut)*bg;
            work_bs[zi]=bf*q;
            work_bq[zi]=bf*edge_sign_on_cell[cs];
            b_zb[zi]=bz; b_zb3[zi]=bz3;
            if (!ad_ok(bz)||!ad_ok(bz3)||!ad_ok(work_bs[zi])||!ad_ok(work_bq[zi]))
                ad_fail(records, 7u,6u,zi);
        }
        s_omega_tendency[i]=0.0f;
    }
    // The original kernel leaves the two outside levels unchanged.
    for (int cell=t; cell<ncells; cell+=nth) {
        b_omega_tendency[cell]+=s_omega_tendency[cell];
        b_omega_tendency[nlev*ncells+cell]+=s_omega_tendency[nlev*ncells+cell];
    }
    __syncthreads();
    // Metric interpolation transposed to zz. Each (level,cell) is unique.
    for (int i=t; i<nlev*ncells; i+=nth) {
        const int k=i/ncells, cell=i%ncells;
        double sum=b_zz[i];
        if (k>=1) sum+=work_bm[i]*fzm[k];
        if (k+1<nlev) sum+=work_bm[(k+1)*ncells+cell]*fzp[k+1];
        if (!ad_ok(sum)) ad_fail(records, 8u,5u,i);
        b_zz[i]=sum;
    }
    // The sign is shared by every level in a cell slot.
    for (int cs=t; cs<ncells*max_edges; cs+=nth) {
        const int cell=cs/max_edges, slot=cs%max_edges;
        double sum=b_edge_sign_on_cell[cs];
        for (int k=1; k<nlev; ++k) sum+=work_bs[(k*ncells+cell)*max_edges+slot];
        if (!ad_ok(sum)) ad_fail(records, 9u,1u,cs);
        b_edge_sign_on_cell[cs]=sum;
    }
    // Transpose the edge gathers via the validated edge -> (cell,slot) CSR.
    for (int ui=t; ui<nlev*nedges; ui+=nth) {
        const int k=ui/nedges, edge=ui%nedges;
        double sum=b_u_tendency[ui];
        for (int p=edge_offsets[edge]; p<edge_offsets[edge+1]; ++p) {
            const int cs=edge_cell_slots[p], cell=cs/max_edges, slot=cs%max_edges;
            if (k>=1) sum+=work_bq[(k*ncells+cell)*max_edges+slot]*fzm[k];
            if (k+1<nlev) sum+=work_bq[((k+1)*ncells+cell)*max_edges+slot]*fzp[k+1];
        }
        if (!ad_ok(sum)) ad_fail(records, 10u,2u,ui);
        b_u_tendency[ui]=sum;
    }
    // Each level thread accumulates in ascending cell/slot order.
    for (int k=t+1; k<nlev; k+=nth) {
        double m=b_fzm[k], p=b_fzp[k];
        for (int cell=0; cell<ncells; ++cell) {
            const int i=k*ncells+cell;
            m+=work_bm[i]*zz[i];
            p+=work_bm[i]*zz[i-ncells];
            const int count=n_edges_on_cell[cell];
            if (count<0 || count>max_edges) { ad_fail(records, 11u,10u,cell); continue; }
            for (int slot=0; slot<count; ++slot) {
                const int edge=edges_on_cell[cell*max_edges+slot];
                if (edge<0 || edge>=nedges) continue;
                const double bq=work_bq[i*max_edges+slot];
                m+=bq*u_tendency[k*nedges+edge];
                p+=bq*u_tendency[(k-1)*nedges+edge];
            }
        }
        if (!ad_ok(m)||!ad_ok(p)) ad_fail(records, 12u,3u,k);
        b_fzm[k]=m; b_fzp[k]=p;
    }
}
#endif

#endif // AD_BUILD_BRANCH
