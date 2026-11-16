// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 6
#define AD_TRACE_ADD_CANCEL 7
#define AD_TRACE_SUB_SUBNORMAL 8
#define AD_TRACE_SUB_CANCEL 9
#define AD_TRACE_MUL_NONZERO 10
#include "branch_trace.cuh"

__device__ __forceinline__ void trace_forward(int cell,int nlev,int ncells,const float *p,const float *d,const float *q,const float *dz,const float *m,const float *n,int mixing,const float *mix,uint32_t *r) {
 for(int k=0;k<=nlev;k++){int i=k*ncells+cell;bool edge=k==0||k==nlev;ad_branch(r,0,edge);if(edge)continue;
  int j=i-ncells;ad_branch(r,1,mixing!=0);float pd=ad_add(p[i],p[j],r,4,true);float pt=ad_mul(dz[k],pd,r,5);
  float buoy=ad_add(ad_mul(m[k],d[i],r,5),ad_mul(n[k],d[j],r,5),r,3);
  float v=ad_add(mixing?mix[i]:0.0f,ad_mul(q[i],ad_add(pt,buoy,r,4,true),r,5),r,4,true);
  bool f=isfinite(q[i]),g=true;ad_branch(r,27,f);if(f){g=isfinite(v);ad_branch(r,29,g);}ad_branch(r,2,!f||!g);
 }
}

extern "C" __global__ void euler_w_moist_v841_f32__tl(
    int nlev, int ncells, const float *pressure_p, const float *dpdz,
    const float *cqw, const float *rdzu, const float *fzm, const float *fzp,
    int has_mixing, const float *mixing, float *result, int *invalid,
    const float *d_pressure_p, const float *d_dpdz, const float *d_cqw,
    const float *d_rdzu, const float *d_fzm, const float *d_fzp,
    const float *d_mixing, float *d_result, int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,pressure_p,dpdz,cqw,rdzu,fzm,fzp,has_mixing,mixing,branch_record);
    (void)result;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    if (*invalid) { ad_refuse(failure, c, 3, 0); return; }
    d_result[c] = 0.0f;
    for (int k = 1; k < nlev; ++k) {
        const int i = k*ncells+c, j = i-ncells;
        const float p = pressure_p[i]-pressure_p[j];
        const float b = fzm[k]*dpdz[i]+fzp[k]*dpdz[j];
        const float q = rdzu[k]*p-b;
        const float dq = d_rdzu[k]*p + rdzu[k]*(d_pressure_p[i]-d_pressure_p[j]) -
                         d_fzm[k]*dpdz[i] - fzm[k]*d_dpdz[i] -
                         d_fzp[k]*dpdz[j] - fzp[k]*d_dpdz[j];
        const float out = (has_mixing ? d_mixing[i] : 0.0f) - d_cqw[i]*q - cqw[i]*dq;
        const float original = (has_mixing ? mixing[i] : 0.0f) - cqw[i]*q;
        if (!isfinite(p)) { ad_refuse(failure, c, 5, 1); return; }
        else if (!isfinite(b)) { ad_refuse(failure, c, 5, 2); return; }
        else if (!isfinite(q)) { ad_refuse(failure, c, 5, 3); return; }
        else if (!isfinite(out)) { ad_refuse(failure, c, 5, 4); return; }
        else if (!isfinite(cqw[i])) { ad_refuse(failure, c, 5, 5); return; }
        else if (!isfinite(original)) { ad_refuse(failure, c, 5, 6); return; }
        d_result[i] = out;
    }
    d_result[nlev*ncells+c] = 0.0f;
}

extern "C" __global__ void euler_w_moist_v841_f32__ad(
    int nlev, int ncells, const float *pressure_p, const float *dpdz,
    const float *cqw, const float *rdzu, const float *fzm, const float *fzp,
    int has_mixing, const float *mixing, float *result, int *invalid,
    float *b_result, float *b_pressure_p, float *b_dpdz, float *b_cqw,
    float *b_rdzu, float *b_fzm, float *b_fzp, float *b_mixing,
    int *failure, uint32_t *branch_record, unsigned int *completed)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<ncells;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,pressure_p,dpdz,cqw,rdzu,fzm,fzp,has_mixing,mixing,branch_record);
    (void)result;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < ncells) {
        if (*invalid) { ad_refuse(failure, c, 3, 0); }
        else {
            // Each thread owns one column, so vertical transpose writes cannot race.
            for (int k = 1; k < nlev; ++k) {
                const int i = k*ncells+c, j = i-ncells;
                const float p = pressure_p[i]-pressure_p[j];
                const float buoy = fzm[k]*dpdz[i]+fzp[k]*dpdz[j];
                const float q = rdzu[k]*p-buoy, b = b_result[i];
                const float orig = (has_mixing ? mixing[i] : 0.0f)-cqw[i]*q;
                if (!isfinite(p)) { ad_refuse(failure, c, 5, 1); break; }
                else if (!isfinite(buoy)) { ad_refuse(failure, c, 5, 2); break; }
                else if (!isfinite(q)) { ad_refuse(failure, c, 5, 3); break; }
                else if (!isfinite(b)) { ad_refuse(failure, c, 5, 4); break; }
                else if (!isfinite(cqw[i])) { ad_refuse(failure, c, 5, 5); break; }
                else if (!isfinite(orig)) { ad_refuse(failure, c, 5, 6); break; }
                const float acc_q = b_cqw[i] - b*q;
                const float acc_mix = has_mixing ? b_mixing[i] + b : 0.0f;
                if (!isfinite(acc_q)) { ad_refuse(failure, c, 5, 10); break; }
                else if (has_mixing && !isfinite(acc_mix)) { ad_refuse(failure, c, 5, 11); break; }
                b_cqw[i] = acc_q;
                if (has_mixing) b_mixing[i] = acc_mix;
            }
            // Each level appears in up to two adjacent interfaces. Gather both
            // contributions in a fixed order and round once into each f32 bar.
            for (int k = 0; k < nlev; ++k) {
                const int i = k*ncells+c;
                double bp = (double)b_pressure_p[i], bd = (double)b_dpdz[i];
                if (k > 0) {
                    const double v = (double)b_result[i]*cqw[i];
                    bp -= v*rdzu[k];
                    bd += v*fzm[k];
                }
                if (k+1 < nlev) {
                    const int next = i+ncells;
                    const double v = (double)b_result[next]*cqw[next];
                    bp += v*rdzu[k+1];
                    bd += v*fzp[k+1];
                }
                const float acc_p = (float)bp, acc_d = (float)bd;
                if (!isfinite(acc_p)) { ad_refuse(failure, c, 5, 12); break; }
                else if (!isfinite(acc_d)) { ad_refuse(failure, c, 5, 13); break; }
                b_pressure_p[i] = acc_p; b_dpdz[i] = acc_d;
            }
        }
    }
    // One block owns each level; the grid may have fewer blocks than levels.
    // All threads participate, including the lanes beyond ncells.
    __shared__ double sums[3][1024];
    const int t = threadIdx.x;
    for (int k = 1 + blockIdx.x; k < nlev; k += gridDim.x) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (int cell = t; cell < ncells; cell += blockDim.x) {
            const int i = k*ncells+cell, j = i-ncells;
            const double v = (double)b_result[i]*cqw[i];
            s0 -= v*(pressure_p[i]-pressure_p[j]);
            s1 += v*dpdz[i];
            s2 += v*dpdz[j];
        }
        sums[0][t] = s0; sums[1][t] = s1; sums[2][t] = s2;
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if ((t % (2*stride)) == 0 && t+stride < blockDim.x) {
                sums[0][t] += sums[0][t+stride];
                sums[1][t] += sums[1][t+stride];
                sums[2][t] += sums[2][t+stride];
            }
            __syncthreads();
        }
        if (t == 0) {
            if (!isfinite(sums[0][0])) { ad_refuse(failure, ncells + blockIdx.x, 6, 7); }
            else if (!isfinite(sums[1][0])) { ad_refuse(failure, ncells + blockIdx.x, 6, 8); }
            else if (!isfinite(sums[2][0])) { ad_refuse(failure, ncells + blockIdx.x, 6, 9); }
            else {
                const float acc_dz = (float)((double)b_rdzu[k] + sums[0][0]);
                const float acc_m = (float)((double)b_fzm[k] + sums[1][0]);
                const float acc_p = (float)((double)b_fzp[k] + sums[2][0]);
                if (!isfinite(acc_dz)) { ad_refuse(failure, ncells + blockIdx.x, 6, 14); }
                else if (!isfinite(acc_m)) { ad_refuse(failure, ncells + blockIdx.x, 6, 15); }
                else if (!isfinite(acc_p)) { ad_refuse(failure, ncells + blockIdx.x, 6, 16); }
                else { b_rdzu[k] = acc_dz; b_fzm[k] = acc_m; b_fzp[k] = acc_p; }
            }
        }
        __syncthreads();
    }
    if (ad_last_block(completed))
        for (int i = threadIdx.x; i < (nlev+1)*ncells; i += blockDim.x)
            b_result[i] = 0.0f;
}
