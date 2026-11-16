// SPDX-License-Identifier: MIT
#include "common.cuh"
#define AD_TRACE_ADD_SUBNORMAL 6
#define AD_TRACE_ADD_CANCEL 7
#define AD_TRACE_MUL_NONZERO 8
#define AD_TRACE_DIV_UNDERFLOW 9
#include "branch_trace.cuh"

__device__ __forceinline__ void trace_forward(int edge,int nlev,int ncells,int nmass,const int *cells,const float *x,uint32_t *r) {
 int a=cells[2*edge],b=cells[2*edge+1];ad_branch(r,0,true);if(a<0||a>=ncells||b<0||b>=ncells)return;
 for(int k=0;k<nlev;k++){
  float sum=0.0f;
  for(int s=0;s<nmass;s++){int j=(s*nlev+k)*ncells;float v=x[j+a],w=x[j+b];bool f=isfinite(v),g=true;ad_branch(r,20,f);if(f){g=isfinite(w);ad_branch(r,21,g);}ad_branch(r,1,!f||!g);float q=ad_add(v,w,r,3);sum=ad_add(sum,ad_mul(0.5f,q,r,4),r,3);}
  float v=ad_div(1.0f,ad_add(1.0f,sum,r,3),r,5);bool f=isfinite(sum),g=true;ad_branch(r,22,f);if(f){g=isfinite(v);ad_branch(r,23,g);}ad_branch(r,2,!f||!g);
 }
}

extern "C" __global__ void moist_edge_coefficients_v841_f32__tl(
    int nlev, int ncells, int nedges, int nmass,
    const int *cells_on_edge, const float *scalars, float *cqu, int *invalid,
    const float *d_scalars, float *d_cqu, int *failure, uint32_t *branch_record)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<nedges;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,nmass,cells_on_edge,scalars,branch_record);
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= nedges) return;
    if (nmass < 0) { ad_refuse(failure, edge, 6, 0); return; }
    else if (*invalid != 0) { ad_refuse(failure, edge, 6, 1); return; }
    const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
    if (c1 < 0) { ad_refuse(failure, edge, 5, 2); return; }
    else if (c1 >= ncells) { ad_refuse(failure, edge, 5, 3); return; }
    else if (c2 < 0) { ad_refuse(failure, edge, 5, 4); return; }
    else if (c2 >= ncells) { ad_refuse(failure, edge, 5, 5); return; }
    for (int k = 0; k < nlev; ++k) {
        float dt = 0.0f;
        for (int s = 0; s < nmass; ++s) {
            const int base = (s * nlev + k) * ncells;
            if (!isfinite(scalars[base + c1])) { ad_refuse(failure, edge, 6, 6); return; }
            else if (!isfinite(scalars[base + c2])) { ad_refuse(failure, edge, 6, 7); return; }
            else if (!isfinite(d_scalars[base + c1])) { ad_refuse(failure, edge, 6, 8); return; }
            else if (!isfinite(d_scalars[base + c2])) { ad_refuse(failure, edge, 6, 9); return; }
            dt += 0.5f * (d_scalars[base + c1] + d_scalars[base + c2]);
        }
        const int i = k * nedges + edge;
        const float result = -cqu[i] * cqu[i] * dt;
        if (!isfinite(cqu[i])) { ad_refuse(failure, edge, 7, 10); return; }
        else if (!isfinite(result)) { ad_refuse(failure, edge, 7, 11); return; }
        d_cqu[i] = result;
    }
}

// One deterministic gather per cell; edge index fixes the accumulation order.
__device__ __forceinline__ void edge_reverse_cell(int cell,int nlev,int ncells,int nedges,int nmass,
    const int *cells_on_edge,const float *scalars,float *cqu,int *invalid,
    const float *b_cqu,float *b_scalars,int *failure) {
    if (nmass < 0) { ad_refuse(failure, cell, 6, 0); return; }
    else if (*invalid != 0) { ad_refuse(failure, cell, 6, 1); return; }
    // Keep the common, low-degree case cheap. Higher-degree cells use the
    // same ascending-edge scan directly, without a limit on their degree.
    int incident[32], degree = 0;
    for (int e = 0; e < nedges; ++e) {
        const int c1 = cells_on_edge[2 * e], c2 = cells_on_edge[2 * e + 1];
        if (c1 < 0) { ad_refuse(failure, cell, 5, 2); return; }
        else if (c1 >= ncells) { ad_refuse(failure, cell, 5, 3); return; }
        else if (c2 < 0) { ad_refuse(failure, cell, 5, 4); return; }
        else if (c2 >= ncells) { ad_refuse(failure, cell, 5, 5); return; }
        if (c1 == cell || c2 == cell) {
            if (degree < 32) incident[degree] = e;
            ++degree;
        }
    }
    for (int k = 0; k < nlev; ++k) {
        double b = 0.0;
        const int scan_count = degree <= 32 ? degree : nedges;
        for (int slot = 0; slot < scan_count; ++slot) {
            const int e = degree <= 32 ? incident[slot] : slot;
            const int c1 = cells_on_edge[2 * e], c2 = cells_on_edge[2 * e + 1];
            if (c1 != cell && c2 != cell) continue;
            const int multiplicity = (c1 == cell) + (c2 == cell);
            const int i = k * nedges + e;
            if (!isfinite(cqu[i])) { ad_refuse(failure, cell, 7, 6); return; }
            else if (!isfinite(b_cqu[i])) { ad_refuse(failure, cell, 7, 7); return; }
            b -= (0.5 * (double)multiplicity) * (double)cqu[i] * cqu[i] * b_cqu[i];
        }
        for (int s = 0; s < nmass; ++s) {
            const int j = (s * nlev + k) * ncells + cell;
            if (!isfinite(scalars[j])) { ad_refuse(failure, cell, 6, 8); return; }
            else if (!isfinite(b)) { ad_refuse(failure, cell, 6, 9); return; }
            const float acc = (float)((double)b_scalars[j] + b);
            if (!isfinite(acc)) { ad_refuse(failure, cell, 6, 10); return; }
            b_scalars[j] = acc;
        }
    }
}

extern "C" __global__ void moist_edge_coefficients_v841_f32__ad(
    int nlev, int ncells, int nedges, int nmass,
    const int *cells_on_edge, const float *scalars, float *cqu, int *invalid,
    float *b_cqu, float *b_scalars,
    int *failure, uint32_t *branch_record, unsigned int *completed)
{
    for(int tr=blockIdx.x*blockDim.x+threadIdx.x;tr<nedges;tr+=blockDim.x*gridDim.x) trace_forward(tr,nlev,ncells,nmass,cells_on_edge,scalars,branch_record);
    for(int cell=blockIdx.x*blockDim.x+threadIdx.x;cell<ncells;cell+=blockDim.x*gridDim.x)
        edge_reverse_cell(cell,nlev,ncells,nedges,nmass,cells_on_edge,scalars,cqu,invalid,
                          b_cqu,b_scalars,failure);
    if (ad_last_block(completed))
        for (int i = threadIdx.x; i < nlev*nedges; i += blockDim.x)
            b_cqu[i] = 0.0f;
}
