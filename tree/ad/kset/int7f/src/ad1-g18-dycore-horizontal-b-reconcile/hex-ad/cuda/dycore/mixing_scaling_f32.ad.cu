// SPDX-License-Identifier: Apache-2.0
// Derivative of cuda_horizontal.py:468; the ordinary forward is unchanged.
#include "common.cuh"
#if !defined(AD_ONLY_AD) && !defined(AD_ONLY_AUG)
extern "C" __global__ void mixing_scaling_f32__tl(
    const int* cells_on_edge, const float* mesh_density, int nedges, int scale_with_mesh,
    float* del2, float* del4, const float* tape_pow2, const float* tape_pow4, const float* d_mesh_density, float* d_del2, float* d_del4,
    int* failures) {
    const int edge=blockIdx.x*blockDim.x+threadIdx.x;
    if (edge>=nedges) return;
    if (!scale_with_mesh) { d_del2[edge]=0.0f; d_del4[edge]=0.0f; return; }
    const int c0=cells_on_edge[2*edge], c1=cells_on_edge[2*edge+1];
    const float m=0.5f*(mesh_density[c0]+mesh_density[c1]);
    const float dm=0.5f*(d_mesh_density[c0]+d_mesh_density[c1]);
    if (!ad_finite(mesh_density[c0]) || !ad_finite(mesh_density[c1]) ||
        !ad_finite(d_mesh_density[c0]) || !ad_finite(d_mesh_density[c1]) || !ad_finite(m) || !(m>0.0f))
        ad_bad(failures,edge,2,edge,0);
    const float y2=tape_pow2[edge], y4=tape_pow4[edge];
    const float t2=(-0.25f/y2/m)*dm, t4=(-0.75f/y4/m)*dm;
    if (!(y2>0.0f) || !(y4>0.0f) || !ad_finite(y2) || !ad_finite(y4) || !ad_finite(t2) || !ad_finite(t4)) ad_bad(failures,edge,2,edge,1);
    d_del2[edge]=t2; d_del4[edge]=t4;
}
// The fixed reverse incidence has one entry per occurrence in cells_on_edge;
// entries within a cell are sorted by edge, with repeated self-incidences kept.
// The caller validates it against cells_on_edge before invoking this kernel.
#endif
#if !defined(AD_ONLY_TL) && !defined(AD_ONLY_AUG)
extern "C" __global__ void mixing_scaling_f32__ad(
    const int* cells_on_edge, const float* mesh_density, int nedges, int scale_with_mesh,
    float* del2, float* del4, const float* tape_pow2, const float* tape_pow4, float* b_mesh_density, float* b_del2, float* b_del4,
    int ncells, const int* cell_edge_offsets, const int* cell_edges, int* failures) {
    if (blockIdx.x != 0) return;
 const int thread=threadIdx.x;
    const int stride=blockDim.x;
    if (!scale_with_mesh) { for(int i=thread;i<nedges;i+=stride) { b_del2[i]=0.0f; b_del4[i]=0.0f; } return; }
    for (int cell=thread;cell<ncells;cell+=stride) {
        double b=mpas_f32_to_f64_ieee(b_mesh_density[cell]);
        if (!ad_finite(mesh_density[cell]) || !ad_finite(b)) ad_bad(failures,cell,2,cell,0);
        for (int q=cell_edge_offsets[cell];q<cell_edge_offsets[cell+1];q++) {
            const int edge=cell_edges[q];
            const int c0=cells_on_edge[2*edge], c1=cells_on_edge[2*edge+1];
            const float m=0.5f*(mesh_density[c0]+mesh_density[c1]);
            if (!ad_finite(m) || !(m>0.0f) || !ad_finite(b_del2[edge]) || !ad_finite(b_del4[edge]))
                ad_bad(failures,cell,2,edge,1);
            const float y2=tape_pow2[edge], y4=tape_pow4[edge];
            const double contribution=0.5*(mpas_f32_to_f64_ieee((-0.25f/y2/m)*b_del2[edge])+mpas_f32_to_f64_ieee((-0.75f/y4/m)*b_del4[edge]));
            b+=contribution;
            if (!(y2>0.0f) || !(y4>0.0f) || !ad_finite(y2) || !ad_finite(y4) || !ad_finite(contribution) || !ad_finite(b))
                ad_bad(failures,cell,2,edge,2);
        }
        b_mesh_density[cell]=mpas_f64_to_f32_ieee(b);
    }
    // A single block completes every deterministic gather before consuming seeds.
    __syncthreads();
    for (int i=threadIdx.x;i<nedges;i+=blockDim.x) { b_del2[i]=0.0f; b_del4[i]=0.0f; }
}

#endif

#if defined(AD_ONLY_AUG) || (!defined(AD_ONLY_AD) && !defined(AD_ONLY_TL))
// The wrapper stores the exact powf result used by the unchanged forward expression.
__device__ __forceinline__ float ad_capture_pow(float x,float exponent,float* p2,float* p4,int i) {
    float y=powf(x,exponent);
    if (exponent==0.25f) p2[i]=y; else p4[i]=y;
    return y;
}
#define powf(x,p) ad_capture_pow((x),(p),tape_pow2,tape_pow4,edge)
extern "C" __global__ void mixing_scaling_f32__aug(
    const int* cells_on_edge,
    const float* mesh_density,
    const int nedges,
    const int scale_with_mesh,
    float* del2,
    float* del4,
    float* tape_pow2,
    float* tape_pow4
) {
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    if (!scale_with_mesh) {
        del2[edge] = 1.0f;
        del4[edge] = 1.0f;
        return;
    }
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const float mean_density = 0.5f * (mesh_density[cell0] + mesh_density[cell1]);
    del2[edge] = 1.0f / powf(mean_density, 0.25f);
    del4[edge] = 1.0f / powf(mean_density, 0.75f);
}
#undef powf
#endif
