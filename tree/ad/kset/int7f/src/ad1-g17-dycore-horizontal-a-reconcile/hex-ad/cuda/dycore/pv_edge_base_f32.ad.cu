// SPDX-License-Identifier: Apache-2.0
// Tangent and deterministic transpose of cuda_horizontal.py:290.
// Compile each entry into its own cubin with --fmad=false --ftz=false.
#include "pv_edge_base_f32.branch.cuh"
#define AD_BRANCH_RECORD() do { \
  __shared__ unsigned int ad_local_counts[20]; \
  for(int ad_i=threadIdx.x;ad_i<20;ad_i+=blockDim.x) ad_local_counts[ad_i]=0; \
  __syncthreads(); \
  ad_branch_replay_pv_edge_base_f32(pv_vertex,vertices_on_edge,nlev,nedges,nvertices,pv_edge,ad_local_counts); \
  __syncthreads(); \
  for(int ad_i=threadIdx.x;ad_i<20;ad_i+=blockDim.x) if(ad_local_counts[ad_i]) atomicAdd(branch_record+ad_i,ad_local_counts[ad_i]); \
} while(0)
#if !defined(AD_BUILD_AD)
extern "C" __global__ void pv_edge_base_f32__branch(
    const float* pv_vertex,
    const int* vertices_on_edge,
    const int nlev,
    const int nedges,
    const int nvertices,
    float* pv_edge
, unsigned int* branch_record) { AD_BRANCH_RECORD(); }
#endif


__device__ __forceinline__ bool ad_refuse_slot(int32_t* f,int slot,bool cause){if(cause) atomicExch(f+slot,1);return cause;}



#if !defined(AD_BUILD_AD)
extern "C" __global__ void pv_edge_base_f32__tl(
    const float* pv_vertex, const int* vertices_on_edge,
    int nlev, int nedges, int nvertices, float* pv_edge,
    const float* d_pv_vertex, float* d_pv_edge, int32_t* failure
, unsigned int* branch_record) {
 AD_BRANCH_RECORD();
    (void)pv_edge;
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int a = vertices_on_edge[2 * edge];
    const int b = vertices_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int t = k * nedges + edge;
        if (ad_refuse_slot(failure,0,(a < 0 ))||ad_refuse_slot(failure,1,( a >= nvertices ))||ad_refuse_slot(failure,2,( b < 0 ))||ad_refuse_slot(failure,3,( b >= nvertices))) {
            (void)0; continue;
        }
        const int ia = k * nvertices + a, ib = k * nvertices + b;
        const float x = pv_vertex[ia], y = pv_vertex[ib];
        const float dx = d_pv_vertex[ia], dy = d_pv_vertex[ib];
        if (ad_refuse_slot(failure,4,(!isfinite(x) ))||ad_refuse_slot(failure,5,( !isfinite(y) ))||ad_refuse_slot(failure,6,( !isfinite(x + y) ))||ad_refuse_slot(failure,7,(
            !isfinite(dx) ))||ad_refuse_slot(failure,8,( !isfinite(dy) ))||ad_refuse_slot(failure,9,( !isfinite(dx + dy)))) {
            (void)0;
            continue;
        }
        d_pv_edge[t] = 0.5f * (dx + dy);
    }
}
#endif

// vertex_offsets/vertex_edges are a validated CSR transpose of vertices_on_edge.
// Entries include duplicates and are sorted by (edge, endpoint slot), so the
// accumulation order is independent of CUDA scheduling and needs no atomics.
#if !defined(AD_BUILD_TL)
extern "C" __global__ void pv_edge_base_f32__ad(
    const float* pv_vertex, const int* vertices_on_edge,
    int nlev, int nedges, int nvertices, float* pv_edge,
    const float* b_pv_edge, const int* vertex_offsets, const int* vertex_edges,
    float* b_pv_vertex, int32_t* failure
, unsigned int* branch_record) {
 AD_BRANCH_RECORD();
    (void)pv_edge;
    for (int vertex = blockDim.x * blockIdx.x + threadIdx.x;
         vertex < nvertices; vertex += blockDim.x * gridDim.x) {
      for (int k = 0; k < nlev; ++k) {
        const int t = k * nvertices + vertex;
        double sum = b_pv_vertex[t];
        bool bad = ad_refuse_slot(failure,22,(!isfinite(pv_vertex[t]) ))||ad_refuse_slot(failure,23,( !isfinite((float)sum)));
        for (int p = vertex_offsets[vertex]; p < vertex_offsets[vertex + 1]; ++p) {
            const int edge = vertex_edges[p];
            if (ad_refuse_slot(failure,10,(edge < 0 ))||ad_refuse_slot(failure,11,( edge >= nedges))) { bad = true; continue; }
            const int a = vertices_on_edge[2 * edge];
            const int b = vertices_on_edge[2 * edge + 1];
            if (ad_refuse_slot(failure,12,(a < 0 ))||ad_refuse_slot(failure,13,( a >= nvertices ))||ad_refuse_slot(failure,14,( b < 0 ))||ad_refuse_slot(failure,15,( b >= nvertices ))||ad_refuse_slot(failure,16,(
                (a != vertex && b != vertex)))) { bad = true; continue; }
            const float x = pv_vertex[k * nvertices + a];
            const float y = pv_vertex[k * nvertices + b];
            const float seed = b_pv_edge[k * nedges + edge];
            if (ad_refuse_slot(failure,17,(!isfinite(x) ))||ad_refuse_slot(failure,18,( !isfinite(y) ))||ad_refuse_slot(failure,19,( !isfinite(x + y) ))||ad_refuse_slot(failure,20,( !isfinite(seed)))) {
                bad = true; continue;
            }
            sum += 0.5f * seed;
            if (ad_refuse_slot(failure,21,(!isfinite((float)sum)))) bad = true;
        }
        if (bad) (void)0;
        else b_pv_vertex[t] = sum;
      }
    }
}
#endif
