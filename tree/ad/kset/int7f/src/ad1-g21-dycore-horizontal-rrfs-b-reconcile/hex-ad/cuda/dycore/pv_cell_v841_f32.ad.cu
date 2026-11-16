// SPDX-License-Identifier: Apache-2.0
// Pinned hex pv_cell_v841_f32: fixed mesh, active vertex potential vorticity.
#include "hex_mpas_fp32_ad.cuh"
#include "hex_rrfs_branch_ad.cuh"

using AdFailureRecord = int;

__device__ __forceinline__ int ad_kite(int v, int c, const int *cov, int vd) {
    for (int j = 0; j < vd; ++j) if (cov[v * vd + j] == c) return j;
    return -1;
}

// One deterministic integer flag per distinct refusal source. Multiple
// threads may detect the same source; the idempotent atomic OR is exact.
__device__ __forceinline__ void ad_fail(AdFailureRecord *r, int slot, bool bad) {
    if (bad) atomicOr(r + slot, 1);
}

#if !defined(AD_ONLY_AD)
extern "C" __global__ void pv_cell_v841_f32__tl(
    const float *pv_vertex, const int *vertices_on_cell,
    const int *n_edges_on_cell, const int *cells_on_vertex,
    const float *kite_areas, const float *inv_area_cell,
    const int nlev, const int ncells, const int nvertices,
    const int max_edges, const int vertex_degree, const float *pv_cell,
    const float *d_pv_vertex, float *d_pv_cell, AdFailureRecord *failure, unsigned int *branch_record)
{
    (void)pv_cell;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < nlev * ncells; i += gridDim.x * blockDim.x) {
        const int cell = i % ncells, k = i / ncells;
        rrfs_pvc_record(i,pv_vertex,vertices_on_cell,n_edges_on_cell,cells_on_vertex,kite_areas,inv_area_cell,ncells,nvertices,max_edges,vertex_degree,branch_record);
        float value = 0.0f, deriv = 0.0f;
        ad_fail(failure, 2, !isfinite(inv_area_cell[cell]));
        for (int j = 0; j < n_edges_on_cell[cell]; ++j) {
            const int v = vertices_on_cell[cell * max_edges + j];
            const int kite = ad_kite(v, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                const float area = kite_areas[v * vertex_degree + kite];
                const float p = pv_vertex[k * nvertices + v];
                const float d = d_pv_vertex[k * nvertices + v];
                const float q0 = mpas_mul(area, p);
                const float q = mpas_mul(q0, inv_area_cell[cell]);
                const float dq0 = mpas_mul(area, d);
                const float dq = mpas_mul(dq0, inv_area_cell[cell]);
                value = mpas_add(value, q);
                deriv = mpas_add(deriv, dq);
                ad_fail(failure, 0, !isfinite(area));
                ad_fail(failure, 1, !isfinite(p));
                ad_fail(failure, 3, !isfinite(q0));
                ad_fail(failure, 4, !isfinite(q));
                ad_fail(failure, 5, !isfinite(value));
                ad_fail(failure, 6, !isfinite(d));
                ad_fail(failure, 7, !isfinite(dq0));
                ad_fail(failure, 8, !isfinite(dq));
                ad_fail(failure, 9, !isfinite(deriv));
            }
        }
        d_pv_cell[i] = deriv;
    }
}

#endif

#if !defined(AD_ONLY_TL)
// seed_pv_cell is an immutable snapshot. b_pv_cell is the consumed mutable
// output cotangent. The binder creates disjoint arrays before this launch.
// cells_on_vertex is already a forward input and gives a bounded transpose
// gather; duplicate cell entries are ignored, as ad_kite selects the first.
extern "C" __global__ void pv_cell_v841_f32__ad(
    const float *pv_vertex, const int *vertices_on_cell,
    const int *n_edges_on_cell, const int *cells_on_vertex,
    const float *kite_areas, const float *inv_area_cell,
    const int nlev, const int ncells, const int nvertices,
    const int max_edges, const int vertex_degree, const float *pv_cell,
    const float *seed_pv_cell, float *b_pv_vertex, float *b_pv_cell,
    AdFailureRecord *failure, unsigned int *branch_record)
{
    (void)pv_cell;
    const int begin = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = begin; i < nlev * ncells; i += stride) {
        const int cell = i % ncells, k = i / ncells;
        rrfs_pvc_record(i,pv_vertex,vertices_on_cell,n_edges_on_cell,cells_on_vertex,kite_areas,inv_area_cell,ncells,nvertices,max_edges,vertex_degree,branch_record);
        float value = 0.0f;
        ad_fail(failure, 2, !isfinite(inv_area_cell[cell]));
        ad_fail(failure, 10, !isfinite(seed_pv_cell[i]));
        for (int j = 0; j < n_edges_on_cell[cell]; ++j) {
            const int v = vertices_on_cell[cell * max_edges + j];
            const int kite = ad_kite(v, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                const float area = kite_areas[v * vertex_degree + kite];
                const float p = pv_vertex[k * nvertices + v];
                const float q0 = mpas_mul(area, p);
                const float q = mpas_mul(q0, inv_area_cell[cell]);
                value = mpas_add(value, q);
                ad_fail(failure, 0, !isfinite(area));
                ad_fail(failure, 1, !isfinite(p));
                ad_fail(failure, 3, !isfinite(q0));
                ad_fail(failure, 4, !isfinite(q));
                ad_fail(failure, 5, !isfinite(value));
            }
        }
        b_pv_cell[i] = 0.0f;
    }
    for (int i = begin; i < nlev * nvertices; i += stride) {
        const int vertex = i % nvertices, k = i / nvertices;
        float sum = 0.0f;
        for (int a = 0; a < vertex_degree; ++a) {
            const int cell = cells_on_vertex[vertex * vertex_degree + a];
            if (cell < 0 || cell >= ncells) continue;
            bool duplicate = false;
            for (int z = 0; z < a; ++z)
                duplicate |= cells_on_vertex[vertex * vertex_degree + z] == cell;
            if (duplicate) continue;
            const int kite = ad_kite(vertex, cell, cells_on_vertex, vertex_degree);
            const float area = kite_areas[vertex * vertex_degree + kite];
            for (int j = 0; j < n_edges_on_cell[cell]; ++j) {
                if (vertices_on_cell[cell * max_edges + j] == vertex) {
                    const float b = mpas_mul(seed_pv_cell[k * ncells + cell],
                                             inv_area_cell[cell]);
                    const float contribution = mpas_mul(b, area);
                    sum = mpas_add(sum, contribution);
                    ad_fail(failure, 11, !isfinite(b));
                    ad_fail(failure, 12, !isfinite(contribution));
                    ad_fail(failure, 13, !isfinite(sum));
                }
            }
        }
        b_pv_vertex[i] = mpas_add(b_pv_vertex[i], sum);
        ad_fail(failure, 14, !isfinite(b_pv_vertex[i]));
    }
}

#endif
