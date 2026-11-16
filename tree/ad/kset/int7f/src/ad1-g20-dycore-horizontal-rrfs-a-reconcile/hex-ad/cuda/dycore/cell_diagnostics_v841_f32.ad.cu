#if defined(AD_BUILD_BRANCH)
#include "rrfs_branch_records.ad.cu"
#else
// SPDX-License-Identifier: Apache-2.0
#include "rrfs_mpas_f32.cuh"
#include "rrfs_ad_support.cuh"

__device__ __forceinline__ int ad_kite_slot(
    int v, int c, const int *cells_on_vertex, int degree)
{
    for (int s = 0; s < degree; ++s)
        if (cells_on_vertex[v * degree + s] == c) return s;
    return -1;
}

#if !defined(AD_BUILD_AD)
extern "C" __global__ void cell_diagnostics_v841_f32__tl(
    const float *normal_velocity, const float *ke_edge, const float *ke_vertex,
    const int *edges_on_cell, const int *n_edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_cell, const int *cells_on_vertex, const float *dv_edge,
    const float *inv_area_cell, const float *kite_areas, int nlev, int ncells,
    int nedges, int nvertices, int max_edges, int vertex_degree,
    float *divergence, float *kinetic_energy,
    const float *d_normal_velocity, const float *d_ke_edge, const float *d_ke_vertex,
    float *d_divergence, float *d_kinetic_energy, AdFailureRecord *failure)
{
    (void)divergence; (void)kinetic_energy;
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int q = t; q < nlev * ncells; q += gridDim.x * blockDim.x) {
        const int c = q % ncells, k = q / ncells, count = n_edges_on_cell[c];
        const float ia = inv_area_cell[c];
        float div = 0.0f, ke = 0.0f, ddiv = 0.0f, dke = 0.0f;
        ad_finite(failure, t, ia, 10, c, 1);
        for (int s = 0; s < count; ++s) {
            const int e = edges_on_cell[c * max_edges + s];
            const int j = ad_index(k, e, nedges);
            const float a = mpas_mul(ad_cell_sign(c, e, cells_on_edge), dv_edge[e]);
            ad_finite(failure, t, dv_edge[e], 9, e, 2);
            ad_finite(failure, t, normal_velocity[j], 1, j, 3);
            ad_finite(failure, t, ke_edge[j], 2, j, 4);
            ad_finite(failure, t, d_normal_velocity[j], 1, j, 5);
            ad_finite(failure, t, d_ke_edge[j], 2, j, 6);
            div = mpas_add(div, mpas_mul(a, normal_velocity[j]));
            ke = mpas_add(ke, mpas_mul(0.25f, ke_edge[j]));
            ddiv = mpas_add(ddiv, mpas_mul(a, d_normal_velocity[j]));
            dke = mpas_add(dke, mpas_mul(0.25f, d_ke_edge[j]));
            ad_finite(failure, t, div, 1, j, 7);
            ad_finite(failure, t, ke, 2, j, 8);
        }
        div = mpas_mul(div, ia);
        ke = mpas_mul(mpas_mul(ke, ia), 0.625f);
        ddiv = mpas_mul(ddiv, ia);
        dke = mpas_mul(mpas_mul(dke, ia), 0.625f);
        for (int s = 0; s < count; ++s) {
            const int v = vertices_on_cell[c * max_edges + s];
            const int kite = ad_kite_slot(v, c, cells_on_vertex, vertex_degree);
            if (kite < 0) continue;
            const int j = ad_index(k, v, nvertices), p = v * vertex_degree + kite;
            const float a = mpas_mul(0.375f, kite_areas[p]);
            ad_finite(failure, t, kite_areas[p], 11, p, 9);
            ad_finite(failure, t, ke_vertex[j], 3, j, 10);
            ad_finite(failure, t, d_ke_vertex[j], 3, j, 11);
            ke = mpas_add(ke, mpas_mul(mpas_mul(a, ke_vertex[j]), ia));
            dke = mpas_add(dke, mpas_mul(mpas_mul(a, d_ke_vertex[j]), ia));
            ad_finite(failure, t, ke, 3, j, 12);
        }
        ad_finite(failure, t, div, 12, q, 13);
        ad_finite(failure, t, ddiv, 12, q, 14);
        ad_finite(failure, t, dke, 13, q, 15);
        d_divergence[q] = ddiv;
        d_kinetic_energy[q] = dke;
    }
}
#endif

#if !defined(AD_BUILD_TL)
extern "C" __global__ void cell_diagnostics_v841_f32__ad(
    const float *normal_velocity, const float *ke_edge, const float *ke_vertex,
    const int *edges_on_cell, const int *n_edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_cell, const int *cells_on_vertex, const float *dv_edge,
    const float *inv_area_cell, const float *kite_areas, int nlev, int ncells,
    int nedges, int nvertices, int max_edges, int vertex_degree,
    float *divergence, float *kinetic_energy,
    float *b_normal_velocity, float *b_ke_edge, float *b_ke_vertex,
    float *b_divergence, float *b_kinetic_energy, AdFailureRecord *failure)
{
    (void)divergence; (void)kinetic_energy;
    if (blockIdx.x != 0) return;
    const unsigned t = threadIdx.x;
    for (int k = t; k < nlev; k += blockDim.x) {
        for (int c = 0; c < ncells; ++c) {
            const int q = ad_index(k, c, ncells), count = n_edges_on_cell[c];
            const float ia = inv_area_cell[c], bd = b_divergence[q], bk = b_kinetic_energy[q];
            float div = 0.0f, ke = 0.0f;
            ad_finite(failure, t, ia, 10, c, 16);
            ad_finite(failure, t, bd, 12, q, 17);
            ad_finite(failure, t, bk, 13, q, 18);
            for (int s = 0; s < count; ++s) {
                const int e = edges_on_cell[c * max_edges + s];
                const int j = ad_index(k, e, nedges);
                const float a = mpas_mul(ad_cell_sign(c, e, cells_on_edge), dv_edge[e]);
                ad_finite(failure, t, dv_edge[e], 9, e, 19);
                ad_finite(failure, t, normal_velocity[j], 1, j, 20);
                ad_finite(failure, t, ke_edge[j], 2, j, 21);
                div = mpas_add(div, mpas_mul(a, normal_velocity[j]));
                ke = mpas_add(ke, mpas_mul(0.25f, ke_edge[j]));
                b_normal_velocity[j] = mpas_add(b_normal_velocity[j], mpas_mul(a, mpas_mul(ia, bd)));
                b_ke_edge[j] = mpas_add(b_ke_edge[j], mpas_mul(0.25f, mpas_mul(ia, mpas_mul(0.625f, bk))));
                ad_finite(failure, t, b_normal_velocity[j], 1, j, 22);
                ad_finite(failure, t, b_ke_edge[j], 2, j, 23);
                ad_finite(failure, t, div, 1, j, 24);
                ad_finite(failure, t, ke, 2, j, 25);
            }
            div = mpas_mul(div, ia);
            ke = mpas_mul(mpas_mul(ke, ia), 0.625f);
            for (int s = 0; s < count; ++s) {
                const int v = vertices_on_cell[c * max_edges + s];
                const int kite = ad_kite_slot(v, c, cells_on_vertex, vertex_degree);
                if (kite < 0) continue;
                const int j = ad_index(k, v, nvertices), p = v * vertex_degree + kite;
                const float a = mpas_mul(0.375f, kite_areas[p]);
                ad_finite(failure, t, kite_areas[p], 11, p, 26);
                ad_finite(failure, t, ke_vertex[j], 3, j, 27);
                ke = mpas_add(ke, mpas_mul(mpas_mul(a, ke_vertex[j]), ia));
                b_ke_vertex[j] = mpas_add(b_ke_vertex[j], mpas_mul(a, mpas_mul(ia, bk)));
                ad_finite(failure, t, b_ke_vertex[j], 3, j, 28);
                ad_finite(failure, t, ke, 3, j, 29);
            }
            ad_finite(failure, t, div, 12, q, 30);
            b_divergence[q] = 0.0f;
            b_kinetic_energy[q] = 0.0f;
        }
    }
}
#endif
#endif
