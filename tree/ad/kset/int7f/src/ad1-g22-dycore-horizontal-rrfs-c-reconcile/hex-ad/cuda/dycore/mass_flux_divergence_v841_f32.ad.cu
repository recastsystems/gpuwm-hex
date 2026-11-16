// SPDX-License-Identifier: Apache-2.0
#include "mass_flux_common.cuh"
extern "C" __global__ void mass_flux_divergence_v841_f32__tl(
    const float *rho_u, const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge, const float *dv_edge, const float *inv_area_cell,
    int nlev, int ncells, int nedges, int max_edges, float *mass_divergence,
    const float *d_rho_u, float *d_mass_divergence, HorizontalFailure *failure) {
    mass_tl(rho_u,edges_on_cell,n_edges_on_cell,cells_on_edge,dv_edge,inv_area_cell,
        nlev,ncells,nedges,max_edges,mass_divergence,d_rho_u,d_mass_divergence,failure);
}
extern "C" __global__ void mass_flux_divergence_v841_f32__ad(
    const float *rho_u, const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge, const float *dv_edge, const float *inv_area_cell,
    int nlev, int ncells, int nedges, int max_edges, float *mass_divergence,
    float *b_mass_divergence, float *b_rho_u, HorizontalFailure *failure) {
    mass_ad(rho_u,edges_on_cell,n_edges_on_cell,cells_on_edge,dv_edge,inv_area_cell,
        nlev,ncells,nedges,max_edges,mass_divergence,b_mass_divergence,b_rho_u,failure);
}
