// SPDX-License-Identifier: Apache-2.0
// Executed-arm tangent and deterministic gathered transpose of the pinned row.
#define AD_B78 4
#define AD_B83 5
#define AD_B93 6
#define AD_B111 7
#define AD_B116 8
#define AD_B123 9
#define AD_B142 10
#define AD_B145 11
#define AD_B42 12
#define AD_B45 13
#define AD_B57 14
#define AD_B62 15
#include "dynamics_v841_ad.cuh"

#if !defined(AD_ONLY_AD)
extern "C" __global__ void vector_momentum_v841_f32__tl(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *inv_dc_edge, const float *reference_u, const float *f_edge,
    const float *initial_tendency, const int has_initial, const float *result,
    const float *d_u, const float *d_rho_edge, const float *d_pv_edge,
    const float *d_kinetic, const float *d_mass_divergence, const float *d_reference_u,
    const float *d_f_edge, const float *d_initial_tendency, float *d_result,
    AdFailureRecord *failures, unsigned *branch_records)
{
    (void)result;
    MomentumData s{nlev,ncells,nedges,max_edges2,has_initial,u,rho_edge,pv_edge,
      kinetic,mass_divergence,cells_on_edge,edges_on_edge,n_edges_on_edge,
      weights_on_edge,inv_dc_edge,reference_u,f_edge,initial_tendency};
    const int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    const int extent=momentum_extent(s);
    MomentumDirection d{d_u,d_rho_edge,d_pv_edge,d_kinetic,d_mass_divergence,
      d_reference_u,d_f_edge,d_initial_tendency};
    for(int i=tid;i<extent;i+=stride) { AdCheck c{failures,i,1ull}; momentum_validate_tl(s,d,i,c); }
    for(int i=tid;i<nlev*nedges;i+=stride)
        momentum_store_tl(s,d,i/nedges,i%nedges,d_result,failures,extent,branch_records);
}
#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void vector_momentum_v841_f32__ad(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *inv_dc_edge, const float *reference_u, const float *f_edge,
    const float *initial_tendency, const int has_initial, const float *result,
    const float *b_result, float *b_result_consumed, float *b_u, float *b_rho_edge, float *b_pv_edge,
    float *b_kinetic, float *b_mass_divergence, float *b_reference_u,
    float *b_f_edge, float *b_initial_tendency, const int *edge_offsets,
    const int *edge_owners, const int *cell_offsets, const int *cell_owners,
    AdFailureRecord *failures, unsigned *branch_records)
{
    (void)result;
    MomentumData s{nlev,ncells,nedges,max_edges2,has_initial,u,rho_edge,pv_edge,
      kinetic,mass_divergence,cells_on_edge,edges_on_edge,n_edges_on_edge,
      weights_on_edge,inv_dc_edge,reference_u,f_edge,initial_tendency};
    const int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    // The input seed is an immutable, disjoint scheduler snapshot.
    for (int i=tid; i<nlev * nedges; i+=stride) b_result_consumed[i] = 0.0f;
    const int extent=momentum_extent(s);
    MomentumReverse b{b_result,b_u,b_rho_edge,b_pv_edge,b_kinetic,b_mass_divergence,
      b_reference_u,b_f_edge,b_initial_tendency,edge_offsets,edge_owners,cell_offsets,cell_owners};
    if (branch_records) {
        MomentumDirection none{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
        for(int i=tid;i<nlev*nedges;i+=stride) { AdCheck c{failures,extent+i,1ull};
            (void)momentum_eval(s,none,i/nedges,i%nedges,999,-1,c,branch_records); }
    }
    for(int i=tid;i<extent;i+=stride) { AdCheck c{failures,i,1ull}; momentum_reverse_element(s,b,i,c); }
}

#endif
