// SPDX-License-Identifier: Apache-2.0
// The shared helper recomputes the executed primal arithmetic and gathers
// the transpose in a fixed topology order; no unordered atomics are used.
#define AD_B78 2
#define AD_B83 3
#define AD_B93 4
#define AD_B111 5
#define AD_B116 6
#define AD_B123 7
#define AD_B142 8
#define AD_B145 9
#define AD_B42 10
#define AD_B45 11
#define AD_B57 12
#define AD_B62 13
#include "dynamics_v841_ad.cuh"

#if !defined(AD_ONLY_AD)
extern "C" __global__ void theta_finish_v841_f32__tl(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux, const float *result,
    const float *d_edge_flux, const float *d_vertical_flux, float *d_result, AdFailureRecord *failures, unsigned *branch_records)
{
    (void)result;
    ThetaData s{nlev,ncells,nedges,max_edges,0,0,
      n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,edge_flux,vertical_flux,nullptr, nullptr, nullptr, nullptr, nullptr};
    const int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    const int extent=theta_extent(s);
    ThetaDirection d{d_edge_flux,d_vertical_flux,nullptr,nullptr,nullptr};
    for(int i=tid;i<extent;i+=stride) { AdCheck c{failures,i,1ull}; theta_validate_tl(s,d,i,c); }
    const int nout=(nlev+0)*ncells;
    for(int i=tid;i<nout;i+=stride) theta_store_tl(s,d,i/ncells,i%ncells,d_result,failures,extent,branch_records);
}

#endif
#if !defined(AD_ONLY_TL)
extern "C" __global__ void theta_finish_v841_f32__ad(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *inv_area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux, const float *result,
    const float *b_result, float *b_result_consumed, float *b_edge_flux, float *b_vertical_flux, const int *edge_offsets, const int *edge_owners, AdFailureRecord *failures, unsigned *branch_records)
{
    (void)result;
    ThetaData s{nlev,ncells,nedges,max_edges,0,0,
      n_edges_on_cell,edges_on_cell,acoustic_sign,inv_area_cell,rdzw,edge_flux,vertical_flux,nullptr, nullptr, nullptr, nullptr, nullptr};
    const int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    // The input seed is an immutable, disjoint scheduler snapshot.
    for (int i=tid; i<nlev * ncells; i+=stride) b_result_consumed[i] = 0.0f;
    const int extent=theta_extent(s);
    ThetaReverse b{b_result,b_edge_flux,b_vertical_flux,nullptr,nullptr,nullptr,edge_offsets,edge_owners,nullptr,nullptr};
    if (branch_records) {
        ThetaDirection none{nullptr,nullptr,nullptr,nullptr,nullptr};
        const int nout=(nlev+int(s.w))*ncells;
        for(int i=tid;i<nout;i+=stride) { AdCheck c{failures,extent+i,1ull};
            (void)theta_eval(s,none,i/ncells,i%ncells,999,-1,c,branch_records); }
    }
    for(int i=tid;i<extent;i+=stride) { AdCheck c{failures,i,1ull}; theta_reverse_element(s,b,i,c); }
}

#endif
