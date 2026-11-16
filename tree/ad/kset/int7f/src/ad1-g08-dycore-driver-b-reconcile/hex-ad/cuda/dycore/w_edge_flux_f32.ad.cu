// SPDX-License-Identifier: Apache-2.0
#include "w_edge_common.cuh"
extern "C" __global__ void w_edge_flux_f32__tl(
    int nlev,int ncells,int nedges,int width,float coefficient,
    const float *w,const float *ru,const float *fzm,const float *fzp,const float *adv,const float *adv3,
    const int *n_adv,const int *adv_cells,
    const float *d_coefficient,const float *dw,const float *dru,const float *dfzm,const float *dfzp,
    const float *dadv,const float *dadv3,float *dflux,unsigned *branch_record,int *failures) {
    const WEdge p={nlev,ncells,nedges,width,coefficient,w,ru,fzm,fzp,adv,adv3,n_adv,adv_cells};
    const WEdgeTangent d={*d_coefficient,dw,dru,dfzm,dfzp,dadv,dadv3};
    AD_LOOP((nlev+1)*nedges) we_tangent_element(p,d,i/nedges,i%nedges,dflux,branch_record,1,failures);
}
extern "C" __global__ void w_edge_flux_f32__ad(
    int nlev,int ncells,int nedges,int width,float coefficient,
    const float *w,const float *ru,const float *fzm,const float *fzp,const float *adv,const float *adv3,
    const int *n_adv,const int *adv_cells,const int *cell_offsets,const int *cell_edges,const int *cell_slots,
    const float *bflux,float *bw,float *bru,float *bfzm,float *bfzp,float *badv,float *badv3,
    float *bcoefficient,int *failures) {
    const WEdge p={nlev,ncells,nedges,width,coefficient,w,ru,fzm,fzp,adv,adv3,n_adv,adv_cells};
    const WEdgeAdjoint b={cell_offsets,cell_edges,cell_slots,bflux,bw,bru,bfzm,bfzp,badv,badv3,bcoefficient};
    const int total=(nlev+1)*ncells+nlev*nedges+nlev+nedges*width;
    AD_LOOP(total) we_adjoint_element(p,b,i,failures);
    we_adjoint_scalar(p,b,failures,total);
}
