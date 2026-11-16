// SPDX-License-Identifier: Apache-2.0
// Branch-only replay derived from cuda_horizontal.py at the recorded pin.
#define AD_BR_ADD_FAST 2
#define AD_BR_ADD_SUBNORMAL 3
#define AD_BR_ADD_CANCEL 4
#define AD_BR_SUB_FAST -1
#define AD_BR_SUB_SUBNORMAL -1
#define AD_BR_SUB_CANCEL -1
#define AD_BR_MUL_ENVELOPE 5
#define AD_BR_MUL_NONZERO 6
#define AD_BR_DIV_SUBNORMAL -1
#define AD_BR_DIV_ENVELOPE -1
#define AD_BR_DECODE_SUBNORMAL 7
#define AD_BR_DECODE_SIGN 8
#define AD_BR_ENCODE_SUBNORMAL 9
#define AD_BR_ENCODE_SIGN 10
#include "ad_branch_ops.cuh"

#define AD_ELEMENT_LOOP(total) for (int element = blockDim.x*blockIdx.x+threadIdx.x; element < (total); element += gridDim.x*blockDim.x)
__device__ __forceinline__ int ad_br_lidx(int k,int i,int n){return k*n+i;}
__device__ __forceinline__ float ad_br_cell_sign(int cell,int edge,const int* x){return x[2*edge]==cell?1.0f:-1.0f;}
__device__ __forceinline__ float ad_br_vertex_sign(int vertex,int edge,const int* x){return x[2*edge+1]==vertex?1.0f:-1.0f;}
__device__ __forceinline__ int ad_br_kite_slot(int vertex,int cell,const int* x,int degree){for(int j=0;j<degree;j++)if(x[vertex*degree+j]==cell)return j;return -1;}

__device__ __forceinline__ void ad_branch_replay_edge_fields_from_saved_u_f32(
    const float* rho,
    const float* normal_velocity,
    const float* dc_edge,
    const float* dv_edge,
    const int* cells_on_edge,
    const int nlev,
    const int ncells,
    const int nedges,
    float* rho_edge,
    float* ke_edge
, unsigned int* branch_shared) {
  AdBranchRecorder recorder{branch_shared};

    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int tid = ad_br_lidx(k, edge, nedges);
        const float edge_rho = recorder.mul(0.5f, recorder.add(
            rho[ad_br_lidx(k, cell0, ncells)], rho[ad_br_lidx(k, cell1, ncells)]
        ));
        const float u = normal_velocity[tid];
        (void)(edge_rho);
        // Native MPAS: efac*u**2. Round the square before the area product.
        (void)(recorder.mul(recorder.mul(
            dc_edge[edge], dv_edge[edge]), recorder.mul(u, u)));
    }
}
