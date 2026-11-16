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

__device__ __forceinline__ void ad_branch_replay_tangential_velocity_f32(
    const float* normal_velocity,
    const int* edges_on_edge,
    const int* n_edges_on_edge,
    const float* weights_on_edge,
    const int nlev,
    const int nedges,
    const int max_edges2,
    float* tangential_velocity
, unsigned int* branch_shared) {
  AdBranchRecorder recorder{branch_shared};

    AD_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int count = n_edges_on_edge[edge];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            value = recorder.add(value, recorder.mul(weights_on_edge[offset],
                normal_velocity[ad_br_lidx(k, neighbor, nedges)]));
        }
        (void)(value);
    }
}
