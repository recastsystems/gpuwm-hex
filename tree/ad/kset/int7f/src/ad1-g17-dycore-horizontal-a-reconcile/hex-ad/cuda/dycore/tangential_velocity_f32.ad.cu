// SPDX-License-Identifier: Apache-2.0
// Tangent and deterministic transpose for cuda_horizontal.py:170.
#include "tangential_velocity_f32.branch.cuh"
#define AD_BRANCH_RECORD() do { \
  __shared__ unsigned int ad_local_counts[24]; \
  for(int ad_i=threadIdx.x;ad_i<24;ad_i+=blockDim.x) ad_local_counts[ad_i]=0; \
  __syncthreads(); \
  ad_branch_replay_tangential_velocity_f32(normal_velocity,edges_on_edge,n_edges_on_edge,weights_on_edge,nlev,nedges,max_edges2,tangential_velocity,ad_local_counts); \
  __syncthreads(); \
  for(int ad_i=threadIdx.x;ad_i<24;ad_i+=blockDim.x) if(ad_local_counts[ad_i]) atomicAdd(branch_record+ad_i,ad_local_counts[ad_i]); \
} while(0)
#if !defined(AD_BUILD_AD)
extern "C" __global__ void tangential_velocity_f32__branch(
    const float* normal_velocity,
    const int* edges_on_edge,
    const int* n_edges_on_edge,
    const float* weights_on_edge,
    const int nlev,
    const int nedges,
    const int max_edges2,
    float* tangential_velocity
, unsigned int* branch_record) { AD_BRANCH_RECORD(); }
#endif

__device__ __forceinline__ bool ad_refuse_slot(int32_t* f,int slot,bool cause){if(cause) atomicExch(f+slot,1);return cause;}

#if !defined(AD_BUILD_AD)
extern "C" __global__ void tangential_velocity_f32__tl(
    const float* normal_velocity, const int* edges_on_edge,
    const int* n_edges_on_edge, const float* weights_on_edge,
    int nlev, int nedges, int max_edges2, float* tangential_velocity,
    const float* d_normal_velocity, float* d_tangential_velocity,
    int32_t* failure
, unsigned int* branch_record) {
 AD_BRANCH_RECORD();
    (void)tangential_velocity;
    for (int t=blockIdx.x*blockDim.x+threadIdx.x;t<nlev*nedges;t+=blockDim.x*gridDim.x) {
        const int e=t%nedges,k=t/nedges,count=n_edges_on_edge[e];
        if(ad_refuse_slot(failure,0,(count<0 ))||ad_refuse_slot(failure,1,( count>max_edges2))) { (void)0;continue; }
        float primal=0.0f,delta=0.0f,correction=0.0f; bool bad=false;
        for(int s=0;s<count;s++) {
            int p=e*max_edges2+s, n=edges_on_edge[p];
            if(ad_refuse_slot(failure,2,(n<0 ))||ad_refuse_slot(failure,3,( n>=nedges))) { bad=true; continue; }
            float w=weights_on_edge[p], x=normal_velocity[k*nedges+n],dx=d_normal_velocity[k*nedges+n];
            primal += w*x;
            // Compensate the fixed-order tangent sum in float32. This keeps
            // cancellation in the transpose identity from magnifying error.
            const float term=w*dx-correction;
            const float next=delta+term;
            correction=(next-delta)-term;
            delta=next;
            bad |= ad_refuse_slot(failure,10,(!isfinite(w)))||ad_refuse_slot(failure,11,(!isfinite(x)))||ad_refuse_slot(failure,12,(!isfinite(dx)))||ad_refuse_slot(failure,13,(!isfinite(primal)))||ad_refuse_slot(failure,14,(!isfinite(delta)));
        }
        if(bad) (void)0; else d_tangential_velocity[t]=delta;
    }
}
#endif

// CSR rows are input edges; entries are original (output edge, slot) offsets,
// built by stable traversal of output edges and their valid slots, duplicates
// retained. This makes the gather order explicit and avoids unordered atomics.
#if !defined(AD_BUILD_TL)
extern "C" __global__ void tangential_velocity_f32__ad(
    const float* normal_velocity, const int* edges_on_edge,
    const int* n_edges_on_edge, const float* weights_on_edge,
    int nlev, int nedges, int max_edges2, float* tangential_velocity,
    const float* b_tangential_velocity, const int* edge_offsets,
    const int* edge_slots, float* b_normal_velocity, int32_t* failure
, unsigned int* branch_record) {
 AD_BRANCH_RECORD();
    (void)tangential_velocity;
    for (int t=blockIdx.x*blockDim.x+threadIdx.x;t<nlev*nedges;t+=blockDim.x*gridDim.x) {
        const int target=t%nedges,k=t/nedges;
        double sum=b_normal_velocity[t];
        bool bad=ad_refuse_slot(failure,15,(!isfinite((float)sum)))||ad_refuse_slot(failure,16,(!isfinite(normal_velocity[t])))||ad_refuse_slot(failure,17,(!isfinite(b_tangential_velocity[t])));
        for(int i=edge_offsets[target];i<edge_offsets[target+1];i++) {
            int p=edge_slots[i], e=p/max_edges2, slot=p%max_edges2;
            if(ad_refuse_slot(failure,4,(p<0))||ad_refuse_slot(failure,5,(e>=nedges))||ad_refuse_slot(failure,6,(slot>=n_edges_on_edge[e]))||ad_refuse_slot(failure,7,(edges_on_edge[p]!=target))) { bad=true;continue; }
            const float w=weights_on_edge[p],seed=b_tangential_velocity[k*nedges+e];
            bad |= ad_refuse_slot(failure,18,(!isfinite(w)))||ad_refuse_slot(failure,19,(!isfinite(seed)));
            float primal=0.0f;
            for(int s=0;s<n_edges_on_edge[e];s++) {
                const int q=e*max_edges2+s, neighbor=edges_on_edge[q];
                if(ad_refuse_slot(failure,8,(neighbor<0))||ad_refuse_slot(failure,9,(neighbor>=nedges))) { bad=true;continue; }
                float v=normal_velocity[k*nedges+neighbor],weight=weights_on_edge[q];
                primal += weight*v;
                bad |= ad_refuse_slot(failure,20,(!isfinite(v)))||ad_refuse_slot(failure,21,(!isfinite(weight)))||ad_refuse_slot(failure,22,(!isfinite(primal)));
            }
            sum+=(double)w*seed; bad |= ad_refuse_slot(failure,23,(!isfinite((float)sum)));
        }
        if(bad) (void)0; else b_normal_velocity[t]=sum;
    }
}
#endif
