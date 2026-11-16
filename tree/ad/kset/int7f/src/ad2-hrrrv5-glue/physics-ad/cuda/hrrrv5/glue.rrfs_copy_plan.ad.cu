// SPDX-License-Identifier: Apache-2.0
// Tangent and deterministic transpose for glue_plan.py's fused word moves.
// A source-typed mirror table preserves each captured shape and stride and
// substitutes a tangent/adjoint address for each active float32 view.  A null
// address denotes an inactive view.  Integer words are never differentiated.
#include "glue.copy_reduce.cuh"

struct RrfsGlueCopyOpAd {
    unsigned long long dst, src;
    long long dst_stride[4], src_stride[4], shape[4];
    long long count;
    unsigned int fill;
    int mode;
};
static_assert(sizeof(RrfsGlueCopyOpAd) == 128, "copy-plan ABI mismatch");

__device__ __forceinline__ void rrfs_ad_offsets(const RrfsGlueCopyOpAd &op,
                                                long long e, long long &d,
                                                long long &s) {
    long long r=e;
    const long long i3=r%op.shape[3]; r/=op.shape[3];
    const long long i2=r%op.shape[2]; r/=op.shape[2];
    const long long i1=r%op.shape[1]; r/=op.shape[1];
    d=r*op.dst_stride[0]+i1*op.dst_stride[1]+i2*op.dst_stride[2]+i3*op.dst_stride[3];
    s=r*op.src_stride[0]+i1*op.src_stride[1]+i2*op.src_stride[2]+i3*op.src_stride[3];
}

// Same geometry as the captured plan.  fill_direction[i] is the direction of
// operation i's continuous fill value, or null for a fixed/integer fill.
// Repeated pointers represent the same scalar in multiple operations.
extern "C" __global__ void rrfs_glue_copy_plan__tl(
    const RrfsGlueCopyOpAd *d_ops, const float *const *fill_direction) {
    const RrfsGlueCopyOpAd &op=d_ops[blockIdx.y];
    float *dst=(float*)op.dst;
    const float *src=(const float*)op.src;
    if (!dst) return;
    for (long long e=(long long)blockIdx.x*blockDim.x+threadIdx.x; e<op.count;
         e+=(long long)gridDim.x*blockDim.x) {
        long long d,s; rrfs_ad_offsets(op,e,d,s);
        if (op.mode) {
            const float *df=fill_direction ? fill_direction[blockIdx.y] : nullptr;
            dst[d]=df ? *df : 0.0f;
        } else {
            dst[d]=src ? src[s] : 0.0f;
        }
    }
}

// The binder validates the captured op table, target bounds, unique written
// addresses, and absence of a read/write overlap.  It supplies a zeroed
// workspace of 2 * sum(count of all captured operations) buckets.
// All contributors are accumulated in descending op, ascending element order
// as float64, including the pre-existing source cotangent, then rounded once.
// This covers zero-stride broadcasts, source aliases and repeated active fill
// scalars across operations.  fill_adjoint entries for the same original
// scalar must point to the same (separate from overwritten outputs) slot.
extern "C" __global__ void rrfs_glue_copy_plan__ad(
    const RrfsGlueCopyOpAd *b_ops, float *const *fill_adjoint,
    glue_copy_bucket *work, int *failure) {
    if (blockIdx.x || blockIdx.y || blockIdx.z || threadIdx.x || threadIdx.y || threadIdx.z) return;
    const int op_count=(int)gridDim.y;
    if (!b_ops) { atomicExch(failure,1); return; }
    unsigned long long capacity=0;
    for (int i=0;i<op_count;++i) {
        const RrfsGlueCopyOpAd &op=b_ops[i];
        if (op.count<0 || (op.count && (op.shape[1]<=0 || op.shape[2]<=0 || op.shape[3]<=0))) {
            atomicExch(failure,1); return;
        }
        unsigned long long n=(unsigned long long)op.count;
        if (n>(~0ULL-capacity)/2) { atomicExch(failure,1); return; }
        capacity+=2*n;
    }
    if (capacity && !work) { atomicExch(failure,1); return; }
    glue_copy_bucket *segments[1]={work};
    unsigned long long capacities[1]={capacity};
    for (int i=op_count-1;i>=0;--i) {
        const RrfsGlueCopyOpAd &op=b_ops[i];
        float *dst=(float*)op.dst, *src=(float*)op.src;
        if (!dst) continue;
        for (long long e=0;e<op.count;++e) {
            long long d,s; rrfs_ad_offsets(op,e,d,s);
            float seed=dst[d];
            float *scalar = (op.mode && fill_adjoint) ? fill_adjoint[i] : nullptr;
            float *source = op.mode ? scalar : (src ? src+s : nullptr);
            if (source) glue_copy_add(segments,capacities,1,capacity,source,seed,failure);
            else if (!isfinite(seed)) atomicExch(failure,1);
            dst[d]=0.0f;
        }
    }
    if (capacity) glue_copy_flush(segments,capacities,1,failure);
}
