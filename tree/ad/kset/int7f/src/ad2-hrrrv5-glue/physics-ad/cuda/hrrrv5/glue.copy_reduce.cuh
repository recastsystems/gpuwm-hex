// SPDX-License-Identifier: Apache-2.0
// General deterministic transpose storage for the RRFS copy plans.
// The caller traverses operations in reverse and elements in ascending order.
// Every distinct source address gets one binary64 running sum; it is written
// back exactly once after all destination seeds have been accumulated.
struct glue_copy_bucket { float *address; double sum; };
__device__ __forceinline__ glue_copy_bucket *glue_copy_slot(
    glue_copy_bucket **segments, const unsigned long long *capacity,
    int nsegments, unsigned long long slot) {
    for (int j=0;j<nsegments;++j) {
        if (slot<capacity[j]) return segments[j]+slot;
        slot-=capacity[j];
    }
    return nullptr;
}
__device__ __forceinline__ void glue_copy_add(
    glue_copy_bucket **segments,const unsigned long long *capacity,int nsegments,
    unsigned long long total,float *address,float seed,int *failure) {
    unsigned long long x=((unsigned long long)address)>>2;
    x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33;
    unsigned long long k=x%total;
    for(unsigned long long tried=0;tried<total;++tried) {
        glue_copy_bucket *b=glue_copy_slot(segments,capacity,nsegments,k);
        if(b==nullptr){ atomicExch(failure,1); return; }
        if(b->address==nullptr) {
            b->address=address; b->sum=(double)*address+(double)seed;
            if(!isfinite(*address) || !isfinite(seed) || !isfinite(b->sum)) atomicExch(failure,1);
            return;
        }
        if(b->address==address) {
            b->sum+=(double)seed;
            if(!isfinite(seed) || !isfinite(b->sum)) atomicExch(failure,1);
            return;
        }
        if(++k==total)k=0;
    }
    atomicExch(failure,1);
}
__device__ __forceinline__ void glue_copy_flush(
    glue_copy_bucket **segments,const unsigned long long *capacity,int nsegments,
    int *failure) {
    for(int j=0;j<nsegments;++j) for(unsigned long long k=0;k<capacity[j];++k) {
        glue_copy_bucket *b=segments[j]+k;
        if(b->address!=nullptr) {
            float result=__double2float_rn(b->sum);
            *(b->address)=result;
            if(!isfinite(result)) atomicExch(failure,1);
        }
    }
}
