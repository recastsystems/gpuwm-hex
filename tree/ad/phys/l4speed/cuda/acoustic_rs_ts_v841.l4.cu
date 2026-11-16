// SPDX-License-Identifier: Apache-2.0
// L4 speed (ad/window-L4-speed): acoustic_rs_ts_v841 reverse, a warp per lane instead of a thread per lane.
// The old reverse: 32 lanes, lane L walks its elements (cells L, L+32, ... ascending, levels ascending), runs each
// element's tape and, at the tape's end, adds every leaf's cotangent once into the lane's binary64 partial row (or
// into the cell-local float32 cotangent); then the lane partials are summed in lane order. A leaf's total within
// one element never depends on other elements; only the order of the per-element adds into a partial word does.
// Here lane L is one block of 32 threads: each thread runs the tape of one of 32 consecutive elements of the lane
// and records its leaves (destination, cotangent) instead of adding them; then the 32 elements' leaves are added
// element by element in the lane's order (one element's leaves have distinct destinations, so its adds run in
// parallel). Every partial word receives the same binary64 terms in the same order: bitwise the old reverse.
// The final lane-ordered sum is the old final loop, one thread per word.
// Difference outside the values: the tape's finiteness check of an accumulator it reads (input()) sees the
// partial as of the batch start, not after the previous element in the batch; a nonfinite partial still refuses
// in the final sum (source 3).
#include "acoustic_rs_ts_v841.ad.cu"

#define L4_RS_BATCH 32
#define L4_RS_MAXLEAF 96
struct L4Leaf { unsigned long long dst; double bar; };
struct L4RsReverse : AcousticReverse {
    L4Leaf *leaves; int *nleaves;
    __device__ void finish() {
        int n = 0;
        for (int i = count - 1; i >= 0; --i) {
            AcousticNode &x = nodes[i];
            if (x.destination) {
                if (n < L4_RS_MAXLEAF) leaves[n++] = {x.destination, x.bar}; else acoustic_fail(failure, 0, i, 4);
            } else {
                if (x.left >= 0) nodes[x.left].bar += x.bar * mpas_f32_to_f64_ieee(x.left_weight);
                if (x.right >= 0) nodes[x.right].bar += x.bar * mpas_f32_to_f64_ieee(x.right_weight);
            }
        }
        *nleaves = n;
    }
};
__device__ __forceinline__ void l4_rs_apply(const L4Leaf &l, AcousticFailure *failure, int i) {
    if (l.dst & 1ULL) {
        double *p = (double *)(l.dst & ~1ULL); double z = *p + l.bar;
        if (!isfinite(z)) acoustic_fail(failure, 0, i, 3); *p = z;
    } else {
        float *p = (float *)l.dst; float z = mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(*p) + l.bar);
        if (!isfinite(z)) acoustic_fail(failure, 0, i, 3); *p = z;
    }
}

extern "C" __global__ void acoustic_rs_ts_v841__l4_lanes(RS_PRIMAL_ARGS,float *b_dts,
            RS_SHADOW_ARGS,AcousticNode *scratch,double *partial,L4Leaf *leafbuf,AcousticFailure *failures) {
    (void)b_dts;
    constexpr int lanes = 32;
    if (gridDim.x != lanes || blockDim.x != L4_RS_BATCH) { if (!blockIdx.x && !threadIdx.x) acoustic_fail(failures,0,0,6); return; }
    if (nlev < 0 || ncells < 0 || nedges < 0 || max_edges < 0) { if (!threadIdx.x) acoustic_fail(failures,0,0,6); return; }
    const int lane = blockIdx.x, t = threadIdx.x;
    const int capacity = 128 + 16 * max_edges;
    const size_t sign_n=(size_t)ncells*max_edges,edge_n=nedges,area_n=ncells;
    const size_t theta_n=(size_t)nlev*ncells,interface_n=(size_t)(nlev+1)*ncells,ru_n=(size_t)nlev*nedges;
    const size_t os=1,od=os+sign_n,oa=od+edge_n,ot=oa+area_n,oz=ot+theta_n;
    const size_t oc=oz+nlev,oft=oc+nlev,oe=oft+interface_n,oru=oe+nlev+1,orw=oru+ru_n;
    const size_t stride=orw+interface_n;
    double *q = partial + (size_t)lane * stride;
    for (size_t i = t; i < stride; i += blockDim.x) q[i] = 0.0;
    // The old lane stops at its first cell with an invalid edge count (refusal, source 5).
    __shared__ int ncell_lane;
    if (t == 0) {
        int n = 0;
        for (int cell = lane; cell < ncells; cell += lanes) {
            int count = n_edges_on_cell[cell];
            if (count < 0 || count > max_edges) { acoustic_fail(failures,2,cell,5); break; }
            ++n;
        }
        ncell_lane = n;
    }
    __shared__ int nleaf[L4_RS_BATCH];
    __syncthreads();
    RS_FIELDS; f.sign.accum=q+os;f.dv.accum=q+od;f.area.accum=q+oa;f.theta.accum=q+ot;
    f.rdzw.accum=q+oz;f.cofrz.accum=q+oc;f.coftz.accum=q+oft;f.ewm.accum=q+oe;
    f.ru.accum=q+oru;f.rw.accum=q+orw;
    const long long nel = (long long)ncell_lane * nlev;
    L4Leaf *mine = leafbuf + ((size_t)lane * L4_RS_BATCH + t) * L4_RS_MAXLEAF;
    for (long long base = 0; base < nel; base += L4_RS_BATCH) {
        const long long e = base + t;
        nleaf[t] = 0;
        if (e < nel) {
            const int cell = lane + lanes * (int)(e / nlev), k = (int)(e % nlev);
            const int count = n_edges_on_cell[cell];
            L4RsReverse o; o.nodes = scratch + ((size_t)lane * L4_RS_BATCH + t) * capacity; o.count = 0; o.capacity = capacity;
            o.failure = failures; o.leaves = mine; o.nleaves = &nleaf[t]; o.reset();
            rs_ts_element(o,k,cell,ncells,nedges,max_edges,count,edges_on_cell,cells_on_edge,dts,q,f);
        }
        __syncthreads();
        const int m = (int)min((long long)L4_RS_BATCH, nel - base);
        for (int i = 0; i < m; ++i) {
            const L4Leaf *li = leafbuf + ((size_t)lane * L4_RS_BATCH + i) * L4_RS_MAXLEAF;
            for (int l = t; l < nleaf[i]; l += blockDim.x) l4_rs_apply(li[l], failures, l);
            __syncthreads();
        }
    }
}
extern "C" __global__ void acoustic_rs_ts_v841__l4_reduce(int nlev,int ncells,int nedges,int max_edges,float *b_dts,
            float *d_sign,float *d_dv,float *d_area,float *d_theta,float *d_rdzw,float *d_cofrz,
            float *d_coftz,float *d_ewm,float *d_ru,float *d_rw,const double *partial,AcousticFailure *failures) {
    constexpr int lanes = 32;
    const size_t sign_n=(size_t)ncells*max_edges,edge_n=nedges,area_n=ncells;
    const size_t theta_n=(size_t)nlev*ncells,interface_n=(size_t)(nlev+1)*ncells,ru_n=(size_t)nlev*nedges;
    const size_t os=1,od=os+sign_n,oa=od+edge_n,ot=oa+area_n,oz=ot+theta_n;
    const size_t oc=oz+nlev,oft=oc+nlev,oe=oft+interface_n,oru=oe+nlev+1,orw=oru+ru_n;
    const size_t stride=orw+interface_n;
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (j >= stride) return;
    float *dst;
    if(j==0) dst=b_dts;
    else if(j<od) dst=d_sign+(j-os);
    else if(j<oa) dst=d_dv+(j-od);
    else if(j<ot) dst=d_area+(j-oa);
    else if(j<oz) dst=d_theta+(j-ot);
    else if(j<oc) dst=d_rdzw+(j-oz);
    else if(j<oft) dst=d_cofrz+(j-oc);
    else if(j<oe) dst=d_coftz+(j-oft);
    else if(j<oru) dst=d_ewm+(j-oe);
    else if(j<orw) dst=d_ru+(j-oru);
    else dst=d_rw+(j-orw);
    double sum=mpas_f32_to_f64_ieee(*dst);
    for(int lane=0;lane<lanes;++lane) sum+=partial[(size_t)lane*stride+j];
    if(!isfinite(sum)) acoustic_fail(failures,1,j,3);
    *dst=mpas_f64_to_f32_ieee(sum);
}
