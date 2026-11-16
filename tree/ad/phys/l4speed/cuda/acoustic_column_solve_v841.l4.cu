// SPDX-License-Identifier: Apache-2.0
// L4 speed (ad/window-L4-speed): acoustic_column_solve_v841 reverse on the whole card.
// The old reverse ran one block of 128 threads (each column's tape on its own thread, cells strided by 128),
// then reduced the per-cell contributions over cells in index order. Every per-cell contribution row and the
// cell-ordered reduction are independent of which thread ran the column, so here:
//   __l4_cells   one thread per column (grid over cells), the old per-column body unchanged; the tape capacity
//                is a launch scalar (a column that overflows it refuses, source 4, exactly like the old tape)
//   __l4_reduce  the old final loop (sum over cells in index order per contribution slot), one thread per slot
// Same nodes, same order, same binary64 sums: bitwise the old reverse. No atomics on floats.
#include "acoustic_column_solve_v841.ad.cu"

extern "C" __global__ void acoustic_column_solve_v841__l4_cells(CS_PRIMAL_ARGS,float *b_dts,
              CS_SHADOW_ARGS,float *seed_rwp,float *seed_rhop,
              float *seed_rtp,float *seed_ww,
              AcousticValue *work,AcousticNode *scratch,
              double *contrib,int capacity,AcousticFailure *failures) {
    (void)b_dts;
    if(nlev<0||capacity<=0){ if(!blockIdx.x&&!threadIdx.x) acoustic_fail(failures,0,0,6); return; }
    const int s=nlev+1,stride=8*s+1;
    const int tid=blockIdx.x*blockDim.x+threadIdx.x,threads=gridDim.x*blockDim.x;
    for(int cell=tid;cell<ncells;cell+=threads) {
        double *partial=contrib+(unsigned long long)cell*stride;
        for(int q=0;q<stride;++q)partial[q]=0.0f;
        CS_FIELDS;
        f.rwp.seed=seed_rwp==d_rwp?nullptr:seed_rwp;
        f.rhop.seed=seed_rhop==d_rhop?nullptr:seed_rhop;
        f.rtp.seed=seed_rtp==d_rtp?nullptr:seed_rtp;
        f.ww.seed=seed_ww==d_ww?nullptr:seed_ww;
        f.fzm.accum=partial;f.fzp.accum=partial+s;f.rdzw.accum=partial+2*s;f.cofrz.accum=partial+3*s;
        f.etp.accum=partial+4*s;f.etm.accum=partial+5*s;f.ewp.accum=partial+6*s;f.ewm.accum=partial+7*s;
        AcousticReverse o{scratch+(unsigned long long)tid*capacity,0,capacity,failures};o.reset();
        column_solve(o,nlev,ncells,cell,dts,partial+8*s,f,work+(unsigned long long)tid*2*s);
        for(int k=0;k<nlev;++k){
            int i=CS_C(k,cell,ncells);
            seed_rwp[i]=0.0f;seed_rhop[i]=0.0f;seed_rtp[i]=0.0f;
            if(k>0)seed_ww[i]=0.0f;
        }
    }
}
extern "C" __global__ void acoustic_column_solve_v841__l4_reduce(int nlev,int ncells,float *b_dts,
              float *d_fzm,float *d_fzp,float *d_rdzw,float *d_cofrz,float *d_etp,float *d_etm,float *d_ewp,float *d_ewm,
              const double *contrib,AcousticFailure *failures) {
    const int s=nlev+1,stride=8*s+1;
    const int q=blockIdx.x*blockDim.x+threadIdx.x; if(q>=stride) return;
    float *dst[8]={d_fzm,d_fzp,d_rdzw,d_cofrz,d_etp,d_etm,d_ewp,d_ewm};
    float *target;int plane=q/s,k=q%s;
    if(q<8*s) {if(k==nlev && plane<6)return;target=dst[plane]+k;}else target=b_dts;
    double sum=mpas_f32_to_f64_ieee(*target);
    for(int cell=0;cell<ncells;++cell)sum+=contrib[(unsigned long long)cell*stride+q];
    if(!isfinite(sum))acoustic_fail(failures,0,q,3);
    *target=mpas_f64_to_f32_ieee(sum);
}
#ifdef L4_MEASURE
// Instrumented copy (L4_MEASURE builds only): the largest tape count over all columns.
extern "C" __global__ void acoustic_column_solve_v841__l4_measure(CS_PRIMAL_ARGS,float *b_dts,
              CS_SHADOW_ARGS,float *seed_rwp,float *seed_rhop,float *seed_rtp,float *seed_ww,
              AcousticValue *work,AcousticNode *scratch,double *contrib,int capacity,AcousticFailure *failures,int *maxcount) {
    (void)b_dts;
    const int s=nlev+1,stride=8*s+1;
    const int tid=blockIdx.x*blockDim.x+threadIdx.x,threads=gridDim.x*blockDim.x;
    for(int cell=tid;cell<ncells;cell+=threads) {
        double *partial=contrib+(unsigned long long)cell*stride;
        for(int q=0;q<stride;++q)partial[q]=0.0f;
        CS_FIELDS;
        f.rwp.seed=seed_rwp; f.rhop.seed=seed_rhop; f.rtp.seed=seed_rtp; f.ww.seed=seed_ww;
        f.fzm.accum=partial;f.fzp.accum=partial+s;f.rdzw.accum=partial+2*s;f.cofrz.accum=partial+3*s;
        f.etp.accum=partial+4*s;f.etm.accum=partial+5*s;f.ewp.accum=partial+6*s;f.ewm.accum=partial+7*s;
        AcousticReverse o{scratch+(unsigned long long)tid*capacity,0,capacity,failures};o.reset();
        column_solve(o,nlev,ncells,cell,dts,partial+8*s,f,work+(unsigned long long)tid*2*s);
        atomicMax(maxcount,o.count);
    }
}
#endif
