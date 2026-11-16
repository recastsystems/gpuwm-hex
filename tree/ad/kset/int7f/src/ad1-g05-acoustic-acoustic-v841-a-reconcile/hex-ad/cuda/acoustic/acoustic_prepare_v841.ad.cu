// SPDX-License-Identifier: Apache-2.0
// Hand interim derivative of the fixed-control overwrites in the forward.
// Section 35 device failure slot: non-finite input or derivative.
#include "acoustic_mpas_fp32.cuh"
typedef unsigned AdFailureRecord;
__device__ __forceinline__ void prepare_fail(AdFailureRecord *records,
                                            int cell, unsigned long long field,
                                            unsigned long long index) {
    (void)cell; (void)field; (void)index; atomicExch(records, 1u);
}

extern "C" __global__ void acoustic_prepare_v841__tl(
        const int nlev, const int ncells, const int small_step,
        float *rw_p, float *rtheta_pp, float *rtheta_pp_old,
        float *rho_pp, float *ww_avg,
        float *d_rw_p, float *d_rtheta_pp, float *d_rtheta_pp_old,
        float *d_rho_pp, float *d_ww_avg, AdFailureRecord *failures) {
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        if (small_step == 1) {
            d_rtheta_pp_old[i] = 0.0f;
            d_rtheta_pp[i] = 0.0f;
            d_rho_pp[i] = 0.0f;
        } else {
            if (!isfinite(rtheta_pp[i]) || !isfinite(d_rtheta_pp[i]))
                prepare_fail(failures, cell, 2, i);
            if (!isfinite(rho_pp[i]) || !isfinite(d_rho_pp[i]))
                prepare_fail(failures, cell, 4, i);
            d_rtheta_pp_old[i] = d_rtheta_pp[i];
        }
    }
    for (int k = 0; k <= nlev; ++k) {
        const int i = k * ncells + cell;
        if (small_step == 1) {
            d_rw_p[i] = 0.0f;
            d_ww_avg[i] = 0.0f;
        } else {
            if (!isfinite(rw_p[i]) || !isfinite(d_rw_p[i]))
                prepare_fail(failures, cell, 1, i);
            if (!isfinite(ww_avg[i]) || !isfinite(d_ww_avg[i]))
                prepare_fail(failures, cell, 5, i);
        }
    }
    (void)rtheta_pp_old;
}

extern "C" __global__ void acoustic_prepare_v841__ad(
        const int nlev, const int ncells, const int small_step,
        float *rw_p, float *rtheta_pp, float *rtheta_pp_old,
        float *rho_pp, float *ww_avg,
        float *b_rw_p, float *b_rtheta_pp, float *b_rtheta_pp_old,
        float *b_rho_pp, float *b_ww_avg,
        float *seed_rw_p, float *seed_rtheta_pp,
        float *seed_rtheta_pp_old, float *seed_rho_pp,
        float *seed_ww_avg, AdFailureRecord *failures) {
    const int cell=blockDim.x*blockIdx.x+threadIdx.x;
    if(cell>=ncells)return;
    for(int k=0;k<nlev;++k){
        int i=k*ncells+cell;
        if(!isfinite(seed_rtheta_pp[i])||!isfinite(seed_rtheta_pp_old[i])||!isfinite(seed_rho_pp[i])||
           !isfinite(b_rtheta_pp[i])||!isfinite(b_rtheta_pp_old[i])||!isfinite(b_rho_pp[i]))
            prepare_fail(failures,cell,2,i);
        // The reconciled ABI binds output seeds separately from input
        // cotangents. Capture each seed before consuming its written slot.
        const float st=seed_rtheta_pp[i],so=seed_rtheta_pp_old[i],sr=seed_rho_pp[i];
        if(small_step!=1){
            if(!isfinite(rtheta_pp[i])||!isfinite(rho_pp[i]))prepare_fail(failures,cell,2,i);
            const double initial=(b_rtheta_pp==seed_rtheta_pp)?0.0:mpas_f32_to_f64_ieee(b_rtheta_pp[i]);
            b_rtheta_pp[i]=mpas_f64_to_f32_ieee(initial+mpas_f32_to_f64_ieee(st)+mpas_f32_to_f64_ieee(so));
            if(b_rho_pp!=seed_rho_pp)b_rho_pp[i]=mpas_add(b_rho_pp[i],sr);
            if(!isfinite(b_rtheta_pp[i])||!isfinite(b_rho_pp[i]))prepare_fail(failures,cell,2,i);
        }else{
            if(b_rtheta_pp==seed_rtheta_pp)b_rtheta_pp[i]=0.0f;
            if(b_rho_pp==seed_rho_pp)b_rho_pp[i]=0.0f;
        }
        // The old field is written on every substep; the others are
        // overwritten only on the first substep. Separate incoming seed
        // buffers are consumed, while pass-through slots remain untouched.
        seed_rtheta_pp_old[i]=0.0f;
        if(small_step==1){seed_rtheta_pp[i]=0.0f;seed_rho_pp[i]=0.0f;}
    }
    for(int k=0;k<=nlev;++k){
        int i=k*ncells+cell;
        if(!isfinite(seed_rw_p[i])||!isfinite(seed_ww_avg[i])||!isfinite(b_rw_p[i])||!isfinite(b_ww_avg[i]))
            prepare_fail(failures,cell,1,i);
        if(small_step!=1){
            if(!isfinite(rw_p[i])||!isfinite(ww_avg[i]))prepare_fail(failures,cell,1,i);
            if(b_rw_p!=seed_rw_p)b_rw_p[i]=mpas_add(b_rw_p[i],seed_rw_p[i]);
            if(b_ww_avg!=seed_ww_avg)b_ww_avg[i]=mpas_add(b_ww_avg[i],seed_ww_avg[i]);
            if(!isfinite(b_rw_p[i])||!isfinite(b_ww_avg[i]))prepare_fail(failures,cell,1,i);
        }else{
            if(b_rw_p==seed_rw_p)b_rw_p[i]=0.0f;
            if(b_ww_avg==seed_ww_avg)b_ww_avg[i]=0.0f;
            seed_rw_p[i]=0.0f;seed_ww_avg[i]=0.0f;
        }
    }
    (void)rtheta_pp_old;
}
