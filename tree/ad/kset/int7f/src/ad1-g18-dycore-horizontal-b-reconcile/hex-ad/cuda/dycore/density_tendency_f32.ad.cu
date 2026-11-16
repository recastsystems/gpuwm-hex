// SPDX-License-Identifier: Apache-2.0
// Derivative of cuda_horizontal.py:412; the ordinary forward is unchanged.
#include "common.cuh"
#include "branch_record.cuh"
#define lidx ad_br_lidx
#define cell_sign ad_br_cell_sign
#define vertex_sign ad_br_vertex_sign
#define HORIZONTAL_ELEMENT_LOOP(total) for(int element=blockDim.x*blockIdx.x+threadIdx.x;element<(total);element+=gridDim.x*blockDim.x)
#define mpas_add(a,b) ad_br_add(rec,a,b)
#define mpas_sub(a,b) ad_br_sub(rec,a,b)
#define mpas_mul(a,b) ad_br_mul(rec,a,b)
#define mpas_div(a,b) ad_br_div(rec,a,b)
#define mpas_sqrt(a) ad_br_sqrt(rec,a)
#define mpas_min(a,b) ad_br_min(rec,a,b)
__device__ __forceinline__ void ad_br_record_density_tendency_f32(
    const float* mass_divergence,
    const float* rho_w,
    const float* rdzw,
    const float* physics_tendency,
    const int has_physics,
    const int nlev,
    const int ncells,
    float* tendency
, unsigned int* branch_record) {
 AdBranchRecorder rec{branch_record,{3,4,5,6,7,8,9,10,11,12,13,14,-1,-1,-1,-1,-1,-1,-1,-1,-1}};
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, cell, ncells);
        float value = mpas_sub(mpas_sub(0.0f, mass_divergence[tid]),
            mpas_mul(rdzw[k], mpas_sub(
                rho_w[lidx(k + 1, cell, ncells)],
                rho_w[lidx(k, cell, ncells)])));
        if (has_physics) value = mpas_add(value, physics_tendency[tid]);
        (void)(value);
    }
}
#undef lidx
#undef cell_sign
#undef vertex_sign
#undef HORIZONTAL_ELEMENT_LOOP
#undef mpas_add
#undef mpas_sub
#undef mpas_mul
#undef mpas_div
#undef mpas_sqrt
#undef mpas_min

#ifndef AD_ONLY_AD
extern "C" __global__ void density_tendency_f32__tl(
    const float* mass_divergence, const float* rho_w, const float* rdzw,
    const float* physics_tendency, int has_physics, int nlev, int ncells, float* tendency,
    const float* d_mass_divergence, const float* d_rho_w, const float* d_rdzw,
    const float* d_physics_tendency, float* d_tendency, int* failures, unsigned int* branch_record){
 if(branch_record) ad_br_record_density_tendency_f32(mass_divergence,rho_w,rdzw,physics_tendency,has_physics,nlev,ncells,tendency,branch_record);
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=ncells) return;
    for (int k=0;k<nlev;k++) {
        const int i=k*ncells+cell, ip=(k+1)*ncells+cell;
        if (!ad_finite(mass_divergence[i]) || !ad_finite(d_mass_divergence[i]))
            ad_bad(failures,cell,1,i,0);
        if (!ad_finite(rho_w[i]) || !ad_finite(rho_w[ip]) || !ad_finite(d_rho_w[i]) || !ad_finite(d_rho_w[ip]))
            ad_bad(failures,cell,2,i,1);
        if (!ad_finite(rdzw[k]) || !ad_finite(d_rdzw[k])) ad_bad(failures,cell,3,k,2);
        if (has_physics && (!ad_finite(physics_tendency[i]) || !ad_finite(d_physics_tendency[i])))
            ad_bad(failures,cell,4,i,3);
        const float dr=mpas_sub(rho_w[ip],rho_w[i]);
        const float ddr=mpas_sub(d_rho_w[ip],d_rho_w[i]);
        float dt=mpas_sub(mpas_sub(0.0f,d_mass_divergence[i]),
                           mpas_add(mpas_mul(d_rdzw[k],dr),mpas_mul(rdzw[k],ddr)));
        if (has_physics) dt=mpas_add(dt,d_physics_tendency[i]);
        if (!ad_finite(dr) || !ad_finite(dt)) ad_bad(failures,cell,8,i,4);
        d_tendency[i]=dt;
    }
}
#endif
#ifndef AD_ONLY_TL
extern "C" __global__ void density_tendency_f32__ad(
    const float* mass_divergence, const float* rho_w, const float* rdzw,
    const float* physics_tendency, int has_physics, int nlev, int ncells, float* tendency,
    float* b_mass_divergence, float* b_rho_w, float* b_rdzw,
    float* b_physics_tendency, float* b_tendency, int* failures) {
    if (blockIdx.x != 0) return;
 const int thread=threadIdx.x;
    const int stride=blockDim.x;
    for (int cell=thread;cell<ncells;cell+=stride) {
        for (int k=0;k<nlev;k++) {
            const int i=k*ncells+cell, ip=(k+1)*ncells+cell;
            const float b=b_tendency[i];
            if (!ad_finite(mass_divergence[i]) || !ad_finite(b_mass_divergence[i])) ad_bad(failures,cell,1,i,0);
            if (!ad_finite(rho_w[i]) || !ad_finite(rho_w[ip]) || !ad_finite(b_rho_w[i]) || !ad_finite(b_rho_w[ip])) ad_bad(failures,cell,2,i,1);
            if (!ad_finite(rdzw[k])) ad_bad(failures,cell,3,k,2);
            if (has_physics && (!ad_finite(physics_tendency[i]) || !ad_finite(b_physics_tendency[i]))) ad_bad(failures,cell,4,i,3);
            if (!ad_finite(b)) ad_bad(failures,cell,8,i,4);
            b_mass_divergence[i]=mpas_sub(b_mass_divergence[i],b);
            if (has_physics) b_physics_tendency[i]=mpas_add(b_physics_tendency[i],b);
        }
        for (int k=0;k<=nlev;k++) {
            const int i=k*ncells+cell;
            double b=mpas_f32_to_f64_ieee(b_rho_w[i]);
            if (k<nlev) b+=mpas_f32_to_f64_ieee(mpas_mul(rdzw[k],b_tendency[i]));
            if (k>0) b-=mpas_f32_to_f64_ieee(mpas_mul(rdzw[k-1],b_tendency[i-ncells]));
            if (!ad_finite(b)) ad_bad(failures,cell,2,i,5);
            b_rho_w[i]=mpas_f64_to_f32_ieee(b);
        }
    }
    // Independent deterministic gather for the shared per-level coefficient.
    for (int k=thread;k<nlev;k+=stride) {
        double b=mpas_f32_to_f64_ieee(b_rdzw[k]);
        if (!ad_finite(b)) ad_bad(failures,(unsigned long long)ncells+k,3,k,6);
        for (int cell=0;cell<ncells;cell++) {
            const int i=k*ncells+cell;
            b-=mpas_f32_to_f64_ieee(mpas_mul(b_tendency[i],mpas_sub(rho_w[i+ncells],rho_w[i])));
        }
        if (!ad_finite(b)) ad_bad(failures,(unsigned long long)ncells+k,3,k,7);
        b_rdzw[k]=mpas_f64_to_f32_ieee(b);
    }
    // A single block completes every deterministic gather before consuming seeds.
    __syncthreads();
    for (int i=threadIdx.x;i<nlev*ncells;i+=blockDim.x) { b_tendency[i]=0.0f; }
}

#endif
