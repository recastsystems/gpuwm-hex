// SPDX-License-Identifier: MIT
#define AD_FAILURE_PAIRS(X) X(1807,8,0) X(1808,9,1) X(1808,11,2) X(1812,7,3) X(1814,10,4)
#include "glue.member_g28.cuh"

__device__ __forceinline__ void edge_record(int i, int nlev, int nedges, int nsolve,
        int pbl, int cu, int first, const float *mass, const float *pb,
        const float *conv, const float *tend, unsigned *r) {
    bool outside = i >= nlev * nedges;
    ad_branch(r,0,outside);
    if (outside) return;
    ad_branch(r,1,first != 0);
    if (first) {
        ad_branch(r,2,pbl != 0);
        float t = pbl ? pb[i] : tend[i];
        ad_branch(r,3,cu != 0);
        if (cu) (void)ad_record_addsub(t,conv[i],false,r,7,12,13);
    }
    bool halo = i % nedges >= nsolve;
    ad_branch(r,4,halo);
    if (halo) return;
    float u=0.0f;
    ad_branch(r,5,pbl != 0);
    if (pbl) u=ad_record_addsub(u,ad_record_mul(pb[i],mass[i],r,10,12,13),false,r,7,12,13);
    ad_branch(r,6,cu != 0);
    if (cu) (void)ad_record_addsub(u,ad_record_mul(conv[i],mass[i],r,10,12,13),false,r,7,12,13);
}

#if !defined(AD_MODE_AD)
extern "C" __global__ void mpas_rk_edges_v842_f32__tl(
    int nlev, int nedges, int nsolve, int pbl, int cu_momentum, int first,
    const float *mass_edge, const float *pbl_edge, const float *cu_edge,
    const float *out_u, const float *tend_u_phys,
    const float *d_mass_edge, const double *d_pbl_edge, const float *d_cu_edge,
    float *d_out_u, float *d_tend_u_phys, int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (branch_record) edge_record(i,nlev,nedges,nsolve,pbl,cu_momentum,first,
                                    mass_edge,pbl_edge,cu_edge,tend_u_phys,branch_record);
    if (i >= nlev * nedges) return;
    bool uses_physics = first || i % nedges < nsolve;
    if (uses_physics && pbl && !isfinite(pbl_edge[i])) { ad_refuse(failure,failure_detail,i,8,1807); return; }
    if (uses_physics && cu_momentum && !isfinite(cu_edge[i])) { ad_refuse(failure,failure_detail,i,9,1808); return; }
    if (first && !pbl && cu_momentum && !isfinite(tend_u_phys[i])) { ad_refuse(failure,failure_detail,i,11,1808); return; }
    // Aliased forward inputs share the pbl field's shadow storage.
    double dm = mass_edge == pbl_edge ? ((const double*)d_mass_edge)[i] : d_mass_edge[i];
    double dc = cu_edge == pbl_edge ? ((const double*)d_cu_edge)[i] : d_cu_edge[i];
    if (first) {
        double value = pbl ? d_pbl_edge[i] : (double)d_tend_u_phys[i];
        if (cu_momentum) value += dc;
        d_tend_u_phys[i] = (float)value;
    }
    if (i % nedges >= nsolve) return;
    if ((pbl || cu_momentum) && !isfinite(mass_edge[i])) {
        ad_refuse(failure,failure_detail,i,7,1812); return;
    }
    float u = 0.0f;
    double du = 0.0;
    if (pbl) { u += pbl_edge[i] * mass_edge[i];
        du += (double)d_pbl_edge[i] * mass_edge[i] + (double)pbl_edge[i] * dm; }
    if (cu_momentum) { u += cu_edge[i] * mass_edge[i];
        du += dc * mass_edge[i] + (double)cu_edge[i] * dm; }
    if (!isfinite(u)) { ad_refuse(failure,failure_detail,i,10,1814); return; }
    d_out_u[i] = (float)du;
    (void)out_u;
}
#endif

#if !defined(AD_MODE_TL)
extern "C" __global__ void mpas_rk_edges_v842_f32__ad(
    int nlev, int nedges, int nsolve, int pbl, int cu_momentum, int first,
    const float *mass_edge, const float *pbl_edge, const float *cu_edge,
    const float *out_u, const float *tend_u_phys,
    float *b_mass_edge, double *b_pbl_edge, float *b_cu_edge,
    float *seed_out_u, float *seed_tend_u_phys,
    float *b_out_u, float *b_tend_u_phys, int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (branch_record) edge_record(i,nlev,nedges,nsolve,pbl,cu_momentum,first,
                                    mass_edge,pbl_edge,cu_edge,tend_u_phys,branch_record);
    if (i >= nlev * nedges) return;
    bool uses_physics = first || i % nedges < nsolve;
    if (uses_physics && pbl && !isfinite(pbl_edge[i])) { ad_refuse(failure,failure_detail,i,8,1807); return; }
    if (uses_physics && cu_momentum && !isfinite(cu_edge[i])) { ad_refuse(failure,failure_detail,i,9,1808); return; }
    if (first && !pbl && cu_momentum && !isfinite(tend_u_phys[i])) { ad_refuse(failure,failure_detail,i,11,1808); return; }
    double increment[3] = {0.0, 0.0, 0.0};
    bool touched[3] = {false, false, false};
    void *bars[3] = {(void*)b_mass_edge, (void*)b_pbl_edge, (void*)b_cu_edge};
    bool use_shadow[3] = {bars[0] == bars[1], true, bars[2] == bars[1]};
    if (i % nedges < nsolve) {
        if ((pbl || cu_momentum) && !isfinite(mass_edge[i])) {
            ad_refuse(failure,failure_detail,i,7,1812); return;
        }
        float u = 0.0f;
        if (pbl) u += pbl_edge[i] * mass_edge[i];
        if (cu_momentum) u += cu_edge[i] * mass_edge[i];
        if (!isfinite(u)) { ad_refuse(failure,failure_detail,i,10,1814); return; }
        float g = seed_out_u[i];
        if (pbl) { increment[1] += (double)g * mass_edge[i];
                   increment[0] += (double)g * pbl_edge[i]; touched[0] = touched[1] = true; }
        if (cu_momentum) { increment[2] += (double)g * mass_edge[i];
                           increment[0] += (double)g * cu_edge[i]; touched[0] = touched[2] = true; }
    } else if (b_out_u != seed_out_u) b_out_u[i] += seed_out_u[i];
    if (first) {
        float g = seed_tend_u_phys[i];
        if (cu_momentum) { increment[2] += g; touched[2] = true; }
        if (pbl) { increment[1] += g; touched[1] = true; }
        else if (b_tend_u_phys != seed_tend_u_phys) b_tend_u_phys[i] += g;
    } else if (b_tend_u_phys != seed_tend_u_phys) b_tend_u_phys[i] += seed_tend_u_phys[i];
    // Aliased inputs are one variable. Gather every contribution in a fixed
    // order and round only once when writing its float32 cotangent.
    for (int k = 0; k < 3; ++k) if (touched[k]) {
        bool prior = false;
        for (int j = 0; j < k; ++j) prior |= touched[j] && bars[j] == bars[k];
        if (prior) continue;
        double sum = use_shadow[k] ? ((double*)bars[k])[i] : ((float*)bars[k])[i];
        for (int j = k; j < 3; ++j) if (touched[j] && bars[j] == bars[k]) sum += increment[j];
        if (use_shadow[k]) ((double*)bars[k])[i] = sum; else ((float*)bars[k])[i] = (float)sum;
    }
    // Clear exactly the slots written by the forward, after their last use.
    if (i % nedges < nsolve) seed_out_u[i] = 0.0f;
    // The non-PBL arm carries the old tendency through, including when it
    // adds a convection tendency.  If the binder uses one in-place bar, that
    // incoming seed is already the old-tendency cotangent and must remain.
    if (first && (pbl || cu_momentum) &&
        (pbl || b_tend_u_phys != seed_tend_u_phys)) seed_tend_u_phys[i] = 0.0f;
    (void)out_u;
}
#endif
