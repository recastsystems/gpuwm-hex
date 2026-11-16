// SPDX-License-Identifier: MIT
#define AD_FAILURE_PAIRS(X) X(1767,15,0) X(1769,9,1) X(1772,15,2) X(1772,17,3) X(1777,15,4) X(1779,10,5) X(1782,15,6) X(1782,17,7) X(1786,15,8) X(1787,15,9) X(1788,9,10) X(1788,11,11) X(1788,12,12) X(1788,13,13) X(1788,14,14) X(1790,17,15) X(1792,14,16) X(1792,16,17)
#include "glue.member_g28.cuh"
#include "glue.member_scalar_sum.cuh"

__device__ __forceinline__ void cells_record(int i,int plane,int ncells,int nsolve,
        int nspecies,int pbl,int cu,int lw,int sw,const int *pi,const int *ci,
        float rvord,const float *mass,const float *theta,const float *qv,
        const float *const *rates,const float *out_q,unsigned *r) {
    bool outside=i>=plane || i%ncells>=nsolve;
    ad_branch(r,0,outside);
    if (outside || pi[0] < 0 || pi[0] >= nspecies) return;
    float th=0.0f, m=mass[i], values[13]; int ids[13], used=0;
    ad_branch(r,1,pbl != 0);
    if (pbl) {
        th=ad_record_addsub(th,ad_record_mul(rates[0][i],m,r,11,15,16),false,r,8,15,16);
        for (int s=0;s<8;++s) {
            int sp=pi[s]; ad_branch(r,2,sp>=0);
            if (sp>=0 && sp<nspecies) {
                int k=0; while(k<used && ids[k]!=sp) ++k;
                if(k==used){ids[used]=sp;values[used++]=out_q[sp*plane+i];}
                float prod=ad_record_mul(rates[1+s][i],m,r,11,15,16);
                values[k]=ad_record_addsub(values[k],prod,false,r,8,15,16);
            }
        }
    }
    ad_branch(r,3,cu != 0);
    if (cu) {
        th=ad_record_addsub(th,ad_record_mul(rates[9][i],m,r,11,15,16),false,r,8,15,16);
        for (int s=0;s<5;++s) {
            int sp=ci[s]; ad_branch(r,4,sp>=0);
            if (sp>=0 && sp<nspecies) {
                int k=0; while(k<used && ids[k]!=sp) ++k;
                if(k==used){ids[used]=sp;values[used++]=out_q[sp*plane+i];}
                float prod=ad_record_mul(rates[10+s][i],m,r,11,15,16);
                values[k]=ad_record_addsub(values[k],prod,false,r,8,15,16);
            }
        }
    }
    ad_branch(r,5,lw != 0);
    if(lw) th=ad_record_addsub(th,ad_record_mul(rates[15][i],m,r,11,15,16),false,r,8,15,16);
    ad_branch(r,6,sw != 0);
    if(sw) th=ad_record_addsub(th,ad_record_mul(rates[16][i],m,r,11,15,16),false,r,8,15,16);
    float coeff=ad_record_addsub(1.0f,ad_record_mul(rvord,qv[i],r,11,15,16),false,r,8,15,16);
    ad_branch(r,7,!isfinite(coeff) || coeff==0.0f);
    int k=0; while(k<used && ids[k]!=pi[0]) ++k;
    float vapor=(k<used)?values[k]:out_q[pi[0]*plane+i];
    float left=ad_record_mul(coeff,th,r,11,15,16);
    float product=ad_record_mul(rvord,theta[i],r,11,15,16);
    float numerator=ad_record_mul(product,vapor,r,11,15,16);
    float ratio=ad_record_div(numerator,coeff,r,13,15,16);
    float coupled=ad_record_addsub(left,ratio,false,r,8,15,16);
    (void)ad_record_addsub(0.0f,coupled,false,r,8,15,16);
}

// Slot zero and its exact physical aliases carry double derivative buffers.
__device__ __forceinline__ double cells_rate_tangent(const float *const *table, int slot, int i) {
    return table[slot] == table[0] ? reinterpret_cast<const double *>(table[slot])[i] : (double)table[slot][i];
}
#if !defined(AD_MODE_AD)
extern "C" __global__ void mpas_rk_cells_v842_f32__tl(
    int nlev, int ncells, int nsolve, int nspecies,
    int pbl, int cu, int lw, int sw, const int *pbl_indices, const int *cu_indices,
    float rvord, const float *mass, const float *theta_m, const float *qv,
    const float *const *rates, const float *out_theta, const float *out_q_before,
    const float *out_rho, const float *d_rvord, const float *d_mass,
    const float *d_theta_m, const float *d_qv, const float *const *d_rates,
    float *d_out_theta, float *d_out_q, float *d_out_rho, int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, plane = nlev * ncells;
    if(branch_record) cells_record(i,plane,ncells,nsolve,nspecies,pbl,cu,lw,sw,pbl_indices,cu_indices,rvord,mass,theta_m,qv,rates,out_q_before,branch_record);
    if (i >= plane || i % ncells >= nsolve) return;
    int vapor_species = pbl_indices[0];
    if (vapor_species < 0 || vapor_species >= nspecies) { ad_refuse(failure,failure_detail,0,9,1788); return; }
    if (!isfinite(rvord)) { ad_refuse(failure,failure_detail,0,11,1788); return; }
    if (!isfinite(mass[i])) { ad_refuse(failure,failure_detail,i,12,1788); return; }
    if (!isfinite(theta_m[i])) { ad_refuse(failure,failure_detail,i,13,1788); return; }
    if (!isfinite(qv[i])) { ad_refuse(failure,failure_detail,i,14,1788); return; }
    int vi = vapor_species * plane + i;
    float vapor = out_q_before[vi], th = 0.0f;
    double dvapor = d_out_q[vi], dth = 0.0;
    int species_ids[13], species_used = 0;
    double species_increment[13] = {};
    float m = mass[i], dm = d_mass[i];
    if (!isfinite(vapor)) { ad_refuse(failure,failure_detail,vi,17,1790); return; }
    if (pbl) {
        if (!isfinite(rates[0][i])) { ad_refuse(failure,failure_detail,i,15,1767); return; }
        th += rates[0][i] * m; dth += cells_rate_tangent(d_rates,0,i) * m + (double)rates[0][i] * dm;
        for (int s = 0; s < 8; ++s) {
            int species = pbl_indices[s];
            if (species >= nspecies) { ad_refuse(failure,failure_detail,s,9,1769); return; }
            if (species >= 0) {
                int si = species * plane + i;
                float r = rates[1+s][i]; double dr = cells_rate_tangent(d_rates,1+s,i);
                if (!isfinite(r)) { ad_refuse(failure,failure_detail,(unsigned long long)(1+s)*plane+i,15,1772); return; }
                if (!isfinite(out_q_before[si])) { ad_refuse(failure,failure_detail,si,17,1772); return; }
                double inc = (double)dr * m + (double)r * dm;
                int k = 0; while (k < species_used && species_ids[k] != species) ++k;
                if (k == species_used) species_ids[species_used++] = species;
                species_increment[k] += inc;
                if (species == vapor_species) { vapor += r * m; dvapor += inc; }
            }
        }
    }
    if (cu) {
        if (!isfinite(rates[9][i])) { ad_refuse(failure,failure_detail,(unsigned long long)9*plane+i,15,1777); return; }
        th += rates[9][i] * m; dth += cells_rate_tangent(d_rates,9,i) * m + (double)rates[9][i] * dm;
        for (int s = 0; s < 5; ++s) {
            int species = cu_indices[s];
            if (species >= nspecies) { ad_refuse(failure,failure_detail,s,10,1779); return; }
            if (species >= 0) {
                int si = species * plane + i;
                float r = rates[10+s][i]; double dr = cells_rate_tangent(d_rates,10+s,i);
                if (!isfinite(r)) { ad_refuse(failure,failure_detail,(unsigned long long)(10+s)*plane+i,15,1782); return; }
                if (!isfinite(out_q_before[si])) { ad_refuse(failure,failure_detail,si,17,1782); return; }
                double inc = (double)dr * m + (double)r * dm;
                int k = 0; while (k < species_used && species_ids[k] != species) ++k;
                if (k == species_used) species_ids[species_used++] = species;
                species_increment[k] += inc;
                if (species == vapor_species) { vapor += r * m; dvapor += inc; }
            }
        }
    }
    if (lw) { if (!isfinite(rates[15][i])) { ad_refuse(failure,failure_detail,(unsigned long long)15*plane+i,15,1786); return; }
        th += rates[15][i] * m; dth += cells_rate_tangent(d_rates,15,i) * m + (double)rates[15][i] * dm; }
    if (sw) { if (!isfinite(rates[16][i])) { ad_refuse(failure,failure_detail,(unsigned long long)16*plane+i,15,1787); return; }
        th += rates[16][i] * m; dth += cells_rate_tangent(d_rates,16,i) * m + (double)rates[16][i] * dm; }
    float coeff = 1.0f + rvord * qv[i];
    if (!isfinite(coeff) || coeff == 0.0f) { ad_refuse(failure,failure_detail,i,14,1792); return; }
    double dcoeff = (double)(*d_rvord) * qv[i] + (double)rvord * d_qv[i];
    float product = rvord * theta_m[i], term = product * vapor;
    float quotient = term / coeff;
    if (!isfinite(th) || !isfinite(vapor) || !isfinite(quotient) || !isfinite(coeff*th+quotient)) {
        ad_refuse(failure,failure_detail,i,16,1792); return;
    }
    double dproduct = (double)(*d_rvord) * theta_m[i] + (double)rvord * d_theta_m[i];
    double dterm = dproduct * vapor + (double)product * dvapor;
    reinterpret_cast<double *>(d_out_theta)[i] = dcoeff * th + (double)coeff * dth + (dterm - (double)quotient * dcoeff) / coeff;
    for (int k = 0; k < species_used; ++k) {
        int si = species_ids[k] * plane + i;
        d_out_q[si] = (float)((double)d_out_q[si] + species_increment[k]);
    }
    d_out_rho[i] = 0.0f;
    (void)out_theta; (void)out_rho;
}
#endif

#if !defined(AD_MODE_TL)
extern "C" __global__ void mpas_rk_cells_v842_f32__ad(
    int nlev, int ncells, int nsolve, int nspecies,
    int pbl, int cu, int lw, int sw, const int *pbl_indices, const int *cu_indices,
    float rvord, const float *mass, const float *theta_m, const float *qv,
    const float *const *rates, const float *out_theta, const float *out_q_before,
    const float *out_rho, float *b_mass, float *b_theta_m, float *b_qv,
    float *const *b_rates, float *seed_out_theta, float *seed_out_q,
    float *seed_out_rho, float *b_out_theta, float *b_out_q, float *b_out_rho,
    float *b_rvord_work, int *failure, unsigned long long *failure_detail, unsigned *branch_record) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, plane = nlev * ncells;
    if(branch_record) cells_record(i,plane,ncells,nsolve,nspecies,pbl,cu,lw,sw,pbl_indices,cu_indices,rvord,mass,theta_m,qv,rates,out_q_before,branch_record);
    if (i >= plane) return;
    b_rvord_work[i] = 0.0f;
    bool active = i % ncells < nsolve;
    int vapor_species = active ? pbl_indices[0] : -1;
    for (int s = 0; s < nspecies; ++s) {
        int si = s * plane + i;
        if (b_out_q != seed_out_q) b_out_q[si] += seed_out_q[si];
    }
    if (!active) {
        if (b_out_theta != seed_out_theta) reinterpret_cast<double *>(b_out_theta)[i] += reinterpret_cast<const double *>(seed_out_theta)[i];
        if (b_out_rho != seed_out_rho) b_out_rho[i] += seed_out_rho[i];
        return;
    }
    if (vapor_species < 0 || vapor_species >= nspecies) { ad_refuse(failure,failure_detail,0,9,1788); return; }
    if (!isfinite(rvord)) { ad_refuse(failure,failure_detail,0,11,1788); return; }
    if (!isfinite(mass[i])) { ad_refuse(failure,failure_detail,i,12,1788); return; }
    if (!isfinite(theta_m[i])) { ad_refuse(failure,failure_detail,i,13,1788); return; }
    if (!isfinite(qv[i])) { ad_refuse(failure,failure_detail,i,14,1788); return; }
    int vi = vapor_species * plane + i;
    float vapor = out_q_before[vi], th = 0.0f, m = mass[i];
    if (!isfinite(vapor)) { ad_refuse(failure,failure_detail,vi,17,1790); return; }
    if (pbl) {
        if (!isfinite(rates[0][i])) { ad_refuse(failure,failure_detail,i,15,1767); return; }
        th += rates[0][i] * m;
        for (int s = 0; s < 8; ++s) {
            int species = pbl_indices[s];
            if (species >= nspecies) { ad_refuse(failure,failure_detail,s,9,1769); return; }
            if (species >= 0) {
                int si = species * plane + i;
                if (!isfinite(rates[1+s][i])) { ad_refuse(failure,failure_detail,(unsigned long long)(1+s)*plane+i,15,1772); return; }
                if (!isfinite(out_q_before[si])) { ad_refuse(failure,failure_detail,si,17,1772); return; }
                if (species == vapor_species) vapor += rates[1+s][i] * m;
            }
        }
    }
    if (cu) {
        if (!isfinite(rates[9][i])) { ad_refuse(failure,failure_detail,(unsigned long long)9*plane+i,15,1777); return; }
        th += rates[9][i] * m;
        for (int s = 0; s < 5; ++s) {
            int species = cu_indices[s];
            if (species >= nspecies) { ad_refuse(failure,failure_detail,s,10,1779); return; }
            if (species >= 0) {
                int si = species * plane + i;
                if (!isfinite(rates[10+s][i])) { ad_refuse(failure,failure_detail,(unsigned long long)(10+s)*plane+i,15,1782); return; }
                if (!isfinite(out_q_before[si])) { ad_refuse(failure,failure_detail,si,17,1782); return; }
                if (species == vapor_species) vapor += rates[10+s][i] * m;
            }
        }
    }
    if (lw) { if (!isfinite(rates[15][i])) { ad_refuse(failure,failure_detail,(unsigned long long)15*plane+i,15,1786); return; } th += rates[15][i] * m; }
    if (sw) { if (!isfinite(rates[16][i])) { ad_refuse(failure,failure_detail,(unsigned long long)16*plane+i,15,1787); return; } th += rates[16][i] * m; }
    float coeff = 1.0f + rvord * qv[i], product = rvord * theta_m[i];
    if (!isfinite(coeff) || coeff == 0.0f) { ad_refuse(failure,failure_detail,i,14,1792); return; }
    float term = product * vapor, quotient = term / coeff;
    if (!isfinite(th) || !isfinite(vapor) || !isfinite(quotient) || !isfinite(coeff*th+quotient)) {
        ad_refuse(failure,failure_detail,i,16,1792); return;
    }
    double g = reinterpret_cast<const double *>(seed_out_theta)[i];
    double bth = g * coeff, bc = g * th, bterm = g / coeff;
    bc -= bterm * quotient;
    double bproduct = bterm * vapor;
    b_theta_m[i] = (float)((double)b_theta_m[i] + bproduct * rvord);
    b_qv[i] = (float)((double)b_qv[i] + bc * rvord);
    b_rvord_work[i] = (float)(bproduct * theta_m[i] + bc * qv[i]);
    double bm = 0.0, rate_increment[17] = {};
    bool rate_used[17] = {};
    if (sw) { rate_increment[16] += bth * m; rate_used[16] = true; bm += bth * rates[16][i]; }
    if (lw) { rate_increment[15] += bth * m; rate_used[15] = true; bm += bth * rates[15][i]; }
    if (cu) {
        for (int s = 4; s >= 0; --s) {
            int species = cu_indices[s];
            if (species >= 0) {
                int si = species * plane + i;
                double bq = (double)seed_out_q[si] + (species == vapor_species ? bterm * product : 0.0);
                rate_increment[10+s] += bq * m; rate_used[10+s] = true; bm += bq * rates[10+s][i];
            }
        }
        rate_increment[9] += bth * m; rate_used[9] = true; bm += bth * rates[9][i];
    }
    if (pbl) {
        for (int s = 7; s >= 0; --s) {
            int species = pbl_indices[s];
            if (species >= 0) {
                int si = species * plane + i;
                double bq = (double)seed_out_q[si] + (species == vapor_species ? bterm * product : 0.0);
                rate_increment[1+s] += bq * m; rate_used[1+s] = true; bm += bq * rates[1+s][i];
            }
        }
        rate_increment[0] += bth * m; rate_used[0] = true; bm += bth * rates[0][i];
    }
    // q is an in-place increment.  Its seed is also read by every selected
    // species update above, so add the vapor-coupling term only afterwards.
    b_out_q[vi] = (float)((double)b_out_q[vi] + bterm * product);
    // The captured rate table may repeat pointers. Consolidate cotangents for
    // each distinct input before the one float32 store, without atomics.
    for (int k = 0; k < 17; ++k) if (rate_used[k]) {
        bool prior = false;
        for (int j = 0; j < k; ++j) prior |= rate_used[j] && b_rates[j] == b_rates[k];
        if (prior) continue;
        double sum = b_rates[k] == b_rates[0] ? reinterpret_cast<double *>(b_rates[k])[i] : (double)b_rates[k][i];
        for (int j = k; j < 17; ++j)
            if (rate_used[j] && b_rates[j] == b_rates[k]) sum += rate_increment[j];
        if (b_rates[k] == b_rates[0]) reinterpret_cast<double *>(b_rates[k])[i] = sum; else b_rates[k][i] = (float)sum;
    }
    b_mass[i] = (float)((double)b_mass[i] + bm);
    // The forward overwrites theta and rho at every active cell.  A species
    // seed is consumed only when at least one enabled map writes it.
    reinterpret_cast<double *>(seed_out_theta)[i] = 0.0;
    seed_out_rho[i] = 0.0f;
    if (b_out_q != seed_out_q && pbl) for (int s = 0; s < 8; ++s)
        if (pbl_indices[s] >= 0) seed_out_q[pbl_indices[s] * plane + i] = 0.0f;
    if (b_out_q != seed_out_q && cu) for (int s = 0; s < 5; ++s)
        if (cu_indices[s] >= 0) seed_out_q[cu_indices[s] * plane + i] = 0.0f;
    (void)out_theta; (void)out_rho;
}
#endif
