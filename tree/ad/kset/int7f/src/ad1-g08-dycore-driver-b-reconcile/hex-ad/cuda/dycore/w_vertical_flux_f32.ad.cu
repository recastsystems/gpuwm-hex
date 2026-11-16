// SPDX-License-Identifier: Apache-2.0
#include "dycore_math.cuh"
// The sign bit reproduces mpas_abs, including the selected side at signed zero.
__device__ __forceinline__ float ad_abs_side(float v) {
    return (__float_as_uint(v) & 0x80000000u) ? -1.0f : 1.0f;
}
extern "C" __global__ void w_vertical_flux_f32__tl(
    int nlev, int ncells, const float *w, const float *rw,
    const float *dw, const float *drw, float *dflux, unsigned *branch_record, int *failures)
{
    AD_LOOP((nlev + 1) * ncells) {
        const int k = i / ncells;
        float d = 0.0f;
        if (k > 0 && k < nlev) {
            const int lo = i - ncells;
            const float v = mpas_mul(0.5f, mpas_add(rw[i], rw[lo]));
            const double dv = 0.5 * ((double)drw[i] + (double)drw[lo]);
            if (k == 1 || k == nlev - 1) {
                const float q = mpas_add(w[i], w[lo]);
                d = (float)(0.5 * dv * (double)q + 0.5 * (double)v * ((double)dw[i] + (double)dw[lo]));
            } else {
                const int lo2 = i - 2 * ncells, hi = i + ncells;
                atomicAdd(&branch_record[4+(__float_as_uint(v)>>31)], 1u);
                const float a = mpas_sub(mpas_mul(7.0f, mpas_add(w[i], w[lo])),
                                         mpas_add(w[hi], w[lo2]));
                const float b = mpas_sub(mpas_sub(w[hi], w[lo2]),
                                         mpas_mul(3.0f, mpas_sub(w[i], w[lo])));
                const double da = 7.0 * ((double)dw[i] + (double)dw[lo]) - ((double)dw[hi] + (double)dw[lo2]);
                const double db = (double)dw[hi] - (double)dw[lo2] - 3.0 * ((double)dw[i] - (double)dw[lo]);
                d = (float)((dv * (double)a + (double)v * da + (double)ad_abs_side(v) * dv * (double)b + (double)mpas_abs(v) * db) / 12.0);
                if (!isfinite(w[lo2]) || !isfinite(w[hi]) || !isfinite(dw[lo2]) || !isfinite(dw[hi]) ||
                    !isfinite(a) || !isfinite(b)) ad_fail(failures, i, 3, i, 1);
            }
            if (!isfinite(w[i]) || !isfinite(w[lo]) || !isfinite(rw[i]) || !isfinite(rw[lo]) ||
                !isfinite(dw[i]) || !isfinite(dw[lo]) || !isfinite(drw[i]) || !isfinite(drw[lo]) ||
                !isfinite(v) || !isfinite(d)) ad_fail(failures, i, 3, i, 1);
        }
        dflux[i] = d;
    }
}
// One thread owns a full column and gathers each input level's contributions
// in increasing output-level order. Each sum is rounded only on final store.
extern "C" __global__ void w_vertical_flux_f32__ad(
    int nlev, int ncells, const float *w, const float *rw,
    const float *bflux, float *bw, float *brw, int *failures)
{
    AD_LOOP(ncells) {
        const int c = i;
        for (int j = 0; j <= nlev; ++j) {
            const int ix = j * ncells + c;
            double wbar = (double)bw[ix], rwbar = (double)brw[ix];
            for (int k = max(1, j - 1); k <= min(nlev - 1, j + 2); ++k) {
                const int x = k * ncells + c, lo = x - ncells;
                const float v = mpas_mul(0.5f, mpas_add(rw[x], rw[lo]));
                const float g = bflux[x];
                double bv;
                if (k == 1 || k == nlev - 1) {
                    const float q = mpas_add(w[x], w[lo]);
                    bv = 0.5 * (double)g * q;
                    if (j == k || j == k - 1) wbar += 0.5 * (double)g * v;
                } else {
                    const int lo2 = x - 2 * ncells, hi = x + ncells;
                    const float a = mpas_sub(mpas_mul(7.0f, mpas_add(w[x], w[lo])),
                                             mpas_add(w[hi], w[lo2]));
                    const float b = mpas_sub(mpas_sub(w[hi], w[lo2]),
                                             mpas_mul(3.0f, mpas_sub(w[x], w[lo])));
                    const double ba = (double)g * v / 12.0;
                    const double bb = (double)g * mpas_abs(v) / 12.0;
                    bv = (double)g * ((double)a + ad_abs_side(v) * (double)b) / 12.0;
                    if (j == k) wbar += 7.0 * ba - 3.0 * bb;
                    if (j == k - 1) wbar += 7.0 * ba + 3.0 * bb;
                    if (j == k + 1) wbar += -ba + bb;
                    if (j == k - 2) wbar += -ba - bb;
                    if (!isfinite(w[lo2]) || !isfinite(w[hi]) || !isfinite(a) || !isfinite(b))
                        ad_fail(failures, c, 1, x, 1);
                }
                if (j == k || j == k - 1) rwbar += 0.5 * bv;
                if (!isfinite(w[x]) || !isfinite(w[lo]) || !isfinite(rw[x]) || !isfinite(rw[lo]) ||
                    !isfinite(v) || !isfinite(g) || !isfinite(bv)) ad_fail(failures, c, 1, x, 1);
            }
            const float wresult = (float)wbar, rwresult = (float)rwbar;
            if (!isfinite(wbar) || !isfinite(rwbar) || !isfinite(wresult) || !isfinite(rwresult))
                ad_fail(failures, c, 1, ix, 1);
            bw[ix] = wresult;
            brw[ix] = rwresult;
        }
    }
}
