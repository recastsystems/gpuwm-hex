// SPDX-License-Identifier: Apache-2.0
// Simplified inner model, boundary layer (0.6.2): the tangent-linear and adjoint of a linear vertical
// diffusion whose exchange coefficients are read from the trajectory and held fixed, in place of the boundary-layer
// scheme's own derivative. Written for this engine; it reads the scheme's launch arguments and nothing of its code.
//
// At a boundary-layer launch the scheme's own state from its call before is in el_pbl, qke, sh3d and sm3d (it carries
// them between calls). The linear model forms MYNN's own diffusivities from them, K_h = el sqrt(qke) sh3d and
// K_m = el sqrt(qke) sm3d (m2/s, on the interface below each level; zero at the ground), from the launch's pre-state,
// so the tangent and the adjoint read the same numbers, and holds them (kmode 1). kmode 2 holds the boundary layer
// whole: no mixing and no surface exchange, so its tendencies' tangent is 0:
//
//   x_new = T^-1 x,   tend = (x_new - x) / delt,   for x = th, qv (with exch_h) and u, v (with exch_m)
//
// T is the scheme's own implicit tridiagonal form (MYNN-EDMF mynn_tendencies: rhoz, dtz, rhoinv) with the
// mass-flux, counter-gradient and cloud terms left out. The lowest level adds a linear surface exchange with held
// coefficients: momentum ust^2 / wspd (the scheme's own drag form), heat ch, moisture ch times the moisture
// availability the trajectory's own flux implies (qfx / (rho ch (qsfc - qv)), in [0, 1]). The skin temperature and
// the surface humidity are held. Every other output of the launch is held (its tangent is 0). The adjoint is the
// exact transpose. One thread a column, float64 throughout (the shadows are float64).
//
// sfc_* switches (AD_IM_SFC on the host): all three on by default; "momentum" keeps only the drag; "none" no
// surface exchange.
//
// Second form (cond_tau > 0, AD_IM_COND on the host; off unless asked): a smooth large-scale condensation added to the
// same tendencies. With s = qv - qs(T, p) (saturation mixing ratio over liquid, Bolton's es) and a blend width
// e = 0.05 qs, the condensation rate is C(s) = (s + sqrt(s^2 + e^2)) / (2 tau): no threshold, about s / tau well above
// saturation and about 0 well below it. Its derivative at the trajectory, g = (1 + s / sqrt(s^2 + e^2)) / (2 tau), is
// held, with the blend width's own change with temperature: C' = dC/ds (qv' - dqs/dT exner th') + dC/de 0.05 dqs/dT
// exner th' at fixed pressure; the vapour tendency takes -C' and the theta tendency
// takes L C' / (cp exner). So a vapour increment where the trajectory is near or past saturation condenses and heats
// the column over about tau; where it is well below saturation it stays vapour.

#define IM_NK_MAX 160
#define IM_MAX_OUT 96

struct ImOut {
    unsigned long long p[IM_MAX_OUT];   // shadow (float64) of each held output's first word
    int dims[IM_MAX_OUT];               // 2: one word a column; 3: one word a column and level
    int n;
};

__device__ static void im_thomas(int n, const double* sub, const double* dia, const double* sup, const double* r,
                                 double* cp, double* dp, double* y) {
    double m = dia[0];
    cp[0] = sup[0] / m;
    dp[0] = r[0] / m;
    for (int k = 1; k < n; ++k) {
        m = dia[k] - sub[k] * cp[k - 1];
        cp[k] = sup[k] / m;
        dp[k] = (r[k] - sub[k] * dp[k - 1]) / m;
    }
    y[n - 1] = dp[n - 1];
    for (int k = n - 2; k >= 0; --k) y[k] = dp[k] - cp[k] * y[k + 1];
}

__device__ static double im_clean(double v) { return (isfinite(v) && v > 0.0) ? v : 0.0; }

extern "C" __global__ void im_bl_fixedk(int mode, int first, int count, int its, int ite, int ims, int ime, int kms,
                                        int kts, int kte, float delt, int sfc_heat, int sfc_moist, int sfc_mom,
                                        float cond_tau, const float* __restrict__ tt, const float* __restrict__ pp,
                                        const float* __restrict__ exner,
                                        const float* __restrict__ dz, const float* __restrict__ rho, int kmode,
                                        const float* __restrict__ el, const float* __restrict__ qke,
                                        const float* __restrict__ shh, const float* __restrict__ smm,
                                        const float* __restrict__ ust, const float* __restrict__ wspd,
                                        const float* __restrict__ ch, const float* __restrict__ qfx,
                                        const float* __restrict__ qsfc, const float* __restrict__ qv,
                                        double* s_th, double* s_qv, double* s_u, double* s_v,
                                        double* s_rth, double* s_rqv, double* s_ru, double* s_rv, ImOut outs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= count) return;
    const int index = first + c;
    const int i = its + index % (ite - its + 1);
    const long ni = (long)(ime - ims + 1);
    const long col = (long)(i - ims);
    const int nk = kte - kts + 1;
    if (nk < 2 || nk > IM_NK_MAX) return;   // refused on the host before the launch
    auto at = [&](int k) -> long { return col + (long)(k + kts - kms) * ni; };

    double dzv[IM_NK_MAX], rh[IM_NK_MAX], kkh[IM_NK_MAX], kkm[IM_NK_MAX];
    double a[IM_NK_MAX], b[IM_NK_MAX], cc[IM_NK_MAX], x[IM_NK_MAX], y[IM_NK_MAX], cp[IM_NK_MAX], dp[IM_NK_MAX];
    for (int k = 0; k < nk; ++k) {
        dzv[k] = (double)dz[at(k)];
        double r = (double)rho[at(k)];
        rh[k] = (isfinite(r) && r > 1e-4) ? r : 1e-4;
    }
    kkh[0] = 0.0; kkm[0] = 0.0;
    for (int m = 1; m < nk; ++m) {
        const double rhoz = (rh[m] * dzv[m - 1] + rh[m - 1] * dzv[m]) / (dzv[m - 1] + dzv[m]);
        const double dzi = 0.5 * (dzv[m] + dzv[m - 1]);
        double kh = 0.0, km = 0.0;
        if (kmode == 1) {
            const double lq = im_clean((double)el[at(m)]) * sqrt(im_clean((double)qke[at(m)]));
            kh = lq * im_clean((double)shh[at(m)]);
            km = lq * im_clean((double)smm[at(m)]);
        }
        kkh[m] = rhoz * kh / dzi;
        kkm[m] = rhoz * km / dzi;
    }
    // held surface exchange velocities (m/s)
    double vm = 0.0, vh = 0.0, vq = 0.0;
    const double chv = im_clean((double)ch[col]);
    if (sfc_mom) {
        const double us = im_clean((double)ust[col]);
        double ws = (double)wspd[col];
        ws = (isfinite(ws) && ws > 0.1) ? ws : 0.1;
        vm = us * us / ws;
    }
    if (sfc_heat) vh = chv;
    if (sfc_moist && chv > 0.0) {
        const double dq = (double)qsfc[col] - (double)qv[at(0)];
        const double fq = (double)qfx[col];
        double beta = 0.0;
        if (isfinite(dq) && isfinite(fq) && fabs(dq) > 1e-6) beta = fq / (rh[0] * chv * dq);
        beta = isfinite(beta) ? fmin(fmax(beta, 0.0), 1.0) : 0.0;
        vq = chv * beta;
    }
    const double dt = (double)delt;
    // second form: the held slopes of C on vapour (gc, 1/s) and on theta (-qt, 1/(K s)) at each level, from the trajectory
    double gc[IM_NK_MAX], qt[IM_NK_MAX], ex[IM_NK_MAX];
    const bool cond = cond_tau > 0.f;
    if (cond) {
        const double tau = (double)cond_tau;
        for (int k = 0; k < nk; ++k) {
            const double T = (double)tt[at(k)], P = (double)pp[at(k)], E = (double)exner[at(k)], q = (double)qv[at(k)];
            gc[k] = 0.0; qt[k] = 0.0; ex[k] = (isfinite(E) && E > 0.0) ? E : 1.0;
            if (!(isfinite(T) && isfinite(P) && isfinite(q)) || T < 150.0 || P < 100.0) continue;
            const double es = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65));
            if (!(es < 0.5 * P)) continue;
            const double qs = 0.622 * es / (P - es);
            const double dqs = qs * P / (P - es) * 17.67 * 243.5 / ((T - 29.65) * (T - 29.65));
            const double s = q - qs, e = 0.05 * qs, r = sqrt(s * s + e * e);
            gc[k] = 0.5 * (1.0 + s / r) / tau;                     // dC/ds
            const double ge = 0.5 * e / r / tau;                    // dC/de, e = 0.05 qs(T)
            qt[k] = (gc[k] - 0.05 * ge) * dqs * ex[k];               // -dC/dth at fixed qv and p
        }
        if (mode == 1) {   // C^T on the tendency bars before the diffusion transposes read and clear them
            for (int k = 0; k < nk; ++k) {
                const double cb = -s_rqv[at(k)] + 2.5e6 / (1004.5 * ex[k]) * s_rth[at(k)];
                s_qv[at(k)] += gc[k] * cb;
                s_th[at(k)] -= qt[k] * cb;
            }
        }
    }
    double* sx[4] = {s_th, s_qv, s_u, s_v};
    double* st[4] = {s_rth, s_rqv, s_ru, s_rv};
    const double* kk[4] = {kkh, kkh, kkm, kkm};
    const double vs[4] = {vh, vq, vm, vm};
    for (int f = 0; f < 4; ++f) {
        for (int k = 0; k < nk; ++k) {
            const double dtz = dt / dzv[k];
            a[k] = (k > 0) ? -dtz * kk[f][k] / rh[k] : 0.0;
            cc[k] = (k < nk - 1) ? -dtz * kk[f][k + 1] / rh[k] : 0.0;
            b[k] = 1.0 - a[k] - cc[k];
        }
        b[0] += dt / dzv[0] * vs[f];
        if (mode == 0) {
            for (int k = 0; k < nk; ++k) x[k] = sx[f][at(k)];
            im_thomas(nk, a, b, cc, x, cp, dp, y);
            for (int k = 0; k < nk; ++k) st[f][at(k)] = (y[k] - x[k]) / dt;
        } else {
            // x_bar += (T^-T - I) tend_bar / delt ; tend_bar = 0
            for (int k = 0; k < nk; ++k) x[k] = st[f][at(k)] / dt;
            // T^T: sub-diagonal c[k-1], super-diagonal a[k+1]
            double sub[IM_NK_MAX], sup[IM_NK_MAX];
            for (int k = 0; k < nk; ++k) {
                sub[k] = (k > 0) ? cc[k - 1] : 0.0;
                sup[k] = (k < nk - 1) ? a[k + 1] : 0.0;
            }
            im_thomas(nk, sub, b, sup, x, cp, dp, y);
            for (int k = 0; k < nk; ++k) {
                sx[f][at(k)] += y[k] - x[k];
                st[f][at(k)] = 0.0;
            }
        }
    }
    if (cond && mode == 0) {   // the tangent adds C x to the two tendencies the diffusion just wrote
        for (int k = 0; k < nk; ++k) {
            const double cx = gc[k] * s_qv[at(k)] - qt[k] * s_th[at(k)];
            s_rqv[at(k)] -= cx;
            s_rth[at(k)] += 2.5e6 / (1004.5 * ex[k]) * cx;
        }
    }
    // every other output the launch writes is held: tangent 0 (and so its adjoint is 0 after the launch)
    for (int o = 0; o < outs.n && o < IM_MAX_OUT; ++o) {
        double* s = (double*)outs.p[o];
        if (outs.dims[o] == 2) { s[col] = 0.0; }
        else { for (int k = 0; k < nk; ++k) s[at(k)] = 0.0; }
    }
}
