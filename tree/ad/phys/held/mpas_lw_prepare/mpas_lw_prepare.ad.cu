// SPDX-License-Identifier: BSD-3-Clause
// Private autodiff program, held lane ad/held-mpas_lw_prepare.
// Tangent (TL) and adjoint (AD) of the RRTMG LW column-preparation launch `mpas_lw_prepare` (hex pin 16251589,
// hexcore/mpasphys/rrtmg_lw_v842.py:412; captured forward source 683d7a07..., lines 9281-9463; MPAS-A v8.4.2
// module_ra_rrtmg_lw.F:11810-12164, module_ra_rrtmg_vinterp.F:21-179), written from the forward CUDA source.
//
// Forward, one thread per column c (pointers offset per column exactly as the forward wrapper does):
//   S1  P(k) = pI[k]/100, T(k) = tI[k]                              k = 0..nm           (lev, working storage)
//   S2  A2 = running dz sum + dz/2; A3 = max(qv,1e-12)*1.607793; k<nm: A0 = p/100, A1 = t;
//       k>=nm: P(k+1) = max(P(k)-4, 0), A0 = (P(k)+P(k+1))/2       k = 0..nl-1
//   S3  P(nl) = 0; A0[nl-1] = (P(nl-1)+P(nl))/2
//   S4  anchor = T(nm-1) - profile(P(nm-1))
//   S5  T(k) = profile(P(k)) + anchor; A1[k-1] = (T(k)+T(k-1))/2  k = nm..nl            (overwrites T(nm), A1[nm-1])
//   S6  A4 = climatological ozone (o3input != 2)
//   S7  A4 = vinterp(A0, pin, ozone) top to bottom with a carried search index  (o3input == 2; overwrites S6)
//   S8  A5..A11 = cloud fraction, liquid/ice/snow paths, effective radii                k < nm (constants above)
// Active inputs: m [nc,12,nm], interfaces [nc,2,nm+1], surface [nc,6], pin [nz], ozone [nc,nz].
// Outputs, fully overwritten: a [nc,12,nl], lev [nc,2,nl+1]. Inactive: the ten integer scalars.
//
// Derivative rule: the exact derivative of the executed branch, evaluated on the forward's own primal, which the
// entries recompute from the pre-launch inputs with the forward's operations in the forward's order (the outputs'
// pre-launch words are stale). Integer search indices (profile j, ice-radius j, vinterp upper) and every switch are
// read from that primal and frozen. Every limiter-like clamp is a frozen coefficient: max(x,c)/min(x,c)/clamp01 pass
// the derivative (coefficient 1) only when the executed side is the unclamped one, strictly (x > c for max, x < c for
// min, 0 < x < 1 for clamp01), and give 0 when the bound is active, including the tie and a NaN operand (fmaxf/fminf
// return the bound there). Sites: qv floor 1e-12, P(k+1) = max(P-4,0), climatology max(.,0) layer overlaps,
// q floors max(q,0), denom = max(0.01,cf), re floors max(2.5/5/10, re*1e6), rei cap min(140,.), liquid-radius clamp01
// (x4), snow re cap at 130 (factor branch). TL and AD use the same scalar partial coefficients, so the AD is the exact
// transpose of the TL (up to float32 accumulation order).
// Precision: float32 throughout, explicit round-to-nearest intrinsics, built with --fmad=false --ftz=false (the live
// schemes' rules); a zero coefficient contributes exactly 0 (no 0*inf).
//
// pin is one array shared by every column: its adjoint is a cross-column sum. To keep the reverse bitwise
// reproducible, each column writes its pin contributions to its own workspace row, and the last block to finish
// (threadfence + ticket) sums the rows in column order 0..nc-1 and adds the sum to the pin adjoint.
//
// Failure slots (one uint32 each, nonzero = fired; ids in the spec):
//   0 5101 shape outside the entry's contract (nm < 1, nl < nm, nl+1 > LWP_NLMAX, o3input == 2 with nz < 2)
//   1 5102 derivative views overlap (any two of the seven)
//   2 5103 nonfinite recomputed primal, partial coefficient or tangent result
//   3 5104 nonfinite incoming tangent (TL) or seed (AD)
// Branch record (TL; exact integer counts, slot s = [false, true]):
//   0 P(k+1) clamp active   1 qv floor active   2 vinterp above pin[0]   3 vinterp below pin[nz-1]
//   4 q floor active (qc/qi/qs)   5 denom floor active (cf <= 0.01)   6 re floor or replacement active
//   7 snow re cap (> 130)   8 climatology overlap mask clamped (o3input != 2)

#define LWP_NLMAX 256
#define LAD(a, b) __fadd_rn((a), (b))
#define LSU(a, b) __fsub_rn((a), (b))
#define LMU(a, b) __fmul_rn((a), (b))
#define LDV(a, b) __fdiv_rn((a), (b))

// Tables, verbatim from the captured forward source (683d7a07, lines 9281-9287), renamed mpas_ -> lwp_.
__device__ const float lwp_retab[95] = {5.92779f,6.26422f,6.61973f,6.99539f,7.39234f,7.81177f,8.25496f,8.72323f,9.21800f,9.74075f,10.2930f,10.8765f,11.4929f,12.1440f,12.8317f,13.5581f,14.2319f,15.0351f,15.8799f,16.7674f,17.6986f,18.6744f,19.6955f,20.7623f,21.8757f,23.0364f,24.2452f,25.5034f,26.8125f,27.7895f,28.6450f,29.4167f,30.1088f,30.7306f,31.2943f,31.8151f,32.3077f,32.7870f,33.2657f,33.7540f,34.2601f,34.7892f,35.3442f,35.9255f,36.5316f,37.1602f,37.8078f,38.4720f,39.1508f,39.8442f,40.5552f,41.2912f,42.0635f,42.8876f,43.7863f,44.7853f,45.9170f,47.2165f,48.7221f,50.4710f,52.4980f,54.8315f,57.4898f,60.4785f,63.7898f,65.5604f,71.2885f,75.4113f,79.7368f,84.2351f,88.8833f,93.6658f,98.5739f,103.603f,108.752f,114.025f,119.424f,124.954f,130.630f,136.457f,142.446f,148.608f,154.956f,161.503f,168.262f,175.248f,182.473f,189.952f,197.699f,205.728f,214.055f,222.694f,231.661f,240.971f,250.639f};
__device__ const float lwp_pprof[60] = {1000.00f,855.47f,731.82f,626.05f,535.57f,458.16f,391.94f,335.29f,286.83f,245.38f,209.91f,179.57f,153.62f,131.41f,112.42f,96.17f,82.27f,70.38f,60.21f,51.51f,44.06f,37.69f,32.25f,27.59f,23.60f,20.19f,17.27f,14.77f,12.64f,10.81f,9.25f,7.91f,6.77f,5.79f,4.95f,4.24f,3.63f,3.10f,2.65f,2.27f,1.94f,1.66f,1.42f,1.22f,1.04f,0.89f,0.76f,0.65f,0.56f,0.48f,0.41f,0.35f,0.30f,0.26f,0.22f,0.19f,0.16f,0.14f,0.12f,0.10f};
__device__ const float lwp_tprof[60] = {286.96f,281.07f,275.16f,268.11f,260.56f,253.02f,245.62f,238.41f,231.57f,225.91f,221.72f,217.79f,215.06f,212.74f,210.25f,210.16f,210.69f,212.14f,213.74f,215.37f,216.82f,217.94f,219.03f,220.18f,221.37f,222.64f,224.16f,225.88f,227.63f,229.51f,231.50f,233.73f,236.18f,238.78f,241.60f,244.44f,247.35f,250.33f,253.32f,256.30f,259.22f,262.12f,264.80f,266.50f,267.59f,268.44f,268.69f,267.76f,266.13f,263.96f,261.54f,258.93f,256.15f,253.23f,249.89f,246.67f,243.48f,240.25f,236.66f,233.86f};
__device__ const float lwp_o3sum[31] = {5.297E-8f,5.852E-8f,6.579E-8f,7.505E-8f,8.577E-8f,9.895E-8f,1.175E-7f,1.399E-7f,1.677E-7f,2.003E-7f,2.571E-7f,3.325E-7f,4.438E-7f,6.255E-7f,8.168E-7f,1.036E-6f,1.366E-6f,1.855E-6f,2.514E-6f,3.240E-6f,4.033E-6f,4.854E-6f,5.517E-6f,6.089E-6f,6.689E-6f,1.106E-5f,1.462E-5f,1.321E-5f,9.856E-6f,5.960E-6f,5.960E-6f};
__device__ const float lwp_ppsum[31] = {955.890f,850.532f,754.599f,667.742f,589.841f,519.421f,455.480f,398.085f,347.171f,301.735f,261.310f,225.360f,193.419f,165.490f,141.032f,120.125f,102.689f,87.829f,75.123f,64.306f,55.086f,47.209f,40.535f,34.795f,29.865f,19.122f,9.277f,4.660f,2.421f,1.294f,0.647f};
__device__ const float lwp_o3win[31] = {4.629E-8f,4.686E-8f,5.017E-8f,5.613E-8f,6.871E-8f,8.751E-8f,1.138E-7f,1.516E-7f,2.161E-7f,3.264E-7f,4.968E-7f,7.338E-7f,1.017E-6f,1.308E-6f,1.625E-6f,2.011E-6f,2.516E-6f,3.130E-6f,3.840E-6f,4.703E-6f,5.486E-6f,6.289E-6f,6.993E-6f,7.494E-6f,8.197E-6f,9.632E-6f,1.113E-5f,1.146E-5f,9.389E-6f,6.135E-6f,6.135E-6f};
__device__ const float lwp_ppwin[31] = {955.747f,841.783f,740.199f,649.538f,568.404f,495.815f,431.069f,373.464f,322.354f,277.190f,237.635f,203.433f,174.070f,148.949f,127.408f,108.915f,93.114f,79.551f,67.940f,58.072f,49.593f,42.318f,36.138f,30.907f,26.362f,16.423f,7.583f,3.620f,1.807f,0.938f,0.469f};

__device__ __forceinline__ bool lwp_fin(float x) { return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u; }
// c * d with an exact zero for a zero coefficient (an unused input's tangent never reaches the result).
__device__ __forceinline__ float lwp_lin(float c, float d) { return c == 0.0f ? 0.0f : LMU(c, d); }
__device__ __forceinline__ void lwp_acc(float *x, float c, float g) { if (c != 0.0f) *x = LAD(*x, LMU(c, g)); }
__device__ __forceinline__ void lwp_fail(unsigned int *fail, int slot, unsigned int id) { if (fail[slot] == 0u) atomicCAS(fail + slot, 0u, id); }

// profile (forward 9297-9306): value, and slope dv/dp of the executed table segment (0 for j == 59).
__device__ float lwp_profile(float p, float *slope) {
    int j = 59;
    if (lwp_pprof[59] < p) {
        for (int kk = 1; kk < 60; ++kk) if (lwp_pprof[kk] < p) { j = kk - 1; break; }
    }
    float weight = j == 59 ? 0.0f : LDV(LSU(p, lwp_pprof[j]), LSU(lwp_pprof[j + 1], lwp_pprof[j]));
    float next = lwp_tprof[j == 59 ? j : j + 1];
    *slope = j == 59 ? 0.0f : LDV(LSU(next, lwp_tprof[j]), LSU(lwp_pprof[j + 1], lwp_pprof[j]));
    return LAD(LMU(weight, LSU(next, lwp_tprof[j])), lwp_tprof[j]);
}
// ice radius (forward 9308-9313): j and the integer part of t are frozen, d corr / dt = 1.
__device__ float lwp_ice_radius(float t, int cap, float *slope) {
    int j = min(max((int)LSU(t, 179.0f), 1), cap) - 1;
    float corr = LSU(t, (float)(int)t);
    *slope = LSU(lwp_retab[j + 1], lwp_retab[j]);
    return LAD(LMU(lwp_retab[j], LSU(1.0f, corr)), LMU(lwp_retab[j + 1], corr));
}
// clamp01 = fminf(1, fmaxf(0, x)); passes the derivative only strictly inside (0, 1).
__device__ __forceinline__ float lwp_clamp01(float x, bool *pass) { *pass = x > 0.0f && x < 1.0f; return fminf(1.0f, fmaxf(0.0f, x)); }
// liquid radius (forward 9315-9321) with its partials in t, land, snow, ice.
__device__ float lwp_liquid_radius(float t, float land, float snow, float ice, float *dt, float *dland, float *dsnow, float *dice) {
    bool p1, p2, p3, p4;
    float c1 = lwp_clamp01(LMU(LSU(273.16f, t), 0.05f), &p1);
    float r1 = LAD(8.0f, LMU(6.0f, c1));
    float c2 = lwp_clamp01(LMU(LMU(0.001f, snow), 10.0f), &p2);
    float r2 = LAD(r1, LMU(LSU(14.0f, r1), c2));
    float c3 = lwp_clamp01(LSU(1.0f, LSU(2.0f, land)), &p3);
    float r3 = LAD(r2, LMU(LSU(14.0f, r2), c3));
    float c4 = lwp_clamp01(ice, &p4);
    float r4 = LAD(r3, LMU(LSU(14.0f, r3), c4));
    float g3 = LSU(1.0f, c4);            // d r4 / d r3
    float g2 = LMU(g3, LSU(1.0f, c3));   // d r4 / d r2
    float g1 = LMU(g2, LSU(1.0f, c2));   // d r4 / d r1
    *dt = p1 ? LMU(g1, LMU(6.0f, -0.05f)) : 0.0f;
    *dsnow = p2 ? LMU(g2, LMU(LSU(14.0f, r1), LMU(0.001f, 10.0f))) : 0.0f;
    *dland = p3 ? LMU(g3, LSU(14.0f, r2)) : 0.0f;
    *dice = p4 ? LSU(14.0f, r3) : 0.0f;
    return r4;
}

// vinterp (forward 9322-9338, called top to bottom with the carried *upper): value and partials.
struct LwpVi { float v, c_p; int pa, pb, oa, ob; float c_pa, c_pb, c_oa, c_ob; int kind; };
__device__ LwpVi lwp_vinterp(float p, const float *pin, const float *oz, int nz, int *upper) {
    for (int kk = *upper; kk < nz - 1; ++kk) {
        if (pin[kk] < p && p <= pin[kk + 1]) { *upper = kk; break; }
    }
    LwpVi r; r.pa = r.pb = r.oa = r.ob = -1; r.c_pa = r.c_pb = r.c_oa = r.c_ob = 0.0f;
    if (p < pin[0]) {
        r.kind = 0; r.v = LDV(LMU(oz[0], p), pin[0]);
        r.c_p = LDV(oz[0], pin[0]); r.oa = 0; r.c_oa = LDV(p, pin[0]); r.pa = 0; r.c_pa = -LDV(r.v, pin[0]);
        return r;
    }
    if (p > pin[nz - 1]) { r.kind = 1; r.v = oz[nz - 1]; r.c_p = 0.0f; r.oa = nz - 1; r.c_oa = 1.0f; return r; }
    const int u = *upper; const bool low = u + 1 < nz;
    float lower_pin = low ? pin[u + 1] : 0.0f, lower_ozone = low ? oz[u + 1] : 0.0f;
    float dpu = LSU(p, pin[u]), dpl = LSU(lower_pin, p), den = LAD(dpl, dpu);
    r.kind = 2; r.v = LDV(LAD(LMU(oz[u], dpl), LMU(lower_ozone, dpu)), den);
    r.c_p = LDV(LSU(lower_ozone, oz[u]), den);
    r.pa = u; r.c_pa = LDV(LSU(r.v, lower_ozone), den);
    r.oa = u; r.c_oa = LDV(dpl, den);
    if (low) { r.pb = u + 1; r.c_pb = LDV(LSU(oz[u], r.v), den); r.ob = u + 1; r.c_ob = LDV(dpu, den); }
    return r;
}

// Ozone climatology (forward 9358-9372): the 31-entry annual column and half levels (constants).
__device__ void lwp_o3_tables(float *annual, float *phalf) {
    phalf[0] = 1100.0f; phalf[31] = 0.0f;
    annual[0] = LMU(0.5f, LAD(lwp_o3sum[0], lwp_o3win[0]));
    for (int j = 1; j < 31; ++j) {
        float interp = LAD(lwp_o3win[j - 1], LMU(LDV(LSU(lwp_o3win[j], lwp_o3win[j - 1]), LSU(lwp_ppwin[j], lwp_ppwin[j - 1])), LSU(lwp_ppsum[j], lwp_ppwin[j - 1])));
        annual[j] = LMU(0.5f, LAD(interp, lwp_o3sum[j]));
        phalf[j] = LDV(LAD(lwp_ppsum[j], lwp_ppsum[j - 1]), 2.0f);
    }
}
// A4 of layer k from the climatology, with partials in P(k) and P(k+1); *clamped counts the active max(.,0) masks.
__device__ float lwp_o3_clim(float pk, float pk1, const float *annual, const float *phalf, float *cpk, float *cpk1, int *clamped) {
    float o3 = 0.0f, sk = 0.0f, sk1 = 0.0f;
    for (int j = 0; j < 31; ++j) {
        float rb1 = LSU(pk, phalf[j]), rb2 = LSU(pk, phalf[j + 1]), rt1 = LSU(pk1, phalf[j]), rt2 = LSU(pk1, phalf[j + 1]);
        float pb1 = fmaxf(rb1, 0.0f), pb2 = fmaxf(rb2, 0.0f), pt1 = fmaxf(rt1, 0.0f), pt2 = fmaxf(rt2, 0.0f);
        o3 = LAD(o3, LMU(LAD(LSU(LSU(pb2, pb1), pt2), pt1), annual[j]));
        int ik = int(rb2 > 0.0f) - int(rb1 > 0.0f), ik1 = int(rt1 > 0.0f) - int(rt2 > 0.0f);
        *clamped += int(!(rb1 > 0.0f)) + int(!(rb2 > 0.0f)) + int(!(rt1 > 0.0f)) + int(!(rt2 > 0.0f));
        if (ik) sk = LAD(sk, LMU((float)ik, annual[j]));
        if (ik1) sk1 = LAD(sk1, LMU((float)ik1, annual[j]));
    }
    float den = LSU(pk, pk1), q = LDV(o3, den);
    *cpk = LMU(LDV(LSU(sk, q), den), 0.603461f);
    *cpk1 = LMU(LDV(LAD(sk1, q), den), 0.603461f);
    return LMU(q, 0.603461f);
}

// Cloud block of layer k < nm (forward 9374-9396): values and partials.
struct LwpCl {
    float cf, liq, ice, snow, rel, rei, res;
    float cf_m7, rel_m8, rel_t, rel_land, rel_snow, rel_ice, rei_m9, rei_m1, rei_t, res_m10;
    float liq_q, liq_dp, liq_cf, ice_q5, ice_q6, ice_dp, ice_cf, snw_q6, snw_m10, snw_dp, snw_cf;
    int qfloor, dfloor, refloor, rescap;
};
#define LWM(f, k) m[(f) * nm + (k)]
__device__ LwpCl lwp_cloud(int k, int nm, int icloud, int reqc, int reqi, int reqs, int presence, const float *m,
                           const float *surface, float a1k, float pk, float pk1) {
    LwpCl r = {};
    const bool cfon = icloud > 0 && (presence & 7) && (presence & 8);
    r.cf = cfon ? LWM(7, k) : 0.0f; r.cf_m7 = cfon ? 1.0f : 0.0f;
    float rel = 10.0f, rei = 10.0f, res = 10.0f;
    if (reqc) {
        float v = LMU(LWM(8, k), 1.e6f); rel = fmaxf(2.5f, v);
        if (v > 2.5f) r.rel_m8 = 1.e6f; else r.refloor++;
        if (rel <= 2.5f && r.cf > 0.0f) { if (surface[2] > 1.5f) rel = 10.5f; else if (surface[2] < 1.5f) rel = 7.5f; }
    } else rel = lwp_liquid_radius(a1k, surface[2], surface[4], surface[3], &r.rel_t, &r.rel_land, &r.rel_snow, &r.rel_ice);
    if (reqi) {
        float v = LMU(LWM(9, k), 1.e6f); rei = fmaxf(5.0f, v);
        if (v > 5.0f) r.rei_m9 = 1.e6f; else r.refloor++;
        if (rei <= 5.0f && r.cf > 0.0f) {
            float s; float x = lwp_ice_radius(LWM(1, k), 75, &s); rei = fmaxf(5.0f, x);
            r.rei_m1 = x > 5.0f ? s : 0.0f;
        }
    } else {
        float s; float x = lwp_ice_radius(a1k, 94, &s); float y = LMU(x, 1.0315f); rei = fminf(140.0f, y);
        r.rei_t = y < 140.0f ? LMU(s, 1.0315f) : 0.0f;
    }
    float res0 = res;
    if (reqs) {
        float v = LMU(LWM(10, k), 1.e6f); res = fmaxf(10.0f, v); res0 = res;
        if (v > 10.0f) r.res_m10 = 1.e6f; else r.refloor++;
    }
    float dp = LSU(pk, pk1), denom = fmaxf(0.01f, r.cf);
    const bool dpass = r.cf > 0.01f; r.dfloor = !dpass;
    const bool qcon = icloud > 0 && (presence & 1), qion = icloud > 0 && (presence & 2), qson = icloud > 0 && (presence & 4);
    float qc = qcon ? fmaxf(LWM(4, k), 0.0f) : 0.0f, qi = qion ? fmaxf(LWM(5, k), 0.0f) : 0.0f, qs = qson ? fmaxf(LWM(6, k), 0.0f) : 0.0f;
    const bool qcp = qcon && LWM(4, k) > 0.0f, qip = qion && LWM(5, k) > 0.0f, qsp = qson && LWM(6, k) > 0.0f;
    r.qfloor = int(qcon && !qcp) + int(qion && !qip) + int(qson && !qsp);
    // g(q) = (((q dp) 100) / 9.80616) 1000 is linear in q and in dp
    const float gq = LMU(LDV(LMU(dp, 100.0f), 9.80616f), 1000.0f);
    r.liq = LDV(LMU(LDV(LMU(LMU(qc, dp), 100.0f), 9.80616f), 1000.0f), denom);
    r.liq_q = qcp ? LDV(gq, denom) : 0.0f;
    r.liq_dp = LDV(LMU(LDV(LMU(qc, 100.0f), 9.80616f), 1000.0f), denom);
    r.liq_cf = dpass ? -LDV(r.liq, denom) : 0.0f;
    const bool sep = reqi || reqs;
    float qice = sep ? qi : LAD(qi, qs);
    r.ice = LDV(LMU(LDV(LMU(LMU(qice, dp), 100.0f), 9.80616f), 1000.0f), denom);
    r.ice_q5 = qip ? LDV(gq, denom) : 0.0f;
    r.ice_q6 = (!sep && qsp) ? LDV(gq, denom) : 0.0f;
    r.ice_dp = LDV(LMU(LDV(LMU(qice, 100.0f), 9.80616f), 1000.0f), denom);
    r.ice_cf = dpass ? -LDV(r.ice, denom) : 0.0f;
    if (reqs) {
        float factor = 1.0f, f_res = 0.0f;
        if (res > 130.0f) {
            float ratio = LDV(130.0f, res); factor = LMU(ratio, ratio); res = 130.0f; r.rescap = 1;
            f_res = LMU(-2.0f, LDV(factor, res0));    // d factor / d res0
            r.res_m10 = 0.0f;                         // res is the constant 130 now
        }
        float gs = LMU(LDV(LMU(LMU(qs, dp), 100.0f), 9.80616f), 1000.0f);
        r.snow = LDV(LMU(factor, gs), denom);
        r.snw_q6 = qsp ? LDV(LMU(factor, gq), denom) : 0.0f;
        r.snw_dp = LDV(LMU(factor, LMU(LDV(LMU(qs, 100.0f), 9.80616f), 1000.0f)), denom);
        r.snw_m10 = r.rescap ? LMU(LDV(gs, denom), LMU(f_res, 1.e6f)) : 0.0f;   // res0 > 130 implies the floor passed
        r.snw_cf = dpass ? -LDV(r.snow, denom) : 0.0f;
    }
    r.rel = rel; r.rei = rei; r.res = res;
    return r;
}
__device__ __forceinline__ bool lwp_cl_fin(const LwpCl &r) {
    const float v[] = {r.cf, r.liq, r.ice, r.snow, r.rel, r.rei, r.res, r.rel_m8, r.rel_t, r.rel_land, r.rel_snow, r.rel_ice,
                       r.rei_m9, r.rei_m1, r.rei_t, r.res_m10, r.liq_q, r.liq_dp, r.liq_cf, r.ice_q5, r.ice_q6, r.ice_dp,
                       r.ice_cf, r.snw_q6, r.snw_m10, r.snw_dp, r.snw_cf};
    bool ok = true;
    for (int i = 0; i < 27; ++i) ok = ok && lwp_fin(v[i]);
    return ok;
}

// Shape contract, and the primal levels shared by the three entries: the final P (S1-S3) into Pv.
__device__ __forceinline__ bool lwp_shape_ok(int nm, int nl, int o3input, int nz) {
    return nm >= 1 && nl >= nm && nl + 1 <= LWP_NLMAX && !(o3input == 2 && nz < 2);
}
__device__ void lwp_levels(int nm, int nl, const float *ifc, float *Pv) {
    for (int k = 0; k <= nm; ++k) Pv[k] = LDV(ifc[k], 100.0f);
    for (int k = nm; k < nl; ++k) Pv[k + 1] = fmaxf(LSU(Pv[k], 4.0f), 0.0f);
    Pv[nl] = 0.0f;
}
// Final A0[k] (S2 or S3) from the final P.
__device__ __forceinline__ float lwp_a0(int k, int nm, int nl, const float *m, const float *Pv) {
    return (k < nm && k != nl - 1) ? LDV(LWM(0, k), 100.0f) : LMU(0.5f, LAD(Pv[k], Pv[k + 1]));
}
// Final A1[k] for k < nm (A1[nm-1] is rewritten by S5 from T(nm) and T(nm-1)).
__device__ __forceinline__ float lwp_a1(int k, int nm, const float *m, const float *ifc, float tnm) {
    return k < nm - 1 ? LWM(1, k) : LMU(0.5f, LAD(tnm, ifc[nm + 1 + nm - 1]));
}

#define LWPD_COLUMN(nm, nl) \
    const float *m = m_ + (size_t)c * 12 * nm, *ifc = ifc_ + (size_t)c * 2 * (nm + 1), *sf = sf_ + (size_t)c * 6; \
    const float *oz = oz_ + (o3input == 2 ? (size_t)c * nz : 0);

// ---------------------------------------------------------------------------------------------------------------
// Diagnostic primal entry (not bound in the window): writes the recomputed primal outputs, so the harness can check
// them bitwise against the forward's post words. Same helpers as TL and AD.
extern "C" __global__ void mpas_lw_prepare__pri(int nc, int nm, int nl, int icloud, int reqc, int reqi, int reqs, int o3input,
    int nz, int presence, const float *m_, const float *ifc_, const float *sf_, const float *pin, const float *oz_,
    float *a_, float *lev_, unsigned int *fail) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    if (!lwp_shape_ok(nm, nl, o3input, nz)) { lwp_fail(fail, 0, 5101u); return; }
    LWPD_COLUMN(nm, nl)
    float *a = a_ + (size_t)c * 12 * nl, *lev = lev_ + (size_t)c * 2 * (nl + 1);
    float Pv[LWP_NLMAX]; lwp_levels(nm, nl, ifc, Pv);
    float s, dzsum = 0.0f;
    for (int k = 0; k < nl; ++k) {
        int kk = min(k, nm - 1); float dz = LWM(2, kk);
        a[2 * nl + k] = LAD(dzsum, LMU(0.5f, dz)); dzsum = LAD(dzsum, dz);
        a[3 * nl + k] = LMU(fmaxf(LWM(3, kk), 1.e-12f), 1.607793f);
        a[0 * nl + k] = lwp_a0(k, nm, nl, m, Pv);
        if (k < nm) a[1 * nl + k] = LWM(1, k);
    }
    float anchor = LSU(ifc[nm + 1 + nm - 1], lwp_profile(Pv[nm - 1], &s));
    for (int k = 0; k < nm; ++k) lev[nl + 1 + k] = ifc[nm + 1 + k];
    for (int k = nm; k <= nl; ++k) {
        lev[nl + 1 + k] = LAD(lwp_profile(Pv[k], &s), anchor);
        a[1 * nl + k - 1] = LMU(0.5f, LAD(lev[nl + 1 + k], lev[nl + 1 + k - 1]));
    }
    for (int k = 0; k <= nl; ++k) lev[k] = Pv[k];
    if (o3input == 2) { int upper = 0; for (int k = nl - 1; k >= 0; --k) a[4 * nl + k] = lwp_vinterp(a[k], pin, oz, nz, &upper).v; }
    else {
        float annual[31], phalf[32]; lwp_o3_tables(annual, phalf); int cl = 0; float x, y;
        for (int k = 0; k < nl; ++k) a[4 * nl + k] = lwp_o3_clim(Pv[k], Pv[k + 1], annual, phalf, &x, &y, &cl);
    }
    float tnm = lev[nl + 1 + nm];
    for (int k = 0; k < nl; ++k) {
        if (k < nm) {
            LwpCl r = lwp_cloud(k, nm, icloud, reqc, reqi, reqs, presence, m, sf, lwp_a1(k, nm, m, ifc, tnm), Pv[k], Pv[k + 1]);
            a[5 * nl + k] = r.cf; a[6 * nl + k] = r.liq; a[7 * nl + k] = r.ice; a[8 * nl + k] = r.snow;
            a[9 * nl + k] = r.rel; a[10 * nl + k] = r.rei; a[11 * nl + k] = r.res;
        } else {
            a[5 * nl + k] = 0.0f; a[6 * nl + k] = 0.0f; a[7 * nl + k] = 0.0f; a[8 * nl + k] = 0.0f;
            a[9 * nl + k] = 10.0f; a[10 * nl + k] = 10.0f; a[11 * nl + k] = 10.0f;
        }
    }
}

// Overlap of the seven derivative views (element counts per the launch scalars).
__device__ bool lwp_views_disjoint(int nc, int nm, int nl, int o3input, int nz, const float *x0, const float *x1, const float *x2,
                                   const float *x3, const float *x4, const float *y0, const float *y1) {
    const unsigned long long u[7] = {(unsigned long long)x0, (unsigned long long)x1, (unsigned long long)x2, (unsigned long long)x3,
                                     (unsigned long long)x4, (unsigned long long)y0, (unsigned long long)y1};
    const unsigned long long n[7] = {4ull * nc * 12 * nm, 4ull * nc * 2 * (nm + 1), 4ull * nc * 6, 4ull * nz,
                                     o3input == 2 ? 4ull * nc * nz : 0ull, 4ull * nc * 12 * nl, 4ull * nc * 2 * (nl + 1)};
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (u[i] && u[j] && n[i] && n[j] && u[i] < u[j] + n[j] && u[j] < u[i] + n[i]) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
extern "C" __global__ void mpas_lw_prepare__tl(int nc, int nm, int nl, int icloud, int reqc, int reqi, int reqs, int o3input,
    int nz, int presence, const float *m_, const float *ifc_, const float *sf_, const float *pin, const float *oz_,
    const float *dm_, const float *difc_, const float *dsf_, const float *dpin, const float *doz_, float *da_, float *dlev_,
    unsigned int *fail, unsigned int *branches) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    if (!lwp_shape_ok(nm, nl, o3input, nz)) { lwp_fail(fail, 0, 5101u); return; }
    if (c == 0 && !lwp_views_disjoint(nc, nm, nl, o3input, nz, dm_, difc_, dsf_, dpin, doz_, da_, dlev_)) lwp_fail(fail, 1, 5102u);
    LWPD_COLUMN(nm, nl)
    const float *dm = dm_ + (size_t)c * 12 * nm, *difc = difc_ + (size_t)c * 2 * (nm + 1), *dsf = dsf_ + (size_t)c * 6;
    const float *doz = o3input == 2 ? doz_ + (size_t)c * nz : nullptr;
    float *da = da_ + (size_t)c * 12 * nl, *dlev = dlev_ + (size_t)c * 2 * (nl + 1);
    bool bad = false, badin = false;
    for (int i = 0; i < 12 * nm; ++i) badin |= !lwp_fin(dm[i]);
    for (int i = 0; i < 2 * (nm + 1); ++i) badin |= !lwp_fin(difc[i]);
    for (int i = 0; i < 6; ++i) badin |= !lwp_fin(dsf[i]);
    if (o3input == 2) { for (int i = 0; i < nz; ++i) badin |= !lwp_fin(doz[i]) || !lwp_fin(dpin[i]); }
#define dP(k) dlev[(k)]
#define dT(k) dlev[nl + 1 + (k)]
#define dA(f, k) da[(f) * nl + (k)]
#define dM(f, k) dm[(f) * nm + (k)]
    float Pv[LWP_NLMAX]; lwp_levels(nm, nl, ifc, Pv);
    for (int k = 0; k <= nm; ++k) bad |= !lwp_fin(Pv[k]);
    // S1
    for (int k = 0; k <= nm; ++k) { dP(k) = LDV(difc[k], 100.0f); dT(k) = difc[nm + 1 + k]; }
    // S2
    float ddzsum = 0.0f; int nclampP = 0, nfloor = 0;
    for (int k = 0; k < nl; ++k) {
        int kk = min(k, nm - 1); float ddz = dM(2, kk);
        dA(2, k) = LAD(ddzsum, LMU(0.5f, ddz)); ddzsum = LAD(ddzsum, ddz);
        const bool qvp = LWM(3, kk) > 1.e-12f; nfloor += !qvp;
        dA(3, k) = qvp ? LMU(dM(3, kk), 1.607793f) : 0.0f;
        if (k < nm) { dA(0, k) = LDV(dM(0, k), 100.0f); dA(1, k) = dM(1, k); }
        else {
            const bool pp = LSU(Pv[k], 4.0f) > 0.0f; nclampP += !pp;
            dP(k + 1) = pp ? dP(k) : 0.0f;
            dA(0, k) = LMU(0.5f, LAD(dP(k), dP(k + 1)));
        }
    }
    // S3
    dP(nl) = 0.0f; dA(0, nl - 1) = LMU(0.5f, LAD(dP(nl - 1), dP(nl)));
    // S4, S5
    float s; float anchor = LSU(ifc[nm + 1 + nm - 1], lwp_profile(Pv[nm - 1], &s));
    float danchor = LSU(dT(nm - 1), lwp_lin(s, dP(nm - 1)));
    float tnm = 0.0f;
    for (int k = nm; k <= nl; ++k) {
        float t = LAD(lwp_profile(Pv[k], &s), anchor); if (k == nm) tnm = t;
        bad |= !lwp_fin(t) || !lwp_fin(s);
        dT(k) = LAD(lwp_lin(s, dP(k)), danchor);
        dA(1, k - 1) = LMU(0.5f, LAD(dT(k), dT(k - 1)));
    }
    // S6 / S7 (S7 overwrites every S6 word when o3input == 2, so S6 is skipped there)
    int ntop = 0, nbot = 0, nclim = 0;
    if (o3input == 2) {
        int upper = 0;
        for (int k = nl - 1; k >= 0; --k) {
            LwpVi v = lwp_vinterp(lwp_a0(k, nm, nl, m, Pv), pin, oz, nz, &upper);
            ntop += v.kind == 0; nbot += v.kind == 1;
            bad |= !lwp_fin(v.v) || !lwp_fin(v.c_p) || !lwp_fin(v.c_pa) || !lwp_fin(v.c_pb) || !lwp_fin(v.c_oa) || !lwp_fin(v.c_ob);
            float t = lwp_lin(v.c_p, dA(0, k));
            if (v.pa >= 0) t = LAD(t, lwp_lin(v.c_pa, dpin[v.pa]));
            if (v.pb >= 0) t = LAD(t, lwp_lin(v.c_pb, dpin[v.pb]));
            if (v.oa >= 0) t = LAD(t, lwp_lin(v.c_oa, doz[v.oa]));
            if (v.ob >= 0) t = LAD(t, lwp_lin(v.c_ob, doz[v.ob]));
            dA(4, k) = t;
        }
    } else {
        float annual[31], phalf[32]; lwp_o3_tables(annual, phalf);
        for (int k = 0; k < nl; ++k) {
            float cpk, cpk1; float v = lwp_o3_clim(Pv[k], Pv[k + 1], annual, phalf, &cpk, &cpk1, &nclim);
            bad |= !lwp_fin(v) || !lwp_fin(cpk) || !lwp_fin(cpk1);
            dA(4, k) = LAD(lwp_lin(cpk, dP(k)), lwp_lin(cpk1, dP(k + 1)));
        }
    }
    // S8
    int nq = 0, nd = 0, nre = 0, nres = 0;
    for (int k = 0; k < nl; ++k) {
        if (k < nm) {
            LwpCl r = lwp_cloud(k, nm, icloud, reqc, reqi, reqs, presence, m, sf, lwp_a1(k, nm, m, ifc, tnm), Pv[k], Pv[k + 1]);
            bad |= !lwp_cl_fin(r);
            nq += r.qfloor; nd += r.dfloor; nre += r.refloor; nres += r.rescap;
            const float dcf = lwp_lin(r.cf_m7, dM(7, k)), da1 = dA(1, k), ddp = LSU(dP(k), dP(k + 1));
            dA(5, k) = dcf;
            dA(6, k) = LAD(LAD(lwp_lin(r.liq_q, dM(4, k)), lwp_lin(r.liq_dp, ddp)), lwp_lin(r.liq_cf, dcf));
            dA(7, k) = LAD(LAD(LAD(lwp_lin(r.ice_q5, dM(5, k)), lwp_lin(r.ice_q6, dM(6, k))), lwp_lin(r.ice_dp, ddp)), lwp_lin(r.ice_cf, dcf));
            dA(8, k) = LAD(LAD(LAD(lwp_lin(r.snw_q6, dM(6, k)), lwp_lin(r.snw_m10, dM(10, k))), lwp_lin(r.snw_dp, ddp)), lwp_lin(r.snw_cf, dcf));
            dA(9, k) = LAD(LAD(LAD(LAD(lwp_lin(r.rel_m8, dM(8, k)), lwp_lin(r.rel_t, da1)), lwp_lin(r.rel_land, dsf[2])),
                               lwp_lin(r.rel_snow, dsf[4])), lwp_lin(r.rel_ice, dsf[3]));
            dA(10, k) = LAD(LAD(lwp_lin(r.rei_m9, dM(9, k)), lwp_lin(r.rei_m1, dM(1, k))), lwp_lin(r.rei_t, da1));
            dA(11, k) = lwp_lin(r.res_m10, dM(10, k));
        } else {
            for (int f = 5; f < 12; ++f) dA(f, k) = 0.0f;
        }
    }
    for (int i = 0; i < 12 * nl; ++i) bad |= !lwp_fin(da[i]);
    for (int i = 0; i < 2 * (nl + 1); ++i) bad |= !lwp_fin(dlev[i]);
    if (badin) lwp_fail(fail, 3, 5104u);
    if (bad) lwp_fail(fail, 2, 5103u);
    atomicAdd(branches + 2 * 0 + 1, (unsigned)nclampP); atomicAdd(branches + 2 * 0 + 0, (unsigned)(nl - nm - nclampP));
    atomicAdd(branches + 2 * 1 + 1, (unsigned)nfloor); atomicAdd(branches + 2 * 1 + 0, (unsigned)(nl - nfloor));
    if (o3input == 2) {
        atomicAdd(branches + 2 * 2 + 1, (unsigned)ntop); atomicAdd(branches + 2 * 2 + 0, (unsigned)(nl - ntop));
        atomicAdd(branches + 2 * 3 + 1, (unsigned)nbot); atomicAdd(branches + 2 * 3 + 0, (unsigned)(nl - nbot));
    } else { atomicAdd(branches + 2 * 8 + 1, (unsigned)nclim); atomicAdd(branches + 2 * 8 + 0, (unsigned)(nl * 124 - nclim)); }
    atomicAdd(branches + 2 * 4 + 1, (unsigned)nq); atomicAdd(branches + 2 * 4 + 0, (unsigned)(3 * nm - nq));
    atomicAdd(branches + 2 * 5 + 1, (unsigned)nd); atomicAdd(branches + 2 * 5 + 0, (unsigned)(nm - nd));
    atomicAdd(branches + 2 * 6 + 1, (unsigned)nre); atomicAdd(branches + 2 * 6 + 0, (unsigned)(3 * nm - nre));
    atomicAdd(branches + 2 * 7 + 1, (unsigned)nres); atomicAdd(branches + 2 * 7 + 0, (unsigned)(nm - nres));
#undef dP
#undef dT
#undef dA
#undef dM
}

// ---------------------------------------------------------------------------------------------------------------
// Transpose. Output cotangents (a, lev) are consumed and cleared in the exact reverse statement order (R8..R1), in
// place on this column's slices; input cotangents accumulate (+=). pin contributions go to this column's workspace
// row and the last block adds the column-ordered sum to the pin cotangent.
__device__ void lwp_ad_column(int c, int nm, int nl, int icloud, int reqc, int reqi, int reqs, int o3input,
    int nz, int presence, const float *m_, const float *ifc_, const float *sf_, const float *pin, const float *oz_,
    float *bm_, float *bifc_, float *bsf_, float *boz_, float *ba_, float *blev_, float *wrow, unsigned int *fail) {
    LWPD_COLUMN(nm, nl)
    float *bm = bm_ + (size_t)c * 12 * nm, *bifc = bifc_ + (size_t)c * 2 * (nm + 1), *bsf = bsf_ + (size_t)c * 6;
    float *boz = o3input == 2 ? boz_ + (size_t)c * nz : nullptr;
    float *ba = ba_ + (size_t)c * 12 * nl, *blev = blev_ + (size_t)c * 2 * (nl + 1);
#define bP(k) blev[(k)]
#define bT(k) blev[nl + 1 + (k)]
#define bA(f, k) ba[(f) * nl + (k)]
#define bM(f, k) bm[(f) * nm + (k)]
    bool bad = false, badin = false;
    for (int i = 0; i < 12 * nl; ++i) badin |= !lwp_fin(ba[i]);
    for (int i = 0; i < 2 * (nl + 1); ++i) badin |= !lwp_fin(blev[i]);
    float Pv[LWP_NLMAX]; lwp_levels(nm, nl, ifc, Pv);
    for (int k = 0; k <= nm; ++k) bad |= !lwp_fin(Pv[k]);
    float s; float anchor = LSU(ifc[nm + 1 + nm - 1], lwp_profile(Pv[nm - 1], &s));
    const float s_nm1 = s;
    float tnm = LAD(lwp_profile(Pv[nm], &s), anchor);
    // R8 (S8)
    for (int k = nl - 1; k >= 0; --k) {
        float g[12];
        for (int f = 5; f < 12; ++f) { g[f] = bA(f, k); bA(f, k) = 0.0f; }
        if (k >= nm) continue;    // constants above the model top
        LwpCl r = lwp_cloud(k, nm, icloud, reqc, reqi, reqs, presence, m, sf, lwp_a1(k, nm, m, ifc, tnm), Pv[k], Pv[k + 1]);
        bad |= !lwp_cl_fin(r);
        float bcf = g[5], bdp = 0.0f;
        lwp_acc(&bM(4, k), r.liq_q, g[6]); lwp_acc(&bdp, r.liq_dp, g[6]); lwp_acc(&bcf, r.liq_cf, g[6]);
        lwp_acc(&bM(5, k), r.ice_q5, g[7]); lwp_acc(&bM(6, k), r.ice_q6, g[7]); lwp_acc(&bdp, r.ice_dp, g[7]); lwp_acc(&bcf, r.ice_cf, g[7]);
        lwp_acc(&bM(6, k), r.snw_q6, g[8]); lwp_acc(&bM(10, k), r.snw_m10, g[8]); lwp_acc(&bdp, r.snw_dp, g[8]); lwp_acc(&bcf, r.snw_cf, g[8]);
        lwp_acc(&bM(8, k), r.rel_m8, g[9]); lwp_acc(&bA(1, k), r.rel_t, g[9]); lwp_acc(&bsf[2], r.rel_land, g[9]);
        lwp_acc(&bsf[4], r.rel_snow, g[9]); lwp_acc(&bsf[3], r.rel_ice, g[9]);
        lwp_acc(&bM(9, k), r.rei_m9, g[10]); lwp_acc(&bM(1, k), r.rei_m1, g[10]); lwp_acc(&bA(1, k), r.rei_t, g[10]);
        lwp_acc(&bM(10, k), r.res_m10, g[11]);
        lwp_acc(&bM(7, k), r.cf_m7, bcf);
        bP(k) = LAD(bP(k), bdp); bP(k + 1) = LSU(bP(k + 1), bdp);
    }
    if (o3input == 2) {
        // R7 (S7; forward order: the levels' contributions are independent given the frozen search indices)
        for (int i = 0; i < nz; ++i) wrow[i] = 0.0f;
        int upper = 0;
        for (int k = nl - 1; k >= 0; --k) {
            LwpVi v = lwp_vinterp(lwp_a0(k, nm, nl, m, Pv), pin, oz, nz, &upper);
            bad |= !lwp_fin(v.v) || !lwp_fin(v.c_p) || !lwp_fin(v.c_pa) || !lwp_fin(v.c_pb) || !lwp_fin(v.c_oa) || !lwp_fin(v.c_ob);
            float g = bA(4, k); bA(4, k) = 0.0f;
            lwp_acc(&bA(0, k), v.c_p, g);
            if (v.pa >= 0) lwp_acc(&wrow[v.pa], v.c_pa, g);
            if (v.pb >= 0) lwp_acc(&wrow[v.pb], v.c_pb, g);
            if (v.oa >= 0) lwp_acc(&boz[v.oa], v.c_oa, g);
            if (v.ob >= 0) lwp_acc(&boz[v.ob], v.c_ob, g);
        }
    } else {
        // R6 (S6)
        float annual[31], phalf[32]; lwp_o3_tables(annual, phalf); int ncl = 0;
        for (int k = nl - 1; k >= 0; --k) {
            float cpk, cpk1; float v = lwp_o3_clim(Pv[k], Pv[k + 1], annual, phalf, &cpk, &cpk1, &ncl);
            bad |= !lwp_fin(v) || !lwp_fin(cpk) || !lwp_fin(cpk1);
            float g = bA(4, k); bA(4, k) = 0.0f;
            lwp_acc(&bP(k), cpk, g); lwp_acc(&bP(k + 1), cpk1, g);
        }
    }
    // R5 (S5), k = nl .. nm
    float banchor = 0.0f;
    for (int k = nl; k >= nm; --k) {
        float g = bA(1, k - 1); bA(1, k - 1) = 0.0f;
        bT(k) = LAD(bT(k), LMU(0.5f, g)); bT(k - 1) = LAD(bT(k - 1), LMU(0.5f, g));
        lwp_profile(Pv[k], &s); bad |= !lwp_fin(s);
        g = bT(k); bT(k) = 0.0f;
        lwp_acc(&bP(k), s, g); banchor = LAD(banchor, g);
    }
    // R4 (S4)
    bT(nm - 1) = LAD(bT(nm - 1), banchor); lwp_acc(&bP(nm - 1), -s_nm1, banchor);
    // R3 (S3)
    {
        float g = bA(0, nl - 1); bA(0, nl - 1) = 0.0f;
        bP(nl - 1) = LAD(bP(nl - 1), LMU(0.5f, g)); bP(nl) = LAD(bP(nl), LMU(0.5f, g));
        bP(nl) = 0.0f;                                        // P(nl) = 0: nothing upstream
    }
    // R2 (S2), k = nl-1 .. 0
    float bdzsum = 0.0f;
    for (int k = nl - 1; k >= 0; --k) {
        int kk = min(k, nm - 1);
        if (k >= nm) {
            float g = bA(0, k); bA(0, k) = 0.0f;
            bP(k) = LAD(bP(k), LMU(0.5f, g)); bP(k + 1) = LAD(bP(k + 1), LMU(0.5f, g));
            g = bP(k + 1); bP(k + 1) = 0.0f;
            if (LSU(Pv[k], 4.0f) > 0.0f) bP(k) = LAD(bP(k), g);
        } else {
            float g = bA(1, k); bA(1, k) = 0.0f; bM(1, k) = LAD(bM(1, k), g);
            g = bA(0, k); bA(0, k) = 0.0f; bM(0, k) = LAD(bM(0, k), LDV(g, 100.0f));
        }
        { float g = bA(3, k); bA(3, k) = 0.0f; if (LWM(3, kk) > 1.e-12f) bM(3, kk) = LAD(bM(3, kk), LMU(g, 1.607793f)); }
        float bdz = bdzsum;                                   // dzsum_new = dzsum_old + dz
        float g = bA(2, k); bA(2, k) = 0.0f;                  // A2 = dzsum_old + dz/2
        bdzsum = LAD(bdzsum, g); bdz = LAD(bdz, LMU(0.5f, g));
        bM(2, kk) = LAD(bM(2, kk), bdz);
    }
    // R1 (S1)
    for (int k = 0; k <= nm; ++k) {
        float g = bP(k); bP(k) = 0.0f; bifc[k] = LAD(bifc[k], LDV(g, 100.0f));
        g = bT(k); bT(k) = 0.0f; bifc[nm + 1 + k] = LAD(bifc[nm + 1 + k], g);
    }
    for (int i = 0; i < 12 * nm; ++i) bad |= !lwp_fin(bm[i]);
    for (int i = 0; i < 2 * (nm + 1); ++i) bad |= !lwp_fin(bifc[i]);
    if (badin) lwp_fail(fail, 3, 5104u);
    if (bad) lwp_fail(fail, 2, 5103u);
#undef bP
#undef bT
#undef bA
#undef bM
}

extern "C" __global__ void mpas_lw_prepare__ad(int nc, int nm, int nl, int icloud, int reqc, int reqi, int reqs, int o3input,
    int nz, int presence, const float *m_, const float *ifc_, const float *sf_, const float *pin, const float *oz_,
    float *bm_, float *bifc_, float *bsf_, float *bpin, float *boz_, float *ba_, float *blev_, float *work,
    unsigned int *fail) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (!lwp_shape_ok(nm, nl, o3input, nz)) { if (c == 0) lwp_fail(fail, 0, 5101u); return; }   // uniform
    if (c == 0 && !lwp_views_disjoint(nc, nm, nl, o3input, nz, bm_, bifc_, bsf_, bpin, boz_, ba_, blev_)) lwp_fail(fail, 1, 5102u);
    if (c < nc) lwp_ad_column(c, nm, nl, icloud, reqc, reqi, reqs, o3input, nz, presence, m_, ifc_, sf_, pin, oz_,
                              bm_, bifc_, bsf_, boz_, ba_, blev_, work + (size_t)c * nz, fail);
    if (o3input != 2) return;                                                                        // uniform
    // Deterministic cross-column pin sum: rows 0..nc-1 in order, by the last block to finish.
    unsigned int *ticket = reinterpret_cast<unsigned int *>(work + (size_t)nc * nz);
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(ticket, 1u) == gridDim.x * gridDim.y * gridDim.z - 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < nz; i += blockDim.x) {
        float sum = 0.0f;
        for (int cc = 0; cc < nc; ++cc) sum = LAD(sum, __ldcg(work + (size_t)cc * nz + i));
        bpin[i] = LAD(bpin[i], sum);
    }
    if (threadIdx.x == 0) *ticket = 0u;                                                              // reusable workspace
}
