// ---- p7 SW entries for rsw_taumol_b (generated section; Apache-2.0; ad/held-rsw_taumol_b) ----
// Active inputs, slot order: colh2o colco2 colch4 colo2 colo3 colmol fac00 fac01 fac10 fac11 selffac selffrac forfac forfrac.
#define P7SW_FWD_PARAMS \
    int ncol, int nlayers, const int* __restrict__ laytrop, const float* __restrict__ tab, const int* __restrict__ ngb, \
    const int* __restrict__ jp, const int* __restrict__ jt, const int* __restrict__ jt1, \
    const float* __restrict__ colh2o, const float* __restrict__ colco2, const float* __restrict__ colch4, \
    const float* __restrict__ colo2, const float* __restrict__ colo3, const float* __restrict__ colmol, \
    const float* __restrict__ fac00, const float* __restrict__ fac01, const float* __restrict__ fac10, \
    const float* __restrict__ fac11, const float* __restrict__ selffac, const float* __restrict__ selffrac, \
    const int* __restrict__ indself, const float* __restrict__ forfac, const float* __restrict__ forfrac, \
    const int* __restrict__ indfor
#define P7SW_D_PARAMS(P, Q) \
    Q double* __restrict__ P##colh2o, Q double* __restrict__ P##colco2, Q double* __restrict__ P##colch4, \
    Q double* __restrict__ P##colo2, Q double* __restrict__ P##colo3, Q double* __restrict__ P##colmol, \
    Q double* __restrict__ P##fac00, Q double* __restrict__ P##fac01, Q double* __restrict__ P##fac10, \
    Q double* __restrict__ P##fac11, Q double* __restrict__ P##selffac, Q double* __restrict__ P##selffrac, \
    Q double* __restrict__ P##forfac, Q double* __restrict__ P##forfrac

// Tangent (augmented forward): the forward's geometry and indexing (rsw_taumol_b: thread -> (col, L, iw)). It writes the
// forward's taug/taur into the given (private) buffers and their tangents into dtaug/dtaur; input tangents are read only.
extern "C" __global__ void rsw_taumol_b__tl(P7SW_FWD_PARAMS, float* __restrict__ taug, float* __restrict__ taur,
                                            P7SW_D_PARAMS(d, const), double* __restrict__ dtaug, double* __restrict__ dtaur)
{
    long long W = (long long)nlayers * RSW_NGPT;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int r = (int)(idx % W);
    int L = r / RSW_NGPT;
    int iw = r % RSW_NGPT;
    size_t o = (size_t)col * nlayers;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    p7swtl::rsw_taumol_body(L, iw, nlayers, laytrop[col], tab, ngb, jp + o, jt + o, jt1 + o,
        TlIn{colh2o + o, dcolh2o + o}, TlIn{colco2 + o, dcolco2 + o}, TlIn{colch4 + o, dcolch4 + o},
        TlIn{colo2 + o, dcolo2 + o}, TlIn{colo3 + o, dcolo3 + o}, TlIn{colmol + o, dcolmol + o},
        TlIn{fac00 + o, dfac00 + o}, TlIn{fac01 + o, dfac01 + o}, TlIn{fac10 + o, dfac10 + o}, TlIn{fac11 + o, dfac11 + o},
        TlIn{selffac + o, dselffac + o}, TlIn{selffrac + o, dselffrac + o}, indself + o,
        TlIn{forfac + o, dforfac + o}, TlIn{forfrac + o, dforfrac + o}, indfor + o,
        TlOut{taug + og, dtaug + og}, TlOut{taur + og, dtaur + og}, true);
}

// Adjoint, first form (kept as rsw_taumol_b__ad_v1 for the bit-equality proof of the compact form below): one thread
// per (column, layer) runs the 112 g-points in ascending order (deterministic, no atomics on
// derivative words). Per g-point: a fresh tape, seeded from btaug/btaur of that word, swept in reverse into 14 float64
// accumulators. Then the accumulators are ADDED to the input cotangents (one rounding each), and the overwritten
// outputs' cotangents btaug/btaur of the layer are consumed (set to +0.0). Grid ceil(ncol*nlayers/64) x 64.
// fail: 1 tape overflow (P7SW_TMAX), 4 an input read off the thread's layer, 5 an output written twice, 6 an output
// word that is not (L, iw) or not written. None is a pattern of the pinned body.
extern "C" __global__ void rsw_taumol_b__ad_v1(P7SW_FWD_PARAMS, P7SW_D_PARAMS(b, ), double* __restrict__ btaug,
                                            double* __restrict__ btaur, int* __restrict__ fail)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * nlayers) return;
    int col = (int)(tid / nlayers);
    int L = (int)(tid % nlayers);
    size_t o = (size_t)col * nlayers;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    P7SwTape tape; tape.n = 0; tape.fail = 0;
    AdCtx ctx; ctx.t = &tape; ctx.L = L; ctx.nout = 0;
    double adj[P7SW_TMAX];
    double acc[P7SW_NIN];
    int high = 0;
    for (int s = 0; s < P7SW_NIN; ++s) acc[s] = 0.0;
    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        tape.n = 0; ctx.nout = 0;
        p7swad::rsw_taumol_body(L, iw, nlayers, laytrop[col], tab, ngb, jp + o, jt + o, jt1 + o,
            AdIn{colh2o + o, 0, &ctx}, AdIn{colco2 + o, 1, &ctx}, AdIn{colch4 + o, 2, &ctx},
            AdIn{colo2 + o, 3, &ctx}, AdIn{colo3 + o, 4, &ctx}, AdIn{colmol + o, 5, &ctx},
            AdIn{fac00 + o, 6, &ctx}, AdIn{fac01 + o, 7, &ctx}, AdIn{fac10 + o, 8, &ctx}, AdIn{fac11 + o, 9, &ctx},
            AdIn{selffac + o, 10, &ctx}, AdIn{selffrac + o, 11, &ctx}, indself + o,
            AdIn{forfac + o, 12, &ctx}, AdIn{forfrac + o, 13, &ctx}, indfor + o,
            AdOut{0, &ctx}, AdOut{1, &ctx}, true);
        if (tape.n > high) high = tape.n;
        long long w = (long long)L * RSW_NGPT + iw;
        if (ctx.nout != 3 || ctx.oi[0] != w || ctx.oi[1] != w) { if (!tape.fail) tape.fail = 6; continue; }
        for (int k = 0; k < tape.n; ++k) adj[k] = 0.0;
        if (ctx.oe[0] >= 0) adj[ctx.oe[0]] += btaug[og + w];
        if (ctx.oe[1] >= 0) adj[ctx.oe[1]] += btaur[og + w];
        for (int k = tape.n - 1; k >= 0; --k) {
            double g = adj[k];
            if (g == 0.0) continue;
            if (tape.leaf[k] >= 0) { acc[tape.leaf[k]] += g; continue; }
            if (tape.a[k] >= 0) adj[tape.a[k]] += tape.pa[k] * g;
            if (tape.b[k] >= 0) adj[tape.b[k]] += tape.pb[k] * g;
        }
    }
    if (tape.fail) atomicMax(fail, tape.fail);
    atomicMax(fail + 1, high);  // tape high-water mark (entries per g-point), reported by the harness
    bcolh2o[o + L] += acc[0]; bcolco2[o + L] += acc[1]; bcolch4[o + L] += acc[2]; bcolo2[o + L] += acc[3];
    bcolo3[o + L] += acc[4]; bcolmol[o + L] += acc[5]; bfac00[o + L] += acc[6]; bfac01[o + L] += acc[7];
    bfac10[o + L] += acc[8]; bfac11[o + L] += acc[9]; bselffac[o + L] += acc[10]; bselffrac[o + L] += acc[11];
    bforfac[o + L] += acc[12]; bforfrac[o + L] += acc[13];
    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        btaug[og + (size_t)L * RSW_NGPT + iw] = 0.0;
        btaur[og + (size_t)L * RSW_NGPT + iw] = 0.0;
    }
}

// Adjoint (ad/last-rad): the first form's arithmetic in the first form's order, on the compact tape (tvc, 12 bytes an
// entry) and, with LM, thread -> (layer, column) layer-major, so a warp holds one layer of 32 columns and takes one
// lower/upper path (column-major warps straddled laytrop and ran both paths). Each thread still owns one (column,
// layer) and runs its 112 g-points in ascending order, so every derivative word is the first form's bit for bit.
// Same arguments, same fail codes, same grid (ceil(ncol*nlayers/64) x 64) as the first form.
template <int LM, int SM>
__device__ __forceinline__ void p7swc_ad(P7SW_FWD_PARAMS, P7SW_D_PARAMS(b, ), double* __restrict__ btaug,
                                         double* __restrict__ btaur, int* __restrict__ fail)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * nlayers) return;
    int col = LM ? (int)(tid % ncol) : (int)(tid / nlayers);
    int L = LM ? (int)(tid / ncol) : (int)(tid % nlayers);
    size_t o = (size_t)col * nlayers;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    P7SwTapeC tape; tape.n = 0; tape.fail = 0;
    AdCtxC ctx; ctx.t = &tape; ctx.L = L; ctx.nout = 0;
    double adjl[SM ? 1 : P7SWC_TMAX];
    __shared__ double adjs[SM ? P7SWC_TMAX * 64 : 1];
#define ADJ(k) (SM ? adjs[(k) * 64 + threadIdx.x] : adjl[(k)])
    double acc[P7SW_NIN];
    int high = 0;
    for (int s = 0; s < P7SW_NIN; ++s) acc[s] = 0.0;
    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        tape.n = 0; ctx.nout = 0;
        p7swac::rsw_taumol_body(L, iw, nlayers, laytrop[col], tab, ngb, jp + o, jt + o, jt1 + o,
            AdInC{colh2o + o, 0, &ctx}, AdInC{colco2 + o, 1, &ctx}, AdInC{colch4 + o, 2, &ctx},
            AdInC{colo2 + o, 3, &ctx}, AdInC{colo3 + o, 4, &ctx}, AdInC{colmol + o, 5, &ctx},
            AdInC{fac00 + o, 6, &ctx}, AdInC{fac01 + o, 7, &ctx}, AdInC{fac10 + o, 8, &ctx}, AdInC{fac11 + o, 9, &ctx},
            AdInC{selffac + o, 10, &ctx}, AdInC{selffrac + o, 11, &ctx}, indself + o,
            AdInC{forfac + o, 12, &ctx}, AdInC{forfrac + o, 13, &ctx}, indfor + o,
            AdOutC{0, &ctx}, AdOutC{1, &ctx}, true);
        if (tape.n > high) high = tape.n;
        long long w = (long long)L * RSW_NGPT + iw;
        if (ctx.nout != 3 || ctx.oi[0] != w || ctx.oi[1] != w) { if (!tape.fail) tape.fail = 6; continue; }
        for (int k = 0; k < tape.n; ++k) ADJ(k) = 0.0;
        if (ctx.oe[0] >= 0) ADJ(ctx.oe[0]) += btaug[og + w];
        if (ctx.oe[1] >= 0) ADJ(ctx.oe[1]) += btaur[og + w];
        for (int k = tape.n - 1; k >= 0; --k) {
            double g = ADJ(k);
            if (g == 0.0) continue;
            int c = tape.code[k], op = c & 255, a = ((c >> 8) & 255) - 1, b = ((c >> 16) & 255) - 1;
            if (op == 0) { acc[a + 1] += g; continue; }
            double pa, pb;
            if (op == 1) { pa = 1.0; pb = 1.0; }
            else if (op == 2) { pa = 1.0; pb = -1.0; }
            else if (op == 3) { pa = (double)tape.vb[k]; pb = (double)tape.va[k]; }
            else { double bv = (double)tape.vb[k]; pa = 1.0 / bv; pb = -(double)tape.va[k] / (bv * bv); }
            if (a >= 0) ADJ(a) += pa * g;
            if (b >= 0) ADJ(b) += pb * g;
        }
    }
    if (tape.fail) atomicMax(fail, tape.fail);
    atomicMax(fail + 1, high);
    bcolh2o[o + L] += acc[0]; bcolco2[o + L] += acc[1]; bcolch4[o + L] += acc[2]; bcolo2[o + L] += acc[3];
    bcolo3[o + L] += acc[4]; bcolmol[o + L] += acc[5]; bfac00[o + L] += acc[6]; bfac01[o + L] += acc[7];
    bfac10[o + L] += acc[8]; bfac11[o + L] += acc[9]; bselffac[o + L] += acc[10]; bselffrac[o + L] += acc[11];
    bforfac[o + L] += acc[12]; bforfrac[o + L] += acc[13];
    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        btaug[og + (size_t)L * RSW_NGPT + iw] = 0.0;
        btaur[og + (size_t)L * RSW_NGPT + iw] = 0.0;
    }
#undef ADJ
}
#ifndef P7SW_AD_LM
#define P7SW_AD_LM 1
#endif
#ifndef P7SW_AD_SM
#define P7SW_AD_SM 0
#endif
extern "C" __global__ void rsw_taumol_b__ad(P7SW_FWD_PARAMS, P7SW_D_PARAMS(b, ), double* __restrict__ btaug,
                                            double* __restrict__ btaur, int* __restrict__ fail)
{
    p7swc_ad<P7SW_AD_LM, P7SW_AD_SM>(ncol, nlayers, laytrop, tab, ngb, jp, jt, jt1, colh2o, colco2, colch4, colo2, colo3, colmol,
        fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
        bcolh2o, bcolco2, bcolch4, bcolo2, bcolo3, bcolmol, bfac00, bfac01, bfac10, bfac11, bselffac, bselffrac,
        bforfac, bforfrac, btaug, btaur, fail);
}
#ifdef P7SW_BENCH
extern "C" __global__ void rsw_taumol_b__ad_c0(P7SW_FWD_PARAMS, P7SW_D_PARAMS(b, ), double* __restrict__ btaug,
                                               double* __restrict__ btaur, int* __restrict__ fail)
{
    p7swc_ad<0, 0>(ncol, nlayers, laytrop, tab, ngb, jp, jt, jt1, colh2o, colco2, colch4, colo2, colo3, colmol,
        fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
        bcolh2o, bcolco2, bcolch4, bcolo2, bcolo3, bcolmol, bfac00, bfac01, bfac10, bfac11, bselffac, bselffrac,
        bforfac, bforfrac, btaug, btaur, fail);
}
#endif
#ifdef P7SW_BENCH
extern "C" __global__ void rsw_taumol_b__ad_s1(P7SW_FWD_PARAMS, P7SW_D_PARAMS(b, ), double* __restrict__ btaug,
                                               double* __restrict__ btaur, int* __restrict__ fail)
{
    p7swc_ad<1, 1>(ncol, nlayers, laytrop, tab, ngb, jp, jt, jt1, colh2o, colco2, colch4, colo2, colo3, colmol,
        fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
        bcolh2o, bcolco2, bcolch4, bcolo2, bcolo3, bcolmol, bfac00, bfac01, bfac10, bfac11, bselffac, bselffrac,
        bforfac, bforfrac, btaug, btaur, fail);
}
extern "C" __global__ void rsw_taumol_b__ad_s0(P7SW_FWD_PARAMS, P7SW_D_PARAMS(b, ), double* __restrict__ btaug,
                                               double* __restrict__ btaur, int* __restrict__ fail)
{
    p7swc_ad<0, 1>(ncol, nlayers, laytrop, tab, ngb, jp, jt, jt1, colh2o, colco2, colch4, colo2, colo3, colmol,
        fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
        bcolh2o, bcolco2, bcolch4, bcolo2, bcolo3, bcolmol, bfac00, bfac01, bfac10, bfac11, bselffac, bselffrac,
        bforfac, bforfrac, btaug, btaur, fail);
}
#endif
