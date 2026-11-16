// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: aug / tl / ad) for rlw_rtrn_accum_rows, RRTMG LW (MPAS-A v8.4.2 port),
// hexcore/mpasphys/rrtmg_lw_v842.py:858 (launch), kernel source captured as .cu 683d7a07 lines 9050-9141
// (image 34da9e09).  Ported from RRTMG_LW rtrnmc flux accumulation (rrtmg_lw_rtrnmc.f90, the totuflux /
// totdflux / totuclfl / totdclfl band sums), statement for statement as rlw_rtrn_accum_coalesced (.cu line 2970).
//
// Forward, one thread per (column, level) row r = col*(nl+1)+lev, block = RLW_ACC_ROWS = 64 rows:
//   for each band b (g-points g_lo..g_hi-1 from ngs), with s_b = wtdiff * delwave[b]:
//     down  (lev <= nl-1):  drad += radld[col,lev,g];   clrd = iclddn[col,lev,g] ? clrd + radclrd[col,lev,g] : drad
//     up    (all lev):      urad += U[g];               clru = iclddn[col,0,g]   ? clru + UC[g]             : urad
//       with U = radlu_sfc[col,g], UC = radclru_sfc[col,g] at lev 0 and U = radlu[col,lev-1,g], UC = radclru[...]
//       at lev >= 1 (iclddn at layer 0 is the column final down-path cloud flag, so the up-path clear test
//       uses it at every level, as rtrnmc does with its scalar iclddn after the downward march)
//     totu += (urad*wtdiff)*dw;  totd += (drad*wtdiff)*dw;  totuc += (clru*wtdiff)*dw;  totdc += (clrd*wtdiff)*dw
//   and the four totals are stored (all ncol*(nl+1) words; totd and totdc are 0 at lev = nl).
//
// The derivative.  The kernel is LINEAR in its six float inputs (radld, radclrd, radlu, radclru, radlu_sfc,
// radclru_sfc).  iclddn (uint8 cloud flags), ngs, delwave and wtdiff are inactive: flags and configuration
// tables / constants.  The clear-sky reset "clr = all-sky running sum" (forward .cu 9111-9112 and 9118-9126)
// is a SELECTION decided by the flags, not a limiter: frozen-coefficient treatment, the flags of the trajectory
// pick the coefficients and have zero derivative.  With k = the last lane of the band whose flag is 0 (-1 if
// none), the band clear sum is  sum_{g<=k} all-sky[g] + sum_{g>k} clear[g]  exactly, so
//   d totd /d radld[col,lev,g]   = s_b                   d totdc/d radld[col,lev,g]  = s_b if g <= k_d(lev)
//   d totdc/d radclrd[col,lev,g] = s_b if g > k_d(lev)   (lev <= nl-1; nothing else reads a down word)
//   d totu /d U[g] = s_b      d totuc/d U[g] = s_b if g <= k_u      d totuc/d UC[g] = s_b if g > k_u
// Every input word inside the launch range is read by exactly ONE row (down word (c,l,g) by row (c,l); radlu
// word (c,l,g) by row (c,l+1); sfc word (c,g) by row (c,0)), so the AD is a gather with no atomics and each
// block owns its rows input and output words: deterministic, block-local, no grid-wide ordering.
// No tape: the TL and AD read only the flags and the tables, never a float primal word, so they bind to any
// state (pre- or post-forward) bitwise alike.  The forward overwrites only its four outputs.
//
// Precision (the live-scheme / glue-row rule): __aug repeats the forward word for word in f32 (__fadd_rn /
// __fmul_rn, built --fmad=false --ftz=false), bitwise to the captured image.  __tl and __ad keep the forward
// tile staging and order, accumulate in double and round once per written word; the coefficient s_b is formed in
// double from the f32 words wtdiff and delwave.  +0.0 is added before each rounding so a zero tangent or
// cotangent is +0 (bitwise zero), never -0.
// Failure (one slot, source "rrtmg_lw_v842.py:858:accum_rows_geometry"): block size other than 64 x 1 x 1
// (rows past blockDim would be skipped, rows past 64 overrun the tiles), a band wider than RLW_ACC_G = 16 or a
// negative band, ngs past NGPTLW, or a grid that does not cover ncol*(nl+1) rows; the TL and AD also flag any
// nonfinite written word (and the AD a nonfinite seed).  On a bad geometry every entry returns before writing.

#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define NBNDLW 16
#define NGPTLW 140
#define RLW_ACC_ROWS 64
#define RLW_ACC_G 16

__device__ __forceinline__ bool rlra_bad(int ncol, int nl, const int* __restrict__ ngs) {
    if (blockDim.x != RLW_ACC_ROWS || blockDim.y != 1 || blockDim.z != 1) return true;
    if (ncol < 0 || nl < 1) return true;
    if ((long long)gridDim.x * RLW_ACC_ROWS < (long long)ncol * (nl + 1)) return true;
    int lo = 0;
    for (int b = 0; b < NBNDLW; ++b) {
        int hi = ngs[b];
        if (hi - lo < 0 || hi - lo > RLW_ACC_G) return true;
        lo = hi;
    }
    return lo > NGPTLW;
}

// ---- aug: the forward, word for word (captured .cu 9050-9141) ------------------------------------------------
extern "C" __global__ void rlw_rtrn_accum_rows__aug(
    int ncol, int nl,
    const float* __restrict__ radld_p, const float* __restrict__ radclrd_p,
    const float* __restrict__ radlu_p, const float* __restrict__ radclru_p,
    const unsigned char* __restrict__ iclddn_p,
    const float* __restrict__ radlu_sfc, const float* __restrict__ radclru_sfc,
    const int* __restrict__ ngs, const float* __restrict__ delwave, float wtdiff,
    float* __restrict__ totuflux, float* __restrict__ totdflux,
    float* __restrict__ totuclfl, float* __restrict__ totdclfl,
    int* __restrict__ failure)
{
    if (rlra_bad(ncol, nl, ngs)) { if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(failure, 1); return; }
    __shared__ float down[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float downc[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float up[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float upc[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag0[RLW_ACC_ROWS][RLW_ACC_G + 1];
    const long long rows = (long long)ncol * (nl + 1);
    const long long base = (long long)blockIdx.x * RLW_ACC_ROWS;
    const int r = threadIdx.x;
    const bool active = base + r < rows;
    const int col = active ? (int)((base + r) / (nl + 1)) : 0;
    const int lev = active ? (int)((base + r) % (nl + 1)) : 0;
    float totu = 0.0f, totd = 0.0f, totuc = 0.0f, totdc = 0.0f;
    for (int iband = 1; iband <= NBNDLW; ++iband) {
        int g_lo = (iband == 1) ? 0 : ngs[iband - 2];
        int g_hi = ngs[iband - 1];
        int ng = g_hi - g_lo;
        __syncthreads();
        for (int slot = threadIdx.x; slot < RLW_ACC_ROWS * ng; slot += blockDim.x) {
            int row = slot / ng, gg = slot % ng, g = g_lo + gg;
            long long t = base + row;
            if (t >= rows) continue;
            int c = (int)(t / (nl + 1)), l = (int)(t % (nl + 1));
            if (l < nl) {
                long long i = ((long long)c * nl + l) * NGPTLW + g;
                down[row][gg] = radld_p[i]; downc[row][gg] = radclrd_p[i];
                flag[row][gg] = iclddn_p[i];
            }
            flag0[row][gg] = iclddn_p[(long long)c * nl * NGPTLW + g];
            long long iu = l == 0 ? (long long)c * NGPTLW + g
                                  : ((long long)c * nl + l - 1) * NGPTLW + g;
            up[row][gg] = l == 0 ? radlu_sfc[iu] : radlu_p[iu];
            upc[row][gg] = l == 0 ? radclru_sfc[iu] : radclru_p[iu];
        }
        __syncthreads();
        if (!active) continue;
        float urad = 0.0f, drad = 0.0f, clru = 0.0f, clrd = 0.0f;
        for (int lane = g_lo; lane < g_hi; ++lane) {
            if (lev <= nl - 1) {
                drad = RLW_AD(drad, down[r][lane - g_lo]);
                if (flag[r][lane - g_lo])
                    clrd = RLW_AD(clrd, downc[r][lane - g_lo]);
                else
                    clrd = drad;
            }
            if (lev == 0) {
                urad = RLW_AD(urad, up[r][lane - g_lo]);
                if (flag0[r][lane - g_lo])
                    clru = RLW_AD(clru, upc[r][lane - g_lo]);
                else
                    clru = urad;
            } else {
                urad = RLW_AD(urad, up[r][lane - g_lo]);
                if (flag0[r][lane - g_lo])
                    clru = RLW_AD(clru, upc[r][lane - g_lo]);
                else
                    clru = urad;
            }
        }
        float dw = delwave[iband - 1];
        totu = RLW_AD(totu, RLW_MU(RLW_MU(urad, wtdiff), dw));
        totd = RLW_AD(totd, RLW_MU(RLW_MU(drad, wtdiff), dw));
        totuc = RLW_AD(totuc, RLW_MU(RLW_MU(clru, wtdiff), dw));
        totdc = RLW_AD(totdc, RLW_MU(RLW_MU(clrd, wtdiff), dw));
    }
    if (!active) return;
    totuflux[(long long)col * (nl + 1) + lev] = totu;
    totdflux[(long long)col * (nl + 1) + lev] = totd;
    totuclfl[(long long)col * (nl + 1) + lev] = totuc;
    totdclfl[(long long)col * (nl + 1) + lev] = totdc;
}

// ---- tl: d_tot* (out, overwritten on the launch rows) from d_rad* (in) ------------------------------------------
extern "C" __global__ void rlw_rtrn_accum_rows__tl(
    int ncol, int nl,
    const unsigned char* __restrict__ iclddn_p,
    const int* __restrict__ ngs, const float* __restrict__ delwave, float wtdiff,
    const float* __restrict__ d_radld, const float* __restrict__ d_radclrd,
    const float* __restrict__ d_radlu, const float* __restrict__ d_radclru,
    const float* __restrict__ d_radlu_sfc, const float* __restrict__ d_radclru_sfc,
    float* __restrict__ d_totuflux, float* __restrict__ d_totdflux,
    float* __restrict__ d_totuclfl, float* __restrict__ d_totdclfl,
    int* __restrict__ failure)
{
    if (rlra_bad(ncol, nl, ngs)) { if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(failure, 1); return; }
    __shared__ float down[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float downc[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float up[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float upc[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag0[RLW_ACC_ROWS][RLW_ACC_G + 1];
    const long long rows = (long long)ncol * (nl + 1);
    const long long base = (long long)blockIdx.x * RLW_ACC_ROWS;
    const int r = threadIdx.x;
    const bool active = base + r < rows;
    const int col = active ? (int)((base + r) / (nl + 1)) : 0;
    const int lev = active ? (int)((base + r) % (nl + 1)) : 0;
    double totu = 0.0, totd = 0.0, totuc = 0.0, totdc = 0.0;
    for (int iband = 1; iband <= NBNDLW; ++iband) {
        int g_lo = (iband == 1) ? 0 : ngs[iband - 2];
        int g_hi = ngs[iband - 1];
        int ng = g_hi - g_lo;
        __syncthreads();
        for (int slot = threadIdx.x; slot < RLW_ACC_ROWS * ng; slot += blockDim.x) {
            int row = slot / ng, gg = slot % ng, g = g_lo + gg;
            long long t = base + row;
            if (t >= rows) continue;
            int c = (int)(t / (nl + 1)), l = (int)(t % (nl + 1));
            if (l < nl) {
                long long i = ((long long)c * nl + l) * NGPTLW + g;
                down[row][gg] = d_radld[i]; downc[row][gg] = d_radclrd[i];
                flag[row][gg] = iclddn_p[i];
            }
            flag0[row][gg] = iclddn_p[(long long)c * nl * NGPTLW + g];
            long long iu = l == 0 ? (long long)c * NGPTLW + g
                                  : ((long long)c * nl + l - 1) * NGPTLW + g;
            up[row][gg] = l == 0 ? d_radlu_sfc[iu] : d_radlu[iu];
            upc[row][gg] = l == 0 ? d_radclru_sfc[iu] : d_radclru[iu];
        }
        __syncthreads();
        if (!active) continue;
        double urad = 0.0, drad = 0.0, clru = 0.0, clrd = 0.0;
        for (int lane = 0; lane < ng; ++lane) {
            if (lev <= nl - 1) {                                   // forward .cu 9108-9113
                drad += (double)down[r][lane];
                if (flag[r][lane]) clrd += (double)downc[r][lane];
                else clrd = drad;                                  // frozen selection (flag of the trajectory)
            }
            urad += (double)up[r][lane];                           // forward .cu 9115-9127 (lev 0 and lev >= 1 alike)
            if (flag0[r][lane]) clru += (double)upc[r][lane];
            else clru = urad;
        }
        double s = (double)wtdiff * (double)delwave[iband - 1];    // d((x*wtdiff)*dw)/dx
        totu += urad * s; totd += drad * s; totuc += clru * s; totdc += clrd * s;
    }
    if (!active) return;
    const long long o = (long long)col * (nl + 1) + lev;
    float tu = (float)(totu + 0.0), td = (float)(totd + 0.0), tuc = (float)(totuc + 0.0), tdc = (float)(totdc + 0.0);
    d_totuflux[o] = tu; d_totdflux[o] = td; d_totuclfl[o] = tuc; d_totdclfl[o] = tdc;
    if (!isfinite(tu) || !isfinite(td) || !isfinite(tuc) || !isfinite(tdc)) atomicExch(failure, 1);
}

// ---- ad: b_rad* (inout, accumulated) from b_tot* (inout: seed read, then consumed = zeroed) ----------------------
extern "C" __global__ void rlw_rtrn_accum_rows__ad(
    int ncol, int nl,
    const unsigned char* __restrict__ iclddn_p,
    const int* __restrict__ ngs, const float* __restrict__ delwave, float wtdiff,
    float* __restrict__ b_radld, float* __restrict__ b_radclrd,
    float* __restrict__ b_radlu, float* __restrict__ b_radclru,
    float* __restrict__ b_radlu_sfc, float* __restrict__ b_radclru_sfc,
    float* __restrict__ b_totuflux, float* __restrict__ b_totdflux,
    float* __restrict__ b_totuclfl, float* __restrict__ b_totdclfl,
    int* __restrict__ failure)
{
    if (rlra_bad(ncol, nl, ngs)) { if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(failure, 1); return; }
    __shared__ double yu[RLW_ACC_ROWS], yd[RLW_ACC_ROWS], yuc[RLW_ACC_ROWS], ydc[RLW_ACC_ROWS];
    __shared__ unsigned char flag[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag0[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ int lastd[RLW_ACC_ROWS], lastu[RLW_ACC_ROWS];
    const long long rows = (long long)ncol * (nl + 1);
    const long long base = (long long)blockIdx.x * RLW_ACC_ROWS;
    const int r = threadIdx.x;
    const bool active = base + r < rows;
    const int lev = active ? (int)((base + r) % (nl + 1)) : 0;
    bool nf = false;
    if (active) {
        // The row four output seeds are read by this thread only; the outputs are overwritten by the forward, so
        // their pre-launch cotangent is zero: consumed here, after the read.
        const long long o = base + r;                              // = col*(nl+1)+lev
        yu[r] = (double)b_totuflux[o]; yd[r] = (double)b_totdflux[o];
        yuc[r] = (double)b_totuclfl[o]; ydc[r] = (double)b_totdclfl[o];
        nf = !isfinite(yu[r]) || !isfinite(yd[r]) || !isfinite(yuc[r]) || !isfinite(ydc[r]);
        b_totuflux[o] = 0.0f; b_totdflux[o] = 0.0f; b_totuclfl[o] = 0.0f; b_totdclfl[o] = 0.0f;
    } else {
        yu[r] = 0.0; yd[r] = 0.0; yuc[r] = 0.0; ydc[r] = 0.0;
    }
    for (int iband = 1; iband <= NBNDLW; ++iband) {
        int g_lo = (iband == 1) ? 0 : ngs[iband - 2];
        int g_hi = ngs[iband - 1];
        int ng = g_hi - g_lo;
        __syncthreads();
        for (int slot = threadIdx.x; slot < RLW_ACC_ROWS * ng; slot += blockDim.x) {
            int row = slot / ng, gg = slot % ng, g = g_lo + gg;
            long long t = base + row;
            if (t >= rows) continue;
            int c = (int)(t / (nl + 1)), l = (int)(t % (nl + 1));
            if (l < nl) flag[row][gg] = iclddn_p[((long long)c * nl + l) * NGPTLW + g];
            flag0[row][gg] = iclddn_p[(long long)c * nl * NGPTLW + g];
        }
        __syncthreads();
        if (active) {                                              // k = last lane with flag 0 (the last reset)
            int kd = -1, ku = -1;
            for (int lane = 0; lane < ng; ++lane) {
                if (lev <= nl - 1 && !flag[r][lane]) kd = lane;
                if (!flag0[r][lane]) ku = lane;
            }
            lastd[r] = kd; lastu[r] = ku;
        }
        __syncthreads();
        const double s = (double)wtdiff * (double)delwave[iband - 1];
        for (int slot = threadIdx.x; slot < RLW_ACC_ROWS * ng; slot += blockDim.x) {
            int row = slot / ng, gg = slot % ng, g = g_lo + gg;
            long long t = base + row;
            if (t >= rows) continue;
            int c = (int)(t / (nl + 1)), l = (int)(t % (nl + 1));
            if (l < nl) {
                long long i = ((long long)c * nl + l) * NGPTLW + g;
                double a = yd[row] + (gg <= lastd[row] ? ydc[row] : 0.0);
                double ac = (flag[row][gg] && gg > lastd[row]) ? ydc[row] : 0.0;
                float va = (float)((double)b_radld[i] + s * a + 0.0);
                float vc = (float)((double)b_radclrd[i] + s * ac + 0.0);
                b_radld[i] = va; b_radclrd[i] = vc;
                nf = nf || !isfinite(va) || !isfinite(vc);
            }
            double au = yu[row] + (gg <= lastu[row] ? yuc[row] : 0.0);
            double auc = (flag0[row][gg] && gg > lastu[row]) ? yuc[row] : 0.0;
            float* bu = l == 0 ? b_radlu_sfc : b_radlu;
            float* buc = l == 0 ? b_radclru_sfc : b_radclru;
            long long iu = l == 0 ? (long long)c * NGPTLW + g : ((long long)c * nl + l - 1) * NGPTLW + g;
            float vu = (float)((double)bu[iu] + s * au + 0.0);
            float vuc = (float)((double)buc[iu] + s * auc + 0.0);
            bu[iu] = vu; buc[iu] = vuc;
            nf = nf || !isfinite(vu) || !isfinite(vuc);
        }
    }
    if (nf) atomicExch(failure, 1);
}
