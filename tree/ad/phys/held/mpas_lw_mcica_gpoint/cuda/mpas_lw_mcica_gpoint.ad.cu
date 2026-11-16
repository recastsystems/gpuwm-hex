// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus harness-only __check and __margin) for mpas_lw_mcica_gpoint, the
// RRTMG LW McICA subcolumn generator (one thread per column x g-point): hexcore/mpasphys/rrtmg_lw_v842.py:532
// (kernel), launched at rrtmg_lw_v842.py:1115 from _lw_mcica_wrapper; CUDA source sha256 683d7a07 (capture
// object, kernel at line 9577; MPAS module_ra_rrtmg_lw.F:2252-2697, 2705-2737). Written by hand from the forward
// CUDA source. The stream helpers below (kiss, lcg / mwc / xorshift jumps) are copied verbatim from that source
// (lines 9465-9473, 9530-9546) so the cloudy decisions are the forward's, word for word.
//
// Forward, per (column c, g-point ig), layer k = 0..nl-1 (a = abase + c*12*nl, A(f,k) = a[f*nl+k]):
//   cf = A(5,k), flushed to 0 below 1e-20                                       (capture source 9593)
//   cdf = KISS draw (or the shared draw for overlap 3); overlap 2: max-random chain on prev / prevcf (9599);
//   overlap >= 4: exponential chain with weights[k] (9595-9598)
//   cloudy = cdf >= 1 - cf                                                      (9600)
//   cloud = cloudy ? 1 : 0;  liquid = cloudy ? A(6,k) : 0;  ice = cloudy ? A(7,k) : 0;  snow = cloudy ? A(8,k) : 0
//
// Linearisation (frozen-coefficient, documented): the only continuous dependence of an output word on an input
// word is the masked copy liquid/ice/snow <- A(6/7/8,k). cf (A(5,k)), the chain words prev / prevcf, the weights
// and the integer stream state enter only through the comparisons at 9593, 9597-9599 and 9600, i.e. through the
// 0/1 mask. That mask is a step function of the inputs: its derivative is 0 almost everywhere and undefined on
// the threshold set cdf == 1 - cf. It is frozen at the trajectory's value (recomputed bitwise from the
// pre-launch seeds / weights / xorshift words, see __check), so
//   d liquid = mask * d A(6,k),  d ice = mask * d A(7,k),  d snow = mask * d A(8,k),  d cloud = 0.
// This is the exact derivative of the real map wherever no draw sits on its threshold (the harness measures the
// smallest |cdf - (1-cf)| margin per launch). The McICA samples are therefore "saved" as the kernel's own
// pre-launch inputs (4 u32 seeds per column plus the 140 x 32 xorshift table and the weights), which the window
// already copies as primal pre-launch words; no 140 x nl x nc mask is stored.
//
// Precision: the TL is a masked copy (no arithmetic, bitwise: written word = d A word or +0). The AD sums the 140
// g-point cotangents of each (c, f, k) in double, in ascending ig order, adds the incoming accumulate word in
// double and rounds once to f32 (the glue-row rule). Built --fmad=false --ftz=false like the forward.
// Geometry: both entries use the forward launch (ceil(nc*140/128) x 128). The TL maps t -> (c, ig) as the
// forward does. The AD maps t -> (c, k) = (t/140, t%140) and returns for k >= nl (nl <= 140 is checked), so one
// thread owns one (c, k) and all 140 g-point words of that layer: deterministic, no atomics on data words.
// In/out: TL d_abase is in (rows 6..8 read), d_cloud / d_liquid / d_ice / d_snow are full overwrites (out).
// AD b_abase rows 6..8 accumulate (rows 0..5, 9..11 untouched), b_cloud / b_liquid / b_ice / b_snow are consumed
// (zeroed). Refusal: slot 0 = nonfinite cf or nonfinite written derivative word; slot 1 = nl outside 1..140 or an
// overlap value outside 1..5 (overlap 0 never launches: the wrapper returns zeros, rrtmg_lw_v842.py:1104).

#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_SU(a, b) __fsub_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define MLG_NG 140

// --- verbatim from capture source 683d7a07 (9465-9473) ---
__device__ float mpas_kiss(unsigned int* seed){
    seed[0]=69069u*seed[0]+1327217885u;
    seed[1]^=seed[1]<<13;seed[1]^=seed[1]>>17;seed[1]^=seed[1]<<5;
    seed[2]=18000u*(seed[2]&65535u)+(seed[2]>>16);
    seed[3]=30903u*(seed[3]&65535u)+(seed[3]>>16);
    int kiss=(int)(seed[0]+seed[1]+(seed[2]<<16)+seed[3]);
    return RLW_AD(RLW_MU((float)kiss,2.328306e-10f),0.5f);
}
// --- verbatim from capture source 683d7a07 (9530-9546) ---
__device__ unsigned int mpas_lcg_jump(unsigned int s,unsigned int n){
    unsigned int ra=1u,rc=0u,pa=69069u,pc=1327217885u;
    while(n){if(n&1u){rc=pa*rc+pc;ra=pa*ra;}pc=pa*pc+pc;pa=pa*pa;n>>=1;}
    return ra*s+rc;
}
__device__ unsigned int mpas_mwc_jump(unsigned int s,unsigned int a,unsigned int n){
    const unsigned long long m=(unsigned long long)a*65536ull-1ull;
    while(n&&(unsigned long long)s>m){s=a*(s&65535u)+(s>>16);--n;}
    if(!n||(unsigned long long)s==m||s==0u)return s;
    unsigned long long r=1ull,base=a;
    while(n){if(n&1u)r=r*base%m;base=base*base%m;n>>=1;}
    return (unsigned int)((unsigned long long)s*r%m);
}
__device__ unsigned int mpas_xorshift_jump(unsigned int s,const unsigned int* columns){
    unsigned int v=0u;
    for(int i=0;i<32;++i)if((s>>i)&1u)v^=columns[i];
    return v;
}

// The forward's per-(c, ig) layer walk (capture source 9583-9602), one layer per step() call.
struct MlgStream {
    unsigned int seed[4]; const float* a; const float* weight; int nl, overlap;
    float prev, prevcf, shared, cf, margin;
    __device__ void init(int nl_, int overlap_, int draws, const float* abase, const unsigned int* seeds,
                         const float* weights, const unsigned int* xorshift, long long c, int ig) {
        nl = nl_; overlap = overlap_; a = abase + c * 12 * nl;
        const unsigned int n = (unsigned int)ig * (unsigned int)draws;
        seed[0] = mpas_lcg_jump(seeds[c * 4], n);
        seed[1] = mpas_xorshift_jump(seeds[c * 4 + 1], xorshift + ig * 32);
        seed[2] = mpas_mwc_jump(seeds[c * 4 + 2], 18000u, n);
        seed[3] = mpas_mwc_jump(seeds[c * 4 + 3], 30903u, n);
        weight = weights + (overlap >= 4 ? c * nl : 0);
        prev = 0.0f; prevcf = 0.0f; shared = 0.0f;
        if (overlap == 3) shared = mpas_kiss(seed);
    }
    __device__ bool step(int k) {
        cf = a[5 * nl + k]; if (cf < 1.e-20f) cf = 0.0f;
        float cdf = overlap == 3 ? shared : mpas_kiss(seed);
        if (overlap >= 4) { float cdf2 = mpas_kiss(seed);
            float alpha = weight[k];
            if (overlap == 5 && cf == 0.0f && prevcf > 0.0f) alpha = 0.0f;
            if (k && cdf2 < alpha) cdf = prev;
        } else if (overlap == 2 && k) { float threshold = RLW_SU(1.0f, prevcf); cdf = prev > threshold ? prev : RLW_MU(cdf, threshold); }
        const float thr = RLW_SU(1.0f, cf);
        bool cloudy = cdf >= thr;
        margin = RLW_SU(cdf, thr);   // harness only (sign and size of the decision's distance to its threshold)
        prev = cdf; prevcf = cf;
        return cloudy;
    }
};

__device__ __forceinline__ bool mlg_bad_args(int nl, int overlap) { return nl < 1 || nl > MLG_NG || overlap < 1 || overlap > 5; }

// Harness only (not in launch_abi): the forward output words from the shared walk, out of place.
extern "C" __global__ void mpas_lw_mcica_gpoint__check(int nc, int nl, int overlap, int draws, const float* abase,
    const unsigned int* seeds, const float* weights, const unsigned int* xorshift,
    float* cloud, float* liquid, float* ice, float* snow) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MLG_NG) return;
    const int ig = (int)(t % MLG_NG); const long long c = t / MLG_NG;
    MlgStream s; s.init(nl, overlap, draws, abase, seeds, weights, xorshift, c, ig);
    for (int k = 0; k < nl; ++k) {
        bool cloudy = s.step(k); const long long idx = (c * nl + k) * MLG_NG + ig;
        cloud[idx] = cloudy ? 1.0f : 0.0f; liquid[idx] = cloudy ? s.a[6 * nl + k] : 0.0f;
        ice[idx] = cloudy ? s.a[7 * nl + k] : 0.0f; snow[idx] = cloudy ? s.a[8 * nl + k] : 0.0f;
    }
}

// Harness only: the signed threshold margin cdf - (1 - cf) of every decision (range hazard 1).
extern "C" __global__ void mpas_lw_mcica_gpoint__margin(int nc, int nl, int overlap, int draws, const float* abase,
    const unsigned int* seeds, const float* weights, const unsigned int* xorshift, float* margin) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MLG_NG) return;
    const int ig = (int)(t % MLG_NG); const long long c = t / MLG_NG;
    MlgStream s; s.init(nl, overlap, draws, abase, seeds, weights, xorshift, c, ig);
    for (int k = 0; k < nl; ++k) { s.step(k); margin[(c * nl + k) * MLG_NG + ig] = s.margin; }
}

// TL: d_cloud = 0, d_{liquid,ice,snow} = mask * d_abase rows 6..8 (masked copy, bitwise).
extern "C" __global__ void mpas_lw_mcica_gpoint__tl(int nc, int nl, int overlap, int draws, const float* abase,
    const unsigned int* seeds, const float* weights, const unsigned int* xorshift,
    const float* d_abase, float* d_cloud, float* d_liquid, float* d_ice, float* d_snow, int* failure) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MLG_NG) return;
    if (mlg_bad_args(nl, overlap)) { atomicExch(failure + 1, 1); return; }
    const int ig = (int)(t % MLG_NG); const long long c = t / MLG_NG;
    const float* da = d_abase + c * 12 * nl;
    MlgStream s; s.init(nl, overlap, draws, abase, seeds, weights, xorshift, c, ig);
    bool bad = false;
    for (int k = 0; k < nl; ++k) {
        bool cloudy = s.step(k); const long long idx = (c * nl + k) * MLG_NG + ig;
        bad |= !isfinite(s.cf);
        float l = cloudy ? da[6 * nl + k] : 0.0f, i = cloudy ? da[7 * nl + k] : 0.0f, w = cloudy ? da[8 * nl + k] : 0.0f;
        bad |= !isfinite(l) || !isfinite(i) || !isfinite(w);
        d_cloud[idx] = 0.0f; d_liquid[idx] = l; d_ice[idx] = i; d_snow[idx] = w;
    }
    if (bad) atomicExch(failure, 1);
}

// AD: thread (c, k): b_abase(6..8, k) += sum_ig mask(c,k,ig) * b_{liquid,ice,snow}(c,k,ig) (double, ig ascending,
// one f32 rounding); b_cloud / b_liquid / b_ice / b_snow of layer k consumed (zeroed).
extern "C" __global__ void mpas_lw_mcica_gpoint__ad(int nc, int nl, int overlap, int draws, const float* abase,
    const unsigned int* seeds, const float* weights, const unsigned int* xorshift,
    float* b_abase, float* b_cloud, float* b_liquid, float* b_ice, float* b_snow, int* failure) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MLG_NG) return;
    if (mlg_bad_args(nl, overlap)) { atomicExch(failure + 1, 1); return; }
    const long long c = t / MLG_NG; const int k = (int)(t % MLG_NG);
    if (k >= nl) return;
    double sl = 0.0, si = 0.0, sw = 0.0; bool bad = false;
    for (int ig = 0; ig < MLG_NG; ++ig) {
        MlgStream s; s.init(nl, overlap, draws, abase, seeds, weights, xorshift, c, ig);
        bool cloudy = false;
        for (int kk = 0; kk <= k; ++kk) cloudy = s.step(kk);
        bad |= !isfinite(s.cf);
        const long long idx = (c * nl + k) * MLG_NG + ig;
        if (cloudy) { sl += (double)b_liquid[idx]; si += (double)b_ice[idx]; sw += (double)b_snow[idx]; }
        b_cloud[idx] = 0.0f; b_liquid[idx] = 0.0f; b_ice[idx] = 0.0f; b_snow[idx] = 0.0f;
    }
    float* ba = b_abase + c * 12 * nl;
    float ol = (float)((double)ba[6 * nl + k] + sl), oi = (float)((double)ba[7 * nl + k] + si), ow = (float)((double)ba[8 * nl + k] + sw);
    bad |= !isfinite(ol) || !isfinite(oi) || !isfinite(ow);
    ba[6 * nl + k] = ol; ba[7 * nl + k] = oi; ba[8 * nl + k] = ow;
    if (bad) atomicExch(failure, 1);
}
