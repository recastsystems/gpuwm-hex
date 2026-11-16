// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus harness-only __check and __margin) for mpas_sw_mcica_gpoint, the
// RRTMG SW McICA subcolumn generator (one thread per column x g-point, 112 g-points): hexcore/mpasphys/
// rrtmg_sw_v842.py:3386 (kernel), launched at rrtmg_sw_v842.py:567 from _wrapper_mcica; CUDA source sha256 3c499c7b
// (capture object, kernel at line 3705; MPAS module_ra_rrtmg_sw.F:1403-2012, 2016-2048). Written by hand from the
// forward CUDA source. The stream helpers below (kiss, lcg / mwc jumps, the xorshift GF(2) jump) are copied verbatim
// from that source (lines 3589-3594, 3666-3678, 3714) so the cloudy decisions are the forward's, word for word.
//
// Forward, per (column c, g-point g), layer k = 0..nl-1 (X(a,k) = a[c*nl+k]; out slab f at ((f*nc+c)*nl+k)*112+g):
//   frac = cf(k), flushed to 0 below 1e-20                                     (capture source 3721)
//   r = KISS draw (or the shared draw for icld 3); icld 2: max-random chain on prev / prev_cf (3722);
//   icld 4/5: exponential chain with weights[c*nl+k] (3723-3724)
//   cloudy = r >= 1 - frac                                                     (3725)
//   out0 = cloudy ? 1 : 0;  out1 = cloudy ? iw(k) : 0;  out2 = cloudy ? lw(k) : 0;  out3 = cloudy ? sw(k) : 0  (3725-3726)
//   out slabs 4..7 (band optics, pre-filled by the wrapper at rrtmg_sw_v842.py:548-550) are not touched.
//
// Linearisation (frozen-coefficient, documented): the only continuous dependence of an output word on an input
// word is the masked copy out1/out2/out3 <- iw/lw/sw. cf, the chain words prev / prev_cf, the weights and the
// integer stream state enter only through the comparisons at 3721, 3722, 3724 and 3725, i.e. through the 0/1 mask.
// That mask is a step function of the inputs: its derivative is 0 almost everywhere and undefined on the threshold
// set r == 1 - frac. It is frozen at the trajectory's value (recomputed bitwise from the pre-launch cf / seeds /
// weights / xorshift words, see __check), so
//   d out1 = mask * d iw,  d out2 = mask * d lw,  d out3 = mask * d sw,  d out0 = 0,  d out4..7 pass through.
// This is the exact derivative of the real map wherever no draw sits on its threshold (the harness measures the
// smallest |r - (1-frac)| margin per launch). cf is declared inactive (frozen coefficient). The McICA samples are
// "saved" as the kernel's own pre-launch inputs (4 u32 seeds per column, the 112 x 32 xorshift table, the weights),
// which the window already copies as primal pre-launch words; no 112 x nl x nc mask is stored.
//
// Precision: the TL is a masked copy (no arithmetic, bitwise: written word = d input word or +0). The AD sums the
// 112 g-point cotangents of each (c, slab, k) in double, in ascending g order, adds the incoming accumulate word in
// double and rounds once to f32 (the glue-row rule). Built --fmad=false --ftz=false like the forward.
// Geometry: both entries use the forward launch (ceil(nc*112/64) x 64). The TL maps t -> (c, g) as the forward
// does. The AD maps t -> (c, k) = (t/112, t%112) and returns for k >= nl (nl <= 112 is checked), so one thread owns
// one (c, k) and all 112 g-point words of that layer: deterministic, no atomics on data words.
// In/out: TL d_iw / d_lw / d_sw are in; d_out is a partial overwrite (slabs 0..3 written, slabs 4..7 untouched,
// so their incoming tangent passes through in place). AD b_iw / b_lw / b_sw accumulate; b_out slabs 0..3 are
// consumed (zeroed), slabs 4..7 untouched (the identity adjoint of the pass-through). Refusal: slot 0 = nonfinite
// cf or nonfinite written derivative word; slot 1 = nl outside 1..112 or icld outside 1..5 (the wrapper sends
// icld outside 1..5 to the per-column mpas_sw_mcica kernel instead, rrtmg_sw_v842.py:551-558).

#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define MSG_NG 112

// --- verbatim from capture source 3c499c7b (3589-3594) ---
__device__ float mpas_kiss(unsigned int& s1,unsigned int& s2,unsigned int& s3,unsigned int& s4) {
 s1=69069u*s1+1327217885u;s2^=s2<<13;s2^=s2>>17;s2^=s2<<5;
 s3=18000u*(s3&65535u)+(s3>>16);s4=30903u*(s4&65535u)+(s4>>16);
 int kiss=(int)(s1+s2+(s3<<16)+s4);
 return FADD(FMUL((float)kiss,2.328306e-10f),.5f);
}
// --- verbatim from capture source 3c499c7b (3666-3678) ---
__device__ unsigned int mpas_sw_lcg_jump(unsigned int s,unsigned int n){
 unsigned int ra=1u,rc=0u,pa=69069u,pc=1327217885u;
 while(n){if(n&1u){rc=pa*rc+pc;ra=pa*ra;}pc=pa*pc+pc;pa=pa*pa;n>>=1;}
 return ra*s+rc;
}
__device__ unsigned int mpas_sw_mwc_jump(unsigned int s,unsigned int a,unsigned int n){
 const unsigned long long m=(unsigned long long)a*65536ull-1ull;
 while(n&&(unsigned long long)s>m){s=a*(s&65535u)+(s>>16);--n;}
 if(!n||(unsigned long long)s==m||s==0u)return s;
 unsigned long long r=1ull,base=a;
 while(n){if(n&1u)r=r*base%m;base=base*base%m;n>>=1;}
 return (unsigned int)((unsigned long long)s*r%m);
}

// The forward's per-(c, g) layer walk (capture source 3712-3727), one layer per step() call.
struct MsgStream {
    unsigned int s1, s2, s3, s4; const float* cf; const float* weights; long long c; int nl, icld;
    float prev, prev_cf, maximum, frac, margin;
    __device__ void init(int nl_, int icld_, int draws, const float* cf_, const unsigned int* seeds,
                         const float* weights_, const unsigned int* xorshift, long long c_, int g) {
        nl = nl_; icld = icld_; cf = cf_; weights = weights_; c = c_;
        const unsigned int n = (unsigned int)g * (unsigned int)draws;                          // 3712
        s1 = mpas_sw_lcg_jump(seeds[c * 4], n);                                                 // 3713
        s2 = 0u; { unsigned int v = seeds[c * 4 + 1]; const unsigned int* col = xorshift + g * 32;
                   for (int i = 0; i < 32; i++) if ((v >> i) & 1u) s2 ^= col[i]; }             // 3714
        s3 = mpas_sw_mwc_jump(seeds[c * 4 + 2], 18000u, n);                                     // 3715
        s4 = mpas_sw_mwc_jump(seeds[c * 4 + 3], 30903u, n);                                     // 3716
        prev = 0.f; prev_cf = 0.f;                                                              // 3717
        maximum = 0.f; if (icld == 3) maximum = mpas_kiss(s1, s2, s3, s4);                     // 3718
    }
    __device__ bool step(int k) {
        float r = icld == 3 ? maximum : mpas_kiss(s1, s2, s3, s4);                              // 3720
        const float x = cf[c * nl + k]; frac = x < 1.e-20f ? 0.f : x;                           // 3721
        if (icld == 2 && k > 0) { if (prev > FSUB(1.f, prev_cf)) r = prev; else r = FMUL(r, FSUB(1.f, prev_cf)); }  // 3722
        if (icld == 4 || icld == 5) { float r2 = mpas_kiss(s1, s2, s3, s4);                    // 3723
            if (k > 0) { float alpha = weights[c * nl + k]; if (icld == 5 && frac == 0.f && prev_cf > 0.f) alpha = 0.f;
                         if (r2 < alpha) r = prev; } }                                          // 3724
        const float thr = FSUB(1.f, frac);
        const bool cloudy = r >= thr;                                                           // 3725
        margin = FSUB(r, thr);   // harness only (sign and size of the decision's distance to its threshold)
        prev = r; prev_cf = frac;                                                               // 3727
        return cloudy;
    }
};

__device__ __forceinline__ bool msg_bad_args(int nl, int icld) { return nl < 1 || nl > MSG_NG || icld < 1 || icld > 5; }
__device__ __forceinline__ long long msg_o(int f, int nc, long long c, int nl, int k, int g) {
    return (((long long)f * nc + c) * nl + k) * MSG_NG + g; }

// Harness only (not in launch_abi): the forward slabs 0..3 from the shared walk, written into out in place.
extern "C" __global__ void mpas_sw_mcica_gpoint__check(int nc, int nl, int icld, int draws, const float* cf,
    const float* iw, const float* lw, const float* sw, const unsigned int* seeds, const float* weights,
    const unsigned int* xorshift, float* out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MSG_NG) return;
    const int g = (int)(t % MSG_NG); const long long c = t / MSG_NG;
    MsgStream s; s.init(nl, icld, draws, cf, seeds, weights, xorshift, c, g);
    for (int k = 0; k < nl; ++k) {
        bool cloudy = s.step(k);
        out[msg_o(0, nc, c, nl, k, g)] = cloudy ? 1.f : 0.f;
        out[msg_o(1, nc, c, nl, k, g)] = cloudy ? iw[c * nl + k] : 0.f;
        out[msg_o(2, nc, c, nl, k, g)] = cloudy ? lw[c * nl + k] : 0.f;
        out[msg_o(3, nc, c, nl, k, g)] = cloudy ? sw[c * nl + k] : 0.f;
    }
}

// Harness only: the signed threshold margin r - (1 - frac) of every decision (range hazard 2), [nc, nl, 112].
extern "C" __global__ void mpas_sw_mcica_gpoint__margin(int nc, int nl, int icld, int draws, const float* cf,
    const unsigned int* seeds, const float* weights, const unsigned int* xorshift, float* margin) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MSG_NG) return;
    const int g = (int)(t % MSG_NG); const long long c = t / MSG_NG;
    MsgStream s; s.init(nl, icld, draws, cf, seeds, weights, xorshift, c, g);
    for (int k = 0; k < nl; ++k) { s.step(k); margin[(c * nl + k) * MSG_NG + g] = s.margin; }
}

// TL: d_out0 = 0, d_out{1,2,3} = mask * d_{iw,lw,sw} (masked copy, bitwise); d_out slabs 4..7 untouched.
extern "C" __global__ void mpas_sw_mcica_gpoint__tl(int nc, int nl, int icld, int draws, const float* cf,
    const float* iw, const float* lw, const float* sw, const unsigned int* seeds, const float* weights,
    const unsigned int* xorshift, const float* d_iw, const float* d_lw, const float* d_sw, float* d_out, int* failure) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MSG_NG) return;
    if (msg_bad_args(nl, icld)) { atomicExch(failure + 1, 1); return; }
    const int g = (int)(t % MSG_NG); const long long c = t / MSG_NG;
    MsgStream s; s.init(nl, icld, draws, cf, seeds, weights, xorshift, c, g);
    bool bad = false;
    for (int k = 0; k < nl; ++k) {
        const bool cloudy = s.step(k);
        bad |= !isfinite(cf[c * nl + k]);
        const float a = cloudy ? d_iw[c * nl + k] : 0.f, b = cloudy ? d_lw[c * nl + k] : 0.f, w = cloudy ? d_sw[c * nl + k] : 0.f;
        bad |= !isfinite(a) || !isfinite(b) || !isfinite(w);
        d_out[msg_o(0, nc, c, nl, k, g)] = 0.f; d_out[msg_o(1, nc, c, nl, k, g)] = a;
        d_out[msg_o(2, nc, c, nl, k, g)] = b;   d_out[msg_o(3, nc, c, nl, k, g)] = w;
    }
    if (bad) atomicExch(failure, 1);
}

// AD: thread (c, k): b_{iw,lw,sw}(c,k) += sum_g mask(c,k,g) * b_out{1,2,3}(c,k,g) (double, g ascending, one f32
// rounding); b_out slabs 0..3 of layer k consumed (zeroed); b_out slabs 4..7 untouched.
extern "C" __global__ void mpas_sw_mcica_gpoint__ad(int nc, int nl, int icld, int draws, const float* cf,
    const float* iw, const float* lw, const float* sw, const unsigned int* seeds, const float* weights,
    const unsigned int* xorshift, float* b_iw, float* b_lw, float* b_sw, float* b_out, int* failure) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (long long)nc * MSG_NG) return;
    if (msg_bad_args(nl, icld)) { atomicExch(failure + 1, 1); return; }
    const long long c = t / MSG_NG; const int k = (int)(t % MSG_NG);
    if (k >= nl) return;
    double si = 0.0, sl = 0.0, ss = 0.0; bool bad = !isfinite(cf[c * nl + k]);
    for (int g = 0; g < MSG_NG; ++g) {
        MsgStream s; s.init(nl, icld, draws, cf, seeds, weights, xorshift, c, g);
        bool cloudy = false;
        for (int kk = 0; kk <= k; ++kk) cloudy = s.step(kk);
        const long long o0 = msg_o(0, nc, c, nl, k, g), o1 = msg_o(1, nc, c, nl, k, g),
                        o2 = msg_o(2, nc, c, nl, k, g), o3 = msg_o(3, nc, c, nl, k, g);
        if (cloudy) { si += (double)b_out[o1]; sl += (double)b_out[o2]; ss += (double)b_out[o3]; }
        b_out[o0] = 0.f; b_out[o1] = 0.f; b_out[o2] = 0.f; b_out[o3] = 0.f;
    }
    const long long i = c * nl + k;
    const float oi = (float)((double)b_iw[i] + si), ol = (float)((double)b_lw[i] + sl), os = (float)((double)b_sw[i] + ss);
    bad |= !isfinite(oi) || !isfinite(ol) || !isfinite(os);
    b_iw[i] = oi; b_lw[i] = ol; b_sw[i] = os;
    if (bad) atomicExch(failure, 1);
}
