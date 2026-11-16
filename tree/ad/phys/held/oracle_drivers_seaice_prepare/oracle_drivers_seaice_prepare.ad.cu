// SPDX-License-Identifier: BSD-3-Clause
// Private autodiff program, held lane ad/held-oracle_drivers_seaice_prepare.
// Tangent and adjoint of the RUC sea-ice prepare launch `oracle_drivers_seaice_prepare` (hex pin 16251589,
// hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5001-5015 body, :5386-5397 wrapper; RRFS-MPAS 1d6e370e
// mpas_atmphys_driver_seaice.F:436-442), written from the forward CUDA source.
//
// Forward, per column c (one thread) and cell i < ncells (k = c + i*nc; ncells is 1 on every RUC route because
// each view holds nc words, ruc.rs address()):
//   if (xice >= xice_threshold && xice < 1)  alb_p = (alb - 0.08 (1-xice)) / xice,  em_p = (em - 0.98 (1-xice)) / xice
//   else                                     alb_p = alb,                           em_p = em
// Active inputs: xice (slot 2), sfc_albedo (3), sfc_emiss (4). Outputs, fully overwritten: sfc_albedo_p (5),
// sfc_emiss_p (6). Inactive: ncells (0, integer), xice_threshold (1, a predicate operand only).
//
// The arm is a switch, not a clamp: the derivative is the executed arm's, chosen by the forward's own pre-launch
// xice and threshold (no frozen coefficient is needed; the arm is the only non-smooth element). In the active arm,
//   dA = (da + (0.08 - A) dx) / x,   dE = (de + (0.98 - E) dx) / x,   A = na / x, na = a - 0.08 (1-x)  (E alike),
// with A and E recomputed from the snapshot inputs in the forward's own rounding (the pre-launch words of the
// outputs are stale). Float32 throughout, explicit round-to-nearest intrinsics, built with --fmad=false --ftz=false
// (the live schemes' rules).
//
// Binding (spec oracle_drivers_seaice_prepare.json, ad.entry/v1): `p` is the primal pointer table with every target
// view copied pre-launch into the tape (indirect.primal_snapshot); `d` / `b` is the mirror of the forward table onto
// the sweep shadows (indirect.mirror). Overlap checks use the mirror (same allocation and offset gives the same
// shadow word), because the snapshot copies are distinct by construction.
//
// Failure slots (one int32 word each, nonzero = fired; source ids in the spec):
//   0 ncells outside [0,1]   1 an output view overlaps an input or the other output   2 x == 0 in the active arm
//   3 nonfinite active-arm primal or result   4 nonfinite active-arm tangent or seed
// Branch record (exact integer counts, slot s = [false, true]): 0 active arm, 1 x >= threshold, 2 x < 1 given 1.

__device__ __forceinline__ bool sip_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ const float *sip_cf(const unsigned long long *p, int s) {
    return reinterpret_cast<const float *>(p[s]);
}
__device__ __forceinline__ float *sip_f(const unsigned long long *p, int s) {
    return reinterpret_cast<float *>(p[s]);
}
__device__ __forceinline__ void sip_fail(unsigned int *fail, int slot, unsigned int source) {
    if (fail[slot] == 0u) fail[slot] = source;
}
// Views are nc*ncells float32 words; true when [u, u+n) and [v, v+n) overlap (both nonnull).
__device__ __forceinline__ bool sip_overlap(unsigned long long u, unsigned long long v, unsigned long long n) {
    return u != 0ull && v != 0ull && u < v + n && v < u + n;
}
// Thread 0 checks the derivative table once: outputs 5 and 6 must not overlap each other or inputs 2-4.
__device__ __forceinline__ bool sip_alias_ok(const unsigned long long *m, int nc, int ncells) {
    const unsigned long long n = 4ull * (unsigned long long)nc * (unsigned long long)(ncells > 0 ? ncells : 1);
    for (int o = 5; o <= 6; ++o) {
        for (int s = 2; s <= 6; ++s) {
            if (s == o) continue;
            if (sip_overlap(m[o], m[s], n)) return false;
        }
    }
    return true;
}

// Active-arm primal of one output: v = (in - k (1-x)) / x, in the forward's rounding.
__device__ __forceinline__ float sip_out(float in, float k, float x) {
    return __fdiv_rn(__fsub_rn(in, __fmul_rn(k, __fsub_rn(1.0f, x))), x);
}

extern "C" __global__ void oracle_drivers_seaice_prepare__tl(
    const unsigned long long *p, const unsigned long long *d, int nc, unsigned int *fail, unsigned int *branches) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int ncells = reinterpret_cast<const int *>(p[0])[c];
    if (ncells < 0 || ncells > 1) { sip_fail(fail, 0, 5006u); return; }
    if (c == 0 && !sip_alias_ok(d, nc, ncells)) sip_fail(fail, 1, 5007u);
    const float threshold = sip_cf(p, 1)[c];
    for (int i = 0; i < ncells; ++i) {
        const int k = c + i * nc;
        const float x = sip_cf(p, 2)[k];
        const float dx = sip_cf(d, 2)[k], da = sip_cf(d, 3)[k], de = sip_cf(d, 4)[k];
        const bool left = x >= threshold, right = left && x < 1.0f;
        atomicAdd(branches + 0 + int(right), 1u);
        atomicAdd(branches + 2 + int(left), 1u);
        if (left) atomicAdd(branches + 4 + int(x < 1.0f), 1u);
        float va = da, ve = de;
        if (right) {
            if (x == 0.0f) { sip_fail(fail, 2, 5008u); return; }
            const float a = sip_cf(p, 3)[k], e = sip_cf(p, 4)[k];
            const float A = sip_out(a, 0.08f, x), E = sip_out(e, 0.98f, x);
            if (!sip_finite(a) || !sip_finite(e) || !sip_finite(A) || !sip_finite(E)) { sip_fail(fail, 3, 5009u); return; }
            if (!sip_finite(dx) || !sip_finite(da) || !sip_finite(de)) { sip_fail(fail, 4, 5010u); return; }
            va = __fdiv_rn(__fadd_rn(da, __fmul_rn(__fsub_rn(0.08f, A), dx)), x);
            ve = __fdiv_rn(__fadd_rn(de, __fmul_rn(__fsub_rn(0.98f, E), dx)), x);
            if (!sip_finite(va) || !sip_finite(ve)) { sip_fail(fail, 3, 5011u); return; }
        }
        sip_f(d, 5)[k] = va;
        sip_f(d, 6)[k] = ve;
    }
}

// Transpose: the outputs' cotangents are consumed and cleared (both are fully overwritten by the forward); inputs
// accumulate. Read-modify-write one input at a time, so inputs that alias each other still accumulate exactly.
extern "C" __global__ void oracle_drivers_seaice_prepare__ad(
    const unsigned long long *p, const unsigned long long *b, int nc, unsigned int *fail) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int ncells = reinterpret_cast<const int *>(p[0])[c];
    if (ncells < 0 || ncells > 1) { sip_fail(fail, 0, 5006u); return; }
    if (c == 0 && !sip_alias_ok(b, nc, ncells)) sip_fail(fail, 1, 5007u);
    const float threshold = sip_cf(p, 1)[c];
    for (int i = ncells - 1; i >= 0; --i) {
        const int k = c + i * nc;
        const float x = sip_cf(p, 2)[k];
        const float bA = sip_cf(b, 5)[k], bE = sip_cf(b, 6)[k];
        sip_f(b, 5)[k] = 0.0f;
        sip_f(b, 6)[k] = 0.0f;
        if (x >= threshold && x < 1.0f) {
            if (x == 0.0f) { sip_fail(fail, 2, 5008u); return; }
            const float a = sip_cf(p, 3)[k], e = sip_cf(p, 4)[k];
            const float A = sip_out(a, 0.08f, x), E = sip_out(e, 0.98f, x);
            if (!sip_finite(a) || !sip_finite(e) || !sip_finite(A) || !sip_finite(E)) { sip_fail(fail, 3, 5009u); return; }
            if (!sip_finite(bA) || !sip_finite(bE)) { sip_fail(fail, 4, 5010u); return; }
            const float tA = __fdiv_rn(bA, x), tE = __fdiv_rn(bE, x);
            const float gx = __fadd_rn(__fmul_rn(__fsub_rn(0.08f, A), tA), __fmul_rn(__fsub_rn(0.98f, E), tE));
            if (!sip_finite(tA) || !sip_finite(tE) || !sip_finite(gx)) { sip_fail(fail, 3, 5011u); return; }
            float *ba = sip_f(b, 3), *be = sip_f(b, 4), *bx = sip_f(b, 2);
            ba[k] = __fadd_rn(ba[k], tA);
            be[k] = __fadd_rn(be[k], tE);
            bx[k] = __fadd_rn(bx[k], gx);
        } else {
            float *ba = sip_f(b, 3), *be = sip_f(b, 4);
            ba[k] = __fadd_rn(ba[k], bA);
            be[k] = __fadd_rn(be[k], bE);
        }
    }
}
