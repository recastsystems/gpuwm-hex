// SPDX-License-Identifier: Apache-2.0
// ad/phys P4: exact executed-branch tangent and reverse types for float32 column physics.
//
// One header, two modes, chosen at compile time:
//   P4_MODE_TL   ADF = {float v; double t}: v is computed by exactly the forward's float
//                operation on the forward's float operands (bitwise primal); t is the
//                tangent in double from the partials below.
//   P4_MODE_REV  ADF = {float v; int i}: same v; i is a node in a per-thread Wengert tape
//                (parents + double partials). The reverse sweep is the exact transpose of
//                the TL, because both use the same partial functions below.
// Branch policy (PHYS-MAP section 3): every comparison, min/max, abs and table index is
// taken on v, exactly as the forward took it; ties in min/max take the left operand.
// Optional branch record: one bit per comparison/selection, per column, for flip counts.
#pragma once

#ifndef P4_MAX_BLOCK
#define P4_MAX_BLOCK 128
#endif

struct P4Ctx {
    // branch record (both modes): bits interleaved [word][thread]
    unsigned *br; int br_words; int br_n; long long br_stride; long long gslot;
    // reverse tape (REV): node k of this thread at k*stride + gslot
    int2 *ab; double2 *w; int cnt; int cap; long long stride; int overflow;
    // counters of declared singular partials (sqrt at 0, log at 0 ...)
    int singular;
};
__shared__ P4Ctx p4ctx[P4_MAX_BLOCK];
__device__ __forceinline__ P4Ctx &P4C() { return p4ctx[threadIdx.x]; }

__device__ __forceinline__ void p4_branch(bool b) {
    P4Ctx &c = P4C();
    if (c.br) {
        int k = c.br_n;
        if ((k >> 5) < c.br_words) {
            unsigned *wp = c.br + (long long)(k >> 5) * c.br_stride + c.gslot;
            if (b) *wp |= (1u << (k & 31));
        }
    }
    c.br_n++;
}

// integer results of float values (table cells, directions) enter the record by parity
__device__ __forceinline__ void p4_branch_int(int r) { p4_branch((r & 1) != 0); }

#if defined(P4_MODE_TL)
struct ADF {
    float v; double t;
    __device__ __forceinline__ ADF() : v(0.0f), t(0.0) {}
    __device__ __forceinline__ ADF(float x) : v(x), t(0.0) {}
    __device__ __forceinline__ ADF(int x) : v((float)x), t(0.0) {}
    __device__ __forceinline__ ADF(double x) : v((float)x), t(0.0) {}
    __device__ __forceinline__ ADF(float x, double d) : v(x), t(d) {}
    __device__ __forceinline__ explicit operator int() const { int r = (int)v; p4_branch_int(r); return r; }
    __device__ __forceinline__ explicit operator double() const { return (double)v; }
    __device__ __forceinline__ explicit operator bool() const { return v != 0.0f; }
};
__device__ __forceinline__ ADF p4_u(float r, const ADF &a, double wa) { return ADF(r, wa * a.t); }
__device__ __forceinline__ ADF p4_b(float r, const ADF &a, double wa, const ADF &b, double wb) {
    return ADF(r, wa * a.t + wb * b.t);
}
__device__ __forceinline__ bool p4_active(const ADF &a) { return a.t != 0.0; }
#elif defined(P4_MODE_REV)
struct ADF {
    float v; int i;
    __device__ __forceinline__ ADF() : v(0.0f), i(-1) {}
    __device__ __forceinline__ ADF(float x) : v(x), i(-1) {}
    __device__ __forceinline__ ADF(int x) : v((float)x), i(-1) {}
    __device__ __forceinline__ ADF(double x) : v((float)x), i(-1) {}
    __device__ __forceinline__ ADF(float x, int node) : v(x), i(node) {}
    __device__ __forceinline__ explicit operator int() const { int r = (int)v; p4_branch_int(r); return r; }
    __device__ __forceinline__ explicit operator double() const { return (double)v; }
    __device__ __forceinline__ explicit operator bool() const { return v != 0.0f; }
};
// a = parent a, b = parent b (or -1); leaf nodes use a = -2 - array id, w.x = element index
__device__ __forceinline__ int p4_push(int a, int b, double wa, double wb) {
    P4Ctx &c = P4C();
    int k = c.cnt++;
    if (c.ab) {
        if (k < c.cap) {
            long long o = (long long)k * c.stride + c.gslot;
            c.ab[o] = make_int2(a, b); c.w[o] = make_double2(wa, wb);
        } else c.overflow = 1;
    }
    return k;
}
__device__ __forceinline__ ADF p4_u(float r, const ADF &a, double wa) {
    if (a.i < 0) return ADF(r);
    return ADF(r, p4_push(a.i, -1, wa, 0.0));
}
__device__ __forceinline__ ADF p4_b(float r, const ADF &a, double wa, const ADF &b, double wb) {
    if (a.i < 0 && b.i < 0) return ADF(r);
    if (b.i < 0) return ADF(r, p4_push(a.i, -1, wa, 0.0));
    if (a.i < 0) return ADF(r, p4_push(b.i, -1, wb, 0.0));
    return ADF(r, p4_push(a.i, b.i, wa, wb));
}
__device__ __forceinline__ ADF p4_leaf(float v, int array_id, long long index) {
    return ADF(v, p4_push(-2 - array_id, -1, (double)index, 0.0));
}
__device__ __forceinline__ bool p4_active(const ADF &a) { return a.i >= 0; }
#else
#error "define P4_MODE_TL or P4_MODE_REV"
#endif

// ---- partials, shared by both modes (double, from the float operands) ----
__device__ __forceinline__ ADF p4_add(float r, const ADF &a, const ADF &b) { return p4_b(r, a, 1.0, b, 1.0); }
__device__ __forceinline__ ADF p4_sub(float r, const ADF &a, const ADF &b) { return p4_b(r, a, 1.0, b, -1.0); }
__device__ __forceinline__ ADF p4_mul(float r, const ADF &a, const ADF &b) {
    return p4_b(r, a, (double)b.v, b, (double)a.v);
}
__device__ __forceinline__ ADF p4_div(float r, const ADF &a, const ADF &b) {
    double bd = (double)b.v;
    return p4_b(r, a, 1.0 / bd, b, -(double)a.v / (bd * bd));
}

// ---- operators: value by the same float op the forward used (--fmad=false build) ----
__device__ __forceinline__ ADF operator+(const ADF &a, const ADF &b) { return p4_add(a.v + b.v, a, b); }
__device__ __forceinline__ ADF operator-(const ADF &a, const ADF &b) { return p4_sub(a.v - b.v, a, b); }
__device__ __forceinline__ ADF operator*(const ADF &a, const ADF &b) { return p4_mul(a.v * b.v, a, b); }
__device__ __forceinline__ ADF operator/(const ADF &a, const ADF &b) { return p4_div(a.v / b.v, a, b); }
__device__ __forceinline__ ADF operator-(const ADF &a) { return p4_u(-a.v, a, -1.0); }
__device__ __forceinline__ ADF operator+(const ADF &a) { return a; }
__device__ __forceinline__ ADF &operator+=(ADF &a, const ADF &b) { a = a + b; return a; }
__device__ __forceinline__ ADF &operator-=(ADF &a, const ADF &b) { a = a - b; return a; }
__device__ __forceinline__ ADF &operator*=(ADF &a, const ADF &b) { a = a * b; return a; }
__device__ __forceinline__ ADF &operator/=(ADF &a, const ADF &b) { a = a / b; return a; }

#define P4_CMP(op) \
    __device__ __forceinline__ bool operator op(const ADF &a, const ADF &b) { bool r = a.v op b.v; p4_branch(r); return r; }
P4_CMP(<) P4_CMP(<=) P4_CMP(>) P4_CMP(>=) P4_CMP(==) P4_CMP(!=)
#undef P4_CMP

// ---- CUDA rounding intrinsics ----
__device__ __forceinline__ ADF __fadd_rn(const ADF &a, const ADF &b) { return p4_add(__fadd_rn(a.v, b.v), a, b); }
__device__ __forceinline__ ADF __fsub_rn(const ADF &a, const ADF &b) { return p4_sub(__fsub_rn(a.v, b.v), a, b); }
__device__ __forceinline__ ADF __fmul_rn(const ADF &a, const ADF &b) { return p4_mul(__fmul_rn(a.v, b.v), a, b); }
__device__ __forceinline__ ADF __fdiv_rn(const ADF &a, const ADF &b) { return p4_div(__fdiv_rn(a.v, b.v), a, b); }
__device__ __forceinline__ ADF p4_sqrt_of(float r, const ADF &a) {
    if (r == 0.0f) { if (p4_active(a)) P4C().singular++; return p4_u(r, a, 0.0); }  // declared: d sqrt at 0 := 0
    return p4_u(r, a, 0.5 / (double)r);
}
__device__ __forceinline__ ADF __fsqrt_rn(const ADF &a) { return p4_sqrt_of(__fsqrt_rn(a.v), a); }

// ---- selections (executed arm, tie -> left, NaN -> the non-NaN operand as fmaxf does) ----
__device__ __forceinline__ ADF fmaxf(const ADF &a, const ADF &b) {
    float r = fmaxf(a.v, b.v);
    bool left = isnan(b.v) || (!isnan(a.v) && a.v >= b.v);
    p4_branch(left);
    return left ? p4_u(r, a, 1.0) : p4_u(r, b, 1.0);
}
__device__ __forceinline__ ADF fminf(const ADF &a, const ADF &b) {
    float r = fminf(a.v, b.v);
    bool left = isnan(b.v) || (!isnan(a.v) && a.v <= b.v);
    p4_branch(left);
    return left ? p4_u(r, a, 1.0) : p4_u(r, b, 1.0);
}
__device__ __forceinline__ ADF fabsf(const ADF &a) {
    bool neg = signbit(a.v);
    p4_branch(neg);
    return p4_u(fabsf(a.v), a, neg ? -1.0 : 1.0);
}
__device__ __forceinline__ float roundf(const ADF &a) { float r = roundf(a.v); p4_branch_int((int)r); return r; }   // integer-valued: no tangent
__device__ __forceinline__ float floorf(const ADF &a) { float r = floorf(a.v); p4_branch_int((int)r); return r; }

// ---- libm wrappers: value from the forward's own float routine, partial from the true function ----
__device__ __forceinline__ ADF p4_log_of(float r, const ADF &x) {
    if (x.v == 0.0f) { if (p4_active(x)) P4C().singular++; return p4_u(r, x, 0.0); }
    return p4_u(r, x, 1.0 / (double)x.v);
}
__device__ __forceinline__ ADF p4_exp_of(float r, const ADF &x) { return p4_u(r, x, exp((double)x.v)); }
__device__ __forceinline__ ADF p4_pow_of(float r, const ADF &x, const ADF &y) {
    double xd = (double)x.v, yd = (double)y.v;
    double wx = 0.0, wy = 0.0;
    if (p4_active(x)) {
        if (xd == 0.0) { wx = (yd == 1.0) ? 1.0 : 0.0; if (yd < 1.0 && yd != 0.0) P4C().singular++; }
        else wx = yd * pow(xd, yd - 1.0);
    }
    if (p4_active(y)) {
        if (xd > 0.0) wy = pow(xd, yd) * log(xd);
        else P4C().singular++;
    }
    return p4_b(r, x, wx, y, wy);
}
__device__ __forceinline__ ADF p4_atan_of(float r, const ADF &x) {
    double xd = (double)x.v; return p4_u(r, x, 1.0 / (1.0 + xd * xd));
}
__device__ __forceinline__ ADF p4_sin_of(float r, const ADF &x) { return p4_u(r, x, cos((double)x.v)); }
