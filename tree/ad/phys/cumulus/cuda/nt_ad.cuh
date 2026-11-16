// SPDX-License-Identifier: Apache-2.0
// Scalar carriers for the New Tiedtke (cu_ntiedtke, MPAS 44c0849 / MMM a4baf7f3) derivative entries.
// Included AFTER the forward source (nt-forward.cu), so every
// primal word below is produced by the forward's own function (mpas_nt_add/sub/mul/div/sqrt/neg, gfk_exp/log/pow,
// fmaxf/fminf/fabsf/sqrtf): the derivative entries' trajectory is the forward's, bit for bit.
//
//   AF: tangent.        v = the forward's f32 word; d = tangent in double.
//   AR: taped reverse.  v = the forward's f32 word; id = tape entry (>= 0), -1 constant, <= -2 input word slot.
//
// Precision (the live-scheme rule): every local partial is formed in double from the f32 primal words; tangents and
// adjoints accumulate in double and are rounded once when they are written out.
// Branch rule (frozen at the recorded path): every comparison, integer conversion (mpas_nt_int: the INTEGER zcbase),
// min/max choice and fabs sign is decided on the primal words exactly as the forward decides it; only the executed arm
// is differentiated. Integer state (ktype, ldcum, loddraf, kcbot, kctop, kdpl, idtop, klab, llo3) is never a carrier,
// so the linear model is the derivative along the recorded branch path and integer counts.
#pragma once

#ifndef NT_PLANT
#define NT_PLANT 0      // planted errors for the proofs: 1 = AR product partial, 2 = AF product tangent (both 1e-3 off)
#endif

struct AF {
  float v; double d;
  __device__ AF() : v(0.f), d(0.0) {}
  __device__ AF(float x) : v(x), d(0.0) {}
  __device__ AF(float x, double t) : v(x), d(t) {}
  __device__ explicit operator int() const { return (int)v; }
  __device__ explicit operator float() const { return v; }
};

// ---- the per-thread tape (one thread = one column), laid out [entry][W] like the forward's level-major arrays ----
struct NtTapeE { int a, b; double pa, pb; };
struct NtTape {
  NtTapeE *tape;   // [cap][W]
  double *adj;     // [cap][W]
  double *adjin;   // [nslot][W]
  int *cnt;        // [W]
  int *ovf;        // [W] per-thread overflow flag
  int cap, W;
};
__device__ NtTape g_nt;   // set by every thread of an AD launch to the same words (as the RUC land entries do)
__device__ __forceinline__ int nt_push(int a, double pa, int b, double pb) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = g_nt.cnt[t];
  g_nt.cnt[t] = n + 1;                    // counts past the capacity too: a cap-0 pass is the tape census
  if (n >= g_nt.cap) { g_nt.ovf[t] = 1; return n; }  // an untaped but ACTIVE id (>= cap): the census keeps counting what follows
  NtTapeE e; e.a = a; e.b = b; e.pa = pa; e.pb = pb;
  g_nt.tape[(size_t)n * g_nt.W + t] = e;
  return n;
}
struct AR {
  float v; int id;
  __device__ AR() : v(0.f), id(-1) {}
  __device__ AR(float x) : v(x), id(-1) {}
  __device__ AR(float x, int i) : v(x), id(i) {}
  __device__ explicit operator int() const { return (int)v; }
  __device__ explicit operator float() const { return v; }
};
__device__ __forceinline__ AR ar_un(float y, const AR &a, double pa) {
  return AR(y, a.id == -1 ? -1 : nt_push(a.id, pa, -1, 0.0));
}
__device__ __forceinline__ AR ar_bin(float y, const AR &a, double pa, const AR &b, double pb) {
  if (a.id == -1 && b.id == -1) return AR(y, -1);
  if (b.id == -1) return AR(y, nt_push(a.id, pa, -1, 0.0));
  if (a.id == -1) return AR(y, nt_push(b.id, pb, -1, 0.0));
  return AR(y, nt_push(a.id, pa, b.id, pb));
}

// ---- comparisons: primal words only (the recorded path) ----
#define NT_CMP(T) \
  __device__ __forceinline__ bool operator<(const T &a, const T &b) { return a.v < b.v; } \
  __device__ __forceinline__ bool operator>(const T &a, const T &b) { return a.v > b.v; } \
  __device__ __forceinline__ bool operator<=(const T &a, const T &b) { return a.v <= b.v; } \
  __device__ __forceinline__ bool operator>=(const T &a, const T &b) { return a.v >= b.v; } \
  __device__ __forceinline__ bool operator==(const T &a, const T &b) { return a.v == b.v; } \
  __device__ __forceinline__ bool operator!=(const T &a, const T &b) { return a.v != b.v; }
NT_CMP(AF) NT_CMP(AR)

// ---- plain f32 forms (constants): the forward's own functions ----
__device__ __forceinline__ float r_add(float a, float b) { return mpas_nt_add(a, b); }
__device__ __forceinline__ float r_sub(float a, float b) { return mpas_nt_sub(a, b); }
__device__ __forceinline__ float r_mul(float a, float b) { return mpas_nt_mul(a, b); }
__device__ __forceinline__ float r_div(float a, float b) { return mpas_nt_div(a, b); }
__device__ __forceinline__ float r_sqrt(float a) { return mpas_nt_sqrt(a); }
__device__ __forceinline__ float r_sqrtf(float a) { return sqrtf(a); }
__device__ __forceinline__ float r_neg(float a) { return mpas_nt_neg(a); }
__device__ __forceinline__ float r_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float r_min(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float r_fabs(float a) { return fabsf(a); }
__device__ __forceinline__ float r_exp(float a) { return gfk_exp(a); }
__device__ __forceinline__ float r_log(float a) { return gfk_log(a); }
__device__ __forceinline__ float r_pow(float a, float b) { return gfk_pow(a, b); }
__device__ __forceinline__ int r_ntint(float a) { return mpas_nt_int(a); }
__device__ __forceinline__ int r_ntint(const AF &a) { return mpas_nt_int(a.v); }
__device__ __forceinline__ int r_ntint(const AR &a) { return mpas_nt_int(a.v); }
__device__ __forceinline__ bool r_nan(float a) { return mpas_nt_nan(a); }
__device__ __forceinline__ bool r_nan(const AF &a) { return mpas_nt_nan(a.v); }
__device__ __forceinline__ bool r_nan(const AR &a) { return mpas_nt_nan(a.v); }

// ---- local partials in double from the f32 words ----
__device__ __forceinline__ double nt_dsqrt(float a) { return a > 0.f ? 0.5 / sqrt((double)a) : 0.0; }
__device__ __forceinline__ double nt_dpow_a(float a, float b) {
  const double A = a, B = b;
  if (a == 0.f) return B == 1.0 ? 1.0 : 0.0;
  if (a < 0.f) return (B == floor(B)) ? B * pow(A, B - 1.0) : 0.0;
  return B * pow(A, B - 1.0);
}
__device__ __forceinline__ double nt_dpow_b(float a, float b) {
  if (a <= 0.f) return 0.0;
  return log((double)a) * pow((double)a, (double)b);
}
// max/min: the executed arm. fmaxf/fminf return the non-NaN operand; on equal words the first operand's arm is taken.
__device__ __forceinline__ bool nt_take_first_max(float a, float b) { return (a >= b) || (b != b); }
__device__ __forceinline__ bool nt_take_first_min(float a, float b) { return (a <= b) || (b != b); }

// ---- AF (tangent) ----
#if NT_PLANT == 2
#define NT_PLANT_AF 1.001
#else
#define NT_PLANT_AF 1.0
#endif
__device__ __forceinline__ AF r_add(const AF &a, const AF &b) { return AF(mpas_nt_add(a.v, b.v), a.d + b.d); }
__device__ __forceinline__ AF r_sub(const AF &a, const AF &b) { return AF(mpas_nt_sub(a.v, b.v), a.d - b.d); }
__device__ __forceinline__ AF r_mul(const AF &a, const AF &b) {
  return AF(mpas_nt_mul(a.v, b.v), NT_PLANT_AF * (double)b.v * a.d + (double)a.v * b.d);
}
__device__ __forceinline__ AF r_div(const AF &a, const AF &b) {
  const double B = b.v; return AF(mpas_nt_div(a.v, b.v), a.d / B - ((double)a.v / (B * B)) * b.d);
}
__device__ __forceinline__ AF r_sqrt(const AF &a) { return AF(mpas_nt_sqrt(a.v), nt_dsqrt(a.v) * a.d); }
__device__ __forceinline__ AF r_sqrtf(const AF &a) { return AF(sqrtf(a.v), nt_dsqrt(a.v) * a.d); }
__device__ __forceinline__ AF r_neg(const AF &a) { return AF(mpas_nt_neg(a.v), -a.d); }
__device__ __forceinline__ AF operator-(const AF &a) { return AF(-a.v, -a.d); }
__device__ __forceinline__ AF r_max(const AF &a, const AF &b) {
  return AF(fmaxf(a.v, b.v), nt_take_first_max(a.v, b.v) ? a.d : b.d);
}
__device__ __forceinline__ AF r_min(const AF &a, const AF &b) {
  return AF(fminf(a.v, b.v), nt_take_first_min(a.v, b.v) ? a.d : b.d);
}
__device__ __forceinline__ AF r_fabs(const AF &a) { return AF(fabsf(a.v), a.v < 0.f ? -a.d : a.d); }
__device__ __forceinline__ AF r_exp(const AF &a) { return AF(gfk_exp(a.v), a.d == 0.0 ? 0.0 : exp((double)a.v) * a.d); }
__device__ __forceinline__ AF r_log(const AF &a) { return AF(gfk_log(a.v), a.d == 0.0 ? 0.0 : a.d / (double)a.v); }
__device__ __forceinline__ AF r_pow(const AF &a, const AF &b) {
  double t = 0.0;
  if (a.d != 0.0) t += nt_dpow_a(a.v, b.v) * a.d;
  if (b.d != 0.0) t += nt_dpow_b(a.v, b.v) * b.d;
  return AF(gfk_pow(a.v, b.v), t);
}

// ---- AR (taped reverse) ----
#if NT_PLANT == 1
#define NT_PLANT_AR 1.001
#else
#define NT_PLANT_AR 1.0
#endif
__device__ __forceinline__ AR r_add(const AR &a, const AR &b) { return ar_bin(mpas_nt_add(a.v, b.v), a, 1.0, b, 1.0); }
__device__ __forceinline__ AR r_sub(const AR &a, const AR &b) { return ar_bin(mpas_nt_sub(a.v, b.v), a, 1.0, b, -1.0); }
__device__ __forceinline__ AR r_mul(const AR &a, const AR &b) {
  return ar_bin(mpas_nt_mul(a.v, b.v), a, NT_PLANT_AR * (double)b.v, b, (double)a.v);
}
__device__ __forceinline__ AR r_div(const AR &a, const AR &b) {
  const double B = b.v; return ar_bin(mpas_nt_div(a.v, b.v), a, 1.0 / B, b, -(double)a.v / (B * B));
}
__device__ __forceinline__ AR r_sqrt(const AR &a) { return ar_un(mpas_nt_sqrt(a.v), a, nt_dsqrt(a.v)); }
__device__ __forceinline__ AR r_sqrtf(const AR &a) { return ar_un(sqrtf(a.v), a, nt_dsqrt(a.v)); }
__device__ __forceinline__ AR r_neg(const AR &a) { return ar_un(mpas_nt_neg(a.v), a, -1.0); }
__device__ __forceinline__ AR operator-(const AR &a) { return ar_un(-a.v, a, -1.0); }
__device__ __forceinline__ AR r_max(const AR &a, const AR &b) {
  return AR(fmaxf(a.v, b.v), nt_take_first_max(a.v, b.v) ? a.id : b.id);
}
__device__ __forceinline__ AR r_min(const AR &a, const AR &b) {
  return AR(fminf(a.v, b.v), nt_take_first_min(a.v, b.v) ? a.id : b.id);
}
__device__ __forceinline__ AR r_fabs(const AR &a) { return ar_un(fabsf(a.v), a, a.v < 0.f ? -1.0 : 1.0); }
__device__ __forceinline__ AR r_exp(const AR &a) { return ar_un(gfk_exp(a.v), a, a.id == -1 ? 0.0 : exp((double)a.v)); }
__device__ __forceinline__ AR r_log(const AR &a) { return ar_un(gfk_log(a.v), a, a.id == -1 ? 0.0 : 1.0 / (double)a.v); }
__device__ __forceinline__ AR r_pow(const AR &a, const AR &b) {
  return ar_bin(gfk_pow(a.v, b.v), a, a.id == -1 ? 0.0 : nt_dpow_a(a.v, b.v), b, b.id == -1 ? 0.0 : nt_dpow_b(a.v, b.v));
}

// ---- plain CUDA rounded intrinsics (kernels outside the NaN-preserving scheme text: the driver's forcing fold) ----
__device__ __forceinline__ float r_fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float r_fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float r_fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float r_fdiv(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ AF r_fadd(const AF &a, const AF &b) { return AF(__fadd_rn(a.v, b.v), a.d + b.d); }
__device__ __forceinline__ AF r_fsub(const AF &a, const AF &b) { return AF(__fsub_rn(a.v, b.v), a.d - b.d); }
__device__ __forceinline__ AF r_fmul(const AF &a, const AF &b) { return AF(__fmul_rn(a.v, b.v), NT_PLANT_AF * (double)b.v * a.d + (double)a.v * b.d); }
__device__ __forceinline__ AF r_fdiv(const AF &a, const AF &b) { const double B = b.v; return AF(__fdiv_rn(a.v, b.v), a.d / B - ((double)a.v / (B * B)) * b.d); }
__device__ __forceinline__ AR r_fadd(const AR &a, const AR &b) { return ar_bin(__fadd_rn(a.v, b.v), a, 1.0, b, 1.0); }
__device__ __forceinline__ AR r_fsub(const AR &a, const AR &b) { return ar_bin(__fsub_rn(a.v, b.v), a, 1.0, b, -1.0); }
__device__ __forceinline__ AR r_fmul(const AR &a, const AR &b) { return ar_bin(__fmul_rn(a.v, b.v), a, NT_PLANT_AR * (double)b.v, b, (double)a.v); }
__device__ __forceinline__ AR r_fdiv(const AR &a, const AR &b) { const double B = b.v; return ar_bin(__fdiv_rn(a.v, b.v), a, 1.0 / B, b, -(double)a.v / (B * B)); }

// ---- the reverse sweep of one thread's tape, and the adjoint of a word with tape id `id` ----
__device__ __forceinline__ void nt_acc(int t, int id, double x) {
  if (id >= 0) { if (id < g_nt.cap) g_nt.adj[(size_t)id * g_nt.W + t] += x; }
  else if (id <= -2) g_nt.adjin[(size_t)(-id - 2) * g_nt.W + t] += x;
}
__device__ __forceinline__ void nt_sweep(int t) {
  const int n = g_nt.cnt[t] < g_nt.cap ? g_nt.cnt[t] : g_nt.cap;
  for (int k = n - 1; k >= 0; --k) {
    const double g = g_nt.adj[(size_t)k * g_nt.W + t];
    if (g == 0.0) continue;
    const NtTapeE e = g_nt.tape[(size_t)k * g_nt.W + t];
    if (e.a != -1) nt_acc(t, e.a, e.pa * g);
    if (e.b != -1) nt_acc(t, e.b, e.pb * g);
  }
}
