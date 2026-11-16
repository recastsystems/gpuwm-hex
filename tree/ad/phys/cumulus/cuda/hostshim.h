// SPDX-License-Identifier: Apache-2.0
// Compile the New Tiedtke forward and its derivative entries for the PROCESSORS (g++,
// -ffp-contract=off, SSE: every f32 operation correctly rounded, no FMA contraction, no flush to zero = the strict
// card numerics). One CUDA thread = one column, run one after another. Used only by the processor proofs.
#pragma once
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__ __restrict
#define __constant__
struct nt_dim3 { unsigned x, y, z; };
extern nt_dim3 threadIdx, blockIdx, blockDim, gridDim;
using std::isfinite; using std::isnan;
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fsqrt_rn(float a) { return std::sqrt(a); }
inline float __fmaf_rn(float a, float b, float c) { return std::fmaf(a, b, c); }
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __dsub_rn(double a, double b) { return a - b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __ddiv_rn(double a, double b) { return a / b; }
inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
inline float nt_h_sqrtf(float a) { return std::sqrt(a); }
inline float nt_h_fabsf(float a) { return std::fabs(a); }
// PTX max.f32 / min.f32: the non-NaN operand; of two zeros, max is +0 and min is -0.
inline float nt_h_fmaxf(float a, float b) {
  if (a != a) return b; if (b != b) return a;
  if (a == b) return std::signbit(a) ? b : a;
  return a > b ? a : b;
}
inline float nt_h_fminf(float a, float b) {
  if (a != a) return b; if (b != b) return a;
  if (a == b) return std::signbit(a) ? a : b;
  return a < b ? a : b;
}
#define sqrtf nt_h_sqrtf
#define fabsf nt_h_fabsf
#define fmaxf nt_h_fmaxf
#define fminf nt_h_fminf
inline unsigned __float_as_uint(float x) { unsigned u; std::memcpy(&u, &x, 4); return u; }
inline int __float_as_int(float x) { int u; std::memcpy(&u, &x, 4); return u; }
inline float __uint_as_float(unsigned u) { float x; std::memcpy(&x, &u, 4); return x; }
inline float __int_as_float(int u) { float x; std::memcpy(&x, &u, 4); return x; }
inline long long __double_as_longlong(double x) { long long u; std::memcpy(&u, &x, 8); return u; }
inline double __longlong_as_double(long long u) { double x; std::memcpy(&x, &u, 8); return x; }
inline int __double2hiint(double x) { return (int)(__double_as_longlong(x) >> 32); }
inline int __double2loint(double x) { return (int)(unsigned)(__double_as_longlong(x) & 0xffffffffull); }
inline double __hiloint2double(int hi, int lo) {
  return __longlong_as_double((long long)(((unsigned long long)(unsigned)hi << 32) | (unsigned)lo));
}
inline float __double2float_rn(double x) { return (float)x; }
inline int atomicAdd(int *p, int v) { int o = *p; *p = o + v; return o; }
inline int atomicMax(int *p, int v) { int o = *p; if (v > o) *p = v; return o; }
inline int atomicOr(int *p, int v) { int o = *p; *p = o | v; return o; }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
