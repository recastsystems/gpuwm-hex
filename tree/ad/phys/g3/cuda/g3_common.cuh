// SPDX-License-Identifier: Apache-2.0
// Lane g3-radland: shared helpers of the hand ad.entry/v1 TL/AD rows for the RRTMG SW tail,
// the SW setup tables and the radiation-shared kernels. Derivative coefficients and sums are float64, computed
// from the float32 primal words the forward reads (or recomputes in the forward's own order); each stored
// derivative word is rounded once to float32, with +0.0 added so a zero direction gives +0 bits.
#pragma once
#define G3D(x) ((double)(x))
__device__ __forceinline__ float g3_st(double v) { return __fadd_rn((float)v, 0.0f); }
// AD accumulate: one rounding of (old + contribution).
__device__ __forceinline__ void g3_acc(float* p, double v) { *p = g3_st(G3D(*p) + v); }
__device__ __forceinline__ void g3_fail(int* f, int slot) { atomicExch(f + slot, 1); }
