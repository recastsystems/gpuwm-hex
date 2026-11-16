// SPDX-License-Identifier: Apache-2.0
// g3-radland: ozone_time (radshared_v842.py:331, captured .cu 9b6ced0f): out = f1 monthly[prev] + f2 monthly[next],
// f1/f2 the CAM cyclic calendar weights of julian (a host constant). Linear in monthly; the weights are recomputed
// exactly as the forward does (float64 calendar, then the forward's (float) casts). Refusal slot 0: out aliasing monthly.
#include "radshared.cu"
#include "g3_common.cuh"
__device__ __forceinline__ void g3_oz_weights(float julian, int& prev, int& next, float& w1, float& w2) {
    const int dates[12] = {16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350};
    double day = (double)FADD(julian, 1.f);
    int iday = (int)day;
    day = DSUB(day, (double)iday); iday = iday % 365; if (iday == 0) iday = 365;
    day = DADD(day, (double)iday);
    next = 0;
    for (int m = 0; m < 12; ++m) if ((double)dates[m] > day) { next = m; break; }
    prev = next > 0 ? next - 1 : 11;
    double plus = dates[next], minus = dates[prev], f1, f2;
    if (next == 0) {
        double delta = DSUB(DADD(plus, 365.), minus);
        if (day > plus) { f1 = DDIV(DSUB(DADD(plus, 365.), day), delta); f2 = DDIV(DSUB(day, minus), delta); }
        else { f1 = DDIV(DSUB(plus, day), delta); f2 = DDIV(DSUB(DADD(day, 365.), minus), delta); }
    } else {
        double delta = DSUB(plus, minus);
        f1 = DDIV(DSUB(plus, day), delta); f2 = DDIV(DSUB(day, minus), delta);
    }
    w1 = (float)f1; w2 = (float)f2;
}
extern "C" __global__ void ozone_time__tl(int n, float julian, const float* d_m, float* d_out, int* failure) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
    if (d_m == d_out) g3_fail(failure, 0);
    int p, q; float w1, w2; g3_oz_weights(julian, p, q, w1, w2);
    d_out[idx] = g3_st(G3D(w1) * G3D(d_m[(size_t)idx * 12 + p]) + G3D(w2) * G3D(d_m[(size_t)idx * 12 + q]));
}
extern "C" __global__ void ozone_time__ad(int n, float julian, float* b_m, float* b_out, int* failure) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
    if (b_m == b_out) { g3_fail(failure, 0); return; }
    int p, q; float w1, w2; g3_oz_weights(julian, p, q, w1, w2);
    double g = b_out[idx];
    g3_acc(b_m + (size_t)idx * 12 + p, G3D(w1) * g); g3_acc(b_m + (size_t)idx * 12 + q, G3D(w2) * g);
    b_out[idx] = 0.0f;
}
