// SPDX-License-Identifier: Apache-2.0
// g3-radland: the step-0 radiation setup tables (constant rule). Each kernel writes words that depend on no model
// state: compiled-in constants (mpas_sw_exp_table, mpas_sw_static_tables, mpas_sw_weights), the RRTMG SW k-table
// reduction of host-read coefficients with those weights (mpas_sw_reduce_coefficients), and the ozone climatology
// interpolated to the mesh latitude (ozone_latitude: decoded table plus fixed geometry). The derivative of every word
// they write is 0 both ways: the TL writes +0 over exactly the forward's written words, the AD consumes the same
// words' cotangents (sets them to +0) and adds nothing to any input. Words the forward does not write are untouched.
// Compiled once per kernel with -DG3_KERNEL=<n> so each fatbin carries only its own pair.
#include "g3_common.cuh"
#if G3_KERNEL == 1
// rrtmg_sw_v842.py:4793 (captured .cu 3c499c7b): writes out[0..10000].
#define G3_BODY(out) int i = blockIdx.x * blockDim.x + threadIdx.x; if (i > 10000) return; out[i] = 0.0f;
extern "C" __global__ void mpas_sw_exp_table__tl(float* d_out, int* failure) { G3_BODY(d_out) }
extern "C" __global__ void mpas_sw_exp_table__ad(float* b_out, int* failure) { G3_BODY(b_out) }
#elif G3_KERNEL == 2
// rrtmg_sw_v842.py:4790: writes out[0..6936).
#define G3_BODY(out) int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < 6936) out[i] = 0.0f;
extern "C" __global__ void mpas_sw_static_tables__tl(float* d_out, int* failure) { G3_BODY(d_out) }
extern "C" __global__ void mpas_sw_static_tables__ad(float* b_out, int* failure) { G3_BODY(b_out) }
#elif G3_KERNEL == 3
// rrtmg_sw_v842.py:4772: band b writes rwgt[b*16 + k] for k over the band's g-points (the forward's own loop).
__device__ const int G3_NGS[15] = {0, 6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 112};
__device__ const int G3_NGN[112] = {2,2,2,2,4,4,1,1,1,1,1,2,1,2,1,2,1,2,1,1,1,1,2,2,4,4,1,1,1,1,2,2,4,4,1,1,1,1,1,1,1,1,2,6,1,1,1,1,1,1,1,1,2,6,8,8,2,2,1,1,1,1,1,1,2,4,2,2,2,2,2,2,2,2,1,1,2,2,4,6,1,1,2,2,4,6,1,1,1,1,1,1,4,6,1,1,2,2,4,6,1,1,1,1,2,2,2,2,1,1,1,1};
__device__ void g3_weights(float* rwgt) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= 14) return; int g = 0;
    for (int j = G3_NGS[b]; j < G3_NGS[b + 1]; j++) { int start = g; g += G3_NGN[j]; for (int k = start; k < g; k++) rwgt[b * 16 + k] = 0.0f; }
}
extern "C" __global__ void mpas_sw_weights__tl(float* d_rwgt, int* failure) { g3_weights(d_rwgt); }
extern "C" __global__ void mpas_sw_weights__ad(float* b_rwgt, int* failure) { g3_weights(b_rwgt); }
#elif G3_KERNEL == 4
// rrtmg_sw_v842.py:4779: writes reduced[0..count). raw (host-read k-coefficients) and rwgt (mpas_sw_weights) are
// constants of the run. Refusal slot 0: reduced aliasing raw or rwgt (the zero would erase a constant table's word).
extern "C" __global__ void mpas_sw_reduce_coefficients__tl(int count, const float* d_raw, const float* d_rwgt, float* d_red, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= count) return;
    if (d_red == d_raw || d_red == d_rwgt) g3_fail(failure, 0);
    d_red[i] = 0.0f;
}
extern "C" __global__ void mpas_sw_reduce_coefficients__ad(int count, float* b_raw, float* b_rwgt, float* b_red, int* failure) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= count) return;
    if (b_red == b_raw || b_red == b_rwgt) { g3_fail(failure, 0); return; }
    b_red[i] = 0.0f;
}
#elif G3_KERNEL == 5
// radshared_v842.py:308 (captured .cu 9b6ced0f): writes monthly[0..nc*nl*12). lat is fixed mesh geometry, lat_oz and
// table the decoded climatology: constants of the run.
extern "C" __global__ void ozone_latitude__tl(int nc, int nl, float* d_m, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long long)nc * nl * 12) return; d_m[i] = 0.0f;
}
extern "C" __global__ void ozone_latitude__ad(int nc, int nl, float* b_m, int* failure) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long long)nc * nl * 12) return; b_m[i] = 0.0f;
}
#endif
