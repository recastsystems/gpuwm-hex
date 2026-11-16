// SPDX-License-Identifier: Apache-2.0
// Synthetic module-global binding test only; this is not a physics proof.
extern "C" __device__ __constant__ unsigned int binder_values[4] = {
    0x12345678U, 0x4b7a2d91U, 0U, 0U
};
extern "C" __global__ void check_module_global(const unsigned int *value,
                                               unsigned int *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = value[0] == 0x4b7a2d91U ? 0x5a17f00dU : 0U;
    }
}
