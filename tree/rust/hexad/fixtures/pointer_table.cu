// SPDX-License-Identifier: Apache-2.0
// Synthetic runner exercise only; it is not a physics proof.
extern "C" __global__ void check_pointer_table(const unsigned long long *table,
                                                 const unsigned char *base,
                                                 unsigned int *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = (table[0] == (unsigned long long)(base + 8) &&
                  table[1] == (unsigned long long)base) ? 0x5a17f00dU : 0U;
    }
}
