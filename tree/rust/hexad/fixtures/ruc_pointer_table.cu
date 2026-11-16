// SPDX-License-Identifier: Apache-2.0
// Synthetic check only; this is not a forecast or physics equality proof.
extern "C" __global__ void check_ruc_pointer_table(const unsigned long long *table,
                                                     const unsigned char *base,
                                                     unsigned int *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        bool ok = true;
        for (unsigned int i = 0; i < 11; ++i)
            ok = ok && table[i] == (unsigned long long)(base + 8 * i);
        out[0] = ok ? 0x5a17f00dU : 0U;
    }
}
