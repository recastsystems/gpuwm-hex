// SPDX-License-Identifier: Apache-2.0
// Count bitwise-different 32-bit words on the device. A short final word is
// compared byte by byte; no read extends beyond the supplied allocation.
// The caller must clear *differing to zero on the same stream first and launch
// exactly 256 threads per block. Count only; no floating-point interpretation.
extern "C" __global__ void binder_count_differing_words(
    const unsigned char *expected, const unsigned char *actual,
    unsigned long long bytes, unsigned long long *differing) {
    __shared__ unsigned long long block_count[256];
    const unsigned int tid = threadIdx.x;
    const unsigned long long words = bytes / 4ULL + (bytes % 4ULL != 0ULL);
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long mine = 0;
    for (unsigned long long word = (unsigned long long)blockIdx.x * blockDim.x + tid;
         word < words; word += stride) {
        unsigned long long offset = word * 4ULL;
        bool changed = false;
        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
            if ((unsigned long long)j < bytes - offset && expected[offset + j] != actual[offset + j]) changed = true;
        mine += changed;
    }
    block_count[tid] = mine;
    __syncthreads();
    for (unsigned int delta = 128; delta; delta >>= 1) {
        if (tid < delta) block_count[tid] += block_count[tid + delta];
        __syncthreads();
    }
    if (!tid) atomicAdd(differing, block_count[0]);
}
