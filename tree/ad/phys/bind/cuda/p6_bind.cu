// SPDX-License-Identifier: Apache-2.0
// L2 binding of the P6 RUC sea-ice whole launch (oracle_ice_ruc_ice) as an ad.scheme.rows/v1 row (schemes.rs,
// kind p6_seaice). Data movement only: no field arithmetic except the float64 accumulation of a cotangent into a
// shadow word (p6b_scatter add), which is the binder's own in-place adjoint convention. The derivative itself is
// the unchanged P6 image (tree/ad/phys/p6: seaice_ice_ruc_ice__{aug,tl,ad}, p6_construct).
//
// Every address list is built on the host at bind time from the launch's pointer table and the bound segments
// (schemes.rs): one 64-bit device address per word, 0 for a word the launch does not have (the four local_*
// helper scalars with no arena row, and the second spelling of an aliased word).
// Row layout of the P6 construct (RucRow, 1,756 bytes): x[72] at byte 0, tso[9] at 288, zs[9] at 324, all float32
// and contiguous, so helper word k of column c is at c*1756 + 4k for k < 90. The rest of the row is not read by
// p6_construct and stays zero.
#define P6B_ROW 1756
#define P6B_WORDS 90

// Primal rows from the pre-launch snapshot: pre[c*90 + k] is the float32 word of helper k (x 0..71, tso 72..80,
// zs 81..89) of the c-th sea-ice column.
extern "C" __global__ void p6b_rows(const unsigned long long *pre, unsigned char *rows, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * P6B_WORDS) return;
    int c = (int)(i / P6B_WORDS), k = (int)(i % P6B_WORDS);
    float v = pre[i] ? *reinterpret_cast<const float *>(pre[i]) : 0.f;
    *reinterpret_cast<float *>(rows + (long long)c * P6B_ROW + 4 * k) = v;
}

// dst[i] = float64 shadow word at src[i] (0 where the word is absent).
extern "C" __global__ void p6b_gather(const unsigned long long *src, double *dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i] ? *reinterpret_cast<const double *>(src[i]) : 0.;
}

// Shadow word at dst[i] = v[i] (add == 0) or += v[i] (add == 1). The host lists hold every word at most once, so
// no two threads touch one word.
extern "C" __global__ void p6b_scatter(const unsigned long long *dst, const double *v, long long n, int add) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !dst[i]) return;
    double *p = reinterpret_cast<double *>(dst[i]);
    *p = add ? *p + v[i] : v[i];
}

// Shadow word at dst[i] = 0 (the adjoint consumes the cotangent of every word the launch writes).
extern "C" __global__ void p6b_zero(const unsigned long long *dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !dst[i]) return;
    *reinterpret_cast<double *>(dst[i]) = 0.;
}

// Mirror of the forward's post state in the augmented primal's output layout (y[c*72 + k], then tso[c*9 + l]):
// the float32 word at post[i], or the primal's own value where the launch has no such word (never counted).
extern "C" __global__ void p6b_mirror(const unsigned long long *post, const float *y, float *mirror, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mirror[i] = post[i] ? *reinterpret_cast<const float *>(post[i]) : y[i];
}
