// SPDX-License-Identifier: Apache-2.0
// Derivative entry (ad.entry/v1: tl / ad, plus a harness-only __check) for rsw_spc_accum_b, the batched RRTMG SW
// g-point flux accumulation: gpuwm/core/rrtmg_sw.py:3825 (launch, gpuwm 2.9.12 sha256 b80bfbb7), kernel source
// hexcore/mpasphys/rrtmg_sw_v842.py pin 44588ef0 (body rsw_spc_accum_body line 2716, batched wrapper line 2828) =
// CUDA capture source 3c499c7b (body line 3035, wrapper line 3147); MPAS module_ra_rrtmg_sw.F:8207-8785.
// Written by hand from the forward CUDA source. Status before: missing (no derivative of any kind).
//
// Forward (grid (ncol, ceil((nl+1)/32)), block 32, one thread per (col, K), K = 0..nl output level; J = nl - K is the
// two-stream level; f32 add.rn / mul.rn, no FMA; the g-point loop iw = 0..111 is sequential, in order):
//   bbfd(K)    = sum_iw w(iw) * zfd(J, iw)          bbfu(K)    = sum_iw w * zfu(J, iw)
//   bbcd(K)    = sum_iw w * zcd(J, iw)              bbcu(K)    = sum_iw w * zcu(J, iw)
//   bbfddir(K) = sum_iw w * ztdbt(J, iw)            bbcddir(K) = sum_iw w * ztdbtc(J, iw)
//   uv{fd,cd,fddir,cddir}(K): the same four sums over the g-points with 10 <= ngb(iw)-15 <= 13 (bands 25-28)
//   ni{fd,cd,fddir,cddir}(K): the same four sums over the g-points with ngb(iw)-15 == 14 or <= 9 (bands 16-24, 29)
// with w = zincflx(col, iw), z(col, J, iw) at (col*(nl+1) + J)*112 + iw and every output at col*(nl+1) + K.
// The interleaved shared-memory tile (6 x 8 g-points x 32 levels) is a pure transpose of the same words; the
// arithmetic and its order are the loop above. So the map is BILINEAR in (zincflx, z): the TL is exact (no
// truncation term at all), and the primal words it needs are the inputs themselves (no forward intermediates).
//
// Precision (the glue-row rule, as the live schemes): partials and sums are formed in double from the f32 primal and
// tangent words; every written derivative word is rounded to f32 once. The harness-only __check reproduces the
// forward's own f32 order (add.rn / mul.rn and the forward's NaN-word rule) and is compared bitwise to the image.
// Limiters: none. No clamp, min/max or state branch: the only branches are the launch guard (level <= nl) and the
// band class of ngb(iw), an integer table (inactive). The tile's input_level clamp to 0 touches only padded lanes
// (level > nl) whose results are never stored. So no frozen-coefficient treatment is needed.
// Inactive: ncol, nlayers, ngb (band index table, role index).
// In / out: zincflx and the six z arrays are read-only in the forward (adjoints accumulate); the 14 outputs are
// fully overwritten at every (col, K <= nl) (tangents out; adjoint seeds consumed: zeroed).
// Determinism: TL is one thread per (col, K) as in the forward. AD runs block (col, 0) only (the other y blocks of
// the forward geometry return at once): phase 1, each lane owns levels K = lane, lane + blockDim.x, ... and writes
// that level's z cotangents (each z word belongs to exactly one K: no race); phase 2, each lane owns g-points
// iw = lane, lane + blockDim.x, ... and sums the zincflx cotangent over K = 0..nl in order; phase 3 zeroes the
// consumed seeds. Barriers separate the phases, so every seed is read before it is zeroed. No atomics on data.
// Refusal: any nonfinite primal input, or nonfinite written word, sets failure slot 0 (source rrtmg_sw.py:3825).

#define RSA_NGPT 112

// Output order = the forward's argument order: 0 bbfd, 1 bbfu, 2 bbcd, 3 bbcu, 4 bbfddir, 5 bbcddir, 6 uvfd, 7 uvcd,
// 8 uvfddir, 9 uvcddir, 10 nifd, 11 nicd, 12 nifddir, 13 nicddir. Input order: 0 zcd, 1 zcu, 2 zfd, 3 zfu,
// 4 ztdbt_nodel, 5 ztdbtc_nodel.
#define RSA_OUT_ARGS(p, c) c float *p##bbfd, c float *p##bbfu, c float *p##bbcd, c float *p##bbcu, c float *p##bbfddir, \
    c float *p##bbcddir, c float *p##uvfd, c float *p##uvcd, c float *p##uvfddir, c float *p##uvcddir, c float *p##nifd, \
    c float *p##nicd, c float *p##nifddir, c float *p##nicddir
#define RSA_OUT_LIST(p) p##bbfd, p##bbfu, p##bbcd, p##bbcu, p##bbfddir, p##bbcddir, p##uvfd, p##uvcd, p##uvfddir, \
    p##uvcddir, p##nifd, p##nicd, p##nifddir, p##nicddir
#define RSA_IN_ARGS(p) const float *p##zincflx, const float *p##zcd, const float *p##zcu, const float *p##zfd, \
    const float *p##zfu, const float *p##ztn, const float *p##ztc
#define RSA_IN_ARGS_W(p) float *p##zincflx, float *p##zcd, float *p##zcu, float *p##zfd, float *p##zfu, float *p##ztn, \
    float *p##ztc

// Band class of g-point iw: 1 = uv (10 <= ibm <= 13), 2 = ni (ibm == 14 or ibm <= 9), 0 = neither (not reached with
// the RRTMG SW ngb table, bands 16..29; kept so the entry follows the forward's predicate exactly).
__device__ __forceinline__ int rsa_class(const int *ngb, int iw) {
  int ibm = ngb[iw] - 15;
  if (ibm >= 10 && ibm <= 13) return 1;
  if (ibm == 14 || ibm <= 9) return 2;
  return 0;
}

// The forward's f32 arithmetic (rsw_rn_add / rsw_rn_mul with mpas_sw_nan_result), for __check only.
__device__ __forceinline__ float rsa_nan(float r, float a, float b) {
  if (!isnan(r)) return r;
  if (isnan(a)) return __uint_as_float(__float_as_uint(a) | 0x00400000u);
  if (isnan(b)) return __uint_as_float(__float_as_uint(b) | 0x00400000u);
  return __uint_as_float(0xffc00000u);
}
__device__ __forceinline__ float rsa_add(float a, float b) { return rsa_nan(__fadd_rn(a, b), a, b); }
__device__ __forceinline__ float rsa_mul(float a, float b) { return rsa_nan(__fmul_rn(a, b), a, b); }

// Harness only (not in launch_abi): the forward output words, out of place, forward order.
extern "C" __global__ void rsw_spc_accum_b__check(int ncol, int nl, const int *ngb, RSA_IN_ARGS(),
    RSA_OUT_ARGS(o_, ), int *failure) {
  int col = blockIdx.x, K = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= ncol || K > nl) return;
  int n1 = nl + 1, J = nl - K;
  const float *w = zincflx + (long long)col * RSA_NGPT;
  long long zb = ((long long)col * n1 + J) * RSA_NGPT;
  float a[14];
#pragma unroll
  for (int o = 0; o < 14; ++o) a[o] = 0.0f;
  bool bad = false;
  for (int iw = 0; iw < RSA_NGPT; ++iw) {
    int c = rsa_class(ngb, iw);
    float ww = w[iw];
    float cd = zcd[zb + iw], cu = zcu[zb + iw], fd = zfd[zb + iw], fu = zfu[zb + iw], tn = ztn[zb + iw], tc = ztc[zb + iw];
    bad |= !isfinite(ww) || !isfinite(cd) || !isfinite(cu) || !isfinite(fd) || !isfinite(fu) || !isfinite(tn) || !isfinite(tc);
    a[1] = rsa_add(a[1], rsa_mul(ww, fu));
    a[0] = rsa_add(a[0], rsa_mul(ww, fd));
    a[3] = rsa_add(a[3], rsa_mul(ww, cu));
    a[2] = rsa_add(a[2], rsa_mul(ww, cd));
    a[4] = rsa_add(a[4], rsa_mul(ww, tn));
    a[5] = rsa_add(a[5], rsa_mul(ww, tc));
    if (c == 1) {
      a[7] = rsa_add(a[7], rsa_mul(ww, cd)); a[6] = rsa_add(a[6], rsa_mul(ww, fd));
      a[9] = rsa_add(a[9], rsa_mul(ww, tc)); a[8] = rsa_add(a[8], rsa_mul(ww, tn));
    } else if (c == 2) {
      a[11] = rsa_add(a[11], rsa_mul(ww, cd)); a[10] = rsa_add(a[10], rsa_mul(ww, fd));
      a[13] = rsa_add(a[13], rsa_mul(ww, tc)); a[12] = rsa_add(a[12], rsa_mul(ww, tn));
    }
  }
  float *O[14] = {RSA_OUT_LIST(o_)};
  long long j = (long long)col * n1 + K;
#pragma unroll
  for (int o = 0; o < 14; ++o) O[o][j] = a[o];
  if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rsw_spc_accum_b__tl(int ncol, int nl, const int *ngb, RSA_IN_ARGS(), RSA_IN_ARGS(d_),
    RSA_OUT_ARGS(d_, ), int *failure) {
  int col = blockIdx.x, K = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= ncol || K > nl) return;
  int n1 = nl + 1, J = nl - K;
  long long wb = (long long)col * RSA_NGPT, zb = ((long long)col * n1 + J) * RSA_NGPT;
  double a[14];
#pragma unroll
  for (int o = 0; o < 14; ++o) a[o] = 0.0;
  bool bad = false;
  for (int iw = 0; iw < RSA_NGPT; ++iw) {
    int c = rsa_class(ngb, iw);
    float wf = zincflx[wb + iw];
    float zf[6] = {zcd[zb + iw], zcu[zb + iw], zfd[zb + iw], zfu[zb + iw], ztn[zb + iw], ztc[zb + iw]};
    double dz[6] = {(double)d_zcd[zb + iw], (double)d_zcu[zb + iw], (double)d_zfd[zb + iw], (double)d_zfu[zb + iw],
                    (double)d_ztn[zb + iw], (double)d_ztc[zb + iw]};
    double w = (double)wf, dw = (double)d_zincflx[wb + iw];
    bad |= !isfinite(wf);
    double t[6];
#pragma unroll
    for (int q = 0; q < 6; ++q) { bad |= !isfinite(zf[q]); t[q] = dw * (double)zf[q] + w * dz[q]; }
    a[0] += t[2]; a[1] += t[3]; a[2] += t[0]; a[3] += t[1]; a[4] += t[4]; a[5] += t[5];
    if (c == 1) { a[6] += t[2]; a[7] += t[0]; a[8] += t[4]; a[9] += t[5]; }
    else if (c == 2) { a[10] += t[2]; a[11] += t[0]; a[12] += t[4]; a[13] += t[5]; }
  }
  float *O[14] = {RSA_OUT_LIST(d_)};
  long long j = (long long)col * n1 + K;
#pragma unroll
  for (int o = 0; o < 14; ++o) { float v = (float)a[o]; O[o][j] = v; bad |= !isfinite(v); }
  if (bad) atomicExch(failure, 1);
}

extern "C" __global__ void rsw_spc_accum_b__ad(int ncol, int nl, const int *ngb, RSA_IN_ARGS(), RSA_IN_ARGS_W(b_),
    RSA_OUT_ARGS(b_, ), int *failure) {
  int col = blockIdx.x;
  if (blockIdx.y != 0 || col >= ncol) return;          // block-uniform: the barriers below see the whole block
  int n1 = nl + 1, lane = threadIdx.x, nt = blockDim.x;
  long long wb = (long long)col * RSA_NGPT, ob = (long long)col * n1;
  const float *Z[6] = {zcd, zcu, zfd, zfu, ztn, ztc};
  float *BZ[6] = {b_zcd, b_zcu, b_zfd, b_zfu, b_ztn, b_ztc};
  float *BO[14] = {RSA_OUT_LIST(b_)};
  bool bad = false;
  // Phase 1: z cotangents, lane-owned levels. Seed of input q at band class c (output indices as above):
  //   zcd: 2 + (uv 7 | ni 11)   zcu: 3   zfd: 0 + (uv 6 | ni 10)   zfu: 1   ztn: 4 + (uv 8 | ni 12)   ztc: 5 + (uv 9 | ni 13)
  for (int K = lane; K <= nl; K += nt) {
    long long j = ob + K, zb = (ob + (nl - K)) * RSA_NGPT;
    double y[14];
#pragma unroll
    for (int o = 0; o < 14; ++o) { y[o] = (double)BO[o][j]; bad |= !isfinite(BO[o][j]); }
    for (int iw = 0; iw < RSA_NGPT; ++iw) {
      int c = rsa_class(ngb, iw);
      float wf = zincflx[wb + iw];
      double w = (double)wf;
      bad |= !isfinite(wf);
      double g[6];
      g[0] = y[2] + (c == 1 ? y[7] : c == 2 ? y[11] : 0.0);
      g[1] = y[3];
      g[2] = y[0] + (c == 1 ? y[6] : c == 2 ? y[10] : 0.0);
      g[3] = y[1];
      g[4] = y[4] + (c == 1 ? y[8] : c == 2 ? y[12] : 0.0);
      g[5] = y[5] + (c == 1 ? y[9] : c == 2 ? y[13] : 0.0);
#pragma unroll
      for (int q = 0; q < 6; ++q) {
        float v = (float)((double)BZ[q][zb + iw] + w * g[q]);
        BZ[q][zb + iw] = v; bad |= !isfinite(v);
      }
    }
  }
  __syncthreads();
  // Phase 2: zincflx cotangent, lane-owned g-points, levels summed K = 0..nl in order.
  for (int iw = lane; iw < RSA_NGPT; iw += nt) {
    int c = rsa_class(ngb, iw);
    double s = 0.0;
    for (int K = 0; K <= nl; ++K) {
      long long j = ob + K, zi = (ob + (nl - K)) * RSA_NGPT + iw;
      double z[6];
#pragma unroll
      for (int q = 0; q < 6; ++q) { float zf = Z[q][zi]; z[q] = (double)zf; bad |= !isfinite(zf); }
      double y[14];
#pragma unroll
      for (int o = 0; o < 14; ++o) y[o] = (double)BO[o][j];
      double t = y[0] * z[2] + y[1] * z[3] + y[2] * z[0] + y[3] * z[1] + y[4] * z[4] + y[5] * z[5];
      if (c == 1) t += y[6] * z[2] + y[7] * z[0] + y[8] * z[4] + y[9] * z[5];
      else if (c == 2) t += y[10] * z[2] + y[11] * z[0] + y[12] * z[4] + y[13] * z[5];
      s += t;
    }
    float v = (float)((double)b_zincflx[wb + iw] + s);
    b_zincflx[wb + iw] = v; bad |= !isfinite(v);
  }
  __syncthreads();
  // Phase 3: the 14 outputs are fully overwritten in the forward, so their seeds are consumed.
  for (int K = lane; K <= nl; K += nt) {
#pragma unroll
    for (int o = 0; o < 14; ++o) BO[o][ob + K] = 0.0f;
  }
  if (bad) atomicExch(failure, 1);
}
