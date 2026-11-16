// ---- p7 entries for rlw_taumol_batched (generated section; Apache-2.0) ----
#define P7_CALL(NS, N) NS::rlw_gband##N(ncol, nl, col, lay, gpoint, laytrop, isv, chi_mls, oneminus, tabs, ctx)
#define P7_SWITCH(NS, BAND) switch (BAND) { \
    case 0: P7_CALL(NS, 1); break; case 1: P7_CALL(NS, 2); break; \
    case 2: P7_CALL(NS, 3); break; case 3: P7_CALL(NS, 4); break; \
    case 4: P7_CALL(NS, 5); break; case 5: P7_CALL(NS, 6); break; \
    case 6: P7_CALL(NS, 7); break; case 7: P7_CALL(NS, 8); break; \
    case 8: P7_CALL(NS, 9); break; case 9: P7_CALL(NS, 10); break; \
    case 10: P7_CALL(NS, 11); break; case 11: P7_CALL(NS, 12); break; \
    case 12: P7_CALL(NS, 13); break; case 13: P7_CALL(NS, 14); break; \
    case 14: P7_CALL(NS, 15); break; case 15: P7_CALL(NS, 16); break; }

#define P7_THREAD \
    long long element = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    long long tid = element / 16; int gpoint = (int)(element % 16) + 1; \
    if (tid >= (long long)ncol * nl) return; \
    int col = (int)(tid / nl); int lay = (int)(tid % nl) + 1; int laytrop = laytrop_v[col]; \
    const float* const* tabs = (const float* const*)bandptrs[blockIdx.y];

// Transform self-check: identical arithmetic through the generated bodies.
extern "C" __global__ void p7_taumol_fwd(
    int ncol, int nl, const int* __restrict__ laytrop_v, const float* __restrict__ fs,
    const int* __restrict__ isv, const float* __restrict__ wx, const float* __restrict__ chi_mls,
    float oneminus, const unsigned long long* __restrict__ bandptrs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    P7_THREAD
    p7fwd::Ctx ctx{fs, wx, taug, fracs};
    P7_SWITCH(p7fwd, blockIdx.y)
}

// Tangent (augmented forward): writes the forward's taug/fracs and their tangents.
// Launch with the forward's geometry. Derivative storage float64.
extern "C" __global__ void rlw_taumol_batched__tl(
    int ncol, int nl, const int* __restrict__ laytrop_v, const float* __restrict__ fs,
    const int* __restrict__ isv, const float* __restrict__ wx, const float* __restrict__ chi_mls,
    float oneminus, const unsigned long long* __restrict__ bandptrs,
    float* __restrict__ taug, float* __restrict__ fracs,
    const double* __restrict__ dfs, const double* __restrict__ dwx,
    double* __restrict__ dtaug, double* __restrict__ dfracs)
{
    P7_THREAD
    p7tl::Ctx ctx{fs, dfs, wx, dwx, taug, fracs, dtaug, dfracs};
    P7_SWITCH(p7tl, blockIdx.y)
}

// Adjoint: one thread per (column, layer) runs every band's g-points in a fixed
// order (deterministic, no atomics). Adds into bfs/bwx; consumes and clears the
// overwritten outputs' cotangents btaug/bfracs. Grid ceil(ncol*nl/64) x 1, block 64.
// G4 frame arena for the adjoint (see P7Tape): base set by the host per launch, bytes per launch thread exported.
__device__ unsigned char* p7_frame_base;
// ad/last-rad: p7_frame_bytes sizes the arena for rlw_taumol_batched__ad (the compact form below); the first form,
// kept as rlw_taumol_batched__ad_v1 for the proof, needs p7_frame_bytes_v1.
__device__ const unsigned long long p7_frame_bytes_v1 = (unsigned long long)P7_TMAX * (3 * sizeof(double) + 3 * sizeof(short));
__device__ const unsigned long long p7_frame_bytes = (unsigned long long)P7_TMAX * (sizeof(double) + sizeof(int) + 2 * sizeof(float));
extern "C" __global__ void rlw_taumol_batched__ad_v1(
    int ncol, int nl, const int* __restrict__ laytrop_v, const float* __restrict__ fs,
    const int* __restrict__ isv, const float* __restrict__ wx, const float* __restrict__ chi_mls,
    float oneminus, const unsigned long long* __restrict__ bandptrs,
    double* __restrict__ bfs, double* __restrict__ bwx,
    double* __restrict__ btaug, double* __restrict__ bfracs, int* __restrict__ fail)
{
    // G4: a grid-stride loop over (column, layer) on the launch's threads (the host caps the grid), each thread with
    // its tape and adjoint vector in the frame arena. Every (column, layer) is still one thread's serial sweep with
    // its own outputs, so the result does not depend on the grid.
    const long long me = (long long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long long)gridDim.x * blockDim.x;
    double* fd = (double*)p7_frame_base; short* fsh = (short*)(fd + 3 * nt * P7_TMAX);
    P7Tape tape;
    tape.pa = P7Col<double>{fd + me, nt}; tape.pb = P7Col<double>{fd + nt * P7_TMAX + me, nt};
    const P7Col<double> adj{fd + 2 * nt * P7_TMAX + me, nt};
    tape.a = P7Col<short>{fsh + me, nt}; tape.b = P7Col<short>{fsh + nt * P7_TMAX + me, nt};
    tape.leaf = P7Col<short>{fsh + 2 * nt * P7_TMAX + me, nt};
    for (long long tid = me; tid < (long long)ncol * nl; tid += nt) {
    int col = (int)(tid / nl); int lay = (int)(tid % nl) + 1; int laytrop = laytrop_v[col];
    tape.n = 0; tape.fail = 0;
    double acc[P7_NLEAF];
    for (int s = 0; s < P7_NLEAF; ++s) acc[s] = 0.0;
    p7ad::Ctx ctx{fs, wx, &tape, nullptr, nullptr, 0};
    for (int band = 0; band < 16; ++band) {
        const float* const* tabs = (const float* const*)bandptrs[band];
        for (int gpoint = 1; gpoint <= 16; ++gpoint) {
            tape.n = 0; ctx.nout = 0;
            P7_SWITCH(p7ad, band)
            if (ctx.nout == 0) continue;
            for (int k = 0; k < tape.n; ++k) adj[k] = 0.0;
            for (int j = 0; j < ctx.nout; ++j)
                if (ctx.oe[j] >= 0) adj[ctx.oe[j]] += (ctx.ok[j] ? bfracs : btaug)[ctx.oi[j]];
            for (int k = tape.n - 1; k >= 0; --k) {
                double g = adj[k];
                if (g == 0.0) continue;
                if (tape.leaf[k] >= 0) { acc[tape.leaf[k]] += g; continue; }
                if (tape.a[k] >= 0) adj[tape.a[k]] += tape.pa[k] * g;
                if (tape.b[k] >= 0) adj[tape.b[k]] += tape.pb[k] * g;
            }
        }
    }
    if (tape.fail) atomicMax(fail, tape.fail);
    for (int s = 0; s < NFSLOT; ++s) bfs[((long long)s * ncol + col) * nl + (lay - 1)] += acc[s];
    for (int x = 0; x < MAXXSEC; ++x) bwx[((long long)col * MAXXSEC + x) * nl + (lay - 1)] += acc[NFSLOT + x];
    long long row = ((long long)col * nl + (lay - 1)) * NGPTLW;
    for (int g = 0; g < NGPTLW; ++g) { btaug[row + g] = 0.0; bfracs[row + g] = 0.0; }
    }
}

// Adjoint (ad/last-rad): the first form (rlw_taumol_batched__ad_v1) on the compact tape (P7TapeC, 20 bytes an entry
// with the adjoint, against 30): same arguments, same grid, same order of every operation, so every derivative word
// is the first form's bit for bit. Arena per launch thread: adjoint doubles, then code, va, vb (entry-major).
extern "C" __global__ void rlw_taumol_batched__ad(
    int ncol, int nl, const int* __restrict__ laytrop_v, const float* __restrict__ fs,
    const int* __restrict__ isv, const float* __restrict__ wx, const float* __restrict__ chi_mls,
    float oneminus, const unsigned long long* __restrict__ bandptrs,
    double* __restrict__ bfs, double* __restrict__ bwx,
    double* __restrict__ btaug, double* __restrict__ bfracs, int* __restrict__ fail)
{
    const long long me = (long long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long long)gridDim.x * blockDim.x;
    double* fd = (double*)p7_frame_base;
    int* fc = (int*)(fd + nt * P7_TMAX);
    float* ff = (float*)(fc + nt * P7_TMAX);
    P7TapeC tape;
    const P7Col<double> adj{fd + me, nt};
    tape.code = P7Col<int>{fc + me, nt};
    tape.va = P7Col<float>{ff + me, nt}; tape.vb = P7Col<float>{ff + nt * P7_TMAX + me, nt};
    for (long long tid = me; tid < (long long)ncol * nl; tid += nt) {
    int col = (int)(tid / nl); int lay = (int)(tid % nl) + 1; int laytrop = laytrop_v[col];
    tape.n = 0; tape.fail = 0;
    double acc[P7_NLEAF];
    for (int s = 0; s < P7_NLEAF; ++s) acc[s] = 0.0;
    p7ac::Ctx ctx{fs, wx, &tape, nullptr, nullptr, 0};
    for (int band = 0; band < 16; ++band) {
        const float* const* tabs = (const float* const*)bandptrs[band];
        for (int gpoint = 1; gpoint <= 16; ++gpoint) {
            tape.n = 0; ctx.nout = 0;
            P7_SWITCH(p7ac, band)
            if (ctx.nout == 0) continue;
            for (int k = 0; k < tape.n; ++k) adj[k] = 0.0;
            for (int j = 0; j < ctx.nout; ++j)
                if (ctx.oe[j] >= 0) adj[ctx.oe[j]] += (ctx.ok[j] ? bfracs : btaug)[ctx.oi[j]];
            for (int k = tape.n - 1; k >= 0; --k) {
                double g = adj[k];
                if (g == 0.0) continue;
                int c = tape.code[k], op = c & 1023, a = ((c >> 10) & 1023) - 1, b = ((c >> 20) & 1023) - 1;
                if (op == 0) { acc[a + 1] += g; continue; }
                double pa, pb;
                p7c_partials(op, tape.va[k], tape.vb[k], pa, pb);
                if (a >= 0) adj[a] += pa * g;
                if (b >= 0) adj[b] += pb * g;
            }
        }
    }
    if (tape.fail) atomicMax(fail, tape.fail);
    for (int s = 0; s < NFSLOT; ++s) bfs[((long long)s * ncol + col) * nl + (lay - 1)] += acc[s];
    for (int x = 0; x < MAXXSEC; ++x) bwx[((long long)col * MAXXSEC + x) * nl + (lay - 1)] += acc[NFSLOT + x];
    long long row = ((long long)col * nl + (lay - 1)) * NGPTLW;
    for (int g = 0; g < NGPTLW; ++g) { btaug[row + g] = 0.0; bfracs[row + g] = 0.0; }
    }
}
