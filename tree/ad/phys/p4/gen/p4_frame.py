#!/usr/bin/env python3
# G4 (2026-11-06): ugwp_oro_run derivative entries keep their per-column work arrays (16 level arrays of ADF in the
# generated body, and the P4Out output slots) in a frame arena in global memory, entry-major across the launch's
# threads, instead of on the thread stack (25,280 B/thread: ~6.8 GiB reserved for every resident thread of a 188-SM
# card). Storage only: every value, slope and tape node is computed as before.
# usage: p4_frame.py <cuda/ugwp_entries.cuh> (in place, idempotent) | p4_frame.py --body <gen-out/ugwp.p4.cu> (in place)
import sys, re
ARR = ["u", "v", "vtj", "vtk", "rho", "del", "za", "bnv2", "ri", "velco", "taup", "taufb", "taud_ls", "taud_bl", "dudt", "dvdt"]
def entries(p):
    s = open(p).read()
    if "p4_frame_base" in s: print("entries: already framed"); return
    old = """    float val[CAP]; bool wr[CAP];
#if defined(P4_MODE_TL)
    double tan[CAP];
#else
    int node[CAP];
#endif
    int rbw;               // reads of a slot this launch had not written yet (expected 0)
    __device__ __forceinline__ void init(float *fw_, size_t plane_, int nc_, int L_) {
        fw = fw_; plane = plane_; nc = nc_; L = L_; rbw = 0;"""
    new = """    // G4: the slots live in the frame arena (P4Col views, entry-major across the launch threads), not on the stack.
    P4Col<float> val; P4Col<bool> wr;
#if defined(P4_MODE_TL)
    P4Col<double> tan;
#else
    P4Col<int> node;
#endif
    int rbw;               // reads of a slot this launch had not written yet (expected 0)
    __device__ __forceinline__ void init(float *fw_, size_t plane_, int nc_, int L_, long long tan_off, long long val_off, long long wr_off) {
#if defined(P4_MODE_TL)
        tan = p4_frame_col<double>(tan_off);
#else
        node = p4_frame_col<int>(tan_off);
#endif
        val = p4_frame_col<float>(val_off); wr = p4_frame_col<bool>(wr_off);
        fw = fw_; plane = plane_; nc = nc_; L = L_; rbw = 0;"""
    assert s.count(old) == 1, "P4Out arrays"; s = s.replace(old, new)
    anchor = "// Output accessor:"
    assert s.count(anchor) == 1
    frame = """// G4 frame arena: base set by the host per launch (module global); element k of a region for launch thread t at
// region + (k * nthreads + t) * sizeof(T), regions ordered 8-byte types first so every region stays aligned.
__device__ unsigned char *p4_frame_base;
template <class T> struct P4Col {
    T *p; long long s;
    __device__ __forceinline__ T &operator[](long long k) const { return p[k * s]; }
};
template <class T> __device__ __forceinline__ P4Col<T> p4_frame_col(long long off) {
    const long long nt = (long long)gridDim.x * blockDim.x, t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    return P4Col<T>{(T *)(p4_frame_base + off * nt) + t, nt};
}
#if defined(P4_MODE_TL)
typedef double P4TanT;
#else
typedef int P4TanT;
#endif
#define P4F_NARR 16
#define P4F_ADF 0LL
#define P4F_OTAN (P4F_ADF + (long long)P4F_NARR * (UGWP_LEVELS + 2) * (long long)sizeof(ADF))
#define P4F_STAN (P4F_OTAN + 12LL * UGWP_LEVELS * (long long)sizeof(P4TanT))
#define P4F_OVAL (P4F_STAN + 10LL * (long long)sizeof(P4TanT))
#define P4F_SVAL (P4F_OVAL + 12LL * UGWP_LEVELS * 4)
#define P4F_OWR (P4F_SVAL + 10LL * 4)
#define P4F_SWR (P4F_OWR + 12LL * UGWP_LEVELS)
#define P4F_BYTES (P4F_SWR + 10LL)
__device__ const unsigned long long p4_frame_bytes = (unsigned long long)((P4F_BYTES + 15) / 16 * 16);
"""
    s = s.replace(anchor, frame + anchor)
    for o, n in [("O_.init(out, n3, nc, nz);", "O_.init(out, n3, nc, nz, P4F_OTAN, P4F_OVAL, P4F_OWR);"),
                 ("S2.init(sout, (size_t)nc, nc, 1);", "S2.init(sout, (size_t)nc, nc, 1, P4F_STAN, P4F_SVAL, P4F_SWR);")]:
        assert s.count(o) == 2, o; s = s.replace(o, n)
    open(p, "w").write(s); print("entries: framed")
def body(p):
    s = open(p).read()
    if "G4 frame: the column arrays" in s: print("body: already framed"); return
    decl = """    ADF u[UGWP_LEVELS+2], v[UGWP_LEVELS+2], vtj[UGWP_LEVELS+2];
    ADF vtk[UGWP_LEVELS+2], rho[UGWP_LEVELS+2], del[UGWP_LEVELS+2];
    ADF za[UGWP_LEVELS+2], bnv2[UGWP_LEVELS+2], ri[UGWP_LEVELS+2];
    ADF velco[UGWP_LEVELS+2], taup[UGWP_LEVELS+2], taufb[UGWP_LEVELS+2];
    ADF taud_ls[UGWP_LEVELS+2], taud_bl[UGWP_LEVELS+2];
    ADF dudt[UGWP_LEVELS+2], dvdt[UGWP_LEVELS+2];
"""
    assert s.count(decl) == 1, "body declarations"
    new = "    // G4 frame: the column arrays live in the frame arena (default-constructed, as the stack arrays were).\n"
    for i, a in enumerate(ARR):
        new += f"    const P4Col<ADF> {a} = p4_frame_col<ADF>(P4F_ADF + {i}LL * (UGWP_LEVELS + 2) * (long long)sizeof(ADF));\n"
    new += "    for (int p4_k = 0; p4_k < UGWP_LEVELS + 2; ++p4_k) {\n"
    new += "        " + " ".join(f"{a}[p4_k] = ADF();" for a in ARR) + "\n    }\n"
    s = s.replace(decl, new)
    open(p, "w").write(s); print("body: framed")
if sys.argv[1] == "--body": body(sys.argv[2])
else: entries(sys.argv[1])
