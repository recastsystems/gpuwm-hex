// SPDX-License-Identifier: Apache-2.0
//! p4-gen: build-time generator for the P4 derivative modules (MYNN surface layer, GSL drag).
//!
//! Input is the exact forward source the hrrrv5 capture recorded (sha256-pinned). The output
//! is one self-contained CUDA file per family: the original source verbatim (tables, libm,
//! original kernels), then the derivative types (p4_ad.cuh), then the physics part with
//! `float` replaced by the derivative type `ADF` and its functions renamed `adx_*`, then the
//! hand-written entries. Nothing in the forward tree is read or written except as input text.
//! Every textual pattern must match the expected number of times, or generation stops.
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn sha(b: &[u8]) -> String { let mut h = Sha256::new(); h.update(b); format!("{:x}", h.finalize()) }

fn die(msg: &str) -> ! { eprintln!("p4-gen: {msg}"); std::process::exit(2) }

/// Byte range of a brace-delimited block starting at the first '{' at or after `from`.
fn block_end(s: &str, from: usize) -> usize {
    let b = s.as_bytes();
    let mut i = s[from..].find('{').map(|x| x + from).unwrap_or_else(|| die("no block"));
    let mut depth = 0i32;
    loop {
        match b[i] { b'{' => depth += 1, b'}' => { depth -= 1; if depth == 0 { return i + 1; } } _ => {} }
        i += 1;
        if i >= b.len() { die("unbalanced block"); }
    }
}

/// Replace the type word `float` with ADF, except in `float *` pointer declarations.
fn subst_float(s: &str) -> String {
    let re = Regex::new(r"\bfloat\b(\s*\*)?").unwrap();
    re.replace_all(s, |c: &regex::Captures| if c.get(1).is_some() { c[0].to_string() } else { "ADF".to_string() })
        .into_owned()
}

fn count(s: &str, pat: &str) -> usize { s.matches(pat).count() }

fn expect(n: usize, want: usize, what: &str) { if n != want { die(&format!("{what}: matched {n}, expected {want}")); } }

fn mynnsfc(src: &str, header: &str, entries: &str) -> String {
    let start = src.find("__device__ float psim_stab[1001];").unwrap_or_else(|| die("table start"));
    let kern = src.find("extern \"C\" __global__ void mynnsfc_columns(").unwrap_or_else(|| die("columns kernel"));
    let mut phys = src[start..kern].to_string();
    // tables stay in the verbatim copy; the substituted part refers to them
    let tab = Regex::new(r"(?m)^__device__ float psi[mh]_(?:un)?stab\[1001\];\n").unwrap();
    expect(tab.find_iter(&phys).count(), 4, "table declarations");
    phys = tab.replace_all(&phys, "").into_owned();
    // the table builder writes the tables: it stays float, in the verbatim copy only
    let proto = "__device__ __noinline__ void m_drv_psi_init(int psi_opt, int &errflg);\n";
    expect(count(&phys, proto), 1, "psi_init prototype");
    phys = phys.replacen(proto, "", 1);
    let def = phys.find("__device__ __noinline__ void m_drv_psi_init(int psi_opt, int &errflg) {").unwrap_or_else(|| die("psi_init body"));
    let end = block_end(&phys, def);
    phys.replace_range(def..end, "");
    expect(count(&phys, "m_drv_psi_init"), 1, "psi_init call sites left (the init branch of the driver)");
    let phys = Regex::new(r"\bm_").unwrap().replace_all(&phys, "adx_m_").into_owned();
    // the driver's own cold-start table call goes to the verbatim float builder
    let phys = phys.replace("adx_m_drv_psi_init(", "m_drv_psi_init(");
    let phys = subst_float(&phys);

    // the column body of mynnsfc_columns becomes a device function over a local field array
    let kend = block_end(src, kern);
    let kernel = &src[kern..kend];
    let open = kernel.find('{').unwrap() + 1;
    let mut body = kernel[open..kernel.len() - 1].to_string();
    let head = "    int c=blockIdx.x*blockDim.x+threadIdx.x;\n    if(c>=n) return;\n    int error=0;\n";
    expect(count(&body, head), 1, "columns prologue");
    body = body.replacen(head, "", 1);
    let tail = "    errors[c]=error;\n";
    expect(count(&body, tail), 1, "columns epilogue");
    body = body.replacen(tail, "", 1);
    let fre = Regex::new(r"\bf\[(\d+)\*n\+c\]").unwrap();
    expect(fre.find_iter(&body).count(), 68, "field references");
    body = fre.replace_all(&body, "fl[$1]").into_owned();
    expect(count(&body, "m_drv_mynnsfc_driver("), 1, "driver call");
    body = body.replace("m_drv_mynnsfc_driver(", "adx_m_drv_mynnsfc_driver(");
    let colfn = format!(
        "// generated from mynnsfc_columns (captured source), field array in registers\n\
         __device__ __forceinline__ void p4_mynnsfc_column(ADF *fl, const int *options, const int *vegtype, int &error,\n\
         \x20   float xice_threshold, unsigned long long present, int n, int c, int step_override, int init_override) {{\n{body}}}\n");
    let mut parts = entries.split("//@@P4_SPLIT@@");
    let (e1, e2) = (parts.next().unwrap(), parts.next().unwrap_or_else(|| die("entries split")));
    format!("{src}\n// ===== p4: derivative types =====\n{header}\n// ===== p4: entries part 1 (libm) =====\n{e1}\n// ===== p4: substituted physics (generated) =====\n{phys}\n{colfn}\n// ===== p4: entries part 2 =====\n{e2}\n")
}

fn ugwp(src: &str, header: &str, entries: &str, freeze: &str) -> String {
    let k = src.find("extern \"C\" __global__ void ugwp_oro_run(").unwrap_or_else(|| die("ugwp kernel"));
    let kend = block_end(src, k);
    let kernel = &src[k..kend];
    let open = kernel.find('{').unwrap();
    let sig = &kernel[..open];
    let mut body = kernel[open..].to_string();
    let want_sig = "extern \"C\" __global__ void ugwp_oro_run(\n    const float *layers, const float *prsi0, const float *surface,\n    const int *kpbl0, float *out, float *sout, int *status,\n    float g, float cp, float rd, float rv, float fv, float pi_c,\n    float dt, int nz, int nc, int diags)\n";
    if sig != want_sig { die(&format!("ugwp signature changed:\n{sig}")); }
    let new_sig = "__device__ __noinline__ void adx_ugwp_oro_run_body(\n    const P4In &layers, const P4In &prsi0, const P4In &surface,\n    const int *kpbl0, P4OutO &out, P4OutS &sout, int *status,\n    float g, float cp, float rd, float rv, float fv, float pi_c,\n    float dt, int nz, int nc, int diags, int p4_col0, int p4_ncols)\n";
    let idx = "const int i = blockDim.x * blockIdx.x + threadIdx.x;\n    if (i >= nc) return;";
    expect(count(&body, idx), 1, "ugwp column index");
    body = body.replacen(idx, "const int p4_t = blockDim.x * blockIdx.x + threadIdx.x;\n    const int i = p4_col0 + p4_t;\n    if (p4_t >= p4_ncols || i >= nc) return;", 1);
    let body = subst_float(&body);
    let (body, frz_defs) = ugwp_freeze(body, freeze);
    // UGWP_LEVELS and the libm live in the verbatim copy
    format!("{src}\n// ===== p4: derivative types =====\n{header}\n// ===== p4: entries part 1 (accessors, libm) =====\n{}\n// ===== p4: substituted ugwp_oro_run body (generated) =====\n{frz_defs}{new_sig}{body}\n// ===== p4: entries part 2 =====\n{}\n",
        entries.split("//@@P4_SPLIT@@").next().unwrap(), entries.split("//@@P4_SPLIT@@").nth(1).unwrap_or_else(|| die("entries split")))
}

/// GSL drag closure frozen from the forward trajectory (ruling 2026-11-05 08:35Z, the limiter/MYNN rule):
/// p4_frz(x) keeps the float value bit for bit and drops the derivative (TL: t = 0; REV: no tape node).
/// Every switch still decides on the forward value. "none" leaves the body byte for byte as before.
///   closure:        N^2 (bnv2, every level, floor applied) and Ri at every level; their reference-layer
///                   averages bnv2[1], ri[1] (so bnv, brvf, the Lindzen temc/tem2 and the blocking pe follow);
///                   the small-scale boundary-layer N (xnbv); the stress limiters dtfac and dtfac_meso.
///   closure+froude: closure plus the Froude number fr (so efact and gfobnv carry no derivative through it).
/// Integer selections (direction bin idir, kbl, komax, kblk, kpbl2) and the saturation arm carry no
/// derivative in either mode: they are the executed choice of the forward.
fn ugwp_freeze(body: String, freeze: &str) -> (String, String) {
    if freeze == "none" { return (body, String::new()); }
    let mut b = body;
    let mut rep = |b: &mut String, from: &str, to: &str, what: &str| {
        expect(count(b, from), 1, what);
        *b = b.replacen(from, to, 1);
    };
    rep(&mut b, "            ri[k]=fmaxf(bvf2/shr2,-100.0f);\n",
        "            ri[k]=p4_frz(fmaxf(bvf2/shr2,-100.0f));\n", "ri per level");
    rep(&mut b, "            bnv2[k]=fmaxf(bnv2[k],1.e-5f);\n",
        "            bnv2[k]=p4_frz(fmaxf(bnv2[k],1.e-5f));\n", "bnv2 per level");
    rep(&mut b, "        ldrag=ldrag || bnv2[1]<=0.0f;\n",
        "        bnv2[1]=p4_frz(bnv2[1]); ri[1]=p4_frz(ri[1]);\n        ldrag=ldrag || bnv2[1]<=0.0f;\n", "reference-layer N^2, Ri");
    rep(&mut b, "            const ADF xnbv=__fsqrt_rn(govrth*(thvk-thv1)/hpbl2);\n",
        "            const ADF xnbv=p4_frz(__fsqrt_rn(govrth*(thvk-thv1)/hpbl2));\n", "small-scale N");
    rep(&mut b, "                if(s!=0.0f) dtfac=fminf(dtfac,fabsf(velco[k]/(dt*s)));\n",
        "                if(s!=0.0f) dtfac=p4_frz(fminf(dtfac,fabsf(velco[k]/(dt*s))));\n", "dtfac limiter");
    rep(&mut b, "                    dtfac_meso=fminf(dtfac_meso,0.5f*fabsf(velco[k]/(dt*taud_ls[k])));\n",
        "                    dtfac_meso=p4_frz(fminf(dtfac_meso,0.5f*fabsf(velco[k]/(dt*taud_ls[k]))));\n", "dtfac_meso limiter");
    if freeze == "closure+froude" {
        rep(&mut b, "            ADF fr=bnv*rulow*var*od; fr=fminf(fr,10.0f);\n",
            "            ADF fr=bnv*rulow*var*od; fr=p4_frz(fminf(fr,10.0f));\n", "Froude number");
    }
    let defs = format!("// p4 freeze: {freeze} (GSL drag closure frozen from the trajectory; value bitwise, derivative dropped)\n\
#if defined(P4_MODE_TL)\n__device__ __forceinline__ ADF p4_frz(const ADF &a) {{ return ADF(a.v, 0.0); }}\n\
#else\n__device__ __forceinline__ ADF p4_frz(const ADF &a) {{ return ADF(a.v); }}\n#endif\n");
    (b, defs)
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() != 7 && a.len() != 8 { die("usage: p4-gen <mynnsfc|ugwp> <captured.cu> <sha256> <p4_ad.cuh> <entries.cuh> <out.cu> [--freeze=none|closure|closure+froude]"); }
    let freeze = match a.get(7).map(|x| x.as_str()) {
        None | Some("--freeze=none") => "none",
        Some("--freeze=closure") => "closure",
        Some("--freeze=closure+froude") => "closure+froude",
        Some(x) => die(&format!("unknown option {x}")),
    };
    if freeze != "none" && a[1] != "ugwp" { die("--freeze applies to the ugwp family only"); }
    let src_b = fs::read(&a[2]).unwrap_or_else(|e| die(&format!("{}: {e}", a[2])));
    let got = sha(&src_b);
    if got != a[3] { die(&format!("source sha256 {got} != pinned {}", a[3])); }
    let src = String::from_utf8(src_b).unwrap();
    let header = fs::read_to_string(&a[4]).unwrap();
    let entries = fs::read_to_string(&a[5]).unwrap();
    let out = match a[1].as_str() { "mynnsfc" => mynnsfc(&src, &header, &entries), "ugwp" => ugwp(&src, &header, &entries, freeze), _ => die("family") };
    let p = PathBuf::from(&a[6]);
    fs::write(&p, out.as_bytes()).unwrap();
    println!("{} {} source={} header={} entries={} freeze={}", sha(out.as_bytes()), p.display(), got, sha(header.as_bytes()), sha(entries.as_bytes()), freeze);
}
