// SPDX-License-Identifier: Apache-2.0
//! tempo-ad-gen: generate the executed-branch tangent/adjoint module of the
//! hrrrv5 TEMPO launch (tempo_warp_steps) from the forward's own CUDA source.
//!
//!   tempo-ad-gen --source FORWARD.cu --sha256 HEX --gen-dir DIR --out tempo_ad.cu --manifest gen.json
//!
//! The forward source is the exact text hex compiles (captured object
//! bb99eb4f..., hex pin 16251589). Output layout:
//!   1. the forward text up to `struct TempoState`, unchanged except the
//!      execution shape (TEMPO_WARP 1 -> 0: one thread per column, the serial
//!      transcription the source documents as the same statements on the same
//!      operands);
//!   2. adreal.cuh (active types, maths partials, tape);
//!   3. namespace adt { the TEMPO column code again, every `float` and
//!      `double` token replaced by AR<float> / AR<double> }, with three named
//!      rules: overloaded one-line helpers keep a plain copy beside the active
//!      one, tmax/tmin promote before comparing (the C++ result type of the
//!      original ternary), and the shared-base pow block is the hand version in
//!      pow_base.cuh;
//!   4. driver.cuh (record, tangent and adjoint entries).
//! Nothing here is read at run time; the output is compiled once with the
//! forward's own strict NVRTC flags.
mod freeze;
use std::collections::BTreeMap;
use std::fs;
use std::process::exit;

fn sha256_hex(data: &[u8]) -> String {
    // Small self-contained SHA-256 (FIPS 180-4), so the generator has no dependency.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut m = data.to_vec();
    let bits = (data.len() as u64) * 8;
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&bits.to_be_bytes());
    for chunk in m.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 { h[i] = h[i].wrapping_add(v[i]); }
    }
    h.iter().map(|x| format!("{:08x}", x)).collect()
}

/// Token-level rewrite of one line: identifiers `float`/`double` outside
/// comments, strings and character literals. `in_block` carries a /* */
/// comment across lines. Returns the new line and the number of rewrites.
fn rewrite_line(line: &str, in_block: &mut bool, promote: bool) -> (String, usize) {
    let b: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len() + 16);
    let mut i = 0;
    let mut n = 0;
    while i < b.len() {
        if *in_block {
            if b[i] == '*' && i + 1 < b.len() && b[i + 1] == '/' { out.push_str("*/"); i += 2; *in_block = false; }
            else { out.push(b[i]); i += 1; }
            continue;
        }
        let c = b[i];
        if c == '/' && i + 1 < b.len() && b[i + 1] == '/' { out.extend(&b[i..]); break; }
        if c == '/' && i + 1 < b.len() && b[i + 1] == '*' { out.push_str("/*"); i += 2; *in_block = true; continue; }
        if c == '"' || c == '\'' {
            out.push(c); i += 1;
            while i < b.len() {
                out.push(b[i]);
                if b[i] == '\\' && i + 1 < b.len() { out.push(b[i + 1]); i += 2; continue; }
                if b[i] == c { i += 1; break; }
                i += 1;
            }
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            // Number: digits, letters, dots; a sign only right after an exponent letter.
            let s = i;
            let hex = c == '0' && i + 1 < b.len() && (b[i + 1] == 'x' || b[i + 1] == 'X');
            while i < b.len() {
                let d = b[i];
                let exp_sign = (d == '+' || d == '-') && i > s && if hex { matches!(b[i - 1], 'p' | 'P') } else { matches!(b[i - 1], 'e' | 'E') };
                if d.is_ascii_alphanumeric() || d == '.' || d == '_' || exp_sign { out.push(d); i += 1; } else { break; }
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') { i += 1; }
            let id: String = b[s..i].iter().collect();
            match id.as_str() {
                "float" => { out.push_str(if promote { "AR<double>" } else { "AR<float>" }); n += 1; }
                "double" => { out.push_str("AR<double>"); n += 1; }
                _ => out.push_str(&id),
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    (out, n)
}

/// `TEMPO_HD[N] ret name(params...)` on one line: (name, params contain float/double, complete one-liner).
fn helper_def(line: &str) -> Option<(String, bool, bool)> {
    let t = line.trim_start();
    let rest = t.strip_prefix("TEMPO_HDN ").or_else(|| t.strip_prefix("TEMPO_HD "))?;
    let open = rest.find('(')?;
    let head = &rest[..open];
    let name = head.split_whitespace().last()?.trim_start_matches('*').to_string();
    let close = rest[open..].find(')').map(|x| x + open).unwrap_or(rest.len());
    let params = &rest[open..close];
    let fp = params.contains("float") || params.contains("double");
    if !rest.contains('{') && rest.trim_end().ends_with(';') { return None; } // declaration only
    let one = rest.contains('{') && rest.trim_end().ends_with('}');
    Some((name, fp, one))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
        .unwrap_or_else(|| { eprintln!("missing {k}"); exit(2) });
    let src_path = get("--source");
    let want = get("--sha256");
    let gen_dir = get("--gen-dir");
    let out_path = get("--out");
    let man_path = get("--manifest");
    // --promote: the float64-promoted grade build (every float of the column code becomes AR<double>).
    let promote = a.iter().any(|x| x == "--promote");
    // --warp: keep the forecast's own warp shape (TEMPO_WARP 1, one 32-lane warp per column) and use the
    // warp tape (adreal_warp.cuh) and its segment driver (driver_warp.cuh); every TEMPO_SYNC of the
    // active copy becomes ad_sync(), the phase boundary of the warp tape.
    let warp_shape = a.iter().any(|x| x == "--warp");
    // --freeze none|closure (default none: the output is byte for byte the unfrozen generator's). closure inserts the
    // statement-level freeze points of freeze.rs into the active copy (value kept, derivative dropped).
    let freeze_mode = a.iter().position(|x| x == "--freeze").and_then(|i| a.get(i + 1)).cloned().unwrap_or_else(|| "none".into());
    let freezing = match freeze_mode.as_str() { "none" => false, "closure" => true,
        m => { eprintln!("--freeze {m}: expected none or closure"); exit(2) } };
    // --freeze-mask: the groups compiled in (a C unsigned literal, e.g. 0x46u); default freeze::DEFAULT_MASK. It is
    // the output's first line, so a bisection build is the same bytes as the generator run with that mask.
    let freeze_mask = a.iter().position(|x| x == "--freeze-mask").and_then(|i| a.get(i + 1)).cloned()
        .unwrap_or_else(|| freeze::DEFAULT_MASK.to_string());
    if !freezing && a.iter().any(|x| x == "--freeze-mask") { eprintln!("--freeze-mask needs --freeze closure"); exit(2); }
    // --freeze-gate R6[,R7] (lane S11): gate the group 6/7 deposition freeze on the forward's species mass (kg m-3):
    // frozen only where the mass is below the bound (freeze::GATE_POINTS). Absent: the output is the ungated bytes.
    let freeze_gate: Option<(String, String)> = a.iter().position(|x| x == "--freeze-gate").and_then(|i| a.get(i + 1)).map(|g| {
        let mut it = g.split(',');
        let r6 = it.next().unwrap_or("").to_string();
        let r7 = it.next().map(str::to_string).unwrap_or_else(|| r6.clone());
        for r in [&r6, &r7] { if r.parse::<f64>().map_or(true, |v| !(v >= 0.0)) { eprintln!("--freeze-gate {g}: expected R6[,R7] >= 0"); exit(2); } }
        (r6, r7) });
    if !freezing && freeze_gate.is_some() { eprintln!("--freeze-gate needs --freeze closure"); exit(2); }
    let point_stmt = |j: usize| -> String {
        if freeze_gate.is_some() { if let Some(g) = freeze::gated_stmt(j) { return g; } }
        freeze::POINTS[j].4.to_string() };
    let raw = fs::read(&src_path).expect("read source");
    let got = sha256_hex(&raw);
    if got != want { eprintln!("source sha256 {got} != pinned {want}"); exit(1); }
    let text = String::from_utf8(raw).expect("utf8");
    let lines: Vec<&str> = text.split('\n').collect();
    // Freeze points by source line; each must hold its text (the source is pinned, so this guards against
    // freezing a different statement).
    let mut frz_before: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut frz_after: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    if freezing {
        if got != freeze::SOURCE_SHA256 { eprintln!("--freeze closure: the freeze table is for source {}, not {got}", freeze::SOURCE_SHA256); exit(1); }
        for (j, (ln, want_txt, at, _bit, _stmt)) in freeze::POINTS.iter().enumerate() {
            let have = lines.get(ln - 1).copied().unwrap_or("");
            if !have.contains(want_txt) { eprintln!("freeze point {j}: source line {ln} does not hold {want_txt:?}: {have:?}"); exit(1); }
            if *at == freeze::At::Before { frz_before.entry(*ln).or_default().push(j); } else { frz_after.entry(*ln).or_default().push(j); }
        }
    }
    let mut frz_applied = vec![0usize; freeze::POINTS.len()];
    let find = |pat: &str, from: usize| -> usize {
        (from..lines.len()).find(|&i| lines[i].starts_with(pat))
            .unwrap_or_else(|| { eprintln!("marker not found: {pat}"); exit(1) })
    };
    let warp_line = find("#define TEMPO_WARP 1", 0);
    let t_start = find("// ty_tend: process rates, real(dp).", 0);
    let cut = find("struct TempoState {float f[17][TEMPO_NZP];};", t_start);
    let pow_a = find("#if defined(TEMPO_SHARED_POW)", t_start);
    let pow_b = find("#endif", pow_a);
    let tmax_line = "template<class A,class B> TEMPO_HD auto tmax(A a,B b)->decltype(a+b) {return a>b?a:b;}";
    let tmin_line = "template<class A,class B> TEMPO_HD auto tmin(A a,B b)->decltype(a+b) {return a<b?a:b;}";
    let rd = |f: &str| fs::read_to_string(format!("{gen_dir}/{f}")).expect(f);
    let header = rd(if warp_shape { "adreal_warp.cuh" } else { "adreal.cuh" });
    let pow_text = rd("pow_base.cuh");
    let driver = rd(if warp_shape { "driver_warp.cuh" } else { "driver.cuh" });

    // Overloaded helpers in the transformed region (two or more definitions with float/double parameters).
    let mut defs: BTreeMap<String, Vec<(usize, bool)>> = BTreeMap::new();
    for i in t_start..cut {
        if i >= pow_a && i <= pow_b { continue; }
        if let Some((name, fp, one)) = helper_def(lines[i]) { if fp { defs.entry(name).or_default().push((i, one)); } }
    }
    let dual: BTreeMap<String, Vec<(usize, bool)>> = defs.into_iter().filter(|(_, v)| v.len() >= 2).collect();
    for (name, v) in &dual {
        for (i, one) in v { if !one { eprintln!("overloaded helper {name} at line {} is not a one-line definition", i + 1); exit(1); } }
    }
    let dual_lines: Vec<usize> = dual.values().flatten().map(|x| x.0).collect();

    let mut out = String::new();
    if freezing { out.push_str(&format!("#define AD_FRZ_MASK {freeze_mask}\n")); }
    if let Some((r6, r7)) = &freeze_gate { out.push_str(&format!("#define AD_FRZ_GATE_R6 {r6}\n#define AD_FRZ_GATE_R7 {r7}\n")); }
    out.push_str("// GENERATED by tempo-ad-gen from the hex TEMPO forward source; do not edit.\n");
    out.push_str(&format!("// source sha256 {got}\n"));
    for (i, l) in lines[..cut].iter().enumerate() {
        if i == warp_line && !warp_shape { out.push_str("#define TEMPO_WARP 0\n"); } else { out.push_str(l); out.push('\n'); }
    }
    out.push_str("\n// ===================== adreal.cuh =====================\n");
    if promote { out.push_str("#define AD_PROMOTE 1\n#define AD_PRIM double\n"); }
    out.push_str(&header);
    if freezing { out.push_str(freeze::HELPER); }
    if freeze_gate.is_some() { out.push_str(freeze::GATE_HELPER); }
    let mut rules: BTreeMap<&str, usize> = BTreeMap::new();
    if warp_shape {
        out.push_str("\n// ===================== active column code =====================\n#undef TEMPO_SYNC\n#define TEMPO_SYNC() ad_sync()\nnamespace adt {\n");
        *rules.entry("warp_sync_is_phase_boundary").or_default() += 1;
    } else {
        out.push_str("\n// ===================== active column code =====================\nnamespace adt {\n");
    }
    let mut in_block = false;
    let mut rewrites = 0usize;
    let mut i = t_start;
    while i < cut {
        let l = lines[i];
        if i == pow_a {
            out.push_str(&pow_text);
            *rules.entry("pow_base_block").or_default() += 1;
            i = pow_b + 1;
            continue;
        }
        if l == tmax_line || l == tmin_line {
            let op = if l == tmax_line { ">" } else { "<" };
            let nm = if l == tmax_line { "tmax" } else { "tmin" };
            out.push_str(&format!("template<class A,class B> TEMPO_HD auto {nm}(A a,B b)->decltype(a+b) {{typedef decltype(a+b) R_; const R_ x_(a), y_(b); return x_{op}y_?x_:y_;}}\n"));
            *rules.entry("promote_minmax").or_default() += 1;
            i += 1;
            continue;
        }
        if dual_lines.contains(&i) {
            out.push_str(l); out.push('\n');
            *rules.entry("plain_overload_copy").or_default() += 1;
            // Promoted grade: the float overload would duplicate the double one once both are AR<double>.
            let params = &l[l.find('(').unwrap()..];
            if promote && params.split(')').next().unwrap().contains("float") {
                *rules.entry("promote_skip_float_overload").or_default() += 1;
                i += 1;
                continue;
            }
        }
        let t = l.trim_start();
        if let Some(rest) = t.strip_prefix("#define ") {
            let name: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            out.push_str(&format!("#undef {name}\n"));
        }
        for &j in frz_before.get(&(i + 1)).map(|v| v.as_slice()).unwrap_or(&[]) {
            out.push_str(&point_stmt(j)); out.push('\n'); frz_applied[j] += 1;
        }
        let (nl, n) = rewrite_line(l, &mut in_block, promote);
        rewrites += n;
        out.push_str(&nl); out.push('\n');
        for &j in frz_after.get(&(i + 1)).map(|v| v.as_slice()).unwrap_or(&[]) {
            out.push_str(&point_stmt(j)); out.push('\n'); frz_applied[j] += 1;
        }
        i += 1;
    }
    // The per-column precipitation and reflectivity launches (P11 objectives read them): their bodies
    // as active device functions, column index a parameter-free 0 so the driver hands each thread
    // one-column views (nc = 1).
    let blocks = [("extern \"C\" __global__ void tempo_precip(", "// RRFS 44c0849 mpas_atmphys_driver_microphysics.F:1038-1060."),
                  ("TEMPO_HD float tempo_mpas_rsif(float p,float t) {", "// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init_tempo.F:30-187.")];
    let colidx = " int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;";
    for (start, end) in blocks {
        let a0 = find(start, cut);
        let a1 = find(end, a0);
        let mut seen = 0;
        for l in &lines[a0..a1] {
            let mut l2 = l.replace("extern \"C\" __global__ void", "__device__ void");
            if l2 == colidx { l2 = " int c=0;".to_string(); seen += 1; }
            let (nl, n) = rewrite_line(&l2, &mut in_block, promote);
            rewrites += n;
            out.push_str(&nl); out.push('\n');
        }
        if start.contains("tempo_precip") && seen != 1 || start.contains("rsif") && seen != 1 { eprintln!("column index line not found once in block {start}"); exit(1); }
        *rules.entry("column_launch_as_device_function").or_default() += 1;
    }
    if warp_shape {
        out.push_str("} // namespace adt\n#undef TEMPO_SYNC\n#define TEMPO_SYNC() __syncwarp()\n\n// ===================== driver_warp.cuh =====================\n");
    } else {
        out.push_str("} // namespace adt\n\n// ===================== driver.cuh =====================\n");
    }
    out.push_str(&driver);
    if freezing {
        for (j, c) in frz_applied.iter().enumerate() {
            if *c != 1 { eprintln!("freeze point {j} (line {}) applied {c} times, expected once", freeze::POINTS[j].0); exit(1); }
        }
        *rules.entry("closure_freeze_points").or_default() += freeze::POINTS.len();
    }
    fs::write(&out_path, &out).expect("write out");
    let manifest = format!(
        "{{\"schema\":\"ad.phys.tempo-gen/v1\",\"promote\":{promote},{}\"source_sha256\":\"{got}\",\"output_sha256\":\"{}\",\"output_bytes\":{},\
\"transformed_lines\":[{},{}],\"type_rewrites\":{rewrites},\"rules\":{{{}}},\"plain_overload_names\":[{}],\
\"header_sha256\":\"{}\",\"pow_base_sha256\":\"{}\",\"driver_sha256\":\"{}\"}}\n",
        format!("{}{}", if warp_shape { "\"warp_shape\":true," } else { "" },
            if freezing { format!("\"freeze\":{{\"mode\":\"closure\",\"mask\":\"{freeze_mask}\",{}\"points\":{},\"groups\":[{}]}},", freeze_gate.as_ref().map_or(String::new(), |(r6, r7)| format!("\"gate\":{{\"r6\":{r6},\"r7\":{r7},\"points\":{}}},", freeze::GATE_POINTS.len())), freeze::POINTS.len(),
                freeze::GROUPS.iter().map(|(b, n, w)| format!("{{\"bit\":{b},\"name\":\"{n}\",\"what\":\"{w}\"}}")).collect::<Vec<_>>().join(",")) } else { String::new() }),
        sha256_hex(out.as_bytes()), out.len(), t_start + 1, cut,
        rules.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(","),
        dual.keys().map(|k| format!("\"{k}\"")).collect::<Vec<_>>().join(","),
        sha256_hex(header.as_bytes()), sha256_hex(pow_text.as_bytes()), sha256_hex(driver.as_bytes()));
    fs::write(&man_path, manifest).expect("write manifest");
    println!("generated {} bytes, {} type rewrites, lines {}..{}, plain overloads {:?}, rules {:?}",
             out.len(), rewrites, t_start + 1, cut, dual.keys().collect::<Vec<_>>(), rules);
}
