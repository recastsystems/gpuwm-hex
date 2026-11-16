// SPDX-License-Identifier: Apache-2.0
//! Physics-on window, lane L1: derivative rule of every command of the physics plan, for the composed
//! tangent and the segment-checkpointed reverse (window.rs). Kernel launches with an ad.entry/v1 row run
//! their entries (deriv.rs); everything else gets one of the rules below, applied on one float shadow arena
//! (P8 / U7 convention: one shadow element per forward 4-byte word, an 8-byte element's shadow at its first
//! word). The copy-row, view, where, max and zero rules are P8's (tree/ad/phys/p8: rules.rs assignment,
//! rows.rs row form and parallel gather transpose, cuda/p8.cu kernels). Nothing is passed silently: every
//! command that has no exact rule is `Refused` (identity in both directions, so the composed operator stays
//! the exact transpose of the tangent) and counted by name.
use crate::{plan::{self, Operation}, submit, Execution};
use std::collections::HashMap;

/// Element type `T` of every module that launches cupy_concatenate_same_size, from the module's recorded CuPy source
/// (`typedef <T> T;` in recording/sources/<sha>.cu, named by any argument of any launch of that module; the runner has
/// already checked each named source against its sha256, crate::verify_copy_indexer_sources /
/// verify_fixed_argument_sources). The kernel name carries no dtype, so this is the only place the type comes from.
/// THE BREAKAGE THIS PREVENTS (step6-sw, review finding ad/phys-p8-check rules.rs:25): the window bound the launch by
/// NAME as inactive (schemes18.json, built on pkg18 whose only instantiation is `typedef long long T`). pkg18m2 also has
/// module 5396101c, `typedef float T`: the SW radiation input stack `f = cp.stack(p3d, t3d, dz8w, pi3d, qv3d, qc3d, qi3d,
/// qs3d, cldfra3d, re_cloud, re_ice, re_snow)` (rrtmg_sw_v842.py:581) that mpas_sw_prep and mpas_sw_outputs read. As an
/// identity the SW tangent never saw the state, and mpas_sw_prep read whatever tangent the reused pool words held.
pub fn concat_module_types(plan: &plan::Plan, base: &std::path::Path) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    let mut want: Vec<&str> = Vec::new();
    for ops in plan.blocks.values() { for op in ops { if let Operation::Kernel { module, function, .. } = op {
        if function == "cupy_concatenate_same_size" && !want.contains(&module.as_str()) { want.push(module); }
    } } }
    for m in want {
        let mut src: Option<String> = None;
        'find: for ops in plan.blocks.values() { for op in ops { if let Operation::Kernel { module, arguments, .. } = op {
            if module != m { continue; }
            for a in arguments {
                let s = match a {
                    plan::Argument::CupyCopyIndexer { source, .. } => Some(&source.source),
                    plan::Argument::Bytes { fixed_source: Some(f), .. } | plan::Argument::Patched { fixed_source: Some(f), .. } => Some(&f.source),
                    _ => None,
                };
                if let Some(s) = s.filter(|s| s.starts_with("recording/sources/")) { src = Some(s.clone()); break 'find; }
            }
        } } }
        // S01b: a module whose launches carry no recorded-source reference takes its source from the proved
        // (image, source) CIndexer pairs (carray.rs): the 3 km LAM's float SW stack image 5396101c, source 1ae76152.
        // THE BREAKAGE THIS PREVENTS: "element type None not established from the module's recorded source" (refused).
        let src = src.or_else(|| match m {
            "5396101c0ef796bd00442a0a8b1e86918acde77f9349378760dc21c23000d056" =>
                Some("recording/sources/1ae7615202e9b32e2d3f0b85e552eb80033e5603c5fe2ae8f0682b4abe33ce43.cu".to_string()),
            "82af2f9c5466c4d839daebc05a7fc5497952d5c47cbe2ea8d059c4211b09e555" =>
                Some("recording/sources/4a66cfe30903320192f854458f2cb1b07f23506c864c1c5863a4db9dfb296160.cu".to_string()),
            // g1 round 2: the SW radiation's deferred-check vector (hexcore cuda_driver.py, cp.concatenate(flags)),
            // `typedef double T`. Its pairing is recorded: pk120e4 step_01 op 1229 launches this image with a
            // cupy_copy_indexer argument naming recording/sources/ed5c8f4a...c676.cu (sha256-checked by the runner),
            // and the module id IS the image's sha256, so the pairing holds in every package that carries that source.
            // THE BREAKAGE THIS PREVENTS: the 29-step 60 km package pk60e29p-20230331 has six launches of this image,
            // none naming a source; five are followed (5 steps later) by a cupy_copy__int32/bool_float64 writing one
            // double at the vector's first byte, which neither host-check proof admits, so the type stayed None and
            // window e29-s2-self stopped at step 1 ("element type None not established from the module's recorded
            // source"). With the type established the launch gets the float64 copy rule below (exact, no proof needed).
            "b15be0f44f1cf96074960f5935c60ed94e6e83adc8cc63b27f820dd990ae6b0e" =>
                Some("recording/sources/ed5c8f4ac11438b7050e7b811d7d0ceeef692b06c4a5d6003b2a37475c53c676.cu".to_string()),
            _ => None,
        }).filter(|s| base.join(s).exists());
        let Some(src) = src else { continue };
        let text = std::fs::read_to_string(base.join(&src)).map_err(|e| format!("concatenate module {m}: {src}: {e}"))?;
        if let Some(t) = text.lines().map(str::trim).find_map(|l| l.strip_prefix("typedef ").and_then(|r| r.strip_suffix(" T;"))) {
            out.insert(m.to_string(), t.to_string());
        }
    }
    Ok(out)
}

/// integrator: enter the host-check proof (concat_host_checked) into the module type map (concat_module_types).
/// A float module keeps its copy rows and an integer module is inactive already: both stay as they are. Any other
/// module that passes the proof becomes "<type>:host-checked", and a module with NO established type that passes becomes
/// ":host-checked" (Context::concatenate binds both as identity).
/// THE BREAKAGE THIS PREVENTS: a host-checked module whose launches carry no recorded-source reference and that is not
/// in concat_module_types' fallback table has no entry in the map; a loop over the map's entries never sees it and the
/// launch stays Refused ("element type None", the 60 km plan pkg60-e4).
pub fn mark_host_checked(types: &mut HashMap<String, String>, checked: &std::collections::HashSet<String>) {
    for m in checked {
        let t = types.entry(m.clone()).or_default();
        if !matches!(t.as_str(), "float" | "long long" | "long" | "int" | "unsigned long long" | "unsigned int" | "bool") { t.push_str(":host-checked"); }
    }
}

/// integrator (2026-11-09): the concatenate modules whose EVERY launch in the plan writes a vector that only the host
/// reads. Proved on the raw plan JSON, so every operation kind is covered with no per-kind list. For each launch the
/// output is argument 2 (a by-value CArray: pointer, size, shape, stride). Among the later operations of the same step
/// that name the output allocation at an offset inside the output bytes (any nested {"allocation","offset"}):
/// * the first must be a device_to_host of exactly the output with a "checked..." host_consumer (more may follow);
/// * the first operation of any other kind, if there is one, must be a clear or an upload that starts exactly at the
///   output: a NEW array in that pool block, which the pool can only hand out once the vector is dead.
/// A reference at a lower offset is another array of the pool (alive beside the vector, or made after it died) and
/// cannot be a view of the vector, which is the kernel's own fresh output. So no device word is ever computed from the
/// vector, and identity in the tangent and the reverse is the exact derivative of everything the model reads.
/// SCOPE OF THE PROOF, as it is: the first two bullets look only inside the launch's own step. When that step holds no
/// clear or upload at the output after the check (the vector is not proven dead inside its step), the proof also
/// requires that no operation in any other block, and none before the launch in its own block, names the bytes before a
/// new array starts exactly there (correction of c1313db4, whose rule stopped at the end of the step).
/// Used for every module that has no float row and is not integer: a recorded type such as double, and also a module
/// whose element type is NOT established at all (mark_host_checked). The proof does not need the type.
/// THE BREAKAGE THIS PREVENTS: the SW radiation's deferred check (hexcore rrtmg_sw_v842) gathers its flags into one
/// double vector, module b15be0f4, that only the host reads. At 120 km (pk120e4: 16 doubles, two launches, one of them
/// names the recorded source) the launch was Refused as "element type Some(double) not established". At 60 km
/// (pkg60-e4: 60 doubles, one launch, step_01 op 2959, argument kinds patched/bytes/patched/bytes with no recorded-source
/// reference) the module has no type at all and was Refused as "element type None not established"; c1313db4 tagged only
/// modules that already had a type, so it did not reach that plan. Either way it was a refused kind beyond the deck's
/// explicit list, which makes a window INVALID by the program's rule although nothing reads the vector but the host.
/// storm3km: the operation kinds that start a NEW array at their destination from host bytes: a clear, an upload, and
/// the table uploads (ruc_pointer_table_upload, descriptor_table_upload, rrfs_copy_plan_upload: every "*_upload" kind
/// writes its destination from the host and reads nothing on the device). THE BREAKAGE THIS PREVENTS: on the 38,278-cell
/// 3 km limited-area plan (pkg-lam-r300-e4 step_01 op 1712, module b15be0f4, 28 doubles at allocation 102 + 21605376) the
/// first operation after the host check that names the vector's bytes is a ruc_pointer_table_upload starting exactly
/// there: the pool slot handed to a new array, the vector dead. The proof accepted only "clear" and "upload", failed, and
/// the window carried one refused kind ("element type Some(double) not established"), so it read INVALID
/// (a test machine <local path>) although only the host ever read the vector.
/// DOORS round 7 (init120v7): a copy plan (rrfs_copy_plan_upload) whose only rows that name the vector's bytes are FILL
/// rows (src null: the row writes a constant, it reads nothing) and one of them starts exactly at the vector. That is a
/// new array born at the vector's first byte, as a clear or an upload there is: the pool can only hand the block out
/// once the vector is dead, and a fill reads no word of it.
/// THE BREAKAGE THIS PREVENTS: a plan the hex door records from a start file under GPUWM_HEX_MEMBER_MEMORY=single
/// (x1.40962, run init120-f) launches the SW deferred-check vector (typedef double, 16 flags) with a CIndexer that is
/// equal in both recordings, so no launch names its source and the type stays None; the vector's block (allocation 169
/// offset 512) is next named by step_01 op 4267, a copy plan with a fill row at exactly that place, which the proof
/// did not count as a new array: all four doors stopped at step 1, "element type None not established".
fn copy_plan_fill_starts_at(q: &serde_json::Value, al: u64, off: u64, nb: u64) -> bool {
    use serde_json::Value;
    if q.get("kind").and_then(Value::as_str) != Some("rrfs_copy_plan_upload") { return false; }
    let inside = |v: Option<&Value>| v.map_or(false, |t| t.get("allocation").and_then(Value::as_u64) == Some(al)
        && t.get("offset").and_then(Value::as_u64).map_or(false, |o| o >= off && o < off + nb));
    let at = |v: Option<&Value>| v.map_or(false, |t| t.get("allocation").and_then(Value::as_u64) == Some(al)
        && t.get("offset").and_then(Value::as_u64) == Some(off));
    if inside(q.get("destination")) || inside(q.get("cache")) { return false; }
    let Some(rows) = q.get("rows").and_then(Value::as_array) else { return false };
    let mut starts = false;
    for r in rows {
        let src = r.get("src").filter(|s| !s.is_null());
        if inside(src) { return false; }
        if inside(r.get("dst")) {
            if src.is_some() { return false; }
            if at(r.get("dst")) { starts = true; }
        }
    }
    starts
}

fn fresh_array_kind(kind: &str) -> bool { kind == "clear" || kind == "upload" || kind.ends_with("_upload") }

/// longwin (2026-11-10): the same claim as concat_host_checked's own proof (only the host check reads the vector the
/// launch `blocks[bname][j]` writes), proved along the plan's execution order. Used as a second way to pass: a launch is
/// host-checked when the proof in concat_host_checked OR this one holds, so no plan that passed before can fail now.
/// THE ORDER: the plan's "cadence" (one list of block names per step). It is used only when it names every block exactly
/// once; then a forward pass, whichever boundary it starts from, runs the blocks in that order and each at most once, so
/// only the operations AFTER the launch can read what it wrote.
/// THE PROOF: walk those operations. Among the ones that name the vector's bytes (any nested {"allocation","offset"}
/// inside them), the exact "checked..." download may come any number of times; the first operation of any other kind must
/// START A NEW ARRAY at the vector's first byte, and the check must have come before it. The plan ending first also
/// proves it. A new array starts there when the operation is
/// * a host write whose destination is exactly the first byte (clear, upload, "*_upload": as in concat_host_checked);
/// * a native_cub_reduce whose work area or result is exactly the first byte, with input and count elsewhere (the work
///   area is CuPy's own fresh allocation, the result an int32 or int64 word: native_cub::word_bytes admits no other);
/// * a cupy_copy__<in>_<out> launch whose only argument naming the bytes is argument 1 (out0, which "out0 = in0" writes
///   and never reads) at exactly the first byte, with <out> an integer, bool or float32 type;
/// * an acoustic_cofrz_v841 launch (nlev, rdzw, cofrz: cofrz[k] = rdzw[k], written and never read) whose only argument
///   naming the bytes is cofrz at exactly the first byte, with nlev * 4 bytes longer than the vector.
/// WHY THAT IS A NEW ARRAY: the vector holds doubles. An array of another element type, or one longer than the vector,
/// that starts at its first byte is neither the vector nor a view of it, so the pool has handed the block out again,
/// which it can only do once the vector and every view of it are dead (the pool fact concat_host_checked already rests
/// on). A float64 copy output is NOT accepted: it could be an assignment into the vector itself. Any other kernel, or a
/// download that is not the exact check, fails the proof.
/// What the forward does (hexcore cuda_driver.py, the deferred check): cp.asnumpy(cp.concatenate(flags)).tolist() in
/// one expression; the concatenated vector is never bound to a name.
/// THE BREAKAGE THIS PREVENTS: a 30-step 120 km package (a test machine <local path>, 24 launches
/// of the two check-vector modules b15be0f4 and 82af2f9c). The pool gives the dead vector's bytes to a device-written
/// array in 20 of them (an int64 index array filled word by word, the cofrz column, a reduce result), and for 13 the
/// order-blind rule also found an EARLIER step's use of the same pool block. The module failed the proof, its launches
/// were Refused ("element type Some(\"double\") not established"), and every window on a long package stopped at step 1
/// (runs/dot30-0331p). The 4-step pk120e4 passed only because its four launches happen to be followed by host writes.
fn concat_dead_in_order(envelope: &serde_json::Value, bname: &str, j: usize) -> bool {
    use serde_json::Value;
    fn refs(v: &Value, out: &mut Vec<(u64, u64)>) {
        match v {
            Value::Object(m) => {
                if let (Some(a), Some(o)) = (m.get("allocation").and_then(Value::as_u64), m.get("offset").and_then(Value::as_u64)) { out.push((a, o)); }
                for x in m.values() { refs(x, out); }
            },
            Value::Array(a) => for x in a { refs(x, out); },
            _ => {},
        }
    }
    (|| -> Option<bool> {
        let blocks = envelope.get("blocks")?.as_object()?;
        let mut order: Vec<&str> = Vec::new();
        for step in envelope.get("cadence")?.as_array()? { for b in step.as_array()? { order.push(b.as_str()?); } }
        let mut seen = std::collections::HashSet::new();
        if order.len() != blocks.len() || !order.iter().all(|b| blocks.contains_key(*b) && seen.insert(*b)) { return None; }
        let at = order.iter().position(|b| *b == bname)?;
        let y = blocks.get(bname)?.as_array()?.get(j)?.get("arguments")?.as_array()?.get(2)?;
        let hex = y.get("bytes_hex")?.as_str()?;
        if hex.len() != 64 || !hex.is_ascii() { return None; }
        let word = |k: usize| u64::from_str_radix(&(0..8).rev().map(|b| &hex[(k * 8 + b) * 2..(k * 8 + b) * 2 + 2]).collect::<String>(), 16).ok();
        let (size, stride) = (word(1)?, word(3)?);
        let p = y.get("pointers")?.as_array()?;
        if p.len() != 1 || p[0].get("byte_offset")?.as_u64()? != 0 { return None; }
        let t = p[0].get("target")?;
        let (al, off) = (t.get("allocation")?.as_u64()?, t.get("offset")?.as_u64()?);
        let nb = size.checked_mul(stride)?;
        if nb == 0 { return None; }
        let inside = |v: &Value| { let mut r = Vec::new(); refs(v, &mut r); r.iter().any(|&(a, o)| a == al && o >= off && o < off + nb) };
        let inside_opt = |v: Option<&Value>| v.map_or(false, |v| inside(v));
        let start = |d: Option<&Value>| d.and_then(|d| Some((d.get("allocation")?.as_u64()?, d.get("offset")?.as_u64()?))) == Some((al, off));
        // the one device address of an argument: a raw device pointer, or a by-value array with one pointer at byte 0
        let target = |a: &Value| -> Option<Value> {
            if let Some(t) = a.get("target") { return Some(t.clone()); }
            let p = a.get("pointers")?.as_array()?;
            if p.len() != 1 || p[0].get("byte_offset")?.as_u64()? != 0 { return None; }
            p[0].get("target").cloned()
        };
        let mut checked = false;
        for (n, b) in order[at..].iter().enumerate() {
            let ops = blocks.get(*b)?.as_array()?;
            for q in ops.iter().skip(if n == 0 { j + 1 } else { 0 }) {
                if !inside(q) { continue; }
                let kind = q.get("kind").and_then(Value::as_str).unwrap_or("");
                if kind == "device_to_host" {
                    let s = q.get("source")?;
                    let exact = s.get("allocation")?.as_u64()? == al && s.get("offset")?.as_u64()? == off && q.get("bytes")?.as_u64()? == nb;
                    if !exact || !q.get("host_consumer")?.as_str()?.starts_with("checked") { return Some(false); }
                    checked = true; continue;
                }
                let fresh = if fresh_array_kind(kind) { start(q.get("destination")) }
                else if kind == "native_cub_reduce" {
                    let (w, o) = (q.get("workspace"), q.get("output"));
                    (start(w) || start(o)) && (start(w) || !inside_opt(w)) && (start(o) || !inside_opt(o))
                        && !inside_opt(q.get("input")) && !inside_opt(q.get("count"))
                } else if kind == "kernel" {
                    let f = q.get("function").and_then(Value::as_str).unwrap_or("");
                    let args = q.get("arguments")?.as_array()?;
                    let hits: Vec<usize> = args.iter().enumerate().filter(|(_, a)| inside(a)).map(|(i, _)| i).collect();
                    if let Some(types) = f.strip_prefix("cupy_copy__") {
                        hits == [1] && start(target(&args[1]).as_ref())
                            && matches!(types.rsplit('_').next().unwrap_or(""), "bool" | "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "int64" | "uint64" | "float32")
                    } else if f == "acoustic_cofrz_v841" {
                        let nlev = args.first().filter(|a| a.get("kind").and_then(Value::as_str) == Some("bytes"))
                            .and_then(|a| a.get("bytes_hex")).and_then(Value::as_str).filter(|h| h.len() == 8 && h.is_ascii())
                            .and_then(|h| u32::from_str_radix(&(0..4).rev().map(|b| &h[b * 2..b * 2 + 2]).collect::<String>(), 16).ok());
                        hits == [2] && args.len() == 3 && args[2].get("kind").and_then(Value::as_str) == Some("device")
                            && start(target(&args[2]).as_ref()) && nlev.map_or(false, |n| n as u64 * 4 > nb)
                    } else { false }
                } else { false };
                return Some(checked && fresh);
            }
        }
        Some(checked)
    })().unwrap_or(false)
}

pub fn concat_host_checked(envelope: &serde_json::Value) -> std::collections::HashSet<String> {
    use serde_json::Value;
    fn refs(v: &Value, out: &mut Vec<(u64, u64)>) {
        match v {
            Value::Object(m) => {
                if let (Some(a), Some(o)) = (m.get("allocation").and_then(Value::as_u64), m.get("offset").and_then(Value::as_u64)) { out.push((a, o)); }
                for x in m.values() { refs(x, out); }
            },
            Value::Array(a) => for x in a { refs(x, out); },
            _ => {},
        }
    }
    let mut ok: HashMap<String, bool> = HashMap::new();
    let Some(blocks) = envelope.get("blocks").and_then(Value::as_object) else { return Default::default() };
    for (bname, ops) in blocks.iter() {
        let Some(ops) = ops.as_array() else { continue };
        for (j, op) in ops.iter().enumerate() {
            if op.get("kind").and_then(Value::as_str) != Some("kernel") || op.get("function").and_then(Value::as_str) != Some("cupy_concatenate_same_size") { continue; }
            let Some(module) = op.get("module").and_then(Value::as_str) else { continue };
            let proved = (|| -> Option<bool> {
                let y = op.get("arguments")?.as_array()?.get(2)?;
                let hex = y.get("bytes_hex")?.as_str()?;
                if hex.len() != 64 || !hex.is_ascii() { return None; }
                // little-endian u64 word k of the by-value CArray
                let word = |k: usize| u64::from_str_radix(&(0..8).rev().map(|b| &hex[(k * 8 + b) * 2..(k * 8 + b) * 2 + 2]).collect::<String>(), 16).ok();
                let (size, stride) = (word(1)?, word(3)?);
                let p = y.get("pointers")?.as_array()?;
                if p.len() != 1 || p[0].get("byte_offset")?.as_u64()? != 0 { return None; }
                let t = p[0].get("target")?;
                let (al, off) = (t.get("allocation")?.as_u64()?, t.get("offset")?.as_u64()?);
                let nb = size.checked_mul(stride)?;
                if nb == 0 { return None; }
                let mut checked = false;
                for q in &ops[j + 1..] {
                    let mut r = Vec::new(); refs(q, &mut r);
                    if !r.iter().any(|&(a, o)| a == al && o >= off && o < off + nb) { continue; }
                    let kind = q.get("kind").and_then(Value::as_str).unwrap_or("");
                    if kind == "device_to_host" {
                        let s = q.get("source")?;
                        let exact = s.get("allocation")?.as_u64()? == al && s.get("offset")?.as_u64()? == off && q.get("bytes")?.as_u64()? == nb;
                        if !exact || !q.get("host_consumer")?.as_str()?.starts_with("checked") { return Some(false); }
                        checked = true; continue;
                    }
                    let d = q.get("destination");
                    let fresh = (fresh_array_kind(kind)
                        && d.and_then(|d| d.get("allocation")).and_then(Value::as_u64) == Some(al)
                        && d.and_then(|d| d.get("offset")).and_then(Value::as_u64) == Some(off))
                        || copy_plan_fill_starts_at(q, al, off, nb);
                    return Some(checked && fresh);
                }
                if !checked { return Some(false); }
                // No new array started at the output inside this step, so the vector may still be alive when the step
                // ends. Then no operation anywhere else in the plan may name its bytes before a new array starts exactly
                // there: every other block (whatever order the blocks run in), and this block's operations before the
                // launch (a block may run again). Anything else, a kernel that writes there included, fails the proof.
                for (other_name, other) in blocks.iter() {
                    let other = other.as_array()?;
                    let upto = if other_name == bname { j } else { other.len() };
                    let first = other[..upto].iter().find(|q| { let mut r = Vec::new(); refs(q, &mut r); r.iter().any(|&(a, o)| a == al && o >= off && o < off + nb) });
                    if let Some(q) = first {
                        let d = q.get("destination");
                        let fresh = fresh_array_kind(q.get("kind").and_then(Value::as_str).unwrap_or(""))
                            && d.and_then(|d| d.get("allocation")).and_then(Value::as_u64) == Some(al)
                            && d.and_then(|d| d.get("offset")).and_then(Value::as_u64) == Some(off);
                        if !fresh { return Some(false); }
                    }
                }
                Some(true)
            })().unwrap_or(false) || concat_dead_in_order(envelope, bname, j);
            let e = ok.entry(module.to_string()).or_insert(true);
            *e = *e && proved;
        }
    }
    ok.into_iter().filter(|(_, v)| *v).map(|(k, _)| k).collect()
}

/// P8 DevRow (cuda/p8.cu struct P8Row).
#[repr(C)] #[derive(Clone, Copy, Debug)]
pub struct DevRow { pub dst: i64, pub src: i64, pub ds: [i64; 4], pub ss: [i64; 4], pub n: [i64; 3], pub count: i64 }
/// P8 DevView (cuda/p8.cu struct P8View).
#[repr(C)] #[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevView { pub base: i64, pub size: i64, pub ndim: i32, pub pad: i32, pub shape: [i64; 4], pub stride: [i64; 4] }

#[derive(Clone, Debug)]
pub enum R {
    /// Kernel with an ad.entry/v1 derivative.
    Entry,
    /// Copy rows (post-transport copy, glue copy plan, cuBLAS transpose, broadcasting view copy): P8 rows.
    Rows { rows: Vec<DevRow>, label: &'static str },
    /// Elementwise float copy (casts exact on the shadow).
    View { dst: DevView, src: DevView },
    /// Overwrite with state-independent data on a strided view.
    ZeroView { dst: DevView, label: String },
    /// out = max(c, x): x's primal span (allocation, first word, words) is taped before the launch.
    Max { out: DevView, x: DevView, prim: (u64, usize, usize), xv: DevView, c: f32 },
    /// dst = mask ? src : dst; the mask's byte span (allocation, first byte, bytes) is taped before the launch.
    Where { dst: DevView, src: DevView, prim: (u64, usize, usize), mv: DevView },
    /// out = c * in (contiguous): element indices.
    Scale { input: i64, output: i64, n: usize, c: f64 },
    /// dst <- src over n words.
    Copy { dst: i64, src: i64, n: usize },
    /// y += c * x over n words (accumulate): tangent the same axpy, reverse x_bar += c * y_bar with y_bar kept.
    Axpy { y: i64, x: i64, n: usize, c: f64 },
    /// multidevice-runner: the source half of a cross-rank copy (n words at shadow element src). Tangent: send the words;
    /// reverse: src += the receiving rank's cotangent.
    PeerSend { src: i64, n: usize, key: String },
    /// multidevice-runner: the destination half. Tangent: dst <- the sending rank's words; reverse: send dst's cotangent
    /// and zero it.
    PeerRecv { dst: i64, n: usize, key: String },
    /// dst overwritten by a constant (clear, fill, immutable upload, fresh allocation, host-frame constant).
    Zero { ranges: Vec<(i64, usize)>, label: String },
    /// Held upload (a held scheme's step result, a host value): zero tangent, cotangent dropped; per origin.
    Held { ranges: Vec<(i64, usize)>, origin: String },
    /// No floating state written (or a held physics launch): identity both ways; counted by label.
    Identity(String),
    /// No exact rule: identity both ways, listed by name. A kind named with --refuse is NOT this unless what it writes
    /// is unknown (hold_refused): it runs as its own rule with the derivative held (fill rows, ZeroView, Zero).
    Refused(String),
    /// Frozen FCT limiter (--fct-frozen 1): fct_minmax_source with the limiter bounds held constant. `e` are the shadow
    /// element indices of old, source, rho_old, source_old, minimum, maximum; `prim` the primal spans of source and
    /// rho_old (allocation, first word, words), taped before the launch.
    FctMinmax { nt: i32, nl: i32, nc: i32, dt: f32, e: [i64; 6], prim: [(u64, usize, usize); 2] },
    /// L2 physics row (P3 MYNN-EDMF, P4 surface layer / GSL drag) bound through the float32-shadow adapter
    /// (--native-schemes 1, window.rs): its float64 entries run on widened copies of the window shadows.
    Native(String),
    /// TEMPO's whole-launch derivative (tempo_warp_steps, lane L5 tempo_adw: warp shape, per-substep checkpoints,
    /// executed-branch tape) bound in the window (--tempo-live 1, window.rs `tempo_apply`): the launch's inputs are
    /// snapshot before it runs, the tangent / adjoint run on float64 copies of the window's float32 shadows.
    Tempo(String),
    /// TEMPO per-column launch (tempo_precip) bound to P5's per-thread entry (--tempo-precip 1, window.rs):
    /// its float64 tape runs on widened copies of the window shadows, outputs rounded back.
    TempoSerial(String),
    /// --tempo-small (window.rs, ad/held-tempo_radar): a TEMPO per-column launch bound to the P5 per-thread entry
    /// (tempo_adw::TempoSmall, float64 tape) through the float32-shadow adapter. Without the flag the launch keeps its
    /// rule below (tempo_radar: inactive, an output diagnostic with no prognostic feedback).
    TempoSmall(String),
}

/// D1 (g2, 2026-11-11): the rule of a kind the run froze with --refuse. `own` is the rule the kind would have had,
/// `row` the shadow words its kernel-set row's tangent entries write (None: no row). The kind is HELD: the tangent of
/// every word it writes becomes zero and the cotangent of those words is consumed, the transpose of that. Copy rows
/// become fill rows; a view, max, scale or copy zeroes its destination; a row zeroes its written tangent views; a
/// kind that writes no float word stays identity, which is exact for it.
/// THE BREAKAGE THIS PREVENTS: --refuse ran the kind as identity both ways, so the tangent words it would have
/// written kept whatever bytes the pooled allocation held, and a later live rule carried them into the model. Arm
/// g2-u2-unpackonly (only rrfs_radiation_unpack live, the radiation bands refused) stopped non-finite at step 11;
/// with unpack refused too (g2-u1-nounpack) the stale words stayed where they were and the arm equalled the base.
/// A kind with no rule and no row (what it writes is unknown) keeps identity and says so: R::Refused, which the
/// window lists in its receipt under refuse_list_identity.
pub(crate) fn hold_refused(f: &str, why: &str, own: Result<R, String>, row: impl FnOnce() -> Option<Result<Vec<(i64, usize)>, String>>) -> R {
    let label = format!("refused:{f}");
    let unknown = match own {
        Ok(R::Rows { mut rows, label }) => { for r in rows.iter_mut() { r.src = -1; } return R::Rows { rows, label }; },
        Ok(R::View { dst, .. }) | Ok(R::ZeroView { dst, .. }) | Ok(R::Max { out: dst, .. }) => return R::ZeroView { dst, label },
        Ok(R::Scale { output, n, .. }) => return R::Zero { ranges: vec![(output, n)], label },
        Ok(R::Copy { dst, n, .. }) => return R::Zero { ranges: vec![(dst, n)], label },
        Ok(R::Axpy { y, n, .. }) => return R::Zero { ranges: vec![(y, n)], label },
        Ok(R::Zero { ranges, .. }) | Ok(R::Held { ranges, .. }) => return R::Zero { ranges, label },
        Ok(R::Identity(s)) => return R::Identity(s),
        Ok(R::Entry) => "row".to_string(),
        Ok(R::Refused(s)) => s,
        Ok(R::Where { .. }) => "a masked write (where)".to_string(),
        Ok(_) => "a scheme or limiter rule".to_string(),
        Err(e) => e,
    };
    // The text must not read like an unlisted refusal (window.rs stops on those two endings).
    let ident = |e: String| R::Refused(format!("{f}: {why} (identity kept, writes unknown: {})", e.replace(": entry not dispatchable: ", ", entry not dispatchable, ")));
    match row() {
        Some(Ok(ranges)) if !ranges.is_empty() => R::Zero { ranges, label },
        Some(Ok(_)) => ident("its row's tangent entries write no tangent view".into()),
        Some(Err(e)) => ident(format!("its row's written views do not resolve: {e}")),
        None => ident(format!("no kernel-set row; own rule: {unknown}")),
    }
}

/// D1b (g2, 2026-11-11): a kind named with --refuse that is a native physics row with no kernel-set row and no
/// rule here (P7 gas optics: rlw_taumol_batched, rsw_taumol_b). hold_refused cannot see what such a kind writes and
/// keeps identity (`r` is then R::Refused); live the kind runs as native:P7, whose schemes.json row names its float32
/// outputs. `written` is those outputs as shadow words (schemes::Native::refused_written; None: not such a row). The
/// kind is held on them: tangent zero, cotangent consumed, like every other listed kind.
/// THE BREAKAGE THIS PREVENTS: refused, the two kinds left the tangent words of taug / fracs / taur at whatever bytes
/// their pooled allocations held, and any live reader downstream would carry them into the model, as
/// rrfs_radiation_unpack did in arm g2-u2-unpackonly (non-finite at step 11). Found by the round-1 checker (00:53Z).
pub(crate) fn hold_refused_native(f: &str, r: R, written: Option<Result<Vec<(i64, usize)>, String>>) -> R {
    match (r, written) {
        (R::Refused(_), Some(Ok(ranges))) if !ranges.is_empty() => R::Zero { ranges, label: format!("refused:{f}") },
        (R::Refused(s), Some(Ok(_))) => R::Refused(format!("{s}; its native row names no float32 output")),
        (R::Refused(s), Some(Err(e))) => R::Refused(format!("{s}; its native row's outputs do not resolve ({e})")),
        (r, _) => r,
    }
}

/// D1c (g2, 2026-11-11): how a kind named with --refuse ran, as the receipt says it (window.json
/// refuse_list_rule). A text that starts with IDENTITY is a kind that ran as identity although it writes, or may
/// write, float words: the window lists exactly those kinds in refuse_list_identity (`refuse_ran_identity`).
/// THE BREAKAGE THIS PREVENTS: a listed kind whose rule is the class-held identity ("held:<scheme>": a held physics
/// launch, wrules `"held" => Identity` and the tempo_ / mynnsfc_ / ugwp_oro_ / oracle_ fallback) was written as
/// "identity (no float word written, or class-held)" and left out of refuse_list_identity, so the receipt said no
/// listed kind passed stale tangent words when one could. Found by the round-1 checker (00:53Z).
pub(crate) fn refuse_how(r: &R) -> String {
    match r {
        R::Rows { rows, .. } => format!("held: {} copy rows run as fill rows (destination tangent zero, cotangent consumed)", rows.len()),
        R::ZeroView { .. } => "held: destination view zeroed, cotangent consumed".to_string(),
        R::Zero { ranges, .. } => format!("held: tangent of the {} view(s) it writes zeroed, cotangent consumed", ranges.len()),
        R::Identity(s) if s.starts_with("held:") => format!("IDENTITY (class-held physics launch: the float words it writes keep their tangent): {s}"),
        R::Identity(s) => format!("identity (no float word written): {s}"),
        R::Refused(s) => format!("IDENTITY: {s}"),
        _ => "?".to_string(),
    }
}
/// True when `how` (refuse_how) says the kind ran as identity over float words it writes or may write.
pub(crate) fn refuse_ran_identity(how: &str) -> bool { how.starts_with("IDENTITY") }

pub struct Ctx<'a> {
    pub base_el: &'a HashMap<u64, i64>,
    pub kernels: &'a HashMap<String, crate::deriv::KernelDeriv>,
    pub classes: &'a HashMap<String, String>,
    pub schemes: &'a HashMap<String, String>,
    pub refuse: &'a HashMap<String, String>,
    /// --fct-frozen 1: the FCT limiter coefficients are frozen from the trajectory (see `fct_frozen`).
    pub fct_frozen: bool,
    /// --mesh-constants 1: see `mesh_scaling`.
    pub mesh_constants: bool,
}

/// Damping is required by the complete multi-rank operator. An identity substitute can pass a dot test
/// while omitting every damping partial (Wa-1p on 947aaa09 refused 96 selected launches).
pub(crate) fn require_damping_row(function: &str, present: bool, unsupported: Option<&str>, refused: Option<&str>, entries: bool) -> Result<(), String> {
    if !matches!(function, "divergence_damping_f32" | "divergence_damping_selected_f32") { return Ok(()); }
    let reason = if let Some(why) = refused { Some(why) }
        else if !present { Some("no derivative row") }
        else if let Some(why) = unsupported { Some(why) }
        else if !entries { Some("missing tangent or adjoint entry") }
        else { None };
    match reason {
        Some(why) => Err(format!("DEFECT: {function}: required damping derivative unavailable: {why}")),
        None => Ok(()),
    }
}

/// Kernels whose output is a constant of the module image: (function, bytes written through argument 0). Each entry is
/// checked against its source and SASS before it is listed.
/// * tempo_warp_sizes (hexcore/mpasphys/rrfs/tempo_v303.py:4024): `out[0]=sizeof(TempoWork); out[1]=sizeof(TempoState);
///   out[2]=sizeof(TempoBlob)`, three uint64. SASS of pkg18 module 477158923b00908c: three MOV.64 immediates
///   (0xfe30, 0x1078, 0x3a647620) and three STG.E.64 through the one pointer parameter, no load. The host reads the
///   words back (checked_value, tempo_v303.py:4312) to size TEMPO's work areas; the pool slot is then rebound as a
///   TEMPO pointer table. Grid 1x1x1, block 1x1x1, step 1 only.
pub fn image_constant(f: &str) -> Option<usize> {
    match f { "tempo_warp_sizes" => Some(24), _ => None }
}

/// A CuPy CArray by value: target and the view (byte strides).
pub struct CArr { pub alloc: u64, pub off: usize, pub size: i64, pub shape: Vec<i64>, pub strides: Vec<i64> }
pub fn carr(a: &submit::Argument) -> Option<CArr> {
    let submit::Argument::Patched { bytes, pointers } = a else { return None };
    if pointers.len() != 1 || pointers[0].byte_offset != 0 || bytes.len() < 16 || (bytes.len() - 16) % 16 != 0 { return None; }
    let w = |k: usize| i64::from_le_bytes(bytes[8 * k..8 * k + 8].try_into().unwrap());
    let ndim = (bytes.len() - 16) / 16;
    Some(CArr { alloc: pointers[0].target.allocation, off: pointers[0].target.offset, size: w(1),
                shape: (0..ndim).map(|d| w(2 + d)).collect(), strides: (0..ndim).map(|d| w(2 + ndim + d)).collect() })
}
/// `accumulate(int n, int begin, int end, float dt, float bucket, int ringing, const float* rate, float* total, int* count)`:
/// total[i] += rate[i] * dt for begin <= i < min(end, n). Derivative: d_total += dt * d_rate there (the bucket subtraction
/// and the counter do not change it); the reverse adds dt * total_bar into rate_bar and keeps total_bar.
pub fn accumulate_rule(args: &[submit::Argument], el: impl Fn(u64, usize) -> Result<i64, String>) -> Result<R, String> {
    if args.len() != 9 { return Err(format!("{} arguments, not 9 (n, begin, end, dt, bucket, ringing, rate, total, count)", args.len())); }
    let word = |k: usize| match &args[k] { submit::Argument::Bytes(b) if b.len() == 4 => Ok(<[u8; 4]>::try_from(&b[..4]).unwrap()),
                                           _ => Err(format!("argument {k} is not a 4-byte scalar")) };
    let (n, begin, end) = (i32::from_le_bytes(word(0)?) as i64, i32::from_le_bytes(word(1)?) as i64, i32::from_le_bytes(word(2)?) as i64);
    let dt = f32::from_le_bytes(word(3)?);
    let dev = |k: usize| match &args[k] { submit::Argument::Device(s) if s.offset % 4 == 0 => Ok(s),
                                          _ => Err(format!("argument {k} is not an aligned device pointer")) };
    let (rate, total) = (dev(6)?, dev(7)?);
    let hi = end.min(n);
    if begin < 0 || hi < begin { return Err(format!("range [{begin}, {hi}) of {n}")); }
    if !dt.is_finite() { return Err(format!("dt {dt}")); }
    let (x, y) = (el(rate.allocation, rate.offset)? + begin, el(total.allocation, total.offset)? + begin);
    let m = (hi - begin) as usize;
    if m > 0 && x < y + m as i64 && y < x + m as i64 { return Err("rate and total overlap".into()); }
    Ok(R::Axpy { y, x, n: m, c: dt as f64 })
}
fn scalar_f32(a: &submit::Argument) -> Option<f32> {
    match a { submit::Argument::Bytes(b) if b.len() >= 4 => Some(f32::from_le_bytes(b[..4].try_into().unwrap())), _ => None }
}

impl Ctx<'_> {
    pub fn el(&self, alloc: u64, off: usize) -> Result<i64, String> {
        Ok(self.base_el.get(&alloc).ok_or_else(|| format!("allocation {alloc} has no shadow"))? + (off / 4) as i64)
    }
    /// Shadow view of a CArray of `item`-byte elements. The view starts at the shadow word of its first byte; a
    /// float64 view steps one shadow word per element (byte stride / 8), the convention the entries' float64
    /// bindings over the float32 sweep shadows use (deriv.rs `widen`: element i of a view at byte offset o is shadow
    /// word o/4 + i), so a whole-array float64 copy and an entry binding the same array agree word for word.
    fn view(&self, c: &CArr, item: i64) -> Result<DevView, String> {
        if c.shape.len() > 4 { return Err(format!("view rank {} above 4", c.shape.len())); }
        if c.off % 4 != 0 { return Err("view origin not word aligned".into()); }
        if item != 4 && item != 8 { return Err(format!("{item}-byte elements have no float shadow rule")); }
        let mut v = DevView { base: self.el(c.alloc, c.off)?, size: c.shape.iter().product::<i64>().max(0), ndim: c.shape.len() as i32,
                              pad: 0, shape: [1; 4], stride: [0; 4] };
        for (k, (&n, &s)) in c.shape.iter().zip(&c.strides).enumerate() {
            if s % item != 0 { return Err(format!("stride {s} not a multiple of the {item}-byte element")); }
            v.shape[k] = n; v.stride[k] = s / item;
        }
        if v.ndim == 0 { v.ndim = 1; }
        Ok(v)
    }
    /// Byte span [lo, hi) of a view relative to its allocation (item bytes per element).
    fn span(c: &CArr, item: i64) -> (i64, i64) {
        let (mut lo, mut hi) = (c.off as i64, c.off as i64 + item);
        for (n, s) in c.shape.iter().zip(&c.strides) { let r = (n - 1).max(0) * s; if r < 0 { lo += r } else { hi += r } }
        (lo, hi)
    }
    fn overlap(a: &CArr, b: &CArr) -> bool {
        if a.alloc != b.alloc { return false; }
        let (x, y) = (Self::span(a, 4), Self::span(b, 4));
        x.0 < y.1 && y.0 < x.1
    }
    pub fn scheme(&self, f: &str) -> String {
        if let Some(s) = self.schemes.get(f) { return s.clone(); }
        if f.starts_with("tempo_") { "tempo".into() } else if f.starts_with("mynnsfc_") { "mynn_sfc".into() }
        else if f.starts_with("ugwp_oro_") { "gsl_drag".into() } else if f.starts_with("oracle_") || f.starts_with("ruc_") { "ruc".into() }
        else if f == "rrfs_mynnedmf_run_l16" || f == "rrfs_mynnedmf_run_l4" { "mynn_edmf".into() } else { "?".into() }
    }

    /// The rule of one kernel launch, from its live arguments.
    pub fn kernel(&self, ex: &Execution, f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3]) -> Result<R, String> {
        let damping = self.kernels.get(f);
        require_damping_row(f, damping.is_some(), damping.and_then(|k| k.unsupported.as_deref()),
            self.refuse.get(f).map(String::as_str),
            damping.is_some_and(|k| !k.tl_order.is_empty() && !k.ad_order.is_empty()))?;
        // D1 (g2, 2026-11-11): a kind named with --refuse is HELD: the tangent of every word it writes is zero and
        // the cotangent of those words is consumed (hold_refused). The kind's own rule says what it writes.
        if let Some(why) = self.refuse.get(f) {
            let own = self.kernel_rule(ex, f, args, grid, block);
            return Ok(hold_refused(f, why, own, || self.kernels.get(f).map(|kd| self.row_written(kd, args, grid, block))));
        }
        self.kernel_rule(ex, f, args, grid, block)
    }
    /// The shadow words (first element, count) a row's tangent entries write at this launch: the float32 views of its
    /// `tangent` out / inout bindings (deriv::KernelDeriv::written_targets), from the spec and the launch alone.
    fn row_written(&self, kd: &crate::deriv::KernelDeriv, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3]) -> Result<Vec<(i64, usize)>, String> {
        let li = crate::deriv::LaunchInfo { args, grid, block, shared: 0 };
        kd.written_targets(&li)?.into_iter().map(|(_, al, off, n)| Ok((self.el(al, off)?, n))).collect()
    }
    /// The rule of one kernel launch that --refuse does not name (see `kernel`).
    fn kernel_rule(&self, ex: &Execution, f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3]) -> Result<R, String> {
        if self.fct_frozen && (f == "fct_scale" || f == "fct_minmax_source") {
            return Ok(self.fct_frozen(ex, f, args).unwrap_or_else(|e| R::Refused(format!("{f}: fct-frozen: {e}"))));
        }
        match f {
            // P8 row form for both directions (overrides the 11-02 glue entries, whose transpose is serial and
            // whose copy-plan descriptor table is mirrored as if every word were a pointer).
            "rrfs_post_transport_copy" => return Ok(self.post_transport(args).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            "rrfs_glue_copy_plan" => return Ok(self.copy_plan_rows(ex, args, grid[1]).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            // multirank (L20): the same-host halo pack/unpack (hexcore/partition_samehost_transport_v841.py _BYTE_COPY /
            // _WORD_COPY_2D), blockIdx.y = field: pure word moves, so P8 rows (tangent copies, reverse sums the halo
            // cotangent back into the owner through the packet and its DeviceCopy rule).
            "halo_copy_bytes" | "halo_copy_words_2d" =>
                return Ok(self.halo_copy_rows(ex, f, args, grid[1]).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            // RUC land tile preflight: two int32 results, an exact zero rule (Ctx::integer_flags).
            "ruc_land_tile_preflight" => return Ok(Self::integer_flags(f, args, |a, o| self.el(a, o), |a| ex.allocations.get(&a).map(|x| x.bytes))
                .unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            "rrfs_radiation_unpack" => return Ok(self.radiation_unpack_rows(args).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            // MPAS physics accumulations (hexcore/mpasphys/surface_update_v842.py `accumulate`: rainc with New Tiedtke on,
            // the radiation sums): total[i] = total[i] + rate[i] * dt over [begin, end), then total -= bucket where it
            // passes the bucket (a constant shift) and count += 1 (an integer). Exact linear rule (accumulate_rule).
            // THE BREAKAGE THIS PREVENTS: with cumulus on the window stopped at step 1 with "accumulate: no derivative
            // row and no rule", so no window check could run on a forecast with New Tiedtke on.
            "accumulate" if !self.kernels.contains_key(f) =>
                return Ok(accumulate_rule(args, |a, o| self.el(a, o)).unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))),
            _ => {}
        }
        if let Some(r) = Self::integer_result(f, args, |a, o| self.el(a, o)) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(r) = Self::module_constant(f, args) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(r) = Self::constant_tables(f, args, grid, block, |a, o| self.el(a, o)) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(r) = Self::regional_driving(f, args, grid, block, |a, o| self.el(a, o)) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if self.mesh_constants {
            if let Some(r) = Self::mesh_scaling(f, args, grid, block, |a, o| self.el(a, o)) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        }
        if let Some(r) = self.speczone_overwrite(ex, f, args, grid, block) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(r) = self.scrub_columns(ex, f, args, grid) { return Ok(r.unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))); }
        if let Some(kd) = self.kernels.get(f) {
            let validation_only = kd.spec["launch_abi"]["entries"].as_object().is_some_and(|e| e.is_empty())
                || (f.starts_with("validate_") && kd.functions.is_empty());
            if validation_only { return Ok(R::Identity(format!("inactive:{f} (validation-only row)"))); }
            if let Some(why) = &kd.unsupported { return Ok(R::Refused(format!("{f}: entry not dispatchable: {why}"))); }
            return Ok(R::Entry);
        }
        // Input-free image constants (image_constant): the only argument is the output, and every word written is an
        // immediate of the module image, so the launch overwrites its output with a state-independent constant: zero
        // tangent, cotangent stops (R::Zero), the exact derivative. Not a held scheme launch (held = identity both ways,
        // which would keep a stale tangent on the overwritten words).
        if let Some(bytes) = image_constant(f) {
            let Some(submit::Argument::Device(out)) = args.first() else {
                return Ok(R::Refused(format!("{f}: image constant whose output is not a device pointer argument")));
            };
            if args.len() != 1 || out.offset % 4 != 0 {
                return Ok(R::Refused(format!("{f}: image constant with {} arguments / unaligned output", args.len())));
            }
            return Ok(R::Zero { ranges: vec![(self.el(out.allocation, out.offset)?, bytes.div_ceil(4))],
                                label: format!("constant:{f} (input-free image constant, {bytes} bytes)") });
        }
        let arrays: Vec<CArr> = args.iter().filter_map(carr).collect();
        if let Some(d) = f.strip_prefix("cupy_copy__") {
            let t: Vec<&str> = d.split('_').collect();
            let float = |s: &str| s.starts_with("float");
            // No float destination: integer or bool result, nothing to differentiate (whatever the argument layout).
            if t.len() == 2 && !float(t[1]) { return Ok(R::Identity(format!("inactive:{f} (integer or bool result)"))); }
            if t.len() == 2 && arrays.len() >= 2 {
                let (src, dst) = (&arrays[0], &arrays[1]);
                let item = |s: &str| if s == "float64" { 8 } else if s == "float32" { 4 } else { 0 };
                if float(t[0]) && float(t[1]) { return Ok(self.copy_view(f, src, dst, item(t[0]), item(t[1]))); }
                if float(t[1]) {
                    return Ok(match self.view(dst, item(t[1])) { Ok(v) => R::ZeroView { dst: v, label: format!("inactive-source copy {f}") },
                                                     Err(e) => R::Refused(format!("{f}: {e}")) });
                }
                return Ok(R::Identity(format!("inactive:{f} (integer or bool result)")));
            }
            let item = |s: &str| if s == "float64" { 8 } else if s == "float32" { 4 } else { 0 };
            // g3-radland: `a[...] = x` with x a host scalar (CuPy names its type `float` or `int`): one float destination
            // view, state-independent words.
            if t.len() == 2 && float(t[1]) && arrays.len() == 1 {
                return Ok(match self.view(&arrays[0], item(t[1])) { Ok(v) => R::ZeroView { dst: v, label: format!("host-scalar fill {f}") },
                                                                    Err(e) => R::Refused(format!("{f}: {e}")) });
            }
            return Ok(R::Refused(format!("{f}: unexpected copy arguments")));
        }
        match f {
            "cupy_fill" => if let Some(dst) = arrays.last() {
                // CuPy names no dtype in `cupy_fill`; the fill value is passed by
                // value at the item width, so a 1- or 2-byte value is a bool / int8 / int16 array (the New Tiedtke column
                // masks). Such an array holds no float word and carries no tangent: identity both ways, as for the copies
                // with an integer or bool result above. Before this the float view refused it (stride 1 not a multiple of 4)
                // and the whole window stopped at step 1.
                if let Some(submit::Argument::Bytes(b)) = args.first() {
                    if b.len() < 4 { return Ok(R::Identity(format!("inactive:{f} ({}-byte item: bool or integer fill)", b.len()))); }
                }
                return Ok(match self.view(dst, 4) { Ok(v) => R::ZeroView { dst: v, label: "constant fill".into() }, Err(e) => R::Refused(format!("{f}: {e}")) });
            },
            "cupy_maximum__float_float32_float32" => if arrays.len() == 2 {
                let (x, out) = (&arrays[0], &arrays[1]);
                let c = args.iter().find_map(|a| if carr(a).is_none() { scalar_f32(a) } else { None }).ok_or("maximum constant")?;
                let (lo, hi) = Self::span(x, 4);
                let (ov, iv) = match (self.view(out, 4), self.view(x, 4)) { (Ok(o), Ok(i)) => (o, i), (Err(e), _) | (_, Err(e)) => return Ok(R::Refused(format!("{f}: {e}"))) };
                if lo < 0 || lo % 4 != 0 { return Ok(R::Refused(format!("{f}: unaligned primal span"))); }
                let mut xv = iv; xv.base = (x.off as i64 - lo) / 4;
                if ov.base != iv.base && Self::overlap(x, out) { return Ok(R::Refused(format!("{f}: partially overlapping in/out"))); }
                return Ok(R::Max { out: ov, x: iv, prim: (x.alloc, lo as usize / 4, ((hi - lo + 3) / 4) as usize), xv, c });
            },
            "cupy_copy_where__float32_bool_float32" => if arrays.len() == 3 {
                let (src, mask, dst) = (&arrays[0], &arrays[1], &arrays[2]);
                let (dv, sv) = match (self.view(dst, 4), self.view(src, 4)) { (Ok(d), Ok(s)) => (d, s), (Err(e), _) | (_, Err(e)) => return Ok(R::Refused(format!("{f}: {e}"))) };
                if Self::overlap(src, dst) && dv.base != sv.base { return Ok(R::Refused(format!("{f}: overlapping src/dst"))); }
                if mask.shape.len() > 4 { return Ok(R::Refused(format!("{f}: mask rank"))); }
                let (lo, hi) = Self::span(mask, 1);
                if lo < 0 { return Ok(R::Refused(format!("{f}: mask span"))); }
                let mut mv = DevView { base: mask.off as i64 - lo, size: mask.shape.iter().product::<i64>().max(0), ndim: mask.shape.len().max(1) as i32,
                                       pad: 0, shape: [1; 4], stride: [0; 4] };
                for (k, (&n, &s)) in mask.shape.iter().zip(&mask.strides).enumerate() { mv.shape[k] = n; mv.stride[k] = s; }
                return Ok(R::Where { dst: dv, src: sv, prim: (mask.alloc, lo as usize, (hi - lo) as usize), mv });
            },
            // g3-radland: x / c with c a host constant: d_out = d_x / c (the multiply rule's Scale, factor 1/c).
            "cupy_true_divide__float32_float_float32" => if arrays.len() == 2 {
                let (i, o) = (&arrays[0], &arrays[1]);
                let contiguous = |c: &CArr| { let mut e = 4; for d in (0..c.shape.len()).rev() { if c.shape[d] > 1 && c.strides[d] != e { return false; } e *= c.shape[d].max(1); } true };
                // the divisor is the one 4-byte by-value argument (CuPy passes a float32 scalar; a float64 one is refused)
                let fours: Vec<f32> = args.iter().filter_map(|a| match a { submit::Argument::Bytes(b) if b.len() == 4 => Some(f32::from_le_bytes(b[..4].try_into().unwrap())), _ => None }).collect();
                if fours.len() != 1 { return Ok(R::Refused(format!("{f}: divisor is not one float32 scalar"))); }
                let c = fours[0];
                if i.size != o.size || !contiguous(i) || !contiguous(o) { return Ok(R::Refused(format!("{f}: strided true_divide"))); }
                if i.alloc == o.alloc && i.off != o.off && Self::overlap(i, o) { return Ok(R::Refused(format!("{f}: partially overlapping true_divide"))); }
                if c == 0.0 || !c.is_finite() { return Ok(R::Refused(format!("{f}: divisor {c} (the forward is not finite)"))); }
                return Ok(R::Scale { input: self.el(i.alloc, i.off)?, output: self.el(o.alloc, o.off)?, n: i.size as usize, c: 1.0 / c as f64 });
            },
            "cupy_multiply__float32_float_float32" => if arrays.len() == 2 {
                let (i, o) = (&arrays[0], &arrays[1]);
                let contiguous = |c: &CArr| { let mut e = 4; for d in (0..c.shape.len()).rev() { if c.shape[d] > 1 && c.strides[d] != e { return false; } e *= c.shape[d].max(1); } true };
                let c = args.iter().find_map(|a| if carr(a).is_none() { scalar_f32(a) } else { None }).ok_or("multiply scalar")?;
                if i.size != o.size || !contiguous(i) || !contiguous(o) { return Ok(R::Refused(format!("{f}: strided multiply"))); }
                if i.alloc == o.alloc && i.off != o.off && Self::overlap(i, o) { return Ok(R::Refused(format!("{f}: partially overlapping multiply"))); }
                return Ok(R::Scale { input: self.el(i.alloc, i.off)?, output: self.el(o.alloc, o.off)?, n: i.size as usize, c: c as f64 });
            },
            _ => {}
        }
        let class = self.classes.get(f).map(String::as_str).unwrap_or("");
        match class {
            // COMMON-AD 87.1: the TEMPO order kernels come in as "held" from the P9 manifest (e2e held.rs: permutation),
            // but they write integer schedule words only, so their identity is their exact derivative, not a held
            // scheme. Labelled inactive so held:tempo counts only real TEMPO physics (rule unchanged: identity).
            "held" if matches!(f, "tempo_order_hist" | "tempo_order_scan" | "tempo_order_scatter") =>
                return Ok(R::Identity(format!("inactive:{f} (COMMON-AD 87.1 column-order permutation)"))),
            "held" => return Ok(R::Identity(format!("held:{}", self.scheme(f)))),
            "output-diagnostic" => return Ok(R::Identity(format!("inactive:{f} (output diagnostic)"))),
            "needs-rule" => return Ok(R::Refused(format!("{f}: P8 rule or glue entry not loaded"))),
            // g3-radland: a kernel-set data row for a launch whose every written word is bool or integer (the radiation
            // slot's CuPy masks, scans and integer arithmetic): no float state written, identity both ways.
            "inactive" => return Ok(R::Identity(format!("inactive:{f} (bool or integer result, kernel-set class)"))),
            _ => {}
        }
        // P8 rules.rs kernel_class: inactive (bool / integer results, validators, fixed geometry, diagnostics,
        // TEMPO column-order permutations) and pending (fused physics owned by P3-P6: held while unbound).
        const INACTIVE: &[&str] = &["cupy_less__float32_float32_bool", "cupy_not_equal__int32_int_bool", "cupy_cub_any_pass1",
            "cupy_cub_any_pass2", "hex_transport_species_words", "hex_transport_species_levels", "validate_density_v841",
            "validate_finite_array_v841_f32", "validate_recovered_v841_f32", "w_filter_geometry_v841_f32", "smooth_helicity",
            "split_helicity_pool", "vertex_helicity_gather_pool", "update_diag_pool", "mpas_output_diagnostics_v842",
            "oracle_diags_sfcdiags_ruclsm", "tempo_radar", "tempo_order_hist", "tempo_order_scan", "tempo_order_scatter",
            // diagnostic pool kernels (kernel-set class output-diagnostic for compute_diag / reset_diag)
            "compute_diag_cols", "reset_diag_pool",
            // S01b (3 km LAM): the regional solve's admission check (route table: inactive, diagnostic refusal only) and
            // CuPy's boolean all-reduction (like cub_any): bool or integer words only
            "solve_region_validate_v841", "cupy_cub_all_pass1", "cupy_cub_all_pass2",
            // seeds: the opt-in McICA column seed rule of the forward (hexcore/mpasphys/mcica_seed_rule.py,
            // --mcica-seeds column). The launch overwrites the uint32 seed words of a batch from integer keys only
            // (global cell id, model time, band, run seed): no float word is read or written, so identity both ways
            // is its exact derivative, and a perturbed forward replays it to the same words. THE BREAKAGE THIS
            // PREVENTS: a plan recorded with the option on listed the launch as "no derivative row and no rule",
            // one kind refused beyond the explicit list, which makes the window invalid by the program rule.
            "mpas_mcica_seeds_column"];
        if INACTIVE.contains(&f) || f.starts_with("validate_") { return Ok(R::Identity(format!("inactive:{f}"))); }
        // S01b (3 km LAM): a CuPy elementwise kernel named cupy_<op>__<in types>_<out type> whose output type is bool or
        // an integer writes no float word: nothing to differentiate (the cupy_copy__ rule above, for every op). THE
        // BREAKAGE THIS PREVENTS: the LAM window listed 12 such kinds (cupy_where__bool_int64_int_int64,
        // cupy_minimum__int64_int_int64, cupy_less__int64_int_bool, ...) as refused.
        if let Some(types) = f.strip_prefix("cupy_").and_then(|r| r.split_once("__")).map(|(_, t)| t) {
            let out = types.rsplit('_').next().unwrap_or("");
            let integer = out == "bool" || out == "int" || out.strip_prefix("int").or(out.strip_prefix("uint")).is_some_and(|w| w.parse::<u32>().is_ok());
            if integer && !f.starts_with("cupy_copy__") { return Ok(R::Identity(format!("inactive:{f} (bool or integer result)"))); }
        }
        if f.starts_with("tempo_") || f.starts_with("mynnsfc_") || f.starts_with("ugwp_oro_") || f.starts_with("oracle_")
            || f == "rrfs_mynnedmf_run_l16" || f == "rrfs_mynnedmf_run_l4" { return Ok(R::Identity(format!("held:{}", self.scheme(f)))); }
        Ok(R::Refused(format!("{f}: no derivative row and no rule")))
    }

    /// Setup launches whose only output is an integer (held lane rrfs_init_microphysics_top, 2026-11-06).
    /// rrfs_init_microphysics_top (TEMPO setup, RRFS 44c0849 mpas_atmphys_init.F:284-292, forward in
    /// hexcore/mpasphys/rrfs/driver_rrfs.py) writes ONE int32: the last level k whose running height
    /// 0.5/rdzw[0] + dzu[1] + ... + dzu[k-1] is <= top. It is piecewise constant in (top, rdzw, dzu), so its derivative
    /// is exactly zero wherever it exists: the tangent of the output word is zero and the reverse drops its cotangent
    /// (the exact transpose; Identity would pass the overwritten word's cotangent through and is not). No float word is
    /// written, no input is read for a derivative. At a switch point (running height == top) the level jumps by one: a
    /// step with no derivative, never approximated here; the margin on the captured inputs is in the lane's RESULT.md.
    /// Before this rule the launch fell through to "no derivative row and no rule" (Refused) if a window ever composed
    /// step-0 setup (pkg18 does not: 0 launches in plan.json d06ab544).
    fn integer_result(f: &str, args: &[submit::Argument], el: impl Fn(u64, usize) -> Result<i64, String>) -> Option<Result<R, String>> {
        if f != "rrfs_init_microphysics_top" { return None; }
        Some((|| {
            if args.len() != 5 { return Err(format!("{} arguments, not 5 (levels, top, rdzw, dzu, level)", args.len())); }
            let submit::Argument::Device(s) = &args[4] else { return Err("level (argument 4) is not a device pointer".into()) };
            if s.offset % 4 != 0 { return Err("level word not aligned".into()); }
            Ok(R::Zero { ranges: vec![(el(s.allocation, s.offset)?, 1)],
                         label: "integer result: microphysics top level (rrfs_init_microphysics_top)".into() })
        })())
    }
    /// Module-constant initializers (held lane ruc_tbq_init, 2026-11-06).
    /// ruc_tbq_init (RUC setup, RRFS-MPAS 1d6e370e module_ruc_land.F90:479-496 and module_ruc_ice.F90:364-380, forward in
    /// hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:86) takes NO arguments and writes only the module global
    /// `ruc_tbq_table[5001]` (the saturation-pressure table) from compiled-in constants: cq = 173.15 + 0.05 (k-1), exp of a
    /// constant expression, times 6.1153 * 0.62198. Nothing it reads is model state, so its derivative is exactly zero:
    /// the tangent of every table word is 0 and the reverse has no input to send a cotangent to. The table is not a
    /// driver allocation, so it has no window shadow; land and sea ice (oracle_ice_ruc_ice, oracle_land_ruc_land) read it
    /// as a primal constant, which is that same zero tangent. On the shadow arena the launch therefore writes no word:
    /// Identity both ways is the exact operator here, not a refusal. Its words do not depend on the launch either
    /// (explicitly rounded arithmetic; the lane's RESULT.md has the table sha256 from three images and two loads).
    /// THE BREAKAGE THIS PREVENTS: a window that composes the step-0 setup (pkg18 runs it as plan.module_initializers
    /// at load, sequence 14003, and never as a command) reached "no derivative row and no rule" and listed the launch as
    /// Refused. Refuses by name if the launch ever carries an argument: then it could write a shadowed word.
    fn module_constant(f: &str, args: &[submit::Argument]) -> Option<Result<R, String>> {
        if f != "ruc_tbq_init" { return None; }
        Some(if args.is_empty() {
            Ok(R::Identity("inactive:ruc_tbq_init (constant module table ruc_tbq_table, no shadowed word)".into()))
        } else {
            Err(format!("{} arguments, not 0: a module-constant initializer takes none", args.len()))
        })
    }
    /// RRTMG LW lookup-table setup (held lane mpas_init_tables, 2026-11-06).
    /// mpas_init_tables (MPAS module_ra_rrtmg_lw.F:8054-8079 rrtmg_lw_ini, forward in hexcore/mpasphys/rrtmg_lw_v842.py)
    /// reads NO input: thread i = blockIdx.x*blockDim.x+threadIdx.x (i <= 10000) overwrites tau[i], exp_table[i] and
    /// tfn[i] from i and compiled-in constants (t = i/1e4, tau = t/0.278/(1-t), exp = max(exp(-tau), 1e-20), tfn the Pade
    /// transition). Its derivative with respect to everything is exactly zero, so the rule is R::Zero on the 3 x 10001
    /// written words: tangent zeroed, cotangent dropped (the exact transpose; Identity would pass the overwritten words'
    /// cotangent through and is not). The fmaxf(.., 1e-20) clamp and the tau < 0.06 branch select between constants of the
    /// index only, so no frozen-coefficient treatment is needed. Words past 10000 (allocation tails) are not written and
    /// keep identity. Refused by name unless the launch has the 3 device-pointer arguments, word-aligned, every table has a
    /// window shadow, and grid.x*block.x covers the 10001 words (a narrower launch would leave words unwritten that
    /// R::Zero would clear). Before this rule a window composing step-0 setup reached "no derivative row and no rule"
    /// (Refused); pkg18 has no radiation launch (0 of plan.json d06ab544), so the 18-step window is unchanged.
    fn constant_tables(f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3],
                       el: impl Fn(u64, usize) -> Result<i64, String>) -> Option<Result<R, String>> {
        if f != "mpas_init_tables" { return None; }
        const WORDS: usize = 10001;
        Some((|| {
            if args.len() != 3 { return Err(format!("{} arguments, not 3 (tau, exp_table, tfn)", args.len())); }
            let span = grid[0] as u64 * block[0] as u64;
            if span < WORDS as u64 { return Err(format!("grid.x*block.x = {span} writes fewer than the {WORDS} table words")); }
            let mut ranges = Vec::with_capacity(3);
            for (k, name) in ["tau", "exp_table", "tfn"].iter().enumerate() {
                let submit::Argument::Device(s) = &args[k] else { return Err(format!("{name} (argument {k}) is not a device pointer")) };
                if s.offset % 4 != 0 { return Err(format!("{name} not word aligned")); }
                ranges.push((el(s.allocation, s.offset)?, WORDS));
            }
            Ok(R::Zero { ranges, label: "constant table: RRTMG LW tau/exp/tfn lookup tables (mpas_init_tables)".into() })
        })())
    }

    /// S01b (--mesh-constants 1): mixing_scaling_f32 (hexcore/cuda_horizontal.py:468) writes del2 = 1/meanDensity^0.25 and
    /// del4 = 1/meanDensity^0.75 per edge from the static mesh density only (or 1.0 without mesh scaling): fixed geometry,
    /// which the derivative contract holds inactive. So both outputs are state-independent constants: zero tangent,
    /// cotangent stops (R::Zero on del2 and del4, nedges words each). THE BREAKAGE THIS PREVENTS: the 3 km LAM window
    /// stopped at step 1 with "mixing_scaling_f32 tangent refused" (mixing_scaling_f32.ad.cu:17/:20): the row's tangent
    /// refuses a non-positive or non-finite mean density, which the LAM's outer edges (a garbage-cell neighbour) give.
    /// Refused unless the launch has the forward's 6 arguments, word-aligned outputs and grid.x*block.x >= nedges.
    fn mesh_scaling(f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3],
                    el: impl Fn(u64, usize) -> Result<i64, String>) -> Option<Result<R, String>> {
        if f != "mixing_scaling_f32" { return None; }
        Some((|| {
            if args.len() != 6 { return Err(format!("{} arguments, not 6", args.len())); }
            let n = match &args[2] { submit::Argument::Bytes(b) if b.len() == 4 => i32::from_le_bytes(b[..4].try_into().unwrap()) as i64,
                                     _ => return Err("nedges is not an int32 scalar".into()) };
            if n <= 0 || (grid[0] as i64) * (block[0] as i64) < n { return Err(format!("grid covers fewer than nedges = {n} edges")); }
            let mut ranges = Vec::new();
            for (k, name) in [(4usize, "del2"), (5, "del4")] {
                let submit::Argument::Device(s) = &args[k] else { return Err(format!("{name} is not a device pointer")) };
                if s.offset % 4 != 0 { return Err(format!("{name} not word aligned")); }
                ranges.push((el(s.allocation, s.offset)?, n as usize));
            }
            Ok(R::Zero { ranges, label: "mesh constant: mixing_scaling_f32 del2/del4 (static mesh density, --mesh-constants 1)".into() })
        })())
    }

    /// Limited-area driving state (S01 convective-3km, 2026-11-07). The four LBC launches of hexcore/cuda_regional_v841.py
    /// (MPAS-A v8.4.1 mpas_atm_boundaries.F:217-309 and :491-551) derive the boundary driving fields from the LBC files:
    /// regional_lbc_derive_v841 (rho_zz = lbc_rho/zz, rtheta_m = lbc_theta*rho_zz*(1 + rvord*lbc_qv)),
    /// regional_lbc_rho_edge_v841 (rho_edge = edge mean of rho_zz where both cells exist, ru = lbc_u*rho_edge),
    /// regional_lbc_tendency_v841 ((new - old)/interval) and regional_lbc_state_at_v841 (state - dt_remaining*tend).
    /// Every word they read is an LBC upload (the plan's `upload` rule: immutable input, zero tangent, cotangent stops),
    /// a value derived from one, or fixed geometry (zz, cellsOnEdge). The LBCs are forcing, not a control variable: the
    /// gradient is dJ/dx0 with the boundary data held, so every word these launches write has a zero tangent and its
    /// cotangent stops there (R::Zero on the written ranges: the exact transpose of zeroing the tangent; Identity would
    /// pass the overwritten words' cotangent through and is not). The lateral boundary's own sensitivity (dJ/dLBC) is
    /// therefore dropped, by declaration, and named in the receipt label. The specified- and relaxation-zone launches that
    /// combine this driving state with the model state are NOT covered here: they need derivative rows.
    /// Refused by name unless the argument layout matches the kernel's signature, outputs are word aligned and have
    /// window shadows, and (for the three one-thread-per-value kernels) grid.x*block.x covers nvalues, so R::Zero never
    /// clears a word the launch did not write. THE BREAKAGE THIS PREVENTS: a LAM window reached "no derivative row and no
    /// rule" for these launches (Refused: identity both ways, so the stale tangent of the overwritten driving words would
    /// enter the specified-zone overwrite and the relaxation terms).
    fn regional_driving(f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3],
                        el: impl Fn(u64, usize) -> Result<i64, String>) -> Option<Result<R, String>> {
        // (argument count, output positions, length rule)
        let (count, outs): (usize, &[usize]) = match f {
            "regional_lbc_derive_v841" => (8, &[6, 7]),
            "regional_lbc_rho_edge_v841" => (8, &[6, 7]),
            "regional_lbc_tendency_v841" => (5, &[4]),
            "regional_lbc_state_at_v841" => (5, &[4]),
            _ => return None,
        };
        Some((|| {
            if args.len() != count { return Err(format!("{} arguments, not {count}", args.len())); }
            let int = |k: usize| match &args[k] { submit::Argument::Bytes(b) if b.len() == 4 => Ok(i32::from_le_bytes(b[..4].try_into().unwrap()) as i64),
                                                  _ => Err(format!("argument {k} is not an int32 scalar")) };
            let words = if f == "regional_lbc_rho_edge_v841" {
                // grid-stride element loop over nlev*nedges: every element is written whatever the grid
                let (nlev, nedges) = (int(0)?, int(2)?);
                if nlev <= 0 || nedges <= 0 { return Err(format!("nlev {nlev} nedges {nedges}")); }
                nlev * nedges
            } else {
                let n = int(0)?;
                if n <= 0 { return Err(format!("nvalues {n}")); }
                let span = grid[0] as i64 * block[0] as i64 * grid[1].max(1) as i64 * grid[2].max(1) as i64;
                if grid[1] > 1 || grid[2] > 1 || block[1] > 1 || block[2] > 1 || span < n {
                    return Err(format!("one-thread-per-value launch grid {grid:?} block {block:?} does not cover nvalues {n}"));
                }
                n
            };
            let mut ranges = Vec::with_capacity(outs.len());
            for &k in outs {
                let submit::Argument::Device(o) = &args[k] else { return Err(format!("output argument {k} is not a device pointer")) };
                if o.offset % 4 != 0 { return Err(format!("output argument {k} not word aligned")); }
                ranges.push((el(o.allocation, o.offset)?, words as usize));
            }
            Ok(R::Zero { ranges, label: format!("lbc-driving:{f} (LBC forcing held: zero tangent, dJ/dLBC dropped)") })
        })())
    }

    /// Specified-zone overwrites of a limited-area run (S01 convective-3km, 2026-11-07): hexcore/cuda_regional_v841.py
    /// regional_speczone_tend_cell_v841 / _tend_edge_v841 (atm_bdy_adjust_dynamics_speczone_tend, MPAS-A v8.4.1
    /// mpas_atm_time_integration.F:7906-7967: tend_rho, tend_rt := the driving tendencies, tend_rw := 0; tend_ru := the
    /// driving tendency), regional_speczone_u_ru_v841 (atm_srk3:2442-2485: u, ru := the driving state),
    /// regional_zero_speczone_w_v841 (F:7868-7902: w := 0), regional_reset_speczone_values_v841 (F:8201-8244: theta_m :=
    /// rt_driving/rho_driving, rho_theta := rt_driving, rtheta_p := rt_driving - rtheta_base) and hexcore/cuda_driver.py
    /// zero_speczone_lagged_diabatic_theta_v842_f32 (F:7949: rt_diabatic_tend := 0). Each writes, at every listed
    /// specified-zone owner (cell or edge index array `spec`, nspec entries) and every level, a word that is a constant or
    /// a function of the driving state, which the LBC driving rule (regional_driving) and the upload rule hold (zero
    /// tangent), and of the fixed reference state (rtheta_base). So the tangent of every written word is zero and its
    /// cotangent stops there: P8 fill rows (src < 0), one per owner, count = levels, element stride = the owner plane. The
    /// owners are read from the live `spec` array at this launch (they are scattered: the 3 km cull of 2026-11-07 has its
    /// 317 specified-zone cells across 0..2786, so a range rule would cost one memset per word run). Words the launch does
    /// not write keep their tangent and cotangent. Refused by name unless the layout matches, every owner index is in
    /// [0, plane), the outputs have shadows covering levels*plane words, and (for the one-thread-per-element
    /// zero_speczone_lagged_diabatic_theta) the grid covers levels*nspec. THE BREAKAGE THIS PREVENTS: these launches
    /// reached "no derivative row and no rule" (identity both ways), which keeps the interior tangent on specified-zone
    /// words the forward overwrites with the driving state.
    fn speczone_overwrite(&self, ex: &Execution, f: &str, args: &[submit::Argument], grid: [u32; 3], block: [u32; 3]) -> Option<Result<R, String>> {
        // (argument count, outputs, grid-stride element loop)
        let (count, outs, stride): (usize, &[usize], bool) = match f {
            "regional_speczone_tend_cell_v841" => (9, &[6, 7, 8], true),
            "regional_speczone_tend_edge_v841" => (6, &[5], true),
            "regional_speczone_u_ru_v841" => (8, &[6, 7], true),
            "regional_zero_speczone_w_v841" => (5, &[4], true),
            "regional_reset_speczone_values_v841" => (10, &[7, 8, 9], true),
            "zero_speczone_lagged_diabatic_theta_v842_f32" => (5, &[4], false),
            _ => return None,
        };
        Some((|| {
            if args.len() != count { return Err(format!("{} arguments, not {count}", args.len())); }
            let int = |k: usize| match &args[k] { submit::Argument::Bytes(b) if b.len() == 4 => Ok(i32::from_le_bytes(b[..4].try_into().unwrap()) as i64),
                                                  _ => Err(format!("argument {k} is not an int32 scalar")) };
            // every one of these kernels: (levels, plane, nspec, spec*, ...)
            let (levels, plane, nspec) = (int(0)?, int(1)?, int(2)?);
            if levels <= 0 || plane <= 0 || nspec < 0 { return Err(format!("levels {levels} plane {plane} nspec {nspec}")); }
            if !stride {
                let span = grid[0] as i64 * block[0] as i64;
                if grid[1] > 1 || grid[2] > 1 || block[1] > 1 || block[2] > 1 || span < levels * nspec {
                    return Err(format!("one-thread-per-element launch grid {grid:?} block {block:?} does not cover {levels} x {nspec}"));
                }
            }
            let submit::Argument::Device(sp) = &args[3] else { return Err("spec (argument 3) is not a device pointer".into()) };
            let table = ex.allocations.get(&sp.allocation).ok_or("spec allocation not live")?;
            let bytes = nspec as usize * 4;
            if sp.offset + bytes > table.bytes { return Err("spec array exceeds its allocation".into()); }
            let raw = unsafe { crate::deriv::predict::table(&ex.cuda, table.address + sp.offset as u64, bytes)? };
            let idx: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut rows = Vec::new();
            for &k in outs {
                let submit::Argument::Device(o) = &args[k] else { return Err(format!("output argument {k} is not a device pointer")) };
                if o.offset % 4 != 0 { return Err(format!("output argument {k} not word aligned")); }
                let a = ex.allocations.get(&o.allocation).ok_or_else(|| format!("output argument {k} allocation not live"))?;
                if o.offset + 4 * (levels * plane) as usize > a.bytes { return Err(format!("output argument {k}: {levels} x {plane} words exceed its allocation")); }
                rows.extend(speczone_fill_rows(&idx, levels, plane, self.el(o.allocation, o.offset)?)?);
            }
            Ok(R::Rows { rows, label: "lbc specified-zone overwrite (driving state held)" })
        })())
    }

    /// The limited-area garbage-column scrub (S01, 2026-11-07): hexcore/cuda_solve_region_v841.py
    /// solve_region_scrub_columns_v841(table, n_arrays) writes, for each descriptor row j (pointer, rows, stride, solve,
    /// pool float32 bits), the constant pool value into element r*stride + solve of that array for r < rows: native's
    /// garbage element, restored after every launch that may have dirtied it (cuda_regional_forecast_v841.py:795). The
    /// written words are constants, so their tangent is zero and their cotangent stops there: P8 fill rows (src < 0),
    /// one per descriptor row. The table is read from the device at this launch (it is the live, possibly cached,
    /// descriptor table the forward reads). Refused unless the launch is (table, n_arrays) with grid.y >= n_arrays, each
    /// pointer is inside a shadowed allocation and the last element stays inside it. THE BREAKAGE THIS PREVENTS: the
    /// launches (about 450 a step on the 3 km cull) reached "no derivative row and no rule" (identity both ways), which
    /// would carry interior tangent through words the forward resets to a constant.
    fn scrub_columns(&self, ex: &Execution, f: &str, args: &[submit::Argument], grid: [u32; 3]) -> Option<Result<R, String>> {
        if f != "solve_region_scrub_columns_v841" { return None; }
        Some((|| {
            if args.len() != 2 { return Err(format!("{} arguments, not 2 (table, n_arrays)", args.len())); }
            let submit::Argument::Device(t) = &args[0] else { return Err("table (argument 0) is not a device pointer".into()) };
            let n = match &args[1] { submit::Argument::Bytes(b) if b.len() == 4 => i32::from_le_bytes(b[..4].try_into().unwrap()) as i64,
                                     _ => return Err("n_arrays (argument 1) is not an int32 scalar".into()) };
            if n <= 0 || (grid[1] as i64) < n { return Err(format!("n_arrays {n} with grid.y {}", grid[1])); }
            let table = ex.allocations.get(&t.allocation).ok_or("table allocation not live")?;
            let bytes = n as usize * 40;
            if t.offset + bytes > table.bytes { return Err("descriptor table exceeds its allocation".into()); }
            let raw = unsafe { crate::deriv::predict::table(&ex.cuda, table.address + t.offset as u64, bytes)? };
            let mut rows = Vec::with_capacity(n as usize);
            for (j, d) in raw.chunks_exact(40).enumerate() {
                let word = |k: usize| u64::from_le_bytes(d[8 * k..8 * k + 8].try_into().unwrap());
                let (addr, count, stride, solve) = (word(0), word(1) as i64, word(2) as i64, word(3) as i64);
                if count <= 0 || stride < 0 || solve < 0 { return Err(format!("row {j}: rows {count} stride {stride} solve {solve}")); }
                let a = ex.allocations.values().find(|x| addr >= x.address && addr < x.address + x.bytes as u64)
                    .ok_or_else(|| format!("row {j}: address {addr:#x} not in a live allocation"))?;
                let off = (addr - a.address) as usize;
                if off % 4 != 0 { return Err(format!("row {j}: unaligned array")); }
                let last = off + 4 * ((count - 1) * stride + solve) as usize;
                if last + 4 > a.bytes { return Err(format!("row {j}: last element outside allocation {}", a.id)); }
                rows.push(DevRow { dst: self.el(a.id, off)? + solve, src: -1, ds: [0, 0, 0, stride], ss: [0; 4], n: [1, 1, count], count });
            }
            Ok(R::Rows { rows, label: "garbage-column scrub (constant pool value)" })
        })())
    }

    fn copy_view(&self, f: &str, src: &CArr, dst: &CArr, si: i64, di: i64) -> R {
        let (dv, sv) = match (self.view(dst, di), self.view(src, si)) { (Ok(d), Ok(s)) => (d, s), (Err(e), _) | (_, Err(e)) => return R::Refused(format!("{f}: {e}")) };
        if src.shape != dst.shape { return R::Refused(format!("{f}: source and destination shapes differ")); }
        if dst.shape.iter().zip(&dst.strides).any(|(&n, &s)| n > 1 && s == 0) { return R::Refused(format!("{f}: repeated destination")); }
        if si == di && dv.base == sv.base && dv.stride == sv.stride { return R::Identity(format!("inactive:{f} (in place)")); }
        if Self::overlap(src, dst) { return R::Refused(format!("{f}: overlapping copy")); }
        if src.shape.iter().zip(&src.strides).any(|(&n, &s)| n > 1 && s == 0) {
            // Broadcasting source: fan-in, so the transpose goes through P8's deterministic gather (rows.rs from_view).
            let pad = 4 - dst.shape.len();
            let (mut n, mut ds, mut ss) = ([1i64; 4], [0i64; 4], [0i64; 4]);
            for k in 0..dst.shape.len() { n[pad + k] = dst.shape[k]; ds[pad + k] = dst.strides[k] / di; ss[pad + k] = src.strides[k] / si; }
            return R::Rows { rows: vec![DevRow { dst: dv.base, src: sv.base, ds, ss, n: [n[1], n[2], n[3]], count: n.iter().product() }], label: "broadcast copy" };
        }
        R::View { dst: dv, src: sv }
    }

    /// rrfs_post_transport_copy: twelve rows of (dst, src, count, n1, n2, n3, d0..d3, s0..s3) (P8 rows::post_transport).
    fn post_transport(&self, a: &[submit::Argument]) -> Result<R, String> {
        if a.len() != 168 { return Err(format!("{} arguments, not 168", a.len())); }
        let int = |x: &submit::Argument| match x { submit::Argument::Bytes(b) if b.len() == 8 => Ok(i64::from_le_bytes(b[..8].try_into().unwrap())), _ => Err("bad scalar".to_string()) };
        let mut rows = Vec::new();
        for r in 0..12 {
            let g = &a[14 * r..14 * r + 14];
            let w: Vec<i64> = g[2..].iter().map(int).collect::<Result<_, _>>()?;
            if w[0] == 0 { continue; }
            let (submit::Argument::Device(d), submit::Argument::Device(s)) = (&g[0], &g[1]) else { return Err("row without pointers".into()) };
            rows.push(DevRow { dst: self.el(d.allocation, d.offset)?, src: self.el(s.allocation, s.offset)?, n: [w[1], w[2], w[3]],
                               ds: [w[4], w[5], w[6], w[7]], ss: [w[8], w[9], w[10], w[11]], count: w[0] });
        }
        Ok(R::Rows { rows, label: "post-transport copy" })
    }

    /// rrfs_glue_copy_plan: the table the live RrfsCopyPlanUpload built (copy_plan::table: 128-byte rows, dims already in
    /// the kernel's order), read from the device and mapped back to plan storage. Fill rows have no source.
    /// G2 (P1 milestone 2): the held radiation tendency rule. rrfs_radiation_unpack (resident_radiation_rrfs.py:19-30,
    /// 23 rows of ARGS(N) = unsigned d, const unsigned s, a, b, c, f, n, w (i64), v (u32)) moves each radiation call's
    /// results into the MPAS pool arrays (rthratenlw/sw, fluxes, radii) that every later step reads until the next
    /// radiation call: d[row*a+col*b] = s ? s[row*c+col*f] : v, row = e/w, col = e%w, word strides. As P8 copy rows
    /// (n1 = n2 = 1, n3 = w, ds = [a, 0, 0, b]) the tangent copies and the reverse moves the cotangent the carried steps
    /// accumulated on the pool arrays back onto the radiation outputs; fill rows (constant v) are zero. THE BREAKAGE
    /// THIS PREVENTS: Refused (identity) claimed d(pool)/d(pool) = 1 for the overwritten tendency and dropped every
    /// later step's cotangent of the radiation step.
    fn radiation_unpack_rows(&self, a: &[submit::Argument]) -> Result<R, String> {
        if a.len() % 9 != 0 { return Err(format!("{} arguments, not rows of 9", a.len())); }
        let int = |x: &submit::Argument| -> Result<i64, String> { match x {
            submit::Argument::Bytes(b) if b.len() == 8 => Ok(i64::from_le_bytes(b[..8].try_into().unwrap())),
            _ => Err("stride/count word is not a by-value i64".into()) } };
        let mut rows = Vec::new();
        for g in a.chunks(9) {
            let n = int(&g[6])?;
            if n == 0 { continue; }
            let w = int(&g[7])?;
            if n < 0 || w <= 0 { return Err(format!("row n {n} w {w}")); }
            let submit::Argument::Device(d) = &g[0] else { return Err("row destination is not a device pointer".into()) };
            let src = match &g[1] {
                submit::Argument::Device(s) => self.el(s.allocation, s.offset)?,
                submit::Argument::Bytes(b) if b.iter().all(|&x| x == 0) => -1,
                _ => return Err("row source is neither a device pointer nor null".into()),
            };
            rows.push(DevRow { dst: self.el(d.allocation, d.offset)?, src, ds: [int(&g[2])?, 0, 0, int(&g[3])?],
                               ss: [int(&g[4])?, 0, 0, int(&g[5])?], n: [1, 1, w], count: n });
        }
        // AD_UNPACK_ZERO=n1,n2 (bisection diagnostic, default off): a launch with that many nonempty rows has every row
        // turned into a fill row, so the radiation outputs it hands to the model carry zero derivative (consistent both
        // ways). Used to split the LW and SW radiation paths in a Taylor check.
        if let Ok(z) = std::env::var("AD_UNPACK_ZERO") {
            if z.split(',').any(|x| x.trim().parse::<usize>().ok() == Some(rows.len())) { for r in rows.iter_mut() { r.src = -1; } }
        }
        // AD_UNPACK_ZERO_OPS=N:i,j,.. (bisection diagnostic): in a launch with N nonempty rows, only rows i, j, .. (their
        // order among the nonempty rows) become fill rows.
        if let Ok(z) = std::env::var("AD_UNPACK_ZERO_OPS") {
            if let Some((n, list)) = z.split_once(':') {
                if n.trim().parse::<usize>().ok() == Some(rows.len()) {
                    for i in list.split(',').filter_map(|x| x.trim().parse::<usize>().ok()) { if let Some(r) = rows.get_mut(i) { r.src = -1; } }
                }
            }
        }
        Ok(R::Rows { rows, label: "radiation unpack (held tendency)" })
    }

    /// cupy_concatenate_same_size (CuPy `_concatenate_kernel_same_size`): y[i] = x[a][top*base + r] with
    /// i = (top*nx + a)*base + r, x a by-value CArray of nx device pointers, base the by-value i64 block length (P8
    /// rows::concatenate, the same rows). `ty` is the module's element type (concat_module_types):
    /// * float: copy rows, one per source array (tangent copies, the reverse moves y's cotangent onto the sources and
    ///   clears y's words: the exact transpose of an overwrite). The pointer table is read from the device (the forward
    ///   uploaded it) and every address mapped back to a live allocation, as copy_plan_rows does;
    /// * double (g1 round 2): the same copy rows on the float64 shadow convention of `view` (element i of a float64
    ///   array at byte offset o is shadow word o/4 + i), so the rows are the float rows and only the byte checks use
    ///   8-byte elements. A double module whose every launch passes the host-check proof keeps its identity
    ///   ("double:host-checked", matched below before this arm), so no plan that bound before binds differently;
    /// * an integer or bool type: no float word moves, identity both ways (inactive);
    /// * any other type, or no established type, whose every launch only feeds a host check ("<type>:host-checked" or
    ///   ":host-checked", concat_host_checked + mark_host_checked): identity both ways (inactive);
    /// * anything else, or no recorded source: Refused, never guessed.
    /// Refused also when y is not contiguous 4-byte words, a source span leaves its allocation, a source overlaps y, or
    /// y's size is not a multiple of nx*base.
    pub fn concatenate(&self, ex: &Execution, f: &str, a: &[submit::Argument], ty: Option<&str>) -> R {
        let item: i64 = match ty {
            Some("float") => 4,
            Some("double") => 8,
            Some(t @ ("long long" | "long" | "int" | "unsigned long long" | "unsigned int" | "bool")) =>
                return R::Identity(format!("inactive:{f} (integer concatenation, typedef {t} T)")),
            // integrator: neither float nor integer, and the plan proves that only a host check reads every vector
            // this module writes (concat_host_checked): no device word depends on it, identity is exact.
            Some(t) if t.ends_with(":host-checked") => return R::Identity(host_checked_label(f, t)),
            t => return R::Refused(format!("{f}: element type {t:?} not established from the module's recorded source")),
        };
        (|| -> Result<R, String> {
            let x = a.first().and_then(carr).ok_or("argument 0 is not the pointer-array CArray")?;
            let base = match a.get(1) { Some(submit::Argument::Bytes(b)) if b.len() == 8 => i64::from_le_bytes(b[..8].try_into().unwrap()),
                                        _ => return Err("argument 1 (base) is not a by-value i64".into()) };
            let y = a.get(2).and_then(carr).ok_or("argument 2 is not the output CArray")?;
            if x.shape.len() != 1 || x.strides.first() != Some(&8) || x.size <= 0 { return Err(format!("pointer array shape {:?} strides {:?}", x.shape, x.strides)); }
            if y.shape.len() != 1 || y.strides.first() != Some(&item) || y.shape[0] != y.size { return Err(format!("output not contiguous {item}-byte elements: shape {:?} strides {:?}", y.shape, y.strides)); }
            let nx = x.size;
            if base <= 0 || y.size % (base * nx) != 0 { return Err(format!("output size {} not a multiple of nx {nx} x base {base}", y.size)); }
            let ntop = y.size / (base * nx);
            let table = ex.allocations.get(&x.alloc).ok_or("pointer table allocation not live")?;
            if x.off + 8 * nx as usize > table.bytes { return Err("pointer table exceeds its allocation".into()); }
            let data = unsafe { crate::deriv::predict::table(&ex.cuda, table.address + x.off as u64, 8 * nx as usize)? };
            let ylo = y.off as i64; let yhi = ylo + item * y.size;
            let mut rows = Vec::with_capacity(nx as usize);
            for (k, w) in data.chunks_exact(8).enumerate() {
                let addr = u64::from_le_bytes(w.try_into().unwrap());
                let s = ex.allocations.values().find(|q| addr >= q.address && addr < q.address + q.bytes as u64)
                    .ok_or_else(|| format!("source {k} address {addr:#x} not in a live allocation"))?;
                let off = (addr - s.address) as usize;
                if off % item as usize != 0 { return Err(format!("source {k} not aligned to {item} bytes")); }
                if off + (item * ntop * base) as usize > s.bytes { return Err(format!("source {k} span leaves allocation {}", s.id)); }
                if s.id == y.alloc && (off as i64) < yhi && ylo < off as i64 + item * ntop * base { return Err(format!("source {k} overlaps the output")); }
                rows.push(DevRow { dst: self.el(y.alloc, y.off)? + k as i64 * base, src: self.el(s.id, off)?,
                                   ds: [0, 0, nx * base, 1], ss: [0, 0, base, 1], n: [1, ntop, base], count: ntop * base });
            }
            Ok(R::Rows { rows, label: if item == 8 { "concatenate (float64 stack)" } else { "concatenate (float32 stack)" } })
        })().unwrap_or_else(|e| R::Refused(format!("{f}: {e}")))
    }

    fn copy_plan_rows(&self, ex: &Execution, a: &[submit::Argument], nrows: u32) -> Result<R, String> {
        let Some(submit::Argument::Device(t)) = a.first() else { return Err("copy plan without table".into()) };
        let table = ex.allocations.get(&t.allocation).ok_or("table allocation not live")?;
        let bytes = nrows as usize * 128;
        if t.offset + bytes > table.bytes { return Err("copy plan table exceeds its allocation".into()); }
        let data = unsafe { crate::deriv::predict::table(&ex.cuda, table.address + t.offset as u64, bytes)? };
        let find = |addr: u64| -> Result<(u64, usize), String> {
            ex.allocations.values().find(|x| addr >= x.address && addr < x.address + x.bytes as u64)
                .map(|x| (x.id, (addr - x.address) as usize)).ok_or_else(|| format!("descriptor address {addr:#x} not in a live allocation"))
        };
        let mut rows = Vec::with_capacity(nrows as usize);
        for r in data.chunks_exact(128) {
            let w = |k: usize| i64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
            let mode = i32::from_le_bytes(r[124..128].try_into().unwrap());
            let (da, doff) = find(w(0) as u64)?;
            if doff % 4 != 0 { return Err("unaligned descriptor".into()); }
            let src = if mode == 1 { -1 } else { let (sa, so) = find(w(1) as u64)?; self.el(sa, so)? };
            rows.push(DevRow { dst: self.el(da, doff)?, src, ds: [w(2), w(3), w(4), w(5)], ss: [w(6), w(7), w(8), w(9)],
                               n: [w(11), w(12), w(13)], count: w(14) });
        }
        Ok(R::Rows { rows, label: "glue copy plan" })
    }

    /// multirank (L20): halo_copy_bytes / halo_copy_words_2d as P8 rows. The launch's descriptor table (9 i64 per field:
    /// live, packet, indices, rows, count, itemsize, ndim, dims, unit; partition_samehost_transport_v841.py) is read from
    /// the device with its index and dims arrays and mapped back to plan storage. Packet words keep the word convention
    /// (packet word p = shadow word base + p, as the DeviceCopy rule moves them); live float32/int32 words too; a float64
    /// field's element k (byte offset 8k from the field origin) is shadow word origin/4 + k (deriv.rs `widen`), carried in
    /// the first of its two packet words (the second gets no shadow move). Byte fields (bool/int8) carry no float word.
    /// THE BREAKAGE THIS PREVENTS: with no rule the launches are Refused (identity both ways), so the tangent never
    /// crosses a rank boundary and the reverse drops every halo cotangent instead of summing it into the owner.
    fn halo_copy_rows(&self, ex: &Execution, f: &str, a: &[submit::Argument], fields: u32) -> Result<R, String> {
        let (Some(submit::Argument::Device(t)), Some(submit::Argument::Bytes(sc))) = (a.first(), a.get(1)) else {
            return Err("arguments are not (table, scatter)".into()) };
        if a.len() != 2 || sc.len() < 4 { return Err(format!("{} arguments", a.len())); }
        let scatter = i32::from_le_bytes(sc[..4].try_into().unwrap()) != 0;
        let table = ex.allocations.get(&t.allocation).ok_or("table allocation not live")?;
        let bytes = fields as usize * 72;
        if t.offset + bytes > table.bytes { return Err("halo table exceeds its allocation".into()); }
        let data = unsafe { crate::deriv::predict::table(&ex.cuda, table.address + t.offset as u64, bytes)? };
        let find = |addr: u64| -> Result<(u64, usize), String> {
            ex.allocations.values().find(|x| addr >= x.address && addr < x.address + x.bytes as u64)
                .map(|x| (x.id, (addr - x.address) as usize)).ok_or_else(|| format!("halo address {addr:#x} not in a live allocation"))
        };
        let words = |addr: u64, n: usize| -> Result<Vec<i64>, String> {
            let (id, off) = find(addr)?;
            let x = &ex.allocations[&id];
            if off + n * 8 > x.bytes { return Err("halo index/dims array exceeds its allocation".into()); }
            let b = unsafe { crate::deriv::predict::table(&ex.cuda, x.address + off as u64, n * 8)? };
            Ok(b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let mut rows = Vec::new();
        let (mut f64_fields, mut byte_fields) = (0usize, 0usize);
        for r in data.chunks_exact(72) {
            let w = |k: usize| i64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
            let (nrows, count) = (w(3), w(4));
            if nrows <= 0 || count <= 0 { continue; }
            let (la, lo) = find(w(0) as u64)?;
            let (pa, po) = find(w(1) as u64)?;
            if po % 4 != 0 || lo % 4 != 0 { return Err("unaligned live/packet origin".into()); }
            let idx = words(w(2) as u64, count as usize)?;
            let (live0, pack0) = (self.el(la, lo)?, self.el(pa, po)?);
            let mut push = |pk: i64, lv: i64, ds: [i64; 4], ss: [i64; 4], n: [i64; 3]| {
                let (dst, src, d, s) = if scatter { (lv, pk, ss, ds) } else { (pk, lv, ds, ss) };
                rows.push(DevRow { dst, src, ds: d, ss: s, n, count: nrows });
            };
            if f == "halo_copy_words_2d" {
                // offset = row * row_stride + indices[column] (words); packet q = row * columns + column.
                let dims = words(w(7) as u64, 2)?;
                if dims[1] % 4 != 0 { return Err("word row stride not a multiple of 4 bytes".into()); }
                let rs = dims[1] / 4;
                for (c, &i) in idx.iter().enumerate() {
                    push(pack0 + c as i64, live0 + i, [count, 0, 0, 0], [rs, 0, 0, 0], [1, 1, 1]);
                }
                continue;
            }
            let (item, ndim, unit) = (w(5), w(6), w(8));
            if unit == 1 {
                if item < 4 { byte_fields += 1; continue; }
                return Err(format!("{item}-byte field moved bytewise"));
            }
            if unit != 4 || (item != 4 && item != 8) { return Err(format!("unit {unit} item {item}")); }
            if !(1..=5).contains(&ndim) { return Err(format!("field rank {ndim}")); }
            let dims = words(w(7) as u64, 2 * ndim as usize)?;
            let upi = item / 4;
            let width = count * upi;
            // leading dims d_0..d_{m-1} (m = ndim - 1), fastest last; P8 decomposition i3 fastest, then i2, i1, q.
            let m = (ndim - 1) as usize;
            let (mut n, mut ls, mut ps) = ([1i64; 4], [0i64; 4], [0i64; 4]);
            let mut span = width;
            for k in 0..m {
                let slot = 3 - k;
                let dim = m - 1 - k;
                let (shape, stride) = (dims[2 * dim], dims[2 * dim + 1]);
                if stride % item != 0 { return Err(format!("stride {stride} not a multiple of the {item}-byte item")); }
                n[slot] = shape; ls[slot] = stride / item; ps[slot] = span; span *= shape;
            }
            if n.iter().product::<i64>() != nrows { return Err(format!("rows {nrows} != leading shape {n:?}")); }
            let cell = dims[2 * m + 1];
            if cell % item != 0 { return Err(format!("cell stride {cell} not a multiple of the {item}-byte item")); }
            if item == 8 { f64_fields += 1; }
            for (c, &i) in idx.iter().enumerate() {
                // float32/int32: live word = origin + i*cell/4 + sum coord*stride/4; float64: element units (deriv.rs widen).
                push(pack0 + c as i64 * upi, live0 + i * (cell / item), ps, ls, [n[1], n[2], n[3]]);
            }
        }
        let label: &'static str = match (f64_fields > 0, byte_fields > 0) {
            (false, false) => "same-host halo copy",
            (true, false) => "same-host halo copy (float64 fields)",
            (false, true) => "same-host halo copy (byte fields inactive)",
            (true, true) => "same-host halo copy (float64 + byte fields)",
        };
        Ok(R::Rows { rows, label })
    }

    /// Frozen FCT limiter (`--fct-frozen 1`, off by default; a deliberate choice over iterating on a limiter
    /// that cannot be differentiated). The limiter coefficients are taken from the forward trajectory and treated as constants in
    /// the tangent and the adjoint: an APPROXIMATION of the forward's derivative (the standard
    /// treatment of a non-differentiable limiter in variational systems), named in the receipt. fct_scale's outputs scale_in / scale_out (the
    /// coefficients, written in place over the low-order flux sums) get zero tangent and their cotangent is dropped;
    /// fct_minmax_source keeps the exact derivative of source_old (transported state) and gives the limiter bounds
    /// none. The other FCT kernels run their entries, so fct_limit_* are linear in the antidiffusive flux with the
    /// trajectory's weights. Why: the exact limiter derivative divides by (flux sum + 1e-20) and overflowed float32
    /// at degenerate tracer bounds (L1 trace4, 2026-11-05 07:15Z).
    fn fct_frozen(&self, ex: &Execution, f: &str, args: &[submit::Argument]) -> Result<R, String> {
        let kd = self.kernels.get(f).ok_or("no kernel row (argument names)")?;
        let pos = |n: &str| kd.names.iter().position(|x| x == n).ok_or_else(|| format!("no argument {n}"));
        let int = |n: &str| -> Result<i32, String> { match args.get(pos(n)?) {
            Some(submit::Argument::Bytes(b)) if b.len() >= 4 => Ok(i32::from_le_bytes(b[..4].try_into().unwrap())),
            _ => Err(format!("{n} is not a by-value int")) } };
        let (nt, nl, nc) = (int("ntracers")?, int("nlev")?, int("ncells")?);
        if nt <= 0 || nl <= 0 || nc <= 0 { return Err(format!("shape {nt} x {nl} x {nc}")); }
        let q = (nt as usize) * (nl as usize) * (nc as usize);
        // (allocation, byte offset) of a device argument, checked to hold `words` float words.
        let dev = |n: &str, words: usize| -> Result<(u64, usize), String> {
            match args.get(pos(n)?) {
                Some(submit::Argument::Device(s)) => {
                    let a = ex.allocations.get(&s.allocation).ok_or_else(|| format!("{n}: allocation not live"))?;
                    if s.offset % 4 != 0 || s.offset + 4 * words > a.bytes {
                        return Err(format!("{n}: {words} words at byte {} exceed allocation {} ({} bytes)", s.offset, s.allocation, a.bytes));
                    }
                    Ok((s.allocation, s.offset))
                },
                _ => Err(format!("{n} is not a device pointer")),
            }
        };
        if f == "fct_scale" {
            let (si, so) = (dev("scale_in", q)?, dev("scale_out", q)?);
            return Ok(R::Zero { ranges: vec![(self.el(si.0, si.1)?, q), (self.el(so.0, so.1)?, q)],
                                label: "fct-frozen: limiter coefficients scale_in/scale_out (fct_scale)".into() });
        }
        let dt = match args.get(pos("dt")?) { Some(submit::Argument::Bytes(b)) if b.len() >= 4 => f32::from_le_bytes(b[..4].try_into().unwrap()),
                                               _ => return Err("dt is not a by-value float".into()) };
        let r = (nl as usize) * (nc as usize);
        let spans = [("old", q), ("source", q), ("rho_old", r), ("source_old", q), ("minimum", q), ("maximum", q)];
        let mut at = Vec::new();
        for (n, w) in spans { let d = dev(n, w)?; at.push((d.0, d.1, w)); }
        // The kernels assume every output is a separate array from every input (true in the forward source).
        for i in 0..6 { for j in i + 1..6 {
            let (a, b) = (at[i], at[j]);
            if a.0 == b.0 && a.1 < b.1 + 4 * b.2 && b.1 < a.1 + 4 * a.2 { return Err(format!("{} and {} overlap", spans[i].0, spans[j].0)); }
        } }
        let mut e = [0i64; 6];
        for (k, x) in at.iter().enumerate() { e[k] = self.el(x.0, x.1)?; }
        Ok(R::FctMinmax { nt, nl, nc, dt, e, prim: [(at[1].0, at[1].1 / 4, q), (at[2].0, at[2].1 / 4, r)] })
    }

    /// ruc_land_tile_preflight (held lane ruc_land_tile_preflight, 2026-11-06): the RUC land driver's tile stop words,
    /// RRFS-MPAS 1d6e370e module_ruc_land.F90:498-582, forward in hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py:5096 (captured
    /// source e6fd85f0 lines 6395-6400). One thread writes TWO int32 words and nothing else: stop = {nc, 0}; then, if
    /// lakemodel (int, ptrs[8][0]) == 1, the first cell c with lakemask (float, ptrs[9][c]) == 1.0f gives stop = {c, 1}.
    /// The result is integer and piecewise constant in every input (an integer test and a float equality), so its
    /// derivative is exactly zero wherever it exists: zero tangent on the two stop words and their cotangent dropped (the
    /// exact transpose; Identity would claim d(stop_out)/d(stop_in) = 1 for an overwritten word, which is wrong). No float
    /// word is written and no input gets a cotangent. The test lakemask == 1.0f is a step: a mask word exactly at 1.0 has
    /// no derivative (one ulp flips stop). It is never approximated: the branch is the trajectory's, like every integer
    /// word downstream (oracle_land_ruc_land reads stop as its cell bound and preparation-only flag). Before this rule
    /// the launch fell through to "no derivative row and no rule" (Refused, identity both ways) whenever a window composed
    /// RUC land live (pkg18 does not: land is a held upload there, 0 launches in plan.json d06ab544).
    fn integer_flags(f: &str, args: &[submit::Argument], el: impl Fn(u64, usize) -> Result<i64, String>,
                     bytes: impl Fn(u64) -> Option<usize>) -> Result<R, String> {
        if args.len() != 3 { return Err(format!("{} arguments, not 3 (ptrs, stop, nc)", args.len())); }
        if !matches!(&args[0], submit::Argument::Device(_)) { return Err("ptrs (argument 0) is not a device pointer".into()); }
        let submit::Argument::Device(s) = &args[1] else { return Err("stop (argument 1) is not a device pointer".into()) };
        if !matches!(&args[2], submit::Argument::Bytes(b) if b.len() == 4) { return Err("nc (argument 2) is not a by-value int32".into()); }
        if s.offset % 4 != 0 { return Err("stop words not aligned".into()); }
        let n = bytes(s.allocation).ok_or_else(|| format!("stop: allocation {} not live", s.allocation))?;
        if s.offset + 8 > n { return Err(format!("stop: 2 words at byte {} exceed allocation {} ({n} bytes)", s.offset, s.allocation)); }
        Ok(R::Zero { ranges: vec![(el(s.allocation, s.offset)?, 2)], label: format!("integer result: RUC land tile stop words ({f})") })
    }

    /// Rule of a non-kernel command.
    pub fn command(&self, ex: &Execution, op: &Operation, base: &std::path::Path,
                   frame: Option<&crate::replay_frame::ReplayFrame>) -> Result<R, String> {
        let range = |s: &plan::Slice, bytes: usize| -> Result<(i64, usize), String> { Ok((self.el(s.allocation, s.offset)?, bytes.div_ceil(4))) };
        Ok(match op {
            Operation::DeviceCopy { destination, source, bytes, .. } => {
                if bytes % 4 != 0 || destination.offset % 4 != 0 || source.offset % 4 != 0 { return Ok(R::Refused("device_copy: unaligned".into())); }
                let (d, s) = (self.el(destination.allocation, destination.offset)?, self.el(source.allocation, source.offset)?);
                let n = bytes / 4;
                if d == s { R::Identity("inactive:device_copy (in place)".into()) }
                else if d < s + n as i64 && s < d + n as i64 { R::Refused("device_copy: overlapping".into()) }
                else { R::Copy { dst: d, src: s, n } }
            },
            Operation::PeerSend { .. } | Operation::PeerRecv { .. } => R::Refused("peer copy: rule is built in the window (rule_inner)".into()),
            Operation::ModuleGlobalCopy { symbol, .. } => R::Identity(format!("inactive:module_global_copy {symbol} (module constant outside every allocation; no shadow)")),
            Operation::Clear { destination, bytes, .. } => R::Zero { ranges: vec![range(destination, *bytes)?], label: "clear".into() },
            Operation::Fill16 { destination, count, .. } => R::Zero { ranges: vec![range(destination, count * 2)?], label: "fill16".into() },
            Operation::Fill32 { destination, count, .. } => R::Zero { ranges: vec![range(destination, count * 4)?], label: "fill32".into() },
            // anyparam: the window's own LW column-order uploads (window.rs lw_order_op) write int32 column indices
            // only, like the stable argsort and its int32 cast before them: no float word, identity both ways.
            Operation::Upload { origin: Some(o), .. } if o.starts_with("live: LW column order") => R::Identity("inactive:live LW column order (integer indices)".into()),
            Operation::Upload { destination, data, pointers, origin, .. } => {
                if !pointers.is_empty() { R::Identity("table:pointer upload".into()) } else {
                    let bytes = match data {
                        plan::Blob::Hex { bytes, .. } => bytes.len() / 2,
                        plan::Blob::File { path, .. } => std::fs::metadata(crate::path(base, path)).map_err(|e| e.to_string())?.len() as usize,
                        plan::Blob::Gzip { bytes, .. } => *bytes,
                    };
                    R::Zero { ranges: vec![range(destination, bytes)?], label: format!("upload:{}", origin.as_deref().unwrap_or("?")) }
                }
            },
            Operation::HeldUploadRanges { ranges, origin, .. } => R::Held {
                ranges: ranges.iter().map(|r| range(&r.destination, r.bytes)).collect::<Result<_, _>>()?, origin: origin.clone() },
            Operation::FrameUpload { destination, name, .. } => {
                let n = frame.ok_or("frame upload without frame")?.scalar(name)?.len();
                R::Zero { ranges: vec![range(destination, n)?], label: format!("host-frame constant {name}") }
            },
            Operation::HostUpload { destination, view, order, .. } => {
                let n = ex.host_state.pack(view.allocation, &view.view(), order.order())?.len();
                // A host value uploaded into device state: held by name (the host copy carries no tangent).
                R::Held { ranges: vec![range(destination, n)?], origin: format!("host-upload:{}", view.allocation) }
            },
            Operation::HostPublish { .. } => R::Identity("noop:host_publish (device to host)".into()),
            Operation::PointerTableUpload { rule, .. } => R::Identity(format!("table:{rule}")),
            Operation::DescriptorTableUpload { rule, .. } => R::Identity(format!("table:{rule}")),
            Operation::TempoPointerTableBind { .. } | Operation::TempoNamedTableBind { .. } => R::Identity("table:tempo bind".into()),
            Operation::TempoTableRelease { .. } => R::Identity("noop:tempo release".into()),
            Operation::SeaicePointerTableUpload { .. } => R::Identity("table:seaice".into()),
            Operation::RucPointerTableUpload { .. } => R::Identity("table:ruc".into()),
            Operation::RrfsCopyPlanUpload { .. } => R::Identity("table:rrfs copy plan".into()),
            Operation::NativeCubReduce { dtype, .. } | Operation::NativeCubSegmentedReduce { dtype, .. } =>
                // CuPy cub dtype codes 0-7 are the integer types (P8 rules.rs command_class).
                if (0..=7).contains(dtype) { R::Identity("inactive:cub integer reduction".into()) }
                else { R::Refused(format!("native cub reduce dtype {dtype}: float reduction has no rule")) },
            // G2 (P1 milestone 2): the LW wrapper's stable argsort of the integer layer counts (rrtmg_lw_v842.py:1018).
            // CuPy Thrust dtype codes 4 and 6 are int32 and int64 (native_thrust::word_bytes); the sorted copy and the
            // size_t indices are integers, so no float word is written. THE BREAKAGE THIS PREVENTS: with radiation live
            // the argsort was Refused (identity both ways) and listed as an unbound radiation command at every LW call.
            Operation::NativeThrustArgsort { dtype, .. } =>
                if matches!(dtype, 4 | 6) { R::Identity("inactive:thrust argsort (integer keys and indices)".into()) }
                else { R::Refused(format!("native thrust argsort dtype {dtype}: float keys have no rule")) },
            Operation::NativeCublasTranspose { input, output, m, n, .. } => {
                let g = ex.library_dimensions(&[*m, *n], i32::MAX as usize)?;
                let (m, n) = (g[0] as i64, g[1] as i64);
                // cublasSgeam(OP_T, OP_T, m, n, 1, A lda=n, 0, A, C ldc=m): C[i + j*m] = A[j + i*n] (P8 rows::cublas_transpose).
                R::Rows { rows: vec![DevRow { dst: self.el(output.allocation, output.offset)?, src: self.el(input.allocation, input.offset)?,
                                              ds: [0, 0, 1, m], ss: [0, 0, n, 1], n: [1, m, n], count: m * n }], label: "cublas transpose" }
            },
            Operation::DeviceToHost { host_consumer, .. } => R::Identity(format!("d2h:{host_consumer}")),
            Operation::Synchronization { .. } | Operation::EventRecord { .. } | Operation::EventWait { .. } => R::Identity("noop:sync".into()),
            Operation::HostBoundary { .. } => R::Identity("noop:host boundary".into()),
            Operation::TempoMainComplete { .. } => R::Identity("noop:tempo main complete".into()),
            Operation::FreeNull { .. } | Operation::Free { .. } => R::Identity("noop:free".into()),
            Operation::Allocate { id, bytes, .. } => R::Zero { ranges: vec![(self.el(*id, 0)?, bytes.div_ceil(4))], label: "fresh allocation".into() },
            Operation::RrfsCopyPlanKernel { table, cache, .. } => {
                let data = ex.host_state.get(cache.allocation)?;
                let (launch, _, _) = crate::copy_plan::decode_cache(data.get(cache.offset..cache.offset + crate::copy_plan::CACHE_BYTES).ok_or("cache")?)?;
                let args = [submit::Argument::Device(submit::Slice { allocation: table.allocation, offset: table.offset })];
                self.copy_plan_rows(ex, &args, launch.grid[1]).unwrap_or_else(|e| R::Refused(format!("rrfs copy plan kernel: {e}")))
            },
            Operation::TransportCopy { .. } => R::Refused("transport_copy op: no rule in this dispatcher".into()),
            // G2 (P1 milestone 2): the held radiation tendency rule. rrfs_radiation_unpack (resident_radiation_rrfs.py:28)
            // moves each radiation call's results into the MPAS pool arrays (rthratenlw/sw, fluxes, radii) that every
            // later step reads until the next radiation call: d[row*a+col*b] = s ? s[row*c+col*f] : v, row = e/w,
            // col = e%w. As P8 copy rows (n1 = n2 = 1, n3 = w, ds = [a, 0, 0, b]) the tangent copies and the reverse
            // moves the pool cotangent accumulated over the carried steps back onto the radiation outputs; fill rows
            // (constant v) are zero. Views are the live host shape/stride words, as the forward's RadiationUnpack.
            // THE BREAKAGE THIS PREVENTS: Refused (identity) claimed d(pool)/d(pool) = 1 for the overwritten tendency
            // and dropped every steps-2..N cotangent of the radiation step.
            Operation::RadiationUnpack { rows, .. } => {
                let words = |view: &plan::RadiationView| -> Result<(i64, i64, [i64; 2], Option<i64>), String> {
                    let shape = ex.host_i64_values(&view.shape)?;
                    let strides = ex.host_i64_values(&view.strides)?;
                    if !(1..=2).contains(&shape.len()) || shape.len() != strides.len() || shape.iter().any(|&n| n < 0)
                        || strides.iter().any(|b| b % 4 != 0) { return Err("radiation unpack: invalid live view".into()); }
                    let n = shape.iter().product::<i64>();
                    let w = *shape.last().unwrap();
                    let st = if shape.len() == 1 { [0, strides[0] / 4] } else { [strides[0] / 4, strides[1] / 4] };
                    let el = match view.target { Some(t) if n > 0 => Some(self.el(t.allocation, t.offset)?), _ => None };
                    Ok((n, w, st, el))
                };
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let (n, w, ds, dst) = words(&row.dst)?;
                    if n == 0 { continue; }
                    let dst = dst.ok_or("radiation unpack: nonempty view without target")?;
                    let (src, ss) = match &row.src {
                        Some(v) => { let (m, _, ss, el) = words(v)?; if m != n { return Err("radiation unpack: copy shape mismatch".into()); }
                                     (el.ok_or("radiation unpack: source without target")?, ss) },
                        None => (-1, [0, 0]),
                    };
                    out.push(DevRow { dst, src, ds: [ds[0], 0, 0, ds[1]], ss: [ss[0], 0, 0, ss[1]], n: [1, 1, w], count: n });
                }
                R::Rows { rows: out, label: "radiation unpack (held tendency)" }
            },
            Operation::AddInplace { .. } => R::Refused("add_inplace op: no rule in this dispatcher".into()),
            Operation::VinterpOzone { .. } => R::Refused("vinterp_ozone op: no rule in this dispatcher".into()),
            Operation::Kernel { .. } | Operation::HostBranch { .. } | Operation::RkSchedule { .. } | Operation::RkSplits { .. }
            | Operation::RkStages { .. } | Operation::RkSubsteps { .. } => return Err("not a leaf command".into()),
        })
    }
}

/// P8's deterministic transpose of a copy-row launch as a gather (algorithm of src/p8/rows.rs `gather`,
/// copied because that file binds P8's own plan module): CSR over distinct source elements of (dst id << 1 | repeat)
/// items in the serial reference's order (rows last to first, elements ascending), plus the distinct destinations.
pub struct Gather { pub src_word: Vec<i64>, pub start: Vec<i64>, pub item: Vec<i64>, pub dst_word: Vec<i64>,
                    pub sourceless: Vec<u8>, pub injective: bool,
                    /// machine (gather_quick only): rows of the launch that share no element with any other row and are
                    /// injective among themselves; the tables above then cover only the other rows.
                    pub direct: Vec<DevRow>,
                    /// machine (gather_quick only): fan-in groups (see fan_plan): rows transposed by p8_rows_ad_fan with
                    /// no tables; fan_groups holds (first row, rows) per group. The tables above are then empty.
                    pub fan: Vec<DevRow>, pub fan_groups: Vec<i64> }
pub fn gather(rows: &[DevRow]) -> Result<Gather, String> {
    let order: Vec<usize> = (0..rows.len()).rev().collect();
    let mut first = vec![0i64; rows.len() + 1];
    for (i, &r) in order.iter().enumerate() { first[i + 1] = first[i] + rows[r].count; }
    let total = first[rows.len()] as usize;
    if total == 0 { return Ok(Gather { src_word: vec![], start: vec![0], item: vec![], dst_word: vec![], sourceless: vec![], injective: true, direct: vec![], fan: vec![], fan_groups: vec![] }); }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let mut visits = vec![(0i64, 0i64, 0i64); total];
    let chunk = total.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (c, part) in visits.chunks_mut(chunk).enumerate() {
            let (order, first) = (&order, &first);
            sc.spawn(move || {
                let k0 = (c * chunk) as i64;
                let mut i = first.partition_point(|&f| f <= k0) - 1;
                for (j, v) in part.iter_mut().enumerate() {
                    let k = k0 + j as i64;
                    while first[i + 1] <= k { i += 1; }
                    let r = &rows[order[i]];
                    let mut q = k - first[i];
                    let i3 = q % r.n[2]; q /= r.n[2]; let i2 = q % r.n[1]; q /= r.n[1]; let i1 = q % r.n[0]; q /= r.n[0];
                    let d = r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2] + i3 * r.ds[3];
                    let s = if r.src >= 0 { r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2] + i3 * r.ss[3] } else { -1 };
                    *v = (d, k, s);
                }
            });
        }
    });
    let mut by_dst: Vec<(i64, i64)> = visits.iter().map(|&(d, k, _)| (d, k)).collect();
    par_sort(&mut by_dst, threads);
    let mut dst_word: Vec<i64> = Vec::new(); let mut first_k: Vec<i64> = Vec::new();
    for &(d, k) in &by_dst { if dst_word.last() != Some(&d) { dst_word.push(d); first_k.push(k); } }
    let mut parts: Vec<Vec<(i64, i64, i64)>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = visits.chunks(chunk).map(|part| { let (dw, fk) = (&dst_word, &first_k); sc.spawn(move || {
            part.iter().filter(|v| v.2 >= 0).map(|&(d, k, s)| {
                let id = dw.binary_search(&d).unwrap() as i64;
                (s, k, (id << 1) | (fk[id as usize] != k) as i64) }).collect::<Vec<_>>() }) }).collect();
        for h in hs { parts.push(h.join().unwrap()); }
    });
    let mut by_src: Vec<(i64, i64, i64)> = parts.concat();
    let mut sourceless = vec![1u8; dst_word.len()];
    for &(_, _, it) in &by_src { sourceless[(it >> 1) as usize] = 0; }
    par_sort(&mut by_src, threads);
    let (mut src_word, mut start, mut item) = (Vec::new(), Vec::new(), Vec::with_capacity(by_src.len()));
    for &(s, _, it) in &by_src {
        if src_word.last() != Some(&s) { src_word.push(s); start.push(item.len() as i64); }
        item.push(it);
    }
    start.push(item.len() as i64);
    if let Some(&s) = src_word.iter().find(|s| dst_word.binary_search(s).is_ok()) {
        return Err(format!("source element {s} is also a destination: transpose must stay serial for this launch"));
    }
    let injective = dst_word.len() == total && src_word.len() == item.len();
    Ok(Gather { src_word, start, item, dst_word, sourceless, injective, direct: vec![], fan: vec![], fan_groups: vec![] })
}
/// machine: `gather` for the window. An injective copy-row launch (every destination written once, every source
/// read at most once, no element both) needs no tables: the window keeps only the flag (install_gather drops them) and
/// transposes it with rows_ad_direct. THE BREAKAGE THIS FIXES: the flag was found by building the whole gather, three
/// i64 per copied word sorted twice on 16 threads per launch; at 120 km the tangent pass then waited 12.7-25.5 s at its
/// end for those builds (a test machine pair1-b / pair2-a, mark gathers_joined) with the card idle. Here the same
/// three facts are read off two bitmaps, one bit per element, filled a run at a time on all cores. Anything it cannot
/// prove injective (a repeat, an overlap, an odd shape) goes to `gather` unchanged, so a non-injective launch gets the
/// same tables and a refused one the same refusal. AD_GATHER_FAST=0 skips the test.
pub fn gather_quick(rows: &[DevRow]) -> Result<Gather, String> {
    let total: i64 = rows.iter().map(|r| r.count).sum();
    if total > 0 && std::env::var("AD_GATHER_FAST").as_deref() != Ok("0") {
        let all = || Gather { src_word: vec![], start: vec![0], item: vec![], dst_word: vec![], sourceless: vec![], injective: true, direct: vec![], fan: vec![], fan_groups: vec![] };
        if injective_fast(rows) { return Ok(all()); }
        // machine: not injective as a whole. Rows are grouped so that no two groups share a destination or a source
        // element (and no element of the launch is both); the transposes of different groups then touch different words
        // and can run in any order. An injective group goes to rows_ad_direct; the tables are built for the other rows
        // only, in their own order, so every source's items and every repeat flag are the ones the whole-launch gather
        // gives. THE BREAKAGE THIS FIXES: at 120 km one glue copy plan of 97 M words built 3.2 GB of tables per step in
        // 15 s on 16 threads because a few of its 52 rows repeat a source (a test machine gp4-b, window_gather): 14.6 GB
        // of device tables per 3-step window and a 10-25 s wait at the end of the tangent pass. AD_GATHER_SPLIT=0 builds
        // the whole launch's tables as before.
        if std::env::var("AD_GATHER_SPLIT").as_deref() != Ok("0") {
            let split = split_independent(rows);
            if let Some((_, rest)) = &split { if rest.is_empty() { return Ok(all()); } }
            // machine, round 2: what is left after the split is, in every plan read so far, a field copied to two
            // places (two rows with the same source map) or one row whose source broadcasts along an axis of 2. Those
            // are fan-in groups: p8_rows_ad_fan sums a source's contributions in the gather's own order and rounds
            // once, so the result is the gather's bit for bit and no table is built, uploaded or kept. THE BREAKAGE
            // THIS FIXES: the tables are 25-32 bytes per copied word, made per step and kept for the window: at 120 km
            // 0.49 GB per step + 1.0 GB on radiation steps + 0.24 GB (a test machine c6-b, window_gather), about
            // four times that at 60 km, so a 60-step (6 h) 60 km window would hold over 100 GB of them, and each build
            // is a 2-4 s 16-thread sort beside the tangent step. Anything fan_plan cannot prove (sources that overlap
            // without being the same map, a destination written twice, a long broadcast) keeps the tables as before.
            // AD_GATHER_FAN=0 turns it off.
            if std::env::var("AD_GATHER_FAN").as_deref() != Ok("0") {
                let (direct, rest): (Vec<DevRow>, Vec<DevRow>) = match &split { Some((d, r)) => (d.clone(), r.clone()), None => (Vec::new(), rows.to_vec()) };
                if let Some((fan, fan_groups)) = fan_plan(&rest) {
                    let mut g = all();
                    g.injective = false; g.direct = direct; g.fan = fan; g.fan_groups = fan_groups;
                    return Ok(g);
                }
            }
            if let Some((direct, rest)) = split {
                if !direct.is_empty() {
                    let mut g = gather(&rest)?;
                    if g.injective { return Ok(all()); }
                    g.direct = direct;
                    return Ok(g);
                }
            }
        }
    }
    gather(rows)
}
/// machine: (rows proven independent of every other row and injective among themselves, the other rows), both in
/// launch order; None when the launch cannot be split safely (then the whole-launch gather decides, as before).
/// Groups are formed on bounding intervals, which is coarser than the elements themselves, so two groups never share
/// an element; a destination interval that meets any source interval leaves the launch whole (gather's own test of
/// "a source element is also a destination" must see every row).
fn split_independent(rows: &[DevRow]) -> Option<(Vec<DevRow>, Vec<DevRow>)> {
    let mut iv: Vec<(usize, (i64, i64), Option<(i64, i64)>)> = Vec::new(); // (row, dst interval, src interval), inclusive
    for (ri, r) in rows.iter().enumerate() {
        if r.count == 0 { continue; }
        if r.count < 0 || r.n.iter().any(|&x| x <= 0) { return None; }
        let inner = r.n[0].checked_mul(r.n[1])?.checked_mul(r.n[2])?;
        let dims = [(r.count - 1) / inner + 1, r.n[0], r.n[1], r.n[2]];
        let ext = |b: i64, st: &[i64; 4]| -> Option<(i64, i64)> {
            let (mut lo, mut hi) = (b, b);
            for j in 0..4 { let v = st[j].checked_mul(dims[j] - 1)?; if v < 0 { lo = lo.checked_add(v)?; } else { hi = hi.checked_add(v)?; } }
            Some((lo, hi))
        };
        let d = ext(r.dst, &r.ds)?;
        let s = if r.src >= 0 { Some(ext(r.src, &r.ss)?) } else { None };
        iv.push((ri, d, s));
    }
    if iv.len() < 2 { return None; }
    // any destination interval meeting any source interval: leave the launch whole
    let mut dd: Vec<(i64, i64)> = iv.iter().map(|x| x.1).collect();
    let mut ss: Vec<(i64, i64)> = iv.iter().filter_map(|x| x.2).collect();
    dd.sort_unstable(); ss.sort_unstable();
    let (mut i, mut j) = (0, 0);
    while i < dd.len() && j < ss.len() {
        if dd[i].1 < ss[j].0 { i += 1; } else if ss[j].1 < dd[i].0 { j += 1; } else { return None; }
    }
    // union rows whose destination intervals meet, and rows whose source intervals meet
    let mut parent: Vec<usize> = (0..iv.len()).collect();
    fn find(p: &mut Vec<usize>, mut x: usize) -> usize { while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x }
    let sweep = |list: &mut Vec<(i64, i64, usize)>, parent: &mut Vec<usize>| {
        list.sort_unstable();
        let mut cur: Option<(i64, usize)> = None; // (largest end so far, a member)
        for &(lo, hi, k) in list.iter() {
            match cur {
                Some((end, m)) if lo <= end => { let (a, b) = (find(parent, m), find(parent, k)); if a != b { parent[a] = b; } cur = Some((end.max(hi), k)); },
                _ => cur = Some((hi, k)),
            }
        }
    };
    let mut dl: Vec<(i64, i64, usize)> = iv.iter().enumerate().map(|(k, x)| (x.1 .0, x.1 .1, k)).collect();
    let mut sl: Vec<(i64, i64, usize)> = iv.iter().enumerate().filter_map(|(k, x)| x.2.map(|s| (s.0, s.1, k))).collect();
    sweep(&mut dl, &mut parent); sweep(&mut sl, &mut parent);
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for k in 0..iv.len() { let root = find(&mut parent, k); groups.entry(root).or_default().push(k); }
    if groups.len() < 2 { return None; }
    let mut is_direct = vec![false; iv.len()];
    for members in groups.values() {
        let subset: Vec<DevRow> = members.iter().map(|&k| rows[iv[k].0]).collect();
        if injective_fast(&subset) { for &k in members { is_direct[k] = true; } }
    }
    let direct: Vec<DevRow> = (0..iv.len()).filter(|&k| is_direct[k]).map(|k| rows[iv[k].0]).collect();
    let rest: Vec<DevRow> = (0..iv.len()).filter(|&k| !is_direct[k]).map(|k| rows[iv[k].0]).collect();
    Some((direct, rest))
}
/// machine: a table-free transpose plan for copy rows that are not injective as a set, or None.
/// Returns (rows, groups): the rows of a group are consecutive and in the serial reference's order (rows last to
/// first; the slices of a broadcasting row ascending), groups = (first row, rows) per group; p8_rows_ad_fan runs it.
/// A row whose source has stride 0 along an axis longer than 1 (at most 16 slices in all) is cut into its slices,
/// which then share one source map. Rows with the same source map (same origin, strides, shape, count) form a group.
/// Proof, on the same bitmaps as injective_fast: every destination element of all rows is written once, the groups'
/// source maps are injective and pairwise disjoint, and no element is both read and written. Then a source element's
/// contributions are exactly its group's rows at the same element number, in the order gather() lists them.
fn fan_plan(rows: &[DevRow]) -> Option<(Vec<DevRow>, Vec<i64>)> {
    const MAX_SLICES: i64 = 16;
    const MAX_UNITS: usize = 1024;
    let mut units: Vec<DevRow> = Vec::new();
    for r in rows.iter().rev() {
        if r.count == 0 { continue; }
        if r.count < 0 || r.n.iter().any(|&x| x <= 0) { return None; }
        let inner = r.n[0].checked_mul(r.n[1])?.checked_mul(r.n[2])?;
        if r.count % inner != 0 { return None; }
        let dims = [r.count / inner, r.n[0], r.n[1], r.n[2]];
        let (mut bc, mut slices) = ([false; 4], 1i64);
        if r.src >= 0 {
            for j in 0..4 {
                if dims[j] > 1 && r.ss[j] == 0 {
                    if r.ds[j] == 0 { return None; }
                    bc[j] = true;
                    slices = slices.checked_mul(dims[j])?;
                    if slices > MAX_SLICES { return None; }
                }
            }
        }
        if slices == 1 { units.push(*r); }
        else {
            for t in 0..slices {
                // slice index along each broadcast axis, axis 0 most significant: ascending element number
                let (mut rem, mut u, mut d2) = (t, *r, dims);
                for j in (0..4).rev() {
                    if bc[j] { let i = rem % dims[j]; rem /= dims[j]; u.dst = u.dst.checked_add(i.checked_mul(r.ds[j])?)?; d2[j] = 1; }
                }
                u.n = [d2[1], d2[2], d2[3]];
                u.count = d2[0] * d2[1] * d2[2] * d2[3];
                units.push(u);
            }
        }
        if units.len() > MAX_UNITS { return None; }
    }
    if units.is_empty() { return None; }
    // groups by source map; a stride along an axis of length 1 is never used, so it is not part of the map
    let key = |u: &DevRow| {
        let dims = [u.count / (u.n[0] * u.n[1] * u.n[2]), u.n[0], u.n[1], u.n[2]];
        let mut ss = u.ss;
        for j in 0..4 { if dims[j] == 1 { ss[j] = 0; } }
        (u.src, ss, u.n, u.count)
    };
    let mut groups: Vec<Vec<DevRow>> = Vec::new();
    let mut at: std::collections::HashMap<(i64, [i64; 4], [i64; 3], i64), usize> = std::collections::HashMap::new();
    for u in &units {
        if u.src < 0 { groups.push(vec![*u]); continue; }
        let k = key(u);
        match at.get(&k) { Some(&g) => groups[g].push(*u), None => { at.insert(k, groups.len()); groups.push(vec![*u]); } }
    }
    // the proof: each group's first row as it is (its destination and the group's source), the others as fills
    let mut probe: Vec<DevRow> = Vec::with_capacity(units.len());
    for g in &groups {
        probe.push(g[0]);
        for u in &g[1..] { let mut f = *u; f.src = -1; f.ss = [0; 4]; probe.push(f); }
    }
    if !injective_fast(&probe) { return None; }
    let (mut fan, mut fan_groups) = (Vec::with_capacity(units.len()), Vec::with_capacity(2 * groups.len()));
    for g in &groups { fan_groups.push(fan.len() as i64); fan_groups.push(g.len() as i64); fan.extend_from_slice(g); }
    Some((fan, fan_groups))
}
/// True only when the launch is proven injective with no element both read and written (see gather_quick).
fn injective_fast(rows: &[DevRow]) -> bool {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
    const TASK_ELEMS: i64 = 1 << 22;
    let (mut dlo, mut dhi, mut slo, mut shi) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
    let mut tasks: Vec<(usize, i64, i64)> = Vec::new(); // (row, first run, runs); a run is n[2] elements
    for (ri, r) in rows.iter().enumerate() {
        if r.count == 0 { continue; }
        if r.count < 0 || r.n.iter().any(|&x| x <= 0) { return false; }
        let Some(inner) = r.n[0].checked_mul(r.n[1]).and_then(|x| x.checked_mul(r.n[2])) else { return false };
        if r.count % r.n[2] != 0 { return false; }
        let dims = [(r.count - 1) / inner + 1, r.n[0], r.n[1], r.n[2]];
        let ext = |b: i64, st: &[i64; 4]| -> Option<(i64, i64)> {
            let (mut lo, mut hi) = (b, b);
            for j in 0..4 { let v = st[j].checked_mul(dims[j] - 1)?; if v < 0 { lo = lo.checked_add(v)?; } else { hi = hi.checked_add(v)?; } }
            Some((lo, hi))
        };
        let Some((lo, hi)) = ext(r.dst, &r.ds) else { return false };
        if lo < 0 { return false; }
        dlo = dlo.min(lo); dhi = dhi.max(hi);
        if r.src >= 0 {
            let Some((lo, hi)) = ext(r.src, &r.ss) else { return false };
            if lo < 0 { return false; }
            slo = slo.min(lo); shi = shi.max(hi);
        }
        let runs = r.count / r.n[2];
        let per = (TASK_ELEMS / r.n[2]).max(1);
        let mut o = 0;
        while o < runs { let c = per.min(runs - o); tasks.push((ri, o, c)); o += c; }
    }
    if dhi < dlo { return false; }
    // bitmaps on absolute 64-element words, so a destination word and a source word line up for the overlap test
    let words = |lo: i64, hi: i64| if hi < lo { 0usize } else { ((hi >> 6) - (lo >> 6) + 1) as usize };
    let (dn, sn) = (words(dlo, dhi), words(slo, shi));
    if (dn + sn) as u64 > (1u64 << 31) { return false; } // more than 16 GiB of bitmap: not a window this path was sized for
    let (dbm, sbm) = (vec![0u64; dn], vec![0u64; sn]);
    // AtomicU64 has the same in-memory representation as u64; the vectors outlive every use of these views.
    let da: &[AtomicU64] = unsafe { std::slice::from_raw_parts(dbm.as_ptr().cast(), dn) };
    let sa: &[AtomicU64] = unsafe { std::slice::from_raw_parts(sbm.as_ptr().cast(), sn) };
    let (dw0, sw0) = (dlo >> 6, if sn > 0 { slo >> 6 } else { 0 });
    // set [lo, lo + len); true when a bit was already set
    fn mark_run(bm: &[AtomicU64], w0: i64, lo: i64, len: i64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let (mut e, end, mut hit) = (lo, lo + len, false);
        while e < end {
            let b = (e & 63) as u32;
            let take = (64 - b as i64).min(end - e);
            let mask = if take == 64 { u64::MAX } else { ((1u64 << take) - 1) << b };
            hit |= bm[((e >> 6) - w0) as usize].fetch_or(mask, Relaxed) & mask != 0;
            e += take;
        }
        hit
    }
    fn mark_strided(bm: &[AtomicU64], w0: i64, first: i64, stride: i64, n: i64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        if stride == 1 { return mark_run(bm, w0, first, n); }
        if stride == -1 { return mark_run(bm, w0, first - (n - 1), n); }
        if stride == 0 { return n > 1 || mark_run(bm, w0, first, 1); }
        let mut hit = false;
        for i in 0..n { let e = first + i * stride; let bit = 1u64 << (e & 63); hit |= bm[((e >> 6) - w0) as usize].fetch_or(bit, Relaxed) & bit != 0; }
        hit
    }
    let (bad, next) = (AtomicBool::new(false), AtomicUsize::new(0));
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(32).min(tasks.len()).max(1);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let t = next.fetch_add(1, Relaxed);
                if t >= tasks.len() || bad.load(Relaxed) { break; }
                let (ri, o0, c) = tasks[t];
                let r = &rows[ri];
                let mut hit = false;
                for o in o0..o0 + c {
                    let i2 = o % r.n[1]; let u = o / r.n[1]; let i1 = u % r.n[0]; let q = u / r.n[0];
                    hit |= mark_strided(da, dw0, r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2], r.ds[3], r.n[2]);
                    if r.src >= 0 { hit |= mark_strided(sa, sw0, r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2], r.ss[3], r.n[2]); }
                    if hit { break; }
                }
                if hit { bad.store(true, Relaxed); break; }
            });
        }
    });
    if bad.load(Relaxed) { return false; }
    // an element both read and written: gather refuses the launch (its text), so it goes there
    if sn > 0 {
        let (a, z) = (dw0.max(sw0), (dw0 + dn as i64).min(sw0 + sn as i64));
        let mut w = a;
        while w < z { if dbm[(w - dw0) as usize] & sbm[(w - sw0) as usize] != 0 { return false; } w += 1; }
    }
    true
}
/// Sort in `threads` chunks in parallel, then merge pairwise in parallel.
fn par_sort<T: Ord + Copy + Send + Sync>(v: &mut Vec<T>, threads: usize) {
    let n = v.len();
    if n < 1 << 16 || threads < 2 { v.sort_unstable(); return; }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|sc| { for p in v.chunks_mut(chunk) { sc.spawn(move || p.sort_unstable()); } });
    let mut width = chunk;
    while width < n {
        let runs: Vec<&[T]> = v.chunks(width).collect();
        let merged: Vec<Vec<T>> = std::thread::scope(|sc| {
            let hs: Vec<_> = runs.chunks(2).map(|pair| { let pair = pair.to_vec(); sc.spawn(move || {
                if pair.len() == 1 { return pair[0].to_vec(); }
                let (a, b) = (pair[0], pair[1]);
                let (mut i, mut j, mut m) = (0, 0, Vec::with_capacity(a.len() + b.len()));
                while i < a.len() && j < b.len() { if a[i] <= b[j] { m.push(a[i]); i += 1 } else { m.push(b[j]); j += 1 } }
                m.extend_from_slice(&a[i..]); m.extend_from_slice(&b[j..]); m }) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        *v = merged.concat(); width *= 2;
    }
}

/// P8 fill rows (src < 0) zeroing `levels` words of stride `plane` from `base + owner` for every distinct owner of a
/// specified-zone index array (Ctx::speczone_overwrite). Owner indices outside [0, plane) are refused.
pub fn speczone_fill_rows(idx: &[i32], levels: i64, plane: i64, base: i64) -> Result<Vec<DevRow>, String> {
    let mut owners: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
    if let Some(bad) = owners.iter().find(|&&i| i < 0 || i >= plane) { return Err(format!("specified-zone owner {bad} outside [0, {plane})")); }
    owners.sort_unstable(); owners.dedup();
    Ok(owners.into_iter().map(|c| DevRow { dst: base + c, src: -1, ds: [0, 0, 0, plane], ss: [0; 4], n: [1, 1, levels], count: levels }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// accumulate: the rule reads n/begin/end/dt and binds rate and total at begin; an overlap or a wrong layout refuses.
    #[test]
    fn accumulate_rule_binds_rate_and_total() {
        let i = |v: i32| submit::Argument::Bytes(v.to_le_bytes().to_vec());
        let f = |v: f32| submit::Argument::Bytes(v.to_le_bytes().to_vec());
        let d = |a: u64, o: usize| submit::Argument::Device(submit::Slice { allocation: a, offset: o });
        let el = |a: u64, o: usize| Ok::<i64, String>(a as i64 * 10_000 + (o / 4) as i64);
        let args = vec![i(2890), i(0), i(2890), f(60.0), f(100.0), i(1), d(1, 0), d(2, 0), d(3, 0)];
        match accumulate_rule(&args, el).unwrap() { R::Axpy { y, x, n, c } => assert_eq!((y, x, n, c), (20_000, 10_000, 2890, 60.0)), r => panic!("{r:?}") }
        let args = vec![i(2890), i(10), i(5000), f(30.0), f(0.0), i(0), d(1, 8), d(2, 0), d(3, 0)];
        match accumulate_rule(&args, el).unwrap() { R::Axpy { y, x, n, c } => assert_eq!((y, x, n, c), (20_010, 10_012, 2880, 30.0)), r => panic!("{r:?}") }
        let args = vec![i(2890), i(0), i(2890), f(60.0), f(100.0), i(1), d(1, 0), d(1, 400), d(3, 0)];
        assert!(accumulate_rule(&args, el).is_err());
        assert!(accumulate_rule(&args[..8], el).is_err());
    }
    /// machine: gather_quick answers as gather does (flag, and the tables whenever the launch is not injective).
    #[test]
    fn gather_quick_agrees_with_gather() {
        let rows: Vec<Vec<DevRow>> = vec![
            vec![DevRow { dst: 100, src: 0, ds: [0, 0, 1, 3], ss: [0, 0, 4, 1], n: [1, 3, 4], count: 12 }],           // transpose
            vec![DevRow { dst: 100, src: 0, ds: [0, 0, 4, 1], ss: [0, 0, 0, 1], n: [1, 3, 4], count: 12 }],           // broadcast
            vec![DevRow { dst: 1000, src: 7, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 300], count: 300 }],       // plain copy
            vec![DevRow { dst: 1000, src: 7, ds: [0, 0, 0, -1], ss: [0, 0, 0, 1], n: [1, 1, 300], count: 300 }],      // reversed
            vec![DevRow { dst: 1000, src: -1, ds: [0, 0, 0, 1], ss: [0; 4], n: [1, 1, 300], count: 300 }],            // sourceless
            vec![DevRow { dst: 1000, src: 7, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 300], count: 300 },
                 DevRow { dst: 1200, src: 4000, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 300], count: 300 }],    // destinations repeat
            vec![DevRow { dst: 1000, src: 7, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 300], count: 300 },
                 DevRow { dst: 2000, src: 200, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 300], count: 300 }],     // sources repeat
            vec![DevRow { dst: 5000, src: 0, ds: [600, 0, 0, 2], ss: [300, 0, 0, 1], n: [1, 1, 300], count: 900 }],   // strided, 3 outer blocks
            vec![DevRow { dst: 64, src: 128, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 64], count: 64 },
                 DevRow { dst: 0, src: 192, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 64], count: 64 }],          // word-aligned runs
        ];
        let mut injective = 0;
        for r in &rows {
            let (g, q) = (gather(r).unwrap(), gather_quick(r).unwrap());
            assert_eq!(g.injective, q.injective, "{r:?}");
            if g.injective { injective += 1; assert!(q.dst_word.is_empty()); }
            else if !q.fan.is_empty() {
                // machine round 2: a fan-in plan has no tables; it must transpose exactly as the tables do
                assert!(q.item.is_empty() && q.dst_word.is_empty());
                let mut x: Vec<f32> = (0..7000).map(|i| ((i * 7919 % 1013) as f32 - 500.0) * 1.0e-3).collect();
                let mut y = x.clone();
                apply_tables(&mut x, &g);
                apply_direct(&mut y, &q.direct); apply_fan(&mut y, &q.fan, &q.fan_groups);
                assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "{r:?}");
            }
            else { assert_eq!((&g.src_word, &g.start, &g.item, &g.dst_word, &g.sourceless), (&q.src_word, &q.start, &q.item, &q.dst_word, &q.sourceless)); }
        }
        assert_eq!(injective, 6);
        // an element both read and written keeps gather's refusal
        let overlap = [DevRow { dst: 10, src: 0, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, 20], count: 20 }];
        assert_eq!(gather(&overlap).err(), gather_quick(&overlap).err());
        assert!(gather_quick(&overlap).is_err());
    }
    /// machine: host model of the device transposes. `whole` applies a gather's tables as p8_gather_ad and
    /// p8_clear_dst do; `direct` applies rows as p8_rows_ad_direct does.
    fn apply_tables(a: &mut [f32], g: &Gather) {
        let old = a.to_vec();
        for (j, &s) in g.src_word.iter().enumerate() {
            let mut sum = old[s as usize] as f64;
            for k in g.start[j]..g.start[j + 1] { let it = g.item[k as usize]; if it & 1 == 0 { sum += old[g.dst_word[(it >> 1) as usize] as usize] as f64; } }
            a[s as usize] = sum as f32;
        }
        for &d in &g.dst_word { a[d as usize] = 0.0; }
    }
    fn apply_direct(a: &mut [f32], rows: &[DevRow]) {
        for r in rows {
            for e in 0..r.count {
                let mut q = e;
                let i3 = q % r.n[2]; q /= r.n[2]; let i2 = q % r.n[1]; q /= r.n[1]; let i1 = q % r.n[0]; q /= r.n[0];
                let d = (r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2] + i3 * r.ds[3]) as usize;
                let seed = a[d]; a[d] = 0.0;
                if r.src >= 0 { let s = (r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2] + i3 * r.ss[3]) as usize; a[s] = (a[s] as f64 + seed as f64) as f32; }
            }
        }
    }
    /// machine round 2: host model of p8_rows_ad_fan.
    fn apply_fan(a: &mut [f32], rows: &[DevRow], groups: &[i64]) {
        let off = |r: &DevRow, e: i64| {
            let mut q = e;
            let i3 = q % r.n[2]; q /= r.n[2]; let i2 = q % r.n[1]; q /= r.n[1]; let i1 = q % r.n[0]; q /= r.n[0];
            (r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2] + i3 * r.ds[3], r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2] + i3 * r.ss[3])
        };
        for g in groups.chunks(2) {
            let r0 = &rows[g[0] as usize];
            for e in 0..r0.count {
                let (_, s) = off(r0, e);
                let mut sum = if r0.src >= 0 { a[s as usize] as f64 } else { 0.0 };
                for j in 0..g[1] { let (d, _) = off(&rows[(g[0] + j) as usize], e); sum += a[d as usize] as f64; a[d as usize] = 0.0; }
                if r0.src >= 0 { a[s as usize] = sum as f32; }
            }
        }
    }
    #[test]
    fn mixed_launch_splits_and_transposes_like_the_whole_gather() {
        let row = |dst: i64, src: i64, n: i64| DevRow { dst, src, ds: [0, 0, 0, 1], ss: [0, 0, 0, 1], n: [1, 1, n], count: n };
        // rows 0 and 3 are plain copies; rows 1 and 2 read the same source words; row 4 is overwritten by row 5; row 6 fills
        let rows = vec![row(1000, 0, 100), row(1200, 200, 100), row(1400, 250, 100), row(1600, 400, 100),
                        row(1800, 500, 100), row(1850, 600, 100), row(2000, -1, 50)];
        let whole = gather(&rows).unwrap();
        let quick = gather_quick(&rows).unwrap();
        assert!(!whole.injective && !quick.injective);
        assert_eq!(quick.direct.len(), 3);
        assert_eq!(quick.direct.iter().map(|r| r.dst).collect::<Vec<_>>(), vec![1000, 1600, 2000]);
        assert!(quick.dst_word.len() < whole.dst_word.len());
        let mut x: Vec<f32> = (0..2100).map(|i| ((i * 7919 % 1013) as f32 - 500.0) * 1.0e-3).collect();
        let mut y = x.clone();
        apply_tables(&mut x, &whole);
        apply_direct(&mut y, &quick.direct); apply_tables(&mut y, &quick);
        assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()));
        std::env::set_var("AD_GATHER_SPLIT", "0");
        let off = gather_quick(&rows).unwrap();
        std::env::remove_var("AD_GATHER_SPLIT");
        assert!(off.direct.is_empty() && off.dst_word == whole.dst_word);
        // a destination interval that meets a source interval is left whole (gather's own refusal or tables)
        let crossing = vec![row(1000, 0, 100), row(50, 3000, 10), row(1200, 3000, 10)];
        assert!(split_independent(&crossing).is_none());
        // machine round 2, fan-in (in this test so that the off switches above and below never overlap).
        // rows 0, 1 and 2 read the same 100 source words (a field copied to three places); row 3 is a plain copy; row 4
        // broadcasts 40 source words along an axis of 3; row 5 fills. No tables, and the gather's result bit for bit.
        let rows = vec![row(1000, 0, 100), row(1200, 0, 100), row(1400, 0, 100), row(1100, 300, 100),
                        DevRow { dst: 2000, src: 500, ds: [0, 0, 40, 1], ss: [0, 0, 0, 1], n: [1, 3, 40], count: 120 }, row(2200, -1, 50)];
        let whole = gather(&rows).unwrap();
        let quick = gather_quick(&rows).unwrap();
        assert!(!whole.injective && !quick.injective);
        assert!(quick.item.is_empty() && quick.dst_word.is_empty() && !quick.fan.is_empty());
        assert_eq!(quick.fan_groups, vec![0, 3, 3, 3]);
        assert_eq!(quick.fan.iter().map(|r| r.dst).collect::<Vec<_>>(), vec![2000, 2040, 2080, 1400, 1200, 1000]);
        assert_eq!(quick.direct.iter().chain(&quick.fan).map(|r| r.count).sum::<i64>(), 4 * 100 + 120 + 50);
        let mut x: Vec<f32> = (0..2300).map(|i| ((i * 7919 % 1013) as f32 - 500.0) * 1.0e-3).collect();
        let mut y = x.clone();
        apply_tables(&mut x, &whole);
        apply_direct(&mut y, &quick.direct); apply_fan(&mut y, &quick.fan, &quick.fan_groups);
        assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()));
        assert!(x.iter().any(|v| *v != 0.0));
        std::env::set_var("AD_GATHER_FAN", "0");
        let off = gather_quick(&rows).unwrap();
        std::env::remove_var("AD_GATHER_FAN");
        assert!(off.fan.is_empty() && !off.item.is_empty());
        // sources that overlap without being the same map, a destination written twice, and a broadcast of 17 slices
        // keep the tables
        let long = DevRow { dst: 2000, src: 500, ds: [0, 0, 40, 1], ss: [0, 0, 0, 1], n: [1, 17, 40], count: 680 };
        for bad in [vec![row(1000, 0, 100), row(1200, 50, 100)], vec![row(1000, 0, 100), row(1050, 300, 100)], vec![long]] {
            let q = gather_quick(&bad).unwrap();
            assert!(q.fan.is_empty() && !q.item.is_empty(), "{bad:?}");
        }
    }
    #[test]
    fn gather_of_a_transpose_is_injective_and_a_broadcast_is_not() {
        let t = DevRow { dst: 100, src: 0, ds: [0, 0, 1, 3], ss: [0, 0, 4, 1], n: [1, 3, 4], count: 12 };
        let g = gather(&[t]).unwrap();
        assert!(g.injective);
        assert_eq!(g.dst_word.len(), 12);
        let b = DevRow { dst: 100, src: 0, ds: [0, 0, 4, 1], ss: [0, 0, 0, 1], n: [1, 3, 4], count: 12 };
        let g = gather(&[b]).unwrap();
        assert!(!g.injective);
        assert_eq!(g.src_word, vec![0, 1, 2, 3]);
        assert_eq!(g.start, vec![0, 3, 6, 9, 12]);
    }
    #[test]
    fn microphysics_top_level_is_an_exact_zero_rule() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| if a == 7 { Ok(100 + (o / 4) as i64) } else { Err(format!("allocation {a} has no shadow")) };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let args = vec![Argument::Bytes(59i32.to_le_bytes().to_vec()), Argument::Bytes(45000f32.to_le_bytes().to_vec()),
                        dev(5, 0), dev(6, 0), dev(7, 8)];
        match Ctx::integer_result("rrfs_init_microphysics_top", &args, el) {
            Some(Ok(R::Zero { ranges, .. })) => assert_eq!(ranges, vec![(102, 1)]),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::integer_result("rrfs_init_ocean_temperature", &args, el).is_none());
        assert!(matches!(Ctx::integer_result("rrfs_init_microphysics_top", &args[..4], el), Some(Err(_))));
        let bad = vec![args[0].clone(), args[1].clone(), dev(5, 0), dev(6, 0), dev(9, 0)];
        assert!(matches!(Ctx::integer_result("rrfs_init_microphysics_top", &bad, el), Some(Err(_))));
    }
    #[test]
    fn tbq_init_is_an_exact_constant_table_rule() {
        use submit::{Argument, Slice};
        match Ctx::module_constant("ruc_tbq_init", &[]) {
            Some(Ok(R::Identity(l))) => assert!(l.starts_with("inactive:ruc_tbq_init")),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::module_constant("oracle_init_rucinit", &[]).is_none());
        let one = vec![Argument::Device(Slice { allocation: 3, offset: 0 })];
        assert!(matches!(Ctx::module_constant("ruc_tbq_init", &one), Some(Err(_))));
    }
    #[test]
    fn speczone_fill_rows_cover_every_level_of_every_owner_once() {
        let rows = speczone_fill_rows(&[7, 2, 7, 0], 3, 10, 1000).unwrap();
        assert_eq!(rows.len(), 3);
        let mut words = Vec::new();
        for r in &rows {
            assert_eq!(r.src, -1);
            for e in 0..r.count { let i3 = e % r.n[2]; words.push(r.dst + i3 * r.ds[3]); }
        }
        words.sort();
        assert_eq!(words, vec![1000, 1002, 1007, 1010, 1012, 1017, 1020, 1022, 1027]);
        assert!(speczone_fill_rows(&[10], 3, 10, 0).is_err());
        assert!(speczone_fill_rows(&[-1], 3, 10, 0).is_err());
    }
    #[test]
    fn lbc_driving_launches_are_exact_zero_rules() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| match a { 10 => Ok((o / 4) as i64), 11 => Ok(5000 + (o / 4) as i64), _ => Err(format!("allocation {a} has no shadow")) };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let i = |v: i32| Argument::Bytes(v.to_le_bytes().to_vec());
        let x = |v: f32| Argument::Bytes(v.to_le_bytes().to_vec());
        // tendency: nvalues 1000 on 4 x 256 threads
        let t = vec![i(1000), x(1.0 / 3600.0), dev(1, 0), dev(2, 0), dev(10, 8)];
        match Ctx::regional_driving("regional_lbc_tendency_v841", &t, [4, 1, 1], [256, 1, 1], el) {
            Some(Ok(R::Zero { ranges, .. })) => assert_eq!(ranges, vec![(2, 1000)]),
            other => panic!("rule {other:?}"),
        }
        // a grid that does not cover nvalues is refused (R::Zero would clear unwritten words)
        assert!(matches!(Ctx::regional_driving("regional_lbc_tendency_v841", &t, [3, 1, 1], [256, 1, 1], el), Some(Err(_))));
        // rho_edge: grid-stride over nlev*nedges, two outputs
        let r = vec![i(55), i(2787), i(8500), dev(3, 0), dev(4, 0), dev(5, 0), dev(10, 0), dev(11, 0)];
        match Ctx::regional_driving("regional_lbc_rho_edge_v841", &r, [8, 1, 1], [256, 1, 1], el) {
            Some(Ok(R::Zero { ranges, .. })) => assert_eq!(ranges, vec![(0, 55 * 8500), (5000, 55 * 8500)]),
            other => panic!("rule {other:?}"),
        }
        // wrong layout, unshadowed output, and kernels outside the four
        assert!(matches!(Ctx::regional_driving("regional_lbc_state_at_v841", &t[..4], [4, 1, 1], [256, 1, 1], el), Some(Err(_))));
        let bad = vec![i(1000), x(15.0), dev(1, 0), dev(2, 0), dev(12, 0)];
        assert!(matches!(Ctx::regional_driving("regional_lbc_state_at_v841", &bad, [4, 1, 1], [256, 1, 1], el), Some(Err(_))));
        assert!(Ctx::regional_driving("regional_speczone_tend_cell_v841", &t, [4, 1, 1], [256, 1, 1], el).is_none());
    }
    #[test]
    fn rrtmg_lw_table_setup_is_an_exact_zero_rule() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| match a { 0 => Ok((o / 4) as i64), 1 => Ok(10112 + (o / 4) as i64), 2 => Ok(20224 + (o / 4) as i64),
                                              _ => Err(format!("allocation {a} has no shadow")) };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let (g, b) = ([79u32, 1, 1], [128u32, 1, 1]);
        let args = vec![dev(0, 0), dev(1, 0), dev(2, 0)];
        match Ctx::constant_tables("mpas_init_tables", &args, g, b, el) {
            Some(Ok(R::Zero { ranges, .. })) => assert_eq!(ranges, vec![(0, 10001), (10112, 10001), (20224, 10001)]),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::constant_tables("mpas_init_rwgt", &args, g, b, el).is_none());
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &args[..2], g, b, el), Some(Err(_))));
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &args, [78, 1, 1], b, el), Some(Err(_))));
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &args, g, b, el), Some(Ok(_))));
        let bad = vec![dev(0, 0), dev(1, 0), dev(9, 0)];
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &bad, g, b, el), Some(Err(_))));
        let unaligned = vec![dev(0, 0), dev(1, 2), dev(2, 0)];
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &unaligned, g, b, el), Some(Err(_))));
        let scalar = vec![dev(0, 0), Argument::Bytes(vec![0; 8]), dev(2, 0)];
        assert!(matches!(Ctx::constant_tables("mpas_init_tables", &scalar, g, b, el), Some(Err(_))));
    }
    #[test]
    fn par_sort_matches_sort() {
        let mut v: Vec<(i64, i64)> = (0..200_000).map(|i| ((i * 7919) % 100_003, i)).collect();
        let mut w = v.clone(); w.sort_unstable();
        par_sort(&mut v, 8);
        assert_eq!(v, w);
    }
    #[test]
    fn ruc_land_tile_preflight_stop_words_are_an_exact_zero_rule() {
        use submit::{Argument, Slice};
        let el = |a: u64, o: usize| if a == 14 { Ok(500 + (o / 4) as i64) } else { Err(format!("allocation {a} has no shadow")) };
        let bytes = |a: u64| if a == 14 || a == 1 { Some(512) } else { None };
        let dev = |a, o| Argument::Device(Slice { allocation: a, offset: o });
        let args = vec![dev(1, 0), dev(14, 8), Argument::Bytes(2562i32.to_le_bytes().to_vec())];
        match Ctx::integer_flags("ruc_land_tile_preflight", &args, el, bytes) {
            Ok(R::Zero { ranges, .. }) => assert_eq!(ranges, vec![(502, 2)]),
            other => panic!("rule {other:?}"),
        }
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &args[..2], el, bytes).is_err());
        let past_end = vec![dev(1, 0), dev(14, 508), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &past_end, el, bytes).is_err());
        let unaligned = vec![dev(1, 0), dev(14, 6), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &unaligned, el, bytes).is_err());
        let no_shadow = vec![dev(1, 0), dev(1, 0), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &no_shadow, el, bytes).is_err());
        let by_value_stop = vec![dev(1, 0), Argument::Bytes(vec![0; 8]), args[2].clone()];
        assert!(Ctx::integer_flags("ruc_land_tile_preflight", &by_value_stop, el, bytes).is_err());
    }
}

#[cfg(test)]
mod required_damping_tests {
    use super::require_damping_row;
    #[test]
    fn missing_local_recipe_cannot_be_an_identity_dot_pass() {
        let why = "derived binding edges_on_cell (local_edge_incidence) not in the derived store and no recipe";
        let e = require_damping_row("divergence_damping_selected_f32", true, Some(why), None, true).unwrap_err();
        assert_eq!(e, format!("DEFECT: divergence_damping_selected_f32: required damping derivative unavailable: {why}"));
    }
    #[test]
    fn both_damping_variants_require_both_entries_and_cannot_be_refused() {
        for f in ["divergence_damping_f32", "divergence_damping_selected_f32"] {
            assert!(require_damping_row(f, false, None, None, false).is_err());
            assert!(require_damping_row(f, true, None, None, false).is_err());
            assert!(require_damping_row(f, true, None, Some("diagnostic identity"), true).is_err());
            assert!(require_damping_row(f, true, None, None, true).is_ok());
        }
    }
}

/// Label of a host-checked concatenate launch; `t` is the map entry ("double:host-checked", or ":host-checked" when the
/// element type is not established).
fn host_checked_label(f: &str, t: &str) -> String {
    let ty = t.trim_end_matches(":host-checked");
    if ty.is_empty() { format!("inactive:{f} (vector read only by a host check, element type not established)") }
    else { format!("inactive:{f} (vector read only by a host check, typedef {ty} T)") }
}

#[cfg(test)]
mod concat_host_checked_tests {
    use serde_json::json;
    // by-value CArray: pointer word, size 16, shape 16, stride 8
    const Y: &str = "00f4bfbb2b7a0000100000000000000010000000000000000800000000000000";
    fn launch(module: &str, al: u64, off: u64) -> serde_json::Value {
        json!({"kind":"kernel","module":module,"function":"cupy_concatenate_same_size","arguments":[
            {"kind":"patched","bytes_hex":Y,"pointers":[{"byte_offset":0,"target":{"allocation":99,"offset":0}}]},
            {"kind":"bytes","bytes_hex":"0100000000000000"},
            {"kind":"patched","bytes_hex":Y,"pointers":[{"byte_offset":0,"target":{"allocation":al,"offset":off}}]}]})
    }
    fn d2h(al: u64, off: u64, bytes: u64, consumer: &str) -> serde_json::Value {
        json!({"kind":"device_to_host","source":{"allocation":al,"offset":off},"bytes":bytes,"host_consumer":consumer})
    }
    fn kernel(al: u64, off: u64) -> serde_json::Value {
        json!({"kind":"kernel","module":"m","function":"reader","arguments":[{"kind":"device","target":{"allocation":al,"offset":off}}]})
    }
    fn clear(al: u64, off: u64) -> serde_json::Value { json!({"kind":"clear","destination":{"allocation":al,"offset":off},"bytes":4,"value":0}) }
    fn set(ops: Vec<serde_json::Value>) -> std::collections::HashSet<String> { super::concat_host_checked(&json!({"blocks":{"step_01":ops}})) }
    #[test] fn check_then_clear_is_host_checked() {
        // a lower-offset neighbour in the same pool allocation does not count; the 128-byte vector is read by the check only
        assert!(set(vec![launch("a", 136, 512), d2h(136, 512, 128, "checked_value"), kernel(136, 0), clear(136, 512), kernel(136, 512)]).contains("a"));
        assert!(set(vec![launch("a", 136, 512), d2h(136, 512, 128, "checked_value")]).contains("a"));
    }
    #[test] fn a_kernel_reader_is_not() {
        assert!(set(vec![launch("a", 136, 512), d2h(136, 512, 128, "checked_value"), kernel(136, 512)]).is_empty());
        assert!(set(vec![launch("a", 136, 512), d2h(136, 512, 128, "checked_value"), kernel(136, 576)]).is_empty());
        assert!(set(vec![launch("a", 136, 512), kernel(136, 512), d2h(136, 512, 128, "checked_value")]).is_empty());
    }
    #[test] fn unchecked_partial_or_missing_download_is_not() {
        assert!(set(vec![launch("a", 136, 512), d2h(136, 512, 128, "value")]).is_empty());
        assert!(set(vec![launch("a", 136, 512), d2h(136, 512, 64, "checked_value")]).is_empty());
        assert!(set(vec![launch("a", 136, 512), clear(136, 512)]).is_empty());
        assert!(set(vec![launch("a", 136, 512)]).is_empty());
    }
    #[test] fn every_launch_of_the_module_must_pass() {
        let s = set(vec![launch("a", 136, 512), d2h(136, 512, 128, "checked_value"), launch("a", 200, 0), kernel(200, 0),
                         launch("b", 300, 0), d2h(300, 0, 128, "checked_zero_words")]);
        assert!(!s.contains("a") && s.contains("b"));
    }
    fn set2(blocks: serde_json::Value) -> std::collections::HashSet<String> { super::concat_host_checked(&json!({"blocks":blocks})) }
    #[test] fn a_vector_alive_at_the_end_of_its_step_must_be_named_nowhere_else() {
        // the 60 km shape: check, then nothing more in the step
        assert!(set2(json!({"step_01":[launch("a", 174, 2048), d2h(174, 2048, 128, "checked_value")], "step_02":[kernel(174, 0)]})).contains("a"));
        // another block names the bytes with a kernel: not proven
        assert!(set2(json!({"step_01":[launch("a", 174, 2048), d2h(174, 2048, 128, "checked_value")], "step_02":[kernel(174, 2048)]})).is_empty());
        assert!(set2(json!({"step_00":[kernel(174, 2100)], "step_01":[launch("a", 174, 2048), d2h(174, 2048, 128, "checked_value")]})).is_empty());
        // another block starts a new array exactly there first: proven
        assert!(set2(json!({"step_01":[launch("a", 174, 2048), d2h(174, 2048, 128, "checked_value")], "step_02":[clear(174, 2048), kernel(174, 2048)]})).contains("a"));
        // the launch's own block may run again: an earlier reader in it is a later reader
        assert!(set2(json!({"step_01":[kernel(174, 2048), launch("a", 174, 2048), d2h(174, 2048, 128, "checked_value")]})).is_empty());
        // dead inside its own step (the 120 km shape): other blocks reuse the pool block freely
        assert!(set2(json!({"step_01":[launch("a", 136, 512), d2h(136, 512, 128, "checked_value"), clear(136, 512)], "step_03":[kernel(136, 512)]})).contains("a"));
    }
    /// longwin: the ordered proof (concat_dead_in_order). Blocks run in the cadence order.
    fn set3(blocks: serde_json::Value, cadence: serde_json::Value) -> std::collections::HashSet<String> { super::concat_host_checked(&json!({"blocks":blocks,"cadence":cadence})) }
    fn copy(f: &str, al: u64, off: u64, arg: usize) -> serde_json::Value {
        let out = json!({"kind":"patched","bytes_hex":"00000000000000000100000000000000","pointers":[{"byte_offset":0,"target":{"allocation":al,"offset":off}}]});
        let other = json!({"kind":"patched","bytes_hex":"00000000000000000100000000000000","pointers":[{"byte_offset":0,"target":{"allocation":7,"offset":0}}]});
        json!({"kind":"kernel","module":"c","function":f,"arguments":if arg == 1 { json!([other, out]) } else { json!([out, other]) }})
    }
    fn cofrz(nlev: u32, al: u64, off: u64) -> serde_json::Value {
        let h: String = nlev.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect();
        json!({"kind":"kernel","module":"k","function":"acoustic_cofrz_v841","arguments":[{"kind":"bytes","bytes_hex":h},
            {"kind":"device","target":{"allocation":8,"offset":0}}, {"kind":"device","target":{"allocation":al,"offset":off}}]})
    }
    fn reduce(ws: (u64, u64), out: (u64, u64), inp: (u64, u64)) -> serde_json::Value {
        json!({"kind":"native_cub_reduce","op":2,"dtype":4,"count":{"allocation":900001,"offset":8},"workspace":{"allocation":ws.0,"offset":ws.1},
               "output":{"allocation":out.0,"offset":out.1},"input":{"allocation":inp.0,"offset":inp.1}})
    }
    #[test] fn in_cadence_order_an_earlier_use_of_the_pool_block_does_not_count() {
        let chk = || d2h(174, 2048, 128, "checked_value");
        // the 30-step shape: an earlier step used the same pool bytes with a kernel; the order-blind rule refuses, the ordered one proves
        let b = json!({"step_01":[kernel(174, 2048)], "step_02":[launch("a", 174, 2048), chk()]});
        assert!(set2(b.clone()).is_empty());
        assert!(set3(b.clone(), json!([["step_01"], ["step_02"]])).contains("a"));
        // the same blocks in the other order: the kernel now reads after the launch
        assert!(set3(b.clone(), json!([["step_02"], ["step_01"]])).is_empty());
        // a cadence that does not name every block exactly once is not used
        assert!(set3(b.clone(), json!([["step_02"]])).is_empty());
        assert!(set3(b, json!([["step_01"], ["step_02"], ["step_02"]])).is_empty());
        // earlier operations of the launch's own block do not count either
        assert!(set3(json!({"step_01":[kernel(174, 2048), launch("a", 174, 2048), chk()]}), json!([["step_01"]])).contains("a"));
    }
    #[test] fn a_device_written_new_array_at_the_first_byte_proves_the_vector_dead() {
        let chk = || d2h(174, 2048, 128, "checked_value");
        let one = |ops: Vec<serde_json::Value>| set3(json!({"step_01":[launch("a", 174, 2048), chk()], "step_02":ops}), json!([["step_01"], ["step_02"]]));
        // an int64 array filled word by word, then read by anything: the first user starts a new array
        assert!(one(vec![copy("cupy_copy__int64_int64", 174, 2048, 1), copy("cupy_copy__int64_int64", 174, 2056, 1), kernel(174, 2048)]).contains("a"));
        assert!(one(vec![copy("cupy_copy__int32_int64", 174, 2048, 1)]).contains("a"));
        assert!(one(vec![copy("cupy_copy__float_float32", 174, 2048, 1)]).contains("a"));
        // a float64 output could be an assignment into the vector itself; an input is a read; a later byte is not the start
        assert!(one(vec![copy("cupy_copy__int32_float64", 174, 2048, 1)]).is_empty());
        assert!(one(vec![copy("cupy_copy__int64_int64", 174, 2048, 0)]).is_empty());
        assert!(one(vec![copy("cupy_copy__int64_int64", 174, 2056, 1)]).is_empty());
        // the cofrz column: longer than the 128-byte vector proves it, not longer does not; 59 levels is the real case
        assert!(one(vec![cofrz(59, 174, 2048)]).contains("a"));
        assert!(one(vec![cofrz(32, 174, 2048)]).is_empty());
        assert!(one(vec![cofrz(59, 174, 2052)]).is_empty());
        // a reduce: result or work area at the first byte; an input there is a read
        assert!(one(vec![reduce((174, 2048), (174, 1536), (9, 0))]).contains("a"));
        assert!(one(vec![reduce((204, 0), (174, 2048), (9, 0))]).contains("a"));
        assert!(one(vec![reduce((204, 0), (9, 0), (174, 2048))]).is_empty());
        assert!(one(vec![reduce((174, 2048), (174, 2056), (9, 0))]).is_empty());
        // any other kernel, and a download that is not the check, still fail; so does a write before the check
        assert!(one(vec![kernel(174, 2048)]).is_empty());
        assert!(one(vec![d2h(174, 2048, 64, "value")]).is_empty());
        assert!(set3(json!({"step_01":[launch("a", 174, 2048), copy("cupy_copy__int64_int64", 174, 2048, 1), chk()]}), json!([["step_01"]])).is_empty());
        assert!(set3(json!({"step_01":[launch("a", 174, 2048)], "step_02":[kernel(9, 0)]}), json!([["step_01"], ["step_02"]])).is_empty());
    }
    #[test] fn a_module_with_no_established_type_is_marked_too() {
        use std::collections::{HashMap, HashSet};
        // the 60 km plan: b15be0f4's only launch names no recorded source, so the type map has no entry for it
        let mut types: HashMap<String, String> = [("f", "float"), ("i", "long long"), ("d", "double"), ("x", "double")].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        let checked: HashSet<String> = ["f", "i", "d", "none"].iter().map(|s| s.to_string()).collect();
        super::mark_host_checked(&mut types, &checked);
        assert_eq!(types["f"], "float"); assert_eq!(types["i"], "long long");
        assert_eq!(types["d"], "double:host-checked"); assert_eq!(types["x"], "double");
        assert_eq!(types["none"], ":host-checked");
        assert_eq!(super::host_checked_label("k", &types["d"]), "inactive:k (vector read only by a host check, typedef double T)");
        assert_eq!(super::host_checked_label("k", &types["none"]), "inactive:k (vector read only by a host check, element type not established)");
    }
    /// The proof on real plan packages (host only, no card):
    /// HEXAD_CONCAT_PLANS=<local path>,<kernel set> cargo test real_plans -- --ignored --nocapture
    #[test] #[ignore] fn real_plans_prove_the_sw_check_vector() {
        let list = std::env::var("HEXAD_CONCAT_PLANS").expect("HEXAD_CONCAT_PLANS");
        for p in list.split(',') {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            let s = super::concat_host_checked(&v);
            let mut names: Vec<&str> = s.iter().map(|m| &m[..8.min(m.len())]).collect(); names.sort();
            eprintln!("{p}: host-checked concatenate modules {names:?}");
            assert!(s.iter().any(|m| m.starts_with("b15be0f4")), "{p}: b15be0f4 not proven");
        }
    }
}

#[cfg(test)]
mod refuse_hold_tests {
    use super::*;
    fn view(base: i64, size: i64) -> DevView { DevView { base, size, ndim: 1, pad: 0, shape: [size, 1, 1, 1], stride: [1, 0, 0, 0] } }
    fn none() -> Option<Result<Vec<(i64, usize)>, String>> { None }
    /// D1: a kind named with --refuse writes zero tangent words (and its cotangent is consumed), never stale ones.
    #[test]
    fn a_listed_refusal_holds_what_the_kind_writes() {
        // the copy-rows rule of rrfs_radiation_unpack: every row becomes a fill row (source -1), destinations kept
        let rows = vec![DevRow { dst: 100, src: 7, ds: [1, 0, 0, 59], ss: [1, 0, 0, 59], n: [1, 1, 59], count: 4 },
                        DevRow { dst: 900, src: -1, ds: [1, 0, 0, 1], ss: [1, 0, 0, 1], n: [1, 1, 1], count: 2 }];
        match hold_refused("rrfs_radiation_unpack", "refused for this run (--refuse)", Ok(R::Rows { rows, label: "radiation unpack (held tendency)" }), none) {
            R::Rows { rows, label } => { assert_eq!(label, "radiation unpack (held tendency)"); assert_eq!(rows.len(), 2);
                assert!(rows.iter().all(|r| r.src == -1)); assert_eq!((rows[0].dst, rows[0].count, rows[1].dst), (100, 4, 900)); },
            r => panic!("{r:?}"),
        }
        // elementwise copy, max: the destination view is zeroed
        match hold_refused("cupy_copy__float32_float32", "w", Ok(R::View { dst: view(40, 8), src: view(0, 8) }), none) {
            R::ZeroView { dst, label } => { assert_eq!(dst, view(40, 8)); assert_eq!(label, "refused:cupy_copy__float32_float32"); }, r => panic!("{r:?}") }
        match hold_refused("k", "w", Ok(R::Max { out: view(5, 3), x: view(9, 3), prim: (1, 0, 3), xv: view(9, 3), c: 0.0 }), none) {
            R::ZeroView { dst, .. } => assert_eq!(dst, view(5, 3)), r => panic!("{r:?}") }
        // scale and copy: the output words
        match hold_refused("k", "w", Ok(R::Scale { input: 3, output: 11, n: 6, c: 2.0 }), none) { R::Zero { ranges, .. } => assert_eq!(ranges, vec![(11, 6)]), r => panic!("{r:?}") }
        match hold_refused("k", "w", Ok(R::Copy { dst: 21, src: 3, n: 9 }), none) { R::Zero { ranges, .. } => assert_eq!(ranges, vec![(21, 9)]), r => panic!("{r:?}") }
        // a row (entry): the tangent views its entries write, also when the entry is not dispatchable
        for own in [Ok(R::Entry), Ok(R::Refused("rlw_taut: entry not dispatchable: derived binding missing".into())), Err("allocation 7 has no shadow".to_string())] {
            match hold_refused("rlw_taut", "w", own, || Some(Ok(vec![(1000, 59), (2000, 118)]))) {
                R::Zero { ranges, label } => { assert_eq!(ranges, vec![(1000, 59), (2000, 118)]); assert_eq!(label, "refused:rlw_taut"); }, r => panic!("{r:?}") }
        }
        // no float word written: identity is exact and stays
        match hold_refused("cupy_invert__bool_bool", "w", Ok(R::Identity("inactive:cupy_invert__bool_bool (bool or integer result)".into())), || panic!("row not consulted")) {
            R::Identity(s) => assert!(s.starts_with("inactive:")), r => panic!("{r:?}") }
    }
    /// D1b: a refused P7 gas-optics kind (no kernel-set row, no rule here) is held on the outputs its native row names.
    #[test]
    fn a_refused_native_row_is_held_on_the_outputs_its_row_names() {
        for f in ["rlw_taumol_batched", "rsw_taumol_b"] {
            let kept = || hold_refused(f, "refused for this run (--refuse)", Ok(R::Refused(format!("{f}: no derivative row and no rule"))), none);
            assert!(matches!(kept(), R::Refused(_)), "without the native row the kind keeps identity");
            match hold_refused_native(f, kept(), Some(Ok(vec![(5000, 840), (9000, 840)]))) {
                R::Zero { ranges, label } => { assert_eq!(ranges, vec![(5000, 840), (9000, 840)]); assert_eq!(label, format!("refused:{f}")); }, r => panic!("{r:?}") }
            // not such a row, no output, or outputs that do not resolve: identity stays and is named, and the text
            // never reads like an unlisted refusal
            for w in [None, Some(Ok(Vec::new())), Some(Err("argument taug is not a device pointer".to_string()))] {
                match hold_refused_native(f, kept(), w) {
                    R::Refused(s) => { assert!(s.contains("identity kept"), "{s}");
                        assert!(!s.ends_with(": no derivative row and no rule") && !s.contains(": entry not dispatchable: "), "{s}"); },
                    r => panic!("{r:?}") }
            }
        }
        // a kind that already has a hold keeps it
        match hold_refused_native("k", R::Zero { ranges: vec![(1, 2)], label: "refused:k".into() }, Some(Ok(vec![(7, 7)]))) {
            R::Zero { ranges, .. } => assert_eq!(ranges, vec![(1, 2)]), r => panic!("{r:?}") }
    }
    /// D1c: the receipt lists every listed kind that ran as identity over float words: unknown writes AND the
    /// class-held physics launches. A kind that writes no float word, or is held, is not listed.
    #[test]
    fn the_receipt_lists_class_held_identity_with_the_unknown_writes() {
        let listed = |r: R| refuse_ran_identity(&refuse_how(&r));
        assert!(listed(R::Identity("held:tempo".into())));
        assert!(listed(R::Identity("held:?".into())));
        assert!(listed(R::Refused("solar: refused for this run (--refuse) (identity kept, writes unknown: no kernel-set row)".into())));
        assert!(!listed(R::Identity("inactive:cupy_invert__bool_bool (bool or integer result)".into())));
        assert!(!listed(R::Identity("inactive:tempo_order_scan (COMMON-AD 87.1 column-order permutation)".into())));
        assert!(!listed(R::Zero { ranges: vec![(1, 2)], label: "refused:k".into() }));
        assert!(!listed(R::ZeroView { dst: view(0, 4), label: "refused:k".into() }));
        assert!(!listed(R::Rows { rows: Vec::new(), label: "radiation unpack (held tendency)" }));
        assert!(refuse_how(&R::Identity("held:gsl_drag".into())).ends_with(": held:gsl_drag"));
    }
    /// What a kind writes is unknown (no rule and no row, a masked write, views that do not resolve): identity is kept
    /// and named, and the text never reads like an unlisted refusal (window.rs stops on those two endings).
    #[test]
    fn a_listed_refusal_with_unknown_writes_keeps_identity_and_says_so() {
        let cases: Vec<(Result<R, String>, Option<Result<Vec<(i64, usize)>, String>>)> = vec![
            (Ok(R::Refused("solar: no derivative row and no rule".into())), None),
            (Ok(R::Refused("x: entry not dispatchable: y".into())), None),
            (Ok(R::Where { dst: view(0, 4), src: view(8, 4), prim: (1, 0, 4), mv: view(16, 4) }), None),
            (Ok(R::Entry), Some(Ok(Vec::new()))),
            (Ok(R::Entry), Some(Err("n is not a device argument".into()))),
            (Err("allocation 3 has no shadow".into()), None)];
        for (own, row) in cases {
            match hold_refused("solar", "refused for this run (--refuse)", own, || row) {
                R::Refused(s) => {
                    assert!(s.starts_with("solar: refused for this run (--refuse) (identity kept, writes unknown: "), "{s}");
                    assert!(!s.ends_with(": no derivative row and no rule") && !s.contains(": entry not dispatchable: "), "{s}");
                },
                r => panic!("{r:?}"),
            }
        }
    }
}
