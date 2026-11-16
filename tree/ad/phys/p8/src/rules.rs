// SPDX-License-Identifier: Apache-2.0
//! Derivative assignment of every command of the hrrrv5 physics plan, and the
//! resolver that binds every kernel launch's pointers to live plan storage.
use super::plan::{args, carray_span, cupy_items, Arg, Plan, Slice};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

const GLUE: &[&str] = &["prep_mass_v841_f32", "prep_interfaces_v841_f32", "project_edges_v841_f32",
    "mpas_transport_source_baseline_v842_f32", "mpas_transport_handoff_v842_f32", "rrfs_pbl_radiation_sum",
    "rrfs_surface_spacing", "fork_albedo", "fork_interp", "fork_minmax"];
const DIAG: &[&str] = &["smooth_helicity", "split_helicity_pool", "vertex_helicity_gather_pool", "update_diag_pool",
    "mpas_output_diagnostics_v842", "oracle_diags_sfcdiags_ruclsm", "tempo_radar",
    // L2 (ad/window-L2-bindings): the output-step diagnostic modules of the 18-step plan, classed "unknown" before
    // (bind18, steps 9 and 18: 508 launches). hexcore/mpasdiag/isobaric.py (the MPAS-A v8.4.2 isobaric module) and
    // mpasdiag/convective_rrfs.py (the RRFS convective diagnostics), by their source modules.
    "fixed_pressure_apply", "fixed_pressure_count", "fixed_pressure_scan", "prepare_mass", "prepare_interfaces",
    "prepare_vertices", "slp_columns", "slp_finish", "layer_mean", "compute_diag_cols", "reset_diag_pool"];
const VALIDATE: &[&str] = &["validate_density_v841", "validate_finite_array_v841_f32", "validate_recovered_v841_f32",
    "w_filter_geometry_v841_f32"];

/// (class, binding) of a kernel. Classes: rule:* (this unit), entry:* (built
/// derivative entry), pending:Pn (fused physics entry owned by unit Pn),
/// inactive:* (no derivative, outputs zero in both directions).
pub fn kernel_class(f: &str, e2e: &HashSet<String>) -> (&'static str, String) {
    if e2e.contains(f) { return ("entry", "ad/e2e kernels.json (U1-U3)".into()); }
    if GLUE.contains(&f) { return ("entry", format!("ad2-hrrrv5-glue {f}__tl/__ad")); }
    if DIAG.contains(&f) { return ("inactive", "diagnostic output, no prognostic use".into()); }
    if VALIDATE.contains(&f) { return ("inactive", "validator / fixed geometry (U7 rule)".into()); }
    match f {
        "rrfs_post_transport_copy" | "rrfs_glue_copy_plan" =>
            return ("rule", "rows: copy TL, parallel gather transpose".into()),
        // step6-sw (review finding ad/phys-p8-check rules.rs:25): the name carries no dtype. A float32 instantiation is a
        // copy rule and an integer one is inactive; only the module's recorded source says which (resolve() below, the
        // window's wrules::concatenate). By name the class is not established, so no table row may bind it by name.
        "cupy_concatenate_same_size" => return ("unknown", "element type is a module property (recorded source typedef); bound per module, never by name".into()),
        "cupy_copy_where__float32_bool_float32" => return ("rule", "where: executed arm of the forward mask".into()),
        "cupy_maximum__float_float32_float32" => return ("rule", "max: executed arm (x wins only if x > c)".into()),
        "cupy_fill" => return ("inactive", "constant fill".into()),
        "cupy_less__float32_float32_bool" | "cupy_not_equal__int32_int_bool" | "cupy_cub_any_pass1" | "cupy_cub_any_pass2"
        | "hex_transport_species_words" | "hex_transport_species_levels" => return ("inactive", "bool or integer result".into()),
        // L2 (ad/window-L2-bindings): P3 and P4 are bound as ad.scheme.rows/v1 rows (tree/ad/phys/bind).
        "rrfs_mynnedmf_run_l16" => return ("entry", "P3 MYNN-EDMF whole launch: ad.scheme.rows/v1 row".into()),
        "mynnsfc_tables" => return ("inactive", "constant psi tables (P4-ENTRIES); the derivative modules build their own".into()),
        // ad/held-oracle_ice_ruc_ice: the P6 sea-ice whole launch is bound as an ad.scheme.rows/v1 row (bind/p6_row.py).
        "oracle_ice_ruc_ice" => return ("entry", "P6 RUC sea-ice whole launch: ad.scheme.rows/v1 row".into()),
        // ad/step6-rucland: the RUC land whole launch (entry ad/held-oracle_land_ruc_land) is bound as an ad.scheme.rows/v1 row
        // (kind ruc_land, bind/ruc_land_row.py). Before, the oracle_ prefix below classed it pending:P6 and the window held it.
        "oracle_land_ruc_land" => return ("entry", "RUC land whole launch: ad.scheme.rows/v1 row (kind ruc_land)".into()),
        "cupy_multiply__float32_float_float32" => return ("rule", "scale: d_out = c d_x, c a host constant".into()),
        // ad/held-rlw_taumol_batched: P7's generated RRTMG-LW gas-optics whole launch, bound as an ad.scheme.rows/v1 row.
        "rlw_taumol_batched" => return ("entry", "P7 RRTMG-LW gas optics whole launch: ad.scheme.rows/v1 row (kind p7_taumol)".into()),
        // ad/held-rsw_taumol_b: P7's generated RRTMG-SW gas-optics whole launch, bound as an ad.scheme.rows/v1 row.
        "rsw_taumol_b" => return ("entry", "P7 RRTMG-SW gas optics whole launch: ad.scheme.rows/v1 row (kind p7_rsw_taumol)".into()),
        _ => {}
    }
    if let Some(d) = f.strip_prefix("cupy_copy__") {
        let t: Vec<&str> = d.split('_').collect();
        let float = |s: &str| s.starts_with("float");
        if t.len() == 2 && float(t[0]) && float(t[1]) { return ("rule", "view copy (linear, cast exact)".into()); }
        return ("inactive", "integer or bool source".into());
    }
    // TEMPO column-order kernels (tempo_v303.py:4027-4046): uint32 bucket counts, their in-place exclusive scan and the
    // int32 column order, read only as a schedule by tempo_warp_steps (columns are independent, so the order changes no
    // word). No float word is read or written (recorded sm_120 SASS: IADD/LDG/STG only), so they are inactive (COMMON-AD
    // 87.1). BREAKAGE THIS FIXES: the tempo_ prefix below classed them pending:P5, so the per-launch dot reported them
    // "no derivative row; P8 table: pending: P5" (held physics) and an inactive row for them was refused as disagreeing.
    if matches!(f, "tempo_order_hist" | "tempo_order_scan" | "tempo_order_scatter") {
        return ("inactive", "TEMPO column-order permutation: integer bucket counts and column order only (COMMON-AD 87.1)".into());
    }
    if f.starts_with("tempo_") { return ("pending", "P5 (TEMPO whole launches)".into()); }
    if f.starts_with("mynnsfc_") || f.starts_with("ugwp_oro_") { return ("entry", "P4 surface layer / GSL drag whole launch: ad.scheme.rows/v1 row".into()); }
    if f.starts_with("oracle_") { return ("pending", "P6 (RUC sea ice and land)".into()); }
    ("unknown", String::new())
}

/// Rule of a non-kernel command.
pub fn command_class(op: &Value) -> (&'static str, &'static str) {
    match op["kind"].as_str().unwrap_or("") {
        "device_copy" => ("rule", "copy: d_dst = d_src; b_src += b_dst, b_dst = 0"),
        "clear" => ("rule", "zero"),
        "upload" if op["pointers"].as_array().is_some_and(|p| !p.is_empty()) => ("table", "pointer table (no derivative)"),
        "upload" => ("rule", "immutable input: zero tangent, cotangent stops"),
        "held_upload_ranges" => ("rule", "held slot: zero tangent; cotangent recorded, then stops"),
        "rrfs_copy_plan_upload" | "ruc_pointer_table_upload" | "tempo_pointer_table_bind" => ("table", "pointer table (no derivative)"),
        // CuPy cub dtype codes 0-7 are the integer types (common.pxd: INT8 0 .. UINT64 7);
        // a float reduction would need a derivative, so it is not silently inactive.
        "native_cub_reduce" if op["dtype"].as_u64().is_some_and(|d| d <= 7) => ("inactive", "integer reduction"),
        "native_cublas_transpose" => ("rule", "rows: float32 transpose (step 1 setup)"),
        "device_to_host" | "synchronization" | "allocate" | "free_null" | "host_boundary" => ("noop", "no derivative work"),
        _ => ("unknown", ""),
    }
}

pub struct Resolved { pub records: Vec<Value>, pub summary: Value }

/// Element type `T` of each module whose recorded CuPy source the plan names
/// (an argument's source reference `recording/sources/<sha>.cu`, read beside the
/// plan). The kernel name of cupy_concatenate_same_size carries no dtype, so its
/// derivative class comes from here.
pub fn module_types(plan: &Plan, plan_path: &str) -> HashMap<String, String> {
    let dir = std::path::Path::new(plan_path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut out = HashMap::new();
    for (_, block) in plan.steps() {
        for op in plan.ops(&block) {
            if op["kind"] != "kernel" { continue; }
            let module = op["module"].as_str().unwrap_or("").to_string();
            if out.contains_key(&module) { continue; }
            for a in op["arguments"].as_array().into_iter().flatten() {
                let src = [&a["source"]["source"], &a["fixed_source"]["source"]].into_iter()
                    .filter_map(|v| v.as_str()).find(|s| s.starts_with("recording/sources/"));
                let Some(src) = src else { continue };
                let Ok(text) = std::fs::read_to_string(dir.join(src)) else { continue };
                if let Some(t) = text.lines().map(str::trim).find_map(|l| l.strip_prefix("typedef ").and_then(|r| r.strip_suffix(" T;"))) {
                    out.insert(module.clone(), t.to_string());
                }
                break;
            }
        }
    }
    out
}

pub fn resolve(plan: &Plan, e2e: &HashSet<String>, types: &HashMap<String, String>) -> Resolved {
    let mut bytes: HashMap<u64, usize> = plan.bytes.clone();
    let mut tables: HashMap<Slice, Vec<Option<Slice>>> = HashMap::new();
    let mut writer: HashMap<Slice, &'static str> = HashMap::new();
    let mut records = Vec::new();
    let mut by_class: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_kind: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_fn: BTreeMap<String, (String, u64)> = BTreeMap::new();
    let (mut launches, mut bound, mut bad_ptr, mut refused) = (0u64, 0u64, 0u64, Vec::new());
    for (step, block) in plan.steps() {
        for (i, op) in plan.ops(&block).iter().enumerate() {
            let kind = op["kind"].as_str().unwrap_or("").to_string();
            *by_kind.entry(kind.clone()).or_default() += 1;
            match kind.as_str() {
                "allocate" => { bytes.insert(op["id"].as_u64().unwrap(), op["bytes"].as_u64().unwrap() as usize); },
                "tempo_pointer_table_bind" => {
                    let e = op["entries"].as_array().into_iter().flatten()
                        .map(|x| if x.is_null() { None } else { Some(super::plan::slice(&x["target"])) }).collect();
                    { let d = super::plan::slice(&op["destination"]); tables.insert(d, e); writer.insert(d, "tempo"); }
                },
                "ruc_pointer_table_upload" => {
                    let e = op["arguments"].as_object().into_iter().flatten()
                        .map(|(_, x)| Some(super::plan::slice(&x["target"]))).collect();
                    { let d = super::plan::slice(&op["destination"]); tables.insert(d, e); writer.insert(d, "ruc"); }
                },
                "rrfs_copy_plan_upload" => {
                    let e = op["rows"].as_array().into_iter().flatten().flat_map(|r|
                        [Some(super::plan::slice(&r["dst"])), if r["src"].is_null() { None } else { Some(super::plan::slice(&r["src"])) }]).collect();
                    { let d = super::plan::slice(&op["destination"]); tables.insert(d, e); writer.insert(d, "copy_plan"); }
                },
                "upload" if op["pointers"].as_array().is_some_and(|p| !p.is_empty()) => {
                    let e = op["pointers"].as_array().unwrap().iter().map(|p| Some(super::plan::slice(&p["target"]))).collect();
                    { let d = super::plan::slice(&op["destination"]); tables.insert(d, e); writer.insert(d, "pointer_upload"); }
                },
                _ => {}
            }
            if kind != "kernel" {
                let (c, _) = command_class(op);
                *by_class.entry(format!("command:{c}")).or_default() += 1;
                if c == "unknown" { refused.push(json!({"step": step, "index": i, "kind": kind})); }
                continue;
            }
            launches += 1;
            let f = op["function"].as_str().unwrap_or("");
            let (class, binding) = if f == "cupy_concatenate_same_size" {
                match types.get(op["module"].as_str().unwrap_or("")).map(String::as_str) {
                    Some("float") => ("rule", "rows: copy TL, parallel gather transpose (float32)".to_string()),
                    Some(t @ ("long long" | "long" | "int" | "unsigned long long" | "unsigned int" | "bool")) =>
                        ("inactive", format!("integer concatenation (typedef {t} T)")),
                    t => ("unknown", format!("element type {t:?} not established")),
                }
            } else { kernel_class(f, e2e) };
            *by_class.entry(format!("kernel:{class}")).or_default() += 1;
            by_fn.entry(f.to_string()).or_insert((format!("{class}: {binding}"), 0)).1 += 1;
            // Bind every pointer argument to live storage.
            let items = cupy_items(f);
            let mut array_k = 0usize;
            let mut ptrs = Vec::new(); let mut ok = class != "unknown";
            let live = |s: Slice, lo: i64, hi: i64| bytes.get(&s.alloc).is_some_and(|&b| s.off as i64 + lo >= 0 && s.off as i64 + hi <= b as i64);
            for (k, a) in args(op).iter().enumerate() {
                match a {
                    Arg::Ptr(s) => {
                        let fine = live(*s, 0, 1);
                        let mut rec = json!({"arg": k, "alloc": s.alloc, "offset": s.off, "bytes": "kernel-defined", "live": fine});
                        if let Some(t) = tables.get(s) {
                            let good = t.iter().flatten().all(|e| live(*e, 0, 1));
                            rec["indirect"] = json!({"entries": t.len(), "null": t.iter().filter(|e| e.is_none()).count(), "live": good});
                            ok &= good;
                        }
                        ok &= fine; ptrs.push(rec);
                    },
                    Arg::CArray { at, size, shape, strides } => {
                        let item = items.get(array_k).copied().unwrap_or_else(|| strides.iter().copied().filter(|&s| s > 0).min().unwrap_or(4));
                        array_k += 1;
                        let (lo, hi) = carray_span(shape, strides, item);
                        let fine = *size == 0 || live(*at, lo, hi);
                        ok &= fine;
                        ptrs.push(json!({"arg": k, "alloc": at.alloc, "offset": at.off, "span": [lo, hi], "item": item,
                                         "shape": shape, "strides": strides, "live": fine}));
                    },
                    Arg::Packed { ptrs: p, .. } => for (o, s) in p {
                        let fine = live(*s, 0, 1); ok &= fine;
                        ptrs.push(json!({"arg": k, "byte": o, "alloc": s.alloc, "offset": s.off, "live": fine}));
                    },
                    _ => {}
                }
            }
            if ptrs.iter().any(|p| p["live"] == false) { bad_ptr += 1; }
            // A by-value pointer array (CuPy concatenate) must have been written by a
            // pointer upload of its own length; anything else is a plan gap.
            let mut reason = Value::Null;
            if f == "cupy_concatenate_same_size" {
                if let Some(Arg::CArray { at, size, .. }) = args(op).first() {
                    let good = writer.get(at) == Some(&"pointer_upload") && tables.get(at).is_some_and(|t| t.len() as i64 == *size);
                    if !good { ok = false; reason = json!(format!("pointer array at allocation {} is not written by a pointer upload in this plan (last writer: {}); one of the small pointer-table uploads v26 did not retain (P1)", at.alloc, writer.get(at).copied().unwrap_or("initial bytes"))); }
                }
            }
            if ok { bound += 1; } else { refused.push(json!({"step": step, "index": i, "function": f, "class": class, "reason": reason})); }
            records.push(json!({"step": step, "block": block, "index": i, "function": f, "class": class,
                                "binding": binding, "pointers": ptrs, "bound": ok}));
        }
    }
    let fns: BTreeMap<String, Value> = by_fn.into_iter().map(|(k, (c, n))| (k, json!({"launches": n, "assignment": c}))).collect();
    let summary = json!({"steps": plan.steps().len(), "kernel_launches": launches, "bound": bound,
        "pointer_failures": bad_ptr, "refused": refused, "classes": by_class, "kinds": by_kind, "functions": fns});
    Resolved { records, summary }
}

#[cfg(test)]
mod order_class_tests {
    use super::kernel_class;
    use std::collections::HashSet;
    // ad/held-tempo_order_scan: the order kernels are inactive (COMMON-AD 87.1); every other TEMPO launch stays P5.
    #[test]
    fn tempo_order_kernels_inactive_rest_pending() {
        let none = HashSet::new();
        for f in ["tempo_order_hist", "tempo_order_scan", "tempo_order_scatter"] { assert_eq!(kernel_class(f, &none).0, "inactive", "{f}"); }
        for f in ["tempo_warp_steps", "tempo_from", "tempo_to", "tempo_precip", "tempo_emissions"] { assert_eq!(kernel_class(f, &none).0, "pending", "{f}"); }
        assert_eq!(kernel_class("tempo_radar", &none).0, "inactive");
        // step6-sw: concatenate's class is a module property, never established by name.
        assert_eq!(kernel_class("cupy_concatenate_same_size", &none).0, "unknown");
    }
}
