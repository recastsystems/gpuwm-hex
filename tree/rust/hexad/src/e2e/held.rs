// SPDX-License-Identifier: Apache-2.0
//! P9 (autodiff physics rung): apply the physics scheme manifest to the reverse dispatcher.
//!
//! With `--schemes tree/ad/phys/p9/schemes.json`, kernels the manifest marks
//! held (fused physics launches P3-P6 have not bound yet), permutation (TEMPO
//! column ordering, integer only) or constant (step-1 tables) get the class
//! "held", and inactive ones the class "output-diagnostic"; the sweep gives
//! "held" the identity rule, the exact transpose of the tangent dispatcher's
//! pass-through. Copies and glue (manifest "rule" or "live-glue") are never
//! held: holding a copy keeps the destination's old tangent instead of the
//! source's, which cuts the state chain. Those with no loaded entry and no
//! exact rule in this sweep get the class "needs-rule" and are refused by
//! name. `--hold s1,s2` (or `physics`) also drops loaded entries of those
//! physics schemes, for one-scheme-live attribution runs. Without `--schemes`
//! nothing changes.
use crate::deriv::KernelDeriv;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// Schemes that move or advance the state: `--hold` refuses them.
const NOT_HOLDABLE: [&str; 4] = ["dycore", "elementwise", "glue", "diagnostics"];
/// Element-wise kernels the reverse sweep already transposes exactly (reverse.rs `rule`).
const SWEEP_RULES: [&str; 3] = ["cupy_copy__float32_float32", "cupy_fill", "cupy_multiply__float32_float_float32"];

pub fn apply(a: &HashMap<String, String>, kernels: &mut HashMap<String, KernelDeriv>,
             classes: &mut HashMap<String, String>) -> Result<Value, String> {
    let Some(path) = a.get("schemes") else { return Ok(Value::Null) };
    let doc: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
        .map_err(|e| format!("{path}: {e}"))?;
    let mut summary = assign(&doc, a.get("hold").map(String::as_str), &|k| kernels.contains_key(k), classes)
        .map_err(|e| format!("{path}: {e}"))?;
    for k in &summary["entries_dropped"].as_array().cloned().unwrap_or_default() {
        if let Some(k) = k.as_str() { kernels.remove(k); }
    }
    summary["manifest"] = json!(path);
    println!("{summary}");
    Ok(summary)
}

/// Class assignment from the manifest; `loaded(k)` says whether kernel k has a loaded entry.
/// Returns the summary ("entries_dropped" lists the loaded entries a forced hold removes).
pub fn assign(doc: &Value, hold_arg: Option<&str>, loaded: &dyn Fn(&str) -> bool,
              classes: &mut HashMap<String, String>) -> Result<Value, String> {
    if doc["schema"] != "ad.phys.p9-schemes/v1" { return Err("not ad.phys.p9-schemes/v1".into()); }
    let rows = doc["rows"].as_object().ok_or("schemes rows")?;
    let mut hold = HashSet::new();
    for s in hold_arg.unwrap_or("").split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if s == "physics" {
            for r in rows.values() {
                let scheme = r["scheme"].as_str().unwrap_or("");
                if !NOT_HOLDABLE.contains(&scheme) { hold.insert(scheme.to_owned()); }
            }
        } else if NOT_HOLDABLE.contains(&s) {
            return Err(format!("--hold {s}: not a physics scheme (holding copies or the dycore cuts the state chain)"));
        } else if rows.values().any(|r| r["scheme"] == s) { hold.insert(s.to_owned()); }
        else { return Err(format!("--hold names unknown scheme {s}")); }
    }
    let (mut held, mut dropped, mut needs) = (0_u64, Vec::new(), Vec::new());
    for (kernel, r) in rows {
        let scheme = r["scheme"].as_str().unwrap_or("");
        let status = r["p9"].as_str().unwrap_or("");
        let forced = hold.contains(scheme);
        if forced && loaded(kernel) { dropped.push(kernel.clone()); }
        if loaded(kernel) && !forced { continue; }
        match status {
            "held" | "permutation" | "constant" => { classes.insert(kernel.clone(), "held".into()); held += 1; },
            "inactive" => { classes.entry(kernel.clone()).or_insert_with(|| "output-diagnostic".into()); },
            _ if forced => { classes.insert(kernel.clone(), "held".into()); held += 1; },
            "rule" if SWEEP_RULES.contains(&kernel.as_str()) => {},
            "rule" | "live-glue" => { classes.insert(kernel.clone(), "needs-rule".into()); needs.push(kernel.clone()); },
            _ => {},
        }
    }
    let mut h: Vec<_> = hold.into_iter().collect();
    h.sort(); dropped.sort(); needs.sort();
    Ok(json!({"event":"p9_schemes","held_kernels":held,"hold":h,"entries_dropped":dropped,"refused_without_rule":needs}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn doc() -> Value {
        json!({"schema":"ad.phys.p9-schemes/v1","rows":{
            "tempo_warp_steps":{"scheme":"tempo","p9":"held"},
            "mynnsfc_columns":{"scheme":"mynn_sfc","p9":"held"},
            "cupy_copy__float32_float64":{"scheme":"elementwise","p9":"rule"},
            "cupy_copy__float32_float32":{"scheme":"elementwise","p9":"rule"},
            "cupy_copy__int32_int64":{"scheme":"elementwise","p9":"inactive"},
            "rrfs_post_transport_copy":{"scheme":"glue","p9":"rule"},
            "prep_mass_v841_f32":{"scheme":"glue","p9":"live-glue"},
            "acoustic_ru_v841":{"scheme":"dycore","p9":"live-e2e"}}})
    }
    #[test]
    fn copies_and_glue_are_never_held() {
        let mut c = HashMap::new();
        let loaded = |k: &str| k == "acoustic_ru_v841" || k == "mynnsfc_columns";
        let s = assign(&doc(), None, &loaded, &mut c).unwrap();
        assert_eq!(c.get("tempo_warp_steps").map(String::as_str), Some("held"));
        assert_eq!(c.get("cupy_copy__int32_int64").map(String::as_str), Some("output-diagnostic"));
        for k in ["cupy_copy__float32_float64", "rrfs_post_transport_copy", "prep_mass_v841_f32"] {
            assert_eq!(c.get(k).map(String::as_str), Some("needs-rule"), "{k}");
        }
        assert!(c.get("cupy_copy__float32_float32").is_none() && c.get("mynnsfc_columns").is_none());
        assert_eq!(s["entries_dropped"], json!([]));
    }
    #[test]
    fn hold_physics_drops_physics_entries_only() {
        let mut c = HashMap::new();
        let loaded = |k: &str| k == "acoustic_ru_v841" || k == "mynnsfc_columns";
        let s = assign(&doc(), Some("physics"), &loaded, &mut c).unwrap();
        assert_eq!(s["entries_dropped"], json!(["mynnsfc_columns"]));
        assert_eq!(c.get("mynnsfc_columns").map(String::as_str), Some("held"));
        assert!(c.get("acoustic_ru_v841").is_none());
        assert!(assign(&doc(), Some("glue"), &loaded, &mut HashMap::new()).is_err());
    }
}
