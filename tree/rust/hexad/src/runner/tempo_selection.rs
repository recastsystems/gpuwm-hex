// SPDX-License-Identifier: Apache-2.0
//! TEMPO carrier order and presence checks from launch_mpas. The caller must
//! establish that these named views and configuration flags are live.
//! Source: rrfs/tempo_v303.py:4580,4596-4614.
#[path = "tempo_layout.rs"]
mod layout;
use super::View;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Carriers {
    pub fields: HashMap<String, Option<View>>,
    pub model: HashMap<String, Option<View>>,
    pub mesh: HashMap<String, Option<View>>,
    pub diagnostics: HashMap<String, Option<View>>,
    pub tendencies: HashMap<String, Option<View>>,
    pub surface: HashMap<String, Option<View>>,
    pub precip: HashMap<String, Option<View>>,
    pub profiles: HashMap<String, Option<View>>,
}

pub struct Table {
    pub name: &'static str,
    pub entries: Vec<Option<View>>,
    pub shape: Option<Vec<usize>>,
    pub optional: bool,
    pub device: u64,
}

fn require(values: &HashMap<String, Option<View>>, names: &[&str]) -> Result<(), String> {
    for name in names {
        if !matches!(values.get(*name), Some(Some(_))) {
            return Err(format!("allocated native TEMPO carrier {name} missing"));
        }
    }
    Ok(())
}

fn table(name: &'static str, names: &[&str], values: &HashMap<String, Option<View>>,
         shape: Option<Vec<usize>>, optional: bool, device: u64) -> Result<Table, String> {
    if !optional { require(values, names)?; }
    Ok(Table { name, entries: names.iter().map(|n| values.get(*n).cloned().flatten()).collect(),
               shape, optional, device })
}

/// Select the same eight tables as the forward. The two flags are the current
/// aerosol-aware and hail-aware values; nonzero has the source's truth value.
/// Pointer addresses are bound separately from current allocations.
pub fn select(carriers: &Carriers, flags: [i32; 2]) -> Result<Vec<Table>, String> {
    let qv = carriers.fields.get("qv").and_then(Option::as_ref)
        .ok_or("TEMPO qv carrier missing")?;
    if qv.shape.len() != 2 { return Err("TEMPO qv rank differs from source".into()); }
    let nz = qv.shape[0];
    let cells = qv.shape[1];
    let device = qv.device;
    require(&carriers.fields, &layout::FIELD[..12])?;
    require(&carriers.fields, &layout::FIELD[17..])?;
    require(&carriers.model, &layout::MODEL[..8])?;
    require(&carriers.model, &layout::MODEL[13..])?;
    require(&carriers.tendencies, &layout::TEND[..9])?;
    require(&carriers.surface, &layout::SURFACE[..10])?;
    if flags[0] != 0 {
        require(&carriers.fields, &layout::FIELD[12..15])?;
        require(&carriers.model, &layout::MODEL[8..11])?;
        require(&carriers.tendencies, &layout::TEND[9..])?;
        require(&carriers.surface, &layout::SURFACE[10..])?;
    }
    if flags[1] != 0 {
        require(&carriers.fields, &layout::FIELD[15..17])?;
        require(&carriers.model, &layout::MODEL[11..13])?;
    }
    let layer_shape = Some(vec![nz, cells]);
    let surface_shape = Some(vec![cells]);
    Ok(vec![
        table("fields", layout::FIELD, &carriers.fields, layer_shape.clone(), true, device)?,
        table("model", layout::MODEL, &carriers.model, None, true, device)?,
        table("mesh", &["zz", "zgrid"], &carriers.mesh, None, false, device)?,
        table("diagnostics", layout::DIAG, &carriers.diagnostics, None, false, device)?,
        table("tendencies", layout::TEND, &carriers.tendencies, layer_shape.clone(), true, device)?,
        table("surface", layout::SURFACE, &carriers.surface, surface_shape.clone(), true, device)?,
        table("precip", layout::PRECIP, &carriers.precip, surface_shape, false, device)?,
        table("profiles", layout::PROFILE, &carriers.profiles, layer_shape, false, device)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Slice;
    fn view(id: u64, shape: &[usize]) -> Option<View> {
        let mut stride = 4i64;
        let mut strides = Vec::new();
        for &n in shape.iter().rev() { strides.push(stride); stride *= n as i64; }
        strides.reverse();
        Some(View {target: Some(Slice {allocation: id, offset: 0}), dtype: "<f4".into(),
                   shape: shape.to_vec(), strides, device: 0})
    }
    fn values(names: &[&str], shape: &[usize]) -> HashMap<String, Option<View>> {
        names.iter().enumerate().map(|(i,n)| ((*n).into(), view((i + 1) as u64, shape))).collect()
    }
    fn carriers() -> Carriers {
        Carriers { fields: values(layout::FIELD, &[3, 5]), model: values(layout::MODEL, &[3, 5]),
            mesh: values(&["zz", "zgrid"], &[3, 5]), diagnostics: values(layout::DIAG, &[3, 5]),
            tendencies: values(layout::TEND, &[3, 5]), surface: values(layout::SURFACE, &[5]),
            precip: values(layout::PRECIP, &[5]), profiles: values(layout::PROFILE, &[3, 5]) }
    }
    #[test]
    fn current_flags_choose_required_optional_fields() {
        let mut c = carriers();
        for name in ["nc", "nwfa", "nifa", "ng", "qb"] { c.fields.remove(name); }
        for name in ["nc", "nwfa", "nifa", "ng", "volg"] { c.model.remove(name); }
        for name in ["rncmpten", "rnifampten", "rnwfampten"] { c.tendencies.remove(name); }
        for name in ["nwfa2d", "nifa2d"] { c.surface.remove(name); }
        let selected = select(&c, [0, 0]).unwrap();
        assert_eq!(selected.iter().map(|v| v.name).collect::<Vec<_>>(),
                   ["fields", "model", "mesh", "diagnostics", "tendencies", "surface", "precip", "profiles"]);
        assert_eq!(selected[0].entries.len(), 20);
        assert_eq!(selected[0].shape.as_deref(), Some(&[3, 5][..]));
        assert!(selected[0].entries[12].is_none());
        assert!(select(&c, [1, 0]).is_err());
        assert!(select(&c, [0, -1]).is_err());
        assert!(select(&carriers(), [1, 1]).is_ok());
    }
    #[test]
    fn missing_required_or_bad_qv_rank_fails() {
        let mut c = carriers(); c.mesh.remove("zz");
        assert!(select(&c, [0, 0]).is_err());
        let mut c = carriers(); c.fields.insert("qv".into(), view(1, &[4]));
        assert!(select(&c, [0, 0]).is_err());
    }
}
