// SPDX-License-Identifier: Apache-2.0
//! Fixed-mesh inverse-incidence arrays that derivative specs bind as `derived`.
//! Each function restates, line for line, the recipe its spec names (the 11-02
//! lane sources listed beside each); a result is used only if its SHA256 equals
//! the value the spec pins, so a restatement that differs is refused, not used.
//! Built once per (kernel, recipe) from the live fixed topology; never per step.
use std::collections::BTreeSet;

/// ad1-g07 hex-ad/runtime/derived/dycore_csr.rs::transpose_fixed (rows -> sorted, de-duplicated columns).
pub fn transpose_fixed(rows: usize, columns: usize, width: usize, counts: &[i32], indices: &[i32]) -> Result<(Vec<i32>, Vec<i32>), String> {
    if counts.len() != rows || indices.len() != rows * width { return Err("topology shape mismatch".into()); }
    let mut inverse = vec![Vec::<i32>::new(); columns];
    for r in 0..rows {
        let count = counts[r];
        if count < 0 || count as usize > width { return Err("invalid row count".into()); }
        for &c in &indices[r * width..r * width + count as usize] {
            if c < 0 || c as usize >= columns { return Err("invalid column".into()); }
            inverse[c as usize].push(r as i32);
        }
    }
    let (mut offsets, mut entries) = (vec![0], Vec::new());
    for row in &mut inverse { row.sort_unstable(); row.dedup(); entries.extend_from_slice(row); offsets.push(entries.len() as i32); }
    Ok((offsets, entries))
}
/// ad1-g08 hex-ad/runtime/dycore/transpose.rs::build (owners then slots ascending; duplicates kept).
pub fn build(nfrom: usize, nto: usize, width: usize, counts: &[i32], indices: &[i32]) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), String> {
    if counts.len() != nfrom || indices.len() != nfrom * width { return Err("invalid topology dimensions".into()); }
    let mut lists = vec![Vec::<(i32, i32)>::new(); nto];
    for owner in 0..nfrom {
        let count = usize::try_from(counts[owner]).map_err(|_| "negative count")?;
        if count > width { return Err("count exceeds width".into()); }
        for slot in 0..count {
            let d = usize::try_from(indices[owner * width + slot]).map_err(|_| "negative index")?;
            if d >= nto { return Err("index outside destination".into()); }
            lists[d].push((owner as i32, slot as i32));
        }
    }
    let (mut offsets, mut owners, mut slots) = (vec![0], Vec::new(), Vec::new());
    for row in lists { for (o, s) in row { owners.push(o); slots.push(s); } offsets.push(owners.len() as i32); }
    Ok((offsets, owners, slots))
}
/// ad1-g17 hex-ad/tools/dycore_csr.rs (group owners or owner*width+slot by row).
pub fn csr_pack(values: &[i32], counts: Option<&[i32]>, nrows: usize, width: usize, owner_pack: bool, skip_negative: bool) -> Result<(Vec<i32>, Vec<i32>), String> {
    if values.len() % width != 0 { return Err("topology shape differs from width".into()); }
    let owners = values.len() / width;
    let full = vec![width as i32; owners];
    let counts = counts.unwrap_or(&full);
    if counts.len() != owners { return Err("counts shape differs from topology".into()); }
    let mut groups = vec![Vec::<i32>::new(); nrows];
    for owner in 0..owners {
        let count = usize::try_from(counts[owner]).map_err(|_| "negative count")?;
        if count > width { return Err("count exceeds width".into()); }
        for slot in 0..count {
            let v = values[owner * width + slot];
            if v < 0 && skip_negative { continue; }
            let row = usize::try_from(v).map_err(|_| "invalid topology index")?;
            if row >= nrows { return Err("invalid topology index".into()); }
            groups[row].push(if owner_pack { owner as i32 } else { (owner * width + slot) as i32 });
        }
    }
    let (mut offsets, mut entries) = (vec![0], Vec::new());
    for g in groups { entries.extend(g); offsets.push(entries.len() as i32); }
    Ok((offsets, entries))
}
/// ad1-g01 hex-ad/rust/acoustic/transpose.rs::edge_to_slots (edge -> cell*max_edges+slot).
pub fn edge_to_slots(ncells: usize, nedges: usize, max_edges: usize, counts: &[i32], edges: &[i32]) -> Result<(Vec<i32>, Vec<i32>), String> {
    if counts.len() != ncells || edges.len() != ncells * max_edges { return Err("connectivity shape".into()); }
    let mut rows = vec![Vec::<i32>::new(); nedges];
    for cell in 0..ncells {
        let count = usize::try_from(counts[cell]).map_err(|_| "negative count")?;
        if count > max_edges { return Err("edge count exceeds width".into()); }
        for slot in 0..count {
            let pos = cell * max_edges + slot;
            let e = edges[pos];
            if e < 0 || e as usize >= nedges { return Err("edge index".into()); }
            rows[e as usize].push(pos as i32);
        }
    }
    let (mut offsets, mut slots) = (vec![0], Vec::new());
    for row in rows { slots.extend(row); offsets.push(slots.len() as i32); }
    Ok((offsets, slots))
}
fn csr(rows: Vec<BTreeSet<usize>>) -> (Vec<i32>, Vec<i32>) {
    let (mut offsets, mut owners) = (vec![0], Vec::new());
    for row in rows { for v in row { owners.push(v as i32); } offsets.push(owners.len() as i32); }
    (offsets, owners)
}
/// hunt dyn-entry-gap: csr() of rows kept as plain vectors: each row sorted ascending and de-duplicated, which is what
/// iterating a BTreeSet gives. One BTreeSet per edge (122,880 at 120 km) made each of these tables 20 to 40 ms of host
/// time on its first launch.
fn csr_vec(rows: Vec<Vec<u32>>) -> (Vec<i32>, Vec<i32>) {
    let (mut offsets, mut owners) = (Vec::with_capacity(rows.len() + 1), Vec::with_capacity(rows.iter().map(Vec::len).sum()));
    offsets.push(0);
    for mut row in rows { row.sort_unstable(); row.dedup(); owners.extend(row.iter().map(|&v| v as i32)); offsets.push(owners.len() as i32); }
    (offsets, owners)
}
fn index(x: i32, limit: usize) -> Result<usize, String> {
    let u = usize::try_from(x).map_err(|_| "negative index")?;
    if u >= limit { Err("index out of range".into()) } else { Ok(u) }
}
/// ad1-g11 hex-ad/cuda/dycore/inverse_topology.rs::theta -> (edge owners, theta-cell owners).
pub fn theta(nc: usize, ne: usize, width: usize, counts: &[i32], edges: &[i32], cells: &[i32]) -> Result<((Vec<i32>, Vec<i32>), (Vec<i32>, Vec<i32>)), String> {
    if counts.len() != nc || edges.len() != nc * width || cells.len() != ne * 2 { return Err("bad incidence lengths".into()); }
    let mut edge_rows: Vec<Vec<u32>> = vec![Vec::new(); ne];
    for c in 0..nc {
        let count = usize::try_from(counts[c]).map_err(|_| "negative count")?;
        if count > width { return Err("count exceeds width".into()); }
        for j in 0..count { edge_rows[index(edges[c * width + j], ne)?].push(c as u32); }
    }
    // rows are pushed with c ascending, so a row is already sorted; dedup makes it the set
    for r in edge_rows.iter_mut() { r.dedup(); }
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for e in 0..ne { for side in 0..2 { let c = index(cells[2 * e + side], nc)?; rows[c].extend_from_slice(&edge_rows[e]); } }
    Ok((csr_vec(edge_rows), csr_vec(rows)))
}
/// multirank-2: theta's first output alone (edge owners: per edge, the cells whose edges_on_cell rows list it), the
/// form the theta/w finish rows declare with five inputs (no cells_on_edge). Same loop as theta().
pub fn theta_edges(nc: usize, ne: usize, width: usize, counts: &[i32], edges: &[i32]) -> Result<(Vec<i32>, Vec<i32>), String> {
    if counts.len() != nc || edges.len() != nc * width { return Err("bad incidence lengths".into()); }
    let mut edge_rows: Vec<Vec<u32>> = vec![Vec::new(); ne];
    for c in 0..nc {
        let count = usize::try_from(counts[c]).map_err(|_| "negative count")?;
        if count > width { return Err("count exceeds width".into()); }
        for j in 0..count { edge_rows[index(edges[c * width + j], ne)?].push(c as u32); }
    }
    Ok(csr_vec(edge_rows))
}
/// ad1-g11 hex-ad/cuda/dycore/inverse_topology.rs::momentum -> (edge owners, cell owners).
pub fn momentum(nc: usize, ne: usize, width: usize, counts: &[i32], neighbors: &[i32], cells: &[i32]) -> Result<((Vec<i32>, Vec<i32>), (Vec<i32>, Vec<i32>)), String> {
    if counts.len() != ne || neighbors.len() != ne * width || cells.len() != ne * 2 { return Err("bad incidence lengths".into()); }
    let mut edge_rows: Vec<Vec<u32>> = (0..ne).map(|e| vec![e as u32]).collect();
    let mut cell_rows: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for e in 0..ne {
        let count = usize::try_from(counts[e]).map_err(|_| "negative count")?;
        if count > width { return Err("count exceeds width".into()); }
        for j in 0..count { edge_rows[index(neighbors[e * width + j], ne)?].push(e as u32); }
        for side in 0..2 { cell_rows[index(cells[2 * e + side], nc)?].push(e as u32); }
    }
    Ok((csr_vec(edge_rows), csr_vec(cell_rows)))
}
/// G6: ad1-g25 hex-ad/recipes/dycore/derive_edge_incidence.py::derive (+ main's packing), restated line for line.
/// From [nedges,2] endpoints: per entity, the incident edges ascending (an edge whose two endpoints are the same entity
/// counts once), padded with -1 to the widest row; returns (edges [entity_count*width], counts, width).
pub fn edge_incidence(endpoints: &[i32], entity_count: usize) -> Result<(Vec<i32>, Vec<i32>, i32), String> {
    if endpoints.len() % 2 != 0 || entity_count == 0 { return Err("expected nonempty [nedges,2] endpoints".into()); }
    let mut rows = vec![Vec::<i32>::new(); entity_count];
    for (edge, ab) in endpoints.chunks_exact(2).enumerate() {
        let (a, b) = (ab[0], ab[1]);
        if a.min(b) < 0 || a.max(b) as usize >= entity_count { return Err(format!("edge {edge} endpoint outside mesh")); }
        let mut both = BTreeSet::new(); both.insert(a); both.insert(b);
        for entity in both { rows[entity as usize].push(edge as i32); }
    }
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if width == 0 { return Err("empty incidence".into()); }
    for (entity, row) in rows.iter().enumerate() {
        if row.windows(2).any(|w| w[0] >= w[1]) { return Err(format!("nonunique or unordered row {entity}")); }
        for &edge in row {
            let e = edge as usize;
            if endpoints[2 * e] != entity as i32 && endpoints[2 * e + 1] != entity as i32 { return Err("inverse incidence is inconsistent".into()); }
        }
    }
    let mut edges = Vec::with_capacity(entity_count * width);
    for row in &rows { edges.extend_from_slice(row); edges.extend(std::iter::repeat(-1).take(width - row.len())); }
    let counts = rows.iter().map(|r| r.len() as i32).collect();
    Ok((edges, counts, width as i32))
}
/// G6: U2 tree/ad/u2_transport/harness/src/main.rs `derived`, recipe "inverse_adv_cells", restated line for line: for
/// each cell, the stencil entries e*width+s (s < n_adv[e]) with adv_cells[e*width+s] == cell, ascending; (offsets, entries).
pub fn inverse_adv_cells(ncells: usize, nedges: usize, width: usize, n_adv: &[i32], adv_cells: &[i32]) -> Result<(Vec<i32>, Vec<i32>), String> {
    if n_adv.len() < nedges || adv_cells.len() < nedges * width { return Err("inverse_adv_cells: topology shape".into()); }
    let mut lists = vec![Vec::<i32>::new(); ncells];
    for e in 0..nedges {
        let n = usize::try_from(n_adv[e]).map_err(|_| "inverse_adv_cells: negative count")?;
        if n > width { return Err("inverse_adv_cells: count exceeds width".into()); }
        for s in 0..n {
            let a = e * width + s;
            lists[index(adv_cells[a], ncells)?].push(a as i32);
        }
    }
    for l in lists.iter_mut() { l.sort_unstable(); }
    let (mut off, mut ent) = (vec![0i32], Vec::new());
    for l in lists { ent.extend(l); off.push(ent.len() as i32); }
    Ok((off, ent))
}
pub fn pack(v: &[i32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

/// Physics-on window L3: ad2-hrrrv5-glue physics-ad/cuda/hrrrv5/glue.interface_shapes.rs::checked_interface_shapes
/// (sha256 d98b4515...), restated line for line. Checked host metadata for the pressure-interface reverse launch
/// (prep_interfaces_v841_f32 ad_passthrough): shapes and strides come from the bound forward views, never from
/// values. Returns (mass capacities, mass levels 0..nlev, interior levels 1..nlev).
pub fn interface_shapes(nlev: i32, ncells: i32, launched_threads: u64, mass: [([usize; 2], [usize; 2]); 4],
                        weights: [([usize; 2], [usize; 2]); 2]) -> Result<([i32; 4], Vec<i32>, Vec<i32>), String> {
    if nlev < 2 || ncells <= 0 || launched_threads < ncells as u64 {
        return Err("invalid interface dimensions or incomplete forward grid".into());
    }
    let nl = nlev as usize;
    let nc = ncells as usize;
    let row_bytes = nc.checked_mul(4).ok_or("stride overflow")?;
    let mut caps = [0; 4];
    for (j, (shape, strides)) in mass.iter().enumerate() {
        if shape[0] < nl || shape[1] != nc || *strides != [row_bytes, 4] { return Err("invalid mass-field view".into()); }
        let len = shape[0].checked_mul(nc).ok_or("mass size overflow")?;
        if len > i32::MAX as usize { return Err("mass index overflow".into()); }
        caps[j] = i32::try_from(shape[0]).map_err(|_| "mass capacity overflow")?;
    }
    for (shape, strides) in weights {
        if shape != [nl + 1, nc] || strides != [row_bytes, 4] { return Err("invalid interface-weight view".into()); }
        if shape[0].checked_mul(nc).ok_or("weight size overflow")? > i32::MAX as usize { return Err("weight index overflow".into()); }
    }
    Ok((caps, (0..nlev).collect(), (1..nlev).collect()))
}
