// SPDX-License-Identifier: Apache-2.0
//! multirank-2 (`ad-binder window --local-topology 1`): the topology "derived" bindings of the dycore derivative rows,
//! built from the launch's OWN mesh instead of the SHA256-pinned blobs of the captured global x1.2562 mesh.
//!
//! THE BREAKAGE THIS PREVENTS (F057 "Seen, not fixed"): in a decomposed plan each rank's launches run on a rank-local
//! mesh (owned + halo cells, local indices), but 28 dycore rows bind mesh-topology arrays keyed by the SHA256 of the
//! captured global mesh. 25 of them silently took the global array from the derived store onto a rank's local mesh
//! (a wrong adjoint, the dot test fails), and the rows whose arrays are constructed at run time (pv_edge_base_f32,
//! tangential_velocity_f32, vector_momentum_v841_f32) failed the SHA check at construct and stopped the window.
//!
//! Two parts, both pure host code (the device side is deriv.rs `prepare_with`):
//! * [`Registry`]: one rank's mesh, found from the plan's own launches. Mesh arguments (by argument name) that occur
//!   together in one launch belong to one rank's mesh; groups are the union of those co-occurrences. Each array keeps
//!   the (allocation, offset) the launch passed and the row width the launch gave it (its width scalar, or 2 / 1);
//!   each group keeps the launch scalars ncells / nedges / nvertices of its launches. A recipe that needs a mesh array
//!   the launch does not pass (momentum_filter_lap4's same-mesh edges_on_cell, ...) finds it in the launch's group.
//! * [`build`]: the restated recipes whose inputs are mesh arrays rather than this launch's arguments (the original
//!   sources: tree/ad/kernels/*/hex-ad/tools/dycore/{transpose,wind_inverse_csr}.rs, the derive_* / validate_* scripts
//!   the specs name, tree/ad/u2_transport/harness/src/main.rs `derived`), each applied to the global mesh in the tests
//!   and checked against its pinned SHA256.
//! Admission is each recipe's own consistency checks (indices in range, rows consistent): refused by name, never guessed.
//!
//! Rank-local index law (tree/src/hexcore/partition_local_mesh_v841.py, measured on the 2-rank x1.2562 partition
//! <kernel set>): a reference to an entity outside the rank's local set is CLAMPED to
//! local index 0 (rank 0: 172 cells_on_edge endpoints and 86 edges_on_vertex slots of the outer halo; rank 1 the
//! same), and a mesh that carries MPAS's one-past garbage element (a LAM) keeps it as local index n. Both are the
//! values the forward reads, so the inverse maps keep clamped references (the adjoint of the forward as run); the
//! garbage element n has no row of its own and is skipped, as S01b e72756f5 does for the LAM. A slot whose
//! incidence does not hold is refused unless it names 0 (the clamp) or n (garbage); anything outside [0, n] is
//! corrupt and refused.
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A plan device argument: (allocation id, byte offset).
pub type Key = (u64, usize);

/// The mesh dimensions a group carries (launch scalars of the same name).
pub const DIMS: [&str; 3] = ["ncells", "nedges", "nvertices"];

/// Row width of a mesh array: fixed, or the first of these launch scalars the launch passes.
#[derive(Clone, Copy, Debug)]
pub enum Width { Fixed(usize), Scalar(&'static [&'static str]) }

/// The integer mesh-topology arrays, by kernel argument name: (row dimension, row width).
pub fn mesh_array(name: &str) -> Option<(&'static str, Width)> {
    const ME: &[&str] = &["max_edges", "maxedges"];
    Some(match name {
        "cells_on_edge" | "vertices_on_edge" => ("nedges", Width::Fixed(2)),
        "n_edges_on_edge" | "n_adv" => ("nedges", Width::Fixed(1)),
        "edges_on_edge" => ("nedges", Width::Scalar(&["max_edges2"])),
        "adv_cells" => ("nedges", Width::Scalar(&["width"])),
        "n_edges_on_cell" => ("ncells", Width::Fixed(1)),
        "edges_on_cell" | "cells_on_cell" | "vertices_on_cell" => ("ncells", Width::Scalar(ME)),
        "edges_on_vertex" | "cells_on_vertex" => ("nvertices", Width::Scalar(&["vertex_degree"])),
        _ => return None,
    })
}

/// Recipes whose outputs are mesh topology (the 28 rows' derived bindings, derived-topology-28.json). Every other
/// derived binding (physics tables, interface shapes) keeps today's path in both modes.
pub fn topology_recipe(r: &str) -> bool {
    ["tools/dycore_csr.rs", "transpose.rs#f32-count", "transpose.rs#pair", "transpose.rs#build", "transpose_fixed",
     "recovery_topology.rs", "edge_to_slots", "u2:inverse_cells_on_edge", "u2:inverse_cells_on_cell", "inverse_adv_cells",
     "derive_reverse_topology.py", "derive_launch_scalar.py", "del4_edge_count", "pressure_topology.rs",
     "derive_edge_incidence.py", "wind_inverse_csr.rs", "validate_inverse_mesh.py",
     "inverse_topology.rs::theta", "inverse_topology.rs::momentum",
     // storm3km: the limited-area rows' gather tables (deriv.rs restate, from the launch's own mesh arguments)
     "acreg:build_kset_acreg.py", "s01b:make_lam_derived.py"].iter().any(|k| r.contains(k))
}

/// Which variant of a mesh array a recipe wants when the launch does not pass it itself. The plan can carry more than
/// one array of a name per rank: x1.2562 pkg18m2 has edges_on_cell at width 10 (the MPAS edgesOnCell [nCells,
/// maxEdges], allocation 27) and a width-6 compaction (allocations 74 and 83, the recovery / reconstruct rows').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick { Any, Wide, Compact }

/// Mesh arrays a mesh-input recipe needs, for output `name`: (array, pick, required). None: the recipe binds only this
/// launch's own arguments (dycore_csr, edge_to_slots, inverse_topology, transpose_fixed/#build, derive_edge_incidence,
/// inverse_adv_cells) and is built by deriv.rs `restate` from them.
pub fn needs(recipe: &str, name: &str) -> Option<Vec<(&'static str, Pick, bool)>> {
    use Pick::*;
    Some(if recipe.contains("transpose.rs#f32-count") || recipe.contains("del4_edge_count") { vec![] }
    else if recipe.contains("transpose.rs#pair") || recipe.contains("pressure_topology.rs") || recipe.contains("u2:inverse_cells_on_edge") {
        vec![("cells_on_edge", Any, true)]
    } else if recipe.contains("recovery_topology.rs") {
        vec![("cells_on_edge", Any, true), ("edges_on_cell", Wide, true), ("n_edges_on_cell", Any, true)]
    } else if recipe.contains("derive_reverse_topology.py") {
        vec![("edges_on_cell", Wide, true), ("n_edges_on_cell", Any, true), ("cells_on_edge", Any, true),
             ("vertices_on_edge", Any, true), ("edges_on_vertex", Any, false)]
    } else if recipe.contains("derive_launch_scalar.py") {
        match name {
            // multirank-2: n_edges_on_cell and cells_on_edge (optional) give the width of the transposed rows
            "max_edges" => vec![("edges_on_cell", Wide, true), ("n_edges_on_cell", Any, false), ("cells_on_edge", Any, false)],
            "vertex_degree" => vec![("vertices_on_edge", Any, true), ("edges_on_vertex", Any, false)],
            _ => vec![],
        }
    } else if recipe.contains("validate_inverse_mesh.py") {
        vec![("edges_on_cell", Wide, true), ("n_edges_on_cell", Any, true), ("cells_on_edge", Any, true)]
    } else if recipe.contains("wind_inverse_csr.rs") {
        vec![("edges_on_cell", Any, true), ("n_edges_on_cell", Any, true)]
    } else if recipe.contains("u2:inverse_cells_on_cell") {
        vec![("n_edges_on_cell", Any, true), ("cells_on_cell", Any, true)]
    } else { return None })
}

/// One launch's mesh: what [`Registry::observe`] records.
#[derive(Clone, Debug, Default)]
pub struct Observation { pub function: String, pub arrays: Vec<(String, Key, Option<usize>)>, pub dims: Vec<(String, usize)> }

#[derive(Clone, Debug, Default)]
pub struct Variant { pub key: Key, pub widths: BTreeSet<usize>, pub functions: BTreeSet<String> }

#[derive(Clone, Debug, Default)]
pub struct Group { pub arrays: BTreeMap<String, Vec<Variant>>, pub dims: BTreeMap<String, BTreeSet<(usize, String)>> }

/// The per-rank mesh registry (see the module comment).
#[derive(Debug, Default)]
pub struct Registry { obs: Vec<Observation>, pub groups: Vec<Group>, group_of: HashMap<Key, usize> }

fn root(p: &mut [usize], mut i: usize) -> usize { while p[i] != i { p[i] = p[p[i]]; i = p[i]; } i }

impl Registry {
    pub fn observe(&mut self, o: Observation) { if !o.arrays.is_empty() { self.obs.push(o); } }
    pub fn observations(&self) -> usize { self.obs.len() }
    /// Union the co-occurrence groups (the order of observation does not matter) and collect each group's arrays and
    /// dims. Groups are numbered by their smallest (allocation, offset).
    pub fn finish(&mut self) {
        let mut keys: Vec<Key> = self.obs.iter().flat_map(|o| o.arrays.iter().map(|a| a.1)).collect();
        keys.sort(); keys.dedup();
        let ids: HashMap<Key, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let mut p: Vec<usize> = (0..keys.len()).collect();
        for o in &self.obs {
            let first = ids[&o.arrays[0].1];
            for (_, k, _) in &o.arrays[1..] {
                let (a, b) = (root(&mut p, first), root(&mut p, ids[k]));
                if a != b { p[a.max(b)] = a.min(b); }
            }
        }
        let mut roots: BTreeMap<usize, usize> = BTreeMap::new();
        self.groups.clear(); self.group_of.clear();
        for (i, k) in keys.iter().enumerate() {
            let r = root(&mut p, i);
            let n = roots.len();
            let g = *roots.entry(r).or_insert(n);
            if g == self.groups.len() { self.groups.push(Group::default()); }
            self.group_of.insert(*k, g);
        }
        for o in &self.obs {
            let grp = &mut self.groups[self.group_of[&o.arrays[0].1]];
            for (name, k, w) in &o.arrays {
                let vs = grp.arrays.entry(name.clone()).or_default();
                let i = match vs.iter().position(|v| v.key == *k) { Some(i) => i, None => { vs.push(Variant { key: *k, ..Default::default() }); vs.len() - 1 } };
                if let Some(w) = w { vs[i].widths.insert(*w); }
                vs[i].functions.insert(o.function.clone());
            }
            for (d, v) in &o.dims { grp.dims.entry(d.clone()).or_default().insert((*v, o.function.clone())); }
        }
        for g in &mut self.groups { for vs in g.arrays.values_mut() { vs.sort_by_key(|v| v.key); } }
    }
    /// The group of a launch, through any of its mesh arguments (all of them lie in one group by construction).
    pub fn group(&self, keys: &[Key]) -> Result<usize, String> {
        let gs: BTreeSet<usize> = keys.iter().filter_map(|k| self.group_of.get(k).copied()).collect();
        match gs.len() {
            1 => Ok(*gs.iter().next().unwrap()),
            0 => Err(format!("none of the launch's mesh arguments {keys:?} is in the plan's mesh registry")),
            _ => Err(format!("the launch's mesh arguments lie in mesh groups {gs:?}")),
        }
    }
    /// A group dimension (ncells / nedges / nvertices): the launch scalar of that name, one value across the group.
    pub fn dim(&self, g: usize, name: &str) -> Result<usize, String> {
        let vals: BTreeSet<usize> = self.groups[g].dims.get(name).map(|s| s.iter().map(|x| x.0).collect()).unwrap_or_default();
        match vals.len() {
            1 => Ok(*vals.iter().next().unwrap()),
            0 => Err(format!("mesh group {g} has no launch passing {name} (needed for its rank-local extent)")),
            _ => Err(format!("mesh group {g}: launches pass {name} = {vals:?} (two meshes in one group: {})",
                             self.groups[g].dims[name].iter().map(|(v, f)| format!("{f}={v}")).collect::<Vec<_>>().join(", "))),
        }
    }
    /// Width of one variant (one value across the launches that passed it).
    pub fn width(&self, g: usize, name: &str, key: Key) -> Result<usize, String> {
        let v = self.groups[g].arrays.get(name).and_then(|vs| vs.iter().find(|v| v.key == key))
            .ok_or_else(|| format!("{name} at allocation {} offset {} is not in mesh group {g}", key.0, key.1))?;
        match v.widths.len() {
            1 => Ok(*v.widths.iter().next().unwrap()),
            0 => Err(format!("{name} at allocation {} offset {}: no launch gives its row width ({:?})", key.0, key.1, v.functions)),
            _ => Err(format!("{name} at allocation {} offset {}: launches give row widths {:?} ({:?})", key.0, key.1, v.widths, v.functions)),
        }
    }
    /// The candidates of `name` in group `g` for `pick`, all of one width: one, or several the caller must find equal
    /// word for word (copies, such as pkg18m2's allocations 74 and 83), else refuse. Empty when the group has none.
    pub fn pick(&self, g: usize, name: &str, pick: Pick) -> Result<Vec<(Key, usize)>, String> {
        let Some(vs) = self.groups[g].arrays.get(name) else { return Ok(Vec::new()) };
        let mut c: Vec<(Key, usize)> = Vec::new();
        for v in vs { c.push((v.key, self.width(g, name, v.key)?)); }
        let ws: BTreeSet<usize> = c.iter().map(|x| x.1).collect();
        let w = match pick {
            Pick::Wide => *ws.iter().next_back().unwrap(),
            Pick::Compact => *ws.iter().next().unwrap(),
            Pick::Any if ws.len() == 1 => *ws.iter().next().unwrap(),
            Pick::Any => return Err(format!("mesh group {g} has {name} at widths {ws:?} and the recipe names no width")),
        };
        Ok(c.into_iter().filter(|x| x.1 == w).collect())
    }
}

/// The resolved host inputs of one local build: logical extents only (rows x width words, never the allocation padding).
#[derive(Clone, Debug, Default)]
pub struct Mesh { pub dims: BTreeMap<String, usize>, pub arrays: BTreeMap<String, (Vec<i32>, usize)> }

impl Mesh {
    fn d(&self, n: &str) -> Result<usize, String> { self.dims.get(n).copied().ok_or_else(|| format!("mesh dimension {n} not resolved")) }
    fn a(&self, n: &str) -> Result<(&[i32], usize), String> {
        self.arrays.get(n).map(|(v, w)| (v.as_slice(), *w)).ok_or_else(|| format!("mesh array {n} not resolved"))
    }
}

fn cell(x: i32, nc: usize, at: usize) -> Result<usize, String> {
    usize::try_from(x).ok().filter(|&u| u < nc).ok_or_else(|| format!("row {at}: index {x} out of range [0, {nc})"))
}
/// A reference into `n` local entities: Some(index), None for the one-past garbage element n, refused outside [0, n].
/// hunt dyn-entry-gap: `what` is a closure, so the refusal text is formatted only when a refusal happens (it was
/// formatted for every word of every mesh array: about 20 ms of host time per mesh table at 120 km).
fn local(x: i32, n: usize, what: impl FnOnce() -> String) -> Result<Option<usize>, String> {
    match usize::try_from(x) {
        Ok(u) if u < n => Ok(Some(u)),
        Ok(u) if u == n => Ok(None),
        _ => Err(format!("{}: index {x} outside [0, {n}] (n = the one-past garbage element)", what())),
    }
}
fn count(x: i32, w: usize, at: usize) -> Result<usize, String> {
    usize::try_from(x).ok().filter(|&u| u <= w).ok_or_else(|| format!("row {at}: count {x} outside [0, {w}]"))
}
fn lists(mut l: Vec<Vec<(i32, i32)>>) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    for x in l.iter_mut() { x.sort(); }
    let mut off = vec![0i32];
    for x in &l { off.push(off.last().unwrap() + x.len() as i32); }
    (off, l.iter().flatten().map(|x| x.0).collect(), l.iter().flatten().map(|x| x.1).collect())
}

/// transpose.rs `pair <owners> <pairs> edge` (pressure_topology.rs gives the same arrays): per owner, the pair rows
/// (slot / 2) that name it, in order.
pub fn pair(owners: usize, pairs: &[i32]) -> Result<(Vec<i32>, Vec<i32>), String> {
    if pairs.len() % 2 != 0 { return Err("pair: odd pair array".into()); }
    let mut rows = vec![Vec::<i32>::new(); owners];
    for (slot, &o) in pairs.iter().enumerate() {
        let Some(i) = local(o, owners, || format!("pair: owner at word {slot}"))? else { continue };
        rows[i].push((slot / 2) as i32);
    }
    let (mut o, mut x) = (vec![0i32], Vec::new());
    for r in rows { x.extend(r); o.push(x.len() as i32); }
    Ok((o, x))
}
/// u2 harness `inverse_cells_on_edge`: per cell, (edge, side) ascending; an edge whose two cells are equal counts once.
pub fn inverse_cells_on_edge(nc: usize, coe: &[i32]) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), String> {
    let mut l = vec![Vec::<(i32, i32)>::new(); nc];
    for (e, ab) in coe.chunks_exact(2).enumerate() {
        let (a, b) = (local(ab[0], nc, || format!("cells_on_edge edge {e}"))?, local(ab[1], nc, || format!("cells_on_edge edge {e}"))?);
        if let Some(a) = a { l[a].push((e as i32, 0)); }
        if let Some(b) = b.filter(|b| Some(*b) != a) { l[b].push((e as i32, 1)); }
    }
    Ok(lists(l))
}
/// u2 harness `inverse_cells_on_cell`: per cell, (neighbour cell, slot) ascending over the active slots.
pub fn inverse_cells_on_cell(nc: usize, me: usize, neoc: &[i32], coc: &[i32]) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), String> {
    if neoc.len() != nc || coc.len() != nc * me { return Err("inverse_cells_on_cell: topology shape".into()); }
    let mut l = vec![Vec::<(i32, i32)>::new(); nc];
    for c in 0..nc {
        let n = count(neoc[c], me, c)?;
        for s in 0..n { if let Some(x) = local(coc[c * me + s], nc, || format!("cells_on_cell cell {c} slot {s}"))? { l[x].push((c as i32, s as i32)); } }
    }
    Ok(lists(l))
}
/// wind_inverse_csr.rs: per edge, the (cell, slot) pairs that list it, cells then slots ascending.
pub fn wind_inverse_csr(nc: usize, ne: usize, nm: usize, edges: &[i32], counts: &[i32]) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), String> {
    if edges.len() != nc * nm || counts.len() != nc { return Err("wind_inverse_csr: unexpected input word count".into()); }
    let mut rows: Vec<Vec<(i32, i32)>> = vec![Vec::new(); ne];
    for c in 0..nc {
        let n = count(counts[c], nm, c).map_err(|e| format!("wind_inverse_csr: cell count exceeds maxedges: {e}"))?;
        for s in 0..n {
            let Some(e) = local(edges[c * nm + s], ne, || format!("wind_inverse_csr: cell {c} slot {s}: active edge"))? else { continue };
            rows[e].push((c as i32, s as i32));
        }
    }
    let (mut o, mut cs, mut ss) = (vec![0i32], Vec::new(), Vec::new());
    for r in rows { for (c, s) in r { cs.push(c); ss.push(s); } o.push(cs.len() as i32); }
    Ok((o, cs, ss))
}
/// Consistency of edges_on_cell / n_edges_on_cell against cells_on_edge (recovery_topology, derive_reverse_topology,
/// validate_inverse_mesh): every endpoint is a local cell, every active slot names a local edge that has the cell.
pub fn check_cell_edges(eoc: &[i32], w: usize, neoc: &[i32], coe: &[i32], nc: usize, ne: usize) -> Result<(), String> {
    if eoc.len() != nc * w || neoc.len() != nc || coe.len() != 2 * ne { return Err("cell-edge incidence: extents".into()); }
    for (e, ab) in coe.chunks_exact(2).enumerate() { for x in ab { local(*x, nc, || format!("cells_on_edge edge {e}"))?; } }
    for c in 0..nc {
        let n = count(neoc[c], w, c).map_err(|x| format!("n_edges_on_cell {x}"))?;
        for s in 0..n {
            let e = eoc[c * w + s];
            let Some(u) = local(e, ne, || format!("edges_on_cell cell {c} slot {s}"))? else { continue };
            if u != 0 && coe[2 * u] != c as i32 && coe[2 * u + 1] != c as i32 {
                return Err(format!("edges_on_cell cell {c} slot {s}: edge {e} does not have cell {c} in cells_on_edge"));
            }
        }
    }
    Ok(())
}
/// multirank-2: an inverse incidence (edges_on_cell from cells_on_edge, edges_on_vertex from vertices_on_edge) as the
/// exact transpose of what the forward reads: each row keeps, in its own order, the entries whose edge names the entity,
/// and gains (ascending) every edge that names it but is not listed. A row that changes is padded with -1; a row that
/// does not keeps its bytes. Returns (rows, width, counts). On a whole mesh every row is already its transpose, so the
/// bytes, the width and the pinned sha256 are unchanged. On a rank-local mesh an out-of-rank reference is clamped to
/// local 0: an outer-halo edge whose second cell is clamped names cell 0 (the forward reads cell 0 there, so its
/// cotangent belongs to cell 0), and an outer vertex's out-of-rank edge slot is clamped to edge 0, which does not name
/// that vertex. THE BREAKAGE THIS PREVENTS: the 2-rank window's reverse stopped at step 2:
/// momentum_filter_lap4_v841_f32 adjoint refused (refusal-11, a vertex's edges_on_vertex edge without that vertex).
pub fn transpose_keep(rows: &[i32], w: usize, counts: Option<&[i32]>, pairs: &[i32], n: usize) -> (Vec<i32>, usize, Vec<i32>) {
    let ne = pairs.len() / 2;
    let mut by: Vec<Vec<i32>> = vec![Vec::new(); n];
    for e in 0..ne {
        for side in 0..2 {
            let x = pairs[2 * e + side];
            if x >= 0 && (x as usize) < n && (side == 0 || x != pairs[2 * e]) { by[x as usize].push(e as i32); }
        }
    }
    let mut out_rows: Vec<Option<Vec<i32>>> = Vec::with_capacity(n);
    let (mut width, mut cnt) = (w, Vec::with_capacity(n));
    for x in 0..n {
        let k = counts.map(|c| (c[x].max(0) as usize).min(w)).unwrap_or(w);
        let row = &rows[x * w..x * w + k];
        let names = |e: i32| e >= 0 && (e as usize) < ne && (pairs[2 * e as usize] == x as i32 || pairs[2 * e as usize + 1] == x as i32);
        let mut kept: Vec<i32> = Vec::new();
        for &e in row { if names(e) && !kept.contains(&e) { kept.push(e); } }
        for &e in &by[x] { if !kept.contains(&e) { kept.push(e); } }
        let same = counts.is_some() && kept.as_slice() == row || counts.is_none() && kept.len() == row.iter().filter(|&&e| e >= 0).count()
            && kept.iter().zip(row.iter().filter(|&&e| e >= 0)).all(|(a, b)| a == b);
        cnt.push(kept.len() as i32);
        width = width.max(kept.len());
        out_rows.push(if same { None } else { Some(kept) });
    }
    let mut out = Vec::with_capacity(n * width);
    for x in 0..n {
        match &out_rows[x] {
            None => { out.extend_from_slice(&rows[x * w..x * w + w]); out.extend(std::iter::repeat(-1).take(width - w)); },
            Some(k) => { out.extend_from_slice(k); out.extend(std::iter::repeat(-1).take(width - k.len())); },
        }
    }
    (out, width, cnt)
}

/// multirank-2: cells_on_edge as the reverse rows read it: an endpoint whose cell does not list the edge among its first
/// n_edges_on_cell slots is replaced by the other endpoint (the rows skip a second endpoint equal to the first). The
/// reverse sums an edge's cotangent from the cells that READ it through edges_on_cell, and on a rank-local mesh an
/// outer-halo edge's out-of-rank endpoint is clamped to local cell 0, which does not list that edge. On a whole mesh
/// every endpoint lists its edge, so the rows are unchanged (the pinned sha256 still holds). THE BREAKAGE THIS
/// PREVENTS: the 2-rank window's reverse stopped at step 2: recover_terrain_w_f32 adjoint refused (refusal-8,
/// recover_terrain_w.ad.cu:121, an edge's endpoint cell without that edge: slot < 0).
pub fn listed_endpoints(eoc: &[i32], w: usize, neoc: &[i32], coe: &[i32], nc: usize) -> Vec<i32> {
    let lists = |c: i32, e: usize| -> bool {
        if c < 0 || c as usize >= nc { return true; }   // garbage element or corrupt: left as recorded (the rows decide)
        let c = c as usize;
        let n = (neoc[c].max(0) as usize).min(w);
        eoc[c * w..c * w + n].iter().any(|&x| x == e as i32)
    };
    let mut out = coe.to_vec();
    for (e, ab) in out.chunks_exact_mut(2).enumerate() {
        let (a, b) = (ab[0], ab[1]);
        match (lists(a, e), lists(b, e)) {
            (true, false) => ab[1] = a,
            (false, true) => ab[0] = b,
            _ => {}
        }
    }
    out
}
/// Consistency of edges_on_vertex against vertices_on_edge: every slot names a local edge that has that vertex.
pub fn check_vertex_edges(eov: &[i32], vd: usize, voe: &[i32], nv: usize, ne: usize) -> Result<(), String> {
    if eov.len() != nv * vd || voe.len() != 2 * ne { return Err("vertex-edge incidence: extents".into()); }
    for (e, ab) in voe.chunks_exact(2).enumerate() { for x in ab { local(*x, nv, || format!("vertices_on_edge edge {e}"))?; } }
    for v in 0..nv {
        for s in 0..vd {
            let e = eov[v * vd + s];
            let Some(u) = local(e, ne, || format!("edges_on_vertex vertex {v} slot {s}"))? else { continue };
            if u != 0 && voe[2 * u] != v as i32 && voe[2 * u + 1] != v as i32 { return Err(format!("edges_on_vertex vertex {v} slot {s}: edge {e} does not have vertex {v}")); }
        }
    }
    Ok(())
}
fn vertex_rows(voe: &[i32], nv: usize) -> Result<(Vec<i32>, Vec<i32>), String> {
    crate::recipes::csr_pack(voe, None, nv, 2, true, false).map_err(|x| format!("vertices_on_edge inverse: {x}"))
}
fn max_incidence(voe: &[i32], nv: usize) -> Result<usize, String> {
    Ok(vertex_rows(voe, nv)?.0.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0))
}

/// Build output `name` (`component`) of mesh-input recipe `recipe` from the resolved local mesh.
pub fn build(recipe: &str, name: &str, component: &str, m: &Mesh) -> Result<Vec<i32>, String> {
    let offsets = name.contains("offsets") || name.ends_with("_off") || component == "offsets";
    if recipe.contains("transpose.rs#f32-count") { return Ok(vec![m.d("ncells")? as i32]); }
    if recipe.contains("del4_edge_count") { return Ok(vec![m.d("nedges")? as i32]); }
    if recipe.contains("transpose.rs#pair") || recipe.contains("pressure_topology.rs") {
        let (o, x) = pair(m.d("ncells")?, m.a("cells_on_edge")?.0)?;
        return Ok(if offsets { o } else { x });
    }
    if recipe.contains("u2:inverse_cells_on_edge") || recipe.contains("u2:inverse_cells_on_cell") {
        let (o, first, second) = if recipe.contains("cells_on_edge") { inverse_cells_on_edge(m.d("ncells")?, m.a("cells_on_edge")?.0)? }
            else { let (coc, w) = m.a("cells_on_cell")?; inverse_cells_on_cell(m.d("ncells")?, w, m.a("n_edges_on_cell")?.0, coc)? };
        return Ok(if offsets { o } else if name.contains("slot") { second } else { first });
    }
    if recipe.contains("wind_inverse_csr.rs") {
        let (eoc, w) = m.a("edges_on_cell")?;
        let (o, cs, ss) = wind_inverse_csr(m.d("ncells")?, m.d("nedges")?, w, eoc, m.a("n_edges_on_cell")?.0)?;
        return Ok(if offsets { o } else if name.contains("slot") { ss } else { cs });
    }
    let ne = m.d("nedges")?;
    if recipe.contains("recovery_topology.rs") || recipe.contains("derive_reverse_topology.py") || recipe.contains("validate_inverse_mesh.py") {
        let nc = m.d("ncells")?;
        let ((eoc, w), (neoc, _), (coe, _)) = (m.a("edges_on_cell")?, m.a("n_edges_on_cell")?, m.a("cells_on_edge")?);
        check_cell_edges(eoc, w, neoc, coe, nc, ne)?;
        // recovery_topology.rs: the mesh edges_on_cell trimmed to its widest row, and that width (the semantics
        // S01b e72756f5 make_lam_derived.py verified byte for byte on x1.2562 pkg18m2).
        if recipe.contains("recovery_topology.rs") && matches!(name, "edges_on_cell" | "max_edges") {
            let maxn = neoc.iter().copied().max().unwrap_or(0).max(0) as usize;
            return Ok(if name == "max_edges" { vec![maxn as i32] } else { eoc.chunks_exact(w).flat_map(|r| r[..maxn].to_vec()).collect() });
        }
        // multirank-2: derive_reverse_topology feeds the reverse gathers (momentum filters): the exact transpose.
        if recipe.contains("derive_reverse_topology.py") && matches!(name, "edges_on_cell" | "eoc" | "n_edges_on_cell" | "neoc" | "max_edges") {
            let (rows, width, counts) = transpose_keep(eoc, w, Some(neoc), coe, nc);
            return Ok(match name { "edges_on_cell" | "eoc" => rows, "max_edges" => vec![width as i32], _ => counts });
        }
        return Ok(match name {
            "edges_on_cell" | "eoc" => eoc.to_vec(),
            "n_edges_on_cell" | "neoc" => neoc.to_vec(),
            // multirank-2: recovery_topology rows (recover_terrain_w) skip a duplicate endpoint but refuse one that does not
            // list the edge (refusal-8); smagorinsky (derive_reverse_topology) refuses equal endpoints (refusal-4) and skips
            // a non-listing one, so only the recovery rows get the rewritten endpoints.
            "cells_on_edge" | "coe" if recipe.contains("recovery_topology.rs") => listed_endpoints(eoc, w, neoc, coe, nc),
            "cells_on_edge" | "coe" => coe.to_vec(),
            "max_edges" => vec![w as i32],
            "edges_on_vertex" | "eov" if recipe.contains("derive_reverse_topology.py") => {
                let nv = m.d("nvertices")?;
                let (voe, _) = m.a("vertices_on_edge")?;
                match m.arrays.get("edges_on_vertex") {
                    Some((eov, vd)) => transpose_keep(eov, *vd, None, voe, nv).0,
                    // No edges_on_vertex in the rank's plan: the inverse of vertices_on_edge, every vertex of one degree.
                    None => {
                        let (o, e) = vertex_rows(voe, nv)?;
                        let vd = max_incidence(voe, nv)?;
                        if let Some(v) = (0..nv).find(|&v| (o[v + 1] - o[v]) as usize != vd) {
                            return Err(format!("edges_on_vertex from vertices_on_edge: vertex {v} has {} edges, not the vertex degree {vd}", o[v + 1] - o[v]));
                        }
                        e
                    },
                }
            },
            other => return Err(format!("{recipe} has no output {other}")),
        });
    }
    if recipe.contains("derive_launch_scalar.py") {
        return Ok(vec![match name {
            // multirank-2: the widths of the transposed rows the same launch binds (derive_reverse_topology above)
            "max_edges" => {
                let (eoc, w) = m.a("edges_on_cell")?;
                match (m.arrays.get("n_edges_on_cell"), m.arrays.get("cells_on_edge"), m.d("ncells")) {
                    (Some((neoc, _)), Some((coe, _)), Ok(nc)) => transpose_keep(eoc, w, Some(neoc), coe, nc).1 as i32,
                    _ => w as i32,
                }
            },
            "nedges" => ne as i32,
            "vertex_degree" => {
                let (voe, nv) = (m.a("vertices_on_edge")?.0, m.d("nvertices")?);
                let mx = max_incidence(voe, nv)?;
                match m.arrays.get("edges_on_vertex") {
                    Some((eov, vd)) => transpose_keep(eov, *vd, None, voe, nv).1 as i32,
                    None => mx as i32,
                }
            },
            other => return Err(format!("{recipe} has no output {other}")),
        }]);
    }
    Err(format!("no local restatement for recipe {recipe}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn transpose_keep_follows_what_the_forward_reads() {
        // multirank-2: edges e0=(0,1) e1=(0,2) e2=(1,0 clamped); cell rows width 3 with counts.
        let eoc = vec![0, 1, -1, 0, 2, -1, 1, 2, -1];
        let neoc = vec![2, 2, 2];
        let coe = vec![0, 1, 0, 2, 1, 0];
        let (rows, w, c) = transpose_keep(&eoc, 3, Some(&neoc), &coe, 3);
        // cell 0 gains e2 (it names cell 0), cell 2 loses e2 (e2 no longer names cell 2)
        assert_eq!((rows, w, c), (vec![0, 1, 2, 0, 2, -1, 1, -1, -1], 3, vec![3, 2, 1]));
        // a consistent mesh: bytes unchanged
        let coe_ok = vec![0, 1, 0, 2, 1, 2];
        assert_eq!(transpose_keep(&eoc, 3, Some(&neoc), &coe_ok, 3), (eoc.clone(), 3, neoc.clone()));
        // a vertex slot clamped to edge 0 that does not name the vertex is dropped (rows without counts)
        let voe = vec![0, 1, 1, 2, 2, 0];
        let eov = vec![0, 2, 0, 1, 2, 0];   // vertex 2: slot 1 clamped to edge 0, which is (0,1)
        let (r, wv, _) = transpose_keep(&eov, 2, None, &voe, 3);
        assert_eq!((r, wv), (vec![0, 2, 0, 1, 2, 1], 2));
    }
    #[test]
    fn listed_endpoints_drop_a_clamped_out_of_rank_cell() {
        // multirank-2: 3 cells, edge 2's second endpoint is clamped to cell 0, which does not list edge 2.
        let eoc = vec![0, 1, -1, 0, 2, -1, 1, 2, -1];
        let neoc = vec![2, 2, 2];
        let coe = vec![0, 1, 0, 2, 1, 0];
        assert_eq!(listed_endpoints(&eoc, 3, &neoc, &coe, 3), vec![0, 1, 0, 2, 1, 1]);
        // a whole mesh (every endpoint lists its edge) is unchanged
        let coe_ok = vec![0, 1, 0, 2, 1, 2];
        assert_eq!(listed_endpoints(&eoc, 3, &neoc, &coe_ok, 3), coe_ok);
    }

    use super::*;
    use std::io::Read;

    /// The global x1.2562 mesh of the 1-rank pkg18m2 plan (its recorded initial allocation payloads, cut to the
    /// logical extents): tree/ad/e2e/fixtures/x1.2562-mesh/<name>.i32.gz (README there names each source payload).
    fn fixture(name: &str) -> Vec<i32> {
        let m = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = [m.join("fixtures/x1.2562-mesh"), m.join("../../../../../../e2e/fixtures/x1.2562-mesh")].into_iter()
            .find(|d| d.is_dir()).expect("tree/ad/e2e/fixtures/x1.2562-mesh");
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(dir.join(format!("{name}.i32.gz"))).unwrap()).read_to_end(&mut raw).unwrap();
        raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn sha(v: &[i32]) -> String { crate::exec::digest(&crate::recipes::pack(v)) }
    const NC: usize = 2562; const NE: usize = 7680; const NV: usize = 5120;
    fn global(eoc_wide: bool) -> Mesh {
        let mut m = Mesh::default();
        for (d, v) in [("ncells", NC), ("nedges", NE), ("nvertices", NV)] { m.dims.insert(d.into(), v); }
        let (e, w) = if eoc_wide { (fixture("edges_on_cell_w10"), 10) } else { (fixture("edges_on_cell_w6"), 6) };
        m.arrays.insert("edges_on_cell".into(), (e, w));
        for (n, w) in [("cells_on_edge", 2), ("n_edges_on_cell", 1), ("vertices_on_edge", 2), ("edges_on_vertex", 3), ("cells_on_cell", 10)] {
            m.arrays.insert(n.into(), (fixture(n), w));
        }
        m
    }
    // Pinned SHA256s of the 28 rows' derived bindings (kset-int8 specs; multirank2/gen/derived-topology-28.json).
    const COE: &str = "7c4b6c781f6001e70a03186e2b58639bf2e4455bd19184e39c0aa63db3f6df72";
    const EOC10: &str = "eedcaaed085bea9e56944986bd8375062b543cc4dcc4daffe415b530fc366dbc";
    const NEOC: &str = "98c7fd5fdab896bb9350099b1bbf2532813c23b3b3c5e8491a7d7b94c33d18b8";
    const EOV: &str = "3c63a50a5c13f7152cb74df6ada3a80481424976c617390aa6ed0cc1e9b704c8";
    const CSR_OFF: &str = "4836aa73664068bacb61d5ff4bef377b654f3e12acfab2e9beab107914cec3f7";
    const CSR_ENT: &str = "ccae6cd072bfb62e3e9659e5d7889ff4347cd2308b0ddaed78fbf37ffb905ba6";

    #[test]
    fn fixtures_are_the_pinned_global_mesh() {
        assert_eq!(sha(&fixture("cells_on_edge")), COE);
        assert_eq!(sha(&fixture("edges_on_cell_w10")), EOC10);
        assert_eq!(sha(&fixture("n_edges_on_cell")), NEOC);
        assert_eq!(sha(&fixture("edges_on_vertex")), EOV);
    }

    /// Every mesh-input recipe on the global x1.2562 mesh reproduces its pinned SHA256 (on one rank local == global).
    #[test]
    fn local_recipes_reproduce_the_pinned_global_arrays() {
        let wide = global(true);
        let compact = global(false);
        let cases: &[(&str, &str, &Mesh, &str)] = &[
            ("hex-ad/tools/dycore/transpose.rs#f32-count", "ncells", &wide, "3e2a9d17d8aea0946ccbc4086f4c0f29cfb1eb1ad83b2fc3228d3193c971bf3e"),
            ("hex-ad/tools/dycore/transpose.rs#pair", "cell_edge_offsets", &wide, CSR_OFF),
            ("hex-ad/tools/dycore/transpose.rs#pair", "cell_edges", &wide, CSR_ENT),
            ("hex-ad/derived/dycore/pressure_topology.rs", "cell_edge_offsets", &wide, CSR_OFF),
            ("hex-ad/derived/dycore/pressure_topology.rs", "cell_edges", &wide, CSR_ENT),
            ("hex-ad/tools/dycore/recovery_topology.rs", "edges_on_cell", &wide, "2310adab74aad8dc62f924232199e595abb4572b4dc77a4124b868dc6ab83d29"),
            ("hex-ad/tools/dycore/recovery_topology.rs", "n_edges_on_cell", &wide, NEOC),
            ("hex-ad/tools/dycore/recovery_topology.rs", "max_edges", &wide, "7aa8ca4a02506da9133d8f889678b76f716ce45d02e22fdb7b70a15e56a0eff8"),
            ("hex-ad/tools/dycore/recovery_topology.rs", "cells_on_edge", &wide, COE),
            ("hex-ad/tools/dycore/derive_reverse_topology.py", "edges_on_cell", &wide, EOC10),
            ("hex-ad/tools/dycore/derive_reverse_topology.py", "n_edges_on_cell", &wide, NEOC),
            ("hex-ad/tools/dycore/derive_reverse_topology.py", "edges_on_vertex", &wide, EOV),
            ("hex-ad/tools/dycore/derive_reverse_topology.py", "cells_on_edge", &wide, COE),
            ("hex-ad/tools/dycore/derive_launch_scalar.py", "max_edges", &wide, "075de2b906dbd7066da008cab735bee896370154603579a50122f9b88545bd45"),
            ("hex-ad/tools/dycore/derive_launch_scalar.py", "vertex_degree", &wide, "9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178"),
            ("hex-ad/tools/dycore/derive_launch_scalar.py", "nedges", &wide, "b8c268933a184f364d04e8ed971ac3a677c91a434acc7e456de26d237e9faf51"),
            ("hex-ad/derive/dycore/horizontal_v841.rs#del4_edge_count", "nedges", &wide, "b8c268933a184f364d04e8ed971ac3a677c91a434acc7e456de26d237e9faf51"),
            ("hex-ad/tools/validate_inverse_mesh.py", "edges_on_cell", &wide, EOC10),
            ("hex-ad/tools/validate_inverse_mesh.py", "n_edges_on_cell", &wide, NEOC),
            ("hex-ad/tools/validate_inverse_mesh.py", "max_edges", &wide, "075de2b906dbd7066da008cab735bee896370154603579a50122f9b88545bd45"),
            ("hex-ad/tools/dycore/wind_inverse_csr.rs", "inverse_offsets", &compact, "8f94553fc42d6d8d8d4fb38368cb54b9fa950c2f7c9d7b177e436616485cd15c"),
            ("hex-ad/tools/dycore/wind_inverse_csr.rs", "inverse_cells", &compact, COE),
            ("hex-ad/tools/dycore/wind_inverse_csr.rs", "inverse_slots", &compact, "2ec4eed43c614d06dbddced074839bcfc4d7809d5d0b4b3423f488c5710bdd98"),
            ("u2:inverse_cells_on_edge", "inv_off", &wide, CSR_OFF),
            ("u2:inverse_cells_on_edge", "inv_edge", &wide, CSR_ENT),
            ("u2:inverse_cells_on_cell", "inv_off", &wide, CSR_OFF),
            ("u2:inverse_cells_on_cell", "inv_nbr", &wide, "5ae160d12e4d1b3d93c85477389e43231ec5ee1cacbe762bad6f58a98b445a24"),
            ("u2:inverse_cells_on_cell", "inv_slot", &wide, "16f01571481cd3b6299877fb87f3bfb82257ffbb1a1ea9128affd9899a339810"),
        ];
        for (recipe, name, m, want) in cases {
            assert!(topology_recipe(recipe) && needs(recipe, name).is_some(), "{recipe}");
            let got = build(recipe, name, "", m).unwrap_or_else(|e| panic!("{recipe} {name}: {e}"));
            assert_eq!(&sha(&got), want, "{recipe} {name}");
        }
        assert_eq!(build("recovery_topology.rs", "edges_on_cell", "", &wide).unwrap(), compact.arrays["edges_on_cell"].0);
        // inverse_topology.rs::theta with five inputs (theta_finish / w_finish edge_offsets, edge_owners).
        let (o, e) = crate::recipes::theta_edges(NC, NE, 10, &wide.arrays["n_edges_on_cell"].0, &wide.arrays["edges_on_cell"].0).unwrap();
        assert_eq!((sha(&o).as_str(), sha(&e).as_str()), ("8f94553fc42d6d8d8d4fb38368cb54b9fa950c2f7c9d7b177e436616485cd15c", COE));
        // edges_on_vertex derived from vertices_on_edge (a rank whose plan passes none) equals the plan's own.
        let mut no_eov = global(true);
        no_eov.arrays.remove("edges_on_vertex");
        assert_eq!(sha(&build("derive_reverse_topology.py", "edges_on_vertex", "", &no_eov).unwrap()), EOV);
        assert_eq!(build("derive_launch_scalar.py", "vertex_degree", "", &no_eov).unwrap(), [3]);
        // The launch-argument recipes are not mesh-input recipes (deriv.rs restate builds them from the launch).
        for r in ["hex-ad/tools/dycore_csr.rs", "transpose.rs::edge_to_slots", "inverse_topology.rs::theta", "dycore_csr.rs:transpose_fixed",
                  "transpose.rs#build", "derive_edge_incidence.py", "u2:inverse_adv_cells"] {
            assert!(topology_recipe(r) && needs(r, "x").is_none(), "{r}");
        }
        assert!(!topology_recipe("physics-ad/cuda/hrrrv5/glue.interface_shapes.rs") && !topology_recipe("local_edge_incidence"));
    }

    /// A rank-local relabelled mesh: 3 local cells (0, 1 owned; 2 halo), 4 edges, 2 vertices.
    fn small() -> Mesh {
        let mut m = Mesh::default();
        for (d, v) in [("ncells", 3), ("nedges", 4), ("nvertices", 2)] { m.dims.insert(d.into(), v); }
        // edges: 0 (0,1), 1 (1,2), 2 (2,0), 3 (2,2: a halo edge in the one-past form)
        m.arrays.insert("cells_on_edge".into(), (vec![0, 1, 1, 2, 2, 0, 2, 2], 2));
        // MPAS order (not ascending), width 4 with -1 padding
        m.arrays.insert("edges_on_cell".into(), (vec![2, 0, -1, -1, 0, 1, -1, -1, 1, 2, 3, -1], 4));
        m.arrays.insert("n_edges_on_cell".into(), (vec![2, 2, 3], 1));
        m.arrays.insert("vertices_on_edge".into(), (vec![0, 1, 1, 0, 0, 1, 1, 1], 2));
        m
    }
    #[test]
    fn local_recipes_on_a_relabelled_two_rank_style_mesh() {
        let m = small();
        assert_eq!(build("transpose.rs#pair", "cell_edge_offsets", "", &m).unwrap(), [0, 2, 4, 8]);
        assert_eq!(build("transpose.rs#pair", "cell_edges", "", &m).unwrap(), [0, 2, 0, 1, 1, 2, 3, 3]);
        assert_eq!(build("u2:inverse_cells_on_edge", "inv_off", "", &m).unwrap(), [0, 2, 4, 7]);
        assert_eq!(build("u2:inverse_cells_on_edge", "inv_edge", "", &m).unwrap(), [0, 2, 0, 1, 1, 2, 3]);
        assert_eq!(build("transpose.rs#f32-count", "ncells", "", &m).unwrap(), [3]);
        assert_eq!(build("del4_edge_count", "nedges", "", &m).unwrap(), [4]);
        assert_eq!(build("derive_reverse_topology.py", "edges_on_cell", "", &m).unwrap(), m.arrays["edges_on_cell"].0);
        assert_eq!(build("derive_launch_scalar.py", "max_edges", "", &m).unwrap(), [4]);
        assert_eq!(build("recovery_topology.rs", "max_edges", "", &m).unwrap(), [3]);
        assert_eq!(build("recovery_topology.rs", "edges_on_cell", "", &m).unwrap(), [2, 0, -1, 0, 1, -1, 1, 2, 3]);
        assert_eq!(build("wind_inverse_csr.rs", "inverse_offsets", "", &m).unwrap(), [0, 2, 4, 6, 7]);
        assert_eq!(build("wind_inverse_csr.rs", "inverse_cells", "", &m).unwrap(), [0, 1, 1, 2, 0, 2, 2]);
        assert_eq!(build("wind_inverse_csr.rs", "inverse_slots", "", &m).unwrap(), [1, 0, 1, 0, 0, 1, 2]);
        // edges_on_vertex from vertices_on_edge: vertex 0 has edges 0,1,2, vertex 1 has 0,1,2,3: a non-uniform degree is
        // refused by name, never padded.
        let e = build("derive_reverse_topology.py", "edges_on_vertex", "", &m).unwrap_err();
        assert!(e.contains("vertex 0 has 3 edges"), "{e}");
        // Admission: a slot naming an edge that does not touch the cell, an out-of-range edge, an out-of-range
        // cells_on_edge endpoint: each refused by name.
        let mut bad = small();
        bad.arrays.get_mut("edges_on_cell").unwrap().0[0] = 1;
        let e = build("recovery_topology.rs", "edges_on_cell", "", &bad).unwrap_err();
        assert!(e.contains("edge 1 does not have cell 0"), "{e}");
        let mut bad = small();
        bad.arrays.get_mut("edges_on_cell").unwrap().0[10] = 5;
        let e = build("validate_inverse_mesh.py", "max_edges", "", &bad).unwrap_err();
        assert!(e.contains("index 5 outside [0, 4]"), "{e}");
        let mut bad = small();
        bad.arrays.get_mut("cells_on_edge").unwrap().0[7] = 4;
        let e = build("transpose.rs#pair", "cell_edges", "", &bad).unwrap_err();
        assert!(e.contains("index 4 outside [0, 3]"), "{e}");
        assert!(build("u2:inverse_cells_on_edge", "inv_edge", "", &bad).is_err());
        assert!(build("derive_reverse_topology.py", "cells_on_edge", "", &bad).is_err());
        // The one-past garbage element (cell 3 of 3, edge 4 of 4; MPAS's LAM form): accepted, no row of its own, so
        // the inverse maps skip it (S01b e72756f5); a garbage slot needs no incidence.
        let mut g = small();
        g.arrays.get_mut("cells_on_edge").unwrap().0[7] = 3;
        assert_eq!(build("transpose.rs#pair", "cell_edges", "", &g).unwrap(), [0, 2, 0, 1, 1, 2, 3]);
        assert_eq!(build("transpose.rs#pair", "cell_edge_offsets", "", &g).unwrap(), [0, 2, 4, 7]);
        assert_eq!(build("u2:inverse_cells_on_edge", "inv_edge", "", &g).unwrap(), [0, 2, 0, 1, 1, 2, 3]);
        g.arrays.get_mut("edges_on_cell").unwrap().0[10] = 4;
        assert_eq!(build("validate_inverse_mesh.py", "edges_on_cell", "", &g).unwrap(), g.arrays["edges_on_cell"].0);
        assert_eq!(build("wind_inverse_csr.rs", "inverse_offsets", "", &g).unwrap(), [0, 2, 4, 6, 6]);
        // A reference clamped to local index 0 (partition_local_mesh_v841: absent from the rank) is the forward's
        // value: accepted though edge 0 does not touch cell 2, and kept in the inverse maps.
        let mut c = small();
        c.arrays.get_mut("edges_on_cell").unwrap().0[10] = 0;
        assert!(build("recovery_topology.rs", "edges_on_cell", "", &c).is_ok());
        assert_eq!(build("wind_inverse_csr.rs", "inverse_cells", "", &c).unwrap(), [0, 1, 2, 1, 2, 0, 2]);
    }

    /// The two rank-local meshes of the 2-rank x1.2562 partition (x1.2562.part2.K3.npz l2g maps applied to the global
    /// fixtures with the partition_local_mesh_v841 remap law): every mesh-input recipe builds on both, none refuses.
    #[test]
    fn local_recipes_build_on_the_real_two_rank_partition() {
        for (p, (nc, ne, nv)) in [(0, (1537, 4694, 3158)), (1, (1541, 4706, 3166))] {
            let f = |n: &str| fixture(&format!("../x1.2562-part2/p{p}.{n}"));
            let mut m = Mesh::default();
            for (d, v) in [("ncells", nc), ("nedges", ne), ("nvertices", nv)] { m.dims.insert(d.into(), v); }
            for (n, file, w) in [("cells_on_edge", "cells_on_edge", 2), ("edges_on_cell", "edges_on_cell_w10", 10), ("n_edges_on_cell", "n_edges_on_cell", 1),
                                 ("vertices_on_edge", "vertices_on_edge", 2), ("edges_on_vertex", "edges_on_vertex", 3)] {
                let v = f(file);
                assert_eq!(v.len(), w * if n.contains("vertex") && n.starts_with("edges") { nv } else if n.ends_with("_cell") { nc } else { ne }, "p{p} {n}");
                m.arrays.insert(n.into(), (v, w));
            }
            // The outer halo's clamped references are there (the partition law), and they are 0, never -1 or n.
            let eov = &m.arrays["edges_on_vertex"].0;
            let voe = &m.arrays["vertices_on_edge"].0;
            let clamped = (0..nv).flat_map(|v| (0..3).map(move |s| (v, s))).filter(|&(v, s)| { let e = eov[3 * v + s] as usize; voe[2 * e] != v as i32 && voe[2 * e + 1] != v as i32 }).count();
            assert_eq!(clamped, 86, "p{p}");
            for (recipe, name) in [("transpose.rs#pair", "cell_edges"), ("pressure_topology.rs", "cell_edge_offsets"), ("transpose.rs#f32-count", "ncells"),
                                   ("recovery_topology.rs", "edges_on_cell"), ("recovery_topology.rs", "max_edges"), ("derive_reverse_topology.py", "edges_on_vertex"),
                                   ("derive_reverse_topology.py", "edges_on_cell"), ("derive_launch_scalar.py", "max_edges"), ("derive_launch_scalar.py", "vertex_degree"),
                                   ("validate_inverse_mesh.py", "n_edges_on_cell"), ("wind_inverse_csr.rs", "inverse_slots"), ("u2:inverse_cells_on_edge", "inv_edge"),
                                   ("del4_edge_count", "nedges")] {
                build(recipe, name, "", &m).unwrap_or_else(|e| panic!("p{p} {recipe} {name}: {e}"));
            }
            assert_eq!(build("transpose.rs#f32-count", "ncells", "", &m).unwrap(), [nc as i32]);
            assert_eq!(build("derive_launch_scalar.py", "vertex_degree", "", &m).unwrap(), [3]);
            assert_eq!(build("transpose.rs#pair", "cell_edge_offsets", "", &m).unwrap()[nc], 2 * ne as i32);
        }
    }

    // ---- partition (2026-11-09): rank-local meshes of ANY METIS partition, any rank count, unbalanced. ----
    // Fixtures tree/ad/e2e/fixtures/x1.2562-parts/<set>/p<k>.{cell,edge,vertex}_l2g.i32.gz + meta.json, exported by
    // a partition export tool from hexcore's same-host layout assets (build_samehost_layouts_v841.py,
    // halo_rings 3): m2, m16, m32 (gpmetis -contig, balanced) and unbal8 (gpmetis -tpwgts 0.30/0.10x7: rank 0 owns 773
    // cells, the others ~255). The local mesh is built with the partition_local_mesh_v841._remap_values law (rows in l2g
    // order; a value absent from the rank clamped to local 0; negative padding kept).
    const PART_SETS: [&str; 4] = ["m2", "m16", "m32", "unbal8"];
    fn part_meta(set: &str) -> serde_json::Value {
        let m = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = [m.join("fixtures/x1.2562-parts"), m.join("../../../../../../e2e/fixtures/x1.2562-parts")].into_iter()
            .find(|d| d.is_dir()).expect("tree/ad/e2e/fixtures/x1.2562-parts");
        serde_json::from_slice(&std::fs::read(dir.join(set).join("meta.json")).unwrap()).unwrap()
    }
    fn remap_rows(global: &[i32], w: usize, l2g: &[i32], g2l: &[i32]) -> Vec<i32> {
        l2g.iter().flat_map(|&g| global[g as usize * w..(g as usize + 1) * w].to_vec())
            .map(|v| if v < 0 { v } else if g2l[v as usize] < 0 { 0 } else { g2l[v as usize] }).collect()
    }
    fn inverse(l2g: &[i32], n: usize) -> Vec<i32> {
        let mut g2l = vec![-1i32; n];
        for (l, &g) in l2g.iter().enumerate() { assert!(g2l[g as usize] < 0, "l2g repeats {g}"); g2l[g as usize] = l as i32; }
        g2l
    }
    /// (local mesh, cell l2g, owned cells) of rank p of a partition set.
    fn part_mesh(set: &str, p: usize) -> (Mesh, Vec<i32>, usize) {
        let f = |k: &str| fixture(&format!("../x1.2562-parts/{set}/p{p}.{k}_l2g"));
        let (cl, el, vl) = (f("cell"), f("edge"), f("vertex"));
        let (cg, eg, vg) = (inverse(&cl, NC), inverse(&el, NE), inverse(&vl, NV));
        let mut m = Mesh::default();
        for (d, v) in [("ncells", cl.len()), ("nedges", el.len()), ("nvertices", vl.len())] { m.dims.insert(d.into(), v); }
        let neoc = fixture("n_edges_on_cell");
        m.arrays.insert("cells_on_edge".into(), (remap_rows(&fixture("cells_on_edge"), 2, &el, &cg), 2));
        m.arrays.insert("edges_on_cell".into(), (remap_rows(&fixture("edges_on_cell_w10"), 10, &cl, &eg), 10));
        m.arrays.insert("n_edges_on_cell".into(), (cl.iter().map(|&g| neoc[g as usize]).collect(), 1));
        m.arrays.insert("vertices_on_edge".into(), (remap_rows(&fixture("vertices_on_edge"), 2, &el, &vg), 2));
        m.arrays.insert("edges_on_vertex".into(), (remap_rows(&fixture("edges_on_vertex"), 3, &vl, &eg), 3));
        let owned = part_meta(set)["ranks"][p]["owned_cells"].as_u64().unwrap() as usize;
        (m, cl, owned)
    }

    /// The remap law here is hexcore's: on the 2-way METIS partition (gpmetis is deterministic, so the a test machine part2) it
    /// reproduces the x1.2562-part2 fixtures the 2-rank recording's sealed plan carries, word for word.
    #[test]
    fn any_partition_remap_reproduces_the_recorded_two_rank_meshes() {
        for p in 0..2 {
            let (m, _, _) = part_mesh("m2", p);
            for (n, file) in [("cells_on_edge", "cells_on_edge"), ("edges_on_cell", "edges_on_cell_w10"), ("n_edges_on_cell", "n_edges_on_cell"),
                              ("vertices_on_edge", "vertices_on_edge"), ("edges_on_vertex", "edges_on_vertex")] {
                assert_eq!(m.arrays[n].0, fixture(&format!("../x1.2562-part2/p{p}.{file}")), "m2 p{p} {n}");
            }
        }
    }

    /// Every rank-local recipe builds on every rank of 2-, 16- and 32-way and an unbalanced 8-way partition; the reverse
    /// gather table (derive_reverse_topology edges_on_cell) is the EXACT transpose of what the forward edge loop reads
    /// (cells_on_edge, clamped endpoints included), so no cotangent is dropped or doubled at a seam; and ownership
    /// covers the mesh exactly once.
    #[test]
    fn local_recipes_build_and_transpose_exactly_on_any_partition() {
        for set in PART_SETS {
            let n = part_meta(set)["n_partitions"].as_u64().unwrap() as usize;
            let mut owner = vec![usize::MAX; NC];
            for p in 0..n {
                let (m, cl, owned) = part_mesh(set, p);
                for &g in &cl[..owned] { assert_eq!(owner[g as usize], usize::MAX, "{set}: cell {g} owned twice"); owner[g as usize] = p; }
                for (recipe, name) in [("transpose.rs#pair", "cell_edges"), ("pressure_topology.rs", "cell_edge_offsets"), ("transpose.rs#f32-count", "ncells"),
                                       ("recovery_topology.rs", "edges_on_cell"), ("recovery_topology.rs", "max_edges"), ("recovery_topology.rs", "cells_on_edge"),
                                       ("derive_reverse_topology.py", "edges_on_vertex"), ("derive_reverse_topology.py", "edges_on_cell"),
                                       ("derive_reverse_topology.py", "n_edges_on_cell"), ("derive_launch_scalar.py", "max_edges"),
                                       ("derive_launch_scalar.py", "vertex_degree"), ("validate_inverse_mesh.py", "n_edges_on_cell"),
                                       ("wind_inverse_csr.rs", "inverse_slots"), ("u2:inverse_cells_on_edge", "inv_edge"), ("del4_edge_count", "nedges")] {
                    build(recipe, name, "", &m).unwrap_or_else(|e| panic!("{set} p{p} {recipe} {name}: {e}"));
                }
                let (nc, ne) = (m.d("ncells").unwrap(), m.d("nedges").unwrap());
                let coe = &m.arrays["cells_on_edge"].0;
                let rows = build("derive_reverse_topology.py", "edges_on_cell", "", &m).unwrap();
                let counts = build("derive_reverse_topology.py", "n_edges_on_cell", "", &m).unwrap();
                let w = build("derive_launch_scalar.py", "max_edges", "", &m).unwrap()[0] as usize;
                assert_eq!(rows.len(), nc * w, "{set} p{p}");
                let mut want: Vec<Vec<i32>> = vec![Vec::new(); nc];
                for e in 0..ne { for s in 0..2 { let c = coe[2 * e + s]; if c >= 0 && !want[c as usize].contains(&(e as i32)) { want[c as usize].push(e as i32); } } }
                for c in 0..nc {
                    let mut got: Vec<i32> = rows[c * w..c * w + counts[c] as usize].to_vec();
                    got.sort_unstable(); want[c].sort_unstable();
                    assert_eq!(got, want[c], "{set} p{p} cell {c}: reverse gather row is not the transpose of cells_on_edge");
                }
                assert_eq!(build("transpose.rs#pair", "cell_edge_offsets", "", &m).unwrap()[nc], 2 * ne as i32, "{set} p{p}");
            }
            assert!(owner.iter().all(|&o| o != usize::MAX), "{set}: a cell has no owner");
        }
    }

    /// Halo depth: with halo_rings 3, every owned cell and every halo cell of ring 1 and 2 sees its whole stencil
    /// unclamped (its edges, their two cells and two vertices, and each vertex's three edges), on every rank of every
    /// set. The reverse is the transpose of the recorded forward, so it reads and writes exactly the entities the forward
    /// read; this pins that a radius-2 stencil around an owned cell never reaches a clamped reference, at any rank count
    /// and on the unbalanced partition. Clamped references exist only in ring 3 (the outermost), as the remap law says.
    #[test]
    fn halo_rings_cover_radius_two_stencils_on_any_partition() {
        let (coc, neoc) = (fixture("cells_on_cell"), fixture("n_edges_on_cell"));
        let (gcoe, gvoe, geov) = (fixture("cells_on_edge"), fixture("vertices_on_edge"), fixture("edges_on_vertex"));
        for set in PART_SETS {
            let n = part_meta(set)["n_partitions"].as_u64().unwrap() as usize;
            for p in 0..n {
                let (m, cl, owned) = part_mesh(set, p);
                let el = fixture(&format!("../x1.2562-parts/{set}/p{p}.edge_l2g"));
                let vl = fixture(&format!("../x1.2562-parts/{set}/p{p}.vertex_l2g"));
                let mut ring = vec![u8::MAX; NC];
                for &g in &cl[..owned] { ring[g as usize] = 0; }
                for r in 0..3u8 {
                    for c in 0..NC { if ring[c] == r { for s in 0..neoc[c] as usize { let d = coc[10 * c + s] as usize; if ring[d] == u8::MAX { ring[d] = r + 1; } } } }
                }
                let local: std::collections::HashSet<i32> = cl.iter().copied().collect();
                assert!((0..NC).all(|c| (ring[c] != u8::MAX) == local.contains(&(c as i32))), "{set} p{p}: local cells are not owned + 3 rings");
                let (eoc, coe, voe, eov) = (&m.arrays["edges_on_cell"].0, &m.arrays["cells_on_edge"].0, &m.arrays["vertices_on_edge"].0, &m.arrays["edges_on_vertex"].0);
                let leoc = &m.arrays["n_edges_on_cell"].0;
                for (lc, &g) in cl.iter().enumerate() {
                    if ring[g as usize] > 2 { continue; }
                    for s in 0..leoc[lc] as usize {
                        let e = eoc[10 * lc + s] as usize;
                        let ge = el[e] as usize;
                        for k in 0..2 {
                            assert_eq!(cl[coe[2 * e + k] as usize], gcoe[2 * ge + k], "{set} p{p} ring {} cell {g}: edge {ge} endpoint clamped", ring[g as usize]);
                            let v = voe[2 * e + k] as usize;
                            assert_eq!(vl[v], gvoe[2 * ge + k], "{set} p{p} cell {g}: edge {ge} vertex clamped");
                            for j in 0..3 {
                                let e2 = eov[3 * v + j];
                                assert!(e2 >= 0 && geov[3 * vl[v] as usize + j] == el[e2 as usize], "{set} p{p} cell {g}: vertex {} edge clamped", vl[v]);
                            }
                        }
                    }
                }
            }
        }
    }

    fn ob(f: &str, arrays: &[(&str, Key, Option<usize>)], dims: &[(&str, usize)]) -> Observation {
        Observation { function: f.into(), arrays: arrays.iter().map(|(n, k, w)| (n.to_string(), *k, *w)).collect(),
                      dims: dims.iter().map(|(d, v)| (d.to_string(), *v)).collect() }
    }
    /// Co-occurrence groups: two ranks' launches, in either order, give two groups; a launch finds its rank's group
    /// through any of its mesh arguments; arrays a launch does not pass resolve in its group; collisions are refused.
    #[test]
    fn registry_groups_by_co_occurrence_and_refuses_collisions() {
        let rank = |base: u64, nc: usize| vec![
            ob("lap4", &[("cells_on_edge", (base, 0), Some(2)), ("vertices_on_edge", (base + 1, 0), Some(2))], &[("ncells", nc)]),
            ob("theta", &[("edges_on_cell", (base + 2, 0), Some(10)), ("n_edges_on_cell", (base + 3, 0), Some(1)), ("cells_on_edge", (base, 0), Some(2))],
               &[("ncells", nc), ("nedges", 3 * nc)]),
            ob("recover", &[("edges_on_cell", (base + 4, 0), Some(6)), ("cells_on_edge", (base, 0), Some(2))], &[]),
            ob("vort", &[("edges_on_vertex", (base + 5, 0), Some(3)), ("vertices_on_edge", (base + 1, 0), Some(2))], &[("nvertices", 2 * nc)]),
        ];
        let mut obs = rank(10, 1283); obs.extend(rank(20, 1281));
        let mut fwd = Registry::default(); for o in obs.iter().cloned() { fwd.observe(o); } fwd.finish();
        let mut rev = Registry::default(); for o in obs.iter().rev().cloned() { rev.observe(o); } rev.finish();
        for r in [&fwd, &rev] {
            assert_eq!(r.groups.len(), 2);
            let (g0, g1) = (r.group(&[(10, 0)]).unwrap(), r.group(&[(25, 0)]).unwrap());
            assert_ne!(g0, g1);
            assert_eq!(r.group(&[(11, 0), (13, 0)]).unwrap(), g0);
            assert_eq!((r.dim(g0, "ncells").unwrap(), r.dim(g0, "nedges").unwrap(), r.dim(g0, "nvertices").unwrap()), (1283, 3849, 2566));
            assert_eq!(r.dim(g1, "ncells").unwrap(), 1281);
            // lap4 passes cells_on_edge + vertices_on_edge only; its same-mesh edges_on_cell / edges_on_vertex:
            assert_eq!(r.pick(g0, "edges_on_cell", Pick::Wide).unwrap(), [((12, 0), 10)]);
            assert_eq!(r.pick(g0, "edges_on_cell", Pick::Compact).unwrap(), [((14, 0), 6)]);
            assert_eq!(r.pick(g1, "edges_on_vertex", Pick::Any).unwrap(), [((25, 0), 3)]);
            assert!(r.pick(g0, "edges_on_cell", Pick::Any).unwrap_err().contains("widths"));
            assert!(r.pick(g0, "cells_on_cell", Pick::Any).unwrap().is_empty());
        }
        // A launch whose mesh arguments come from both ranks merges them: the group then has two cells_on_edge of one
        // width (the caller refuses unless they are equal word for word) and two values of ncells: refused by name.
        let mut obs2 = obs.clone();
        obs2.push(ob("bridge", &[("cells_on_edge", (10, 0), Some(2)), ("n_edges_on_cell", (23, 0), Some(1))], &[]));
        let mut r = Registry::default(); for o in obs2 { r.observe(o); } r.finish();
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.pick(0, "cells_on_edge", Pick::Any).unwrap(), [((10, 0), 2), ((20, 0), 2)]);
        let e = r.dim(0, "ncells").unwrap_err();
        assert!(e.contains("ncells") && e.contains("1281") && e.contains("1283"), "{e}");
        // One array given two widths by two launches: refused by name.
        let mut r = Registry::default();
        r.observe(ob("a", &[("edges_on_cell", (1, 0), Some(10))], &[]));
        r.observe(ob("b", &[("edges_on_cell", (1, 0), Some(6))], &[]));
        r.finish();
        assert!(r.width(0, "edges_on_cell", (1, 0)).unwrap_err().contains("widths {6, 10}"));
        assert!(r.group(&[(99, 0)]).is_err());
    }
}
