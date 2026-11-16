// SPDX-License-Identifier: Apache-2.0
//! Copy-row launches (rrfs_post_transport_copy, rrfs_glue_copy_plan,
//! cupy_concatenate_same_size) in one form, and the host build of their
//! parallel transpose. Geometry is integer and fixed by the plan, so the gather
//! is built once per launch geometry, never inside a step.
use super::plan::{args, le_i64, Arg, Plan, Slice};
use serde_json::Value;
use std::collections::HashMap;

/// Strides in 4-byte words; the forward kernels decompose e as i3 = e % n3,
/// i2, i1, i0 = rest; d = dst + i0*ds0 + i1*ds1 + i2*ds2 + i3*ds3.
#[derive(Clone, Debug)]
pub struct Row { pub dst: Slice, pub src: Option<Slice>, pub ds: [i64; 4], pub ss: [i64; 4],
                 pub n: [i64; 3], pub count: i64 }
impl Row {
    #[inline] pub fn at(&self, e: i64) -> (i64, i64) {
        let mut q = e;
        let i3 = q % self.n[2]; q /= self.n[2];
        let i2 = q % self.n[1]; q /= self.n[1];
        let i1 = q % self.n[0]; q /= self.n[0];
        (q * self.ds[0] + i1 * self.ds[1] + i2 * self.ds[2] + i3 * self.ds[3],
         q * self.ss[0] + i1 * self.ss[1] + i2 * self.ss[2] + i3 * self.ss[3])
    }
}

/// rrfs_post_transport_copy: twelve rows of (dst, src, count, n1, n2, n3, d0..d3, s0..s3).
pub fn post_transport(op: &Value) -> Result<Vec<Row>, String> {
    let a = args(op);
    if a.len() != 168 { return Err(format!("post-transport copy has {} arguments, not 168", a.len())); }
    let int = |x: &Arg| match x { Arg::Scalar(b) if b.len() == 8 => Ok(le_i64(b, 0)), _ => Err("bad scalar".to_string()) };
    let mut rows = Vec::new();
    for r in 0..12 {
        let g = &a[14 * r..14 * r + 14];
        let w: Vec<i64> = g[2..].iter().map(int).collect::<Result<_, _>>()?;
        if w[0] == 0 { continue; }
        let (Arg::Ptr(d), Arg::Ptr(s)) = (&g[0], &g[1]) else { return Err("post-transport row without pointers".into()) };
        rows.push(Row { dst: *d, src: Some(*s), n: [w[1], w[2], w[3]], ds: [w[4], w[5], w[6], w[7]],
                        ss: [w[8], w[9], w[10], w[11]], count: w[0] });
    }
    Ok(rows)
}

/// rrfs_copy_plan_upload rows, materialized exactly as copy_plan::table orders
/// them (stable sort: size-1 dims first, then by |dst stride| descending).
pub fn copy_plan(plan: &Plan, op: &Value) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    for r in op["rows"].as_array().ok_or("copy plan without rows")? {
        let words = |k: &str| -> Result<[i64; 4], String> {
            let v = r[k].as_array().ok_or(format!("copy plan row without {k}"))?;
            let w: Vec<i64> = v.iter().map(|s| plan.host_i64(s)).collect::<Result<_, _>>()?;
            w.try_into().map_err(|_| format!("{k} rank"))
        };
        let shape = words("shape")?; let dsr = words("dst_stride")?;
        let (src, ssr) = if r["src"].is_null() { (None, [0; 4]) } else { (Some(super::plan::slice(&r["src"])), words("src_stride")?) };
        let mut order = [0usize, 1, 2, 3];
        order.sort_by(|&a, &b| (shape[a] > 1).cmp(&(shape[b] > 1))
            .then_with(|| dsr[b].unsigned_abs().cmp(&dsr[a].unsigned_abs())));
        let sh = order.map(|k| shape[k]);
        out.push(Row { dst: super::plan::slice(&r["dst"]), src, ds: order.map(|k| dsr[k]), ss: order.map(|k| ssr[k]),
                       n: [sh[1], sh[2], sh[3]], count: sh.iter().product() });
    }
    Ok(out)
}

/// cupy_concatenate_same_size: y[i] = x[array][offset] with y laid out [top][array][base].
/// Only the float32 instantiation is a copy rule (rows move 4-byte words); the
/// caller classifies by the module's element type first (the hrrrv5 plan's only
/// instantiation, module 82af2f9c, is `typedef long long T`: integer, inactive).
/// y must be contiguous float32 (CArray stride 4 bytes); anything else is refused.
pub fn concatenate(table: &[Slice], y: Slice, ysize: i64, ystride: i64, base: i64) -> Result<Vec<Row>, String> {
    if ystride != 4 { return Err(format!("concatenate output stride {ystride} bytes: not contiguous float32, no rows rule")); }
    let nx = table.len() as i64;
    let ntop = ysize / (base * nx);
    Ok(table.iter().enumerate().map(|(a, &src)| Row {
        dst: Slice { alloc: y.alloc, off: y.off + 4 * (a as usize) * base as usize }, src: Some(src),
        ds: [0, 0, nx * base, 1], ss: [0, 0, base, 1], n: [1, ntop, base], count: ntop * base }).collect())
}

/// native_cublas_transpose (step-1 setup): the binder replays it as
/// cublasSgeam(OP_T, OP_T, m, n, 1, A lda=n, 0, A ldb=n, C ldc=m), so in
/// column-major terms C[i + j*m] = A[j + i*n] for i < m, j < n, float32.
/// A pure permutation: tangent copies, adjoint is the direct transpose.
pub fn cublas_transpose(plan: &Plan, op: &Value) -> Result<Row, String> {
    let (m, n) = (plan.host_i64(&op["m"])?, plan.host_i64(&op["n"])?);
    if m <= 0 || n <= 0 { return Err("bad cuBLAS transpose geometry".into()); }
    Ok(Row { dst: super::plan::slice(&op["output"]), src: Some(super::plan::slice(&op["input"])),
             ds: [0, 0, 1, m], ss: [0, 0, n, 1], n: [1, m, n], count: m * n })
}

/// A CuPy view copy (rank <= 4, byte strides) as one row. Used when the source
/// broadcasts (a zero stride over an extent > 1): the transpose then has fan-in
/// and goes through the deterministic gather instead of the elementwise rule.
pub fn from_view(dst: Slice, shape: &[i64], dst_strides: &[i64], src: Slice, src_strides: &[i64]) -> Row {
    let pad = 4 - shape.len();
    let (mut n, mut ds, mut ss) = ([1i64; 4], [0i64; 4], [0i64; 4]);
    for k in 0..shape.len() { n[pad + k] = shape[k]; ds[pad + k] = dst_strides[k] / 4; ss[pad + k] = src_strides[k] / 4; }
    Row { dst, src: Some(src), ds, ss, n: [n[1], n[2], n[3]], count: n.iter().product() }
}
pub fn broadcasts(shape: &[i64], strides: &[i64]) -> bool { shape.iter().zip(strides).any(|(&n, &s)| n > 1 && s == 0) }

/// Arena layout: every allocation a launch touches gets a base element index.
pub struct Arena { pub base: HashMap<u64, i64>, pub words: i64 }
impl Arena {
    pub fn new(plan: &Plan, allocs: impl IntoIterator<Item = u64>) -> Result<Self, String> {
        let mut ids: Vec<u64> = allocs.into_iter().collect(); ids.sort(); ids.dedup();
        let (mut base, mut at) = (HashMap::new(), 0i64);
        for id in ids {
            let b = *plan.bytes.get(&id).or(plan.later.get(&id)).ok_or(format!("unknown allocation {id}"))?;
            base.insert(id, at); at += ((b as i64 + 3) / 4 + 63) / 64 * 64;
        }
        Ok(Arena { base, words: at })
    }
    pub fn el(&self, s: Slice) -> i64 { self.base[&s.alloc] + (s.off / 4) as i64 }
}

/// Device form of a row (matches struct P8Row in cuda/p8.cu).
#[repr(C)] #[derive(Clone, Copy)]
pub struct DevRow { pub dst: i64, pub src: i64, pub ds: [i64; 4], pub ss: [i64; 4], pub n: [i64; 3], pub count: i64 }
pub fn dev_rows(rows: &[Row], ar: &Arena) -> Vec<DevRow> {
    rows.iter().map(|r| DevRow { dst: ar.el(r.dst), src: r.src.map_or(-1, |s| ar.el(s)), ds: r.ds, ss: r.ss, n: r.n, count: r.count }).collect()
}

/// The serial reference's order, as a gather: CSR over distinct source
/// elements of (dst id << 1 | repeat) items, plus the distinct destinations.
pub struct Gather { pub src_word: Vec<i64>, pub start: Vec<i64>, pub item: Vec<i64>, pub dst_word: Vec<i64>,
                    pub sourceless: Vec<u8>, pub threads: usize, pub build_ms: f64,
                    /// Every destination and source visited once: the direct kernel applies.
                    pub injective: bool }
pub fn gather(rows: &[DevRow]) -> Result<Gather, String> {
    let t0 = std::time::Instant::now();
    // Serial order: rows last to first, elements ascending. Position k is global.
    let order: Vec<usize> = (0..rows.len()).rev().collect();
    let mut first = vec![0i64; rows.len() + 1];
    for (i, &r) in order.iter().enumerate() { first[i + 1] = first[i] + rows[r].count; }
    let total = first[rows.len()] as usize;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    // (dst, k, src or -1) for every visit, generated in parallel chunks.
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
                    let e = k - first[i];
                    let mut q = e;
                    let i3 = q % r.n[2]; q /= r.n[2]; let i2 = q % r.n[1]; q /= r.n[1]; let i1 = q % r.n[0]; q /= r.n[0];
                    let d = r.dst + q * r.ds[0] + i1 * r.ds[1] + i2 * r.ds[2] + i3 * r.ds[3];
                    let s = if r.src >= 0 { r.src + q * r.ss[0] + i1 * r.ss[1] + i2 * r.ss[2] + i3 * r.ss[3] } else { -1 };
                    *v = (d, k, s);
                }
            });
        }
    });
    // Destinations: distinct words, first visit owns the seed.
    let mut by_dst: Vec<(i64, i64)> = visits.iter().map(|&(d, k, _)| (d, k)).collect();
    par_sort(&mut by_dst, threads);
    // dst_word is ascending (by_dst sorted), so ids are found by binary search.
    let mut dst_word: Vec<i64> = Vec::new(); let mut first_k: Vec<i64> = Vec::new();
    for &(d, k) in &by_dst { if dst_word.last() != Some(&d) { dst_word.push(d); first_k.push(k); } }
    // Sources: (src, k, item) sorted by source then serial position; built in parallel chunks.
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
    // The gather reads every seed before any destination is cleared, which is
    // the serial result only if no source element is also a destination.
    if let Some(&s) = src_word.iter().find(|s| dst_word.binary_search(s).is_ok()) {
        return Err(format!("source element {s} is also a destination: transpose must stay serial for this launch"));
    }
    // A destination fed from a fill row and a copy row is not sourceless.
    let injective = dst_word.len() == total && src_word.len() == item.len();
    Ok(Gather { src_word, start, item, dst_word, sourceless, threads, build_ms: t0.elapsed().as_secs_f64() * 1e3, injective })
}

/// Sort in `threads` chunks in parallel, then merge pairwise in parallel.
fn par_sort<T: Ord + Copy + Send>(v: &mut Vec<T>, threads: usize) {
    let n = v.len();
    if n < 1 << 16 || threads < 2 { v.sort_unstable(); return; }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|sc| { for p in v.chunks_mut(chunk) { sc.spawn(move || p.sort_unstable()); } });
    let mut runs: Vec<Vec<T>> = v.chunks(chunk).map(|c| c.to_vec()).collect();
    while runs.len() > 1 {
        let mut next = Vec::new();
        std::thread::scope(|sc| {
            let hs: Vec<_> = runs.chunks(2).map(|p| { let p = p.to_vec(); sc.spawn(move || {
                if p.len() == 1 { return p.into_iter().next().unwrap(); }
                let (a, b) = (&p[0], &p[1]); let mut m = Vec::with_capacity(a.len() + b.len());
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() { if a[i] <= b[j] { m.push(a[i]); i += 1 } else { m.push(b[j]); j += 1 } }
                m.extend_from_slice(&a[i..]); m.extend_from_slice(&b[j..]); m }) }).collect();
            for h in hs { next.push(h.join().unwrap()); }
        });
        runs = next;
    }
    *v = runs.pop().unwrap_or_default();
}
