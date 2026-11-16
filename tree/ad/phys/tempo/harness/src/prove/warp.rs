// SPDX-License-Identifier: Apache-2.0
//! Whole-launch proof and cost of the warp-shaped TEMPO tangent/adjoint
//! (tempo_adw_*, generated with tempo-ad-gen --warp) on captured hex
//! tempo_warp_steps launches, every column at once:
//!  A. binding (optional, card of the captured image): the captured forward replays its post-state;
//!  B. checkpoint + count pass: identity of the augmented primal against the captured outputs on
//!     every column, per-substep node / phase / cross-lane edge counts, per-substep checkpoints;
//!  C. the launch tangent (segments forward) and adjoint (segments in reverse) over all columns,
//!     per-column dot, zero direction both ways, two reverse runs and two tangent runs bitwise;
//!  D. (optional) the serial P5 module's tangent on named columns, same direction: bitwise;
//!  E. cost: whole-launch adjoint wall (checkpoint pass + every segment record + sweep), per stage.
//! All scoring is here in Rust; CUDA does the column arithmetic.
use super::*;

const AD_BLOCK: u32 = 128;
const NCNT: usize = 8;

struct WGen { rec: *mut std::ffi::c_void, tl: *mut std::ffi::c_void, adj: *mut std::ffi::c_void, g: u64, work: usize, state: usize, prim: usize, _m: Module }
impl WGen {
    fn load(cubin: &[u8]) -> WGen {
        let m = Module::load(cubin);
        let (g, gn) = m.global("g_ad");
        let sz = Buf::new(32);
        Args::new().ptr(sz.p).launch(m.func("tempo_adw_sizes"), 1, 1);
        let s: Vec<u64> = sz.get(4);
        assert_eq!(gn, 144, "AdTape (warp) layout"); assert_eq!(s[3], 144);
        let rec = m.func("tempo_adw_record");
        eprintln!("[gen] warp record: local bytes/thread {}, registers {}, max threads/block {}; work {} B, state {} B per column",
                  gpu::func_attr(rec, 3), gpu::func_attr(rec, 4), gpu::func_attr(rec, 0), s[0], s[1]);
        WGen { rec, tl: m.func("tempo_adw_tl"), adj: m.func("tempo_adw_adj"), g, work: s[0] as usize, state: s[1] as usize, prim: s[2] as usize, _m: m }
    }
}

/// Host view of the tape pointers written into the module global before every launch.
#[derive(Default, Clone, Copy)]
struct TapeSet { a: u64, b: u64, pa: u64, pb: u64, br: u64, il: u64, count: u64, rbr: u64, ril: u64, t: i64, bcap: i64, icap: i64,
                 base: u64, ccap: u64, phase: u64, pcap: i64, mark: u64, nmark: i64 }
impl TapeSet {
    fn put(&self, g: u64) {
        let mut v: Vec<u8> = vec![];
        for x in [self.a, self.b, self.pa, self.pb, self.br, self.il, self.count, self.rbr, self.ril] { v.extend(x.to_le_bytes()); }
        for x in [self.t, self.bcap, self.icap] { v.extend(x.to_le_bytes()); }
        for x in [self.base, self.ccap, self.phase] { v.extend(x.to_le_bytes()); }
        v.extend(self.pcap.to_le_bytes()); v.extend(self.mark.to_le_bytes()); v.extend(self.nmark.to_le_bytes());
        assert_eq!(v.len(), 144);
        unsafe_put(g, &v);
    }
}

/// One tile: a contiguous range of the capture's columns, every buffer the record and the sweeps need.
struct WTile { c0: usize, n: usize, cols: Buf, st: Buf, wk: Buf, yout: Buf, oidx: Buf, iidx: Buf, cnt: Buf,
               slots: usize, pcap: usize, seg_base: Vec<Vec<i64>>, seg_ccap: Vec<Vec<i32>>, obcap: usize, a: Buf, b: Buf, pa: Buf, pb: Buf, base: Buf, ccap: Buf, phase: Buf, sc: Buf,
               obid: Buf, obv: Buf, fault: Buf }

pub fn run(a: &[String]) {
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let caps: Vec<String> = a.iter().enumerate().filter(|(i, x)| *x == "--capture" && *i + 1 < a.len()).map(|(i, _)| a[i + 1].clone()).collect();
    let out = arg("--out").expect("--out");
    let seeds: u64 = arg("--seeds").unwrap_or("3".into()).parse().unwrap();
    let budget_gb: f64 = arg("--budget-gb").unwrap_or("12".into()).parse().unwrap();
    let cmp_cols: Vec<usize> = arg("--cmp-columns").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
    let binding = a.iter().any(|x| x == "--binding");
    gpu::init();
    let cubin = std::fs::read(arg("--cubin").expect("--cubin")).unwrap();
    let gen = WGen::load(&cubin);
    let serial = arg("--serial-cubin").map(|p| std::fs::read(p).unwrap());
    let mut receipt = json!({"schema": "ad.phys.tempo-warp-proof/v1", "cubin_sha256": format!("{:x}", Sha256::digest(&cubin)),
        "serial_cubin_sha256": serial.as_ref().map(|c| format!("{:x}", Sha256::digest(c))),
        "work_bytes_per_column": gen.work, "state_bytes_per_column": gen.state, "prim_bytes": gen.prim, "captures": []});
    let (free_start, total) = gpu::mem_info();
    receipt["card_free_bytes_at_start"] = json!(free_start); receipt["card_total_bytes"] = json!(total);
    for cd in &caps {
        let tcap = Instant::now();
        let cap = Cap::load(Path::new(cd));
        let (nz, nc, steps) = (cap.nz, cap.columns, cap.steps as usize);
        let nin = 18 * nz + 10;
        let mut r = json!({"capture": cd, "step": cap.m["step"], "config": cap.m["config"], "nz": nz, "columns": nc, "substeps": steps, "dt": cap.dt,
                           "flags": {"aero": cap.flags[0], "hail": cap.flags[1], "melt": cap.flags[3], "first": cap.flags[4], "presence": cap.flags[5], "radii": cap.flags[6]}});
        eprintln!("[load] {} ({} columns, {} levels, {} substeps) in {:.1} s", cd, nc, nz, steps, tcap.elapsed().as_secs_f64());
        if a.iter().any(|x| x == "--serial-only") {
            // Serial P5 module alone: count-only record over the named columns, then all columns (diagnostic).
            let sg = Gen::load(serial.as_ref().expect("--serial-cubin"));
            for cols in [cmp_cols.clone(), (0..nc).collect::<Vec<usize>>()] {
                let tc = Tile::new(&sg, &cols, nin, 0, 0, 0);
                let t = Instant::now(); tc.record(&sg, &cap);
                let cnt = tc.counts();
                eprintln!("[S] serial count-only record over {} columns: {:.1} s, nodes max {}", cols.len(), t.elapsed().as_secs_f64(), cnt.iter().map(|c| c[0]).max().unwrap());
            }
            continue;
        }
        // ---------------------------------------------------------------- A. binding (optional)
        let mut fwd_ms = vec![];
        if binding {
            let img = std::fs::read(cap.dir.join(cap.m["image"]["path"].as_str().unwrap())).unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(&img)), cap.m["image"]["sha256"].as_str().unwrap());
            let fwd = Module::load(&img);
            let ff = fwd.func("tempo_warp_steps");
            for rep in 0..3 {
                cap.restore(&cap.pre);
                let t = Instant::now(); cap.run_captured(ff); fwd_ms.push(t.elapsed().as_secs_f64() * 1e3);
                if rep == 0 {
                    let mut d = 0usize;
                    for (i, b) in cap.dev.iter().enumerate() {
                        if i == cap.cost_alloc || i == cap.blob_alloc || cap.scratch.contains(&i) { continue; }
                        let got: Vec<u8> = b.get(b.n);
                        d += got.chunks(4).zip(cap.post[i].chunks(4)).filter(|(x, y)| x != y).count();
                    }
                    eprintln!("[A] captured image replay: {d} differing words");
                    r["binding"] = json!({"differing_words": d});
                }
            }
            cap.restore(&cap.pre);
            fwd_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
            r["forward_launch_ms"] = json!(fwd_ms);
        }
        // ---------------------------------------------------------------- B. checkpoint + count pass, all columns
        let all32: Vec<i32> = (0..nc as i32).collect();
        let t_all = nc * 32;
        let colsb = Buf::from(&all32);
        let st_all = Buf::new(gen.state * nc); let wk_all = Buf::new(gen.work * nc);
        let yall = Buf::new(gen.prim * nin * nc); let oall = Buf::new(4 * nin * nc); let iall = Buf::new(4 * nin * nc);
        let cnt_all = Buf::new(4 * NCNT * t_all);
        let mark = Buf::new(4 * 3 * (steps + 1) * t_all);
        let ck = Buf::new(gen.prim * steps * nin * nc);
        let ckm = Buf::new(steps * nin * nc);
        let rec_args = |tile_cols: u64, n: usize, s0: usize, s1: usize, ckin: u64, ckout: u64, ckoff: usize, st: u64, wk: u64, yo: u64, oi: u64, ii: u64| -> Args {
            Args::new().i32(nz as i32).i32(nc as i32).i32(steps as i32).f32(cap.dt)
                .ptr(cap.dev[cap.f_tab].p).ptr(cap.dev[cap.s_tab].p).ptr(cap.dev[cap.blob_alloc].p)
                .i32(cap.flags[0]).i32(cap.flags[1]).i32(cap.flags[3]).i32(cap.flags[4]).i32(cap.flags[5]).i32(cap.flags[6])
                .ptr(tile_cols).i32(n as i32).i32(s0 as i32).i32(s1 as i32).ptr(ckin).ptr(ckout).ptr(ckm.p).raw((nc as i64).to_le_bytes().to_vec()).i32(ckoff as i32)
                .ptr(st).ptr(wk).ptr(yo).ptr(oi).ptr(ii).ptr(0)
        };
        let grid = |n: usize| ((n * 32) as u32 + AD_BLOCK - 1) / AD_BLOCK;
        let mut ckpt_ms = vec![];
        for _rep in 0..2 {
            TapeSet { count: cnt_all.p, t: t_all as i64, mark: mark.p, nmark: (steps + 1) as i64, ..Default::default() }.put(gen.g);
            let t = Instant::now();
            rec_args(colsb.p, nc, 0, steps, 0, ck.p, 0, st_all.p, wk_all.p, yall.p, oall.p, iall.p).launch(gen.rec, grid(nc), AD_BLOCK);
            ckpt_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ckv: Vec<f32> = if gen.prim == 4 { ck.get(steps * nin * nc) } else { ck.get::<f64>(steps * nin * nc).iter().map(|&v| v as f32).collect() };
        let yv: Vec<f32> = if gen.prim == 4 { yall.get(nin * nc) } else { yall.get::<f64>(nin * nc).iter().map(|&v| v as f32).collect() };
        let mut idiff = vec![0usize; 3]; let mut icmp = vec![0usize; 3]; let mut ex = vec![];
        for c in 0..nc {
            let y = cap.outputs(&cap.post, c);
            for o in 0..nin {
                if y[o].is_nan() && y[o].to_bits() == f32::NAN.to_bits() { continue; }
                let g = if o < 14 * nz { 0 } else if o < 14 * nz + 10 { 1 } else { 2 };
                icmp[g] += 1;
                let got = yv[o * nc + c];
                if got.to_bits() != y[o].to_bits() { idiff[g] += 1; if ex.len() < 10 { ex.push(json!({"column": c, "slot": o, "got": got, "want": y[o]})); } }
            }
        }
        let cn: Vec<i32> = cnt_all.get(NCNT * t_all);
        let ovf = (0..t_all).filter(|&t| cn[NCNT * t + 3] != 0).count();
        let mk: Vec<i32> = mark.get(3 * (steps + 1) * t_all);
        let mkv = |i: usize, t: usize, f: usize| mk[(i * t_all + t) * 3 + f];
        // Per substep: node capacity per lane (max lane of each column), phases, cross-lane edges per phase.
        let mut seg_cap = vec![vec![0usize; nc]; steps]; let mut seg_ph = vec![0usize; steps]; let mut seg_rm = vec![0usize; steps];
        let mut lane_nodes_total = 0u64;
        for s in 0..steps { for c in 0..nc { for l in 0..32 {
            let t = c * 32 + l;
            let dn = (mkv(s + 1, t, 0) - mkv(s, t, 0)) as usize;
            lane_nodes_total += dn as u64;
            seg_cap[s][c] = seg_cap[s][c].max(dn + 64);
            seg_ph[s] = seg_ph[s].max((mkv(s + 1, t, 1) - mkv(s, t, 1)) as usize);
            seg_rm[s] = seg_rm[s].max(mkv(s + 1, t, 2) as usize);
        } } }
        let col_cap: Vec<usize> = (0..nc).map(|c| (0..steps).map(|s| seg_cap[s][c]).max().unwrap()).collect();
        let pcap = seg_ph.iter().max().unwrap() + 4; let obcap = seg_rm.iter().max().unwrap() + 8;
        let mut cc = col_cap.clone(); cc.sort();
        let slots_all: usize = (0..steps).map(|s| seg_cap[s].iter().map(|x| x * 32).sum::<usize>()).max().unwrap();
        eprintln!("[B] identity: fields {}/{} differ, accumulators {}/{}, profiles {}/{}; overflow threads {}; checkpoint+count pass {:.1} / {:.1} ms (forward {:?} ms)",
                  idiff[0], icmp[0], idiff[1], icmp[1], idiff[2], icmp[2], ovf, ckpt_ms[0], ckpt_ms[1], fwd_ms);
        eprintln!("[B] nodes {} total ({:.0} per column per substep); lane capacity per column per substep: median {} max {}; phases/substep {:?}; max cross-lane edges/phase {:?}; largest segment tape slots {:.2e} ({:.1} GB with sweep scratch)",
                  lane_nodes_total, lane_nodes_total as f64 / (nc * steps) as f64, cc[nc / 2], cc[nc - 1], seg_ph, seg_rm, slots_all as f64, slots_all as f64 * 32.0 / 1e9);
        r["identity"] = json!({"columns": nc, "compared_words": icmp.iter().sum::<usize>(), "differing_words": idiff.iter().sum::<usize>(),
            "fields": [idiff[0], icmp[0]], "accumulators": [idiff[1], icmp[1]], "profiles": [idiff[2], icmp[2]], "examples": ex, "overflow_threads": ovf});
        r["counts"] = json!({"nodes_total": lane_nodes_total, "nodes_per_column_per_substep_mean": lane_nodes_total as f64 / (nc * steps) as f64,
            "lane_capacity_per_column": {"min": cc[0], "median": cc[nc / 2], "p90": cc[nc * 9 / 10], "max": cc[nc - 1]},
            "phases_per_substep": seg_ph, "max_cross_lane_edges_per_phase": seg_rm, "tape_slots_all_columns": slots_all});
        drop(st_all); drop(wk_all); drop(yall); drop(oall); drop(iall); drop(cnt_all); drop(mark);
        // ---------------------------------------------------------------- tiles by the memory budget
        // Tile budget: the asked budget, never more than the card's free memory now less 2 GB (shared card).
        let (free_now, _) = gpu::mem_info();
        let budget = ((budget_gb * 1e9) as usize).min(free_now.saturating_sub(2usize << 30));
        let per_col_fixed = gen.state + gen.work + gen.prim * nin + 3 * 4 * nin + 4 * NCNT * 32 + 8 + 4 + 4 * pcap * 32 + 12 * obcap * 32 + 4 + 2 * 8 * nin;
        // Tape blocks are laid out per segment (each segment's exact lane capacity per column), so a tile needs
        // the largest segment's slots (24 B tape + 8 B sweep scratch each) plus its fixed per-column buffers.
        let mut tiles: Vec<(usize, usize)> = vec![];
        { let mut c0 = 0usize; let mut seg_acc = vec![0usize; steps]; let mut fixed = 0usize;
          for c in 0..nc {
              let need = |acc: &Vec<usize>, f: usize| acc.iter().zip(0..steps).map(|(a, s)| a + seg_cap[s][c] * 32 * 32).max().unwrap() + f + per_col_fixed;
              if need(&seg_acc, fixed) > budget && c > c0 { tiles.push((c0, c - c0)); c0 = c; seg_acc = vec![0; steps]; fixed = 0; }
              for s in 0..steps { seg_acc[s] += seg_cap[s][c] * 32 * 32; } fixed += per_col_fixed; }
          tiles.push((c0, nc - c0)); }
        eprintln!("[C] {} tile(s) of {:?} columns (budget {:.1} GB, card free {:.1} GB)", tiles.len(), tiles.iter().map(|t| t.1).collect::<Vec<_>>(), budget as f64 / 1e9, free_now as f64 / 1e9);
        let (free0, _) = gpu::mem_info();
        let mk_tile = |c0: usize, n: usize| -> WTile {
            let cols: Vec<i32> = (c0..c0 + n).map(|c| c as i32).collect();
            let mut seg_base = vec![]; let mut seg_ccap = vec![]; let mut slots = 0usize;
            for s in 0..steps {
                let mut base = vec![0i64; n]; let mut acc = 0i64;
                for t in 0..n { base[t] = acc; acc += (seg_cap[s][c0 + t] * 32) as i64; }
                slots = slots.max(acc as usize);
                seg_base.push(base); seg_ccap.push((0..n).map(|t| seg_cap[s][c0 + t] as i32).collect::<Vec<i32>>());
            }
            let (base, ccap) = (seg_base[0].clone(), seg_ccap[0].clone());
            WTile { c0, n, cols: Buf::from(&cols), st: Buf::new(gen.state * n), wk: Buf::new(gen.work * n), yout: Buf::new(gen.prim * nin * n),
                    oidx: Buf::new(4 * nin * n), iidx: Buf::new(4 * nin * n), cnt: Buf::new(4 * NCNT * 32 * n), slots, pcap, obcap, seg_base, seg_ccap,
                    a: Buf::new(4 * slots), b: Buf::new(4 * slots), pa: Buf::new(8 * slots), pb: Buf::new(8 * slots), base: Buf::from(&base), ccap: Buf::from(&ccap),
                    phase: Buf::new(4 * pcap * 32 * n), sc: Buf::new(8 * slots), obid: Buf::new(4 * obcap * 32 * n), obv: Buf::new(8 * obcap * 32 * n), fault: Buf::new(4 * n) }
        };
        let tape = |tl: &WTile| TapeSet { a: tl.a.p, b: tl.b.p, pa: tl.pa.p, pb: tl.pb.p, count: tl.cnt.p, t: (tl.n * 32) as i64,
                                          base: tl.base.p, ccap: tl.ccap.p, phase: tl.phase.p, pcap: tl.pcap as i64, ..Default::default() };
        // record segment s of tile tl (tape on)
        let record = |tl: &WTile, s: usize| -> f64 {
            tl.base.put(&tl.seg_base[s]); tl.ccap.put(&tl.seg_ccap[s]);
            tape(tl).put(gen.g);
            let ckin = if s > 0 { ck.p + (gen.prim * s * nin * nc) as u64 } else { 0 };
            let t = Instant::now();
            rec_args(tl.cols.p, tl.n, s, s + 1, ckin, 0, tl.c0, tl.st.p, tl.wk.p, tl.yout.p, tl.oidx.p, tl.iidx.p).launch(gen.rec, grid(tl.n), AD_BLOCK);
            let dt = t.elapsed().as_secs_f64();
            let c: Vec<i32> = tl.cnt.get(NCNT * 32 * tl.n);
            let bad = (0..32 * tl.n).filter(|&t| c[NCNT * t + 3] != 0).count();
            assert!(bad == 0, "segment {s}: {bad} threads overflowed their tape/phase capacity");
            dt
        };
        let sweep_tl = |tl: &WTile, xd: &Buf, yd: &Buf| -> f64 {
            let t = Instant::now();
            Args::new().i32(tl.n as i32).i32(nin as i32).i32(nin as i32).ptr(xd.p).ptr(tl.sc.p).ptr(tl.iidx.p).ptr(tl.oidx.p).ptr(yd.p).launch(gen.tl, grid(tl.n), AD_BLOCK);
            t.elapsed().as_secs_f64()
        };
        let sweep_adj = |tl: &WTile, yb: &Buf, xb: &Buf| -> f64 {
            let t = Instant::now();
            Args::new().i32(tl.n as i32).i32(nin as i32).i32(nin as i32).ptr(yb.p).ptr(tl.sc.p).ptr(tl.iidx.p).ptr(tl.oidx.p).ptr(xb.p)
                .ptr(tl.obid.p).ptr(tl.obv.p).i32(tl.obcap as i32).ptr(tl.fault.p).launch(gen.adj, grid(tl.n), AD_BLOCK);
            let dt = t.elapsed().as_secs_f64();
            let f: Vec<i32> = tl.fault.get(tl.n);
            let bad: Vec<(usize, i32)> = f.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (tl.c0 + i, v)).take(5).collect();
            assert!(bad.is_empty(), "adjoint fault (column, bits): {bad:?}");
            dt
        };
        // Launch tangent / adjoint over one tile; vectors [slot][n] f64.
        // Segment vectors stay on the card: two buffers swap roles (input of one segment, output of the next).
        let launch_tl = |tl: &WTile, xdot: &[f64], st: &mut Vec<(f64, f64)>| -> Vec<f64> {
            let mut v = [Buf::from(xdot), Buf::new(8 * nin * tl.n)];
            for s in 0..steps { let r_ = record(tl, s); let w_ = sweep_tl(tl, &v[0], &v[1]); st.push((r_, w_)); v.swap(0, 1); }
            v[0].get(nin * tl.n)
        };
        let launch_adj = |tl: &WTile, ybar: &[f64], st: &mut Vec<(f64, f64)>| -> Vec<f64> {
            let mut v = [Buf::from(ybar), Buf::new(8 * nin * tl.n)];
            for s in (0..steps).rev() { let r_ = record(tl, s); let w_ = sweep_adj(tl, &v[0], &v[1]); st.push((r_, w_)); v.swap(0, 1); }
            v[0].get(nin * tl.n)
        };
        let xin: Vec<Vec<f32>> = (0..nc).map(|c| cap.inputs(&cap.pre, c)).collect();
        let mut dot_rows = vec![]; let mut worst_col = 0f64; let mut zero_ok = true; let mut det_adj = true; let mut det_tl = true;
        let mut adj_wall = vec![]; let mut tl_wall = vec![]; let mut adj_first: Vec<f64> = vec![]; let mut adj_warm: Vec<f64> = vec![]; let mut stage = json!({}); let mut mem_peak = 0usize;
        let mut ydot_keep: Vec<Vec<f64>> = vec![vec![]; nc]; let mut xbar_keep: Vec<Vec<f64>> = vec![vec![]; nc]; let mut xdot_keep: Vec<Vec<f64>> = vec![vec![]; nc]; let mut ybar_keep: Vec<Vec<f64>> = vec![vec![]; nc];
        let (mut seg_diff, mut seg_cmp, mut seg_extra_max) = (0usize, 0usize, i64::MIN);
        let mut zero_words = [0usize; 3]; // nonzero tangent words, nonzero adjoint words, negative zeros
        let mut lhs_tot = vec![0f64; seeds as usize]; let mut rhs_tot = vec![0f64; seeds as usize];
        for &(c0, n) in &tiles {
            let tl = mk_tile(c0, n);
            let (free1, _) = gpu::mem_info(); mem_peak = mem_peak.max(free0.saturating_sub(free1));
            // Segment identity: each segment record (from its checkpoint) gives the next checkpoint / the launch outputs
            // bit for bit, and its node count per lane is the count pass's for that substep plus its leaves.
            for s in 0..steps {
                record(&tl, s);
                let y: Vec<f32> = if gen.prim == 4 { tl.yout.get(nin * n) } else { tl.yout.get::<f64>(nin * n).iter().map(|&v| v as f32).collect() };
                let cnt: Vec<i32> = tl.cnt.get(NCNT * 32 * n);
                for t in 0..n {
                    let c = c0 + t;
                    for q in 0..nin {
                        let want = if s + 1 < steps { ckv[(s + 1) * nin * nc + q * nc + c] } else { yv[q * nc + c] };
                        if s + 1 == steps { let po = cap.outputs(&cap.post, c)[q]; if po.is_nan() && po.to_bits() == f32::NAN.to_bits() { continue; } }
                        if s + 1 < steps && q >= 18 * nz && cap.s_t[q - 18 * nz].is_none() { continue; }
                        seg_cmp += 1; if y[q * n + t].to_bits() != want.to_bits() { seg_diff += 1; }
                    }
                    for l in 0..32 { let th = t * 32 + l; let dn = (mkv(s + 1, c * 32 + l, 0) - mkv(s, c * 32 + l, 0)) as i64;
                        seg_extra_max = seg_extra_max.max(cnt[NCNT * th] as i64 - 1 - dn); }
                }
            }
            for seed in 1..=seeds {
                let mut g = Rng::new(seed); let mut xd = vec![0f64; nin * n];
                for t in 0..n { for q in 0..nin { xd[q * n + t] = g.normal() * in_scale(nz, &xin[c0 + t], q); } }
                let mut g2 = Rng::new(100 + seed); let mut yb = vec![0f64; nin * n];
                for t in 0..n { for q in 0..nin { yb[q * n + t] = g2.normal(); } }
                let mut st_tl = vec![]; let mut st_ad = vec![];
                let t0 = Instant::now(); let yd = launch_tl(&tl, &xd, &mut st_tl); let tlw = t0.elapsed().as_secs_f64();
                let t1 = Instant::now(); let xb = launch_adj(&tl, &yb, &mut st_ad); let adw = t1.elapsed().as_secs_f64();
                tl_wall.push(tlw); adj_wall.push(adw);
                if seed == 1 { adj_first.push(adw); } else if seed == seeds { adj_warm.push(adw); }
                for t in 0..n {
                    let (mut lc, mut rc) = (0f64, 0f64);
                    for q in 0..nin { lc += yd[q * n + t] * yb[q * n + t]; rc += xd[q * n + t] * xb[q * n + t]; }
                    let rel = (lc - rc).abs() / lc.abs().max(rc.abs()).max(1e-300);
                    if rel > worst_col { worst_col = rel; }
                    lhs_tot[seed as usize - 1] += lc; rhs_tot[seed as usize - 1] += rc;
                    if seed == 1 {
                        ydot_keep[c0 + t] = (0..nin).map(|q| yd[q * n + t]).collect(); xbar_keep[c0 + t] = (0..nin).map(|q| xb[q * n + t]).collect();
                        xdot_keep[c0 + t] = (0..nin).map(|q| xd[q * n + t]).collect(); ybar_keep[c0 + t] = (0..nin).map(|q| yb[q * n + t]).collect();
                    }
                }
                if seed == 1 {
                    stage[format!("tile{}", c0)] = json!({"columns": n, "record_ms_per_segment_tl": st_tl.iter().map(|x| x.0 * 1e3).collect::<Vec<_>>(),
                        "tl_sweep_ms_per_segment": st_tl.iter().map(|x| x.1 * 1e3).collect::<Vec<_>>(),
                        "record_ms_per_segment_adj": st_ad.iter().map(|x| x.0 * 1e3).collect::<Vec<_>>(),
                        "adj_sweep_ms_per_segment": st_ad.iter().map(|x| x.1 * 1e3).collect::<Vec<_>>(), "tl_wall_s": tlw, "adj_wall_s": adw});
                    let xb2 = launch_adj(&tl, &yb, &mut vec![]);
                    if xb2.iter().zip(&xb).any(|(p, q)| p.to_bits() != q.to_bits()) { det_adj = false; }
                    let yd2 = launch_tl(&tl, &xd, &mut vec![]);
                    if yd2.iter().zip(&yd).any(|(p, q)| p.to_bits() != q.to_bits()) { det_tl = false; }
                    let yz = launch_tl(&tl, &vec![0f64; nin * n], &mut vec![]);
                    let xz = launch_adj(&tl, &vec![0f64; nin * n], &mut vec![]);
                    // exact zero: every word compares equal to 0.0 (a negative zero, 0 * negative partial, is zero)
                    let nz_tl = yz.iter().filter(|v| **v != 0.0 || v.is_nan()).count();
                    let nz_ad = xz.iter().filter(|v| **v != 0.0 || v.is_nan()).count();
                    let negz = yz.iter().chain(xz.iter()).filter(|v| v.to_bits() == 0x8000_0000_0000_0000).count();
                    zero_words[0] += nz_tl; zero_words[1] += nz_ad; zero_words[2] += negz;
                    if nz_tl + nz_ad > 0 { zero_ok = false; }
                }
                eprintln!("[C] tile {c0}+{n} seed {seed}: tangent {tlw:.2} s, adjoint {adw:.2} s (segments: record {:.0} ms, sweep {:.0} ms total)",
                          st_ad.iter().map(|x| x.0).sum::<f64>() * 1e3, st_ad.iter().map(|x| x.1).sum::<f64>() * 1e3);
            }
            drop(tl);
        }
        eprintln!("[C] segment identity: {seg_diff}/{seg_cmp} words differ; segment nodes per lane minus count-pass nodes, max {seg_extra_max} (leaves)");
        r["segment_identity"] = json!({"differing_words": seg_diff, "compared_words": seg_cmp, "max_extra_nodes_per_lane_vs_count_pass": seg_extra_max});
        for s in 0..seeds as usize {
            let rel = (lhs_tot[s] - rhs_tot[s]).abs() / lhs_tot[s].abs().max(rhs_tot[s].abs());
            dot_rows.push(json!({"seed": s + 1, "lhs": lhs_tot[s], "rhs": rhs_tot[s], "rel": rel}));
        }
        let dot_worst = dot_rows.iter().map(|x| x["rel"].as_f64().unwrap()).fold(0f64, f64::max);
        // whole-launch adjoint wall: checkpoint pass + every tile's segments (record + sweep)
        // whole-launch adjoint: the checkpoint/count pass plus every tile's reverse segments (record + sweep), first seed
        // of each tile (cold buffers) and last seed (warm)
        let adj_launch_s = ckpt_ms[0] / 1e3 + adj_first.iter().sum::<f64>();
        let adj_launch_warm_s = ckpt_ms[1] / 1e3 + if adj_warm.len() == tiles.len() { adj_warm.iter().sum::<f64>() } else { adj_first.iter().sum::<f64>() };
        eprintln!("[C] dot (all {nc} columns): worst total rel {dot_worst:.2e}, worst column {worst_col:.2e}; zero direction exact {zero_ok} (nonzero words {zero_words:?}); reverse repeat bitwise {det_adj}; tangent repeat bitwise {det_tl}; whole-launch adjoint {adj_launch_s:.2} s (warm {adj_launch_warm_s:.2} s); device peak {:.2} GB",
                  mem_peak as f64 / 1e9);
        r["dot"] = json!({"rows": dot_rows, "worst_rel": dot_worst, "worst_column_rel": worst_col, "columns": nc, "zero_direction_exact": zero_ok, "zero_direction_nonzero_words": {"tangent": zero_words[0], "adjoint": zero_words[1], "negative_zero_words": zero_words[2]},
                          "reverse_repeat_bitwise": det_adj, "tangent_repeat_bitwise": det_tl, "precision": "tape partials and sweeps in float64; primal float32 (the forward's own)"});
        r["cost"] = json!({"checkpoint_count_pass_ms": ckpt_ms, "whole_launch_adjoint_s": adj_launch_s, "whole_launch_adjoint_warm_s": adj_launch_warm_s, "adjoint_wall_s_per_tile_seed": adj_wall, "tangent_wall_s_per_tile_seed": tl_wall,
                           "tiles": tiles, "stages": stage, "device_bytes_peak_tiles": mem_peak, "checkpoint_bytes": gen.prim * steps * nin * nc, "forward_launch_ms": fwd_ms});
        receipt["captures"].as_array_mut().unwrap().push(r.clone());
        std::fs::write(&out, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
        receipt["captures"].as_array_mut().unwrap().pop();
        // ---------------------------------------------------------------- D. serial P5 module, named columns
        if let (Some(sc), false) = (serial.as_ref(), cmp_cols.is_empty()) {
            let sg = Gen::load(sc);
            let tc = Tile::new(&sg, &cmp_cols, nin, 0, 0, 0);
            tc.record(&sg, &cap);
            let cnt = tc.counts();
            let capn = cnt.iter().map(|c| c[0] as usize).max().unwrap() + 1;
            drop(tc);
            let ts = Tile::new(&sg, &cmp_cols, nin, capn, 0, 0);
            let t = Instant::now(); ts.record(&sg, &cap); let srec = t.elapsed().as_secs_f64();
            let m = cmp_cols.len();
            let mut xd = vec![0f64; nin * m]; let mut yb = vec![0f64; nin * m];
            for (t, &c) in cmp_cols.iter().enumerate() { for q in 0..nin { xd[q * m + t] = xdot_keep[c][q]; yb[q * m + t] = ybar_keep[c][q]; } }
            let (yd, stl) = ts.tl(&sg, nin, &xd); let (xb, sad) = ts.adj(&sg, nin, &yb);
            let mut tl_diff = 0usize; let mut tl_cmp = 0usize; let mut adj_rel = 0f64; let mut rows = vec![];
            for (t, &c) in cmp_cols.iter().enumerate() {
                let mut d = 0usize; let (mut num, mut den) = (0f64, 0f64);
                for q in 0..nin {
                    tl_cmp += 1; if yd[q * m + t].to_bits() != ydot_keep[c][q].to_bits() { d += 1; }
                    num += (xb[q * m + t] - xbar_keep[c][q]).powi(2); den += xbar_keep[c][q].powi(2);
                }
                tl_diff += d; let rel = (num / den.max(1e-300)).sqrt(); adj_rel = adj_rel.max(rel);
                rows.push(json!({"column": c, "tangent_differing_words": d, "adjoint_rel_l2": rel, "serial_nodes": cnt[t][0]}));
            }
            eprintln!("[D] serial P5 module on columns {:?}: tangent {}/{} words differ (bitwise check), adjoint rel L2 worst {:.2e}; serial record {:.1} s, TL {:.1} s, adjoint {:.1} s",
                      cmp_cols, tl_diff, tl_cmp, adj_rel, srec, stl, sad);
            r["serial_cross_check"] = json!({"columns": rows, "tangent_differing_words": tl_diff, "tangent_compared_words": tl_cmp, "adjoint_rel_l2_worst": adj_rel,
                                             "serial_record_s": srec, "serial_tl_s": stl, "serial_adjoint_s": sad});
        }
        receipt["captures"].as_array_mut().unwrap().push(r);
        std::fs::write(&out, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    }
    std::fs::write(&out, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
}
