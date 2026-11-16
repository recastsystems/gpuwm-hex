// SPDX-License-Identifier: Apache-2.0
//! bl-forward (opt-in, default off): the launch shape of the plan's MYNN-EDMF forward launch
//! `rrfs_mynnedmf_run_l16`. The forward image holds three entries of one column routine, `rrfs_mynnedmf_run_l1`,
//! `_l4` and `_l16` (1, 4 or 16 lanes a column, the same arithmetic in the same order per column). The recording chose
//! 16 lanes a column, which fills a card on a 2,562-column mesh; on the 3 km storm box (38,278 columns, launches of
//! 29,216 and 9,062) and on the 60 km mesh (163,842 columns, launches of 29,216 and 17,762) fewer lanes run fewer
//! idle lanes. Measured on a 5070 Ti with real columns (bl-forward bltime.py, receipts n4-3km-same.json and
//! n4-60km-same.json, tiled columns): 3 km 0.244 -> 0.119 s a step and 60 km 1.236 -> 0.528 s a step, both with l1;
//! on every real column of each mesh (n4-m-3km-same.json, n4-m-60km-same.json) 0.213 -> 0.095 s and 0.845 -> 0.465 s.
//!
//! Words: AD_MYNN_FWD_LANES = 1 | 4 | 16 picks the entry; AD_MYNN_FWD_BLOCK = 32..=256, a multiple of 32, the block
//! (default 32, the recorded block). Unset: the recorded launch is issued unchanged (function, grid and block).
//! The grid covers batch_count x lanes threads (argument 1), as the recorded grid covers batch_count x 16.
//!
//! THE BREAKAGE THE REFUSALS PREVENT: a launch whose batch_count argument is not 4 bytes, or whose recorded grid does
//! not cover exactly batch_count x 16 threads in 32-thread blocks, is not the launch this shape was measured on: the
//! re-shaped grid could then leave columns unrun. Such a launch is an error, never a quiet fallback.
//!
//! Same bytes, the workspace included (bl-forward wire): entries of different lanes place a column's words at different
//! workspace addresses, the plans carry the workspace allocation and hash it at step boundaries, and the routine copies
//! five TKE-budget diagnostics (dqke, qwt, qshear, qbuoy, qdiss) out of workspace words it never writes when
//! bl_mynn_tkebudget is 0. So the program moves the launch's columns of the workspace into the entry's layout before
//! the launch and back after it ("the workspace layout" below): outside the launch the workspace holds the recorded
//! entry's bytes. Measured on a home card over every column of the 3 km and 60 km meshes (bltime.py --relayout
//! --ws-hash): the 86 argument arrays and the whole workspace equal l16's with l1 and l4, also when the workspace starts
//! from a pattern instead of a clear.

use crate::submit::{Argument, Command, Slice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape { pub lanes: u32, pub block: u32 }

pub const RECORDED: &str = "rrfs_mynnedmf_run_l16";

/// The words, read once. None: off.
pub fn from_env() -> Result<Option<Shape>, String> {
    parse(std::env::var("AD_MYNN_FWD_LANES").ok().as_deref(), std::env::var("AD_MYNN_FWD_BLOCK").ok().as_deref())
}

pub fn parse(lanes: Option<&str>, block: Option<&str>) -> Result<Option<Shape>, String> {
    let Some(l) = lanes else {
        if block.is_some() { return Err("AD_MYNN_FWD_BLOCK needs AD_MYNN_FWD_LANES".into()); }
        return Ok(None);
    };
    let lanes: u32 = match l.trim() { "1" => 1, "4" => 4, "16" => 16, x => return Err(format!("AD_MYNN_FWD_LANES {x}: 1, 4 or 16")) };
    let block: u32 = match block {
        None => 32,
        Some(b) => match b.trim().parse::<u32>() {
            Ok(v) if (32..=256).contains(&v) && v % 32 == 0 => v,
            _ => return Err(format!("AD_MYNN_FWD_BLOCK {b}: 32 to 256, a multiple of 32")),
        },
    };
    Ok(Some(Shape { lanes, block }))
}

/// The entry, grid and block to issue for one plan launch; None when the launch is not the recorded MYNN-EDMF forward.
pub fn reshape(shape: Shape, function: &str, arguments: &[Argument], grid: [u32; 3], block: [u32; 3])
               -> Result<Option<(String, [u32; 3], [u32; 3])>, String> {
    if function != RECORDED { return Ok(None); }
    let count = match arguments.get(1) {
        Some(Argument::Bytes(b)) if b.len() == 4 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        _ => return Err(format!("{RECORDED}: argument 1 (batch_count) is not 4 by-value bytes; launch shape refused")),
    };
    if count <= 0 { return Err(format!("{RECORDED}: batch_count {count}")); }
    let threads = count as u64 * 16;
    if block != [32, 1, 1] || grid[1] != 1 || grid[2] != 1 || grid[0] as u64 != threads.div_ceil(32) {
        return Err(format!("{RECORDED}: recorded grid {grid:?} block {block:?} is not batch_count {count} x 16 lanes in 32-thread blocks"));
    }
    let want = count as u64 * shape.lanes as u64;
    let g = want.div_ceil(shape.block as u64);
    if g > u32::MAX as u64 { return Err(format!("{RECORDED}: grid {g}")); }
    Ok(Some((format!("rrfs_mynnedmf_run_l{}", shape.lanes), [g as u32, 1, 1], [shape.block, 1, 1])))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(count: i32) -> Vec<Argument> { vec![Argument::Bytes(0i32.to_le_bytes().to_vec()), Argument::Bytes(count.to_le_bytes().to_vec())] }

    #[test]
    fn off_unless_asked() {
        assert_eq!(parse(None, None).unwrap(), None);
        assert!(parse(None, Some("64")).is_err());
        assert!(parse(Some("2"), None).is_err());
        assert!(parse(Some("4"), Some("48")).is_err());
        assert_eq!(parse(Some("1"), None).unwrap(), Some(Shape { lanes: 1, block: 32 }));
    }

    #[test]
    fn recorded_3km_launches_reshape_to_cover_every_column() {
        // the recorded 3 km step: grid 14,608 (29,216 columns) and 4,531 (9,062 columns), block 32
        for (count, grid) in [(29216, 14608u32), (9062, 4531)] {
            for (lanes, block) in [(1u32, 32u32), (1, 128), (4, 128), (16, 32)] {
                let (f, g, b) = reshape(Shape { lanes, block }, RECORDED, &args(count), [grid, 1, 1], [32, 1, 1]).unwrap().unwrap();
                assert_eq!(f, format!("rrfs_mynnedmf_run_l{lanes}"));
                assert_eq!(b, [block, 1, 1]);
                assert!(g[0] as u64 * block as u64 >= count as u64 * lanes as u64);
                assert!((g[0] as u64 - 1) * (block as u64) < count as u64 * lanes as u64);
            }
            // l16 at block 32 is the recorded launch itself
            assert_eq!(reshape(Shape { lanes: 16, block: 32 }, RECORDED, &args(count), [grid, 1, 1], [32, 1, 1]).unwrap().unwrap().1, [grid, 1, 1]);
        }
    }

    #[test]
    fn other_launches_untouched_and_odd_records_refused() {
        let s = Shape { lanes: 1, block: 32 };
        assert_eq!(reshape(s, "mynnsfc_columns", &args(10), [5, 1, 1], [32, 1, 1]).unwrap(), None);
        // a planted record: one block short of batch_count x 16 (it would leave columns unrun)
        assert!(reshape(s, RECORDED, &args(29216), [14607, 1, 1], [32, 1, 1]).is_err());
        assert!(reshape(s, RECORDED, &args(29216), [7304, 1, 1], [64, 1, 1]).is_err());
        assert!(reshape(s, RECORDED, &[Argument::Bytes(vec![0; 4])], [1, 1, 1], [32, 1, 1]).is_err());
    }
}

static WORDS: std::sync::OnceLock<Result<Option<Shape>, String>> = std::sync::OnceLock::new();
/// The words of this process, read once.
pub fn get() -> Result<Option<Shape>, String> { WORDS.get_or_init(from_env).clone() }

// ------------------------------------------------------------------------------------------------------------------
// bl-forward wire (speed16): the workspace layout. Entries of different lanes place a column's words at different
// workspace addresses (fwd.cu MynnArena: word w of column c at word (c - c % T) * col_words + w * T + c % T, with
// T = 32 / lanes columns a warp tile). The recorded plans carry the workspace allocation (197 and 198 at 3 km, 421,
// 462 and 463 at 60 km) and hash it whole at step boundaries, with no residue declared, and the routine reads words
// it never writes (the five TKE-budget outputs when bl_mynn_tkebudget is 0). THE BREAKAGE THIS PREVENTS: with the
// l1 or l4 entry and no relayout, step 1's boundary refuses (state hash mismatch, allocation 197; checker of
// 2026-11-13 18:06Z, measured ws digests l16 ecb836c5, l4 8fd6bdca, l1 0b25b6f1), and a later launch of the same step
// would read the earlier launch's residue at other addresses. So around every re-shaped launch the program moves the
// launch's whole 32-column groups from the l16 layout into the entry's before it and back after it. Outside the
// launch the workspace is then always in the recorded layout: the tape, the AD hooks, the boundary hash and every
// later launch see the bytes of the recorded entry. The move before is skipped when the command just before was a
// clear covering those bytes (a uniform byte moves to itself): the recorded plans clear the workspace in the
// operation just before each step's first MYNN-EDMF launch.

/// The relayout kernel, NVRTC PTX of mynn_ws_relayout.cu for compute_80 with the header pinned to ISA 7.0
/// (bl-forward/mkptx.py), so every driver that runs sm_80 or later JITs it; no NVCC is needed at build time.
pub const RELAYOUT_PTX: &str = include_str!("mynn_ws_relayout.ptx");
pub const RELAYOUT_ENTRY: &str = "ad_mynn_ws_relayout";
/// Columns a warp tile holds in the recorded entry (16 lanes a column).
pub const RECORDED_TILE: u32 = 2;
/// Every entry's tile (2, 8 or 32 columns) nests in a group of 32 columns, so a group moves on its own.
pub const GROUP: usize = 32;
/// Groups moved per chunk through the scratch (16 x 32 x 73,448 bytes = 37.6 MB at 3 km and 60 km).
pub const CHUNK_GROUPS: usize = 16;
/// The id the scratch carries in a relayout batch's allocation list; refused if a plan allocation has it.
pub const SCRATCH_ID: u64 = 0x6d79_6e6e_7265_6c61;

pub fn tile(lanes: u32) -> u32 { 32 / lanes }

/// One re-shaped launch's workspace: its allocation and offset, the bytes a column, the whole groups that cover the
/// launch's columns, and the entry's tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relay { pub allocation: u64, pub offset: usize, pub stride: usize, pub columns: usize, pub tile: u32 }

impl Relay {
    pub fn bytes(&self) -> usize { self.columns * self.stride }
    /// True when a clear of `bytes` at (allocation, offset) covers every byte this relay moves.
    pub fn covered_by(&self, clear: (u64, usize, usize)) -> bool {
        let (a, o, n) = clear;
        a == self.allocation && o <= self.offset && o.checked_add(n).map_or(false, |end| end >= self.offset + self.bytes())
    }
}

pub fn scratch_bytes(stride: usize) -> usize { CHUNK_GROUPS * GROUP * stride }

/// The relayout a re-shaped launch needs; None when the entry's tile is the recorded one (l16 at any block).
pub fn relay(shape: Shape, arguments: &[Argument], allocation_bytes: impl Fn(u64) -> Option<usize>) -> Result<Option<Relay>, String> {
    let t = tile(shape.lanes);
    if t == RECORDED_TILE { return Ok(None); }
    let count = match arguments.get(1) {
        Some(Argument::Bytes(b)) if b.len() == 4 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        _ => return Err(format!("{RECORDED}: argument 1 (batch_count) is not 4 by-value bytes")),
    };
    if count <= 0 { return Err(format!("{RECORDED}: batch_count {count}")); }
    let (allocation, offset) = match arguments.get(2) {
        Some(Argument::Device(s)) => (s.allocation, s.offset),
        _ => return Err(format!("{RECORDED}: argument 2 (workspace) is not a device pointer; its layout cannot be moved")),
    };
    let stride = match arguments.get(3) {
        Some(Argument::Bytes(b)) if b.len() == 8 => u64::from_le_bytes(b[..8].try_into().unwrap()) as usize,
        _ => return Err(format!("{RECORDED}: argument 3 (workspace_stride) is not 8 by-value bytes")),
    };
    if stride == 0 || stride % 8 != 0 { return Err(format!("{RECORDED}: workspace_stride {stride} is not a positive multiple of 8")); }
    let columns = (count as usize).div_ceil(GROUP) * GROUP;
    let need = columns.checked_mul(stride).ok_or("workspace bytes overflow")?;
    let have = allocation_bytes(allocation).ok_or_else(|| format!("{RECORDED}: workspace allocation {allocation} not live"))?
        .checked_sub(offset).ok_or_else(|| format!("{RECORDED}: workspace offset {offset} past allocation {allocation}"))?;
    // THE BREAKAGE THIS PREVENTS: entry lX writes whole tiles of 32/X columns; a workspace that does not hold whole
    // 32-column groups would be written past its end by the last partial tile.
    if need > have {
        return Err(format!("{RECORDED}: workspace allocation {allocation} holds {have} bytes from offset {offset}, fewer than \
                            {columns} columns (whole 32-column groups) x {stride}: entry l{} would write past it", shape.lanes));
    }
    if scratch_bytes(stride) / 4 >= 1 << 31 { return Err(format!("{RECORDED}: workspace_stride {stride} too large for a relayout chunk")); }
    Ok(Some(Relay { allocation, offset, stride, columns, tile: t }))
}

/// The commands that move `r`'s groups from tile `from` to tile `to`: per chunk a copy into the scratch, then the
/// relayout kernel from the scratch back into place.
pub fn commands(r: &Relay, from: u32, to: u32, function: *mut std::ffi::c_void) -> Vec<Command> {
    let scratch = Slice { allocation: SCRATCH_ID, offset: 0 };
    let chunk = CHUNK_GROUPS * GROUP;
    let mut v = Vec::with_capacity(2 * r.columns.div_ceil(chunk));
    let mut c0 = 0;
    while c0 < r.columns {
        let n = chunk.min(r.columns - c0);
        let at = Slice { allocation: r.allocation, offset: r.offset + c0 * r.stride };
        let bytes = n * r.stride;
        let words = (bytes / 4) as u32;
        v.push(Command::DeviceCopy { destination: scratch, source: at, bytes });
        v.push(Command::Kernel { function, grid: [words.div_ceil(256).min(4096), 1, 1], block: [256, 1, 1], shared_bytes: 0,
            arguments: vec![Argument::Device(at), Argument::Device(scratch), Argument::Bytes(words.to_le_bytes().to_vec()),
                            Argument::Bytes(((r.stride / 4) as u32).to_le_bytes().to_vec()),
                            Argument::Bytes(from.to_le_bytes().to_vec()), Argument::Bytes(to.to_le_bytes().to_vec())] });
        c0 += n;
    }
    v
}

/// The kernel's index map (mynn_ws_relayout.cu, line for line): destination word d of a chunk reads source word s.
#[cfg(test)]
pub fn kernel_source_word(d: u32, col_words: u32, tile_src: u32, tile_dst: u32) -> u32 {
    let td = tile_dst * col_words;
    let (t, r) = (d / td, d % td);
    let (w, lane) = (r / tile_dst, r % tile_dst);
    let c = t * tile_dst + lane;
    let (ts, ls) = (c / tile_src, c % tile_src);
    ts * tile_src * col_words + w * tile_src + ls
}

/// fwd.cu's arena address: word w of column c with tile T.
#[cfg(test)]
pub fn arena_word(c: usize, w: usize, col_words: usize, t: usize) -> usize { (c - c % t) * col_words + w * t + c % t }

#[cfg(test)]
mod relay_tests {
    use super::*;
    fn ws_args(count: i32, alloc: u64, stride: u64) -> Vec<Argument> {
        vec![Argument::Bytes(0i32.to_le_bytes().to_vec()), Argument::Bytes(count.to_le_bytes().to_vec()),
             Argument::Device(Slice { allocation: alloc, offset: 0 }), Argument::Bytes(stride.to_le_bytes().to_vec())]
    }

    /// Run `commands` on host memory exactly as the card would: copy into the scratch, then the kernel's index map with
    /// the words the commands carry (destination, source, n_words, col_words, tile_src, tile_dst).
    fn host_run(ws: &mut [u32], r: &Relay, from: u32, to: u32) {
        let mut scratch = vec![0u32; scratch_bytes(r.stride) / 4];
        let word = |a: &Argument| -> u32 { let Argument::Bytes(b) = a else { panic!() }; u32::from_le_bytes(b[..4].try_into().unwrap()) };
        for c in commands(r, from, to, std::ptr::null_mut()) {
            match c {
                Command::DeviceCopy { destination, source, bytes } => {
                    assert_eq!(destination.allocation, SCRATCH_ID);
                    let o = source.offset / 4;
                    scratch[..bytes / 4].copy_from_slice(&ws[o..o + bytes / 4]);
                },
                Command::Kernel { arguments, .. } => {
                    let Argument::Device(at) = arguments[0] else { panic!() };
                    let Argument::Device(src) = arguments[1] else { panic!() };
                    assert_eq!(src.allocation, SCRATCH_ID);
                    let (n, cw, ts, td) = (word(&arguments[2]), word(&arguments[3]), word(&arguments[4]), word(&arguments[5]));
                    let o = at.offset / 4;
                    for d in 0..n { ws[o + d as usize] = scratch[kernel_source_word(d, cw, ts, td) as usize]; }
                },
                _ => panic!(),
            }
        }
    }

    #[test]
    fn kernel_map_is_the_arena_layout() {
        // every (column, word) of 64 columns: the kernel's source word for the destination address in tile `to` is the
        // arena address in tile `from`, for every pair of entries
        let cw = 7usize;
        for (from, to) in [(2u32, 32u32), (32, 2), (2, 8), (8, 2), (8, 32)] {
            for c in 0..64 { for w in 0..cw {
                let d = arena_word(c, w, cw, to as usize) as u32;
                assert_eq!(kernel_source_word(d, cw as u32, from, to) as usize, arena_word(c, w, cw, from as usize), "c {c} w {w} {from}->{to}");
            } }
        }
    }

    #[test]
    fn relay_round_trip_moves_only_the_launch_groups_and_restores_the_recorded_layout() {
        // a small workspace of 3 chunks + 3 groups; a launch of 1,573 columns (not a multiple of 32: last group partial)
        let (stride, total_cols) = (5 * 8usize, (3 * CHUNK_GROUPS + 3) * GROUP);
        let r = relay(Shape { lanes: 1, block: 32 }, &ws_args(1573, 9, stride as u64), |_| Some(total_cols * stride)).unwrap().unwrap();
        assert_eq!((r.columns, r.tile), (1600, 32));
        let cw = stride / 4;
        // the recorded layout holds value (c, w) -> c * 1000 + w
        let mut ws = vec![0u32; total_cols * cw];
        for c in 0..total_cols { for w in 0..cw { ws[arena_word(c, w, cw, 2)] = (c * 1000 + w) as u32; } }
        let base = ws.clone();
        host_run(&mut ws, &r, RECORDED_TILE, r.tile);
        for c in 0..total_cols { for w in 0..cw {
            let want = (c * 1000 + w) as u32;
            if c < r.columns { assert_eq!(ws[arena_word(c, w, cw, 32)], want, "moved c {c} w {w}"); }
            else { assert_eq!(ws[arena_word(c, w, cw, 2)], want, "untouched c {c} w {w}"); }
        } }
        host_run(&mut ws, &r, r.tile, RECORDED_TILE);
        assert_eq!(ws, base);
        // a planted error: moving back with the l4 tile does not restore the recorded layout
        host_run(&mut ws, &r, RECORDED_TILE, r.tile);
        host_run(&mut ws, &r, r.tile, 8);
        assert_ne!(ws, base);
    }

    #[test]
    fn relay_refusals_and_skips() {
        let s1 = Shape { lanes: 1, block: 32 };
        // l16 at any block keeps the recorded tile: nothing to move
        assert_eq!(relay(Shape { lanes: 16, block: 128 }, &ws_args(29216, 1, 73448), |_| Some(1 << 40)).unwrap(), None);
        // the recorded 3 km workspace: 29,216 x 73,448 bytes; launches of 29,216 and 9,062 columns
        let a = |_| Some(29216 * 73448);
        assert_eq!(relay(s1, &ws_args(29216, 197, 73448), a).unwrap().unwrap().columns, 29216);
        assert_eq!(relay(s1, &ws_args(9062, 197, 73448), a).unwrap().unwrap().columns, 9088);
        // 60 km last launch 17,762 columns
        assert_eq!(relay(Shape { lanes: 4, block: 128 }, &ws_args(17762, 421, 73448), a).unwrap().unwrap(),
                   Relay { allocation: 421, offset: 0, stride: 73448, columns: 17792, tile: 8 });
        // a workspace sized to the launch's columns exactly, not whole groups: l1 would write past it
        assert!(relay(s1, &ws_args(17762, 5, 73448), |_| Some(17762 * 73448)).is_err());
        assert!(relay(s1, &ws_args(32, 5, 73448), |_| None).is_err());
        let mut bad = ws_args(32, 5, 73448); bad[2] = Argument::Bytes(vec![0; 8]);
        assert!(relay(s1, &bad, |_| Some(1 << 30)).is_err());
        assert!(relay(s1, &ws_args(32, 5, 73444), |_| Some(1 << 30)).is_err());
        // the clear skip covers only a clear of every moved byte
        let r = relay(s1, &ws_args(9062, 197, 73448), a).unwrap().unwrap();
        assert!(r.covered_by((197, 0, 29216 * 73448)));
        assert!(!r.covered_by((197, 0, 9062 * 73448)));
        assert!(!r.covered_by((198, 0, 29216 * 73448)));
        assert!(!r.covered_by((197, 8, 29216 * 73448)));
    }

    #[test]
    fn ptx_is_pinned_and_names_the_entry() {
        assert!(RELAYOUT_PTX.starts_with(".version 7.0\n.target sm_80\n"));
        assert!(RELAYOUT_PTX.contains(".visible .entry ad_mynn_ws_relayout("));
        assert_eq!(RELAYOUT_PTX.matches("\t.param .u").count(), 6);  // dst, src, n_words, col_words, tile_src, tile_dst
    }
}
