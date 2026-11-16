// SPDX-License-Identifier: Apache-2.0
//! Opt-in live Python window: operation-boundary tape planning, no recorded payloads.
//! The caller restores a complete step start and executes its real Python prefix.
//! All costs include pre AND post saves, padding, rules and conservative margins.

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CommandBytes {
    pub pre: u64,
    pub post: u64,
    pub overhead: u64,
}

/// A suffix [lo,end) fitting one tape. No allocation retry or host spill.
/// Return codes: 0 success, 1 single command too large, 2 invalid arguments,
/// 3 byte-count overflow. On error no output is written.
pub fn tape_range(costs: &[CommandBytes], end: usize, budget: u64)
    -> Result<(usize, u64), i32>
{
    if end > costs.len() || budget == 0 { return Err(2); }
    let (mut lo, mut used) = (end, 0u64);
    while lo > 0 {
        let c = costs[lo - 1];
        let need = c.pre.checked_add(c.post).and_then(|n| n.checked_add(c.overhead)).ok_or(3)?;
        if need > budget {
            if lo == end { return Err(1); }
            break;
        }
        if need > budget - used { break; }
        used += need;
        lo -= 1;
    }
    Ok((lo, used))
}

/// Stateless C interface used by hexcore.live_window.NativeRanges. Costs are
/// host metadata only. The ABI never reads a device pointer or owns model memory.
///
/// # Safety
/// `costs` points to `count` aligned CommandBytes (unless count=0), and out_lo /
/// out_used point to distinct writable u64 words not aliasing costs. Caller keeps
/// these host ranges alive for this synchronous call. No exceptions cross the ABI.
#[no_mangle]
pub unsafe extern "C" fn hexad_live_tape_range(
    costs: *const CommandBytes, count: u64, end: u64, budget: u64,
    out_lo: *mut u64, out_used: *mut u64,
) -> i32 {
    if out_lo.is_null() || out_used.is_null() || end > count || budget == 0
        || (count != 0 && costs.is_null())
        || count > (isize::MAX as u64) / std::mem::size_of::<CommandBytes>() as u64
    { return 2; }
    let rows = if count == 0 { &[] } else { std::slice::from_raw_parts(costs, count as usize) };
    match tape_range(rows, end as usize, budget) {
        Ok((lo, used)) => { *out_lo = lo as u64; *out_used = used; 0 }
        Err(code) => code,
    }
}
