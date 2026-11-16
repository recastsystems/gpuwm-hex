// SPDX-License-Identifier: Apache-2.0
//! Interval checks for the binder poison test. This module does not discover
//! allocations and does not certify that its input includes every live extent.
//! The native recorder must supply that complete live set at each boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WordKind { Float, Float64, Integer, IntegerBytes }

impl WordKind {
    pub const fn poison(self) -> u32 {
        match self { Self::Float => 0x7fc0_0000, Self::Float64 => 0,
                    Self::Integer | Self::IntegerBytes => 0xdead_beef }
    }
    pub const fn alignment(self) -> u64 {
        match self { Self::Float64 => 8, Self::IntegerBytes => 1, _ => 4 }
    }
}

/// A word-aligned region of an allocation actually touched by the replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Extent {
    pub allocation: u64,
    pub offset: u64,
    pub bytes: u64,
    pub kind: WordKind,
}

impl Extent {
    fn end(self) -> Result<u64, &'static str> {
        let alignment = self.kind.alignment();
        if self.bytes == 0 || self.offset % alignment != 0 || self.bytes % alignment != 0 {
            return Err("empty or unaligned extent");
        }
        self.offset.checked_add(self.bytes).ok_or("extent end overflow")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoisonWrite { pub extent: Extent, pub word: u32 }

fn ordered(extents: &[Extent]) -> Result<Vec<Extent>, &'static str> {
    let mut result = extents.to_vec();
    for e in &result { e.end()?; }
    result.sort_unstable_by_key(|e| (e.allocation, e.offset));
    for pair in result.windows(2) {
        if pair[0].allocation == pair[1].allocation && pair[0].end()? > pair[1].offset {
            return Err("overlapping extents need prior alias resolution");
        }
    }
    Ok(result)
}

/// Return every word of the supplied live set except its carried sub-extents.
/// Reject ambiguous aliases, a carried region outside the live set, and a
/// conflicting type. A caller must not turn an incomplete live set into proof.
pub fn noncarried_poison(live: &[Extent], carried: &[Extent]) -> Result<Vec<PoisonWrite>, &'static str> {
    let live = ordered(live)?;
    let carried = ordered(carried)?;
    if live.is_empty() || carried.is_empty() { return Err("empty live or carried set"); }
    let mut writes = Vec::new();
    let mut index = 0;
    for region in live {
        let end = region.end()?;
        if index < carried.len() && (carried[index].allocation, carried[index].offset) < (region.allocation, region.offset) {
            return Err("carried extent outside live set");
        }
        let mut cursor = region.offset;
        while index < carried.len() && carried[index].allocation == region.allocation && carried[index].offset < end {
            let keep = carried[index];
            if keep.end()? > end || keep.kind != region.kind { return Err("carried extent exceeds live region or has wrong type"); }
            if keep.offset > cursor {
                let extent = Extent { offset: cursor, bytes: keep.offset - cursor, ..region };
                writes.push(PoisonWrite { extent, word: region.kind.poison() });
            }
            cursor = keep.end()?;
            index += 1;
        }
        if cursor < end {
            let extent = Extent { offset: cursor, bytes: end - cursor, ..region };
            writes.push(PoisonWrite { extent, word: region.kind.poison() });
        }
    }
    if index != carried.len() { return Err("carried extent outside live set"); }
    Ok(writes)
}

/// Produce a one-word bite within an explicitly selected carried extent.
pub fn bite(carried: &[Extent], allocation: u64, offset: u64) -> Result<PoisonWrite, &'static str> {
    let carried = ordered(carried)?;
    let region = carried.into_iter().find(|e| e.allocation == allocation && e.offset <= offset &&
                                               e.end().is_ok_and(|limit| offset < limit))
        .ok_or("bite target is not a carried extent")?;
    let alignment = region.kind.alignment();
    if offset % alignment != 0 { return Err("unaligned bite target"); }
    let available = region.end()? - offset;
    let bytes = match region.kind { WordKind::Float64 => 8, WordKind::IntegerBytes => available.min(4), _ => 4 };
    if bytes > available { return Err("bite target exceeds carried extent"); }
    let extent = Extent { offset, bytes, ..region };
    Ok(PoisonWrite { extent, word: region.kind.poison() })
}

/// Count bitwise different 32-bit words; callers must compare every carried
/// allocation and every required step and reject missing or duplicate IDs.
pub fn differing_words(expected: &[u8], actual: &[u8]) -> Result<u64, &'static str> {
    if expected.len() != actual.len() || expected.len() % 4 != 0 { return Err("incompatible word buffers"); }
    let mut count = 0_u64;
    for (left, right) in expected.chunks_exact(4).zip(actual.chunks_exact(4)) {
        if left != right { count = count.checked_add(1).ok_or("word count overflow")?; }
    }
    Ok(count)
}
