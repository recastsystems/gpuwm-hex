// SPDX-License-Identifier: Apache-2.0
//! Pure owner transitions. The CPU C stub translates this file mechanically, line for line.
//! refs=0 is a tombstone: generation cannot be resurrected. CONST keeps one session pin.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Owner {
    pub id: u64, pub generation: u64, pub address: u64, pub bytes: u64,
    pub shadow: u64, pub refs: u64, pub constant: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct View {
    pub id: u64, pub generation: u64, pub offset: u64, pub itemsize: u64,
    pub rank: u64, pub shape: [u64; 8], pub strides: [i64; 8],
}
// BEGIN SHARED: only fixed-width scalar arithmetic; no allocation or device calls.
pub fn valid(n: &Owner) -> i32 {
    if n.id == 0 || n.generation == 0 || n.address == 0 || n.bytes == 0 || n.constant > 1 { return -1; }
    if n.bytes > u64::MAX - n.address { return -1; }
    return 0;
}
pub fn admit(o: &Owner, n: &Owner) -> i32 {
    if o.refs != 0 {
        if o.generation == n.generation && o.address == n.address && o.bytes == n.bytes && o.constant == n.constant { return 1; }
        return -2;
    }
    if n.generation <= o.generation { return -2; }
    return 0;
}
pub fn overlaps(o: &Owner, n: &Owner) -> bool {
    return o.refs != 0 && o.address < n.address + n.bytes && n.address < o.address + o.bytes;
}
pub fn resolve(o: &Owner, v: &View) -> i32 {
    if o.refs == 0 || o.id != v.id || o.generation != v.generation { return -2; }
    if v.rank > 8 || v.itemsize == 0 || v.offset >= o.bytes { return -1; }
    let mut lo: u64 = v.offset;
    let mut hi: u64 = v.offset;
    let mut i: usize = 0;
    while i < v.rank as usize {
        if v.shape[i] == 0 || v.strides[i] == i64::MIN { return -1; }
        let mut stride: u64 = v.strides[i] as u64;
        if v.strides[i] < 0 { stride = (-v.strides[i]) as u64; }
        let steps: u64 = v.shape[i] - 1;
        if stride != 0 && steps > u64::MAX / stride { return -1; }
        let delta: u64 = steps * stride;
        if v.strides[i] < 0 {
            if delta > lo { return -1; }
            lo -= delta;
        } else {
            if delta > o.bytes - hi { return -1; }
            hi += delta;
        }
        i += 1;
    }
    if v.itemsize > o.bytes - hi { return -1; }
    return 0;
}
pub fn pin(o: &mut Owner) -> i32 {
    if o.refs == 0 || o.refs == u64::MAX { return -2; }
    o.refs += 1;
    return 0;
}
pub fn release(o: &mut Owner) -> i32 {
    if o.refs == 0 { return -2; }
    if o.constant == 1 && o.refs == 1 { return 1; }
    o.refs -= 1;
    if o.refs == 0 { return 0; }
    return 1;
}
// END SHARED
