// SPDX-License-Identifier: Apache-2.0
//! Session-owned staging preparation. The worker and every allocation are owned even when opening a window fails.
use std::{ffi::c_void, thread::JoinHandle};

type Alloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type Free = unsafe extern "C" fn(*mut c_void) -> i32;

pub(crate) struct Batch {
    pub buffers: Vec<(usize, usize)>,
    pub failed: usize,
    pub first_error: Option<i32>,
    free: Free,
}
impl Batch {
    unsafe fn allocate(sizes: Vec<usize>, alloc: Alloc, free: Free) -> Self {
        let mut b = Self { buffers: Vec::new(), failed: 0, first_error: None, free };
        for n in sizes.into_iter().filter(|n| *n > 0) {
            let mut h = std::ptr::null_mut();
            let rc = alloc(&mut h, n, 0);
            if rc == 0 { b.buffers.push((h as usize, n)); }
            else { b.failed += 1; b.first_error.get_or_insert(rc); }
        }
        b
    }
}
impl Drop for Batch {
    fn drop(&mut self) {
        for (h, _) in self.buffers.drain(..) { unsafe { (self.free)(h as *mut c_void); } }
    }
}

/// Owned by Execution, which joins this worker before unloading the CUDA driver or destroying its context.
pub(crate) struct Ahead(Option<JoinHandle<Result<Batch, String>>>);
impl Ahead {
    pub unsafe fn start(context: usize, set: Free, alloc: Alloc, free: Free, sizes: Vec<usize>) -> Result<Self, String> {
        std::thread::Builder::new().name("tls-staging-ahead".into()).spawn(move || {
            let rc = set(context as *mut c_void);
            if rc != 0 { return Err(format!("cuCtxSetCurrent staging-ahead: {rc}")); }
            Ok(Batch::allocate(sizes, alloc, free))
        }).map(|h| Self(Some(h))).map_err(|e| format!("staging-ahead worker: {e}"))
    }
    pub fn join(mut self) -> Result<Batch, String> {
        self.0.take().unwrap().join().map_err(|_| "staging-ahead worker panicked".to_owned())?
    }
}
impl Drop for Ahead {
    fn drop(&mut self) {
        // A failed session open still waits for its worker and frees any completed allocations.
        if let Some(h) = self.0.take() { let _ = h.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, atomic::{AtomicUsize, Ordering::SeqCst}};
    static LOCK: Mutex<()> = Mutex::new(());
    static FREED: AtomicUsize = AtomicUsize::new(0);
    unsafe extern "C" fn alloc(p: *mut *mut c_void, n: usize, _: u32) -> i32 {
        if n == 13 { return 2; }
        *p = n as *mut c_void;
        0
    }
    unsafe extern "C" fn free(p: *mut c_void) -> i32 { FREED.fetch_add(p as usize, SeqCst); 0 }
    unsafe extern "C" fn set(_: *mut c_void) -> i32 { 0 }
    #[test]
    fn failed_allocations_and_abandoned_worker_release_owned_buffers() {
        let _lock = LOCK.lock().unwrap();
        FREED.store(0, SeqCst);
        let a = unsafe { Ahead::start(1, set, alloc, free, vec![0, 4, 13, 8]).unwrap() };
        let b = a.join().unwrap();
        assert_eq!(b.buffers, vec![(4, 4), (8, 8)]);
        assert_eq!((b.failed, b.first_error), (1, Some(2)));
        drop(b);
        assert_eq!(FREED.load(SeqCst), 12);
        drop(unsafe { Ahead::start(1, set, alloc, free, vec![16]).unwrap() });
        assert_eq!(FREED.load(SeqCst), 28);
    }
    #[test]
    fn adoption_transfers_ownership_without_freeing_buffers() {
        let _lock = LOCK.lock().unwrap();
        FREED.store(0, SeqCst);
        let mut b = unsafe { Batch::allocate(vec![4, 8], alloc, free) };
        let buffers = std::mem::take(&mut b.buffers);
        drop(b);
        assert_eq!(FREED.load(SeqCst), 0);
        for (p, _) in buffers { unsafe { free(p as *mut c_void); } }
        assert_eq!(FREED.load(SeqCst), 12);
    }
}
