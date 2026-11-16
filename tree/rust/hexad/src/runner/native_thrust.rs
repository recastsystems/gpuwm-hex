// SPDX-License-Identifier: Apache-2.0
//! Original one-dimensional Thrust argsort with live replay buffers.
//! The caller must separately prove the live shape, ordering and ownership.
use sha2::{Digest, Sha256};
use std::{ffi::{c_char, c_int, c_void, CString}, fs, os::unix::ffi::OsStrExt, path::Path, ptr};

extern "C" {
    fn binder_thrust_open(path: *const c_char, out: *mut *mut c_void) -> c_int;
    fn binder_thrust_argsort(handle: *mut c_void, dtype: c_int, indices: *mut c_void,
                             input: *mut c_void, count: usize, stream: isize,
                             workspace: *mut c_void, capacity: usize, used: *mut usize,
                             allocations: *mut usize) -> c_int;
    fn binder_thrust_close(handle: *mut c_void);
}

pub struct NativeThrust { handle: *mut c_void }
impl NativeThrust {
    pub unsafe fn open(path: &Path, expected: &str) -> Result<Self, String> {
        super::native_cub::no_python()?;
        if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err("invalid Thrust library SHA-256".into());
        }
        let path = fs::canonicalize(path).map_err(|e| format!("cannot resolve Thrust library: {e}"))?;
        let bytes = fs::read(&path).map_err(|e| format!("cannot read Thrust library: {e}"))?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected { return Err("Thrust library SHA-256 mismatch".into()); }
        let name = CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid Thrust library path")?;
        let mut handle = ptr::null_mut();
        let status = binder_thrust_open(name.as_ptr(), &mut handle);
        if status != 0 || handle.is_null() { return Err(format!("cannot load original Thrust host routine: {status}")); }
        if let Err(error) = super::native_cub::no_python() { binder_thrust_close(handle); return Err(error); }
        Ok(Self { handle })
    }
    pub unsafe fn argsort(&self, dtype: i32, indices: u64, input: u64, count: usize,
                          stream: *mut c_void, workspace: u64, capacity: usize) -> Result<(usize, usize), String> {
        let (mut used, mut allocations) = (0, 0);
        let status = binder_thrust_argsort(self.handle, dtype, indices as *mut c_void, input as *mut c_void,
            count, stream as isize, workspace as *mut c_void, capacity, &mut used, &mut allocations);
        if status != 0 { return Err(format!("original Thrust argsort failed ({status}, workspace {used}/{capacity})")); }
        Ok((used, allocations))
    }
}
impl Drop for NativeThrust { fn drop(&mut self) { unsafe { binder_thrust_close(self.handle) } } }

/// Only the one-dimensional integer signatures observed in this forecast.
pub fn word_bytes(dtype: i32) -> Result<usize, String> {
    match dtype { 4 => Ok(4), 6 => Ok(8), _ => Err("unproved Thrust argsort dtype".into()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_abis_and_library_pins_fail_closed() {
        assert_eq!(word_bytes(4), Ok(4));
        assert_eq!(word_bytes(6), Ok(8));
        assert!(word_bytes(9).is_err());
        assert!(unsafe { NativeThrust::open(Path::new("/dev/null"), &"0".repeat(64)) }.is_err());
    }
}
