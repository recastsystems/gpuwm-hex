// SPDX-License-Identifier: Apache-2.0
//! Calls the SHA-pinned original CUB host routine with replay-owned buffers.
//! This is a building block; a caller must still prove every live binding and
//! the single-stream ordering against the ordinary physics forecast.
use sha2::{Digest, Sha256};
use std::{ffi::{c_char, c_int, c_void, CString}, fs, os::unix::ffi::OsStrExt,
          path::Path, ptr};

extern "C" {
    fn binder_cub_open(path: *const c_char, out: *mut *mut c_void) -> c_int;
    fn binder_cub_reduce(handle: *mut c_void, workspace: *mut c_void, capacity: usize,
                         input: *mut c_void, output: *mut c_void, count: c_int,
                         stream: *mut c_void, op: c_int, dtype: c_int, required: *mut usize) -> c_int;
    fn binder_cub_segmented(handle: *mut c_void, workspace: *mut c_void, capacity: usize,
                            input: *mut c_void, output: *mut c_void, segments: c_int, segment_size: c_int,
                            stream: *mut c_void, op: c_int, dtype: c_int, required: *mut usize) -> c_int;
    fn binder_cub_close(handle: *mut c_void);
}

/// The symbols the pinned CuPy CUB/Thrust extensions import from Python and CuPy. This runner supplies inert copies
/// (native_cub_bridge.cc / native_thrust_bridge.cc weak definitions in the ad-binder executable, capi_exports.rs strong
/// exports in the libhexad-phys cdylib); no Python code ever runs.
const STUBBED: &[&str] = &[
    "PyExc_ArithmeticError", "PyExc_AssertionError", "PyExc_AttributeError", "PyExc_DeprecationWarning", "PyExc_IOError",
    "PyExc_ImportError", "PyExc_IndexError", "PyExc_MemoryError", "PyExc_NameError", "PyExc_NotImplementedError",
    "PyExc_OverflowError", "PyExc_RuntimeError", "PyExc_StopIteration", "PyExc_SystemError", "PyExc_TypeError",
    "PyExc_UnboundLocalError", "PyExc_ValueError", "PyExc_ZeroDivisionError", "PyBaseObject_Type", "PyCFunction_Type",
    "PyDict_Type", "PyFloat_Type", "PyList_Type", "PyLong_Type", "PyMethod_Type", "PyRange_Type", "PyTuple_Type",
    "PyUnicode_Type", "_Py_FalseStruct", "_Py_NoneStruct", "_Py_TrueStruct", "Py_Version", "PyObject_GenericGetAttr",
    "cupy_malloc", "cupy_free",
];

#[repr(C)]
struct DlInfo { fname: *const c_char, fbase: *mut c_void, sname: *const c_char, saddr: *mut c_void }

#[link(name = "dl")]
extern "C" {
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
}

/// Loading the pinned CuPy extension is safe when its Python and CuPy imports bind to this runner's inert copies and no
/// interpreter runs. A process without libpython satisfies that by construction (the original rule). A host that maps
/// libpython without running it (a host whose SDK library has DT_NEEDED libpython3.11) is admitted only
/// if Py_IsInitialized() is 0 and every stubbed symbol resolves, in the global scope the extension's RTLD_LOCAL dlopen
/// searches first, to the object this function lives in; otherwise it is refused, naming the symbol and its object.
pub(super) fn no_python() -> Result<(), String> {
    let maps = fs::read_to_string("/proc/self/maps").map_err(|e| format!("cannot inspect process maps: {e}"))?;
    if !maps.lines().any(|line| line.contains("libpython")) {
        return Ok(());
    }
    const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();
    unsafe {
        let init = dlsym(RTLD_DEFAULT, c"Py_IsInitialized".as_ptr());
        if !init.is_null() {
            let initialised: extern "C" fn() -> c_int = std::mem::transmute(init);
            if initialised() != 0 {
                return Err("libpython is mapped and its interpreter is initialised in the native replay process".into());
            }
        }
        let mut own = DlInfo { fname: ptr::null(), fbase: ptr::null_mut(), sname: ptr::null(), saddr: ptr::null_mut() };
        if dladdr(no_python as *const c_void, &mut own) == 0 || own.fbase.is_null() {
            return Err("libpython is mapped and the runner cannot locate its own object (dladdr)".into());
        }
        for name in STUBBED {
            let c = CString::new(*name).map_err(|e| e.to_string())?;
            let p = dlsym(RTLD_DEFAULT, c.as_ptr());
            if p.is_null() {
                return Err(format!("libpython is mapped and {name} is not in the global scope (the runner's inert copy is not visible: load libhexad-phys globally)"));
            }
            let mut info = DlInfo { fname: ptr::null(), fbase: ptr::null_mut(), sname: ptr::null(), saddr: ptr::null_mut() };
            if dladdr(p, &mut info) == 0 || info.fbase != own.fbase {
                let from = if info.fname.is_null() { "an unknown object".to_string() }
                           else { std::ffi::CStr::from_ptr(info.fname).to_string_lossy().into_owned() };
                return Err(format!("libpython is mapped and {name} resolves to {from}, not the runner's inert copy"));
            }
        }
    }
    Ok(())
}

/// A verified host binary, retained until all work submitted from it completes.
pub struct NativeCub { handle: *mut c_void }

impl NativeCub {
    pub unsafe fn open(path: &Path, expected_sha256: &str) -> Result<Self, String> {
        no_python()?;
        if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err("invalid CUB library SHA-256".into());
        }
        let path = fs::canonicalize(path).map_err(|e| format!("cannot resolve CUB library: {e}"))?;
        let bytes = fs::read(&path).map_err(|e| format!("cannot read CUB library: {e}"))?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
            return Err("CUB library SHA-256 mismatch".into());
        }
        let name = CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid CUB library path")?;
        let mut handle = ptr::null_mut();
        let status = binder_cub_open(name.as_ptr(), &mut handle);
        if status != 0 || handle.is_null() { return Err(format!("cannot load original CUB host routine: {status}")); }
        if let Err(error) = no_python() { binder_cub_close(handle); return Err(error); }
        Ok(Self { handle })
    }

    pub unsafe fn reduce(&self, workspace: u64, capacity: usize, input: u64, output: u64,
                         count: usize, stream: *mut c_void, op: i32, dtype: i32) -> Result<usize, String> {
        let count = i32::try_from(count).map_err(|_| "CUB count exceeds int32")?;
        let mut required = 0;
        let status = binder_cub_reduce(self.handle, workspace as *mut c_void, capacity,
            input as *mut c_void, output as *mut c_void, count, stream, op, dtype, &mut required);
        if status != 0 { return Err(format!("original CUB reduce failed ({status}, workspace {required}/{capacity})")); }
        Ok(required)
    }

    pub unsafe fn segmented(&self, workspace: u64, capacity: usize, input: u64, output: u64,
                            segments: usize, segment_size: usize, stream: *mut c_void,
                            op: i32, dtype: i32) -> Result<usize, String> {
        let segments = i32::try_from(segments).map_err(|_| "CUB segment count exceeds int32")?;
        let segment_size = i32::try_from(segment_size).map_err(|_| "CUB segment size exceeds int32")?;
        let mut required = 0;
        let status = binder_cub_segmented(self.handle, workspace as *mut c_void, capacity,
            input as *mut c_void, output as *mut c_void, segments, segment_size, stream, op, dtype, &mut required);
        if status != 0 { return Err(format!("original CUB segmented reduce failed ({status}, workspace {required}/{capacity})")); }
        Ok(required)
    }
}

impl Drop for NativeCub {
    fn drop(&mut self) { unsafe { binder_cub_close(self.handle); } }
}

/// Only the reduction ABIs observed in the physics host calls are admitted.
/// The CuPy dispatcher defines 0=sum, 1=min, 2=max; 4=int32 and 6=int64.
/// Argmin/argmax have a different output layout and must not enter this path.
pub fn word_bytes(op: i32, dtype: i32) -> Result<usize, String> {
    if !matches!(op, 0..=2) { return Err("unproved CUB operation or dtype".into()); }
    match dtype {
        4 => Ok(4),
        6 => Ok(8),
        _ => Err("unproved CUB operation or dtype".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_cub_abis_fail_closed() {
        assert_eq!(word_bytes(2, 4), Ok(4));
        assert_eq!(word_bytes(0, 4), Ok(4));
        assert_eq!(word_bytes(1, 6), Ok(8));
        assert_eq!(word_bytes(2, 6), Ok(8));
        assert!(word_bytes(2, 9).is_err());
        assert!(word_bytes(3, 4).is_err());
    }
    #[test]
    fn library_must_match_pin_before_loading() {
        assert!(unsafe { NativeCub::open(Path::new("/dev/null"), &"0".repeat(64)) }.is_err());
        assert!(unsafe { NativeCub::open(Path::new("/dev/null"), "bad") }.is_err());
    }
}
