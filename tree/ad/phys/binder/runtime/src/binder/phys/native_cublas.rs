// SPDX-License-Identifier: Apache-2.0
//! Call the original, SHA-pinned cuBLAS SGEAM routine for a live transpose.
//! The plan producer must prove the input geometry, ownership, selected branch,
//! dependency versions and ordering against the ordinary forecast.
use sha2::{Digest, Sha256};
use std::{ffi::{c_char, c_int, c_void, CString}, fs::{self, File}, io::Read,
          mem, os::unix::ffi::OsStrExt, path::Path, ptr};

#[link(name = "dl")]
extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

type Create = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type SetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type SetPointerMode = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type Geam = unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, c_int,
    *const f32, *const c_void, c_int, *const f32, *const c_void, c_int, *mut c_void, c_int) -> c_int;

unsafe fn symbol<T: Copy>(library: *mut c_void, name: &str) -> Result<T, String> {
    let name = CString::new(name).map_err(|e| e.to_string())?;
    let pointer = dlsym(library, name.as_ptr());
    if pointer.is_null() || mem::size_of::<T>() != mem::size_of_val(&pointer) {
        return Err(format!("missing or incompatible cuBLAS symbol {name:?}"));
    }
    Ok(mem::transmute_copy(&pointer))
}

pub fn matrix_bytes(m: usize, n: usize) -> Result<usize, String> {
    if m == 0 || n == 0 || m > i32::MAX as usize || n > i32::MAX as usize {
        return Err("invalid cuBLAS transpose dimensions".into());
    }
    m.checked_mul(n).and_then(|x| x.checked_mul(4)).ok_or_else(|| "cuBLAS transpose extent overflow".into())
}

pub struct NativeCublas {
    library: *mut c_void,
    handle: *mut c_void,
    destroy: Destroy,
    set_stream: SetStream,
    geam: Geam,
}

impl NativeCublas {
    pub unsafe fn open(path: &Path, expected_sha256: &str) -> Result<Self, String> {
        super::native_cub::no_python()?;
        if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
            return Err("invalid cuBLAS library SHA-256".into());
        }
        let path = fs::canonicalize(path).map_err(|e| format!("cannot resolve cuBLAS library: {e}"))?;
        let mut input = File::open(&path).map_err(|e| format!("cannot read cuBLAS library: {e}"))?;
        let mut hash = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = input.read(&mut buf).map_err(|e| format!("cannot hash cuBLAS library: {e}"))?;
            if n == 0 { break; }
            hash.update(&buf[..n]);
        }
        if format!("{:x}", hash.finalize()) != expected_sha256 {
            return Err("cuBLAS library SHA-256 mismatch".into());
        }
        let name = CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid cuBLAS library path")?;
        let library = dlopen(name.as_ptr(), 2);
        if library.is_null() { return Err("cannot load pinned cuBLAS library".into()); }
        let result = (|| {
            let create: Create = symbol(library, "cublasCreate_v2")?;
            let destroy: Destroy = symbol(library, "cublasDestroy_v2")?;
            let set_stream: SetStream = symbol(library, "cublasSetStream_v2")?;
            let set_mode: SetPointerMode = symbol(library, "cublasSetPointerMode_v2")?;
            let geam: Geam = symbol(library, "cublasSgeam")?;
            let mut handle = ptr::null_mut();
            let status = create(&mut handle);
            if status != 0 || handle.is_null() { return Err(format!("cublasCreate_v2 failed: {status}")); }
            let status = set_mode(handle, 0); // CUBLAS_POINTER_MODE_HOST
            if status != 0 { destroy(handle); return Err(format!("cublasSetPointerMode_v2 failed: {status}")); }
            if let Err(error) = super::native_cub::no_python() { destroy(handle); return Err(error); }
            Ok(Self { library, handle, destroy, set_stream, geam })
        })();
        if result.is_err() { dlclose(library); }
        result
    }

    /// The observed CuPy copy branch is an SGEAM transpose with B=A,
    /// alpha=1 and beta=0. These values are the operation's identity and zero,
    /// not a cached forecast coefficient. Dimensions and addresses are live.
    pub unsafe fn transpose(&self, input: u64, output: u64, m: usize, n: usize,
                            stream: *mut c_void) -> Result<(), String> {
        let bytes = matrix_bytes(m, n)?;
        let input_end = input.checked_add(bytes as u64).ok_or("cuBLAS input range overflow")?;
        let output_end = output.checked_add(bytes as u64).ok_or("cuBLAS output range overflow")?;
        if input == 0 || output == 0 || (input < output_end && output < input_end) {
            return Err("invalid or overlapping cuBLAS transpose buffers".into());
        }
        let status = (self.set_stream)(self.handle, stream);
        if status != 0 { return Err(format!("cublasSetStream_v2 failed: {status}")); }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = (self.geam)(self.handle, 1, 1, m as i32, n as i32, &alpha,
            input as *const c_void, n as i32, &beta, input as *const c_void, n as i32,
            output as *mut c_void, m as i32);
        if status != 0 { return Err(format!("original cublasSgeam failed: {status}")); }
        Ok(())
    }
}

impl Drop for NativeCublas {
    fn drop(&mut self) { unsafe { (self.destroy)(self.handle); dlclose(self.library); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_dimensions_and_library_pin() {
        assert_eq!(matrix_bytes(12, 2562).unwrap(), 122976);
        assert!(matrix_bytes(0, 12).is_err());
        assert!(matrix_bytes(usize::MAX, 2).is_err());
        assert!(unsafe { NativeCublas::open(Path::new("/dev/null"), &"0".repeat(64)) }.is_err());
    }
}
