// SPDX-License-Identifier: Apache-2.0
//! Build time only: NVRTC compile of the generated whole-launch unit with the forward image's own options.
use std::ffi::{c_char, c_int, c_void, CStr, CString};

extern "C" {
    fn dlopen(f: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
type NvCreate = unsafe extern "C" fn(*mut *mut c_void, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int;
type NvCompile = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
type NvSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
type NvGet = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;

fn sym<T>(h: *mut c_void, s: &str) -> T {
    let c = CString::new(s).unwrap();
    let p = unsafe { dlsym(h, c.as_ptr()) };
    assert!(!p.is_null(), "missing NVRTC symbol {}", s);
    unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) }
}

/// Returns (cubin, log). Panics with the log on a compile error.
pub fn compile(lib: &str, src: &[u8], name: &str, opts: &[String]) -> (Vec<u8>, String) {
    let c = CString::new(lib).unwrap();
    let h = unsafe { dlopen(c.as_ptr(), 2) };
    if h.is_null() {
        panic!("dlopen {}: {}", lib, unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy());
    }
    let create: NvCreate = sym(h, "nvrtcCreateProgram");
    let comp: NvCompile = sym(h, "nvrtcCompileProgram");
    let lsz: NvSize = sym(h, "nvrtcGetProgramLogSize");
    let lget: NvGet = sym(h, "nvrtcGetProgramLog");
    let csz: NvSize = sym(h, "nvrtcGetCUBINSize");
    let cget: NvGet = sym(h, "nvrtcGetCUBIN");
    let mut p: *mut c_void = std::ptr::null_mut();
    let s = CString::new(src.to_vec()).unwrap();
    let n = CString::new(name).unwrap();
    assert_eq!(unsafe { create(&mut p, s.as_ptr(), n.as_ptr(), 0, std::ptr::null(), std::ptr::null()) }, 0);
    let co: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
    let cp: Vec<*const c_char> = co.iter().map(|o| o.as_ptr()).collect();
    let rc = unsafe { comp(p, cp.len() as c_int, cp.as_ptr()) };
    let mut ls = 0usize;
    unsafe { lsz(p, &mut ls) };
    let mut lb = vec![0u8; ls.max(1)];
    unsafe { lget(p, lb.as_mut_ptr() as *mut c_char) };
    let log = String::from_utf8_lossy(&lb).trim_end_matches('\0').to_string();
    if rc != 0 { panic!("NVRTC compile failed ({}):\n{}", rc, log); }
    let mut cs = 0usize;
    unsafe { csz(p, &mut cs) };
    let mut cb = vec![0u8; cs];
    unsafe { cget(p, cb.as_mut_ptr() as *mut c_char) };
    (cb, log)
}
