// SPDX-License-Identifier: Apache-2.0
//! NVRTC through libloading (the same compiler family the forward uses through CuPy).
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};

type Prog = *mut c_void;

pub fn compile(lib_path: &str, src: &str, name: &str, opts: &[String]) -> Result<(Vec<u8>, String), String> {
    unsafe {
        let lib = libloading::Library::new(lib_path).map_err(|e| format!("load {lib_path}: {e}"))?;
        let create: libloading::Symbol<unsafe extern "C" fn(*mut Prog, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int> =
            lib.get(b"nvrtcCreateProgram").map_err(|e| e.to_string())?;
        let comp: libloading::Symbol<unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> c_int> =
            lib.get(b"nvrtcCompileProgram").map_err(|e| e.to_string())?;
        let logsz: libloading::Symbol<unsafe extern "C" fn(Prog, *mut usize) -> c_int> = lib.get(b"nvrtcGetProgramLogSize").map_err(|e| e.to_string())?;
        let getlog: libloading::Symbol<unsafe extern "C" fn(Prog, *mut c_char) -> c_int> = lib.get(b"nvrtcGetProgramLog").map_err(|e| e.to_string())?;
        let cubsz: libloading::Symbol<unsafe extern "C" fn(Prog, *mut usize) -> c_int> = lib.get(b"nvrtcGetCUBINSize").map_err(|e| e.to_string())?;
        let getcub: libloading::Symbol<unsafe extern "C" fn(Prog, *mut c_char) -> c_int> = lib.get(b"nvrtcGetCUBIN").map_err(|e| e.to_string())?;
        let destroy: libloading::Symbol<unsafe extern "C" fn(*mut Prog) -> c_int> = lib.get(b"nvrtcDestroyProgram").map_err(|e| e.to_string())?;
        let csrc = CString::new(src).unwrap();
        let cname = CString::new(name).unwrap();
        let mut p: Prog = std::ptr::null_mut();
        let r = create(&mut p, csrc.as_ptr(), cname.as_ptr(), 0, std::ptr::null(), std::ptr::null());
        if r != 0 { return Err(format!("nvrtcCreateProgram {r}")); }
        let copts: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
        let popts: Vec<*const c_char> = copts.iter().map(|c| c.as_ptr()).collect();
        let rc = comp(p, popts.len() as c_int, popts.as_ptr());
        let mut n = 0usize;
        logsz(p, &mut n);
        let mut log = vec![0u8; n.max(1)];
        getlog(p, log.as_mut_ptr() as *mut c_char);
        let log = String::from_utf8_lossy(&log).trim_end_matches('\0').to_string();
        if rc != 0 { destroy(&mut p); return Err(format!("nvrtcCompileProgram {rc}\n{log}")); }
        let mut m = 0usize;
        cubsz(p, &mut m);
        let mut cub = vec![0u8; m];
        getcub(p, cub.as_mut_ptr() as *mut c_char);
        destroy(&mut p);
        Ok((cub, log))
    }
}
