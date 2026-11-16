// SPDX-License-Identifier: Apache-2.0
//! Minimal CUDA driver + NVRTC access for the P2 proofs, loaded with dlopen (no build-time link).
#![allow(dead_code)]
use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr::null_mut;

extern "C" {
    fn dlopen(name: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
}

macro_rules! sym {
    ($h:expr, $name:literal, $t:ty) => {{
        let n = CString::new($name).unwrap();
        let p = unsafe { dlsym($h, n.as_ptr()) };
        if p.is_null() { return Err(format!("missing symbol {}", $name)); }
        unsafe { std::mem::transmute::<*mut c_void, $t>(p) }
    }};
}

type CuRes = c_int;
pub struct Cuda {
    init: unsafe extern "C" fn(u32) -> CuRes,
    dev_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuRes,
    attr: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuRes,
    ctx_create: unsafe extern "C" fn(*mut *mut c_void, u32, c_int) -> CuRes,
    sync: unsafe extern "C" fn() -> CuRes,
    mod_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CuRes,
    get_fn: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuRes,
    malloc: unsafe extern "C" fn(*mut u64, usize) -> CuRes,
    free: unsafe extern "C" fn(u64) -> CuRes,
    memset: unsafe extern "C" fn(u64, u8, usize) -> CuRes,
    htod: unsafe extern "C" fn(u64, *const c_void, usize) -> CuRes,
    dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> CuRes,
    launch: unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CuRes,
    set_limit: unsafe extern "C" fn(c_int, usize) -> CuRes,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuRes,
    pub arch: (i32, i32),
}

fn ck(rc: c_int, what: &str) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(format!("{what}: CUDA error {rc}")) }
}

impl Cuda {
    pub fn new() -> Result<Cuda, String> {
        let lib = CString::new("libcuda.so.1").unwrap();
        let h = unsafe { dlopen(lib.as_ptr(), 2) };
        if h.is_null() { return Err("dlopen libcuda.so.1".into()); }
        let c = Cuda {
            init: sym!(h, "cuInit", _), dev_get: sym!(h, "cuDeviceGet", _), attr: sym!(h, "cuDeviceGetAttribute", _),
            ctx_create: sym!(h, "cuCtxCreate_v2", _), sync: sym!(h, "cuCtxSynchronize", _),
            mod_load: sym!(h, "cuModuleLoadData", _), get_fn: sym!(h, "cuModuleGetFunction", _),
            malloc: sym!(h, "cuMemAlloc_v2", _), free: sym!(h, "cuMemFree_v2", _), memset: sym!(h, "cuMemsetD8_v2", _),
            htod: sym!(h, "cuMemcpyHtoD_v2", _), dtoh: sym!(h, "cuMemcpyDtoH_v2", _), launch: sym!(h, "cuLaunchKernel", _),
            set_limit: sym!(h, "cuCtxSetLimit", _), mem_info: sym!(h, "cuMemGetInfo_v2", _), arch: (0, 0),
        };
        let mut c = c;
        unsafe {
            ck((c.init)(0), "cuInit")?;
            let mut d = 0;
            ck((c.dev_get)(&mut d, 0), "cuDeviceGet")?;
            let (mut ma, mut mi) = (0, 0);
            ck((c.attr)(&mut ma, 75, d), "cc major")?;
            ck((c.attr)(&mut mi, 76, d), "cc minor")?;
            c.arch = (ma, mi);
            let mut ctx = null_mut();
            ck((c.ctx_create)(&mut ctx, 0, d), "cuCtxCreate")?;
        }
        Ok(c)
    }
    pub fn stack_limit(&self, bytes: usize) -> Result<(), String> { ck(unsafe { (self.set_limit)(0, bytes) }, "stack limit") }
    pub fn free_mem(&self) -> (usize, usize) { let (mut f, mut t) = (0, 0); unsafe { (self.mem_info)(&mut f, &mut t) }; (f, t) }
    pub fn load(&self, image: &[u8]) -> Result<*mut c_void, String> {
        let mut m = null_mut();
        ck(unsafe { (self.mod_load)(&mut m, image.as_ptr().cast()) }, "cuModuleLoadData")?;
        Ok(m)
    }
    pub fn func(&self, m: *mut c_void, name: &str) -> Result<*mut c_void, String> {
        let n = CString::new(name).unwrap();
        let mut f = null_mut();
        ck(unsafe { (self.get_fn)(&mut f, m, n.as_ptr()) }, &format!("function {name}"))?;
        Ok(f)
    }
    pub fn alloc(&self, n: usize) -> Result<u64, String> {
        let mut p = 0;
        ck(unsafe { (self.malloc)(&mut p, n.max(16)) }, &format!("alloc {n}"))?;
        ck(unsafe { (self.memset)(p, 0, n.max(16)) }, "memset")?;
        Ok(p)
    }
    pub fn zero(&self, p: u64, n: usize) -> Result<(), String> { ck(unsafe { (self.memset)(p, 0, n) }, "memset") }
    pub fn release(&self, p: u64) { unsafe { (self.free)(p) }; }
    pub fn put<T: Copy>(&self, p: u64, v: &[T]) -> Result<(), String> {
        ck(unsafe { (self.htod)(p, v.as_ptr().cast(), std::mem::size_of_val(v)) }, "HtoD")
    }
    pub fn get<T: Copy + Default>(&self, p: u64, n: usize) -> Result<Vec<T>, String> {
        self.synchronize()?;
        let mut v = vec![T::default(); n];
        ck(unsafe { (self.dtoh)(v.as_mut_ptr().cast(), p, n * std::mem::size_of::<T>()) }, "DtoH")?;
        Ok(v)
    }
    pub fn get_bytes(&self, p: u64, n: usize) -> Result<Vec<u8>, String> { self.get::<u8>(p, n) }
    pub fn synchronize(&self) -> Result<(), String> { ck(unsafe { (self.sync)() }, "cuCtxSynchronize") }
    /// Launch with argument pointers built from a list of 8-byte or 4-byte values.
    pub fn run(&self, f: *mut c_void, grid: u32, block: u32, args: &mut [Arg]) -> Result<(), String> {
        let mut ptrs: Vec<*mut c_void> = args.iter_mut().map(|a| a.ptr()).collect();
        ck(unsafe { (self.launch)(f, grid, 1, 1, block, 1, 1, 0, null_mut(), ptrs.as_mut_ptr(), null_mut()) }, "launch")
    }
}

pub enum Arg { P(u64), I(i32), F(f32) }
impl Arg {
    fn ptr(&mut self) -> *mut c_void {
        match self { Arg::P(v) => (v as *mut u64).cast(), Arg::I(v) => (v as *mut i32).cast(), Arg::F(v) => (v as *mut f32).cast() }
    }
}

type NvRes = c_int;
pub struct Nvrtc {
    create: unsafe extern "C" fn(*mut *mut c_void, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> NvRes,
    compile: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> NvRes,
    log_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> NvRes,
    log: unsafe extern "C" fn(*mut c_void, *mut c_char) -> NvRes,
    cubin_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> NvRes,
    cubin: unsafe extern "C" fn(*mut c_void, *mut c_char) -> NvRes,
}

impl Nvrtc {
    pub fn new(path: &str) -> Result<Nvrtc, String> {
        let lib = CString::new(path).unwrap();
        let h = unsafe { dlopen(lib.as_ptr(), 2) };
        if h.is_null() { return Err(format!("dlopen {path}")); }
        Ok(Nvrtc {
            create: sym!(h, "nvrtcCreateProgram", _), compile: sym!(h, "nvrtcCompileProgram", _),
            log_size: sym!(h, "nvrtcGetProgramLogSize", _), log: sym!(h, "nvrtcGetProgramLog", _),
            cubin_size: sym!(h, "nvrtcGetCUBINSize", _), cubin: sym!(h, "nvrtcGetCUBIN", _),
        })
    }
    /// Compile `src` (named `name`) with `opts`; returns (cubin, log).
    pub fn build(&self, src: &str, name: &str, opts: &[String]) -> Result<(Vec<u8>, String), String> {
        let s = CString::new(src).map_err(|_| "NUL in source")?;
        let n = CString::new(name).unwrap();
        let mut p = null_mut();
        if unsafe { (self.create)(&mut p, s.as_ptr(), n.as_ptr(), 0, std::ptr::null(), std::ptr::null()) } != 0 {
            return Err("nvrtcCreateProgram".into());
        }
        let co: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
        let cp: Vec<*const c_char> = co.iter().map(|c| c.as_ptr()).collect();
        let rc = unsafe { (self.compile)(p, cp.len() as c_int, cp.as_ptr()) };
        let mut ls = 0usize;
        unsafe { (self.log_size)(p, &mut ls) };
        let mut lb = vec![0u8; ls.max(1)];
        unsafe { (self.log)(p, lb.as_mut_ptr().cast()) };
        let log = String::from_utf8_lossy(&lb).trim_end_matches('\0').to_string();
        if rc != 0 { return Err(format!("NVRTC {rc} building {name}:\n{log}")); }
        let mut cs = 0usize;
        unsafe { (self.cubin_size)(p, &mut cs) };
        let mut cb = vec![0u8; cs];
        if unsafe { (self.cubin)(p, cb.as_mut_ptr().cast()) } != 0 { return Err("nvrtcGetCUBIN".into()); }
        Ok((cb, log))
    }
}
