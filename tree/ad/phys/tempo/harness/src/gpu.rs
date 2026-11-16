// SPDX-License-Identifier: Apache-2.0
//! NVRTC (dlopen) and CUDA driver API (libcuda) bindings, nothing else.
#![allow(non_camel_case_types, dead_code)]
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;

extern "C" {
    fn dlopen(f: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
fn sym<T>(h: *mut c_void, s: &str) -> T {
    let c = CString::new(s).unwrap();
    let p = unsafe { dlsym(h, c.as_ptr()) };
    assert!(!p.is_null(), "missing symbol {s}");
    unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) }
}
type NvCreate = unsafe extern "C" fn(*mut *mut c_void, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int;
type NvCompile = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
type NvSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
type NvGet = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
type NvDestroy = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type NvVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
pub struct Nvrtc { create: NvCreate, compile: NvCompile, log_size: NvSize, log: NvGet, cubin_size: NvSize, cubin: NvGet, destroy: NvDestroy, pub version: (i32, i32) }
impl Nvrtc {
    pub fn open(path: &str) -> Nvrtc {
        let c = CString::new(path).unwrap();
        let h = unsafe { dlopen(c.as_ptr(), 2) };
        if h.is_null() { panic!("dlopen {path}: {}", unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy()); }
        let ver: NvVersion = sym(h, "nvrtcVersion");
        let (mut a, mut b) = (0, 0);
        unsafe { ver(&mut a, &mut b) };
        Nvrtc { create: sym(h, "nvrtcCreateProgram"), compile: sym(h, "nvrtcCompileProgram"), log_size: sym(h, "nvrtcGetProgramLogSize"),
                log: sym(h, "nvrtcGetProgramLog"), cubin_size: sym(h, "nvrtcGetCUBINSize"), cubin: sym(h, "nvrtcGetCUBIN"),
                destroy: sym(h, "nvrtcDestroyProgram"), version: (a, b) }
    }
    pub fn build(&self, src: &str, opts: &[String]) -> Result<(Vec<u8>, String), String> {
        let s = CString::new(src).unwrap();
        let copts: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
        let popts: Vec<*const c_char> = copts.iter().map(|c| c.as_ptr()).collect();
        let mut prog = ptr::null_mut();
        let name = CString::new("tempo_ad.cu").unwrap();
        if unsafe { (self.create)(&mut prog, s.as_ptr(), name.as_ptr(), 0, ptr::null(), ptr::null()) } != 0 { return Err("nvrtcCreateProgram".into()); }
        let rc = unsafe { (self.compile)(prog, popts.len() as c_int, popts.as_ptr()) };
        let mut n = 0usize;
        unsafe { (self.log_size)(prog, &mut n) };
        let mut lg = vec![0u8; n.max(1)];
        unsafe { (self.log)(prog, lg.as_mut_ptr().cast()) };
        let log = String::from_utf8_lossy(&lg).trim_end_matches('\0').to_string();
        if rc != 0 { unsafe { (self.destroy)(&mut prog) }; return Err(format!("NVRTC rc={rc}:\n{log}")); }
        unsafe { (self.cubin_size)(prog, &mut n) };
        let mut out = vec![0u8; n];
        unsafe { (self.cubin)(prog, out.as_mut_ptr().cast()) };
        unsafe { (self.destroy)(&mut prog) };
        Ok((out, log))
    }
}

pub type CUdeviceptr = u64;
#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> c_int;
    fn cuDeviceGet(dev: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDevicePrimaryCtxRetain(ctx: *mut *mut c_void, dev: c_int) -> c_int;
    fn cuDevicePrimaryCtxSetFlags_v2(dev: c_int, flags: c_uint) -> c_int;
    fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
    fn cuCtxSetLimit(limit: c_int, value: usize) -> c_int;
    fn cuModuleLoadData(m: *mut *mut c_void, image: *const c_void) -> c_int;
    fn cuModuleGetFunction(f: *mut *mut c_void, m: *mut c_void, name: *const c_char) -> c_int;
    fn cuModuleGetGlobal_v2(p: *mut CUdeviceptr, bytes: *mut usize, m: *mut c_void, name: *const c_char) -> c_int;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> c_int;
    fn cuMemFree_v2(p: CUdeviceptr) -> c_int;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> c_int;
    fn cuMemsetD8_v2(dst: CUdeviceptr, v: u8, n: usize) -> c_int;
    fn cuLaunchKernel(f: *mut c_void, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
                      shared: c_uint, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuGetErrorString(r: c_int, s: *mut *const c_char) -> c_int;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
    fn cuFuncGetAttribute(v: *mut c_int, attr: c_int, f: *mut c_void) -> c_int;
}
pub fn check(r: c_int, what: &str) {
    if r != 0 {
        let mut s: *const c_char = ptr::null();
        unsafe { cuGetErrorString(r, &mut s) };
        let msg = if s.is_null() { "?".into() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
        panic!("CUDA error {r} ({msg}) in {what}");
    }
}
pub fn init() {
    unsafe {
        check(cuInit(0), "cuInit");
        let mut d = 0; check(cuDeviceGet(&mut d, 0), "device");
        // Blocking sync: the host thread sleeps while the card works instead of spinning one core at 100 %.
        check(cuDevicePrimaryCtxSetFlags_v2(d, 4), "ctx flags");
        let mut ctx = ptr::null_mut(); check(cuDevicePrimaryCtxRetain(&mut ctx, d), "ctx"); check(cuCtxSetCurrent(ctx), "set");
        // Call stack per thread for the non-inlined active column code (AD_STACK bytes if set; otherwise the driver's own sizing);
        // the driver reserves it for every resident thread, so it is kept as small as the code allows.
        if let Some(st) = std::env::var("AD_STACK").ok().and_then(|v| v.parse::<usize>().ok()) { check(cuCtxSetLimit(0, st), "stack limit"); }
    }
}
pub fn mem_info() -> (usize, usize) { let (mut f, mut t) = (0, 0); unsafe { check(cuMemGetInfo_v2(&mut f, &mut t), "meminfo") }; (f, t) }
pub fn sync() { unsafe { check(cuCtxSynchronize(), "sync") } }

pub struct Buf { pub p: CUdeviceptr, pub n: usize }
impl Buf {
    pub fn new(n: usize) -> Buf { let mut p = 0; unsafe { check(cuMemAlloc_v2(&mut p, n.max(8)), "alloc"); check(cuMemsetD8_v2(p, 0, n.max(8)), "zero") }; Buf { p, n } }
    pub fn from<T: Copy>(v: &[T]) -> Buf { let b = Buf::new(std::mem::size_of_val(v)); b.put(v); b }
    pub fn put<T: Copy>(&self, v: &[T]) { let n = std::mem::size_of_val(v); assert!(n <= self.n); unsafe { check(cuMemcpyHtoD_v2(self.p, v.as_ptr().cast(), n), "htod") } }
    pub fn put_at(&self, off: usize, v: &[u8]) { assert!(off + v.len() <= self.n); unsafe { check(cuMemcpyHtoD_v2(self.p + off as u64, v.as_ptr().cast(), v.len()), "htod@") } }
    pub fn get<T: Copy + Default>(&self, count: usize) -> Vec<T> {
        let mut v = vec![T::default(); count]; let n = std::mem::size_of_val(&v[..]); assert!(n <= self.n);
        unsafe { check(cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), self.p, n), "dtoh") }; v
    }
    pub fn zero(&self) { unsafe { check(cuMemsetD8_v2(self.p, 0, self.n), "memset") } }
}
impl Drop for Buf { fn drop(&mut self) { unsafe { cuMemFree_v2(self.p) }; } }

pub struct Module { pub m: *mut c_void }
impl Module {
    pub fn load(image: &[u8]) -> Module { let mut m = ptr::null_mut(); let mut v = image.to_vec(); v.push(0); unsafe { check(cuModuleLoadData(&mut m, v.as_ptr().cast()), "module load") }; Module { m } }
    pub fn func(&self, name: &str) -> *mut c_void { let mut f = ptr::null_mut(); let c = CString::new(name).unwrap(); unsafe { check(cuModuleGetFunction(&mut f, self.m, c.as_ptr()), name) }; f }
    pub fn global(&self, name: &str) -> (CUdeviceptr, usize) { let (mut p, mut n) = (0, 0); let c = CString::new(name).unwrap(); unsafe { check(cuModuleGetGlobal_v2(&mut p, &mut n, self.m, c.as_ptr()), name) }; (p, n) }
}
pub fn func_attr(f: *mut c_void, attr: c_int) -> i32 { let mut v = 0; unsafe { check(cuFuncGetAttribute(&mut v, attr, f), "attr") }; v }

/// Kernel arguments, each stored as raw little-endian bytes.
pub struct Args { vals: Vec<Vec<u8>> }
impl Args {
    pub fn new() -> Args { Args { vals: vec![] } }
    pub fn i32(mut self, v: i32) -> Args { self.vals.push(v.to_le_bytes().to_vec()); self }
    pub fn f32(mut self, v: f32) -> Args { self.vals.push(v.to_le_bytes().to_vec()); self }
    pub fn ptr(mut self, v: u64) -> Args { self.vals.push(v.to_le_bytes().to_vec()); self }
    pub fn raw(mut self, v: Vec<u8>) -> Args { self.vals.push(v); self }
    pub fn launch(&mut self, f: *mut c_void, grid: u32, block: u32) {
        let mut store: Vec<Vec<u64>> = self.vals.iter().map(|b| { let mut w = vec![0u64; (b.len() + 7) / 8]; unsafe { ptr::copy_nonoverlapping(b.as_ptr(), w.as_mut_ptr().cast::<u8>(), b.len()) }; w }).collect();
        let mut ps: Vec<*mut c_void> = store.iter_mut().map(|w| w.as_mut_ptr().cast()).collect();
        unsafe { check(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, ptr::null_mut(), ps.as_mut_ptr(), ptr::null_mut()), "launch") };
        sync();
    }
}
