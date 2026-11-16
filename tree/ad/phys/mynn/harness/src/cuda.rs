// SPDX-License-Identifier: Apache-2.0
//! Minimal CUDA driver API bindings (libcuda), no other dependency. Adapted from tree/ad/u2_transport/harness.
#![allow(non_camel_case_types, dead_code)]
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = u64;

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxSetLimit(limit: c_int, value: usize) -> CUresult;
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuModuleGetGlobal_v2(p: *mut CUdeviceptr, bytes: *mut usize, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemsetD8_v2(dst: CUdeviceptr, v: u8, n: usize) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
        shared: c_uint, stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuEventCreate(e: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventRecord(e: CUevent, stream: *mut c_void) -> CUresult;
    fn cuEventSynchronize(e: CUevent) -> CUresult;
    fn cuEventElapsedTime(ms: *mut f32, a: CUevent, b: CUevent) -> CUresult;
    fn cuGetErrorString(r: CUresult, s: *mut *const c_char) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuFuncGetAttribute(v: *mut c_int, attr: c_int, f: CUfunction) -> CUresult;
}

pub fn check(r: CUresult, what: &str) {
    if r != 0 {
        let mut s: *const c_char = std::ptr::null();
        unsafe { cuGetErrorString(r, &mut s) };
        let msg = if s.is_null() { "?".to_string() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
        panic!("CUDA error {} ({}) in {}", r, msg, what);
    }
}

pub fn init() {
    unsafe {
        check(cuInit(0), "cuInit");
        let mut d: CUdevice = 0;
        check(cuDeviceGet(&mut d, 0), "cuDeviceGet");
        let mut c: CUcontext = std::ptr::null_mut();
        check(cuDevicePrimaryCtxRetain(&mut c, d), "ctx");
        check(cuCtxSetCurrent(c), "ctxset");
    }
}
/// CU_LIMIT_STACK_SIZE = 0
pub fn set_stack(bytes: usize) { unsafe { check(cuCtxSetLimit(0, bytes), "stack limit") } }

pub fn mem_info() -> (usize, usize) {
    let (mut f, mut t) = (0usize, 0usize);
    unsafe { check(cuMemGetInfo_v2(&mut f, &mut t), "meminfo") };
    (f, t)
}

pub struct Module(CUmodule);
impl Module {
    pub fn load(bytes: &[u8]) -> Module {
        let mut v = bytes.to_vec();
        v.push(0);
        let mut m: CUmodule = std::ptr::null_mut();
        unsafe { check(cuModuleLoadData(&mut m, v.as_ptr() as *const c_void), "cuModuleLoadData") };
        Module(m)
    }
    pub fn func(&self, name: &str) -> CUfunction {
        let n = CString::new(name).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        unsafe { check(cuModuleGetFunction(&mut f, self.0, n.as_ptr()), &format!("getfunc {}", name)) };
        f
    }
    pub fn set_global(&self, name: &str, bytes: &[u8]) {
        let n = CString::new(name).unwrap();
        let (mut p, mut s) = (0u64, 0usize);
        unsafe { check(cuModuleGetGlobal_v2(&mut p, &mut s, self.0, n.as_ptr()), &format!("global {}", name)) };
        assert_eq!(s, bytes.len(), "global {} size", name);
        unsafe { check(cuMemcpyHtoD_v2(p, bytes.as_ptr() as *const c_void, bytes.len()), "h2d global") };
    }
}
/// (registers, local bytes per thread) of a kernel
pub fn func_info(f: CUfunction) -> (i32, i32) {
    let (mut r, mut l) = (0, 0);
    unsafe { cuFuncGetAttribute(&mut r, 4, f); cuFuncGetAttribute(&mut l, 3, f); }
    (r, l)
}

pub struct DBuf { pub ptr: CUdeviceptr, pub bytes: usize }
impl DBuf {
    pub fn new(bytes: usize) -> DBuf {
        let mut p: CUdeviceptr = 0;
        unsafe { check(cuMemAlloc_v2(&mut p, bytes.max(16)), &format!("alloc {} bytes", bytes)) };
        unsafe { check(cuMemsetD8_v2(p, 0, bytes.max(16)), "memset") };
        DBuf { ptr: p, bytes }
    }
    pub fn upload(&self, data: &[u8]) { self.upload_at(0, data) }
    pub fn upload_at(&self, off: usize, data: &[u8]) {
        assert!(off + data.len() <= self.bytes.max(16));
        if data.is_empty() { return; }
        unsafe { check(cuMemcpyHtoD_v2(self.ptr + off as u64, data.as_ptr() as *const c_void, data.len()), "h2d") };
    }
    pub fn download(&self) -> Vec<u8> { self.download_at(0, self.bytes) }
    pub fn download_at(&self, off: usize, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        if n == 0 { return v; }
        unsafe { check(cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr + off as u64, n), "d2h") };
        v
    }
    pub fn copy_from(&self, o: &DBuf) { assert!(o.bytes <= self.bytes); unsafe { check(cuMemcpyDtoD_v2(self.ptr, o.ptr, o.bytes), "d2d") } }
    pub fn zero(&self) { unsafe { check(cuMemsetD8_v2(self.ptr, 0, self.bytes.max(16)), "memset") }; }
}
impl Drop for DBuf { fn drop(&mut self) { unsafe { cuMemFree_v2(self.ptr) }; } }

pub enum KArg { Bytes(Vec<u8>), Ptr(u64) }

pub fn launch(f: CUfunction, grid: [u32; 3], block: [u32; 3], shared: u32, args: &[KArg]) {
    let mut store: Vec<Vec<u8>> = args.iter().map(|a| match a {
        KArg::Bytes(b) => { let mut v = b.clone(); while v.len() < 8 { v.push(0) }; v }
        KArg::Ptr(p) => p.to_le_bytes().to_vec(),
    }).collect();
    let mut ptrs: Vec<*mut c_void> = store.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
    unsafe {
        check(cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
            std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut()), "launch");
    }
}
pub fn sync() { unsafe { check(cuCtxSynchronize(), "sync") }; }

pub struct Timer(CUevent, CUevent);
impl Timer {
    pub fn new() -> Timer {
        let (mut a, mut b): (CUevent, CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe { check(cuEventCreate(&mut a, 0), "ev"); check(cuEventCreate(&mut b, 0), "ev"); }
        Timer(a, b)
    }
    pub fn start(&self) { unsafe { check(cuEventRecord(self.0, std::ptr::null_mut()), "rec") }; }
    pub fn stop_ms(&self) -> f64 {
        let mut ms = 0f32;
        unsafe {
            check(cuEventRecord(self.1, std::ptr::null_mut()), "rec");
            check(cuEventSynchronize(self.1), "evsync");
            check(cuEventElapsedTime(&mut ms, self.0, self.1), "elapsed");
        }
        ms as f64
    }
}
