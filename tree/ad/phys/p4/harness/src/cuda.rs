// SPDX-License-Identifier: Apache-2.0
//! Minimal CUDA driver API bindings (libcuda), no other dependency.
#![allow(non_camel_case_types)]
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
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const std::os::raw::c_char) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
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
}

pub fn check(r: CUresult, what: &str) {
    if r != 0 {
        let mut s: *const c_char = std::ptr::null();
        unsafe { cuGetErrorString(r, &mut s) };
        let msg = if s.is_null() { "?".to_string() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
        panic!("CUDA error {} ({}) in {}", r, msg, what);
    }
}

static CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Readers: ordinary launches. Writer: a timing region, so no other thread's
/// kernel lands between its events on the legacy stream.
pub static DEVICE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Make the primary context current on this thread (worker threads).
pub fn set_current() { unsafe { check(cuCtxSetCurrent(CTX.load(std::sync::atomic::Ordering::SeqCst) as CUcontext), "ctxset") }; }

pub fn init() {
    unsafe {
        check(cuInit(0), "cuInit");
        let mut d: CUdevice = 0;
        check(cuDeviceGet(&mut d, 0), "cuDeviceGet");
        let mut c: CUcontext = std::ptr::null_mut();
        check(cuDevicePrimaryCtxRetain(&mut c, d), "ctx");
        check(cuCtxSetCurrent(c), "ctxset");
        CTX.store(c as usize, std::sync::atomic::Ordering::SeqCst);
    }
}

pub fn mem_info() -> (usize, usize) {
    let (mut f, mut t) = (0usize, 0usize);
    unsafe { check(cuMemGetInfo_v2(&mut f, &mut t), "meminfo") };
    (f, t)
}

pub struct Module(CUmodule);
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
impl Module {
    pub fn load(bytes: &[u8]) -> Module {
        let mut v = bytes.to_vec();
        v.push(0);
        let mut m: CUmodule = std::ptr::null_mut();
        unsafe { check(cuModuleLoadData(&mut m, v.as_ptr() as *const c_void), "cuModuleLoadData") };
        Module(m)
    }
    /// G4: (address, bytes) of a module global, or None when the module has no such symbol.
    pub fn global(&self, name: &str) -> Option<(CUdeviceptr, usize)> {
        let n = CString::new(name).unwrap();
        let (mut a, mut z): (CUdeviceptr, usize) = (0, 0);
        let rc = unsafe { cuModuleGetGlobal_v2(&mut a, &mut z, self.0, n.as_ptr()) };
        if rc == 0 { Some((a, z)) } else { None }
    }
    /// G4: allocate a frame arena for up to `threads` launch threads (p4_frame_bytes each) and point the module
    /// global p4_frame_base at it; the buffer lives as long as the process.
    pub fn p4_frame(&self, threads: usize) {
        let (fa, _) = self.global("p4_frame_bytes").expect("ugwp image without p4_frame_bytes (G4 frame arena)");
        let mut v = [0u8; 8];
        unsafe { check(cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, fa, 8), "d2h") };
        let buf = DBuf::new(threads * u64::from_le_bytes(v) as usize);
        let (ba, _) = self.global("p4_frame_base").expect("ugwp image without p4_frame_base");
        unsafe { check(cuMemcpyHtoD_v2(ba, (&buf.ptr as *const CUdeviceptr) as *const c_void, 8), "h2d") };
        std::mem::forget(buf);
    }
    pub fn func(&self, name: &str) -> CUfunction {
        let n = CString::new(name).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        unsafe { check(cuModuleGetFunction(&mut f, self.0, n.as_ptr()), &format!("getfunc {}", name)) };
        f
    }
}

pub struct DBuf { pub ptr: CUdeviceptr, pub bytes: usize }
impl DBuf {
    pub fn new(bytes: usize) -> DBuf {
        let mut p: CUdeviceptr = 0;
        unsafe { check(cuMemAlloc_v2(&mut p, bytes.max(16)), "alloc") };
        unsafe { check(cuMemsetD8_v2(p, 0, bytes.max(16)), "memset") };
        DBuf { ptr: p, bytes }
    }
    pub fn upload_at(&self, off: usize, data: &[u8]) {
        assert!(off + data.len() <= self.bytes.max(16));
        if data.is_empty() { return; }
        unsafe { check(cuMemcpyHtoD_v2(self.ptr + off as u64, data.as_ptr() as *const c_void, data.len()), "h2d") };
    }
    pub fn download_at(&self, off: usize, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        if n == 0 { return v; }
        unsafe { check(cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr + off as u64, n), "d2h") };
        v
    }
    pub fn zero(&self) { unsafe { check(cuMemsetD8_v2(self.ptr, 0, self.bytes.max(16)), "memset") }; }
}
impl Drop for DBuf { fn drop(&mut self) { unsafe { cuMemFree_v2(self.ptr) }; } }

/// One kernel argument as raw bytes (scalar value or device pointer).
pub enum KArg { Bytes(Vec<u8>), Ptr(u64) }

pub fn launch(f: CUfunction, grid: [u32; 3], block: [u32; 3], shared: u32, args: &[KArg]) {
    let _g = DEVICE.read().unwrap();
    launch_unlocked(f, grid, block, shared, args)
}

pub fn launch_unlocked(f: CUfunction, grid: [u32; 3], block: [u32; 3], shared: u32, args: &[KArg]) {
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
