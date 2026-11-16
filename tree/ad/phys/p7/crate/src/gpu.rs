// SPDX-License-Identifier: Apache-2.0
//! Minimal CUDA driver + NVRTC access for the P7 radiation derivative proofs.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(f: u32) -> c_int;
    fn cuDeviceGet(d: *mut c_int, i: c_int) -> c_int;
    fn cuCtxCreate_v2(c: *mut *mut c_void, f: u32, d: c_int) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuMemAlloc_v2(p: *mut u64, n: usize) -> c_int;
    fn cuMemFree_v2(p: u64) -> c_int;
    fn cuMemcpyHtoD_v2(d: u64, s: *const c_void, n: usize) -> c_int;
    fn cuMemcpyDtoH_v2(d: *mut c_void, s: u64, n: usize) -> c_int;
    fn cuMemcpyDtoD_v2(d: u64, s: u64, n: usize) -> c_int;
    fn cuMemsetD8_v2(d: u64, v: u8, n: usize) -> c_int;
    fn cuModuleLoadData(m: *mut *mut c_void, image: *const c_void) -> c_int;
    fn cuModuleGetFunction(f: *mut *mut c_void, m: *mut c_void, n: *const c_char) -> c_int;
    fn cuLaunchKernel(f: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32,
                      sh: u32, st: *mut c_void, a: *mut *mut c_void, e: *mut *mut c_void) -> c_int;
    fn cuCtxSetLimit(limit: c_int, value: usize) -> c_int;
}
unsafe extern "C" {
    fn dlopen(p: *const c_char, f: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
}

pub fn ck(r: c_int, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("{what}: CUDA error {r}")) }
}

pub struct Gpu { _ctx: *mut c_void, allocs: Vec<u64> }

impl Gpu {
    pub fn new() -> Result<Self, String> {
        unsafe {
            ck(cuInit(0), "cuInit")?;
            let mut d = 0; ck(cuDeviceGet(&mut d, 0), "cuDeviceGet")?;
            let mut c = ptr::null_mut(); ck(cuCtxCreate_v2(&mut c, 0, d), "cuCtxCreate")?;
            // CU_LIMIT_STACK_SIZE = 0: the adjoint keeps its tape in thread-local memory.
            ck(cuCtxSetLimit(0, 32 * 1024), "stack limit")?;
            Ok(Self { _ctx: c, allocs: Vec::new() })
        }
    }
    pub fn alloc(&mut self, n: usize) -> Result<u64, String> {
        let mut a = 0; ck(unsafe { cuMemAlloc_v2(&mut a, n.max(1)) }, "alloc")?;
        self.allocs.push(a); Ok(a)
    }
    pub fn put(&mut self, data: &[u8]) -> Result<u64, String> {
        let a = self.alloc(data.len())?; self.upload(a, data)?; Ok(a)
    }
    pub fn upload(&self, a: u64, data: &[u8]) -> Result<(), String> {
        ck(unsafe { cuMemcpyHtoD_v2(a, data.as_ptr().cast(), data.len()) }, "upload")
    }
    pub fn download(&self, a: u64, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        ck(unsafe { cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), a, n) }, "download")?; Ok(v)
    }
    pub fn copy(&self, dst: u64, src: u64, n: usize) -> Result<(), String> {
        ck(unsafe { cuMemcpyDtoD_v2(dst, src, n) }, "dtod")
    }
    pub fn clear(&self, a: u64, n: usize) -> Result<(), String> { ck(unsafe { cuMemsetD8_v2(a, 0, n) }, "clear") }
    pub fn sync(&self) -> Result<(), String> { ck(unsafe { cuCtxSynchronize() }, "synchronize") }
    pub fn module(&self, image: &[u8]) -> Result<Module, String> {
        let mut m = ptr::null_mut();
        ck(unsafe { cuModuleLoadData(&mut m, image.as_ptr().cast()) }, "module load")?;
        Ok(Module(m))
    }
    pub fn free_all(&mut self) { for a in self.allocs.drain(..) { unsafe { cuMemFree_v2(a); } } }
}

pub struct Module(*mut c_void);
impl Module {
    pub fn func(&self, name: &str) -> Result<Func, String> {
        let mut f = ptr::null_mut(); let c = CString::new(name).unwrap();
        ck(unsafe { cuModuleGetFunction(&mut f, self.0, c.as_ptr()) }, &format!("function {name}"))?;
        Ok(Func(f))
    }
}
pub struct Func(*mut c_void);

/// One kernel argument, stored by value so its address is stable while launching.
pub enum Arg { I32(i32), F32(f32), Ptr(u64) }
impl Func {
    pub fn launch(&self, grid: [u32; 3], block: [u32; 3], args: &[Arg]) -> Result<(), String> {
        let mut store: Vec<[u8; 8]> = args.iter().map(|a| match a {
            Arg::I32(v) => { let mut b = [0u8; 8]; b[..4].copy_from_slice(&v.to_ne_bytes()); b }
            Arg::F32(v) => { let mut b = [0u8; 8]; b[..4].copy_from_slice(&v.to_ne_bytes()); b }
            Arg::Ptr(v) => v.to_ne_bytes() }).collect();
        let mut ptrs: Vec<*mut c_void> = store.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();
        ck(unsafe { cuLaunchKernel(self.0, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0,
                                   ptr::null_mut(), ptrs.as_mut_ptr(), ptr::null_mut()) }, "launch")
    }
}

type Prog = *mut c_void;
pub fn nvrtc_compile(lib: &str, source: &str, name: &str, options: &[&str]) -> Result<Vec<u8>, String> {
    unsafe {
        let l = CString::new(lib).unwrap();
        let h = dlopen(l.as_ptr(), 2);
        if h.is_null() { return Err(format!("cannot load {lib}")); }
        let sym = |n: &CStr| -> Result<*mut c_void, String> {
            let x = dlsym(h, n.as_ptr()); if x.is_null() { Err(format!("missing {n:?}")) } else { Ok(x) } };
        let create: unsafe extern "C" fn(*mut Prog, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int =
            std::mem::transmute(sym(c"nvrtcCreateProgram")?);
        let compile: unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> c_int = std::mem::transmute(sym(c"nvrtcCompileProgram")?);
        let log_size: unsafe extern "C" fn(Prog, *mut usize) -> c_int = std::mem::transmute(sym(c"nvrtcGetProgramLogSize")?);
        let log: unsafe extern "C" fn(Prog, *mut c_char) -> c_int = std::mem::transmute(sym(c"nvrtcGetProgramLog")?);
        let size: unsafe extern "C" fn(Prog, *mut usize) -> c_int = std::mem::transmute(sym(c"nvrtcGetCUBINSize")?);
        let get: unsafe extern "C" fn(Prog, *mut c_char) -> c_int = std::mem::transmute(sym(c"nvrtcGetCUBIN")?);
        let src = CString::new(source).unwrap(); let nm = CString::new(name).unwrap();
        let mut p: Prog = ptr::null_mut();
        if create(&mut p, src.as_ptr(), nm.as_ptr(), 0, ptr::null(), ptr::null()) != 0 { return Err("nvrtcCreateProgram".into()); }
        let opts: Vec<CString> = options.iter().map(|o| CString::new(*o).unwrap()).collect();
        let optp: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        let rc = compile(p, optp.len() as c_int, optp.as_ptr());
        let mut n = 0; log_size(p, &mut n); let mut buf = vec![0u8; n.max(1)]; log(p, buf.as_mut_ptr().cast());
        let text = String::from_utf8_lossy(&buf[..n.saturating_sub(1)]).to_string();
        if rc != 0 { return Err(format!("NVRTC failed ({rc}):\n{}", &text[..text.len().min(20000)])); }
        let mut cn = 0; size(p, &mut cn); let mut cubin = vec![0u8; cn]; get(p, cubin.as_mut_ptr().cast());
        Ok(cubin)
    }
}
