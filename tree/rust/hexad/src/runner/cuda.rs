// SPDX-License-Identifier: Apache-2.0
//! Minimal dynamically loaded CUDA Driver API used by the experimental replay.
use super::submit::{Clear, Copy, Launch};
use std::{ffi::{c_char, c_int, c_void, CString}, mem};

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

pub(super) unsafe fn symbol<T: std::marker::Copy>(library: *mut c_void, name: &str) -> Result<T, String> {
    let name = CString::new(name).map_err(|e| e.to_string())?;
    let address = dlsym(library, name.as_ptr());
    if address.is_null() || mem::size_of::<T>() != mem::size_of_val(&address) {
        return Err(format!("missing or incompatible CUDA symbol {name:?}"));
    }
    Ok(mem::transmute_copy(&address))
}

pub type ContextCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
pub type PtrCall = unsafe extern "C" fn(*mut c_void) -> i32;
pub type Create = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
pub type Record = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
pub type Wait = unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> i32;
/// Explicit legacy graph-update ABI. dlsym does not apply cuda.h's _v2 aliases.
#[derive(Clone, Copy)]
pub struct GraphApi {
    pub begin: unsafe extern "C" fn(*mut c_void, i32) -> i32,
    pub end: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub instantiate: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, u64) -> i32,
    pub update: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut *mut c_void, *mut i32) -> i32,
    pub launch: Record,
    pub destroy: PtrCall,
    pub exec_destroy: PtrCall,
}
// Explicit unversioned (v1) CUDA node ABI, matching cuda.h's legacy declarations.
#[repr(C)]
pub struct KernelNodeParams {
    pub function: *mut c_void,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared: u32,
    pub arguments: *mut *mut c_void,
    pub extra: *mut *mut c_void,
}
#[repr(C)]
pub struct MemsetNodeParams {
    pub destination: u64,
    pub pitch: usize,
    pub value: u32,
    pub element_size: u32,
    pub width: usize,
    pub height: usize,
}
#[derive(Clone, Copy)]
pub struct BoundGraphApi {
    pub create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32,
    pub add_kernel: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const *mut c_void, usize, *const KernelNodeParams) -> i32,
    pub add_clear: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const *mut c_void, usize, *const MemsetNodeParams, *mut c_void) -> i32,
    pub set_kernel: unsafe extern "C" fn(*mut c_void, *mut c_void, *const KernelNodeParams) -> i32,
    pub set_clear: unsafe extern "C" fn(*mut c_void, *mut c_void, *const MemsetNodeParams, *mut c_void) -> i32,
    pub get_nodes: unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut usize) -> i32,
}
#[repr(C)]
pub struct IpcMemoryHandle { pub reserved: [u8; 64] }

/// CUDA_MEMCPY3D_PEER_v1 from the CUDA Driver API. Each endpoint has the same
/// C layout; memory_type=2 means device memory. Depth=1 describes our rows.
#[repr(C)]
#[derive(Default)]
pub struct PeerCopyEndpoint {
    pub x_bytes: usize, pub y: usize, pub z: usize, pub lod: usize,
    pub memory_type: i32, pub host: *const c_void, pub device: u64,
    pub array: *mut c_void, pub context: *mut c_void,
    pub pitch: usize, pub height: usize,
}
#[repr(C)]
pub struct PeerCopy3D {
    pub src: PeerCopyEndpoint, pub dst: PeerCopyEndpoint,
    pub width_bytes: usize, pub height: usize, pub depth: usize,
}

pub struct Cuda {
    pub(super) library: *mut c_void,
    pub init: unsafe extern "C" fn(u32) -> i32,
    pub device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub device_can_access_peer: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
    pub context_enable_peer: unsafe extern "C" fn(*mut c_void, u32) -> i32,
    pub copy_peer_async: unsafe extern "C" fn(u64, *mut c_void, u64, *mut c_void, usize, *mut c_void) -> i32,
    pub copy_peer_3d_async: unsafe extern "C" fn(*const PeerCopy3D, *mut c_void) -> i32,
    pub event_elapsed: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32,
    pub context_create: ContextCreate,
    pub context_destroy: PtrCall,
    pub context_synchronize: unsafe extern "C" fn() -> i32,
    /// host-leftovers-2: a helper thread makes the window's context current before it page-locks a buffer.
    pub ctx_set_current: unsafe extern "C" fn(*mut c_void) -> i32,
    /// multicard-merge: the MYNN shard pool finds the dycore context it maps its helpers onto.
    pub ctx_get_current: unsafe extern "C" fn(*mut *mut c_void) -> i32,
    pub allocate: unsafe extern "C" fn(*mut u64, usize) -> i32,
    pub free: unsafe extern "C" fn(u64) -> i32,
    pub allocate_async: unsafe extern "C" fn(*mut u64, usize, *mut c_void) -> i32,
    pub free_async: unsafe extern "C" fn(u64, *mut c_void) -> i32,
    pub host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32,
    pub host_free: PtrCall,
    /// host-leftovers-2: page-lock existing host memory (pinned::PinMem).
    pub host_register: unsafe extern "C" fn(*mut c_void, usize, u32) -> i32,
    pub host_unregister: PtrCall,
    pub upload: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    pub upload_async: unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> i32,
    pub download: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    pub download_async: unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32,
    pub module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32,
    pub module_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32,
    pub module_global: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> i32,
    pub module_unload: PtrCall,
    pub stream_create: Create,
    pub stream_destroy: PtrCall,
    pub stream_synchronize: PtrCall,
    pub event_create: Create,
    pub event_destroy: PtrCall,
    pub event_synchronize: PtrCall,
    pub event_record: Record,
    pub stream_wait_event: Wait,
    pub launch: Launch,
    pub copy: Copy,
    pub clear: Clear,
    pub fill16: unsafe extern "C" fn(u64, u16, usize, *mut c_void) -> i32,
    pub fill32: unsafe extern "C" fn(u64, u32, usize, *mut c_void) -> i32,
    pub fill2d8: unsafe extern "C" fn(u64, usize, u8, usize, usize, *mut c_void) -> i32,
    pub fill2d32: unsafe extern "C" fn(u64, usize, u32, usize, usize, *mut c_void) -> i32,
    pub ipc_open: unsafe extern "C" fn(*mut u64, IpcMemoryHandle, u32) -> i32,
    pub ipc_close: unsafe extern "C" fn(u64) -> i32,
    pub address_range: unsafe extern "C" fn(*mut u64, *mut usize, u64) -> i32,
    pub mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    /// mynn-adjoint-v2: resident blocks per SM of a kernel, the context's device and a device attribute (persistent P3
    /// launches size their grid to what the card keeps resident).
    pub occupancy: unsafe extern "C" fn(*mut i32, *mut c_void, i32, usize) -> i32,
    pub ctx_device: unsafe extern "C" fn(*mut i32) -> i32,
    pub device_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
}

impl Cuda {
    pub unsafe fn bound_graph_api(&self) -> Result<BoundGraphApi, String> {
        Ok(BoundGraphApi {
            create: symbol(self.library, "cuGraphCreate")?,
            add_kernel: symbol(self.library, "cuGraphAddKernelNode")?,
            add_clear: symbol(self.library, "cuGraphAddMemsetNode")?,
            set_kernel: symbol(self.library, "cuGraphExecKernelNodeSetParams")?,
            set_clear: symbol(self.library, "cuGraphExecMemsetNodeSetParams")?,
            get_nodes: symbol(self.library, "cuGraphGetNodes")?,
        })
    }
    pub unsafe fn graph_api(&self) -> Result<GraphApi, String> {
        Ok(GraphApi {
            begin: symbol(self.library, "cuStreamBeginCapture_v2")?,
            end: symbol(self.library, "cuStreamEndCapture")?,
            instantiate: symbol(self.library, "cuGraphInstantiateWithFlags")?,
            update: symbol(self.library, "cuGraphExecUpdate")?,
            launch: symbol(self.library, "cuGraphLaunch")?,
            destroy: symbol(self.library, "cuGraphDestroy")?,
            exec_destroy: symbol(self.library, "cuGraphExecDestroy")?,
        })
    }
    pub unsafe fn load() -> Result<Self, String> {
        let library = dlopen(c"libcuda.so.1".as_ptr(), 2);
        if library.is_null() { return Err("cannot load libcuda.so.1".into()); }
        // Close the handle if any required symbol is absent.
        let result = (|| Ok(Self {
            library,
            init: symbol(library, "cuInit")?,
            device_get: symbol(library, "cuDeviceGet")?,
            device_can_access_peer: symbol(library, "cuDeviceCanAccessPeer")?,
            context_enable_peer: symbol(library, "cuCtxEnablePeerAccess")?,
            copy_peer_async: symbol(library, "cuMemcpyPeerAsync")?,
            copy_peer_3d_async: symbol(library, "cuMemcpy3DPeerAsync")?,
            event_elapsed: symbol(library, "cuEventElapsedTime")?,
            context_create: symbol(library, "cuCtxCreate_v2")?,
            context_destroy: symbol(library, "cuCtxDestroy_v2")?,
            context_synchronize: symbol(library, "cuCtxSynchronize")?,
            ctx_set_current: symbol(library, "cuCtxSetCurrent")?,
            ctx_get_current: symbol(library, "cuCtxGetCurrent")?,
            allocate: symbol(library, "cuMemAlloc_v2")?,
            free: symbol(library, "cuMemFree_v2")?,
            allocate_async: symbol(library, "cuMemAllocAsync")?,
            free_async: symbol(library, "cuMemFreeAsync")?,
            host_alloc: symbol(library, "cuMemHostAlloc")?,
            host_free: symbol(library, "cuMemFreeHost")?,
            host_register: symbol(library, "cuMemHostRegister_v2")?,
            host_unregister: symbol(library, "cuMemHostUnregister")?,
            upload: symbol(library, "cuMemcpyHtoD_v2")?,
            upload_async: symbol(library, "cuMemcpyHtoDAsync_v2")?,
            download: symbol(library, "cuMemcpyDtoH_v2")?,
            download_async: symbol(library, "cuMemcpyDtoHAsync_v2")?,
            module_load: symbol(library, "cuModuleLoadData")?,
            module_function: symbol(library, "cuModuleGetFunction")?,
            module_global: symbol(library, "cuModuleGetGlobal_v2")?,
            module_unload: symbol(library, "cuModuleUnload")?,
            stream_create: symbol(library, "cuStreamCreate")?,
            stream_destroy: symbol(library, "cuStreamDestroy_v2")?,
            stream_synchronize: symbol(library, "cuStreamSynchronize")?,
            event_create: symbol(library, "cuEventCreate")?,
            event_destroy: symbol(library, "cuEventDestroy_v2")?,
            event_synchronize: symbol(library, "cuEventSynchronize")?,
            event_record: symbol(library, "cuEventRecord")?,
            stream_wait_event: symbol(library, "cuStreamWaitEvent")?,
            launch: symbol(library, "cuLaunchKernel")?,
            copy: symbol(library, "cuMemcpyDtoDAsync_v2")?,
            clear: symbol(library, "cuMemsetD8Async")?,
            fill16: symbol(library, "cuMemsetD16Async")?,
            fill32: symbol(library, "cuMemsetD32Async")?,
            fill2d8: symbol(library, "cuMemsetD2D8Async")?,
            fill2d32: symbol(library, "cuMemsetD2D32Async")?,
            ipc_open: symbol(library, "cuIpcOpenMemHandle")?,
            ipc_close: symbol(library, "cuIpcCloseMemHandle")?,
            address_range: symbol(library, "cuMemGetAddressRange_v2")?,
            mem_info: symbol(library, "cuMemGetInfo_v2")?,
            occupancy: symbol(library, "cuOccupancyMaxActiveBlocksPerMultiprocessor")?,
            ctx_device: symbol(library, "cuCtxGetDevice")?,
            device_attribute: symbol(library, "cuDeviceGetAttribute")?,
        }))();
        if result.is_err() { let _ = dlclose(library); }
        // host-syncs: host copies into piece-registered page-locked buffers go one driver call per piece (pinned.rs).
        result.map(|mut c| { super::pinned::install_split(&mut c); poison_install(&mut c); c })
    }
}
impl Drop for Cuda {
    fn drop(&mut self) { unsafe { let _ = dlclose(self.library); } }
}

// ---- mynn (diagnostic; nothing changes unless AD_ALLOC_POISON is set) ------------------------------------------
// THE QUESTION IT ANSWERS: does any result depend on what a fresh cuMemAlloc block happens to hold? The driver hands
// a new process zeroed pages, but a block this process freed comes back with its old bytes (schemes.rs Grow: "a freed
// block is reused by the plan's own later allocations"), and how often that happens depends on how full the card is.
// a test machine <local path> g4-B9count (pkg18m2 on a card with 2.8 GB free) ended rc 0, 4/4 boundaries, tangent side equal,
// adjoint dot 1.25e-5 with every setting equal to legs that give the reference words: only addresses differed.
// AD_ALLOC_POISON=<hex 32-bit word> fills every block cuMemAlloc returns with that word before anyone uses it (a finite
// float such as 42F60000 = 123.0 keeps masked arithmetic finite). A run whose words or boundaries move under it reads
// memory it never wrote. AD_ALLOC_POISON_RANGE=lo:hi limits it to allocation ordinals lo..hi (bisect);
// AD_ALLOC_POISON_BYTES=min:max to block sizes; AD_ALLOC_POISON_LOG=1 prints one line per allocation;
// AD_ALLOC_POISON_BT=1 adds the call stack of each filled block. It sits under the spill layer (spill.rs wraps
// `allocate` after this), so it sees every card block; managed (spilled) blocks and cuMemAllocAsync are not covered.
static POISON_REAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static POISON_SET32: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static POISON_SET8: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static POISON_WORD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static POISON_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static POISON_LO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static POISON_HI: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
static POISON_BMIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static POISON_BMAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
static POISON_FLAGS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static POISON_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static POISON_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

unsafe extern "C" fn poison_allocate(out: *mut u64, bytes: usize) -> i32 {
    use std::sync::atomic::Ordering::Relaxed;
    let real: unsafe extern "C" fn(*mut u64, usize) -> i32 = mem::transmute(POISON_REAL.load(Relaxed));
    let rc = real(out, bytes);
    if rc != 0 { return rc; }
    let n = POISON_N.fetch_add(1, Relaxed);
    let hit = n >= POISON_LO.load(Relaxed) && n < POISON_HI.load(Relaxed) && bytes >= POISON_BMIN.load(Relaxed) && bytes <= POISON_BMAX.load(Relaxed);
    let flags = POISON_FLAGS.load(Relaxed);
    if hit {
        let set32: unsafe extern "C" fn(u64, u32, usize) -> i32 = mem::transmute(POISON_SET32.load(Relaxed));
        let set8: unsafe extern "C" fn(u64, u8, usize) -> i32 = mem::transmute(POISON_SET8.load(Relaxed));
        let w = POISON_WORD.load(Relaxed);
        if bytes >= 4 { let r = set32(*out, w, bytes / 4); if r != 0 { return r; } }
        if bytes % 4 != 0 { let r = set8(*out + (bytes & !3usize) as u64, w as u8, bytes % 4); if r != 0 { return r; } }
        POISON_BYTES.fetch_add(bytes as u64, Relaxed);
        POISON_HITS.fetch_add(1, Relaxed);
    }
    if flags & 1 != 0 || (hit && flags & 2 != 0) {
        let bt = if hit && flags & 2 != 0 {
            let t = format!("{}", std::backtrace::Backtrace::force_capture());
            let keep: Vec<&str> = t.lines().map(|l| l.trim()).filter(|l| !l.starts_with("at ") && !l.contains("backtrace") && !l.contains("poison_allocate")).take(14).collect();
            format!(",\"stack\":{:?}", keep.join(" | "))
        } else { String::new() };
        eprintln!("{{\"event\":\"alloc_poison\",\"n\":{n},\"bytes\":{bytes},\"at\":{},\"hit\":{hit}{bt}}}", *out);
    }
    rc
}

/// Totals for the run's receipt: (allocations seen, blocks filled, bytes filled). All zero when the diagnostic is off.
pub fn poison_totals() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (POISON_N.load(Relaxed), POISON_HITS.load(Relaxed), POISON_BYTES.load(Relaxed))
}

unsafe fn poison_install(c: &mut Cuda) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(word) = std::env::var("AD_ALLOC_POISON").ok().and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) else { return };
    let (Ok(set32), Ok(set8)) = (symbol::<unsafe extern "C" fn(u64, u32, usize) -> i32>(c.library, "cuMemsetD32_v2"),
                                 symbol::<unsafe extern "C" fn(u64, u8, usize) -> i32>(c.library, "cuMemsetD8_v2")) else {
        eprintln!("{{\"event\":\"alloc_poison_install\",\"on\":false,\"why\":\"cuMemsetD32_v2 / cuMemsetD8_v2 not found\"}}");
        return;
    };
    let pair = |k: &str| -> Option<(u64, u64)> {
        let v = std::env::var(k).ok()?; let (a, b) = v.split_once(':')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    };
    if let Some((lo, hi)) = pair("AD_ALLOC_POISON_RANGE") { POISON_LO.store(lo, Relaxed); POISON_HI.store(hi, Relaxed); }
    if let Some((lo, hi)) = pair("AD_ALLOC_POISON_BYTES") { POISON_BMIN.store(lo as usize, Relaxed); POISON_BMAX.store(hi as usize, Relaxed); }
    let on = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
    POISON_FLAGS.store(u32::from(on("AD_ALLOC_POISON_LOG")) | (u32::from(on("AD_ALLOC_POISON_BT")) << 1), Relaxed);
    POISON_WORD.store(word, Relaxed);
    POISON_SET32.store(set32 as usize, Relaxed);
    POISON_SET8.store(set8 as usize, Relaxed);
    POISON_REAL.store(c.allocate as usize, Relaxed);
    c.allocate = poison_allocate;
    eprintln!("{{\"event\":\"alloc_poison_install\",\"on\":true,\"word_hex\":\"{word:08x}\",\"range\":[{},{}],\"bytes\":[{},{}]}}",
              POISON_LO.load(Relaxed), POISON_HI.load(Relaxed), POISON_BMIN.load(Relaxed), POISON_BMAX.load(Relaxed));
}

pub fn checked(operation: &str, status: i32) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("{operation} failed with CUDA status {status}{}", status_note(status))) }
}

// ---- central home-card (2026-11-11): a refused kernel image says why ---------------------------------------------------
// THE BREAKAGE THIS ENDS: the first run of the home-built program on a home card (an RTX 4090, class sm_89) with the one
// sealed package that is at home ended in "cuModuleLoadData failed with CUDA status 209" and nothing else. No image
// was named and no card class, and a package image, a scheme-bundle image and a kernel-set image all gave the same
// line. 209 is the driver saying the image holds no code for this card's class: the program checks no class itself.
// The words before the note are what they always were, so anything that reads the old line still reads it.
pub fn status_note(status: i32) -> &'static str {
    match status {
        209 => " (209 = this image holds no code for this card's class: it was built or recorded for another class of card; a recording replays only on the class it was recorded on)",
        _ => "",
    }
}
/// A card class as nvcc names it, from the driver's compute capability: (8, 9) is "sm_89", (12, 0) is "sm_120".
pub fn class_name(major: i32, minor: i32) -> String { format!("sm_{major}{minor}") }
/// The refusal of one image: the operation and status as `checked` words them, then what the image is and this card's class.
pub fn load_refusal(operation: &str, status: i32, image: &str, class: &str) -> String {
    let image = if image.is_empty() { String::new() } else { format!("{image}, ") };
    format!("{operation} failed with CUDA status {status}: {image}this card is {class}{}", status_note(status))
}
impl Cuda {
    /// The class of the current context's card ("sm_89", "sm_120"); "sm_?" when the driver will not say. Asked only
    /// when a refusal is being written (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76).
    pub unsafe fn card_class(&self) -> String {
        let (mut device, mut major, mut minor) = (0i32, 0i32, 0i32);
        if (self.ctx_device)(&mut device) != 0 || (self.device_attribute)(&mut major, 75, device) != 0
            || (self.device_attribute)(&mut minor, 76, device) != 0 { return "sm_?".into(); }
        class_name(major, minor)
    }
}

#[cfg(test)]
mod refused_image_tests {
    use super::*;
    #[test]
    fn a_refused_image_says_why_and_every_other_status_reads_as_before() {
        assert_eq!(checked("cuMemAlloc arena", 0), Ok(()));
        assert_eq!(checked("cuMemAlloc arena", 2).unwrap_err(), "cuMemAlloc arena failed with CUDA status 2");
        assert_eq!(checked("cuModuleGetFunction win_fill_f32", 500).unwrap_err(), "cuModuleGetFunction win_fill_f32 failed with CUDA status 500");
        let e = checked("cuModuleLoadData p8 rules", 209).unwrap_err();
        assert!(e.starts_with("cuModuleLoadData p8 rules failed with CUDA status 209"), "{e}");
        assert!(e.contains("no code for this card's class"), "{e}");
    }
    #[test]
    fn a_refused_package_image_is_named_with_the_card_class() {
        assert_eq!(class_name(8, 9), "sm_89");
        assert_eq!(class_name(12, 0), "sm_120");
        let e = load_refusal("cuModuleLoadData", 209, "plan module ab12 (recording/images/ab12.gz)", &class_name(8, 9));
        // the old line is still the start of the new one
        assert!(e.starts_with("cuModuleLoadData failed with CUDA status 209"), "{e}");
        assert!(e.contains("plan module ab12 (recording/images/ab12.gz), this card is sm_89"), "{e}");
        assert!(e.contains("no code for this card's class"), "{e}");
        assert_eq!(load_refusal("cuModuleLoadData /b/p8-sm_120.cubin", 200, "", "sm_89"),
                   "cuModuleLoadData /b/p8-sm_120.cubin failed with CUDA status 200: this card is sm_89");
    }
}
