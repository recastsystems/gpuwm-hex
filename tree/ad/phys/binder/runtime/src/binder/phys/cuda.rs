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

unsafe fn symbol<T: std::marker::Copy>(library: *mut c_void, name: &str) -> Result<T, String> {
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
#[repr(C)]
pub struct IpcMemoryHandle { pub reserved: [u8; 64] }

pub struct Cuda {
    library: *mut c_void,
    pub init: unsafe extern "C" fn(u32) -> i32,
    pub device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub context_create: ContextCreate,
    pub context_destroy: PtrCall,
    pub context_synchronize: unsafe extern "C" fn() -> i32,
    /// host-leftovers-2: a helper thread makes the window's context current before it page-locks a buffer.
    pub ctx_set_current: unsafe extern "C" fn(*mut c_void) -> i32,
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
    pub unsafe fn load() -> Result<Self, String> {
        let library = dlopen(c"libcuda.so.1".as_ptr(), 2);
        if library.is_null() { return Err("cannot load libcuda.so.1".into()); }
        // Close the handle if any required symbol is absent.
        let result = (|| Ok(Self {
            library,
            init: symbol(library, "cuInit")?,
            device_get: symbol(library, "cuDeviceGet")?,
            context_create: symbol(library, "cuCtxCreate_v2")?,
            context_destroy: symbol(library, "cuCtxDestroy_v2")?,
            context_synchronize: symbol(library, "cuCtxSynchronize")?,
            ctx_set_current: symbol(library, "cuCtxSetCurrent")?,
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
        result
    }
}
impl Drop for Cuda {
    fn drop(&mut self) { unsafe { let _ = dlclose(self.library); } }
}

pub fn checked(operation: &str, status: i32) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("{operation} failed with CUDA status {status}")) }
}
