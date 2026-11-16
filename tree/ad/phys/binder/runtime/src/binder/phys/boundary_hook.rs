// SPDX-License-Identifier: Apache-2.0
//! Optional same-process checkpoint consumer at a fenced replay boundary.
use super::checkpoint::Binding;
use std::{ffi::{c_char, c_int, c_void, CString}, mem, path::Path};
use std::os::unix::ffi::OsStrExt;

#[link(name = "dl")]
extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

// Pointers in bindings are valid only until the callback returns. Its work
// must finish or be enqueued in this CUDA context before the return; the
// binder synchronizes that context before it advances the frame.
pub type Callback = unsafe extern "C" fn(u64, *mut c_void, *mut c_void, *const Binding, usize) -> i32;
pub struct Hook { library: *mut c_void, callback: Callback }

impl Hook {
    pub unsafe fn load(path: &Path) -> Result<Self, String> {
        let name = CString::new(path.as_os_str().as_bytes()).map_err(|_| "NUL in checkpoint hook path")?;
        let library = dlopen(name.as_ptr(), 2);
        if library.is_null() { return Err("cannot load checkpoint hook".into()); }
        let pointer = dlsym(library, c"ad_binder_boundary_v1".as_ptr());
        if pointer.is_null() || mem::size_of::<Callback>() != mem::size_of_val(&pointer) {
            let _ = dlclose(library);
            return Err("checkpoint hook has no ad_binder_boundary_v1 entry".into());
        }
        Ok(Self { library, callback: mem::transmute_copy(&pointer) })
    }

    pub unsafe fn call(&self, step: u64, context: *mut c_void, stream: *mut c_void,
                       bindings: &[Binding]) -> Result<(), String> {
        let status = (self.callback)(step, context, stream, bindings.as_ptr(), bindings.len());
        if status != 0 { return Err(format!("checkpoint hook refused step {step}: {status}")); }
        Ok(())
    }
}
impl Drop for Hook { fn drop(&mut self) { unsafe { if !self.library.is_null() { let _ = dlclose(self.library); } } } }

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    unsafe extern "C" fn callback(step: u64, _: *mut c_void, _: *mut c_void,
                                   fields: *const Binding, count: usize) -> i32 {
        if step != 7 || count != 1 || fields.is_null() { return 9; }
        let field = &*fields;
        if field.id != 2 || field.bytes != 1 || field.memory != 2 { return 10; }
        *(field.address as *mut u8) = 42;
        0
    }
    #[test]
    fn passes_live_host_binding_and_propagates_refusal() {
        let hook = Hook { library: ptr::null_mut(), callback };
        let mut value = 0u8;
        let fields = [Binding { id: 2, bytes: 1, memory: 2, address: (&mut value as *mut u8) as u64 }];
        unsafe { hook.call(7, ptr::null_mut(), ptr::null_mut(), &fields).unwrap(); }
        assert_eq!(value, 42);
        assert!(unsafe { hook.call(8, ptr::null_mut(), ptr::null_mut(), &fields) }.is_err());
    }
}
