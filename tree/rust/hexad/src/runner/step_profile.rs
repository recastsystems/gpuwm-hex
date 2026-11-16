// SPDX-License-Identifier: Apache-2.0
//! Step wall ranges for a separate Nsight run (AD_STEP_NVTX=1).
//! No CUDA synchronization is added. The ordinary timing path loads no NVTX
//! library and does not allocate a label per step. Missing requested profiling
//! support is an error, not a silently empty trace.

use std::{ffi::{c_char, c_int, c_void, CString}, marker::PhantomData, rc::Rc, sync::OnceLock};

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

struct Api {
    push: unsafe extern "C" fn(*const c_char) -> c_int,
    pop: unsafe extern "C" fn() -> c_int,
}
static API: OnceLock<Result<Option<Api>, String>> = OnceLock::new();

fn load() -> Result<Option<Api>, String> {
    if std::env::var("AD_STEP_NVTX").as_deref() != Ok("1") { return Ok(None); }
    // CUDA 13 ships the NVTX3 interop ABI; older installations have nvToolsExt.
    // Both expose the same C range functions used here. Load neither by default.
    for name in [c"libnvtx3interop.so.1", c"libnvToolsExt.so.1"] { unsafe {
        let library = dlopen(name.as_ptr(), 2);
        if library.is_null() { continue; }
        let push = dlsym(library, c"nvtxRangePushA".as_ptr());
        let pop = dlsym(library, c"nvtxRangePop".as_ptr());
        if push.is_null() || pop.is_null() {
            let _ = dlclose(library);
            continue;
        }
        // Process-lifetime handle, matching the process-lifetime OnceLock.
        return Ok(Some(Api { push: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_char) -> c_int>(push),
                            pop: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> c_int>(pop) }));
    } }
    Err("DEFECT AD_STEP_NVTX=1 requires nvtxRangePushA/nvtxRangePop in libnvtx3interop.so.1 or libnvToolsExt.so.1".into())
}

/// Thread-scoped range; !Send/!Sync prevents popping on another thread.
pub struct StepRange { api: Option<&'static Api>, _thread: PhantomData<Rc<()>> }

impl StepRange {
    pub fn begin(pass: &str, step: usize) -> Result<Self, String> {
        let api = match API.get_or_init(load) {
            Ok(a) => a.as_ref(),
            Err(e) => return Err(e.clone()),
        };
        if let Some(api) = api {
            let label = CString::new(format!("adspeed/{pass}/{step}")).map_err(|e| e.to_string())?;
            // NVTX may be a no-op without an attached profiler. The analyzer
            // refuses absent ranges; no science depends on the NVTX return code.
            unsafe { (api.push)(label.as_ptr()); }
        }
        Ok(Self { api, _thread: PhantomData })
    }
}

impl Drop for StepRange {
    fn drop(&mut self) { if let Some(api) = self.api { unsafe { (api.pop)(); } } }
}
