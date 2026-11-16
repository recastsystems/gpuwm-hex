// SPDX-License-Identifier: Apache-2.0
//! One graph launch for a fully lowered, single-stream device program.
//!
//! This is a backend, not a lowering of window.rs. The caller must supply every
//! forward, TL, reverse, and glue command in order, with all device storage live.
//! BoundBatch admits only kernels, device copies and byte clears. It cannot hide
//! host nodes, allocations, readbacks, or callbacks inside the executable graph.
//! A changed by-value frame or changed pointer requires a newly prepared graph.

use super::{cuda::{self, Cuda}, submit::{BoundBatch, NativeDriver, Submit}};
use std::{ffi::c_void, marker::PhantomData, ptr};

type Handle = *mut c_void;
type PtrCall = unsafe extern "C" fn(Handle) -> i32;

/// Optional symbols: ordinary eager execution does not require graph support.
/// The lifetime keeps libcuda loaded while any executable uses these functions.
pub struct GraphDriver<'cuda> {
    pub(crate) native: NativeDriver,
    pub(crate) stream_create: unsafe extern "C" fn(*mut Handle, u32) -> i32,
    pub(crate) stream_destroy: PtrCall,
    pub(crate) stream_synchronize: PtrCall,
    pub(crate) begin: unsafe extern "C" fn(Handle, i32) -> i32,
    pub(crate) end: unsafe extern "C" fn(Handle, *mut Handle) -> i32,
    pub(crate) instantiate: unsafe extern "C" fn(*mut Handle, Handle, u64) -> i32,
    pub(crate) get_nodes: unsafe extern "C" fn(Handle, *mut Handle, *mut usize) -> i32,
    pub(crate) upload: unsafe extern "C" fn(Handle, Handle) -> i32,
    pub(crate) launch: unsafe extern "C" fn(Handle, Handle) -> i32,
    pub(crate) destroy: PtrCall,
    pub(crate) exec_destroy: PtrCall,
    pub(crate) _cuda: PhantomData<&'cuda Cuda>,
}

impl Cuda {
    pub unsafe fn graph_driver(&self) -> Result<GraphDriver<'_>, String> {
        Ok(GraphDriver {
            native: NativeDriver { launch: self.launch, copy: self.copy, clear: self.clear },
            stream_create: self.stream_create,
            stream_destroy: self.stream_destroy,
            stream_synchronize: self.stream_synchronize,
            begin: cuda::symbol(self.library, "cuStreamBeginCapture_v2")?,
            end: cuda::symbol(self.library, "cuStreamEndCapture")?,
            instantiate: cuda::symbol(self.library, "cuGraphInstantiateWithFlags")?,
            get_nodes: cuda::symbol(self.library, "cuGraphGetNodes")?,
            upload: cuda::symbol(self.library, "cuGraphUpload")?,
            launch: cuda::symbol(self.library, "cuGraphLaunch")?,
            destroy: cuda::symbol(self.library, "cuGraphDestroy")?,
            exec_destroy: cuda::symbol(self.library, "cuGraphExecDestroy")?,
            _cuda: PhantomData,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Driver { operation: &'static str, status: i32 },
    Submit(super::submit::Error),
    Poisoned,
}

fn checked(operation: &'static str, status: i32) -> Result<(), Error> {
    if status == 0 { Ok(()) } else { Err(Error::Driver { operation, status }) }
}

pub struct DeviceGraph<'driver, 'cuda> {
    driver: &'driver GraphDriver<'cuda>,
    stream: Handle,
    graph: Handle,
    executable: Handle,
    capturing: bool,
    poisoned: bool,
    commands: usize,
    nodes: usize,
    launches: u64,
}

impl<'driver, 'cuda> DeviceGraph<'driver, 'cuda> {
    /// Build once before the measured interval. No warm-up execution occurs:
    /// capture records operations without modifying primal or derivative words.
    /// Upload and its one preparation-only stream wait finish before return.
    ///
    /// Safety: `submit` must be the pinned submitter, and all handles and device
    /// addresses in `batch` must be valid in the current context. The allocations
    /// and modules must outlive every launch. External producers must finish (or
    /// be ordered by events on the launch stream) before the first launch.
    pub unsafe fn prepare(driver: &'driver GraphDriver<'cuda>, batch: &BoundBatch<'_>,
                          submit: Submit) -> Result<Self, Error> {
        Self::prepare_impl(driver, batch, submit, None)
    }

    /// Prepare without a host wait. Upload is ordered on the eventual launch
    /// stream. Capture itself contains no executing operations. Caller must
    /// launch on that same stream; the graph cannot outlive its bound resources.
    pub unsafe fn prepare_queued(driver: &'driver GraphDriver<'cuda>, batch: &BoundBatch<'_>,
                                 submit: Submit, stream: Handle) -> Result<Self, Error> {
        Self::prepare_impl(driver, batch, submit, Some(stream))
    }

    unsafe fn prepare_impl(driver: &'driver GraphDriver<'cuda>, batch: &BoundBatch<'_>,
                           submit: Submit, upload_stream: Option<Handle>) -> Result<Self, Error> {
        let mut g = Self { driver, stream: ptr::null_mut(), graph: ptr::null_mut(),
            executable: ptr::null_mut(), capturing: false, poisoned: false,
            commands: batch.len(), nodes: 0, launches: 0 };
        // CU_STREAM_NON_BLOCKING, so capture never uses the legacy null stream.
        checked("cuStreamCreate", (driver.stream_create)(&mut g.stream, 1))?;
        // THREAD_LOCAL: other binder threads may prepare their own allocations.
        checked("cuStreamBeginCapture_v2", (driver.begin)(g.stream, 1))?;
        g.capturing = true;
        let issued = batch.issue(&driver.native, submit, g.stream);
        // Always terminate capture, including a partial submission failure.
        let ended = (driver.end)(g.stream, &mut g.graph);
        g.capturing = false;
        issued.map_err(Error::Submit)?;
        checked("cuStreamEndCapture", ended)?;
        checked("cuGraphGetNodes", (driver.get_nodes)(g.graph, ptr::null_mut(), &mut g.nodes))?;
        checked("cuGraphInstantiateWithFlags", (driver.instantiate)(&mut g.executable, g.graph, 0))?;
        checked("cuGraphUpload", (driver.upload)(g.executable, upload_stream.unwrap_or(g.stream)))?;
        if upload_stream.is_none() {
            checked("cuStreamSynchronize preparation", (driver.stream_synchronize)(g.stream))?;
        }
        checked("cuStreamDestroy_v2 preparation", (driver.stream_destroy)(g.stream))?;
        g.stream = ptr::null_mut();
        Ok(g)
    }

    pub fn nodes(&self) -> usize { self.nodes }
    pub fn commands(&self) -> usize { self.commands }
    pub fn launches(&self) -> u64 { self.launches }

    /// The entire replay path: one driver call, no allocation, host readback,
    /// per-entry launch, graph update, or host wait. Errors poison this instance.
    ///
    /// Safety: stream, context, modules and allocations must still be valid.
    /// Order callers' input writes and output reads on this stream or via events.
    /// GPU completion/errors are checked by the enclosing window boundary gate.
    pub unsafe fn launch(&mut self, stream: Handle) -> Result<(), Error> {
        if self.poisoned { return Err(Error::Poisoned); }
        let status = (self.driver.launch)(self.executable, stream);
        if status != 0 {
            self.poisoned = true;
            return Err(Error::Driver { operation: "cuGraphLaunch", status });
        }
        self.launches += 1;
        Ok(())
    }
}

impl Drop for DeviceGraph<'_, '_> {
    fn drop(&mut self) { unsafe {
        if self.capturing {
            let _ = (self.driver.end)(self.stream, &mut self.graph);
        }
        if !self.executable.is_null() { let _ = (self.driver.exec_destroy)(self.executable); }
        if !self.graph.is_null() { let _ = (self.driver.destroy)(self.graph); }
        if !self.stream.is_null() { let _ = (self.driver.stream_destroy)(self.stream); }
    } }
}
