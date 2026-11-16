// SPDX-License-Identifier: Apache-2.0
//! Reverse scheduling resources. No arithmetic or CPU implementation of a GPU stage.
use crate::{cuda::{checked, Cuda, GraphApi, PtrCall, Record, Wait}, dev, pinned::PinnedBuf};
use std::{collections::HashMap, ffi::c_void, ptr};

/// A private nonblocking stream, with explicit dependencies on the legacy queue.
/// Copies of driver function pointers remain valid while the owning Execution lives.
struct Queue {
    stream: *mut c_void,
    ready: *mut c_void,
    done: *mut c_void,
    record: Record,
    wait: Wait,
    sync: PtrCall,
    event_sync: PtrCall,
    stream_destroy: PtrCall,
    event_destroy: PtrCall,
}
impl Queue {
    unsafe fn new(cuda: &Cuda) -> Result<Self, String> {
        let mut q = Self { stream: ptr::null_mut(), ready: ptr::null_mut(), done: ptr::null_mut(),
            record: cuda.event_record, wait: cuda.stream_wait_event, sync: cuda.stream_synchronize,
            event_sync: cuda.event_synchronize, stream_destroy: cuda.stream_destroy, event_destroy: cuda.event_destroy };
        checked("cuStreamCreate reverse machinery", (cuda.stream_create)(&mut q.stream, 1))?;
        checked("cuEventCreate reverse ready", (cuda.event_create)(&mut q.ready, 2))?;
        checked("cuEventCreate reverse done", (cuda.event_create)(&mut q.done, 3))?;
        Ok(q)
    }
    unsafe fn after_legacy(&self) -> Result<(), String> {
        checked("cuEventRecord reverse producer", (self.record)(self.ready, ptr::null_mut()))?;
        checked("cuStreamWaitEvent reverse producer", (self.wait)(self.stream, self.ready, 0))
    }
    unsafe fn finish(&self) -> Result<(), String> {
        checked("cuEventRecord reverse done", (self.record)(self.done, self.stream))
    }
    unsafe fn before_legacy(&self) -> Result<(), String> {
        checked("cuStreamWaitEvent reverse consumer", (self.wait)(ptr::null_mut(), self.done, 0))
    }
}
impl Drop for Queue {
    fn drop(&mut self) { unsafe {
        // Also fences a partially enqueued copy on an error path.
        if !self.stream.is_null() { let _ = (self.sync)(self.stream); }
        for e in [self.ready, self.done] { if !e.is_null() { let _ = (self.event_destroy)(e); } }
        if !self.stream.is_null() { let _ = (self.stream_destroy)(self.stream); }
    } }
}

/// Owns the destination until DMA completes, including errors and early returns.
/// No host slice of its storage is exposed before wait().
pub struct Download {
    // Queue is dropped before the buffer, including unwinding from wait().
    queue: Queue,
    pin: Option<PinnedBuf>,
    set: PtrCall,
    context: usize,
}
// CUDA stream/event handles may move between threads; wait() makes their context current.
unsafe impl Send for Download {}
impl Download {
    pub unsafe fn start(cuda: &Cuda, context: *mut c_void, pin: PinnedBuf,
                        spans: &[(usize, u64, usize)], protect_primal: bool) -> Result<Self, String> {
        for &(off, _, n) in spans {
            if off.checked_add(n).is_none_or(|end| end > pin.len()) {
                return Err("DEFECT redo download exceeds its pinned buffer".into());
            }
        }
        let mut d = Self { queue: Queue::new(cuda)?, pin: Some(pin), set: cuda.ctx_set_current, context: context as usize };
        d.queue.after_legacy()?;
        let dst = d.pin.as_mut().unwrap().as_mut_ptr();
        for &(off, src, n) in spans { if n != 0 {
            checked("cuMemcpyDtoHAsync reverse redo", (cuda.download_async)(dst.add(off).cast(), src, n, d.queue.stream))?;
        } }
        d.queue.finish()?;
        // Segmented replay will overwrite primal allocations. Only that path
        // waits on the device; the one-pass reverse writes tape and shadows.
        if protect_primal { d.queue.before_legacy()?; }
        Ok(d)
    }
    pub fn wait(mut self) -> Result<PinnedBuf, String> { unsafe {
        checked("cuCtxSetCurrent redo worker", (self.set)(self.context as *mut c_void))?;
        checked("cuEventSynchronize redo worker", (self.queue.event_sync)(self.queue.done))?;
        Ok(self.pin.take().unwrap())
    } }
}
impl Drop for Download {
    fn drop(&mut self) { unsafe {
        if self.pin.is_some() {
            // A worker may fail to make the context current. Do not unpin a
            // possibly live DMA destination on that error path.
            let set = (self.set)(self.context as *mut c_void);
            let fence = if set == 0 { (self.queue.sync)(self.queue.stream) } else { set };
            if fence != 0 {
                eprintln!("DEFECT reverse download cleanup could not fence DMA (CUDA {fence}); retaining pinned destination");
                std::mem::forget(self.pin.take());
            }
        }
    } }
}

/// Keep borrowed checkpoint storage live through all queued copies, including
/// early errors and unwinding. One stream fence replaces per-allocation waits.
pub unsafe fn legacy_batch<T>(cuda: &Cuda, enqueue: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    fenced(cuda.stream_synchronize, enqueue)
}
unsafe fn fenced<T>(sync: PtrCall, enqueue: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    struct Fence(PtrCall, bool);
    impl Drop for Fence { fn drop(&mut self) { if self.1 { unsafe { let _ = (self.0)(ptr::null_mut()); } } } }
    let mut fence = Fence(sync, true);
    let result = enqueue();
    let completion = checked("cuStreamSynchronize checkpoint batch", sync(ptr::null_mut()));
    fence.1 = false;
    completion?;
    result
}

/// A cached executable retains topology, never permission to reuse its old
/// addresses. Every launch consumes a freshly lowered invocation epoch.
#[derive(Clone, Copy, Debug)]
pub struct BindingEpoch {
    pub tape: u64,
    pub scratch: u64,
    pub scratch_base: u64,
    pub scratch_bytes: usize,
}
struct BoundNodes {
    template: *mut c_void,
    destroy: PtrCall,
    nodes: Vec<*mut c_void>,
    shapes: Vec<(u32, usize, usize)>,
    context: *mut c_void,
    epoch: BindingEpoch,
}
impl Drop for BoundNodes {
    fn drop(&mut self) { unsafe {
        if !self.template.is_null() { let _ = (self.destroy)(self.template); }
    } }
}
pub struct BoundRun {
    pub nodes: usize,
    pub builds: u64,
    pub node_updates: usize,
}

struct Graph { raw: *mut c_void, destroy: PtrCall, bound: Option<BoundNodes> }
impl Drop for Graph { fn drop(&mut self) { unsafe { if !self.raw.is_null() { let _ = (self.destroy)(self.raw); } } } }

/// Ends capture on failure or unwind, so a failed recording cannot leave a
/// poisoned capture active in the runner's context.
struct Capture { api: GraphApi, stream: *mut c_void, active: bool }
impl Capture {
    unsafe fn end(mut self) -> Result<Graph, String> {
        let mut g = Graph { raw: ptr::null_mut(), destroy: self.api.destroy, bound: None };
        let status = (self.api.end)(self.stream, &mut g.raw);
        self.active = false;
        checked("cuStreamEndCapture reverse entry", status)?;
        Ok(g)
    }
}
impl Drop for Capture { fn drop(&mut self) { unsafe {
    if self.active {
        let mut g = ptr::null_mut();
        let _ = (self.api.end)(self.stream, &mut g);
        if !g.is_null() { let _ = (self.api.destroy)(g); }
    }
} } }

struct Graphs { queue: Queue, api: GraphApi, execs: HashMap<String, Graph>, bound_api: Option<crate::cuda::BoundGraphApi> }
impl Drop for Graphs { fn drop(&mut self) { unsafe {
    // Executables precede queue teardown. Any previous launch has finished.
    let _ = (self.queue.sync)(self.queue.stream);
    self.execs.clear();
} } }

#[derive(Default)]
pub struct GraphCache {
    graphs: Option<Graphs>,
    pub captures: u64,
    pub updates: u64,
    pub builds: u64,
    pub rebuilds: u64,
    pub skipped: u64,
    bound_poisoned: bool,
}
impl GraphCache {
    /// Reuse this cache's queue and executable storage for an already validated
    /// straight-line segment. No stream capture or per-entry kernel launch.
    ///
    /// Safety: caller owns all resolved addresses/modules until the legacy
    /// consumer fence completes. All producers, reuse and retirement must be
    /// ordered on that legacy stream. `epoch.scratch` increases for every new
    /// binding and `epoch.tape` increases on every walk_back/segment. No two host
    /// threads may update this cache. CUDA copies node parameter values; updates
    /// affect future launches only, including when a prior launch is in flight.
    pub unsafe fn run_bound(&mut self, cuda: &Cuda, context: *mut c_void, key: &str,
                            batch: &crate::submit::BoundBatch<'_>, epoch: BindingEpoch) -> Result<BoundRun, String> {
        if self.bound_poisoned { return Err("DEFECT bound graph cache poisoned".into()); }
        self.bound_poisoned = true;
        let result = self.run_bound_inner(cuda, context, key, batch, epoch);
        if result.is_ok() { self.bound_poisoned = false; }
        result
    }

    unsafe fn run_bound_inner(&mut self, cuda: &Cuda, context: *mut c_void, key: &str,
                              batch: &crate::submit::BoundBatch<'_>, epoch: BindingEpoch) -> Result<BoundRun, String> {
        use crate::submit::GraphNodeFrame;
        if context.is_null() || epoch.tape == 0 || epoch.scratch == 0 || epoch.scratch_base == 0 || epoch.scratch_bytes == 0 {
            return Err("DEFECT bound graph missing context or binding epoch".into());
        }
        // Resolve and validate the complete frame before changing any node.
        let frames = batch.graph_parameters().map_err(|e| format!("DEFECT bound graph parameters: {e:?}"))?;
        let shapes: Vec<_> = frames.nodes.iter().map(GraphNodeFrame::shape).collect();
        if self.graphs.is_none() {
            self.graphs = Some(Graphs { queue: Queue::new(cuda)?, api: cuda.graph_api()?,
                execs: HashMap::new(), bound_api: None });
        }
        let g = self.graphs.as_mut().unwrap();
        if g.bound_api.is_none() { g.bound_api = Some(cuda.bound_graph_api()?); }
        let api = g.bound_api.unwrap();
        let mut run = BoundRun { nodes: shapes.len(), builds: 0, node_updates: 0 };
        if let Some(exec) = g.execs.get_mut(key) {
            let bound = exec.bound.as_mut().ok_or("DEFECT bound graph key belongs to capture path")?;
            if bound.context != context || bound.shapes != shapes || bound.epoch.tape > epoch.tape || bound.epoch.scratch >= epoch.scratch {
                return Err("DEFECT bound graph changed topology/context or stale binding epoch".into());
            }
            // Update ALL nodes even if their address bits compare equal. Arena
            // addresses can be recycled for a different logical allocation.
            // Any failed update poisons the cache and no launch is attempted.
            for (&node, frame) in bound.nodes.iter().zip(&frames.nodes) {
                match frame {
                    GraphNodeFrame::Kernel(p, _) => checked("cuGraphExecKernelNodeSetParams bound", (api.set_kernel)(exec.raw, node, p))?,
                    GraphNodeFrame::Clear(p) => checked("cuGraphExecMemsetNodeSetParams bound", (api.set_clear)(exec.raw, node, p, context))?,
                }
                run.node_updates += 1;
            }
            bound.epoch = epoch;
            self.updates += 1;
        } else {
            // The selected function has one fixed topology per cache. Keep the
            // original graph alive: exec node updates identify its actual nodes.
            if g.execs.len() >= 64 { return Err("DEFECT bound graph cache capacity exhausted".into()); }
            let mut bound = BoundNodes { template: ptr::null_mut(), destroy: g.api.destroy,
                nodes: Vec::with_capacity(shapes.len()), shapes, context, epoch };
            checked("cuGraphCreate bound", (api.create)(&mut bound.template, 0))?;
            for frame in &frames.nodes {
                let mut node = ptr::null_mut();
                let prior = bound.nodes.last();
                let deps = prior.map_or(ptr::null(), |p| p as *const _);
                let ndep = usize::from(prior.is_some());
                match frame {
                    GraphNodeFrame::Kernel(p, _) => checked("cuGraphAddKernelNode bound", (api.add_kernel)(&mut node, bound.template, deps, ndep, p))?,
                    GraphNodeFrame::Clear(p) => checked("cuGraphAddMemsetNode bound", (api.add_clear)(&mut node, bound.template, deps, ndep, p, context))?,
                }
                bound.nodes.push(node);
            }
            let mut actual = 0;
            checked("cuGraphGetNodes bound", (api.get_nodes)(bound.template, ptr::null_mut(), &mut actual))?;
            if actual != frames.nodes.len() { return Err("DEFECT bound graph node count mismatch".into()); }
            let mut exec = Graph { raw: ptr::null_mut(), destroy: g.api.exec_destroy, bound: None };
            checked("cuGraphInstantiateWithFlags bound", (g.api.instantiate)(&mut exec.raw, bound.template, 0))?;
            exec.bound = Some(bound);
            g.execs.insert(key.to_owned(), exec);
            self.builds += 1;
            run.builds = 1;
        }
        // Reuse revmach-c's exact producer/consumer lifetime boundary. These
        // are two device event waits, not CPU synchronization calls.
        g.queue.after_legacy()?;
        checked("cuGraphLaunch bound", (g.api.launch)(g.execs[key].raw, g.queue.stream))?;
        g.queue.finish()?;
        g.queue.before_legacy()?;
        Ok(run)
    }

    /// Capture an audited dispatch with its current bindings. Recapture plus
    /// exec-update changes all addresses and scalar values before every launch.
    /// This caches executable entry graphs, not the host bind or a whole step.
    pub unsafe fn run(&mut self, cuda: &Cuda, key: &str, dispatch: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
        if self.graphs.is_none() {
            let api = cuda.graph_api().map_err(|e| format!("DEFECT reverse graph API: {e}"))?;
            self.graphs = Some(Graphs { queue: Queue::new(cuda)?, api, execs: HashMap::new(), bound_api: None });
        }
        let g = self.graphs.as_mut().unwrap();
        g.queue.after_legacy()?;
        // THREAD_LOCAL capture mode. No legacy launch is issued until capture ends.
        checked("cuStreamBeginCapture reverse entry", (g.api.begin)(g.queue.stream, 1))?;
        let capture = Capture { api: g.api, stream: g.queue.stream, active: true };
        {
            let _stream = dev::DispatchStream::enter(g.queue.stream)?;
            dispatch()?;
        }
        let graph = capture.end()?;
        self.captures += 1;
        let mut replace = !g.execs.contains_key(key);
        if let Some(exec) = g.execs.get(key) {
            let (mut node, mut result) = (ptr::null_mut(), 0i32);
            let status = (g.api.update)(exec.raw, graph.raw, &mut node, &mut result);
            // CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE means changed topology or
            // node parameters cannot be updated. Instantiate the NEW graph;
            // never replay the previous graph after a failed update.
            if status == 910 {
                replace = true;
                self.rebuilds += 1;
                eprintln!("reverse_graph_rebuild function={key} update_result={result}");
            } else {
                checked("cuGraphExecUpdate reverse entry", status)?;
                if result != 0 { return Err(format!("DEFECT reverse graph update result {result} without update failure")); }
                self.updates += 1;
            }
        }
        if replace {
            let mut exec = Graph { raw: ptr::null_mut(), destroy: g.api.exec_destroy, bound: None };
            checked("cuGraphInstantiateWithFlags reverse entry", (g.api.instantiate)(&mut exec.raw, graph.raw, 0))?;
            g.execs.insert(key.to_owned(), exec);
            self.builds += 1;
        }
        checked("cuGraphLaunch reverse entry", (g.api.launch)(g.execs[key].raw, g.queue.stream))?;
        g.queue.finish()?;
        g.queue.before_legacy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, panic::{catch_unwind, AssertUnwindSafe}};

    thread_local! { static FENCES: Cell<u32> = const { Cell::new(0) }; }
    unsafe extern "C" fn count_fence(stream: *mut c_void) -> i32 {
        if !stream.is_null() { return 1; }
        FENCES.with(|n| n.set(n.get() + 1));
        0
    }

    #[test]
    fn checkpoint_error_fences_before_borrowed_storage_returns() {
        FENCES.with(|n| n.set(0));
        let r: Result<(), String> = unsafe { fenced(count_fence, || Err("copy failed".into())) };
        assert_eq!(r.unwrap_err(), "copy failed");
        assert_eq!(FENCES.with(Cell::get), 1);
    }

    #[test]
    fn checkpoint_unwind_fences_and_dispatch_scope_restores() {
        FENCES.with(|n| n.set(0));
        let r = catch_unwind(AssertUnwindSafe(|| unsafe {
            let _: Result<(), String> = fenced(count_fence, || {
                let _scope = dev::DispatchStream::enter(1usize as *mut c_void).unwrap();
                assert!(dev::DispatchStream::enter(2usize as *mut c_void).is_err());
                panic!("injected enqueue panic");
            });
        }));
        assert!(r.is_err());
        assert!(dev::dispatch_stream().is_null());
        assert_eq!(FENCES.with(Cell::get), 1);
    }

    /// This is a driver test, not a physics gate or a performance measurement.
    /// Build outside a GPU lease; run this exact ignored test on a machine with an NVIDIA card.
    #[test]
    #[ignore = "requires a GPU lease (an NVIDIA card)"]
    fn cuda_graph_bindings_download_order_and_capture_cleanup() -> Result<(), String> { unsafe {
        struct Device { cuda: Cuda, ctx: *mut c_void, buffers: Vec<u64> }
        impl Drop for Device { fn drop(&mut self) { unsafe {
            let _ = (self.cuda.ctx_set_current)(self.ctx);
            let _ = (self.cuda.context_synchronize)();
            for &p in &self.buffers { let _ = (self.cuda.free)(p); }
            let _ = (self.cuda.context_destroy)(self.ctx);
        } } }
        let cuda = Cuda::load()?;
        checked("test cuInit", (cuda.init)(0))?;
        let mut device = 0;
        checked("test cuDeviceGet", (cuda.device_get)(&mut device, 0))?;
        let mut ctx = ptr::null_mut();
        checked("test cuCtxCreate", (cuda.context_create)(&mut ctx, 0, device))?;
        let mut d = Device { cuda, ctx, buffers: Vec::new() };
        const N: usize = 1 << 20;
        for _ in 0..4 {
            let mut p = 0;
            checked("test cuMemAlloc", (d.cuda.allocate)(&mut p, N))?;
            d.buffers.push(p);
        }
        let (a, b, c, z) = (d.buffers[0], d.buffers[1], d.buffers[2], d.buffers[3]);
        let cuda = &d.cuda;
        let mut graphs = GraphCache::default();
        for step in 0..8u8 {
            let (src, dst) = if step % 2 == 0 { (a, b) } else { (b, c) };
            checked("test producer", (cuda.clear)(src, step + 1, N, ptr::null_mut()))?;
            graphs.run(cuda, "same-entry", || {
                dev::copy(cuda, dst, src, N)?;
                dev::zero(cuda, z, 128 + step as usize * 4)
            })?;
            // A legacy consumer must see the current graph's destination.
            let got = dev::download(cuda, dst, N)?;
            assert!(got.iter().all(|&v| v == step + 1), "stale graph binding at step {step}");
        }
        assert_eq!(graphs.updates, 7);
        // Incompatible topology must instantiate the new graph, never replay old addresses.
        graphs.run(cuda, "same-entry", || { dev::zero(cuda, a, N) })?;
        assert!(graphs.rebuilds > 0);
        assert!(dev::download(cuda, a, N)?.iter().all(|&v| v == 0));

        let err = graphs.run(cuda, "abort", || {
            dev::zero(cuda, c, N)?;
            Err("injected dispatch failure".into())
        }).unwrap_err();
        assert_eq!(err, "injected dispatch failure");
        assert!(dev::dispatch_stream().is_null());
        let err = graphs.run(cuda, "host-drain", || dev::sync(cuda)).unwrap_err();
        assert!(err.contains("DEFECT"));
        graphs.run(cuda, "after-abort", || dev::zero(cuda, c, N))?;
        assert!(dev::download(cuda, c, N)?.iter().all(|&v| v == 0));

        // The segmented download must retain the pre-restore state even when
        // the overwrite is queued before the compare worker waits.
        checked("test pre-download state", (cuda.clear)(a, 23, N, ptr::null_mut()))?;
        let download = Download::start(cuda, ctx, PinnedBuf::new(cuda, N)?, &[(0, a, N)], true)?;
        checked("test next replay overwrite", (cuda.clear)(a, 91, N, ptr::null_mut()))?;
        let pin = std::thread::spawn(move || download.wait()).join().expect("download worker panicked")?;
        assert!(pin.iter().all(|&v| v == 23));
        assert!(dev::download(cuda, a, N)?.iter().all(|&v| v == 91));
        drop(pin);

        // One-pass mode omits the consumer wait and overlaps independent reverse work.
        let download = Download::start(cuda, ctx, PinnedBuf::new(cuda, N)?, &[(0, a, N)], false)?;
        graphs.run(cuda, "independent-reverse", || dev::zero(cuda, b, N))?;
        let pin = std::thread::spawn(move || download.wait()).join().expect("download worker panicked")?;
        assert!(pin.iter().all(|&v| v == 91));
        drop(pin);

        let mut pin = PinnedBuf::new(cuda, N)?;
        let batch: Result<(), String> = legacy_batch(cuda, || {
            checked("test partial batch", (cuda.download_async)(pin.as_mut_ptr().cast(), a, N, ptr::null_mut()))?;
            Err("injected batch failure".into())
        });
        assert_eq!(batch.unwrap_err(), "injected batch failure");
        assert!(pin.iter().all(|&v| v == 91));
        let bad = Download::start(cuda, ctx, pin, &[(N - 1, a, 2)], false);
        assert!(matches!(bad, Err(e) if e.contains("exceeds")));
        eprintln!("reverse_machinery_driver_test captures={} updates={} builds={} rebuilds={} bytes_per_buffer={N}",
            graphs.captures, graphs.updates, graphs.builds, graphs.rebuilds);
        Ok(())
    } }
}
