// SPDX-License-Identifier: Apache-2.0
//! Experimental sealed-plan input. Partial observations are not valid plans.
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Plan {
    pub schema: String,
    /// P1 (ad/phys): carried allocations that hold live-rebuilt device-address tables (see main.rs).
    #[serde(default)]
    pub address_bearing: Vec<u64>,
    pub complete: bool,
    pub source_pin: String,
    /// Producer certification that a barrier at each sampled step is valid.
    pub end_step_barrier_proven: bool,
    pub modules: Vec<Module>,
    /// P1 (ad/phys): zero-argument kernels the ordinary forward launched before the first replayed step.
    /// Their only effect is module-scope device state (`__device__` globals such as RUC's saturation table
    /// ruc_tbq_table, filled by ruc_tbq_init when the module loads at physics_init, step 0). That state is not
    /// a driver allocation, so the step-1 initial snapshot never held it, and without these launches every
    /// replayed module started from its image's static values (sea ice read a zero table: 38 allocations
    /// wrong at step 1, identically in every replay). Run in recorded order right after the modules load.
    #[serde(default)] pub module_initializers: Vec<ModuleInitializer>,
    /// P1 (ad/phys): each allocation's base address in the recording the reference hashes came from (recording
    /// A), as [id, base]. The forward's carried state holds device addresses (pointer tables, copy plans,
    /// relocated initial tables); the replay holds its own. Step hashes are taken in the forward's address space:
    /// every aligned 8-byte word that points into a live replay allocation is rewritten to the recorded address
    /// of the same allocation and offset before hashing (state_sha256 reports how many). A data word cannot pass
    /// this way unless the forward held exactly that address there.
    #[serde(default)] pub recorded_bases: Vec<[u64; 2]>,
    /// P1 (ad/phys): declared residue, excluded from the step compare the way native-undefined words are (the
    /// RUC tsnav precedent): byte ranges of carried allocations whose forward values are timing-based and differ
    /// run to run in the ordinary forward itself (TEMPO per-column clock64 cycle counts, the column order from an
    /// atomic scatter, its bucket starts, and the per-warp state/work scratch whose warp-to-column assignment
    /// follows that order). Each entry names its witness. The runner zeroes exactly these ranges in its
    /// forward-address view and compares with `masked_sha256` (the forward's boundary bytes, same ranges zeroed),
    /// so every other byte of the allocation is still compared. THE BREAKAGE THIS PREVENTS: allocations 414, 416,
    /// 417 and 446 failed the step hash in every replay although no model word depends on those ranges.
    #[serde(default)] pub excluded_residue: Vec<Residue>,
    pub allocations: Vec<Allocation>,
    /// multidevice-runner (split_plan.py --global-arena): [id, bytes] of every allocation of the UNSPLIT plan. A rank
    /// window then lays its shadow arena out exactly as the one-process window of the unsplit plan does (ascending id,
    /// 64-word padding), so its directions x, y (keyed by arena word) and its reverse words are word-for-word comparable
    /// with the one-process run: the proof that the multi-device reverse is the same operator. Proof runs only: every
    /// rank then holds the whole arena.
    #[serde(default)] pub peer_arena_sizes: Option<Vec<[u64; 2]>>,
    /// multidevice-runner: host owners whose hash is logged, not checked, in a rank process (split_plan.py).
    #[serde(default)] pub peer_host_unchecked: Vec<u64>,
    /// Each host allocation starts from an authenticated initial model input.
    /// Ownership, aliasing and completeness require an independent census.
    #[serde(default)] pub host_allocations: Vec<HostAllocation>,
    #[serde(default)] pub host_carried: Vec<u64>,
    #[serde(default)] pub streams: Vec<Stream>,
    #[serde(default)] pub events: Vec<Event>,
    pub carried: Vec<u64>,
    /// Optional logical carried extents at each completed step. When absent,
    /// carried IDs denote entire physical allocations. When present, the
    /// census producer must prove these are all carried extents, including
    /// aliases and host state, before sealing the plan.
    #[serde(default)] pub carried_views: Option<Vec<Vec<CarriedView>>>,
    /// Every independently enumerated physical owner live at each fenced
    /// step, including scratch and host owners, for checkpoint consumers.
    /// Merely supplying this list does not prove the census is complete.
    #[serde(default)] pub boundaries: Vec<CheckpointBoundary>,
    pub blocks: HashMap<String, Vec<Operation>>,
    /// Optional ordinary-forward hashes for the default replay command.
    #[serde(default)] pub reference: Option<String>,
    #[serde(default)]
    pub cadence: Vec<Vec<String>>,
    #[serde(default)]
    pub physics: Option<super::replay_frame::PhysicsPlan>,
}
#[derive(Deserialize)]
pub struct Module {
    pub id: String,
    pub path: String,
    pub sha256: String,
    /// When set to gzip, `bytes` is the exact uncompressed image size.
    #[serde(default)] pub compression: Option<String>,
    #[serde(default)] pub bytes: Option<usize>,
}
#[derive(Deserialize)]
pub struct Allocation {
    pub id: u64,
    pub bytes: usize,
    pub initial: Option<Blob>,
    /// The producer must certify scratch is overwritten before it is read.
    #[serde(default)] pub overwritten_before_read: bool,
    /// P1 (ad/phys): words of the initial bytes that are recorded device addresses (pointer tables the forward
    /// built before step 1). Uploaded verbatim they point into the recording's address space: update_diag_pool
    /// faulted (CUDA 700) at its first replayed launch, step 2, reading allocation 125's output table.
    #[serde(default)] pub relocations: Vec<Relocation>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relocation { pub byte_offset: usize, pub target: Slice, pub recorded_value: u64 }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostAllocation { pub id: u64, pub bytes: usize, pub initial: Blob, pub origin: String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostView {
    pub allocation: u64, pub offset: i64, pub shape: Vec<usize>, pub strides: Vec<i64>, pub item_bytes: usize,
}
impl HostView {
    pub fn view(&self) -> super::host_array::View {
        super::host_array::View { offset: self.offset, shape: self.shape.clone(),
                                  strides: self.strides.clone(), item_bytes: self.item_bytes }
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostOrder { C, Fortran }
impl HostOrder {
    pub fn order(&self) -> super::host_array::Order {
        match self { Self::C => super::host_array::Order::C, Self::Fortran => super::host_array::Order::Fortran }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleInitializer {
    pub module: String,
    pub function: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub recorded_sequence: u64,
}
#[derive(Deserialize)]
pub struct Stream { pub id: u64, pub flags: u32 }
#[derive(Deserialize)]
pub struct Event { pub id: u64, pub flags: u32 }
#[derive(Clone, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum Blob {
    Hex { bytes: String, sha256: String },
    File { path: String, sha256: String },
    Gzip { path: String, sha256: String, bytes: usize },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub struct Slice { pub allocation: u64, pub offset: usize }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarriedView { pub id: u64, pub target: Slice, pub bytes: usize }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointBoundary {
    pub live: Vec<CarriedView>,
    pub host_live: Vec<u64>,
}
#[derive(Deserialize)]
pub struct Patch { pub byte_offset: usize, pub target: Slice }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Residue {
    pub allocation: u64, pub step: usize, pub ranges: Vec<ResidueRange>, pub masked_sha256: String,
    pub reason: String, pub witness: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidueRange { pub offset: usize, pub bytes: usize, pub array: String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldRange { pub destination: Slice, pub source_offset: usize, pub bytes: usize }
/// Shape/stride words are bound to live host state, not captured launch bytes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadiationView { pub target: Option<Slice>, pub shape: Vec<Slice>, pub strides: Vec<Slice> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadiationRow { pub dst: RadiationView, pub src: Option<RadiationView>,
                          pub fill_f32: Option<Slice> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportRow { pub dst: RadiationView, pub src: RadiationView }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedArgumentSource {
    /// This is an assertion by the plan producer, not a proof of immutability.
    pub class: String,
    pub source: String,
    pub source_sha256: String,
    pub line: usize,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Argument {
    Device { target: Slice },
    /// Cast the current live host scalar at each launch. The producer must
    /// prove its source owner, updates, conversion and selected branch.
    HostScalar { source: Slice, cast: super::host_scalar::Cast },
    /// CuPy CArray by value. Each shape and stride is a live host i64; the
    /// target and view provenance must be established by the plan producer.
    CupyArray { target: Option<Slice>, shape: Vec<Slice>, strides: Vec<Slice>, item_bytes: usize },
    /// CIndexer only for an image/source pair with a proved copy-loop
    /// overwrite of its index scratch. Shape words are read live from host.
    CupyCopyIndexer { shape: Vec<Slice>, source: FixedArgumentSource },
    /// Pointer table selected from the current TEMPO cache, not a trace value.
    TempoTable { binding: String },
    /// A CUDA module symbol is rebound in this process; its captured address
    /// is never a replay input. A separate census must classify its state.
    ModuleGlobal { module: String, symbol: String, offset: usize, expected_bytes: usize },
    Bytes { bytes_hex: String, #[serde(default)] fixed_source: Option<FixedArgumentSource> },
    Patched { bytes_hex: String, pointers: Vec<Patch>,
              #[serde(default)] fixed_source: Option<FixedArgumentSource> },
    StepInteger { scale: i64, bias: i64, width: u8 },
    StepSeconds { start: f64, interval: f64 },
    PhysicsFrame { name: String },
    /// Values derived from the active live RK/acoustic loop scope.
    RkFrame { name: String },
}
/// G2: the LW wrapper launch shapes the plan carries for one rrtmg_lwrad_device call (nc columns, nm model
/// layers, batches = [layer count, columns] per mpas_lw_prepare launch, in order).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LwLayerGroups { pub nc: usize, pub nm: i32, pub batches: Vec<[i64; 2]> }
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Operation {
    /// Construct the source RK schedule from current host configuration.
    RkSchedule { dt_f64: Slice, order_i64: Slice, acoustic_substeps_i64: Slice,
                 dynamics_splits_i64: Slice, body: String },
    /// Execute the source split, stage and acoustic loops as nested blocks.
    RkSplits { body: String },
    RkStages { body: String },
    RkSubsteps { body: String },
    Kernel { stream: u64, module: String, function: String, grid: [u32; 3],
             block: [u32; 3], shared_bytes: u32, arguments: Vec<Argument> },
    /// Source-derived descending ozone-interpolation loop. Array shapes are
    /// live host metadata; the producer proves their owners and contiguity.
    VinterpOzone { stream: u64, module: String,
        pmid: Slice, pin: Slice, ozone: Slice, starts: Slice, upper: Slice, out: Slice,
        pmid_shape: [Slice; 2], pin_size: Slice, ozone_shape: [Slice; 2] },
    /// Rebuild the unpack kernel arguments from current views and live fills.
    RadiationUnpack { stream: u64, module: String, rows: Vec<RadiationRow> },
    /// Rebuild the post-transport copy frame from current device views.
    TransportCopy { stream: u64, module: String, rows: Vec<TransportRow> },
    /// Derive the add kernel's scalar frame from current two-dimensional views.
    AddInplace { stream: u64, module: String, target: Slice, increment: Slice,
                 target_shape: [Slice; 2], increment_shape: [Slice; 2] },
    DeviceCopy { stream: u64, destination: Slice, source: Slice, bytes: usize },
    /// multidevice-runner (peer.rs): the source half of a device copy that crosses a rank boundary of the decomposed
    /// forward (tools/multidevice/split_plan.py): the bytes go to rank `peer`'s PeerRecv of the same channel.
    PeerSend { stream: u64, channel: u64, peer: usize, source: Slice, bytes: usize },
    /// multidevice-runner: the destination half: `bytes` from rank `peer`'s PeerSend of the same channel.
    PeerRecv { stream: u64, channel: u64, peer: usize, destination: Slice, bytes: usize },
    /// S14: a device-to-device copy into a CUDA module global of a recorded image (fast-numerics TEMPO writes its
    /// binary32 parameter block `tempo_k` this way, tempo_v303.py _set_scalars, before every launch that reads it).
    /// The symbol is looked up in the replay's own loaded image; the recorded device address is never an input.
    ModuleGlobalCopy { stream: u64, module: String, symbol: String, offset: usize, source: Slice, bytes: usize },
    Clear { stream: u64, destination: Slice, bytes: usize, value: u8 },
    Fill16 { stream: u64, destination: Slice, count: usize, value: u16 },
    Fill32 { stream: u64, destination: Slice, count: usize, value: u32 },
    /// Scratch must be proved overwritten before any read on each lifetime.
    Allocate { stream: u64, id: u64, bytes: usize, asynchronous: bool,
               overwritten_before_read: bool },
    Free { stream: u64, id: u64, asynchronous: bool },
    /// Preserve an observed successful CUDA free of the null device pointer.
    FreeNull { stream: u64, asynchronous: bool },
    /// The captured blob is authenticated before its explicitly identified
    /// device addresses are relocated for this process.
    Upload { stream: u64, destination: Slice, data: Blob,
             #[serde(default)] pointers: Vec<Patch>,
             /// Physics uploads may use captured bytes only for certified
             /// immutable inputs, never as a substitute for host results.
             #[serde(default)] origin: Option<String> },
    /// COMMON-AD 103.3 milestone-1 holds: the words one held slot changed in the ordinary forward
    /// (tracer HeldSlotWrite records, packed into one blob per slot call), each range uploaded to its
    /// live destination. Origin must be a held label; never used for a live slot.
    /// P1 (ad/phys): `pointers` are the blob words that are recording-A device addresses (a held slot built a
    /// pointer table, e.g. land's 11-entry table in allocation 172 at step 1); each is checked against the
    /// recorded base of its target and relocated to the replay's address. Uploaded verbatim they pointed into
    /// the recording's address space, and two of them (0x1000ad32000, 0x1000ad4ee00) fell inside a different
    /// replay allocation, so table 172 failed the step hash at step 1 (16 words of pool root 347 the same way).
    HeldUploadRanges { stream: u64, data: Blob, ranges: Vec<HeldRange>, origin: String,
                       #[serde(default)] pointers: Vec<Patch> },
    /// Source bytes are calculated from the live Rust frame on every call.
    FrameUpload { stream: u64, destination: Slice, name: String },
    /// Pack the currently live host view; recorded later outputs are forbidden.
    HostUpload { stream: u64, destination: Slice, view: HostView, order: HostOrder },
    /// Publish a current device result to a live, nonoverlapping host view.
    HostPublish { stream: u64, source: Slice, view: HostView, order: HostOrder },
    /// An address table is built from live allocation bindings, not captured
    /// host output. The rule names the pinned forward expression whose ordered
    /// entries the plan producer must independently establish. For the CuPy
    /// same-size stack path the producer must also prove that this path is
    /// selected from live inputs; this operation does not choose the branch.
    PointerTableUpload { stream: u64, destination: Slice, rule: String, entries: Vec<Slice> },
    /// Derive TEMPO's pointer words and use the live 64-entry cache. A new
    /// destination is consumed only on a miss; kernels use the binding name.
    TempoPointerTableBind { stream: u64, binding: String, destination: Slice,
                            entries: Vec<Option<super::tempo::View>>,
                            expected_shape: Option<Vec<usize>>, device: u64, optional: bool },
    /// Recompute the source's named carrier order and required-field checks.
    /// Each flag points to a live native-endian i32 in host state. Ownership,
    /// live names and the order of the eight calls still need a producer proof.
    TempoNamedTableBind { stream: u64, binding: String, destination: Slice,
                          carriers: super::tempo::selection::Carriers,
                          flags: [Slice; 2], group: String },
    /// Drop the caller's local reference; the cache itself can retain a table.
    TempoTableRelease { binding: String },
    /// Clear the shared SAVE flag after a successful TEMPO main launch. The
    /// producer must establish both placement and the live owner's binding.
    TempoMainComplete { first_call_main: Slice },
    /// Construct the pinned sea-ice driver's ordered pointer table from the
    /// live named arrays and its current dummy allocation. Missing ocean-flux
    /// entries are null; other missing entries point at the live dummy.
    SeaicePointerTableUpload { stream: u64, destination: Slice,
                               arrays: HashMap<String, Slice>, dummy: Slice },
    /// Assemble the RUC boundary table from live named views and presence
    /// flags. The producer must independently prove each binding and extent.
    RucPointerTableUpload { stream: u64, destination: Slice, entry: String,
                            arguments: HashMap<String, super::ruc::View>,
                            flags: HashMap<String, super::ruc::View>,
                            tile_stop: Option<super::ruc::View> },
    /// Build RRFS descriptors from live inputs and publish the source's
    /// cached launch state into a carried host owner.
    RrfsCopyPlanUpload { stream: u64, destination: Slice, rows: Vec<super::copy_plan::RowSpec>, cache: Slice },
    /// Run the already-built table using its carried cached launch state.
    RrfsCopyPlanKernel { stream: u64, table: Slice, cache: Slice,
                         normal_module: String, tiled_module: String },
    /// Invoke the original SHA-pinned CuPy CUB host routine with live buffers.
    /// Counts are read from current host state at each call. The producer must
    /// prove their owner and updates, operation, dtype, and buffer bindings.
    NativeCubReduce { stream: u64, library_path: String, library_sha256: String,
                      workspace: Slice, input: Slice, output: Slice,
                      count: Slice, op: i32, dtype: i32 },
    NativeCubSegmentedReduce { stream: u64, library_path: String, library_sha256: String,
                               workspace: Slice, input: Slice, output: Slice,
                               segments: Slice, segment_size: Slice, op: i32, dtype: i32 },
    /// The original stable one-dimensional argsort mutates its live data copy
    /// and writes size_t indices. Shape and buffer ownership need separate proof.
    NativeThrustArgsort { stream: u64, library_path: String, library_sha256: String,
                          workspace: Slice, input: Slice, indices: Slice,
                          count: Slice, dtype: i32 },
    /// CuPy's observed float32 contiguous-copy transpose, submitted through
    /// the original SGEAM host routine. Producer proves live geometry/ownership.
    NativeCublasTranspose { stream: u64, library_path: String, library_sha256: String,
                            input: Slice, output: Slice, m: Slice, n: Slice },
    /// Allowed only once the producer has proved the host copy is output-only.
    /// `checked_value`: the producer proved the bytes are a pure function of plan inputs
    /// (expected_hex); the replay downloads them and refuses on any difference.
    DeviceToHost { stream: u64, source: Slice, bytes: usize, asynchronous: bool,
                   host_consumer: String, #[serde(default)] expected_hex: Option<String>,
                   /// `checked_zero_words`: exactly these u32 words are zero; empty = at most one is.
                   #[serde(default)] expected_zero_words: Option<Vec<usize>>,
                   /// G2 (P1 milestone 2): on the LW wrapper's histogram read-back (rrtmg_lw_v842.py:1019) the
                   /// recorded wrapper batches (layer count, columns) in launch order; the replay rebuilds them from
                   /// the live histogram with layer_groups.rs and refuses on any difference.
                   #[serde(default)] layer_groups: Option<LwLayerGroups> },
    /// The recorded event handle identifies a lifetime, not a native pointer.
    Synchronization { stream: u64, operation: String, event_handle_recorded: u64, flags: u32 },
    EventRecord { stream: u64, event: u64 },
    EventWait { stream: u64, event: u64, flags: u32 },
    /// Choose both possible blocks from a live host byte. For a device result,
    /// the producer must first publish it to this host allocation.
    HostBranch { condition: Slice, if_zero: String, if_nonzero: String },
    HostBoundary { name: String },
}
