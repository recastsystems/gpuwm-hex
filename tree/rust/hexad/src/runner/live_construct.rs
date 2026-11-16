// SPDX-License-Identifier: Apache-2.0
//! Window metadata from resolved live commands. No recorder or payload reader.
//! Generation numbers are lifetime identities, NOT per-write value versions.
use super::*;
use crate::live_owner::Owner;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Span {
    pub owner: u64,
    pub generation: u64,
    pub offset: usize,
    /// Conservative complete accessed extent, not merely the pointer word.
    pub bytes: usize,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OwnerMeta {
    pub owner: u64,
    pub generation: u64,
    pub bytes: usize,
    pub initial: bool,
    pub constant: bool,
    pub address_bearing: bool,
    /// Pointer-bearing byte offsets used only for direction masking. The
    /// forward/checkpoint provider relocates their contents, not this builder.
    pub pointer_offsets: Vec<usize>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Patch {
    pub byte_offset: usize,
    pub target: Span,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Argument {
    Device { target: Span },
    Bytes { bytes_hex: String },
    Patched { bytes_hex: String, pointers: Vec<Patch> },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Command {
    Kernel { stream: u64, module: String, function: String, grid: [u32; 3],
        block: [u32; 3], shared_bytes: u32, arguments: Vec<Argument> },
    DeviceCopy { stream: u64, destination: Span, source: Span, bytes: usize },
    Clear { stream: u64, destination: Span, bytes: usize, value: u8 },
    Fill16 { stream: u64, destination: Span, count: usize, value: u16 },
    Fill32 { stream: u64, destination: Span, count: usize, value: u32 },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Row {
    /// Stable producer command identity; not a CUDA function address.
    pub key: String,
    pub command: Command,
    /// Includes indirect writes; never inferred from the pointer arguments.
    pub writes: Vec<Span>,
    /// Pre + post + augmentation + alignment upper bound, supplied by planner.
    pub tape_upper_bytes: usize,
    /// Labels only; arguments already contain all resolved loop scalars.
    pub rk: Option<[i64; 3]>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Step {
    pub step: usize,
    pub commands: Vec<Row>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Input {
    pub schema: String,
    pub owners: Vec<OwnerMeta>,
    pub steps: Vec<Step>,
    /// Boundary 0, then one after each step: complete live device extents.
    /// This is an inventory, not checkpoint bytes or a host rewind certificate.
    pub boundaries: Vec<Vec<Span>>,
    /// Full module identity -> C typedef T, obtained from the live compiler.
    pub concat_types: HashMap<String, String>,
}

pub(crate) struct LiveProgram {
    // Metadata adapter for the existing initializer only; NEVER execute this plan.
    pub(super) plan: Plan,
    pub(super) fixed: HashSet<u64>,
    pub(super) concat_types: HashMap<String, String>,
    pub input: Input,
    requests: Vec<Vec<Value>>,
}
fn target(s: &Span, owners: &BTreeMap<u64, OwnerMeta>) -> Result<Value, String> {
    let o = owners.get(&s.owner).ok_or("unknown live span owner")?;
    if o.generation != s.generation { return Err("stale live span generation".into()); }
    if s.offset > o.bytes || s.bytes > o.bytes - s.offset { return Err("live span out of bounds".into()); }
    Ok(json!({"allocation":s.owner,"offset":s.offset}))
}
fn exact_span(s: &Span, bytes: usize, owners: &BTreeMap<u64, OwnerMeta>) -> Result<Value, String> {
    if bytes > s.bytes { return Err("command exceeds declared live span".into()); }
    target(s, owners)
}
fn hex(s: &str) -> Result<Vec<u8>, String> { crate::decode_hex(s) }
fn argument(a: &Argument, owners: &BTreeMap<u64, OwnerMeta>) -> Result<Value, String> {
    Ok(match a {
        Argument::Device { target: s } => json!({"kind":"device","target":target(s, owners)?}),
        Argument::Bytes { bytes_hex } => { hex(bytes_hex)?; json!({"kind":"bytes","bytes_hex":bytes_hex}) },
        Argument::Patched { bytes_hex, pointers } => {
            let mut bytes = hex(bytes_hex)?;
            let mut used = HashSet::new();
            let mut patches = Vec::new();
            for p in pointers {
                let end = p.byte_offset.checked_add(8).ok_or("pointer patch overflow")?;
                if end > bytes.len() || (p.byte_offset..end).any(|b| !used.insert(b)) {
                    return Err("overlapping or out-of-bounds pointer patch".into());
                }
                bytes[p.byte_offset..end].fill(0); // raw addresses are not identity
                patches.push(json!({"byte_offset":p.byte_offset,"target":target(&p.target, owners)?}));
            }
            json!({"kind":"patched","bytes_hex":bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),"pointers":patches})
        },
    })
}
fn operation(c: &Command, owners: &BTreeMap<u64, OwnerMeta>) -> Result<Value, String> {
    let (stream, op) = match c {
        Command::Kernel { stream, module, function, grid, block, shared_bytes, arguments } => {
            if module.is_empty() || function.is_empty() || grid.contains(&0) || block.contains(&0) {
                return Err("incomplete live launch identity or dimensions".into());
            }
            (*stream, json!({"kind":"kernel","stream":stream,"module":module,"function":function,
                "grid":grid,"block":block,"shared_bytes":shared_bytes,
                "arguments":arguments.iter().map(|a| argument(a, owners)).collect::<Result<Vec<_>, _>>()?}))
        },
        Command::DeviceCopy { stream, destination, source, bytes } => (*stream,
            json!({"kind":"device_copy","stream":stream,"destination":exact_span(destination,*bytes,owners)?,
                "source":exact_span(source,*bytes,owners)?,"bytes":bytes})),
        Command::Clear { stream, destination, bytes, value } => (*stream,
            json!({"kind":"clear","stream":stream,"destination":exact_span(destination,*bytes,owners)?,"bytes":bytes,"value":value})),
        Command::Fill16 { stream, destination, count, value } => (*stream,
            json!({"kind":"fill16","stream":stream,"destination":exact_span(destination,count.checked_mul(2).ok_or("fill overflow")?,owners)?,"count":count,"value":value})),
        Command::Fill32 { stream, destination, count, value } => (*stream,
            json!({"kind":"fill32","stream":stream,"destination":exact_span(destination,count.checked_mul(4).ok_or("fill overflow")?,owners)?,"count":count,"value":value})),
    };
    // Execution::from_live registers the caller's actual stream as logical 0.
    if stream != 0 { return Err("live window requires joined caller stream 0".into()); }
    Ok(op)
}
impl LiveProgram {
    pub(crate) fn new(input: Input, registered: &BTreeMap<u64, Owner>) -> Result<Self, String> {
        if input.schema != "hexad.live-window/v1" { return Err("unknown live window schema".into()); }
        if input.steps.is_empty() || input.boundaries.len() != input.steps.len() + 1 {
            return Err("live window needs boundary 0 and one boundary per step".into());
        }
        let mut owners = BTreeMap::new();
        let mut fixed = HashSet::new();
        for m in &input.owners {
            let o = registered.get(&m.owner).ok_or("unregistered live owner")?;
            if o.refs == 0 || o.generation != m.generation || o.bytes != m.bytes as u64 || o.constant != m.constant as u64 {
                return Err("live owner generation, extent or constant flag differs".into());
            }
            if crate::live_owner::valid(o) != 0 || m.bytes > (i64::MAX as usize) - 256 {
                return Err("invalid live owner or shadow size".into());
            }
            if m.pointer_offsets.iter().any(|p| p.checked_add(8).map_or(true, |end| end > m.bytes)) {
                return Err("live pointer mask out of bounds".into());
            }
            if owners.insert(m.owner, m.clone()).is_some() { return Err("duplicate live owner id; pin one generation per window".into()); }
            if m.constant { fixed.insert(m.owner); }
        }
        if owners.is_empty() || registered.values().filter(|o| o.refs != 0).count() != owners.len() {
            return Err("live owner inventory is not the registered owner set".into());
        }
        let mut total = 0usize;
        for m in owners.values() {
            total = total.checked_add(m.bytes.div_ceil(4).div_ceil(64).checked_mul(256).ok_or("shadow layout overflow")?)
                .ok_or("shadow layout overflow")?;
        }
        if total > i64::MAX as usize { return Err("shadow layout exceeds signed indexing".into()); }
        let mut blocks = serde_json::Map::new();
        let mut requests = Vec::new();
        let mut prior = None;
        for step in &input.steps {
            if prior.is_some_and(|p: usize| p.checked_add(1) != Some(step.step)) { return Err("live steps must be consecutive".into()); }
            prior = Some(step.step);
            let mut keys = HashSet::new();
            let mut ops = Vec::new();
            let mut reqs = Vec::new();
            for row in &step.commands {
                if row.key.is_empty() || !keys.insert(&row.key) { return Err("missing or repeated live command key".into()); }
                let op = operation(&row.command, &owners)?;
                if let Command::Kernel { function, module, .. } = &row.command {
                    if function == "cupy_concatenate_same_size" && !input.concat_types.contains_key(module) {
                        return Err("live concatenate needs module dtype".into());
                    }
                }
                let mut writes = Vec::new();
                for s in &row.writes {
                    target(s, &owners)?;
                    if owners[&s.owner].constant { return Err("live command writes a constant owner".into()); }
                    if !writes.contains(&s.owner) { writes.push(s.owner); }
                }
                let dst = match &row.command {
                    Command::DeviceCopy { destination, bytes, .. } | Command::Clear { destination, bytes, .. } => Some((destination, *bytes)),
                    Command::Fill16 { destination, count, .. } => Some((destination, count * 2)),
                    Command::Fill32 { destination, count, .. } => Some((destination, count * 4)),
                    _ => None,
                };
                if let Some((d, n)) = dst {
                    if !row.writes.iter().any(|s| s.owner == d.owner && s.generation == d.generation && s.offset <= d.offset
                        && d.offset - s.offset <= s.bytes && n <= s.bytes - (d.offset - s.offset)) {
                        return Err("live destination not covered by writes".into());
                    }
                }
                reqs.push(json!({"operation":op.clone(),"writes":writes}));
                ops.push(op);
            }
            blocks.insert(format!("step_{:08}",step.step), Value::Array(ops));
            requests.push(reqs);
        }
        // These Allocate rows are layout declarations, NEVER forward commands.
        blocks.insert("__live_layout".into(), json!(owners.values().filter(|m| !m.initial).map(|m|
            json!({"kind":"allocate","stream":0,"id":m.owner,"bytes":m.bytes,"asynchronous":false,"overwritten_before_read":false})).collect::<Vec<_>>()));
        let boundaries = input.boundaries.iter().map(|spans| {
            spans.iter().enumerate().map(|(i,s)| Ok(json!({"id":i,"target":target(s,&owners)?,"bytes":s.bytes})))
                .collect::<Result<Vec<_>, String>>().map(|live| json!({"live":live,"host_live":[]}))
        }).collect::<Result<Vec<_>, _>>()?;
        let plan: Plan = serde_json::from_value(json!({"schema":"hexad.live-metadata/v1","complete":false,
            "source_pin":"live","end_step_barrier_proven":false,"modules":[],
            "address_bearing":owners.values().filter(|m| m.address_bearing).map(|m| m.owner).collect::<Vec<_>>(),
            "allocations":owners.values().filter(|m| m.initial).map(|m| json!({"id":m.owner,"bytes":m.bytes,"initial":null,
                "relocations":m.pointer_offsets.iter().map(|p| json!({"byte_offset":p,"target":{"allocation":m.owner,"offset":0},"recorded_value":0})).collect::<Vec<_>>() })).collect::<Vec<_>>(),
            "carried":[],"blocks":blocks,"boundaries":boundaries})).map_err(|e| e.to_string())?;
        Ok(Self { plan, fixed, concat_types: input.concat_types.clone(), input, requests })
    }
    /// Validate again at use, including every indirect owner and constant.
    pub(crate) fn request(&self, step_index: usize, command_index: usize,
        registered: &BTreeMap<u64, Owner>) -> Result<Vec<u8>, String> {
        for m in &self.input.owners {
            let o = registered.get(&m.owner).ok_or("live owner disappeared")?;
            if o.refs == 0 || o.generation != m.generation || o.bytes != m.bytes as u64 || o.constant != m.constant as u64 {
                return Err("live window owner changed since construction".into());
            }
        }
        let r = self.requests.get(step_index).and_then(|s| s.get(command_index)).ok_or("live command index out of bounds")?;
        serde_json::to_vec(r).map_err(|e| e.to_string())
    }
}

/// Owns the fixed generation set for a live window. Boxes keep the Win and
/// Execution stable for LiveSave. The schedule driver must drop its capture
/// before dropping this object, and must never free caller owners while pinned.
/// No legacy recorded Session or replay entry point is exposed here.
pub(crate) struct LiveWindow {
    win: Box<Win>,
    session: Box<super::capi::live::LiveSession>,
    pub program: LiveProgram,
    attached: bool,
}
impl LiveWindow {
    pub(crate) unsafe fn from_live(session: super::capi::live::LiveSession, input: Input,
        options: &HashMap<String, String>) -> Result<Self, String> {
        if opt(options, "live-window") != Some("1") { return Err("live-window=1 required".into()); }
        session.current()?;
        let program = LiveProgram::new(input, &session.owners)?;
        let win = Box::new(Win::from_live(&session.ex, &program, options)?);
        let mut out = Self { win, session: Box::new(session), program, attached: false };
        let cu = &out.session.ex.cuda;
        // Preserve any supplied seed. Rules need one contiguous float arena;
        // publish that same arena's views to the owner table and derivative rows.
        for m in &out.program.input.owners {
            let o = &out.session.owners[&m.owner];
            if !m.constant {
                if o.shadow == 0 { return Err("nonconstant live owner has no shadow".into()); }
                dev::copy(cu, out.win.shadows[&m.owner].address, o.shadow, m.bytes)?;
            }
        }
        dev::sync(cu)?;
        let old: Vec<u64> = out.session.owners.values().filter_map(|o| (o.shadow != 0).then_some(o.shadow)).collect();
        for o in out.session.owners.values_mut() {
            o.shadow = if o.constant == 0 { out.win.shadows[&o.id].address } else { 0 };
        }
        out.attached = true;
        for p in old { checked("cuMemFree replaced live shadow", (cu.free)(p))?; }
        out.session.ex.address_bearing = out.program.input.owners.iter().filter(|m| m.address_bearing).map(|m| m.owner).collect();
        // Mask pointers for ALL lifetimes, including temporaries absent at step 0.
        out.win.relocs = out.program.input.owners.iter().filter(|m| !m.pointer_offsets.is_empty())
            .map(|m| (m.owner, m.pointer_offsets.iter().map(|p| (*p,8)).collect())).collect();
        out.win.address_bearing = out.session.ex.address_bearing.clone();
        Ok(out)
    }
    pub(crate) fn request(&self, step_index: usize, command_index: usize) -> Result<Vec<u8>, String> {
        self.program.request(step_index, command_index, &self.session.owners)
    }
    /// Driver seam: caller must serialize use and destroy LiveSave before self.
    /// No generation can be retired through this wrapper while a tape is alive.
    pub(crate) fn parts(&mut self) -> (&mut Win, &mut Execution) {
        (&mut self.win, &mut self.session.ex)
    }
    pub(crate) fn shadow(&self, owner: u64, generation: u64) -> Result<u64, String> {
        Ok(self.session.owner(owner, generation)?.shadow)
    }
}
impl Drop for LiveWindow {
    fn drop(&mut self) { unsafe {
        let cu = &self.session.ex.cuda;
        // Once attached, the owner entries are views, never separate allocations.
        if self.attached { for o in self.session.owners.values_mut() { o.shadow = 0; } }
        let mut previous = ptr::null_mut();
        if (cu.ctx_get_current)(&mut previous) != 0 || (cu.ctx_set_current)(self.session.ex.context) != 0 { return; }
        if (cu.context_synchronize)() == 0 {
            let w = &self.win;
            for p in [w.sh_base, w.tape.base, w.scratch.base, w.tmp[0], w.tmp[1], w.tmp[2], w.part, w.fail] {
                if p != 0 { let _ = (cu.free)(p); }
            }
            if let Some((_, ar)) = &w.tempo { let _ = (cu.free)(ar.base); }
            for (p, _) in &w.dev_anchors { let _ = (cu.free)(*p); }
            for k in w.kernels.values() {
                for (p, _) in k.derived.borrow().values() { let _ = (cu.free)(*p); }
                if !k.module.is_null() { let _ = (cu.module_unload)(k.module); }
            }
        }
        let _ = (cu.ctx_set_current)(previous);
    } }
}
