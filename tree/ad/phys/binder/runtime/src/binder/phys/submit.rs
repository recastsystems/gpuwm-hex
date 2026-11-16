// SPDX-License-Identifier: Apache-2.0
//! A single-stream batch adapter for hex's pinned C submitter.
//!
//! This module does not capture, seal, allocate, load modules, or schedule a
//! forecast. The caller must establish those facts independently and must
//! preserve the lifetimes of all driver functions, allocations and streams.
//! Uploads, events and host boundaries must be ordered outside a batch.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

#[repr(C)]
pub struct NativeArgument {
    value: *const c_void,
    offset: usize,
    width: usize,
    from_frame: u32,
}

#[repr(C)]
pub struct NativeCommand {
    kind: u32,
    grid: [u32; 3],
    block: [u32; 3],
    shared_bytes: u32,
    argc: u32,
    function: *mut c_void,
    args: *const NativeArgument,
    destination: u64,
    source: u64,
    bytes: usize,
    fill: u8,
}

pub type Launch = unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32,
                                        u32, *mut c_void, *mut *mut c_void,
                                        *mut *mut c_void) -> i32;
pub type Copy = unsafe extern "C" fn(u64, u64, usize, *mut c_void) -> i32;
pub type Clear = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;

#[repr(C)]
pub struct NativeDriver { pub launch: Launch, pub copy: Copy, pub clear: Clear }

pub type Validate = unsafe extern "C" fn(*const NativeCommand, usize, usize) -> i32;
pub type Submit = unsafe extern "C" fn(*const NativeDriver, *const NativeCommand, usize,
                                      *const c_void, *mut c_void, *mut usize) -> i32;

#[derive(Clone, Copy, Debug)]
pub struct Allocation { pub id: u64, pub address: u64, pub bytes: usize }
#[derive(Clone, Copy, Debug)]
pub struct Slice { pub allocation: u64, pub offset: usize }
#[derive(Clone, Debug)]
pub struct PointerPatch { pub byte_offset: usize, pub target: Slice }

#[derive(Clone, Debug)]
pub enum Argument {
    Bytes(Vec<u8>),
    Device(Slice),
    /// CUDA by-value argument with explicitly identified embedded pointers.
    /// Unknown pointer positions must be refused before this function is used.
    Patched { bytes: Vec<u8>, pointers: Vec<PointerPatch> },
    Frame { offset: usize, bytes: usize, alignment: usize },
}

#[derive(Clone, Debug)]
pub enum Command {
    Kernel { function: *mut c_void, grid: [u32; 3], block: [u32; 3],
             shared_bytes: u32, arguments: Vec<Argument> },
    DeviceCopy { destination: Slice, source: Slice, bytes: usize },
    Clear { destination: Slice, value: u8, bytes: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Poisoned,
    EmptyBatch,
    InvalidAllocation(u64),
    DuplicateAllocation(u64),
    MissingAllocation(u64),
    OutOfRange(u64),
    InvalidArgument,
    NativeValidation(i32),
    PartialSubmission { status: i32, completed: usize },
}

#[repr(align(16))]
#[derive(Clone)]
struct AlignedWord([u8; 16]);

struct FixedValue { words: Vec<AlignedWord>, len: usize }
impl FixedValue {
    fn new(bytes: &[u8]) -> Self {
        let mut words = vec![AlignedWord([0; 16]); bytes.len().div_ceil(16)];
        for (index, chunk) in bytes.chunks(16).enumerate() {
            words[index].0[..chunk.len()].copy_from_slice(chunk);
        }
        Self { words, len: bytes.len() }
    }
}

fn address(allocations: &HashMap<u64, Allocation>, slice: Slice, bytes: usize)
    -> Result<u64, Error>
{
    let a = allocations.get(&slice.allocation).ok_or(Error::MissingAllocation(slice.allocation))?;
    if slice.offset >= a.bytes || bytes > a.bytes - slice.offset {
        return Err(Error::OutOfRange(slice.allocation));
    }
    a.address.checked_add(slice.offset as u64).ok_or(Error::OutOfRange(slice.allocation))
}

#[derive(Default)]
pub struct Submitter { poisoned: bool }

impl Submitter {
    pub fn is_poisoned(&self) -> bool { self.poisoned }

    /// The API functions and all device handles/addresses must be valid for the
    /// current CUDA context. Commands must have been independently sealed and
    /// ordered; a successful return means issued, not device-synchronized.
    pub unsafe fn submit(&mut self, driver: &NativeDriver, validate: Validate,
                         submit: Submit, allocations: &[Allocation], commands: &[Command],
                         frame: &[u8], stream: *mut c_void) -> Result<(), Error> {
        if self.poisoned { return Err(Error::Poisoned); }
        if commands.is_empty() { return Err(Error::EmptyBatch); }
        if !cfg!(target_endian = "little") || std::mem::size_of::<usize>() != 8 {
            return Err(Error::InvalidArgument);
        }
        let mut map = HashMap::new();
        for &a in allocations {
            if a.address == 0 || a.bytes == 0 || a.address.checked_add(a.bytes as u64).is_none() {
                return Err(Error::InvalidAllocation(a.id));
            }
            if map.insert(a.id, a).is_some() { return Err(Error::DuplicateAllocation(a.id)); }
        }
        // Each inner buffer has a stable heap address even when the outer Vec moves.
        // Fixed argument storage is 16-byte aligned; callers must refuse any
        // kernel ABI demanding stronger alignment before submitting a batch.
        let mut values: Vec<FixedValue> = Vec::new();
        let mut argument_arrays: Vec<Vec<NativeArgument>> = Vec::new();
        let mut native = Vec::with_capacity(commands.len());
        for command in commands {
            let mut entry = NativeCommand { kind: 0, grid: [0; 3], block: [0; 3],
                shared_bytes: 0, argc: 0, function: ptr::null_mut(), args: ptr::null(),
                destination: 0, source: 0, bytes: 0, fill: 0 };
            match command {
                Command::Kernel { function, grid, block, shared_bytes, arguments } => {
                    if function.is_null() || grid.contains(&0) || block.contains(&0) || arguments.len() > 256 {
                        return Err(Error::InvalidArgument);
                    }
                    let mut native_args = Vec::with_capacity(arguments.len());
                    for argument in arguments {
                        let mut fixed = match argument {
                            Argument::Bytes(bytes) => Some(bytes.clone()),
                            Argument::Device(slice) => Some(address(&map, *slice, 0)?.to_le_bytes().to_vec()),
                            Argument::Patched { bytes, pointers } => {
                                let mut data = bytes.clone();
                                let mut ranges = Vec::new();
                                for patch in pointers {
                                    let end = patch.byte_offset.checked_add(8).ok_or(Error::InvalidArgument)?;
                                    if end > data.len() || ranges.iter().any(|&(a,b)| patch.byte_offset < b && a < end) {
                                        return Err(Error::InvalidArgument);
                                    }
                                    data[patch.byte_offset..end].copy_from_slice(&address(&map, patch.target, 0)?.to_le_bytes());
                                    ranges.push((patch.byte_offset, end));
                                }
                                Some(data)
                            },
                            Argument::Frame { offset, bytes, alignment } => {
                                if *bytes == 0 || !alignment.is_power_of_two()
                                    || *offset > frame.len() || *bytes > frame.len() - *offset
                                    || (frame.as_ptr() as usize + *offset) % *alignment != 0 {
                                    return Err(Error::InvalidArgument);
                                }
                                native_args.push(NativeArgument { value: ptr::null(), offset: *offset,
                                    width: *bytes, from_frame: 1 });
                                None
                            },
                        };
                        if let Some(bytes) = fixed.take() {
                            if bytes.is_empty() { return Err(Error::InvalidArgument); }
                            values.push(FixedValue::new(&bytes));
                            let bytes = values.last().unwrap();
                            native_args.push(NativeArgument { value: bytes.words.as_ptr().cast(), offset: 0,
                                width: bytes.len, from_frame: 0 });
                        }
                    }
                    entry.kind = 1; entry.grid = *grid; entry.block = *block;
                    entry.shared_bytes = *shared_bytes; entry.argc = native_args.len() as u32;
                    entry.function = *function;
                    argument_arrays.push(native_args);
                    entry.args = argument_arrays.last().unwrap().as_ptr();
                },
                Command::DeviceCopy { destination, source, bytes } => {
                    if *bytes == 0 { return Err(Error::InvalidArgument); }
                    entry.kind = 2; entry.destination = address(&map, *destination, *bytes)?;
                    entry.source = address(&map, *source, *bytes)?; entry.bytes = *bytes;
                },
                Command::Clear { destination, value, bytes } => {
                    if *bytes == 0 { return Err(Error::InvalidArgument); }
                    entry.kind = 3; entry.destination = address(&map, *destination, *bytes)?;
                    entry.fill = *value; entry.bytes = *bytes;
                },
            }
            native.push(entry);
        }
        let validation = validate(native.as_ptr(), native.len(), frame.len());
        if validation != 0 { return Err(Error::NativeValidation(validation)); }
        let mut completed = 0;
        let status = submit(driver, native.as_ptr(), native.len(), frame.as_ptr().cast(), stream, &mut completed);
        if status != 0 || completed != native.len() {
            self.poisoned = true;
            return Err(Error::PartialSubmission { status, completed });
        }
        Ok(())
    }
}
