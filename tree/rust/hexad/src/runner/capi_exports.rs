// SPDX-License-Identifier: Apache-2.0
//! Lane runner-lib: the symbols the SHA-pinned CuPy CUB / Thrust host libraries resolve from their loader, exported
//! from the libhexad-phys cdylib (feature capi-exports). In the ad-binder executable they come from
//! native_cub_bridge.cc / native_thrust_bridge.cc through -Wl,--export-dynamic; a cdylib's version script exports only
//! Rust #[no_mangle] items, so without these the pinned libraries' dlopen fails ("cannot load original CUB host
//! routine: -2", a test machine held smoke 2026-11-07T08:33Z) and their cupy_malloc / cupy_free calls would not reach the
//! replay-owned arena. The bridge's own definitions are weak; these strong ones replace them in the cdylib only.
//! cupy_malloc / cupy_free are the bridge's arena allocator, line for line (the Arena layout is the bridge's struct).
#![allow(non_upper_case_globals, non_snake_case)]
use std::ffi::c_void;

#[repr(C)]
pub struct Arena { base: usize, capacity: usize, used: usize, count: usize, addresses: [usize; 1024], active: [bool; 1024], invalid: bool }

/// Bounded native arena allocation (the pinned Thrust extension's allocator hook; its Python allocator never runs).
#[no_mangle]
pub unsafe extern "C" fn cupy_malloc(memory: *mut c_void, bytes: usize) -> *mut u8 {
    let arena = memory as *mut Arena;
    if arena.is_null() { return std::ptr::null_mut(); }
    let a = &mut *arena;
    if bytes == 0 || a.count == 1024 || a.used > usize::MAX - 255 { a.invalid = true; return std::ptr::null_mut(); }
    let offset = (a.used + 255) & !255usize;
    if offset > a.capacity || bytes > a.capacity - offset || a.base > usize::MAX - offset { a.invalid = true; return std::ptr::null_mut(); }
    let pointer = a.base + offset;
    a.addresses[a.count] = pointer;
    a.active[a.count] = true;
    a.count += 1;
    a.used = offset + bytes;
    pointer as *mut u8
}
#[no_mangle]
pub unsafe extern "C" fn cupy_free(memory: *mut c_void, pointer: *mut u8) -> i32 {
    let arena = memory as *mut Arena;
    if arena.is_null() || pointer.is_null() { return -1; }
    let a = &mut *arena;
    for i in 0..a.count {
        if a.addresses[i] == pointer as usize && a.active[i] { a.active[i] = false; return 0; }
    }
    a.invalid = true;
    -1
}

// Inert Python data the extensions reference (GLOB_DAT) without ever running Python.
macro_rules! pyexc { ($($n:ident),*) => { $( #[no_mangle] pub static mut $n: *mut c_void = std::ptr::null_mut(); )* } }
pyexc!(PyExc_ArithmeticError, PyExc_AssertionError, PyExc_AttributeError, PyExc_DeprecationWarning, PyExc_IOError, PyExc_ImportError,
       PyExc_IndexError, PyExc_MemoryError, PyExc_NameError, PyExc_NotImplementedError, PyExc_OverflowError, PyExc_RuntimeError,
       PyExc_StopIteration, PyExc_SystemError, PyExc_TypeError, PyExc_UnboundLocalError, PyExc_ValueError, PyExc_ZeroDivisionError);
#[repr(C, align(64))]
pub struct PyData([u8; 4096]);
macro_rules! pydata { ($($n:ident),*) => { $( #[no_mangle] pub static mut $n: PyData = PyData([0; 4096]); )* } }
pydata!(PyBaseObject_Type, PyCFunction_Type, PyDict_Type, PyFloat_Type, PyList_Type, PyLong_Type, PyMethod_Type, PyRange_Type,
        PyTuple_Type, PyUnicode_Type, _Py_FalseStruct, _Py_NoneStruct, _Py_TrueStruct);
#[no_mangle] pub static mut Py_Version: u32 = 0;
#[no_mangle] pub extern "C" fn PyObject_GenericGetAttr(_o: *mut c_void, _n: *mut c_void) -> *mut c_void { std::process::abort() }
