// SPDX-License-Identifier: Apache-2.0
// Native access to the original CUB host functions exported by a CuPy build.
// The enclosing Rust code verifies the library's SHA-256 and bounds every
// device argument. No Python entry point may be called by this bridge.
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>

// The Python extension has GLOB_DAT references even though the exported CUB
// routines do not use the interpreter. Provide inert definitions for loading
// the library without mapping libpython; executing Python code is forbidden.
// Weak (lane runner-lib): the libhexad-phys cdylib supplies exported strong copies (capi_exports.rs), since a
// cdylib does not export this archive's symbols; the ad-binder executable keeps these.
extern "C" {
#define PYEXC(name) __attribute__((weak)) void* PyExc_##name = nullptr
PYEXC(ArithmeticError); PYEXC(AssertionError); PYEXC(AttributeError);
PYEXC(DeprecationWarning); PYEXC(IOError); PYEXC(ImportError);
PYEXC(IndexError); PYEXC(MemoryError); PYEXC(NameError);
PYEXC(NotImplementedError); PYEXC(OverflowError); PYEXC(RuntimeError);
PYEXC(StopIteration); PYEXC(SystemError); PYEXC(TypeError);
PYEXC(UnboundLocalError); PYEXC(ValueError); PYEXC(ZeroDivisionError);
#undef PYEXC
#define PYDATA(name) __attribute__((weak)) alignas(64) unsigned char name[4096] = {};
PYDATA(PyBaseObject_Type) PYDATA(PyCFunction_Type) PYDATA(PyDict_Type)
PYDATA(PyFloat_Type) PYDATA(PyList_Type) PYDATA(PyLong_Type)
PYDATA(PyMethod_Type) PYDATA(PyRange_Type) PYDATA(PyTuple_Type)
PYDATA(PyUnicode_Type) PYDATA(_Py_FalseStruct) PYDATA(_Py_NoneStruct)
PYDATA(_Py_TrueStruct)
#undef PYDATA
__attribute__((weak)) unsigned int Py_Version = 0;
__attribute__((weak)) void* PyObject_GenericGetAttr(void*, void*) { std::abort(); }
}

struct CUstream_st;
using Reduce = void (*)(void*, size_t&, void*, void*, int, CUstream_st*, int, int);
using Segmented = void (*)(void*, size_t&, void*, void*, int, int, CUstream_st*, int, int);
struct CubLibrary { void* handle; Reduce reduce; Segmented segmented; };

extern "C" int binder_cub_open(const char* path, void** out) noexcept {
    if (!path || !out) return -1;
    *out = nullptr;
    void* handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) return -2;
    auto reduce = reinterpret_cast<Reduce>(dlsym(handle, "_Z17cub_device_reducePvRmS_S_iP11CUstream_stii"));
    auto segmented = reinterpret_cast<Segmented>(dlsym(handle, "_Z27cub_device_segmented_reducePvRmS_S_iiP11CUstream_stii"));
    if (!reduce || !segmented) { dlclose(handle); return -3; }
    try { *out = new CubLibrary{handle, reduce, segmented}; }
    catch (...) { dlclose(handle); return -4; }
    return 0;
}

extern "C" int binder_cub_reduce(void* raw, void* workspace, size_t capacity,
                                  void* input, void* output, int count,
                                  CUstream_st* stream, int op, int dtype,
                                  size_t* required) noexcept {
    if (!raw || !workspace || !input || !output || !required || count <= 0) return -1;
    try {
        auto* library = static_cast<CubLibrary*>(raw);
        size_t bytes = 0;
        library->reduce(nullptr, bytes, input, output, count, stream, op, dtype);
        *required = bytes;
        if (bytes > capacity) return -2;
        library->reduce(workspace, bytes, input, output, count, stream, op, dtype);
        return 0;
    } catch (...) { return -3; }
}

extern "C" int binder_cub_segmented(void* raw, void* workspace, size_t capacity,
                                     void* input, void* output, int segments, int segment_size,
                                     CUstream_st* stream, int op, int dtype,
                                     size_t* required) noexcept {
    if (!raw || !workspace || !input || !output || !required || segments <= 0 || segment_size <= 0) return -1;
    try {
        auto* library = static_cast<CubLibrary*>(raw);
        size_t bytes = 0;
        library->segmented(nullptr, bytes, input, output, segments, segment_size, stream, op, dtype);
        *required = bytes;
        if (bytes > capacity) return -2;
        library->segmented(workspace, bytes, input, output, segments, segment_size, stream, op, dtype);
        return 0;
    } catch (...) { return -3; }
}

extern "C" void binder_cub_close(void* raw) noexcept {
    if (auto* library = static_cast<CubLibrary*>(raw)) {
        dlclose(library->handle);
        delete library;
    }
}
