// SPDX-License-Identifier: Apache-2.0
// Call the original one-dimensional CuPy Thrust host sort with a replay-owned
// arena. The Rust caller verifies the image SHA and bounds every live buffer.
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <vector>

struct Arena {
    uintptr_t base;
    size_t capacity, used, count;
    uintptr_t addresses[1024];
    bool active[1024];
    bool invalid;
};

// The pinned extension calls these through its PLT. Never let its Python
// allocator run: the passed memory pointer is our bounded native arena. Weak (lane runner-lib): the
// libhexad-phys cdylib exports a line-for-line Rust copy (capi_exports.rs); the ad-binder executable keeps these.
extern "C" __attribute__((weak)) char* cupy_malloc(void* memory, size_t bytes) noexcept {
    auto* arena = static_cast<Arena*>(memory);
    if (!arena || !bytes || arena->count == 1024 || arena->used > SIZE_MAX - 255) {
        if (arena) arena->invalid = true;
        return nullptr;
    }
    size_t offset = (arena->used + 255) & ~size_t(255);
    if (offset > arena->capacity || bytes > arena->capacity - offset ||
        arena->base > UINTPTR_MAX - offset) {
        arena->invalid = true;
        return nullptr;
    }
    uintptr_t pointer = arena->base + offset;
    arena->addresses[arena->count] = pointer;
    arena->active[arena->count++] = true;
    arena->used = offset + bytes;
    return reinterpret_cast<char*>(pointer);
}

extern "C" __attribute__((weak)) int cupy_free(void* memory, char* pointer) noexcept {
    auto* arena = static_cast<Arena*>(memory);
    if (!arena || !pointer) return -1;
    for (size_t i = 0; i < arena->count; ++i) {
        if (arena->addresses[i] == reinterpret_cast<uintptr_t>(pointer) && arena->active[i]) {
            arena->active[i] = false;
            return 0;
        }
    }
    arena->invalid = true;
    return -1;
}

using Argsort = void (*)(int, size_t*, void*, void*, const std::vector<ptrdiff_t>&, intptr_t, void*);
struct ThrustLibrary { void* handle; Argsort argsort; };

extern "C" int binder_thrust_open(const char* path, void** out) noexcept {
    if (!path || !out) return -1;
    *out = nullptr;
    void* handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) return -2;
    auto sort = reinterpret_cast<Argsort>(dlsym(handle, "_Z14thrust_argsortiPmPvS0_RKSt6vectorIlSaIlEElS0_"));
    if (!sort) { dlclose(handle); return -3; }
    try { *out = new ThrustLibrary{handle, sort}; }
    catch (...) { dlclose(handle); return -4; }
    return 0;
}

extern "C" int binder_thrust_argsort(void* raw, int dtype, void* indices, void* input,
                                      size_t count, intptr_t stream, void* workspace,
                                      size_t capacity, size_t* used, size_t* allocations) noexcept {
    if (!raw || !indices || !input || !workspace || !used || !allocations || !count ||
        count > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
        (reinterpret_cast<uintptr_t>(workspace) & 255)) return -1;
    Arena arena{reinterpret_cast<uintptr_t>(workspace), capacity, 0, 0, {}, {}, false};
    try {
        std::vector<ptrdiff_t> shape{static_cast<ptrdiff_t>(count)};
        static_cast<ThrustLibrary*>(raw)->argsort(dtype, static_cast<size_t*>(indices),
                                                  input, nullptr, shape, stream, &arena);
    } catch (...) {
        *used = arena.used;
        *allocations = arena.count;
        return arena.invalid ? -2 : -3;
    }
    *used = arena.used;
    *allocations = arena.count;
    if (arena.invalid) return -2;
    for (size_t i = 0; i < arena.count; ++i) if (arena.active[i]) return -4;
    return 0;
}

extern "C" void binder_thrust_close(void* raw) noexcept {
    if (auto* library = static_cast<ThrustLibrary*>(raw)) {
        dlclose(library->handle);
        delete library;
    }
}
