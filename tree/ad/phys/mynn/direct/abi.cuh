#pragma once
#include <cstdint>
// Binary-compatible with gen_whole.py MwLaunch. Unused legacy fields must be zero.
struct MynnAdNode;
struct MwBind { const float* x; float* y; const double* tin; double* tout; const double* ybar; double* xbar; };
struct MwLaunch {
 MwBind b[91];
 const int* kpbl_in; int* kpbl_out; int* error_flags;
 unsigned char* arena; unsigned long long arena_col_bytes;
 unsigned char* store; unsigned long long store_col_elems;
 int* leaf; int* kstore; MynnAdNode* nodes; double* bars; long long capacity; long long* used;
 int* failures; int col_first; int col_count; int mode; int layout;
};
static_assert(sizeof(MwBind)==48 && sizeof(MwLaunch)==4496, "MYNN direct ABI requires 64-bit pointers");
