/* SPDX-License-Identifier: Apache-2.0 */
#ifndef AD_BINDER_BOUNDARY_HOOK_H
#define AD_BINDER_BOUNDARY_HOOK_H
#include <stddef.h>
#include <stdint.h>

/* Match runtime/src/binder/checkpoint.rs. memory: 1 CUDA device, 2 host. */
typedef struct ad_binder_binding_v1 {
    uint64_t id;
    uint64_t bytes;
    uint32_t memory;
    uint64_t address;
} ad_binder_binding_v1;

/* Export this symbol from the optional --checkpoint-hook shared object.
 * The binder calls it after fencing the selected --checkpoint-bindings-at
 * boundaries. The current CUDA context is passed as context; stream is the
 * replay's default stream. IDs and extents join to the independently sealed
 * census. The array and its process-local addresses are valid only until the
 * callback returns. The hook may capture or restore the same boundary, must
 * return nonzero on failure, and must not retain those addresses. The binder
 * synchronizes the CUDA context and reloads the frame owner after return.
 */
int32_t ad_binder_boundary_v1(uint64_t step, void *context, void *stream,
                              const ad_binder_binding_v1 *bindings, size_t count);
#endif
