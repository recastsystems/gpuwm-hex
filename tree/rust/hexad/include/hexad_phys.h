/* SPDX-License-Identifier: Apache-2.0
 * libhexad-phys C ABI v2 (the library face of `ad-binder window`).
 *
 * The hex physics-window tangent-linear and adjoint, one plan step per call, for a host program's linear model.
 * The host owns its state and increment; the runner owns the trajectory (sealed plan package P_k and its boundaries), the
 * tangent / cotangent shadows (one float32 arena over every plan allocation), the power-of-two scale table and the
 * operator manifest. The runner sees only this ABI, device pointers and the allocation table.
 *
 * Call order (the host's tangent and adjoint steps):
 *   initializeTL(dx): hexad_tl_init(s, INCREMENT dx)         T0(dx) primes the arena at boundary 0
 *   stepTL(dx):       hexad_tl_step(s); hexad_tl_get(s, dx)   one tangent plan step, then dx = Ts(arena)
 *   initializeAD(dx): hexad_ad_init(s, INCREMENT dx); zero dx  arena = T_N^T(dx)
 *   stepAD(dx):       hexad_ad_put(s, dx); zero dx; hexad_ad_step(s)   add T_s^T(dx), reverse s -> s-1
 *   finalizeAD(dx):   hexad_ad_get(s, INCREMENT dx)           dx += T0^T(arena at boundary 0)
 * The first tangent sweep of a session binds the trajectory: every boundary hash is checked against the plan's
 * reference.jsonl (a mismatch refuses) and the boundaries are stored (pinned host RAM within the budget, else local
 * NVMe with prefetch). One session = one plan package = one trajectory; a new outer loop opens a new session on its
 * recorded plan package and passes "expect_manifest_sha256" (a manifest change between outer loops is refused).
 *
 * Every pointer in a hexad_field / hexad_vec is a DEVICE pointer valid in the session's context: open with
 * {"context":"primary"} to share the CUDA runtime's primary context with the client (a host's resident fields).
 * Return codes: 0 ok, -1 error, -2 refused (a DEFECT line; hexad_last_error() has the text). No CPU fallback exists.
 */
#ifndef HEXAD_PHYS_H
#define HEXAD_PHYS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define HEXAD_ABI_VERSION 2

typedef struct hexad_session hexad_session;

/* One allocation of the plan: its device primal and its float32 shadow (tangent or cotangent) in the arena. */
typedef struct hexad_alloc {
  uint64_t id, bytes;
  uint64_t primal;   /* device address of the live allocation (0 when not live yet) */
  uint64_t shadow;   /* device address of its float32 shadow words */
  uint32_t flags;    /* HEXAD_ALLOC_* */
  int32_t scale_exp; /* tangent = 2^scale_exp * physical (uniform; see the scale table) */
} hexad_alloc;
#define HEXAD_ALLOC_INITIAL 1u
#define HEXAD_ALLOC_FIXED 2u           /* unchanged at every boundary (geometry, tables) */
#define HEXAD_ALLOC_ADDRESS_BEARING 4u /* holds device addresses: never perturb */
#define HEXAD_ALLOC_LIVE 8u

/* One increment field on the device. Element (k, h) is at dptr[k*lev_stride + h*horiz_stride].
 * Names: "theta" (K), "rho" (dry density, kg m-3), "u" (edge-normal, m s-1, nhoriz = nEdges), "w" (nlev = nVertLevels+1),
 * and the species of the alloc map ("qv", "qc", "qr", "qi", "qs", "qg", "nifa", "nwfa", ...). "ps" is refused: the hex
 * write-back consumes rho; the ps -> rho hydrostatic map belongs to the host's variable change. Column-major host layout
 * (nVertLevels fastest): lev_stride = 1, horiz_stride = nlev. Plan layout: lev_stride = nhoriz, horiz_stride = 1. */
typedef struct hexad_field {
  const char* name;
  uint64_t dptr;
  int32_t dtype;  /* HEXAD_F64 or HEXAD_F32 */
  int32_t nlev;
  int64_t nhoriz;
  int64_t lev_stride, horiz_stride;
} hexad_field;
#define HEXAD_F64 0
#define HEXAD_F32 1

typedef struct hexad_vec {
  int32_t kind;      /* HEXAD_VEC_* */
  int32_t nfields;
  const hexad_field* fields;  /* INCREMENT */
  uint64_t seed;     /* SEED: the window's seed direction (x side at boundary 0, y side at the window end) */
  uint64_t arena;    /* ARENA: device float32 words, the whole arena (hexad_arena_words) */
  uint64_t words;
} hexad_vec;
#define HEXAD_VEC_ZERO 0
#define HEXAD_VEC_INCREMENT 1
#define HEXAD_VEC_SEED 2
#define HEXAD_VEC_ARENA 3

/* Transfer and barrier counters by category: [0] sweep (rule machinery), [1] checkpoint (boundary store / restore /
 * anchors / redo compare), [2] forward replay (the recorded forward's own transfers), [3] glue (what the ABI adds:
 * increment maps, seeds, dots, hashes). Counted at the driver entry points; nothing synchronises for them. */
typedef struct hexad_counters {
  uint64_t h2d_bytes[4], d2h_bytes[4], h2d_calls[4], d2h_calls[4];
  uint64_t ctx_syncs[4], stream_syncs[4], event_syncs[4];
  uint64_t boundaries_ram, boundaries_disk, boundary_ram_bytes, disk_reads_on_path;
  double disk_wait_s;
  uint64_t tl_steps, ad_steps;
} hexad_counters;

uint32_t hexad_abi_version(void);
const char* hexad_last_error(void);

/* plan_dir: the sealed package (plan.json, reference.jsonl, recording/, catalog-a.json, adjoint/alloc-map.json).
 * kset_dir: kernels.json + cubins/.  bundle_dir: the scheme bundle (schemes.json).  manifest: hexad.operator-manifest/v1.
 * options_json (NULL for defaults): {"steps":18, "bstore":"/local/nvme/dir" (required), "boundary_ram_gib":G
 * (default half of MemAvailable), "context":"own"|"primary", "alloc_map_dir":..., "scale_table":...,
 * "expect_manifest_sha256":"...", "check_every_pass":false,
 * "test_direction":{"direction":"field","stable_dir":true,"x_scale":1e-12,"y_scale":1e-20}} */
hexad_session* hexad_open(const char* plan_dir, const char* kset_dir, const char* bundle_dir, const char* manifest,
                          const char* options_json);
void hexad_close(hexad_session* s);
int hexad_steps(const hexad_session* s);
uint64_t hexad_arena_words(const hexad_session* s);
int hexad_alloc_table(const hexad_session* s, hexad_alloc* rows, size_t cap, size_t* n);
/* Zero-direction pass over every step, every boundary hash checked (binds the trajectory if not yet bound). */
int hexad_forward_check(hexad_session* s, uint64_t* steps_equal, uint64_t* nonzero_words);

int hexad_tl_init(hexad_session* s, const hexad_vec* dx);
int hexad_tl_step(hexad_session* s);
int hexad_tl_get(hexad_session* s, const hexad_vec* dx);   /* INCREMENT: dx = Ts(arena); ARENA: copy out */
/* The recorded nonlinear state at boundary `step` into INCREMENT-kind fields (assign; theta, rho,
 * u, w, species). Binds the trajectory first if needed (forward_check). */
int hexad_state_get(hexad_session* s, int step, const hexad_vec* out);
int hexad_ad_init(hexad_session* s, const hexad_vec* dx);  /* INCREMENT: arena = T_N^T(dx); SEED; ARENA */
int hexad_ad_put(hexad_session* s, const hexad_vec* dx);   /* arena += T_s^T(dx) at the current boundary */
int hexad_ad_step(hexad_session* s);                       /* reverse s -> s-1 */
int hexad_ad_get(hexad_session* s, const hexad_vec* dx);   /* INCREMENT: dx += T0^T(arena) (at boundary 0) */
int hexad_position(const hexad_session* s);
int hexad_get_counters(const hexad_session* s, hexad_counters* c);
/* print(): the manifest, the window arguments, the trajectory state, counters, held_upload_dropped_cotangent_sq. */
int hexad_print(const hexad_session* s, char* buf, size_t cap, size_t* needed);

/* ---- tests (the window's proofs through the ABI; T-map tests on a small card) ---- */
int hexad_test_dot(hexad_session* s, int which, uint64_t seed, double* p, double* q, uint64_t* nonfinite);
int hexad_result_sha256(const hexad_session* s, char out65[65]);
int hexad_test_nonzero(const hexad_session* s, uint64_t* out);
int hexad_test_boundary_bytes(hexad_session* s, int step, uint64_t alloc, uint64_t offset, uint64_t bytes, uint8_t* out);
hexad_session* hexad_tmap_open(const char* plan_dir, const char* alloc_map_dir, const char* options_json);
int hexad_test_mirror(const hexad_session* s, uint64_t alloc, uint64_t offset, uint64_t* dptr, uint64_t* words);
int hexad_test_alloc(hexad_session* s, uint64_t bytes, uint64_t* dptr);
int hexad_test_upload(const hexad_session* s, uint64_t dptr, const void* host, uint64_t bytes);
int hexad_test_download(const hexad_session* s, uint64_t dptr, void* host, uint64_t bytes);
int hexad_test_randn(const hexad_session* s, uint64_t dptr, uint64_t n, uint64_t seed);
int hexad_test_dot64(const hexad_session* s, uint64_t a, uint64_t b, uint64_t n, double* out);

#ifdef __cplusplus
}
#endif
#endif
