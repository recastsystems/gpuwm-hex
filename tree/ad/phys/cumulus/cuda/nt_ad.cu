// SPDX-License-Identifier: Apache-2.0
// Derivative entries of every New Tiedtke launch (cu_ntiedtke; MPAS 44c0849 / MMM a4baf7f3 as hex ports it),
// generated from the forward's own text.
//
//   NAME__tl  tangent: every f32 column array the launch writes gets J tan (in place semantics: words the forward
//             leaves alone pass through); arrays it only reads give their tangent in.
//   NAME__ad  taped reverse: J^T of the written arrays' adjoints (snapshot in), accumulated into every array's adjoint.
//
// Both are the forward source itself (nt-forward.cu, included verbatim), its scheme region re-typed over AF / AR by
// gen/gen_nt.py: every primal word is the forward's own operation, so the trajectory is the forward's, bit for bit.
// Frozen at the recorded path: every branch, integer conversion, min/max arm, level index, cloud type and the slab
// latch are decided on the pre-launch primal words exactly as the forward decides them.
#include "nt-forward.cu"
#include "nt_ad.cuh"
namespace tl {
typedef AF R;
#include "nt_R.inc"
}  // namespace tl
namespace adr {
typedef AR R;
#include "nt_R.inc"
}  // namespace adr
#include "nt_entries.inc"
