// SPDX-License-Identifier: BSD-3-Clause
// Held lane oracle_drivers_seaice_finish (private autodiff program): the launch derivative image the composed
// window binds for the RUC sea-ice finish driver (hex pin 16251589, tree/src/hexcore/mpasphys/rrfs/
// ruc_lsm_8c9f1a8.py:5019-5054 drivers_seaice_finish, launch adapter :5398-5435). The TL and AD are P6's
// hand rules (tree/ad/phys/p6/cuda/seaice.ad.cu:309-402), included unchanged; this file adds only the extent guard.
#define SI_COMPILE_ONE 1
#define SI_FINISH_TL 1
#define SI_FINISH_AD 1
#include "../../../p6/cuda/seaice.ad.cu"

// Extent guard, run before the TL and before the AD. The adapter reads ncells per column from the table
// (python driver _finish_ice passes _scalar('ncells', n, 1): every column holds 1) and indexes k = c + i*nc for
// i < ncells. The window binds each table target as nc words (spec indirect.target_bytes = 4*nc), so a column with
// ncells != 1 would make the entries read and write past the bound shadows and pre-launch copies. Refused by
// name (source 5398) instead. Also refuses a null table word for any of the 33 named arrays (the forward's
// ruc.rs pointer_table never writes one for this entry).
extern "C" __global__ void oracle_drivers_seaice_finish__guard(unsigned long long *p, int nc, SeaiceFailure *fail) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nc) return;
    for (int k = 0; k < 33; ++k) if (p[k] == 0ull) { si_fail(fail, c, k + 1, c, 5398); return; }
    if (si_i(p, 0)[c] != 1) si_fail(fail, c, 1, c, 5398);
}
