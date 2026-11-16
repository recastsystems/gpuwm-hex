// the byte layout of ColumnFrame<float> and ColumnBars (CPU build of join.hpp), written as JSON for
// capture_gen.py (the card forward's probes write the frame by offset) and direct_gen.py. Build-time tool only.
#include <cstdio>
#include <cstddef>
#include "join.hpp"
using namespace direct; using namespace direct::join;
typedef ColumnFrame<float> F; typedef ColumnBars B;
#define O(T, m) printf("  \"%s\": %zu,\n", #m, offsetof(T, m))
int main() {
  printf("{\n \"frame\": {\n");
  O(F, n); O(F, dt); O(F, dheat); O(F, plume_active); O(F, plume_live); O(F, ktop); O(F, tail); O(F, plant); O(F, pre); O(F, drv);
  O(F, cloud); O(F, ccfg); O(F, blend); O(F, ess); O(F, qkwx); O(F, prod); O(F, qdiv); O(F, pred); O(F, diss); O(F, heat); O(F, solve); O(F, post);
  printf("  \"ccfg_pblh1\": %zu, \"ccfg_xland\": %zu, \"ccfg_closure\": %zu, \"ccfg_spp_pbl\": %zu, \"ccfg_k_tropo\": %zu,\n",
         offsetof(F, ccfg) + offsetof(M3Col<float>, pblh1), offsetof(F, ccfg) + offsetof(M3Col<float>, xland), offsetof(F, ccfg) + offsetof(M3Col<float>, closure),
         offsetof(F, ccfg) + offsetof(M3Col<float>, spp_pbl), offsetof(F, ccfg) + offsetof(M3Col<float>, k_tropo));
#ifdef JOIN_PMOM
  O(F, pm);
#define OP(m) printf("  \"pm_%s\": %zu,\n", #m, offsetof(F, pm) + offsetof(PlumeMom<float>, m))
  OP(on); OP(started); OP(adj_set); OP(pass); OP(dz1); OP(dz2); OP(psig); OP(adj); OP(om); OP(en); OP(c1); OP(dc); OP(cs); OP(dzi); OP(rhoi); OP(rho); OP(ea);
  printf("  \"pm_nup\": %d,\n", NUPM);
#endif
#ifdef JOIN_PHEAT
  O(F, ph);
#define OH(m) printf("  \"ph_%s\": %zu,\n", #m, offsetof(F, ph) + offsetof(PlumeHeat<float>, m))
  OH(on); OH(g1); OH(g2); OH(om); OH(en); OH(dc);
#endif
#ifdef JOIN_MIX
  printf("  \"mx\": %zu, \"mx_on\": %zu, \"mx_bad\": %zu, \"mx_t\": %zu, \"mx_qtw\": %zu, \"mx_sm2\": %zu, \"mx_sh2\": %zu, \"mx_jac\": %zu,\n",
         offsetof(F, mx), offsetof(F, mx) + offsetof(MixJac<float>, on), offsetof(F, mx) + offsetof(MixJac<float>, bad),
         offsetof(F, mx) + offsetof(MixJac<float>, t), offsetof(F, mx) + offsetof(MixJac<float>, qtw), offsetof(F, mx) + offsetof(MixJac<float>, sm2),
         offsetof(F, mx) + offsetof(MixJac<float>, sh2), offsetof(F, mx) + offsetof(MixJac<float>, jac));
#ifdef JOIN_CLOSURE
  printf("  \"mx_jq\": %zu,\n", offsetof(F, mx) + offsetof(MixJac<float>, jq));
#endif
#endif
  printf("  \"sizeof\": %zu\n },\n", sizeof(F));
  printf(" \"bars_sizeof\": %zu,\n", sizeof(B));
  printf(" \"const\": {\"MAX\": %d, \"DNI\": %d, \"CNI\": %d, \"ENI\": %d, \"TPNI\": %d, \"HNI\": %d, \"NHF\": %d, \"SNI\": %d,\n", MAX, DNI, CNI, ENI, TPNI, HNI, NHF, int(solves::NI));
  printf("  \"DZ\": %d, \"RHO\": %d, \"DF\": %d, \"SA\": %d, \"SD\": %d, \"Q\": %d, \"FU\": %d, \"FD\": %d, \"SUB\": %d, \"DET\": %d, \"COLD\": %d, \"CNEW\": %d, \"ZW\": %d, \"TK\": %d, \"QV\": %d,\n",
         solves::DZ, solves::RHO, solves::DF, solves::SA, solves::SD, solves::Q, solves::FU, solves::FD, solves::SUB, solves::DET, solves::COLD, solves::CNEW, solves::ZW, solves::TK, solves::QV);
  printf("  \"PS\": %d, \"UST\": %d, \"WSPD\": %d, \"OCE\": %d, \"PBLH\": %d, \"LAND\": %d}\n}\n", solves::PS, solves::UST, solves::WSPD, solves::OCE, solves::PBLH, solves::LAND);
}
