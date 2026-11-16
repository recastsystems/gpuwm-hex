#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The processor tests of the joined (tape-free) boundary-layer backward, every stage the default form builds with
# (-DJOIN_PMOM -DJOIN_MIX -DJOIN_CLOSURE -DJOIN_PHEAT -DJOIN_PBLH): each transpose against the forward statements it
# reverses, a planted error in each, and the workspace test of the arena form the card runs. CPU only, a few minutes.
#   run_cpu_tests.sh [BUILD_DIR]      exit 0: every test passes
set -u
J=$(cd "$(dirname "$0")/.." && pwd); O=${1:-$J/build/cpu}; mkdir -p "$O"; rc=0
one() { local n=$1; shift; echo "== $n"; "$@" || { echo "FAIL $n"; rc=1; }; }
one mix   bash "$J/code/mix_test.sh" "$O/mix"
one clos  bash "$J/code/clos_test.sh" "$O/clos"
one ws    bash "$J/code/ws_test.sh" "$O/ws"
one pblh  bash "$J/code/pblh_test.sh" "$O/pblh" 4000
for t in "pmom:-DJOIN_PMOM" "pheat:-DJOIN_PMOM -DJOIN_PHEAT" \
         "pblhj:-DJOIN_PBLH -DJOIN_PMOM -DJOIN_MIX -DJOIN_CLOSURE -DJOIN_PHEAT -ffp-contract=off"; do
  n=${t%%:*}; d=${t#*:}
  one $n bash -c "g++ -std=c++17 -O1 $d -I'$J/blocks' '$J/code/${n}_test.cpp' -o '$O/${n}_test' && '$O/${n}_test'"
done
echo "== $([ $rc = 0 ] && echo PASS || echo FAIL)"; exit $rc
