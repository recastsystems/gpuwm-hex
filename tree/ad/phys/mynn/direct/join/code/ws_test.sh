#!/bin/bash
# build and run the workspace test (code/ws_test.cpp) for the four forms of the joined backward, in the
# arena form the card ships (m150o0). CPU only, about a minute. ws_test.sh [BUILD_DIR]   exit 0: all four PASS.
set -u
J=$(cd "$(dirname "$0")/.." && pwd); O=${1:-$J/build/ws}; mkdir -p $O; rc=0
( cd $J/card && JW_MIN_WORDS=150 JW_OBJECTS=0 python3 arena_rewrite.py ../blocks $O/blocks > $O/arena.log 2>&1 ) || { echo "arena_rewrite failed"; exit 2; }
for v in "default:" "mix:-DJOIN_MIX" "pmom:-DJOIN_PMOM" "pmmix:-DJOIN_PMOM -DJOIN_MIX"; do
  n=${v%%:*}; d=${v#*:}
  g++ -std=c++17 -O1 -DJW_ARENA $d -I$O/blocks $J/code/ws_test.cpp -o $O/ws_test-$n 2> $O/ws_test-$n.err || { echo "$n: build failed"; rc=2; continue; }
  out=$($O/ws_test-$n); r=$?; echo "$n: $(echo $out)"; [ $r = 0 ] || rc=1
done
exit $rc
