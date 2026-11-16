#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The processor proofs on every captured launch bundle in DIR (nthost built in B).
#   proof.sh B DIR OUT   -> OUT/proof.log, OUT/census.json (largest tape per column, per kernel)
B=$1; DIR=$2; OUT=$3; mkdir -p $OUT; : > $OUT/proof.log
for f in $DIR/*.bin; do
  n=$(basename $f .bin)
  { echo "== $n"; $B/nthost-p0 $f fwd | tail -1; echo "fwd rc $?"
    for s in 1 2; do $B/nthost-p0 $f dot $s; echo "dot$s rc $?"; done
    $B/nthost-p0 $f fd 1; echo "fd rc $?"
    $B/nthost-p1 $f dot 1 | sed 's/^DOT/PLANT-AR/'; echo "plantAR rc $?"
    $B/nthost-p2 $f dot 1 | sed 's/^DOT/PLANT-AF/'; echo "plantAF rc $?"; } >> $OUT/proof.log 2>&1
done
python3 - $OUT <<'PY'
import re, sys, json
c = {}
for l in open(sys.argv[1] + '/proof.log'):
    m = re.match(r'DOT (\w+) seed 1 .*tape max (\d+)', l)
    if m: c[m.group(1)] = max(c.get(m.group(1), 0), int(m.group(2)))
json.dump(c, open(sys.argv[1] + '/census.json', 'w'), indent=1); print(c)
PY
