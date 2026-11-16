#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Byte test of gen_whole.py --weakshear (the weak-shear rule of the exact boundary-layer form). CPU, about a minute.
#   test_weakshear.sh CAPTURED_SOURCE          or          CAPTURES=<capture root> test_weakshear.sh   (as run_p3.sh)
# 1. Option off: the generator's base output for ten option sets has the sha256 it had before the option, so no route
#    that does not ask for the rule changes by a byte (the first line is the plain command of run_p3.sh).
# 2. Option on (exact form): the base differs from the exact form's (968addc5) only by the header line, the mw_pv helpers
#    and ONE statement, with the threshold as given. THE DEFAULT IS PINNED: `--weakshear` alone reports 0.0001 and writes
#    d2d0319a (a changed default passed this test before it was pinned). THE RULE'S PLACE IS PINNED: the statement is in
#    f_mym_turbulence and the line after it is `elq = r6_mul(el(k), qkw(k));`, so it sits after the scheme's own floors
#    on sm and sh and before the diffusivities (a rule moved before the floors passed this test before it was pinned;
#    it is another derivative wherever a floor binds).
# 3. Refusals: under --freeze closure (the functions are held already), with --mix-live (no rule on that path yet), and a
#    threshold that is not a finite positive number (inf would hold every interface under the exact form's name).
# 4. The composed source (integrated/compose.sh BASE OUT auto) is 793084cd, the source the three images of the kit
#    bl-weakshear-849608c8 were built from (weakshear/kit-bl-weakshear-849608c8.SHA256SUMS).
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-${CAPTURES:+$CAPTURES/hrrrv5_mynn_dd_c6/objects/70da4af74192b47fef6dd002101d64a22659b38a93c496a7832ec49071689727.cu}}
if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
  echo "test_weakshear.sh: give the captured forward source (70da4af7....cu) or CAPTURES=<capture root>" >&2; exit 2
fi
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; fail=0
RULE_A='{ const bool mw_ws_hold = mw_pv(gm(k)) < (double)('
RULE_B='); if (mw_ws_hold) { sm(k) = mw_frz(sm(k)); sh(k) = mw_frz(sh(k)); qdiv = mw_frz(qdiv); gamt = mw_frz(gamt); gamq = mw_frz(gamq); gamv = mw_frz(gamv); } }'
while IFS='|' read -r want opts; do
  python3 $D/gen_whole.py $opts "$SRC" $T/o.cu > /dev/null || { echo "FAIL rc: ${opts:-(no option)}"; fail=1; continue; }
  got=$(sha256sum $T/o.cu | cut -c1-8)
  if [ "$got" = "$want" ]; then echo "ok   $got  ${opts:-(no option)}"; else echo "FAIL $got (want $want)  ${opts:-(no option)}"; fail=1; fi
done <<EOF
b8d132a0|
bf9516c0|--fused m8
b603c942|--freeze none --fused m8
968addc5|--freeze none --fused m8 --l25-quot exact
0f49b8b4|--freeze none --l25-quot hold --fused m8
643d8398|--freeze closure-plume --plume-hold w --freeze-mask 0x1482 --fused m8
1b423144|--bl-live q1-ramp --fused m8
4c0d22ec|--freeze closure --mix-live q:1,s:1 --fused m8
5b956642|--freeze closure --freeze-mask 0x37ff --gate-q 0.5 --fused m8
06a948d9|--freeze none --frame noinline
EOF
python3 $D/gen_whole.py --freeze none --fused m8 --l25-quot exact "$SRC" $T/x.cu > /dev/null
while IFS='|' read -r t want wsha; do
  python3 $D/gen_whole.py --freeze none --fused m8 --l25-quot exact --weakshear $t "$SRC" $T/w.cu > $T/w.json || { echo "FAIL rc --weakshear $t"; fail=1; continue; }
  v=$(python3 -c "import json;print(repr(json.load(open('$T/w.json'))['weakshear']))")
  n=$(diff $T/x.cu $T/w.cu | grep -c '^>'); r=$(diff $T/x.cu $T/w.cu | grep -c '^<'); s=$(grep -cF "$RULE_A$v$RULE_B" $T/w.cu)
  got=$(sha256sum $T/w.cu | cut -c1-8)
  # the rule's place: the function the statement is in, and the line after it
  place=$(awk -v a="$RULE_A" '
    /^__device__ __(noinline|forceinline)__ (void|MR) f_[A-Za-z0-9_]+\(/ { fn = $0; sub(/^__device__ __(noinline|forceinline)__ (void|MR) /, "", fn); sub(/\(.*/, "", fn) }
    hit == 1 { print fn "|" $0; hit = 2 }
    index($0, a) == 1 { hit = 1 }' $T/w.cu)
  if [ "$n" = 10 ] && [ "$s" = 1 ] && [ "$r" = 0 ] && [ "$v" = "$want" ] && [ "$got" = "$wsha" ] && [ "$place" = "f_mym_turbulence|elq = r6_mul(el(k), qkw(k));" ]; then
    echo "ok   $got  --weakshear ${t:-(default)} = $v: 10 lines added, none removed, one rule statement, before elq in f_mym_turbulence"
  else
    echo "FAIL --weakshear ${t:-(default)}: threshold $v (want $want), sha $got (want $wsha), added $n, removed $r, rule statements $s, place '$place'"; fail=1
  fi
done <<EOF
|0.0001|d2d0319a
3e-5|3e-05|3e427330
3e-4|0.0003|4bc025d8
EOF
for opts in "--freeze closure --weakshear" "--freeze closure --mix-live q:1 --weakshear" "--freeze none --weakshear 0" \
            "--freeze none --weakshear inf" "--freeze none --weakshear nan" "--freeze none --weakshear -1e-4"; do
  if python3 $D/gen_whole.py $opts "$SRC" $T/r.cu > /dev/null 2>&1; then echo "FAIL not refused: $opts"; fail=1; else echo "ok   refused: $opts"; fi
done
python3 $D/gen_whole.py --freeze none --fused m8 --l25-quot exact --weakshear "$SRC" $T/base.gen.cu > /dev/null \
  && bash $D/integrated/compose.sh $T/base.gen.cu $T/out auto > $T/compose.log 2>&1
got=$(sha256sum $T/out/mynn_whole.gen.cu 2>/dev/null | cut -c1-8)
if [ "$got" = 793084cd ]; then echo "ok   $got  composed source of the exact form with the rule (the kit's source)"
else echo "FAIL composed source ${got:-none} (want 793084cd)"; tail -3 $T/compose.log; fail=1; fi
[ $fail = 0 ] && echo "PASS" || { echo "FAILED"; exit 1; }
