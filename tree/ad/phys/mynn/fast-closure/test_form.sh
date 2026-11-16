#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The tape-free route's default boundary-layer form, pinned at its source: from the captured MYNN-EDMF forward
# (sha256 70da4af7...), the generator's output, the source after the height knockout (pknock.py O) and the composed
# source the three generated images are built from must be these bytes, and the capture forward's source must be the
# same with and without the returned height's stage (the frame does not move). Processors only, seconds.
#   test_form.sh CAPTURED.cu        exit 0: PASS
set -u
M=$(cd "$(dirname "$0")/.." && pwd); J=$M/direct/join; CAP=$(readlink -f "${1:?the captured MYNN-EDMF forward source}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; fail=0; export CUDA_VISIBLE_DEVICES=
ok() { if [ "$2" = "$3" ]; then echo "ok   $1 $2"; else echo "FAIL $1: $2, want $3"; fail=1; fi; }
ok "captured forward" "$(sha256sum "$CAP" | cut -c1-8)" 70da4af7
python3 "$M/gen_whole.py" --freeze closure --freeze-mask 0x7c9 --weakshear 1e-5 --plume-live momheat --fused m8 "$CAP" \
  "$T/p.gen.cu" > "$T/gen-info.json" 2> /dev/null || { echo "FAIL generator"; exit 1; }
ok "generated source (--freeze closure --freeze-mask 0x7c9 --weakshear 1e-5 --plume-live momheat --fused m8)" \
  "$(sha256sum "$T/p.gen.cu" | cut -c1-8)" 307b51e7
python3 "$M/fast-closure/pknock.py" "$T/p.gen.cu" "$T/base.gen.cu" O > /dev/null || { echo "FAIL pknock"; exit 1; }
ok "height held inside the column, live in what it returns (pknock.py O)" "$(sha256sum "$T/base.gen.cu" | cut -c1-8)" 62bd322c
bash "$M/fast-closure/compose-m4l.sh" "$T/base.gen.cu" "$T/src" > /dev/null 2>&1 || { echo "FAIL compose"; exit 1; }
ok "composed source (compose-m4l.sh)" "$(sha256sum "$T/src/mynn_whole.gen.cu" | cut -c1-8)" 33ba6f36
for v in "a:-DJOIN_PMOM -DJOIN_MIX -DJOIN_CLOSURE -DJOIN_PHEAT" "b:-DJOIN_PMOM -DJOIN_MIX -DJOIN_CLOSURE -DJOIN_PHEAT -DJOIN_PBLH"; do
  n=${v%%:*}; g++ -std=c++17 ${v#*:} -I"$J/blocks" "$J/card/frame_layout.cpp" -o "$T/fl$n" && "$T/fl$n" > "$T/layout$n.json" \
    && MIX_WS_T=1e-5 python3 "$J/card/capture_gen.py" "$CAP" "$T/layout$n.json" "$T/capture$n.cu" > "$T/cap$n.json" || { echo "FAIL capture $n"; exit 1; }
done
ok "capture forward source, with the height's stage" "$(sha256sum "$T/captureb.cu" | cut -c1-8)" "$(sha256sum "$T/capturea.cu" | cut -c1-8)"
ok "frame bytes a column" "$(grep -o '"frame_bytes": [0-9]*' "$T/capb.json" | grep -o '[0-9]*$')" 90960
# F42: the plume transposes hold cdet = 1/45 and have no pressure-gradient term, so the capture refuses a forward with
# pgfac != 0 or another cdet (it would give a joined backward that is the transpose of no tangent).
sed 's/constexpr float pgfac = 0.00f;/constexpr float pgfac = 0.10f;/' "$CAP" > "$T/pg.cu"
sed 's|constexpr float cdet = ((1.f) / (45.f));|constexpr float cdet = ((1.f) / (40.f));|' "$CAP" > "$T/cd.cu"
for v in pg cd; do
  MIX_WS_T=1e-5 python3 "$J/card/capture_gen.py" "$T/$v.cu" "$T/layoutb.json" "$T/cap-$v.cu" > /dev/null 2> "$T/cap-$v.err"
  ok "capture refuses the plume constant changed ($v)" "$(grep -c 'REFUSED' "$T/cap-$v.err")$( [ -e "$T/cap-$v.cu" ] && echo ' wrote' )" 1
done
[ $fail = 0 ] && echo PASS || echo FAIL; exit $fail
