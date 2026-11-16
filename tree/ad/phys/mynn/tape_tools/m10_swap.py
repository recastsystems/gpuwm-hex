#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""x20 M10 swap-in (proof vehicle): patch a generated mynn_whole.gen.cu so the tape carrier's f_mynnedmf_post_run uses
the hand stage (cuda/stages/m10_out_rev.cuh, M10_TAPE_BRIDGE). TL and every other routine stay byte-identical.
Usage: m10_swap.py IN.gen.cu OUT.gen.cu"""
import sys, hashlib
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
inc = '#include "mw_types.cuh"\n'
assert s.count(inc) == 1
s = s.replace(inc, inc + '#define M10_TAPE_BRIDGE 1\n#include "stages/m10_out_rev.cuh"\n', 1)
i = s.index('void f_mynnedmf_post_run(int kte,')
anchor = 'dqs.rebase({1});\n'
j = s.index(anchor, i) + len(anchor)
s = s[:j] + ('if constexpr (mw_same<MR, MynnTapeF>::v) { m10_post_run_tape(kte, f_qc, f_qi, f_qs, delt, qv, qc, qi, qs, '
             'dqv, dqc, dqi, dqs); return; }\n') + s[j:]
open(out, 'w').write(s)
print(hashlib.sha256(open(src, 'rb').read()).hexdigest()[:8], '->', hashlib.sha256(s.encode()).hexdigest()[:8])
