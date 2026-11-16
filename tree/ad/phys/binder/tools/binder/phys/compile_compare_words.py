#!/usr/bin/env python3
"""Compile the source-owned comparison kernel; not used by native replay."""
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from cupy.cuda import nvrtc

root = Path(__file__).resolve().parent
source = (root / 'compare_words.cu').read_text()
for arch in ('89', '120'):
    program = nvrtc.createProgram(source, 'compare_words.cu', (), ())
    try:
        nvrtc.compileProgram(program, (f'--gpu-architecture=sm_{arch}', '--std=c++17'))
        image = nvrtc.getCUBIN(program)
        (root / 'runner' / 'fixtures' / f'compare_words-sm_{arch}.cubin').write_bytes(image)
    finally:
        nvrtc.destroyProgram(program)
