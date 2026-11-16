#!/usr/bin/env python3
# G4: move the P6 augmented-primal fragment tapes (one SiSiceClosureTape of 128 nodes each, ~3.7 KB) from the
# per-thread stack into a per-column frame in the P6 global scratch arena. Storage moves; no arithmetic changes.
import sys, re
path = sys.argv[1]
src = open(path).read().split('\n')
SITES = [  # (1-based line, type, name)
 (3852,'SiSnowPrepTape','prep'),(3876,'SiSnowCoupleTape','couple'),(3894,'SiSnowClosureTape','closure'),
 (3943,'SiSnowMeltSelectTape','select'),(3957,'SiSnowMeltEnergyTape','energy'),(3962,'SiSnowMeltWaterTape','water'),
 (3972,'SiSnowPostTape','post'),(3985,'SiSnowFluxTape','flux'),(3993,'SiSnowEnergyTape','final'),
 (4070,'SiSiceClosureTape','closure'),(4099,'SiSiceFluxTape','flux'),(4265,'SiIceSnowStateTape','state'),
 (4284,'SiIceOpticsTape','optics'),(4365,'SiIceOpticsBlendTape','blend'),(4519,'SiIcePreTape','prep')]
for slot,(ln,t,n) in enumerate(SITES):
    line = src[ln-1]; old = f'{t} {n};'
    assert line.count(old) == 1, (ln, line)
    src[ln-1] = line.replace(old, f'{t} &{n}=si_frame_slot<{t}>({slot});')
anchor = '__device__ __forceinline__ bool si_snow_column_record('
i = [k for k,l in enumerate(src) if l.startswith(anchor)]
assert len(i) == 1 and i[0] < SITES[0][0]-1
helper = f'''#ifndef SI_FRAME_DEFINED
#define SI_FRAME_DEFINED
// G4 (2026-11-06): the augmented primal's fragment tapes live in a per-column frame of the caller's global scratch
// arena, not on the thread stack. On the stack they made a 49,616-byte frame that the driver reserves for every
// resident thread of the card (13.1 GiB on an RTX PRO 6000). The host sets si_frame_base (column c's frame is
// si_frame_base + c * si_frame_stride, c the launch's global thread index, which every __aug kernel uses as its column)
// and sizes the arena from si_frame_bytes. Storage only: every value and slope is computed exactly as before.
enum {{ SI_FRAME_SLOT=3712, SI_FRAME_NSLOT={len(SITES)} }};
__device__ unsigned char *si_frame_base;
__device__ unsigned long long si_frame_stride;
__device__ const unsigned long long si_frame_bytes=(unsigned long long)SI_FRAME_SLOT*SI_FRAME_NSLOT;
template<class T> __device__ __forceinline__ T &si_frame_slot(int slot) {{
    static_assert(sizeof(T)<=SI_FRAME_SLOT&&alignof(T)<=16,"G4 frame slot too small");
    unsigned long long c=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    return *reinterpret_cast<T*>(si_frame_base+c*si_frame_stride+(unsigned long long)slot*SI_FRAME_SLOT);
}}
#endif'''
src[i[0]:i[0]] = helper.split('\n')
open(path,'w').write('\n'.join(src))
print('sites', len(SITES))
