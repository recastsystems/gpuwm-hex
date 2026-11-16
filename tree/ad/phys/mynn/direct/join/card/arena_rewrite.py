"""Arena form of the joined MYNN backward's block headers (card stack from ~58 KB to a few KB a thread).

usage: python3 arena_rewrite.py BLOCKS_DIR OUT_DIR   (writes OUT_DIR/*.hpp and OUT_DIR/arena-rewrite.txt)

Why: one thread a column keeps every block's fixed work arrays (HeatState, the tridiagonal sweeps' bars, the
predict and cloud work arrays, join.hpp's s[] and b[]) in its stack frame. The card driver reserves that frame for
every thread the card can hold (58,720 B x 288,768 threads = 17 GB on a 188-SM RTX PRO 6000), which does not fit
beside the 3 km program. Here every function-scope array of 32 elements or more, and every HeatState / State /
RepairState object, is rewritten to a macro (jw.hpp):
  - JW_ARENA undefined: the macro gives back the same declaration (T name[N]{} or T name[N]), so the CPU build of the
    rewritten headers is the same program;
  - JW_ARENA defined: the array is carved from a per-thread workspace the caller hands over (global memory on the
    card, a thread_local buffer on the CPU), zero-filled where the original had {} and released at the end of its
    scope (jw::Mark). Same operations on the same values in the same order: only where the words live changes.
Struct members, namespace-scope arrays and arrays with an initializer list are left as they are. Every rewrite is
listed in arena-rewrite.txt (the receipt). Statements the strict declaration pattern does not match are not touched.
"""
import re, sys
from pathlib import Path

TYPES = r'(?:P|double|float|int)'
DECL = re.compile(r'(?:(?<=[;{}])|(?<=^)|(?<=\n))([ \t]*)(' + TYPES + r')[ \t]+([^;(){}=]*?(?:\{\}|=[^;,(){}]+)?(?:,[^;(){}=]*?(?:\{\}|=[^;,(){}]+)?)*);',
                  re.M)
ONE = re.compile(r'^\s*([A-Za-z_]\w*)\s*((?:\[[^\]\[]+\])*)\s*(\{\}|=\s*[^,]+)?\s*$')
OBJ = re.compile(r'(?:(?<=[;{}])|(?<=\n))([ \t]*)((?:solves::)?(?:HeatState|State|RepairState)<P>)[ \t]*([A-Za-z_]\w*)\{\};')
CONST = {'MAX': 61}
# JW_MIN_WORDS: smallest array (elements) moved to the workspace (default 32: every array the 11:36Z arena moved).
# JW_OBJECTS=0 keeps the HeatState / State / RepairState objects in the stack frame (default 1: moved).
import os
MIN_WORDS = int(os.environ.get('JW_MIN_WORDS', '32')); OBJECTS = os.environ.get('JW_OBJECTS', '1') != '0'


def dims_of(s):
    return re.findall(r'\[([^\]]+)\]', s)


def count(d, consts):
    try:
        return eval(d.replace('solves::', ''), {}, consts)
    except Exception:
        return None


def scope_kinds(text):
    """for each '{' position: 'struct', 'namespace', 'init' or 'code'"""
    kinds, stack, out = {}, [], []
    for i, ch in enumerate(text):
        if ch == '{':
            head = text[max(0, i - 200):i]
            if re.search(r'struct\s+\w+\s*(?::[^{;]*)?$', head): k = 'struct'
            elif re.search(r'namespace\s+\w*\s*$', head): k = 'namespace'
            elif re.search(r'[\]=,(]\s*$', head) or re.search(r'\w\s*$', head) and not re.search(r'(\)|else|do|const|try)\s*$', head): k = 'init'
            else: k = 'code'
            kinds[i] = k; stack.append(k)
        elif ch == '}':
            if stack: stack.pop()
        out.append(stack[-1] if stack else 'file')
    return out


def rewrite(text, consts, name, log):
    inner = scope_kinds(text)
    # constants this header defines (constexpr int A=..., B=...)
    for m in re.finditer(r'constexpr\s+int\s+([^;]+);', text):
        for part in m.group(1).split(','):
            if '=' in part:
                k, v = part.split('=', 1); v = count(v.strip(), consts)
                if v is not None: consts[k.strip()] = v
    pieces, last, n_rw = [], 0, 0
    for m in DECL.finditer(text):
        if inner[m.start(2)] != 'code': continue
        ind, typ, body = m.group(1), m.group(2), m.group(3)
        decls = [d for d in re.split(r',(?![^\[]*\])', body)]
        parsed = [ONE.match(d) for d in decls]
        if not all(parsed): continue
        arrays = []
        for p in parsed:
            nm, dm, init = p.group(1), p.group(2), (p.group(3) or '').strip()
            if not dm: continue
            ds = dims_of(dm); tot = 1
            for d in ds:
                v = count(d, consts); tot = None if v is None or tot is None else tot * v
            if tot is not None and tot < MIN_WORDS: continue
            if init not in ('', '{}'): continue
            arrays.append((nm, ds, init == '{}', tot))
        if not arrays: continue
        keep = [d.strip() for d, p in zip(decls, parsed) if p.group(1) not in {a[0] for a in arrays}]
        out = [f'JW_SCOPE({n_rw});']
        for nm, ds, z, tot in arrays:
            if len(ds) == 1: out.append(f'JW_A({typ},{nm},({ds[0]}),{int(z)});')
            elif len(ds) == 2: out.append(f'JW_A2({typ},{nm},({ds[0]}),({ds[1]}),{int(z)});')
            else: raise SystemExit(f'{name}: 3-d array {nm}')
            log.append(f'{name}: {typ} {nm}{"".join("[" + d + "]" for d in ds)}{"{}" if z else ""} -> arena ({tot} words)')
        if keep: out.append(f'{typ} {",".join(keep)};')
        pieces.append(text[last:m.start(2)]); pieces.append(''.join(out)); last = m.end(); n_rw += 1
    pieces.append(text[last:]); text = ''.join(pieces)
    inner = scope_kinds(text); pieces, last = [], 0
    for m in OBJ.finditer(text):
        if inner[m.start(2)] != 'code' or not OBJECTS: continue
        pieces.append(text[last:m.start(2)]); pieces.append(f'JW_SCOPE(o{n_rw});JW_O({m.group(2)},{m.group(3)});'); last = m.end(); n_rw += 1
        log.append(f'{name}: {m.group(2)} {m.group(3)}{{}} -> arena')
    pieces.append(text[last:]); text = ''.join(pieces)
    return text, n_rw


JW = r'''#pragma once
// Workspace macros for the arena form (arena_rewrite.py). Without JW_ARENA each macro is the plain declaration.
#define JW_Z1 {}
#define JW_Z0
#define JW_CAT(a, b) a##b
#ifndef JW_ARENA
#define JW_SCOPE(id)
#define JW_A(T, name, N, Z) T name[N] JW_CAT(JW_Z, Z)
#define JW_A2(T, name, D1, D2, Z) T name[D1][D2] JW_CAT(JW_Z, Z)
#define JW_O(T, name) T name{}
#else
#ifndef JW_BLOCK
#define JW_BLOCK 32
#endif
namespace jw {
#if defined(__CUDACC_RTC__) || defined(__CUDA_ARCH__)
// one slot a thread of the block: the next free byte of the thread's workspace, its end and an overflow flag
__device__ inline char** slot() { __shared__ char* t[JW_BLOCK]; return &t[threadIdx.x]; }
__device__ inline char** stop() { __shared__ char* e[JW_BLOCK]; return &e[threadIdx.x]; }
__device__ inline int* over() { __shared__ int o[JW_BLOCK]; return &o[threadIdx.x]; }
__device__ inline void begin(char* base, long long bytes) { *slot() = base; *stop() = base + bytes; *over() = 0; }
#else
// CPU: one 1 MiB buffer a thread; the high-water mark is measured from its start
inline char* base() { static thread_local char buf[1 << 20]; return buf; }
inline char** slot() { static thread_local char* t = base(); return &t; }
inline char** stop() { static thread_local char* e = base() + (1 << 20); return &e; }
inline int* over() { static thread_local int o = 0; return &o; }
inline long long& hwm() { static thread_local long long h = 0; return h; }
#endif
struct Mark { char* m; Mark() : m(*slot()) {} ~Mark() { *slot() = m; } };
template<class T> inline T* take(long long n, int zero) {
  char* p = *slot(); char* q = p + ((n * (long long)sizeof(T) + 15) & ~15LL);
  if (q > *stop()) { *over() = 1; q = p; }            // overflow: flagged (the caller turns it into a status); never written past the end
  *slot() = q;
#if !(defined(__CUDACC_RTC__) || defined(__CUDA_ARCH__))
  if (q - base() > hwm()) hwm() = q - base();
#endif
  T* r = reinterpret_cast<T*>(p); if (zero) for (long long i = 0; i < n; ++i) r[i] = T(0); return r;
}
template<class T> inline T& obj() {
  T* r = reinterpret_cast<T*>(take<char>((long long)sizeof(T), 0));
  char* c = reinterpret_cast<char*>(r); for (long long i = 0; i < (long long)sizeof(T); ++i) c[i] = 0;   // value-init of a POD: all bytes 0
  return *r;
}
}
#define JW_SCOPE(id) ::jw::Mark JW_CAT(jw_mark_, id)
#define JW_A(T, name, N, Z) T* name = ::jw::take<T>((N), Z)
#define JW_A2(T, name, D1, D2, Z) T (*name)[D2] = reinterpret_cast<T(*)[D2]>(::jw::take<T>((long long)(D1) * (D2), Z))
#define JW_O(T, name) T& name = ::jw::obj<T>()
#endif
'''


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
    log, consts = [], dict(CONST)
    # headers that define constants first, so later headers can size their arrays
    order = ['numeric.hpp', 'column.hpp', 'repair.hpp', 'heat.hpp', 'predict.hpp', 'conversions.hpp', 'solves.hpp', 'dissipation.hpp',
             'driver.hpp', 'ess_constants.hpp', 'ess_scalar.hpp', 'ess.hpp', 'turbprod.hpp', 'cloud.hpp', 'plume_column.hpp', 'plume.hpp',
             'plume_tail.hpp', 'join.hpp']
    names = order + sorted(p.name for p in src.glob('*.hpp') if p.name not in order)
    total = 0
    for nm in names:
        p = src / nm
        if not p.exists(): continue
        t = p.read_text()
        if nm == 'solves.hpp':
            consts['NO'] = consts['MAX']
        new, k = rewrite(t, consts, nm, log); total += k
        if k:
            new = new.replace('#pragma once', '#pragma once\n#include "jw.hpp"', 1)
        (dst / nm).write_text(new)
    (dst / 'jw.hpp').write_text(JW)
    (dst / 'arena-rewrite.txt').write_text('\n'.join(log) + '\n')
    print(f'arena_rewrite: {total} declarations rewritten in {len(set(l.split(":")[0] for l in log))} headers; '
          f'{len(log)} arrays/objects moved to the workspace (list in {dst / "arena-rewrite.txt"})')


if __name__ == '__main__':
    main()
