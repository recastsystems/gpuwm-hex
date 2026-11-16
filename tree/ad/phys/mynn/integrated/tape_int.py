#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# x20 mynn-integrated (2026-11-09): the tape header for the integrated MYNN image (stages M8 + M9 as external records on
# the merge-all batched sweep). Build time only; writes a copy, the tree's cuda/mynn_tape.cuh is not edited.
#
# Breakage it prevents: on ad/x20-merge-all the sweep is batched (MW_REV_BATCH 8, physics-adjoints-fused bb9ddd5e).
#  - M8's marker hook (`if (n.left == -2) mw_ext_rev`) exists only in the unbatched #else sweep, so with the batched sweep an
#    M8 marker (left -2, right n, weight 0/1) would be swept as a plain node: rb_add(right = n, weight * bar) into a wrong cell.
#  - patch_m9.py derives reverse_until from the unbatched loop text; on the batched header its anchors miss.
# Here: one batched reverse_until(start) with the same values and order as the batched reverse() for plain and fused nodes.
# At a marker (left == -2) the batch's register bars are written back first (so the stage reads and adds into memory), then
#   right == -2 (M9, gen_m9.py marker {-2, -2, nrec}): return the marker index (the AD entry runs m9_ext_reverse, resumes below);
#   otherwise (M8, mw_m8_predict.cuh marker {-2, n, edmf}): mw_ext_rev(this, marker) and the next batch starts one slot below.
# reverse() becomes reverse_until(used - 1) followed by the M9 handoff only through the AD entry's loop (patch_m9.py), and a
# plain reverse() call on an image with no M9 marker is reverse_until(used - 1) == -1.
# usage: tape_int.py IN/mynn_tape.cuh OUT/mynn_tape.cuh
import sys
t = open(sys.argv[1]).read()
a = '#if MW_REV_BATCH > 0\n'
assert t.count(a) == 1
b = '    __device__ void reverse() {\n        constexpr int BR = MW_REV_BATCH;\n'
assert t.count(b) == 1, 'batched reverse() anchor'
NEW = r'''    // x20 mynn-integrated: n consecutive slots for an external stage (M9 record); count mode only counts (= reserve()).
    __device__ int ext_reserve(int n) {
        if (used < 0 || (long long)used + n > (long long)capacity || used >= (1 << 29) - n) { fail(0); return -1; }
        const int id = used; used += n; return id;
    }
    // x20 mynn-integrated: the batched sweep from slot `start` down; returns the index of the first M9 marker met, or -1.
    __device__ int reverse_until(int start) {
        constexpr int BR = MW_REV_BATCH;
        int i = start;
        while (i >= 0) {
            const int lo = i - BR + 1 > 0 ? i - BR + 1 : 0;
            const int nb = i - lo + 1;
            double rb[BR]; int nl[BR], nr[BR]; double nw[BR];
#pragma unroll
            for (int k = 0; k < BR; ++k) {
                if (k < nb) { rb[k] = bars[i - k]; const MynnAdNode n = nodes[i - k]; nl[k] = n.left; nr[k] = n.right; nw[k] = n.weight; }
                else { rb[k] = 0.0; nl[k] = -1; nr[k] = -1; nw[k] = 0.0; }
            }
            unsigned dirty = 0u;
            bool skip = false, stop = false;
            int next = lo - 1;
#pragma unroll
            for (int k = 0; k < BR; ++k) {
                if (k < nb && !stop) {
                    if (skip) { skip = false; }
                    else {
                        const int c = i - k;
                        const double b = rb[k];
                        if (b != 0.0) {
                            if (nl[k] == -2) {
#pragma unroll
                                for (int q = 0; q < BR; ++q) if (dirty & (1u << q)) bars[i - q] = rb[q];
                                dirty = 0u;
                                if (nr[k] == -2) return c;
#ifdef MW_EXT_REV
                                mw_ext_rev(this, c);
#else
                                fail(3);
#endif
                                next = c - 1; stop = true;
                            } else if (nr[k] >= (1 << 29)) {
                                const int kind = nr[k] >> 29, r = nr[k] & ((1 << 29) - 1);
                                const double w0 = (k + 1 < nb) ? nw[k + 1 < BR ? k + 1 : BR - 1] : nodes[c - 1].weight;
                                if (kind == 1) {
                                    rb_add(rb, dirty, i, lo, r, nw[k] * b);
                                    rb_add(rb, dirty, i, lo, nl[k], w0 * b);
                                } else {
                                    const double tq = nw[k] * b;
                                    if (tq != 0.0) {
                                        if (kind == 2) rb_add(rb, dirty, i, lo, nl[k], tq);
                                        rb_add(rb, dirty, i, lo, r, w0 * (-tq));
                                    }
                                }
                                if (k + 1 < nb) skip = true; else next = c - 2;
                            } else {
                                const double lw = nr[k] >= 0 ? 1.0 : nw[k];
                                if (nl[k] >= 0) rb_add(rb, dirty, i, lo, nl[k], lw * b);
                                if (nr[k] >= 0) rb_add(rb, dirty, i, lo, nr[k], nw[k] * b);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < BR; ++k) if (dirty & (1u << k)) bars[i - k] = rb[k];
            i = next;
        }
        return -1;
    }
    __device__ void reverse() {
        const int m = reverse_until(used - 1);
        if (m >= 0) fail(3);   // an M9 marker outside the AD entry's reverse_until loop
    }
    __device__ void reverse_batched_plain() {
        constexpr int BR = MW_REV_BATCH;
'''
t = t.replace(b, NEW, 1)
open(sys.argv[2], 'w').write(t)
print('tape_int: batched reverse_until + ext_reserve; reverse() = reverse_until(used - 1)')
