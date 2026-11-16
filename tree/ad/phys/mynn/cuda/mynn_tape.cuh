// Per-column linear tape for the generated MYNN differentiator.
// This file contains no forward MYNN or library implementation.
#pragma once

// mynn-tape (2026-11-08): 16 B per node (was 24). Every binary node the MYNN rules record has left weight 1.0
// (operator+ pushes (1, 1), operator- pushes (1, -1)), so a node stores one weight: the right weight of a binary node
// (right >= 0), the left weight of a unary node, 0 for a leaf. The reverse multiplies the left adjoint by 1.0 exactly
// as before (1.0 * b == b in IEEE, --fmad=false), so the reverse words are unchanged. A binary push with any other
// left weight is refused (failure slot 2), never stored wrong.
#ifndef MW_REV_PF
#define MW_REV_PF 128
#endif
// speed16-precision (2026-11-13): MW_TAPE_PREC sets the storage and sweep precision of the tape. A precision option,
// off by default; it moves the gradient by rounding, and its error against the double tape is a measured figure
// of each use, never claimed as the same bytes.
//   0 (default): double weights, double bars, double sweep: every preprocessed token is the one before this option.
//   1: float bars (the adjoint accumulators), double weights; each contribution weight * bar is formed in double from the
//      float bar and rounded once where it is added into a float bar. Node layout unchanged (16 B), bar 4 B: 20 B a slot
//      against 24. External records (M8) keep working: their record words live in the node weights.
//   2: float weights and float bars, float sweep (node 12 B + bar 4 B = 16 B a slot). Refused with external records
//      (MW_EXT_REV: M8 packs two floats into a weight word) and by the binder (it takes 16 or 24 B nodes only).
//   3: float weights, double bars, double sweep (the partials stored in float, every accumulation in double): node 12 B
//      + bar 8 B = 20 B a slot. Refused with external records and by the binder, as 2.
#ifndef MW_TAPE_PREC
#define MW_TAPE_PREC 0
#endif
#if MW_TAPE_PREC == 2 || MW_TAPE_PREC == 3
#define MW_WCAST(x) (float)(x)
#else
#define MW_WCAST(x) x
#endif
#if MW_TAPE_PREC == 0
#define MW_WT double
#define MW_BT double
#define MW_AT double
#elif MW_TAPE_PREC == 1
#define MW_WT double
#define MW_BT float
#define MW_AT double
#elif MW_TAPE_PREC == 2
#ifdef MW_EXT_REV
#error "MW_TAPE_PREC=2 stores float weights; external reverse records (gen_whole.py --fused) keep data in double weights"
#endif
#define MW_WT float
#define MW_BT float
#define MW_AT float
#elif MW_TAPE_PREC == 3
#ifdef MW_EXT_REV
#error "MW_TAPE_PREC=3 stores float weights; external reverse records (gen_whole.py --fused) keep data in double weights"
#endif
#define MW_WT float
#define MW_BT double
#define MW_AT double
#else
#error "MW_TAPE_PREC must be 0, 1, 2 or 3"
#endif
struct MynnAdNode {
    int left, right;
    MW_WT weight;
};

// The caller owns each context and all its buffers for one column. Buffer
// capacities are in nodes. The caller clears failure slots before launch.
#ifdef MW_EXT_REV
// x20 M8 (gen_whole.py --fused m8): an external reverse record. Its marker node has left == -2 and its bar preset to
// 1.0; the sweep hands the marker to mw_ext_rev (defined by the fused stage header), which reads the record's data
// nodes below the marker and the output leaves above it and adds the stage's adjoint into its input words' bars.
struct MynnAdContext;
__device__ __noinline__ void mw_ext_rev(MynnAdContext* c, int marker);
#endif
struct MynnAdContext {
    MynnAdNode* nodes;
    MW_BT* bars;
    int capacity;
    int used;
    int* failures; // provisional slots: overflow, domain, mixed contexts / unencodable node, arena
    int count_only = 0;
    // x20 M8: the column's arena tile (free after the forward), scratch for external reverse records; set by the AD
    // entry before reverse() when the image has fused stages (gen_whole.py --fused), else unused.
    unsigned char* scr = nullptr; unsigned long long scr_cap = 0; int scr_lane = 0;
    // x20 M8: n raw slots for an external record (data nodes, marker, leaves); the caller writes them unless count_only.
    __device__ int reserve(int n) {
        if (used < 0 || (long long)used + n > (long long)capacity || used >= (1 << 29) - n) { fail(0); return -1; }
        const int id = used; used += n; return id;
    }

    __device__ void fail(int slot) { if (failures) failures[slot] = 1; }
    __device__ int push(int l, double lw, int r, double rw) {
        if (used >= capacity || used < 0 || used >= (1 << 29)) { fail(0); return -1; } // right < 2^29 marks a plain node (push2)
        if (r >= 0 && lw != 1.0) { fail(2); return -1; }
        const int id = used++;
        if (count_only) return id;
        nodes[id] = {l, r, MW_WCAST(r >= 0 ? rw : lw)};
        bars[id] = 0.0;
        return id;
    }
    // mynn-adjoint-v2 (2026-11-08): a fused node takes two slots, k and k+1; consumers reference k+1. Slot k+1 holds
    // left, right | kind << 29 and its weight; slot k holds the second weight, and its bar stays 0 (nothing references
    // slot k), so a reverse that reaches it skips it as before. kind 1 (product a*bv + b*av): replaces the two unary
    // nodes and the binary sum the separate rules push; kind 2 (quotient (a - b*w) * (1/den)) and kind 3 (the same
    // with a passive) replace unary + binary/negation + unary. Plain nodes keep right < 2^29.
    __device__ int push2(int l, double w0, int r, double w1, int kind) {
        if (used + 2 > capacity || used < 0 || used >= (1 << 29) - 2) { fail(0); return -1; }
        const int id = used + 1;
        used += 2;
        if (count_only) return id;
        nodes[id - 1] = {-1, -1, MW_WCAST(w0)};
        bars[id - 1] = 0.0;
        nodes[id] = {l, r | (kind << 29), MW_WCAST(w1)};
        bars[id] = 0.0;
        return id;
    }
    __device__ int leaf() { return push(-1, 0.0, -1, 0.0); }
    __device__ void seed(int id, double b) {
        if (id >= 0 && id < used) bars[id] += b;
        else if (id >= 0) fail(0);
    }
    // mynn-adjoint-v2: a plain node whose operand is the slot the sweep reads next (i - 1, the common chain of
    // temporaries) hands its contribution over in a register instead of a store the next iteration reloads. The next
    // iteration adds it to the cell exactly as the store would have (memory value + c0, then + c1 when both operands
    // are that slot), and writes the sum back, so every cell ends with the value and rounding order of before (the
    // leaf adjoints read after the sweep included).
    // x20 physics-adjoints-fused (2026-11-09): batched reverse sweep (MW_REV_BATCH slots per batch, default 8). The
    // sweep was latency-bound: each slot waited on its own bar load, then on its node load, then on the parent's
    // read-modify-write (~2,500 cycles per slot, prof4), with 12 warps per SM. Now a batch loads the bars and nodes of
    // slots i..i-B+1 together (independent loads in flight at once), keeps those bars in registers, and walks the
    // batch in the old order. A contribution to a cell inside the batch is added to its register (cells receive
    // contributions only from higher slots, so the register already holds every earlier contribution: memory value
    // first, then the batch's slots in descending order, the program order of the old sweep, which also covers the old
    // distance-1 register hand-off); a contribution below the batch is the old read-modify-write. At the end of the
    // batch every cell that received a register contribution is written back with that value. Same values, same
    // order, same IEEE operations (--fmad=false): every bar cell ends with the bits of the old sweep, leaves included.
    // A fused node's second slot is skipped exactly as before (nothing references it); when that slot is the first
    // below the batch, the next batch starts one slot lower. MW_REV_BATCH=0 keeps the old sweep.
#ifndef MW_REV_BATCH
#define MW_REV_BATCH 8
#endif
#if MW_REV_BATCH > 0
    __device__ __forceinline__ void rb_add(MW_BT (&rb)[MW_REV_BATCH], unsigned& dirty, int top, int lo, int cell, MW_AT v) {
        if (cell >= lo) {
            const int k = top - cell;
#pragma unroll
            for (int q = 0; q < MW_REV_BATCH; ++q) if (q == k) { rb[q] = rb[q] + v; dirty |= 1u << q; }
        } else {
            bars[cell] += v;
        }
    }
    __device__ void reverse() {
        constexpr int BR = MW_REV_BATCH;
        int i = used - 1;
        while (i >= 0) {
            const int lo = i - BR + 1 > 0 ? i - BR + 1 : 0;
            const int nb = i - lo + 1;
#if MW_REV_PF > 0
            if (i - MW_REV_PF >= 4) {
                asm volatile("prefetch.L2 [%0];" :: "l"(nodes + (i - MW_REV_PF)));
                asm volatile("prefetch.L2 [%0];" :: "l"(nodes + (i - MW_REV_PF) - 4));
                asm volatile("prefetch.L2 [%0];" :: "l"(bars + (i - MW_REV_PF)));
            }
#endif
            MW_BT rb[BR]; int nl[BR], nr[BR]; MW_AT nw[BR];
#pragma unroll
            for (int k = 0; k < BR; ++k) {
                if (k < nb) { rb[k] = bars[i - k]; const MynnAdNode n = nodes[i - k]; nl[k] = n.left; nr[k] = n.right; nw[k] = n.weight; }
                else { rb[k] = 0.0; nl[k] = -1; nr[k] = -1; nw[k] = 0.0; }
            }
            unsigned dirty = 0u;
            bool skip = false;
            int next = lo - 1;
#pragma unroll
            for (int k = 0; k < BR; ++k) {
                if (k < nb) {
                    if (skip) { skip = false; }
                    else {
                        const int c = i - k;
                        const MW_AT b = rb[k];
                        if (b != 0.0) {
                            if (nr[k] >= (1 << 29)) {
                                // Fused node (push2): the old sweep's arithmetic and order, see push2 and the old reverse.
                                const int kind = nr[k] >> 29, r = nr[k] & ((1 << 29) - 1);
                                const MW_AT w0 = (k + 1 < nb) ? nw[k + 1 < BR ? k + 1 : BR - 1] : nodes[c - 1].weight;
                                if (kind == 1) {
                                    rb_add(rb, dirty, i, lo, r, nw[k] * b);
                                    rb_add(rb, dirty, i, lo, nl[k], w0 * b);
                                } else {
                                    const MW_AT tq = nw[k] * b;
                                    if (tq != 0.0) {
                                        if (kind == 2) rb_add(rb, dirty, i, lo, nl[k], tq);
                                        rb_add(rb, dirty, i, lo, r, w0 * (-tq));
                                    }
                                }
                                if (k + 1 < nb) skip = true; else next = c - 2;
                            } else {
                                const MW_AT lw = nr[k] >= 0 ? 1.0 : nw[k];
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
    }
#else
    __device__ void reverse() {
        int pc = 0; MW_AT c0 = 0.0, c1 = 0.0;
        for (int i = used - 1; i >= 0; --i) {
#if MW_REV_PF > 0
            // mynn-adjoint-v2: the sweep walks its column's tape backwards; ask for the lines MW_REV_PF slots ahead (64 B
            // of nodes every 4 slots, 64 B of bars every 8) so they are in L2 when the sweep gets there. Hint only.
            if ((i & 3) == 0 && i >= MW_REV_PF) {
                asm volatile("prefetch.L2 [%0];" :: "l"(nodes + (i - MW_REV_PF)));
                if ((i & 7) == 0) asm volatile("prefetch.L2 [%0];" :: "l"(bars + (i - MW_REV_PF)));
            }
#endif
            MW_AT b = bars[i];
            if (pc) { b += c0; if (pc == 2) b += c1; pc = 0; bars[i] = b; }
            if (b == 0.0) continue;
            MynnAdNode n = nodes[i];
#ifdef MW_EXT_REV
            if (n.left == -2) { mw_ext_rev(this, i); continue; }
#endif
            if (n.right >= (1 << 29)) {
                // Fused node (push2). The values and their order per adjoint cell are those of the nodes it replaces:
                // kind 1: the sum node gives both unary nodes bar b exactly (0.0 + 1.0*b, b != 0); each then adds
                //         weight*b to its operand. a != b or equal weights (push2's caller), so the order of the
                //         two cells does not matter.
                // kind 2: t = w1*b is the bar of a - b*w (0.0 + t; t == 0 skips it, as the old reverse skips a zero
                //         bar); a gets 1.0*t, the negated bar -t reaches b*w, so b gets w*(-t). a first, then b.
                // kind 3: as kind 2 with a passive: the negation node holds t, b*w gets -1.0*t, b gets w*(-t).
                const int kind = n.right >> 29, r = n.right & ((1 << 29) - 1);
                const MW_AT w0 = nodes[i - 1].weight;
                if (kind == 1) {
                    bars[r] += n.weight * b;
                    bars[n.left] += w0 * b;
                } else {
                    const MW_AT tq = n.weight * b;
                    if (tq != 0.0) {
                        if (kind == 2) bars[n.left] += tq;
                        bars[r] += w0 * (-tq);
                    }
                }
                --i;
                continue;
            }
            const MW_AT lw = n.right >= 0 ? 1.0 : n.weight;
            if (n.left >= 0) { if (n.left == i - 1) { c0 = lw * b; pc = 1; } else bars[n.left] += lw * b; }
            if (n.right >= 0) {
                const MW_AT v = n.weight * b;
                if (n.right == i - 1) { if (pc) { c1 = v; pc = 2; } else { c0 = v; pc = 1; } } else bars[n.right] += v;
            }
        }
    }
#endif
};

// S10 (2026-11-07): a carrier holds a one-byte reference to its column's context instead of the pointer. Each
// thread of a P3 launch owns exactly one context (mynn_whole_column's ctx), so the reference only says whether
// there is one; the address lives once per thread in shared memory (P3 entries launch 32-thread blocks). The
// carriers shrink from 24 to 16 B (tangent) and from 24 to 8 B (tape), and with them the live carrier state the
// thread stack holds across the operation calls. Mixed contexts (failure slot 2) cannot occur with one context per
// thread, which was already the only case.
__shared__ MynnAdContext* mw_ctxp[32];
struct MynnCtxRef {
    bool on;
    __device__ __forceinline__ static MynnCtxRef of(MynnAdContext* c) {
        if (c) mw_ctxp[threadIdx.x & 31] = c;
        return MynnCtxRef{c != nullptr};
    }
    __device__ __forceinline__ explicit operator bool() const { return on; }
    __device__ __forceinline__ MynnAdContext* operator->() const { return mw_ctxp[threadIdx.x & 31]; }
};
__device__ __forceinline__ MynnCtxRef mynn_ad_context(MynnCtxRef a, MynnCtxRef b) {
    return MynnCtxRef{a.on || b.on};
}

// The context is retained even for a structural zero, so refusal sites can
// still report errors. depends tracks local branch dependence separately from
// the numerical tangent or the presence of a first-order tape node.
struct MynnTangent {
    double v; MynnCtxRef context; bool depends;
    __device__ MynnTangent() : v(0.0), context{false}, depends(false) {}
    __device__ MynnTangent(double x) : v(x), context{false}, depends(false) {}
    __device__ MynnTangent(double x, MynnAdContext* c) : v(x), context(MynnCtxRef::of(c)), depends(c!=nullptr) {}
    __device__ MynnTangent(double x, MynnCtxRef c, bool d) : v(x), context(c), depends(d) {}
};
__device__ __forceinline__ bool dt_zero(MynnTangent x) { return !x.depends; }
__device__ __forceinline__ MynnTangent operator+(MynnTangent a, MynnTangent b) {
    return {a.v+b.v, mynn_ad_context(a.context,b.context), a.depends||b.depends};
}
__device__ __forceinline__ MynnTangent operator-(MynnTangent a) { return {-a.v,a.context,a.depends}; }
__device__ __forceinline__ MynnTangent operator-(MynnTangent a, MynnTangent b) {
    return {a.v-b.v, mynn_ad_context(a.context,b.context), a.depends||b.depends};
}
__device__ __forceinline__ MynnTangent operator*(MynnTangent a, double b) {
    return {a.v*b,a.context,a.depends && b!=0.0};
}
__device__ __forceinline__ MynnTangent operator/(MynnTangent a, double b) {
    return {a.v/b,a.context,a.depends};
}

struct MynnLinear {
    int id; MynnCtxRef context; bool depends;
    __device__ MynnLinear() : id(-1), context{false}, depends(false) {}
    __device__ MynnLinear(double) : id(-1), context{false}, depends(false) {}
    __device__ MynnLinear(int i, MynnAdContext* c) : id(i), context(MynnCtxRef::of(c)), depends(i>=0) {}
    __device__ MynnLinear(int i, MynnCtxRef c, bool d) : id(i), context(c), depends(d) {}
};
__device__ __forceinline__ bool dt_zero(MynnLinear x) { return !x.depends; }
__device__ __noinline__ MynnLinear operator+(MynnLinear a, MynnLinear b) {
    MynnCtxRef c=mynn_ad_context(a.context,b.context); bool d=a.depends||b.depends;
    if(a.id<0)return {b.id,c,d}; if(b.id<0)return {a.id,c,d};
    return {c->push(a.id,1.0,b.id,1.0),c,d};
}
__device__ __noinline__ MynnLinear operator-(MynnLinear a) {
    return a.id<0 ? a : MynnLinear(a.context->push(a.id,-1.0,-1,0.0),a.context,a.depends);
}
__device__ __noinline__ MynnLinear operator-(MynnLinear a, MynnLinear b) {
    MynnCtxRef c=mynn_ad_context(a.context,b.context); bool d=a.depends||b.depends;
    if(b.id<0)return {a.id,c,d}; if(a.id<0){auto z=-b;z.context=c;z.depends=d;return z;}
    return {c->push(a.id,1.0,b.id,-1.0),c,d};
}
__device__ __noinline__ MynnLinear operator*(MynnLinear a, double b) {
    if(b==0.0)return {-1,a.context,false};
    return a.id<0 ? a : MynnLinear(a.context->push(a.id,b,-1,0.0),a.context,a.depends);
}
__device__ __noinline__ MynnLinear operator/(MynnLinear a, double b) { return a*(1.0/b); }

template<class D> __device__ __forceinline__ D mynn_ad_depend(D d, bool b) { d.depends=d.depends||b; return d; }
template<class D> __device__ __forceinline__ D mynn_ad_multiply(D a,double av,D b,double bv) {
    return mynn_ad_depend(a*bv+b*av,a.depends && b.depends);
}
// mynn-adjoint-v2: the tape's product. When both operand nodes would be pushed (a, b active, both weights nonzero)
// and the order of their two contributions cannot matter (distinct cells, or the same cell with equal weights), one
// fused node replaces x = a*bv, y = b*av, x + y; the id, context and depends of the result are those of x + y.
// Every other case is the separate rules, unchanged.
__device__ __noinline__ MynnLinear mynn_ad_multiply(MynnLinear a,double av,MynnLinear b,double bv) {
    if (a.id >= 0 && b.id >= 0 && av != 0.0 && bv != 0.0 && (a.id != b.id || av == bv)) {
        MynnCtxRef c = mynn_ad_context(a.context, b.context);
        return mynn_ad_depend(MynnLinear(c->push2(a.id, bv, b.id, av, 1), c, a.depends || b.depends), a.depends && b.depends);
    }
    return mynn_ad_depend(a*bv+b*av,a.depends && b.depends);
}
// mynn-adjoint-v2: quotient derivative (a - b*w) / den. The tangent (and any carrier but the tape) evaluates exactly
// that expression. On the tape, when every node of it would be pushed, one fused node replaces b*w, the difference
// (or the negation, a passive) and the product by 1/den.
template<class D> __device__ __forceinline__ D mw_quot(D a, D b, double w, double den) { return (a - b * w) / den; }
__device__ __noinline__ MynnLinear mw_quot(MynnLinear a, MynnLinear b, double w, double den) {
    const double w3 = 1.0 / den;
    if (b.id >= 0 && w != 0.0 && w3 != 0.0) {
        MynnCtxRef c = mynn_ad_context(a.context, b.context);
        const bool d = a.depends || b.depends;
        if (a.id >= 0) return MynnLinear(c->push2(a.id, w, b.id, w3, 2), c, d);
        return MynnLinear(c->push2(-1, w, b.id, w3, 3), c, d);
    }
    return (a - b * w) / den;
}
__device__ __forceinline__ void mynn_ad_domain(MynnTangent d) { if(d.depends && d.context)d.context->fail(1); }
__device__ __forceinline__ void mynn_ad_domain(MynnLinear d) { if(d.depends && d.context)d.context->fail(1); }
__device__ __forceinline__ MynnTangent mynn_ad_zero(MynnTangent a,MynnTangent b) {
    return {0.0,mynn_ad_context(a.context,b.context),false};
}
__device__ __forceinline__ MynnLinear mynn_ad_zero(MynnLinear a,MynnLinear b) {
    return {-1,mynn_ad_context(a.context,b.context),false};
}
