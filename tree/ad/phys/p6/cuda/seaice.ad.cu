// SPDX-License-Identifier: BSD-3-Clause
// Hand rules for the three RUC sea-ice driver launch boundaries at hex pin
// 16251589f19aec0012fd164d1ea060126208aed1. The main ice kernel is not here.
// The binder retains the complete pre-launch state until both derivatives end.
// Pointer-table entries and strides follow the pinned oracle wrappers. See the
// corresponding specs for extent, non-alias, launch and failure-slot contracts.

// COMMON-AD section 35: the device has one int32 refusal slot per column.
// The host decodes a nonzero source line to its full refusal record.
using SeaiceFailure = int;

__device__ __forceinline__ bool si_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
__device__ __forceinline__ void si_fail(SeaiceFailure *f, int c,
                                        unsigned long long field,
                                        unsigned long long element,
                                        unsigned long long source) {
    (void)field; (void)element;
    if (!f[c]) f[c] = int(source);
}
__device__ __forceinline__ float *si_f(unsigned long long *p, int k) {
    return reinterpret_cast<float *>(p[k]);
}
__device__ __forceinline__ int *si_i(unsigned long long *p, int k) {
    return reinterpret_cast<int *>(p[k]);
}
__device__ __forceinline__ bool si_check(float x, SeaiceFailure *f, int c,
                                         unsigned long long field,
                                         unsigned long long element,
                                         unsigned long long source) {
    if (si_finite(x)) return true;
    si_fail(f, c, field, element, source); return false;
}
// Each slot is [false count, true count]. Counts use exact integer addition.
__device__ __forceinline__ void si_branch(unsigned int *record, int slot, bool arm) {
    atomicAdd(record + 2 * slot + int(arm), 1u);
}

// Building blocks for the ice kernel. These do not constitute an implementation
// of oracle_ice_ruc_ice: its caller must supply its executed iteration index and
// accumulate table cotangents in a fixed order. Table slots are zero based here.
struct SiQsnPoint { int slot; float weight, dr_dtn; };
__device__ __forceinline__ bool si_qsn_point(float tn, SiQsnPoint &q) {
    if (!si_finite(tn)) return false;
    float r=__fadd_rn(__fdiv_rn(__fsub_rn(tn,173.15f),0.05f),1.f);
    // Conversion outside the representable int range would be invalid.
    if (!si_finite(r) || r<=-2147483648.f || r>=2147483648.f) return false;
    int i=int(r); q.dr_dtn=__fdiv_rn(1.f,0.05f);
    if (i<1) {i=1; r=1.f; q.dr_dtn=0.f;}
    if (i>5000) {i=5000; r=5001.f; q.dr_dtn=0.f;}
    q.slot=i-1; q.weight=__fsub_rn(r,float(i)); return true;
}
__device__ __forceinline__ bool si_qsn_tl(float tn,float dtn,const float *table,
                                          const float *dtable,float &out) {
    SiQsnPoint q; if(!si_qsn_point(tn,q)||!si_finite(dtn))return false;
    float a=table[q.slot],b=table[q.slot+1],da=dtable[q.slot],db=dtable[q.slot+1];
    if(!si_finite(a)||!si_finite(b)||!si_finite(da)||!si_finite(db))return false;
    float dw=__fmul_rn(q.dr_dtn,dtn);
    out=__fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(db,da),q.weight),
                             __fmul_rn(__fsub_rn(b,a),dw)),da);
    return si_finite(out);
}
// The caller consumes bo and adds b0/b1 to its deterministic table workspace.
__device__ __forceinline__ bool si_qsn_ad(float tn,const float *table,float bo,
                                          int &slot,float &btn,float &b0,float &b1) {
    SiQsnPoint q; if(!si_qsn_point(tn,q)||!si_finite(bo))return false;
    float a=table[q.slot],b=table[q.slot+1]; if(!si_finite(a)||!si_finite(b))return false;
    slot=q.slot; float bd=__fmul_rn(bo,q.weight);
    b0=__fsub_rn(bo,bd); b1=bd;
    btn=__fmul_rn(__fmul_rn(bo,__fsub_rn(b,a)),q.dr_dtn);
    return si_finite(btn)&&si_finite(b0)&&si_finite(b1);
}

// Compact per-call form for composing and testing ice_qsn. `slot` is the
// 1-based selected table index saved on the actual forward call; a changed
// index is refused rather than silently extrapolating a captured table pair.
__device__ __forceinline__ bool si_qsn_compact(float tn,float a,float b,int slot,
                                               SiQsnPoint &q) {
    return si_qsn_point(tn,q) && q.slot+1==slot && si_finite(a) && si_finite(b);
}
__device__ __forceinline__ bool si_qsn_compact_tl(float tn,float a,float b,int slot,
                         float dtn,float da,float db,float &dy) {
    SiQsnPoint q;if(!si_qsn_compact(tn,a,b,slot,q)||!si_finite(dtn)||
       !si_finite(da)||!si_finite(db))return false;
    float dw=__fmul_rn(q.dr_dtn,dtn);
    dy=__fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(db,da),q.weight),
                           __fmul_rn(__fsub_rn(b,a),dw)),da);
    return si_finite(dy);
}
__device__ __forceinline__ bool si_qsn_compact_ad(float tn,float a,float b,int slot,
                         float by,float &btn,float &ba,float &bb) {
    SiQsnPoint q;if(!si_qsn_compact(tn,a,b,slot,q)||!si_finite(by))return false;
    float w=__fmul_rn(by,q.weight);ba=__fsub_rn(by,w);bb=w;
    btn=__fmul_rn(__fmul_rn(by,__fsub_rn(b,a)),q.dr_dtn);
    return si_finite(btn)&&si_finite(ba)&&si_finite(bb);
}
#if !defined(SI_COMPILE_ONE) || defined(SI_QSN_AUG)
extern "C" __global__ void seaice_ice_qsn__aug(const float *tn,const float *a,
 const float *b,const int *slot,int n,float *y,SeaiceFailure *fail) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;SiQsnPoint q;
    if(!si_qsn_compact(tn[c],a[c],b[c],slot[c],q)){si_fail(fail,c,1,c,4736);return;}
    float v=__fadd_rn(__fmul_rn(__fsub_rn(b[c],a[c]),q.weight),a[c]);
    if(!si_check(v,fail,c,5,c,4736))return;y[c]=v;
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_QSN_TL)
extern "C" __global__ void seaice_ice_qsn__tl(const float *tn,const float *a,
 const float *b,const int *slot,int n,const float *dtn,const float *da,const float *db,
 float *dy,SeaiceFailure *fail,unsigned int *branches) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float r=__fadd_rn(__fdiv_rn(__fsub_rn(tn[c],173.15f),.05f),1.f);
    if(!si_finite(r)||r<=-2147483648.f||r>=2147483648.f){si_fail(fail,c,1,c,4736);return;}
    int i=int(r);si_branch(branches,0,i>=1);if(i<1)i=1;si_branch(branches,1,i<=5000);
    float v;if(!si_qsn_compact_tl(tn[c],a[c],b[c],slot[c],dtn[c],da[c],db[c],v)){
       si_fail(fail,c,1,c,4736);return;}dy[c]=v;
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_QSN_AD)
extern "C" __global__ void seaice_ice_qsn__ad(const float *tn,const float *a,
 const float *b,const int *slot,int n,float *btn,float *ba,float *bb,
 const float *by,SeaiceFailure *fail) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;float x,y,z;
    if(!si_qsn_compact_ad(tn[c],a[c],b[c],slot[c],by[c],x,y,z)){
       si_fail(fail,c,1,c,4736);return;}
    x=__fadd_rn(btn[c],x);y=__fadd_rn(ba[c],y);z=__fadd_rn(bb[c],z);
    if(!si_check(x,fail,c,1,c,4736)||!si_check(y,fail,c,2,c,4736)||
       !si_check(z,fail,c,3,c,4736))return;btn[c]=x;ba[c]=y;bb[c]=z;
}
#endif

// ice_vilka's continuous output depends on the final integer index only. The
// parent records and validates every executed loop iteration before calling it.
struct SiVilkaPoint { float a,b,t1,f,den,rn,num,pp; };
__device__ __forceinline__ bool si_vilka_point(int i,float d1,float d2,float pp,
                                               const float *table,SiVilkaPoint &q) {
    if(i<1||i>5000||!si_finite(d1)||!si_finite(d2)||!si_finite(pp)||pp==0.f)return false;
    q.a=table[i-1];q.b=table[i];q.pp=pp;
    if(!si_finite(q.a)||!si_finite(q.b))return false;
    q.t1=__fadd_rn(173.1f,__fmul_rn(float(i),.05f));
    q.f=__fsub_rn(__fadd_rn(q.t1,__fmul_rn(d1,q.a)),d2);
    q.den=__fadd_rn(.05f,__fmul_rn(d1,__fsub_rn(q.b,q.a)));
    if(!si_finite(q.den)||q.den==0.f)return false;
    q.rn=__fdiv_rn(q.f,q.den);
    q.num=__fadd_rn(q.a,__fmul_rn(__fsub_rn(q.a,q.b),q.rn));
    return si_finite(q.f)&&si_finite(q.rn)&&si_finite(q.num);
}
__device__ __forceinline__ bool si_vilka_tl(int i,float d1,float d2,float pp,
        const float *table,float dd1,float dd2,float dpp,float da,float db,
        float &dqs,float &dts) {
    SiVilkaPoint q;if(!si_vilka_point(i,d1,d2,pp,table,q))return false;
    if(!si_finite(dd1)||!si_finite(dd2)||!si_finite(dpp)||!si_finite(da)||!si_finite(db))return false;
    float df=__fsub_rn(__fadd_rn(__fmul_rn(dd1,q.a),__fmul_rn(d1,da)),dd2);
    float dden=__fadd_rn(__fmul_rn(dd1,__fsub_rn(q.b,q.a)),__fmul_rn(d1,__fsub_rn(db,da)));
    float drn=__fdiv_rn(__fsub_rn(df,__fmul_rn(q.rn,dden)),q.den);
    float dnum=__fadd_rn(da,__fadd_rn(__fmul_rn(__fsub_rn(da,db),q.rn),__fmul_rn(__fsub_rn(q.a,q.b),drn)));
    dts=__fmul_rn(-.05f,drn);
    dqs=__fdiv_rn(__fsub_rn(dnum,__fmul_rn(__fdiv_rn(q.num,pp),dpp)),pp);
    return si_finite(dts)&&si_finite(dqs);
}
__device__ __forceinline__ bool si_vilka_ad(int i,float d1,float d2,float pp,
        const float *table,float bqs,float bts,float &bd1,float &bd2,float &bpp,
        float &ba,float &bb) {
    SiVilkaPoint q;if(!si_vilka_point(i,d1,d2,pp,table,q)||!si_finite(bqs)||!si_finite(bts))return false;
    float bnum=__fdiv_rn(bqs,pp);
    bpp=__fmul_rn(-bnum,__fdiv_rn(q.num,pp));
    float brn=__fadd_rn(__fmul_rn(-.05f,bts),__fmul_rn(bnum,__fsub_rn(q.a,q.b)));
    ba=__fadd_rn(bnum,__fmul_rn(bnum,q.rn)); bb=__fmul_rn(-bnum,q.rn);
    float bf=__fdiv_rn(brn,q.den), bden=__fmul_rn(-bf,q.rn);
    bd1=__fadd_rn(__fmul_rn(bf,q.a),__fmul_rn(bden,__fsub_rn(q.b,q.a)));
    bd2=-bf;
    ba=__fadd_rn(ba,__fsub_rn(__fmul_rn(bf,d1),__fmul_rn(bden,d1)));
    bb=__fadd_rn(bb,__fmul_rn(bden,d1));
    return si_finite(bd1)&&si_finite(bd2)&&si_finite(bpp)&&si_finite(ba)&&si_finite(bb);
}

// One layer of ice_icetmp's material-property update (pinned lines 3839-44).
// The caller owns the surrounding executed nzs loop and reverse ordering.
struct SiIceMaterial { float temp,den,rho,heat,capacity,diffusivity; };
__device__ __forceinline__ bool si_ice_material(float ts,SiIceMaterial &v) {
    if(!si_finite(ts))return false;
    v.temp=__fsub_rn(ts,273.15f);
    v.den=__fsub_rn(1.f,__fmul_rn(0.000165f,v.temp));
    if(!si_finite(v.den)||v.den==0.f)return false;
    v.rho=__fdiv_rn(917.6f,v.den);
    v.heat=__fadd_rn(2115.85f,__fmul_rn(7.7948f,v.temp));
    v.capacity=__fmul_rn(v.heat,v.rho);
    if(!si_finite(v.capacity)||v.capacity==0.f)return false;
    v.diffusivity=__fdiv_rn(2.260872f,v.capacity);
    return si_finite(v.rho)&&si_finite(v.heat)&&si_finite(v.diffusivity);
}
__device__ __forceinline__ bool si_ice_material_tl(float ts,float dts,
        float &dtemp,float &drho,float &dcapacity,float &ddiffusivity) {
    SiIceMaterial v;if(!si_ice_material(ts,v)||!si_finite(dts))return false;
    dtemp=dts;
    float dden=__fmul_rn(-0.000165f,dts);
    drho=__fdiv_rn(__fmul_rn(-v.rho,dden),v.den);
    float dheat=__fmul_rn(7.7948f,dts);
    dcapacity=__fadd_rn(__fmul_rn(dheat,v.rho),__fmul_rn(v.heat,drho));
    ddiffusivity=__fdiv_rn(__fmul_rn(-v.diffusivity,dcapacity),v.capacity);
    return si_finite(drho)&&si_finite(dcapacity)&&si_finite(ddiffusivity);
}
__device__ __forceinline__ bool si_ice_material_ad(float ts,float btemp,float brho,
        float bcapacity,float bdiffusivity,float &bts) {
    SiIceMaterial v;if(!si_ice_material(ts,v)||!si_finite(btemp)||!si_finite(brho)||
        !si_finite(bcapacity)||!si_finite(bdiffusivity))return false;
    float bc=__fsub_rn(bcapacity,__fdiv_rn(__fmul_rn(bdiffusivity,v.diffusivity),v.capacity));
    float br=__fadd_rn(brho,__fmul_rn(bc,v.heat));
    float bh=__fmul_rn(bc,v.rho);
    float bd=__fdiv_rn(__fmul_rn(-br,v.rho),v.den);
    bts=__fadd_rn(__fadd_rn(btemp,__fmul_rn(7.7948f,bh)),__fmul_rn(-0.000165f,bd));
    return si_finite(bts);
}

// The threshold is a predicate only: its executed-arm derivative is zero.
#if !defined(SI_COMPILE_ONE) || defined(SI_PREP_TL)
extern "C" __global__ void oracle_drivers_seaice_prepare__tl(
    unsigned long long *p, unsigned long long *d, int nc, SeaiceFailure *fail,
    unsigned int *branches) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int n = si_i(p, 0)[c];
    const float threshold = si_f(p, 1)[c];
    if (!si_check(threshold, fail, c, 2, c, 5007)) return;
    for (int i = 0; i < n; ++i) {
        const int k = c + i * nc;
        const float x = si_f(p, 2)[k], a = si_f(p, 3)[k], e = si_f(p, 4)[k];
        const float dx = si_f(d, 2)[k], da = si_f(d, 3)[k], de = si_f(d, 4)[k];
        if (!si_check(x, fail, c, 3, k, 5007) || !si_check(a, fail, c, 4, k, 5008) ||
            !si_check(e, fail, c, 5, k, 5009) || !si_check(dx, fail, c, 3, k, 5007) ||
            !si_check(da, fail, c, 4, k, 5008) || !si_check(de, fail, c, 5, k, 5009)) return;
        float va = da, ve = de;
        const bool left = x >= threshold, right = left && x < 1.0f;
        si_branch(branches,0,right); si_branch(branches,1,left);
        if (left) si_branch(branches,2,x < 1.0f);
        if (right) {
            if (x == 0.0f) { si_fail(fail, c, 3, k, 5008); return; }
            const float inv = __fdiv_rn(1.0f, x);
            const float na = __fsub_rn(a, __fmul_rn(0.08f, __fsub_rn(1.0f, x)));
            const float ne = __fsub_rn(e, __fmul_rn(0.98f, __fsub_rn(1.0f, x)));
            va = __fsub_rn(__fmul_rn(__fadd_rn(da, __fmul_rn(0.08f, dx)), inv),
                           __fmul_rn(__fmul_rn(__fdiv_rn(na, x), inv), dx));
            ve = __fsub_rn(__fmul_rn(__fadd_rn(de, __fmul_rn(0.98f, dx)), inv),
                           __fmul_rn(__fmul_rn(__fdiv_rn(ne, x), inv), dx));
        }
        if (!si_check(va, fail, c, 6, k, 5008) || !si_check(ve, fail, c, 7, k, 5009)) return;
        si_f(d, 5)[k] = va; si_f(d, 6)[k] = ve;
    }
}
#endif

#if !defined(SI_COMPILE_ONE) || defined(SI_PREP_AD)
extern "C" __global__ void oracle_drivers_seaice_prepare__ad(
    unsigned long long *p, unsigned long long *b, int nc, SeaiceFailure *fail) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int n = si_i(p, 0)[c];
    const float threshold = si_f(p, 1)[c];
    if (!si_check(threshold, fail, c, 2, c, 5007)) return;
    for (int i = n - 1; i >= 0; --i) {
        const int k = c + i * nc;
        const float x = si_f(p, 2)[k], a = si_f(p, 3)[k], e = si_f(p, 4)[k];
        const float ba = si_f(b, 5)[k], be = si_f(b, 6)[k];
        if (!si_check(x, fail, c, 3, k, 5007) || !si_check(a, fail, c, 4, k, 5008) ||
            !si_check(e, fail, c, 5, k, 5009) || !si_check(ba, fail, c, 6, k, 5008) ||
            !si_check(be, fail, c, 7, k, 5009)) return;
        float ax = 0.f, aa = ba, ae = be;
        if (x >= threshold && x < 1.0f) {
            if (x == 0.0f) { si_fail(fail, c, 3, k, 5008); return; }
            const float inv = __fdiv_rn(1.f, x);
            const float na = __fsub_rn(a, __fmul_rn(0.08f, __fsub_rn(1.f, x)));
            const float ne = __fsub_rn(e, __fmul_rn(0.98f, __fsub_rn(1.f, x)));
            aa = __fmul_rn(ba, inv); ae = __fmul_rn(be, inv);
            ax = __fadd_rn(__fmul_rn(ba, __fsub_rn(__fmul_rn(0.08f, inv), __fmul_rn(__fdiv_rn(na, x), inv))),
                           __fmul_rn(be, __fsub_rn(__fmul_rn(0.98f, inv), __fmul_rn(__fdiv_rn(ne, x), inv))));
        }
        if (!si_check(ax, fail, c, 3, k, 5008) || !si_check(aa, fail, c, 4, k, 5008) ||
            !si_check(ae, fail, c, 5, k, 5009)) return;
        float rx=__fadd_rn(si_f(b,2)[k],ax),ra=__fadd_rn(si_f(b,3)[k],aa);
        float re=__fadd_rn(si_f(b,4)[k],ae);
        if(!si_check(rx,fail,c,3,k,5008)||!si_check(ra,fail,c,4,k,5008)||
           !si_check(re,fail,c,5,k,5009))return;
        si_f(b, 5)[k] = 0.f; si_f(b, 6)[k] = 0.f;
        si_f(b, 2)[k]=rx;si_f(b, 3)[k]=ra;si_f(b, 4)[k]=re;
    }
}
#endif

__device__ __forceinline__ bool si_mix_tl(unsigned long long *p, unsigned long long *d,
                                          int pi, int si, int k, float x, float dx,
                                          SeaiceFailure *f, int c) {
    float a=si_f(p,pi)[k], s=si_f(p,si)[k], da=si_f(d,pi)[k], ds=si_f(d,si)[k];
    if (!si_check(a,f,c,pi+1,k,5028) || !si_check(s,f,c,si+1,k,5028) ||
        !si_check(da,f,c,pi+1,k,5028) || !si_check(ds,f,c,si+1,k,5028)) return false;
    float z=__fadd_rn(__fadd_rn(__fmul_rn(x,da),__fmul_rn(__fsub_rn(1.f,x),ds)),
                      __fmul_rn(dx,__fsub_rn(a,s)));
    if (!si_check(z,f,c,pi+1,k,5028)) return false;
    si_f(d,pi)[k]=z; return true;
}
__device__ __forceinline__ bool si_constmix_tl(unsigned long long *p, unsigned long long *d,
                                               int pi,int k,float x,float dx,float s,
                                               SeaiceFailure *f,int c) {
    float a=si_f(p,pi)[k], da=si_f(d,pi)[k];
    if (!si_check(a,f,c,pi+1,k,5046) || !si_check(da,f,c,pi+1,k,5046)) return false;
    float z=__fadd_rn(__fmul_rn(x,da),__fmul_rn(dx,__fsub_rn(a,s)));
    if (!si_check(z,f,c,pi+1,k,5046)) return false;
    si_f(d,pi)[k]=z; return true;
}
#if !defined(SI_COMPILE_ONE) || defined(SI_FINISH_TL)
extern "C" __global__ void oracle_drivers_seaice_finish__tl(
    unsigned long long *p, unsigned long long *d, int nc, SeaiceFailure *fail,
    unsigned int *branches) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x; if (c>=nc) return;
    const int n=si_i(p,0)[c], config=si_i(p,2)[c], mynn=si_i(p,3)[c];
    if (!config) return;
    const float threshold=si_f(p,1)[c];
    if (!si_check(threshold,fail,c,2,c,5026)) return;
    for(int i=0;i<n;++i) {
        int k=c+i*nc; float x=si_f(p,4)[k];
        if (!si_check(x,fail,c,5,k,5026)) return;
        const bool left=x>=threshold, right=left && x<1.f;
        si_branch(branches,0,right); si_branch(branches,1,left);
        if(left) si_branch(branches,2,x<1.f);
        if (!right) continue;
        float dx=si_f(d,4)[k]; if (!si_check(dx,fail,c,5,k,5026)) return;
        if (!si_mix_tl(p,d,5,6,k,x,dx,fail,c) || !si_mix_tl(p,d,7,8,k,x,dx,fail,c)) return;
        if (mynn) {
            if (!si_mix_tl(p,d,9,10,k,x,dx,fail,c)) return;
        } else {
            // The original cqs value is overwritten by a blend of the OLD chs.
            float a=si_f(p,13)[k], s=si_f(p,14)[k], da=si_f(d,13)[k], ds=si_f(d,14)[k];
            if (!si_check(a,fail,c,14,k,5033) || !si_check(s,fail,c,15,k,5033) ||
                !si_check(da,fail,c,14,k,5033) || !si_check(ds,fail,c,15,k,5033)) return;
            float z=__fadd_rn(__fadd_rn(__fmul_rn(x,da),__fmul_rn(__fsub_rn(1.f,x),ds)),
                              __fmul_rn(dx,__fsub_rn(a,s)));
            if (!si_check(z,fail,c,10,k,5033)) return;
            si_f(d,9)[k]=z;
        }
        for (int j=11;j<=29;j+=2) if (!si_mix_tl(p,d,j,j+1,k,x,dx,fail,c)) return;
        if (!si_constmix_tl(p,d,31,k,x,dx,0.08f,fail,c) ||
            !si_constmix_tl(p,d,32,k,x,dx,0.98f,fail,c)) return;
    }
}
#endif
__device__ __forceinline__ bool si_mix_ad(unsigned long long *p,unsigned long long *b,
                                          int pi,int si,int k,float x,float &bx,
                                          SeaiceFailure *f,int c) {
    float a=si_f(p,pi)[k],s=si_f(p,si)[k],v=si_f(b,pi)[k];
    if (!si_check(a,f,c,pi+1,k,5028) || !si_check(s,f,c,si+1,k,5028) ||
        !si_check(v,f,c,pi+1,k,5028)) return false;
    float xx=__fadd_rn(bx,__fmul_rn(v,__fsub_rn(a,s)));
    float pp=__fmul_rn(x,v);
    float ss=__fadd_rn(si_f(b,si)[k],__fmul_rn(__fsub_rn(1.f,x),v));
    if(!si_check(xx,f,c,5,k,5028)||!si_check(pp,f,c,pi+1,k,5028)||
       !si_check(ss,f,c,si+1,k,5028))return false;
    bx=xx;si_f(b,pi)[k]=pp;si_f(b,si)[k]=ss;
    return true;
}
__device__ __forceinline__ bool si_constmix_ad(unsigned long long *p,unsigned long long *b,
                                               int pi,int k,float x,float &bx,float s,
                                               SeaiceFailure *f,int c) {
    float a=si_f(p,pi)[k],v=si_f(b,pi)[k];
    if (!si_check(a,f,c,pi+1,k,5046) || !si_check(v,f,c,pi+1,k,5046)) return false;
    float xx=__fadd_rn(bx,__fmul_rn(v,__fsub_rn(a,s))),pp=__fmul_rn(x,v);
    if(!si_check(xx,f,c,5,k,5046)||!si_check(pp,f,c,pi+1,k,5046))return false;
    bx=xx;si_f(b,pi)[k]=pp;return true;
}
#if !defined(SI_COMPILE_ONE) || defined(SI_FINISH_AD)
extern "C" __global__ void oracle_drivers_seaice_finish__ad(
    unsigned long long *p,unsigned long long *b,int nc,SeaiceFailure *fail) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
    const int n=si_i(p,0)[c],config=si_i(p,2)[c],mynn=si_i(p,3)[c];
    if(!config)return;
    const float threshold=si_f(p,1)[c]; if(!si_check(threshold,fail,c,2,c,5026))return;
    for(int i=n-1;i>=0;--i) {
        int k=c+i*nc; float x=si_f(p,4)[k]; if(!si_check(x,fail,c,5,k,5026))return;
        if(!(x>=threshold && x<1.f))continue;
        float bx=0.f;
        if(!si_constmix_ad(p,b,32,k,x,bx,0.98f,fail,c) ||
           !si_constmix_ad(p,b,31,k,x,bx,0.08f,fail,c))return;
        for(int j=29;j>=11;j-=2) if(!si_mix_ad(p,b,j,j+1,k,x,bx,fail,c))return;
        if(mynn) {
            if(!si_mix_ad(p,b,9,10,k,x,bx,fail,c))return;
        } else {
            float a=si_f(p,13)[k],s=si_f(p,14)[k],v=si_f(b,9)[k];
            if(!si_check(a,fail,c,14,k,5033) || !si_check(s,fail,c,15,k,5033) ||
               !si_check(v,fail,c,10,k,5033))return;
            float xx=__fadd_rn(bx,__fmul_rn(v,__fsub_rn(a,s)));
            float pp=__fadd_rn(si_f(b,13)[k],__fmul_rn(x,v));
            float ss=__fadd_rn(si_f(b,14)[k],__fmul_rn(__fsub_rn(1.f,x),v));
            if(!si_check(xx,fail,c,5,k,5033)||!si_check(pp,fail,c,14,k,5033)||
               !si_check(ss,fail,c,15,k,5033))return;
            bx=xx;si_f(b,9)[k]=0.f;si_f(b,13)[k]=pp;si_f(b,14)[k]=ss;
        }
        if(!si_mix_ad(p,b,7,8,k,x,bx,fail,c) || !si_mix_ad(p,b,5,6,k,x,bx,fail,c))return;
        if(!si_check(bx,fail,c,5,k,5026))return;
        float accumulated=__fadd_rn(si_f(b,4)[k],bx);
        if(!si_check(accumulated,fail,c,5,k,5026))return;
        si_f(b,4)[k]=accumulated;
    }
}
#endif

#if !defined(SI_COMPILE_ONE) || defined(SI_CORRECT_TL)
extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice__tl(
    unsigned long long *p,unsigned long long *d,int nc,SeaiceFailure *fail,
    unsigned int *branches) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
    int ims=si_i(p,0)[c],ime=si_i(p,1)[c],jms=si_i(p,2)[c];
    int its=si_i(p,4)[c],ite=si_i(p,5)[c],jts=si_i(p,6)[c],jte=si_i(p,7)[c];
    float t=si_f(p,8)[c]; if(!si_check(t,fail,c,9,c,5064))return;
    for(int j=jts;j<=jte;++j) for(int i=its;i<=ite;++i) {
        int k=c+((j-jms)*(ime-ims+1)+(i-ims))*nc;
        float x=si_f(p,9)[k],v=si_f(p,10)[k],dv=si_f(d,10)[k];
        if(!si_check(x,fail,c,10,k,5064)||!si_check(v,fail,c,11,k,5065)||
           !si_check(dv,fail,c,11,k,5065))return;
        bool sea=true,ice=true;
        const bool left=x>=t, right=left && x<=1.f;
        si_branch(branches,0,right); si_branch(branches,1,left);
        if(left) si_branch(branches,2,x<=1.f);
        if(right) {
            sea=(v>271.4f); // ruc_max(a,b) selects b when a == b
            si_branch(branches,3,sea);
            bool a=x<0.2f, b=a && v<253.15f;
            si_branch(branches,4,b); si_branch(branches,5,a);
            if(a) si_branch(branches,6,v<253.15f);
            float vi=b?253.15f:v;
            bool d1=x<0.1f, d2=d1 && vi<263.15f;
            si_branch(branches,7,d2); si_branch(branches,8,d1);
            if(d1) si_branch(branches,9,vi<263.15f);
            if(b || d2) ice=false;
        }
        si_f(d,11)[k]=sea?dv:0.f; si_f(d,12)[k]=ice?dv:0.f;
    }
}
#endif
#if !defined(SI_COMPILE_ONE) || defined(SI_CORRECT_AD)
extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice__ad(
    unsigned long long *p,unsigned long long *b,int nc,SeaiceFailure *fail) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
    int ims=si_i(p,0)[c],ime=si_i(p,1)[c],jms=si_i(p,2)[c];
    int its=si_i(p,4)[c],ite=si_i(p,5)[c],jts=si_i(p,6)[c],jte=si_i(p,7)[c];
    float t=si_f(p,8)[c]; if(!si_check(t,fail,c,9,c,5064))return;
    for(int j=jte;j>=jts;--j) for(int i=ite;i>=its;--i) {
        int k=c+((j-jms)*(ime-ims+1)+(i-ims))*nc;
        float x=si_f(p,9)[k],v=si_f(p,10)[k],bs=si_f(b,11)[k],bi=si_f(b,12)[k];
        if(!si_check(x,fail,c,10,k,5064)||!si_check(v,fail,c,11,k,5065)||
           !si_check(bs,fail,c,12,k,5065)||!si_check(bi,fail,c,13,k,5066))return;
        bool sea=true,ice=true;
        if(x>=t && x<=1.f) {
            sea=(v>271.4f);
            if(x<0.2f && v<253.15f)ice=false;
            if(x<0.1f && v<263.15f)ice=false;
        }
        float r=__fadd_rn(sea?bs:0.f,ice?bi:0.f);
        float accumulated=__fadd_rn(si_f(b,10)[k],r);
        if(!si_check(accumulated,fail,c,11,k,5065))return;
        si_f(b,11)[k]=0.f;si_f(b,12)[k]=0.f;si_f(b,10)[k]=accumulated;
    }
}
#endif
