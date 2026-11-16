// SPDX-License-Identifier: Apache-2.0
// Branch-only primal recomputation, derived from pinned cuda_horizontal_v841.py
// and cuda_fp32.py. It writes only integer branch counts, never primal outputs.
// The ordinary captured forward image is not altered or used from this header.
#ifndef HEX_AD_HORIZONTAL_V841_BRANCHES_CUH
#define HEX_AD_HORIZONTAL_V841_BRANCHES_CUH
__device__ __forceinline__ void ad_branch_bump(unsigned *r,int s,bool v) {
    if (r) atomicAdd(&r[2*s+(v?1:0)],1u);
}
__device__ __forceinline__ float ad_probe_ftz(float x) {
    unsigned b=__float_as_uint(x); return (b&0x7f800000u)==0u ? __uint_as_float(b&0x80000000u) : x;
}
// The fast add/sub arm returns an ordinary FTZ operation. On this arm only,
// an operand that the hardware flushes has local derivative zero.
__device__ __forceinline__ bool ad_probe_add_operand_live(float a,float b,bool sub,bool left) {
    float x=ad_probe_ftz(a),y=ad_probe_ftz(b);
    float z=ad_probe_ftz(sub ? __fsub_rn(x,y) : __fadd_rn(x,y));
    unsigned bits=__float_as_uint(left?a:b)&0x7fffffffu;
    bool subnormal=(bits!=0u && (bits&0x7f800000u)==0u);
    return !(((__float_as_uint(z)&0x7fffffffu)>>23)>24u && subnormal);
}
// Jacobian weights of the executed multiply arm. If the helper takes its
// ordinary FTZ arm, it flushes subnormal operands before the multiplication;
// if it takes the nonzero fallback, both operands are decoded exactly.
__device__ __forceinline__ float ad_probe_mul_weight(float a,float b,bool left) {
    unsigned am=__float_as_uint(a)&0x7fffffffu,bm=__float_as_uint(b)&0x7fffffffu;
    float ordinary=ad_probe_ftz(__fmul_rn(ad_probe_ftz(a),ad_probe_ftz(b)));
    bool fallback=((__float_as_uint(ordinary)&0x7f800000u)==0u)&&am!=0u&&bm!=0u;
    unsigned active=left?am:bm;
    if(!fallback && active!=0u && (active&0x7f800000u)==0u)return 0.0f;
    float other=left?b:a;return fallback?other:ad_probe_ftz(other);
}
__device__ __forceinline__ double ad_probe_decode(unsigned *r,int base,float x) {
    unsigned b=__float_as_uint(x), m=b&0x7fffffffu;
    bool q=m!=0u && (m&0x7f800000u)==0u; ad_branch_bump(r,base+10,q);
    if(q) { double d=ldexp((double)(m&0x007fffffu),-149); return (b&0x80000000u)?-d:d; }
    return (double)x;
}
__device__ __forceinline__ float ad_probe_pack(unsigned *r,int base,double x) {
    double m=fabs(x); bool q=m>0.0 && m<1.1754943508222875e-38; ad_branch_bump(r,base+11,q);
    if(q) { unsigned u=(unsigned)rint(m*7.1362384635297994e+44);
        return __uint_as_float((__double_as_longlong(x)<0LL ? 0x80000000u:0u)|u); }
    return __double2float_rn(x);
}
__device__ __forceinline__ float ad_probe_addsub(unsigned *r,int base,float a,float b,bool sub) {
    float ordinary=ad_probe_ftz(sub ? __fsub_rn(ad_probe_ftz(a),ad_probe_ftz(b)) : __fadd_rn(ad_probe_ftz(a),ad_probe_ftz(b)));
    unsigned z=__float_as_uint(ordinary)&0x7fffffffu, ze=z>>23; int slot=base+(sub?3:0);
    bool q=ze>24u; ad_branch_bump(r,slot,q); if(q)return ordinary;
    unsigned x=__float_as_uint(a)&0x7fffffffu,y=__float_as_uint(b)&0x7fffffffu,xe=x>>23,ye=y>>23;
    q=(xe==0u&&x!=0u)||(ye==0u&&y!=0u);ad_branch_bump(r,slot+1,q);
    if(q) { double da=ad_probe_decode(r,base,a),db=ad_probe_decode(r,base,b);return ad_probe_pack(r,base,sub?da-db:da+db); }
    q=xe<=23u&&ye<=23u&&ze==0u&&x!=y&&x!=0u&&y!=0u;ad_branch_bump(r,slot+2,q);
    if(q) { double da=ad_probe_decode(r,base,a),db=ad_probe_decode(r,base,b);return ad_probe_pack(r,base,sub?da-db:da+db); }
    return ordinary;
}
__device__ __forceinline__ float ad_probe_mul(unsigned *r,int base,float a,float b) {
    float ordinary=ad_probe_ftz(__fmul_rn(ad_probe_ftz(a),ad_probe_ftz(b)));
    bool q=((__float_as_uint(ordinary)&0x7fffffffu)&0x7f800000u)==0u;ad_branch_bump(r,base+6,q);
    if(q) {unsigned x=__float_as_uint(a)&0x7fffffffu,y=__float_as_uint(b)&0x7fffffffu;
        q=x!=0u&&y!=0u;ad_branch_bump(r,base+7,q);
        if(q){double da=ad_probe_decode(r,base,a),db=ad_probe_decode(r,base,b);return ad_probe_pack(r,base,da*db);}}
    return ordinary;
}
__device__ __forceinline__ float ad_probe_div(unsigned *r,int base,float a,float b) {
    unsigned x=__float_as_uint(a)&0x7fffffffu,y=__float_as_uint(b)&0x7fffffffu,xe=x>>23,ye=y>>23;
    bool q=(xe==0u&&x!=0u)||(ye==0u&&y!=0u);ad_branch_bump(r,base+8,q);
    if(q) {double da=ad_probe_decode(r,base,a),db=ad_probe_decode(r,base,b);return ad_probe_pack(r,base,da/db);}
    float ordinary=ad_probe_ftz(__fdiv_rn(ad_probe_ftz(a),ad_probe_ftz(b)));
    q=x!=0u&&y!=0u&&xe+126u<=ye&&((__float_as_uint(ordinary)&0x7fffffffu)&0x7f800000u)==0u;
    ad_branch_bump(r,base+9,q);
    if(q){double da=ad_probe_decode(r,base,a),db=ad_probe_decode(r,base,b);return ad_probe_pack(r,base,da/db);}
    return ordinary;
}
#define mpas_add(a,b) ad_probe_addsub(branch_record,probe_base,(a),(b),false)
#define mpas_sub(a,b) ad_probe_addsub(branch_record,probe_base,(a),(b),true)
#define mpas_mul(a,b) ad_probe_mul(branch_record,probe_base,(a),(b))
#define mpas_div(a,b) ad_probe_div(branch_record,probe_base,(a),(b))
#define HORIZONTAL_ELEMENT_LOOP(total) for (int element = blockDim.x*blockIdx.x+threadIdx.x; element < (total); element += gridDim.x*blockDim.x)
#ifndef HEX_HOIST_EOE
#define HEX_HOIST_EOE 10
#endif
__device__ __forceinline__ int lidx_v841(int l,int i,int n){return l*n+i;}
__device__ __forceinline__ float cell_sign_v841(int c,int e,const int *coe){return coe[2*e]==c?1.0f:-1.0f;}
__device__ __forceinline__ float vertex_sign_v841(int v,int e,const int *voe){return voe[2*e]==v?-1.0f:1.0f;}
__device__ __forceinline__ int kite_slot_v841(int v,int c,const int *cov,int d){for(int s=0;s<d;++s)if(cov[v*d+s]==c)return s;return -1;}

__device__ __forceinline__ void ad_probe_cell_diagnostics_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const float *ke_vertex, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_cell, const int *cells_on_vertex,
    const float *dv_edge, const float *inv_area_cell,
    const float *kite_areas, const int nlev, const int ncells,
    const int nedges, const int nvertices, const int max_edges,
    const int vertex_degree, float *divergence, float *kinetic_energy, unsigned *branch_record)
{
    const int probe_base=4;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const float inv_area = inv_area_cell[cell];
        const float ke_fact = 0.625f;
        float div = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            div = mpas_add(div, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
        for (int slot = 0; slot < count; ++slot) {
            const int vertex = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                    kite_areas[vertex * vertex_degree + kite]);
                correction = mpas_mul(correction,
                    ke_vertex[lidx_v841(k, vertex, nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        (void)(div);
        (void)(ke);
    }
}

__device__ __forceinline__ void ad_probe_laplacian_divergence_v841_f32(
    const float* delsq_u,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dv_edge,
    const float* area_cell,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* delsq_divergence
, unsigned *branch_record) {
    const int probe_base=2;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(mpas_div(1.0f, area_cell[cell]), dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), delsq_u[lidx_v841(k, edge, nedges)]));
        }
        (void)(value);
    }
}

__device__ __forceinline__ void ad_probe_laplacian_vorticity_v841_f32(
    const float* delsq_u,
    const int* edges_on_vertex,
    const int* vertices_on_edge,
    const float* dc_edge,
    const float* area_triangle,
    const int nlev,
    const int nedges,
    const int nvertices,
    const int vertex_degree,
    float* delsq_vorticity
, unsigned *branch_record) {
    const int probe_base=4;
    const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex >= nvertices) return;
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(mpas_div(1.0f, area_triangle[vertex]), dc_edge[edge]),
                vertex_sign_v841(vertex, edge, vertices_on_edge)), delsq_u[lidx_v841(k, edge, nedges)]));
        }
        (void)(value);
    }
}

__device__ __forceinline__ void ad_probe_tangential_velocity_hoist_v841_f32(
    const float* normal_velocity,
    const int* edges_on_edge,
    const int* n_edges_on_edge,
    const float* weights_on_edge,
    const int nlev,
    const int nedges,
    const int max_edges2,
    float* tangential_velocity, const int lpt, unsigned *branch_record)
{
    const int probe_base=6;
    const int hex_rows = nlev;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (hex_element >= nedges * hex_chunks) return;
    const int edge = hex_element % nedges;
    const int hex_k0 = (hex_element / nedges) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int count = n_edges_on_edge[edge];
    if (count > HEX_HOIST_EOE) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            float value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int offset = edge * max_edges2 + slot;
                const int neighbor = edges_on_edge[offset];
                value = mpas_add(value, mpas_mul(weights_on_edge[offset],
                    normal_velocity[lidx_v841(k, neighbor, nedges)]));
            }
            (void)(value);
        }
        return;
    }
    int nb[HEX_HOIST_EOE];
    float wt[HEX_HOIST_EOE];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
        nb[slot] = slot < count ? edges_on_edge[edge * max_edges2 + slot] : edge;
        wt[slot] = slot < count ? weights_on_edge[edge * max_edges2 + slot] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        float value = 0.0f;
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_EOE; ++slot) {
            if (slot < count) {
                value = mpas_add(value, mpas_mul(wt[slot],
                    normal_velocity[lidx_v841(k, nb[slot], nedges)]));
            }
        }
        (void)(value);
    }
}

__device__ __forceinline__ void ad_probe_theta_filter_lap2_v841_f32(
    const float* theta_m,
    const float* rho_edge,
    const float* kdiff,
    const float* del2,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float* area_cell,
    const int nlev,
    const int ncells,
    const int nedges,
    const int max_edges,
    float* delsq_theta,
    float* tendency
, unsigned *branch_record) {
    const int probe_base=2;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_div(1.0f, area_cell[cell]),
                cell_sign_v841(cell, edge, cells_on_edge)), dv_edge[edge]),
                mpas_div(1.0f, dc_edge[edge]));
            const float flux = mpas_mul(mpas_mul(factor, mpas_sub(
                theta_m[lidx_v841(k, cell1, ncells)],
                theta_m[lidx_v841(k, cell0, ncells)])),
                rho_edge[lidx_v841(k, edge, nedges)]);
            lap = mpas_add(lap, flux);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, 0.5f), mpas_add(
                    kdiff[lidx_v841(k, cell0, ncells)],
                    kdiff[lidx_v841(k, cell1, ncells)])), del2[edge]));
        }
        const int tid = lidx_v841(k, cell, ncells);
        (void)(lap);
        (void)(tend);
    }
}

__device__ __forceinline__ void ad_probe_theta_filter_lap4_v841_f32(
    const float* delsq_theta,
    const float* del4,
    const int* edges_on_cell,
    const int* n_edges_on_cell,
    const int* cells_on_edge,
    const float* dc_edge,
    const float* dv_edge,
    const float* area_cell,
    const float h4,
    const int nlev,
    const int ncells,
    const int max_edges,
    float* tendency
, unsigned *branch_record) {
    const int probe_base=2;
    HORIZONTAL_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx_v841(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_mul(mpas_mul(mpas_mul(mpas_mul(del4[edge],
                mpas_mul(h4, mpas_div(1.0f, area_cell[cell]))), dv_edge[edge]),
                cell_sign_v841(cell, edge, cells_on_edge)), mpas_div(1.0f, dc_edge[edge]));
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_theta[lidx_v841(k, cell1, ncells)],
                delsq_theta[lidx_v841(k, cell0, ncells)])));
        }
        (void)(tend);
    }
}

__device__ __forceinline__ void ad_probe_vertex_diagnostics_v841_f32(
    const float *normal_velocity, const float *ke_edge,
    const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *inv_area_triangle,
    const float *f_vertex, const int nlev, const int nedges,
    const int nvertices, const int vertex_degree,
    float *vorticity, float *ke_vertex, float *pv_vertex, unsigned *branch_record)
{
    const int probe_base=3;
    HORIZONTAL_ELEMENT_LOOP(nlev * nvertices) {
        const int vertex = element % nvertices;
        const int k = element / nvertices;
        float vort = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0) continue;
            vort = mpas_add(vort, mpas_mul(mpas_mul(
                vertex_sign_v841(vertex, edge, vertices_on_edge), dc_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, ke_edge[lidx_v841(k, edge, nedges)]);
        }
        vort = mpas_mul(vort, inv_area_triangle[vertex]);
        const float r = mpas_mul(0.25f, inv_area_triangle[vertex]);
        ke = mpas_mul(ke, r);
        const int index = lidx_v841(k, vertex, nvertices);
        (void)(vort);
        (void)(ke);
        (void)(mpas_add(f_vertex[vertex], vort));
    }
}

#undef mpas_add
#undef mpas_sub
#undef mpas_mul
#undef mpas_div
#undef HORIZONTAL_ELEMENT_LOOP
#endif
