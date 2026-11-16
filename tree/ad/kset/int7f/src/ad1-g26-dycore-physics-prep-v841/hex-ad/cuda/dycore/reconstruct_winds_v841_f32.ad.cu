// SPDX-License-Identifier: Apache-2.0
// Geometry and connectivity are fixed and have already passed the original
// sealed-geometry validation.  The adjoint uses a deterministic inverse CSR:
// each edge's (cell,slot) pairs are in ascending cell, then slot order.
#include "prep_branch_record.cuh"

#define WIND_ARGS \
    int nlev, int ncells, int nedges, int maxedges, \
    const float *normal_velocity, const int *edges_on_cell, \
    const int *n_edges_on_cell, const float *coeffs_reconstruct, \
    const float *lat_cell, const float *lon_cell, \
    const int *sealed_edges_on_cell, const int *sealed_n_edges_on_cell, \
    const float *sealed_coeffs_reconstruct, const float *sealed_lat_cell, \
    const float *sealed_lon_cell, float *x, float *y, float *z, \
    float *u_p, float *v_p, int *invalid

#if !defined(AD_AD_ONLY)
extern "C" __global__ void reconstruct_winds_v841_f32__tl(
    WIND_ARGS, const float *d_normal_velocity, float *d_x, float *d_y,
    float *d_z, float *d_u_p, float *d_v_p, unsigned int *branch_record)
{
    (void)normal_velocity; (void)sealed_edges_on_cell;
    (void)sealed_n_edges_on_cell; (void)sealed_coeffs_reconstruct;
    (void)sealed_lat_cell; (void)sealed_lon_cell; (void)x; (void)y; (void)z;
    (void)u_p; (void)v_p; (void)invalid;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    PrepBranchRecorder br{branch_record, 10};
    br.count(0,i>=nlev*ncells);
    if (i >= nlev*ncells) { br.flush(24); return; }
    int c = i % ncells, k = i / ncells;
    bool changed=n_edges_on_cell[c]!=sealed_n_edges_on_cell[c]
        || __float_as_uint(lat_cell[c])!=__float_as_uint(sealed_lat_cell[c])
        || __float_as_uint(lon_cell[c])!=__float_as_uint(sealed_lon_cell[c]);
    br.count(1,changed); br.count_n(2,true,(unsigned)maxedges);br.count(2,false);
    for (int s=0;s<maxedges;++s) {
        int e=c*maxedges+s, q=3*e;
        changed=changed || edges_on_cell[e]!=sealed_edges_on_cell[e]
          || __float_as_uint(coeffs_reconstruct[q])!=__float_as_uint(sealed_coeffs_reconstruct[q])
          || __float_as_uint(coeffs_reconstruct[q+1])!=__float_as_uint(sealed_coeffs_reconstruct[q+1])
          || __float_as_uint(coeffs_reconstruct[q+2])!=__float_as_uint(sealed_coeffs_reconstruct[q+2]);
    }
    br.count(3,changed); if(changed) { br.flush(24); return; }
    int count=n_edges_on_cell[c]; bool badcount=count<0 || count>maxedges;
    br.count(4,badcount); if(badcount) { br.flush(24); return; }
    br.count_n(5,true,(unsigned)count);br.count(5,false);
    float rx=0.0f,ry=0.0f,rz=0.0f;
    for(int s=0;s<count;++s) {
        int e=edges_on_cell[c*maxedges+s]; bool bad=e<0||e>=nedges;
        br.count(6,bad); if(bad) continue;
        int q=3*(c*maxedges+s);float cx=coeffs_reconstruct[q];
        float cy=coeffs_reconstruct[q+1],cz=coeffs_reconstruct[q+2];
        br.count(7,!isfinite(cx)||!isfinite(cy)||!isfinite(cz));
        float vel=normal_velocity[k*nedges+e]; br.count(8,!isfinite(vel));
        rx=br.add(rx,br.mul(cx,vel));ry=br.add(ry,br.mul(cy,vel));
        rz=br.add(rz,br.mul(cz,vel));
    }
    float rclat=cosf(lat_cell[c]),rslat=sinf(lat_cell[c]);
    float rclon=cosf(lon_cell[c]),rslon=sinf(lon_cell[c]);
    br.count(9,!isfinite(rclat)||!isfinite(rslat)||!isfinite(rclon)||!isfinite(rslon));
    (void)br.add(br.mul(br.sub(0.0f,rx),rslon),br.mul(ry,rclon));
    float horiz=br.add(br.mul(rx,rclon),br.mul(ry,rslon));
    (void)br.add(br.mul(br.sub(0.0f,horiz),rslat),br.mul(rz,rclat));
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    for (int s = 0; s < n_edges_on_cell[c]; ++s) {
        int e = edges_on_cell[c*maxedges+s];
        // The forward flags an invalid edge and skips it.  Keep the tangent
        // in bounds while the caller propagates that original refusal.
        if (e < 0 || e >= nedges) continue;
        int j = (c*maxedges+s)*3;
        float dv = d_normal_velocity[k*nedges+e];
        dx += coeffs_reconstruct[j]*dv;
        dy += coeffs_reconstruct[j+1]*dv;
        dz += coeffs_reconstruct[j+2]*dv;
    }
    float clat = cosf(lat_cell[c]), slat = sinf(lat_cell[c]);
    float clon = cosf(lon_cell[c]), slon = sinf(lon_cell[c]);
    d_x[i]=dx; d_y[i]=dy; d_z[i]=dz;
    d_u_p[i] = -dx*slon + dy*clon;
    d_v_p[i] = -(dx*clon + dy*slon)*slat + dz*clat;
    br.flush(24);
}
#endif

#if !defined(AD_TL_ONLY)
extern "C" __global__ void reconstruct_winds_v841_f32__ad(
    WIND_ARGS, float *b_normal_velocity, float *b_x, float *b_y,
    float *b_z, float *b_u_p, float *b_v_p,
    const int *inverse_offsets, const int *inverse_cells,
    const int *inverse_slots, int phase)
{
    (void)normal_velocity; (void)edges_on_cell; (void)n_edges_on_cell;
    (void)sealed_edges_on_cell; (void)sealed_n_edges_on_cell;
    (void)sealed_coeffs_reconstruct; (void)sealed_lat_cell;
    (void)sealed_lon_cell; (void)x; (void)y; (void)z; (void)u_p; (void)v_p;
    (void)invalid;
    // A separate ordered launch is required: the gather below reads a cell
    // seed from several edge threads.  Clear it only after every gather ends.
    if (phase == 1) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < nlev*ncells) {
            b_x[i] = 0.0f; b_y[i] = 0.0f; b_z[i] = 0.0f;
            b_u_p[i] = 0.0f; b_v_p[i] = 0.0f;
        }
        return;
    }
    // The forward launch grid may have fewer threads than nlev*nedges.
    // A fixed grid-stride traversal preserves one deterministic owner per edge
    // and lets the binder use the capture's forward geometry unchanged.
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < nlev*nedges; i += blockDim.x * gridDim.x) {
    int e = i % nedges, k = i / nedges;
    double sum = 0.0;
    for (int p = inverse_offsets[e]; p < inverse_offsets[e+1]; ++p) {
        int c = inverse_cells[p], s = inverse_slots[p];
        int j = (c*maxedges+s)*3, ci = k*ncells+c;
        // Only forward threads in the launched prefix overwrote cell outputs.
        if ((unsigned long long)ci >= (unsigned long long)blockDim.x * gridDim.x) continue;
        float clat = cosf(lat_cell[c]), slat = sinf(lat_cell[c]);
        float clon = cosf(lon_cell[c]), slon = sinf(lon_cell[c]);
        double bx = (double)b_x[ci] - (double)slon*b_u_p[ci] - (double)clon*slat*b_v_p[ci];
        double by = (double)b_y[ci] + (double)clon*b_u_p[ci] - (double)slon*slat*b_v_p[ci];
        double bz = (double)b_z[ci] + (double)clat*b_v_p[ci];
        sum += (double)coeffs_reconstruct[j]*bx + (double)coeffs_reconstruct[j+1]*by
             + (double)coeffs_reconstruct[j+2]*bz;
    }
    b_normal_velocity[i] = (float)((double)b_normal_velocity[i] + sum);
    }
}
#endif
