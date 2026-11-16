// SPDX-License-Identifier: MIT
#include "horizontal_v841_ad_common.cuh"
extern "C" __global__ void pv_apvm_v841_f32__tl(
    const float *normal_velocity,const float *tangential_velocity,const float *pv_vertex,const float *pv_cell,
    const int *vertices_on_edge,const int *cells_on_edge,const float *inv_dc_edge,const float *inv_dv_edge,
    float scale,int nlev,int ncells,int nedges,int nvertices,float *pv_edge,float *grad_pv_normal,float *grad_pv_tangential,
    const float *d_normal_velocity,const float *d_tangential_velocity,const float *d_pv_vertex,const float *d_pv_cell,
    const float *d_scale,float *d_pv_edge,float *d_grad_pv_normal,float *d_grad_pv_tangential,HorizontalAdFailure *failure)
{
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nlev*nedges;i+=gridDim.x*blockDim.x){
        const int e=i%nedges,k=i/nedges,v0=vertices_on_edge[2*e],v1=vertices_on_edge[2*e+1],c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
        if(v0<0||v0>=nvertices||v1<0||v1>=nvertices||c0<0||c0>=ncells||c1<0||c1>=ncells||
           ((grad_pv_normal==grad_pv_tangential)!=(d_grad_pv_normal==d_grad_pv_tangential))){horizontal_ad_fail(failure,0,5,i,6);continue;}
        const int va=k*nvertices+v0,vb=k*nvertices+v1,ca=k*ncells+c0,cb=k*ncells+c1;
        if(!isfinite(normal_velocity[i])||!isfinite(tangential_velocity[i])||!isfinite(pv_vertex[va])||!isfinite(pv_vertex[vb])||
           !isfinite(pv_cell[ca])||!isfinite(pv_cell[cb])||!isfinite(inv_dc_edge[e])||!isfinite(inv_dv_edge[e])||
           !isfinite(scale)||!isfinite(pv_edge[i])||!isfinite(d_normal_velocity[i])||!isfinite(d_tangential_velocity[i])||
           !isfinite(d_pv_vertex[va])||!isfinite(d_pv_vertex[vb])||!isfinite(d_pv_cell[ca])||!isfinite(d_pv_cell[cb])||
           !isfinite(*d_scale)||!isfinite(d_pv_edge[i])){horizontal_ad_fail(failure,1,1,i,6);continue;}
        const float gt=(pv_vertex[vb]-pv_vertex[va])*inv_dv_edge[e],gn=(pv_cell[cb]-pv_cell[ca])*inv_dc_edge[e];
        const double dgt=((double)d_pv_vertex[vb]-(double)d_pv_vertex[va])*(double)inv_dv_edge[e],dgn=((double)d_pv_cell[cb]-(double)d_pv_cell[ca])*(double)inv_dc_edge[e];
        const float value=tangential_velocity[i]*gt+normal_velocity[i]*gn;
        const double delta=(double)d_tangential_velocity[i]*(double)gt+(double)tangential_velocity[i]*dgt+(double)d_normal_velocity[i]*(double)gn+(double)normal_velocity[i]*dgn;
        d_grad_pv_tangential[i]=dgt;d_grad_pv_normal[i]=dgn;
        d_pv_edge[i]=(float)((double)d_pv_edge[i]-(double)(*d_scale)*(double)value-(double)scale*delta);
        if(!isfinite(d_pv_edge[i])||!isfinite(dgn)||!isfinite(dgt))horizontal_ad_fail(failure,2,14,i,6);
    }
}
// Auxiliary, fixed inverse topology is verified against each forward edge.
// One CTA snapshots the two overwrite seeds, allowing deterministic gathers.
extern "C" __global__ void pv_apvm_v841_f32__ad(
    const float *normal_velocity,const float *tangential_velocity,const float *pv_vertex,const float *pv_cell,
    const int *vertices_on_edge,const int *cells_on_edge,const float *inv_dc_edge,const float *inv_dv_edge,
    float scale,int nlev,int ncells,int nedges,int nvertices,float *pv_edge,float *grad_pv_normal,float *grad_pv_tangential,
    const int *edges_on_cell,const int *n_edges_on_cell,int max_edges,const int *edges_on_vertex,int vertex_degree,
    float *b_normal_velocity,float *b_tangential_velocity,float *b_pv_vertex,float *b_pv_cell,float *b_scale,
    const float *b_pv_edge_out,float *b_pv_edge_in,float *b_grad_pv_normal,float *b_grad_pv_tangential,float *workspace,HorizontalAdFailure *failure)
{
    if(gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1||max_edges<=0||vertex_degree<=0||
       ((grad_pv_normal==grad_pv_tangential)!=(b_grad_pv_normal==b_grad_pv_tangential))){
        if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)horizontal_ad_fail(failure,0,14,0,6);
        return;
    }
    const int ne=nlev*nedges,nc=nlev*ncells,nv=nlev*nvertices;
    for(int i=threadIdx.x;i<ne;i+=blockDim.x){
        const int e=i%nedges,k=i/nedges,v0=vertices_on_edge[2*e],v1=vertices_on_edge[2*e+1],c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
        workspace[i]=b_pv_edge_out[i];workspace[ne+i]=b_grad_pv_normal[i];
        workspace[2*ne+i]=grad_pv_normal==grad_pv_tangential?0.0f:b_grad_pv_tangential[i];
        if(v0<0||v0>=nvertices||v1<0||v1>=nvertices||c0<0||c0>=ncells||c1<0||c1>=ncells){horizontal_ad_fail(failure,1,5,i,6);continue;}
        for(int side=0;side<2;++side){
            int c=side?c1:c0,count=n_edges_on_cell[c],matches=0;
            if(count<0||count>max_edges){horizontal_ad_fail(failure,2,6,i,6);continue;}
            for(int s=0;s<count;++s)if(edges_on_cell[c*max_edges+s]==e)++matches;
            if(matches!=1)horizontal_ad_fail(failure,3,6,i,6);
            int v=side?v1:v0;matches=0;
            for(int s=0;s<vertex_degree;++s)if(edges_on_vertex[v*vertex_degree+s]==e)++matches;
            if(matches!=1)horizontal_ad_fail(failure,4,5,i,6);
        }
        const int va=k*nvertices+v0,vb=k*nvertices+v1,ca=k*ncells+c0,cb=k*ncells+c1;
        if(!isfinite(normal_velocity[i])||!isfinite(tangential_velocity[i])||!isfinite(pv_vertex[va])||!isfinite(pv_vertex[vb])||
           !isfinite(pv_cell[ca])||!isfinite(pv_cell[cb])||!isfinite(inv_dc_edge[e])||!isfinite(inv_dv_edge[e])||
           !isfinite(scale)||!isfinite(pv_edge[i])||!isfinite(workspace[i])||!isfinite(workspace[ne+i])||!isfinite(workspace[2*ne+i]))
            horizontal_ad_fail(failure,5,1,i,6);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<ne;i+=blockDim.x){b_pv_edge_in[i]+=workspace[i];b_grad_pv_tangential[i]=0.0f;b_grad_pv_normal[i]=0.0f;}
    __syncthreads();
    __shared__ double partial[128];
    double bscale=0.0;
    for(int i=threadIdx.x;i<ne;i+=blockDim.x){
        const int e=i%nedges,k=i/nedges,v0=vertices_on_edge[2*e],v1=vertices_on_edge[2*e+1],c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
        if(v0<0||v0>=nvertices||v1<0||v1>=nvertices||c0<0||c0>=ncells||c1<0||c1>=ncells)continue;
        const float gt=(pv_vertex[k*nvertices+v1]-pv_vertex[k*nvertices+v0])*inv_dv_edge[e];
        const float gn=(pv_cell[k*ncells+c1]-pv_cell[k*ncells+c0])*inv_dc_edge[e], w=workspace[i];
        b_normal_velocity[i]+=-scale*gn*w;b_tangential_velocity[i]+=-scale*gt*w;
        bscale-=((double)tangential_velocity[i]*(double)gt+(double)normal_velocity[i]*(double)gn)*(double)w;
        if(!isfinite(b_normal_velocity[i])||!isfinite(b_tangential_velocity[i]))horizontal_ad_fail(failure,6,1,i,6);
    }
    partial[threadIdx.x]=bscale;
    __syncthreads();
    for(int stride=64;stride>0;stride>>=1){if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0){*b_scale=(float)((double)*b_scale+partial[0]);if(!isfinite(*b_scale))horizontal_ad_fail(failure,7,9,0,6);}
    for(int j=threadIdx.x;j<nc;j+=blockDim.x){
        const int c=j%ncells,k=j/ncells,count=n_edges_on_cell[c];double b=(double)b_pv_cell[j];
        if(!isfinite(b)||!isfinite(pv_cell[j])||count<0||count>max_edges){horizontal_ad_fail(failure,8,4,j,6);continue;}
        for(int s=0;s<count;++s){
            const int e=edges_on_cell[c*max_edges+s];if(e<0||e>=nedges){horizontal_ad_fail(failure,9,6,j,6);continue;}
            const int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1],i=k*nedges+e;
            if(c!=c0&&c!=c1){horizontal_ad_fail(failure,10,6,j,6);continue;}
            const float sign=(c==c1?1.0f:0.0f)-(c==c0?1.0f:0.0f);
            const double bg=(double)workspace[ne+i]-(double)scale*(double)normal_velocity[i]*(double)workspace[i];
            b+=(double)sign*(double)inv_dc_edge[e]*bg;
        }
        b_pv_cell[j]=(float)b;if(!isfinite(b_pv_cell[j]))horizontal_ad_fail(failure,11,4,j,6);
    }
    for(int j=threadIdx.x;j<nv;j+=blockDim.x){
        const int v=j%nvertices,k=j/nvertices;double b=(double)b_pv_vertex[j];
        if(!isfinite(b)||!isfinite(pv_vertex[j]))horizontal_ad_fail(failure,12,3,j,6);
        for(int s=0;s<vertex_degree;++s){
            const int e=edges_on_vertex[v*vertex_degree+s];if(e<0)continue;
            if(e>=nedges){horizontal_ad_fail(failure,13,5,j,6);continue;}
            const int v0=vertices_on_edge[2*e],v1=vertices_on_edge[2*e+1],i=k*nedges+e;
            if(v!=v0&&v!=v1){horizontal_ad_fail(failure,14,5,j,6);continue;}
            const float sign=(v==v1?1.0f:0.0f)-(v==v0?1.0f:0.0f);
            const double bg=(double)workspace[2*ne+i]-(double)scale*(double)tangential_velocity[i]*(double)workspace[i];
            b+=(double)sign*(double)inv_dv_edge[e]*bg;
        }
        b_pv_vertex[j]=(float)b;if(!isfinite(b_pv_vertex[j]))horizontal_ad_fail(failure,15,3,j,6);
    }
}

// Branch reconstruction is a separate derivative entry. Its output buffers are
// scratch copies, so it never writes the ordinary forward state.
#include "horizontal_v841_branch.cuh"

#define HB_ELEMENT_LOOP(total) for (int element=blockDim.x*blockIdx.x+threadIdx.x; element<(total); element+=gridDim.x*blockDim.x)
#define HEX_HOIST_CELL 10
#define HB_CELL_CHUNK(rows) const int hex_rows=(rows); const int hex_chunks=(hex_rows+lpt-1)/lpt; const int hex_element=blockDim.x*blockIdx.x+threadIdx.x; if(hex_element>=ncells*hex_chunks)return; const int cell=hex_element%ncells; const int hex_k0=(hex_element/ncells)*lpt; const int hex_k1=min(hex_rows,hex_k0+lpt);
__device__ __forceinline__ int hb_lidx(int k,int x,int n){return k*n+x;}
__device__ __forceinline__ float hb_sign(int c,int e,const int *p){return p[2*e]==c?1.0f:-1.0f;}
__device__ __forceinline__ int hb_kite(int v,int c,const int *p,int d){for(int j=0;j<d;++j)if(p[v*d+j]==c)return j;return -1;}

__device__ __forceinline__ void pv_apvm_v841_f32__branch_body(
    const float *normal_velocity, const float *tangential_velocity,
    const float *pv_vertex, const float *pv_cell,
    const int *vertices_on_edge, const int *cells_on_edge,
    const float *inv_dc_edge, const float *inv_dv_edge, const float scale,
    const int nlev, const int ncells, const int nedges, const int nvertices,
    float *pv_edge, float *grad_pv_normal, float *grad_pv_tangential, const float *pv_edge_pre, unsigned *r)
{
    HB_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int vertex0 = vertices_on_edge[2 * edge];
        const int vertex1 = vertices_on_edge[2 * edge + 1];
        const int cell0 = cells_on_edge[2 * edge];
        const int cell1 = cells_on_edge[2 * edge + 1];
        const int index = hb_lidx(k, edge, nedges);
        const float grad_t = hb_mul(r,hb_sub(r,
            pv_vertex[hb_lidx(k, vertex1, nvertices)],
            pv_vertex[hb_lidx(k, vertex0, nvertices)]), inv_dv_edge[edge]);
        const float grad_n = hb_mul(r,hb_sub(r,
            pv_cell[hb_lidx(k, cell1, ncells)],
            pv_cell[hb_lidx(k, cell0, ncells)]), inv_dc_edge[edge]);
        grad_pv_tangential[index] = grad_t;
        grad_pv_normal[index] = grad_n;
        pv_edge[index] = hb_sub(r,pv_edge_pre[index], hb_mul(r,scale, hb_add(r,
            hb_mul(r,tangential_velocity[index], grad_t),
            hb_mul(r,normal_velocity[index], grad_n))));
    }
}
extern "C" __global__ void pv_apvm_v841_f32__branch(
    const float *normal_velocity, const float *tangential_velocity,
    const float *pv_vertex, const float *pv_cell,
    const int *vertices_on_edge, const int *cells_on_edge,
    const float *inv_dc_edge, const float *inv_dv_edge, const float scale,
    const int nlev, const int ncells, const int nedges, const int nvertices,
    float *pv_edge, float *grad_pv_normal, float *grad_pv_tangential, const float *pv_edge_pre, unsigned *branch_record)
{
    __shared__ unsigned r[26];
    for (int i=threadIdx.x; i<26; i+=blockDim.x) r[i]=0u;
    __syncthreads();
    pv_apvm_v841_f32__branch_body(normal_velocity, tangential_velocity, pv_vertex, pv_cell, vertices_on_edge, cells_on_edge, inv_dc_edge, inv_dv_edge, scale, nlev, ncells, nedges, nvertices, pv_edge, grad_pv_normal, grad_pv_tangential, pv_edge_pre, r);
    __syncthreads();
    for (int i=threadIdx.x; i<26; i+=blockDim.x) if(r[i]) atomicAdd(branch_record+i,r[i]);
}
