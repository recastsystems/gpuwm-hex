// SPDX-License-Identifier: Apache-2.0
// Hand transcription of the v8.4.1 divergence, pressure and heat update.
// The reverse replays each executed expression onto a bounded device scratch
// tape, then reduces per-lane input contributions in fixed lane order. No atomics.
#include "acoustic_ad_tape.cuh"
#define RS_C(k,c,n) ((k)*(n)+(c))
#define RS_E(k,e,n) ((k)*(n)+(e))
#define RS_CE(c,s,m) ((c)*(m)+(s))
struct RsFields {
    AcousticField sign,dv,area,theta,rdzw,cofrz,coftz,ewm,ru,rw,rho,rt,trho,trt,rs,ts;
};
template<class O>
__device__ void rs_ts_element(O &o,int k,int cell,int ncells,int nedges,int max_edges,
        int count,const int *edges,const int *cells,float dts,typename O::S *ddts,RsFields f) {
    typedef typename O::V V;
    V dt=o.scalar(dts,ddts,1),half=o.constant(0.5f);
    V r=o.constant(0.0f),t=o.constant(0.0f);
    for(int slot=0;slot<count;++slot) {
        int ces=RS_CE(cell,slot,max_edges),edge=edges[ces];
        if(edge<0||edge>=nedges){acoustic_fail(o.failure,2,ces,5);return;}
        int c0=cells[2*edge],c1=cells[2*edge+1];
        if(c0<0||c0>=ncells||c1<0||c1>=ncells){acoustic_fail(o.failure,3,edge,5);return;}
        V flux=o.mul(o.input(f.sign,ces),dt);
        flux=o.mul(flux,o.input(f.dv,edge));
        flux=o.mul(flux,o.input(f.ru,RS_E(k,edge,nedges)));
        flux=o.mul(flux,o.input(f.area,cell));
        r=o.sub(r,flux);
        V theta_sum=o.add(o.input(f.theta,RS_C(k,c1,ncells)),o.input(f.theta,RS_C(k,c0,ncells)));
        t=o.sub(t,o.mul(o.mul(flux,half),theta_sum));
    }
    int i=RS_C(k,cell,ncells),ip=RS_C(k+1,cell,ncells);
    r=o.add(o.add(o.input(f.rho,i),o.mul(dt,o.input(f.trho,i))),r);
    V vr=o.sub(o.mul(o.input(f.ewm,k+1),o.input(f.rw,ip)),o.mul(o.input(f.ewm,k),o.input(f.rw,i)));
    r=o.sub(r,o.mul(o.mul(dt,o.input(f.cofrz,k)),vr));
    t=o.add(o.add(o.input(f.rt,i),o.mul(dt,o.input(f.trt,i))),t);
    V vt=o.sub(o.mul(o.mul(o.input(f.ewm,k+1),o.input(f.coftz,ip)),o.input(f.rw,ip)),
               o.mul(o.mul(o.input(f.ewm,k),o.input(f.coftz,i)),o.input(f.rw,i)));
    t=o.sub(t,o.mul(o.mul(dt,o.input(f.rdzw,k)),vt));
    o.output(f.rs,i,r);o.output(f.ts,i,t);o.finish();
}
#define RS_PRIMAL_ARGS \
    const int nlev,const int ncells,const int nedges,const int max_edges,const float dts, \
    const int *n_edges_on_cell,const int *edges_on_cell,const int *cells_on_edge, \
    const float *edge_sign_on_cell,const float *dv_edge,const float *inv_area_cell, \
    const float *theta_m,const float *rdzw,const float *cofrz,const float *coftz, \
    const float *ewm,const float *ru_p,const float *rw_p,const float *rho_pp, \
    const float *rtheta_pp,const float *tend_rho,const float *tend_rt,float *rs,float *ts
#define RS_SHADOW_ARGS \
    float *d_sign,float *d_dv,float *d_area,float *d_theta,float *d_rdzw,float *d_cofrz, \
    float *d_coftz,float *d_ewm,float *d_ru,float *d_rw,float *d_rho,float *d_rt, \
    float *d_trho,float *d_trt,float *d_rs,float *d_ts
#define RS_FIELDS RsFields f={ \
    {edge_sign_on_cell,d_sign,4},{dv_edge,d_dv,5},{inv_area_cell,d_area,6}, \
    {theta_m,d_theta,7},{rdzw,d_rdzw,8},{cofrz,d_cofrz,9},{coftz,d_coftz,10}, \
    {ewm,d_ewm,11},{ru_p,d_ru,12},{rw_p,d_rw,13},{rho_pp,d_rho,14}, \
    {rtheta_pp,d_rt,15},{tend_rho,d_trho,16},{tend_rt,d_trt,17},{rs,d_rs,18},{ts,d_ts,19} }
extern "C" __global__ void acoustic_rs_ts_v841__tl(RS_PRIMAL_ARGS,const float *d_dts,
                                                    RS_SHADOW_ARGS,AcousticFailure *failures,unsigned *branch_record) {
    int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=ncells)return;
    RS_FIELDS; unsigned branch_local[30]={0};unsigned *branch_counter=branch_record?branch_local:nullptr;
    AcousticTangent o{failures,branch_counter,1};float ddts=*d_dts;
    int count=n_edges_on_cell[cell];
    if(count<0||count>max_edges){acoustic_fail(o.failure,2,cell,5);return;}
    for(int k=0;k<nlev;++k)
        rs_ts_element(o,k,cell,ncells,nedges,max_edges,count,edges_on_cell,cells_on_edge,dts,&ddts,f);
    if(branch_record)for(int i=0;i<30;++i)if(branch_local[i]) atomicAdd(branch_record+i,branch_local[i]);
}
// The workspace contains 32 private copies of the fields shared between
// cells. Every lane handles a disjoint subset of cells; the final reduction
// gathers those copies in increasing lane order. Cell-local cotangents and
// output seeds have a unique writer. The runner supplies a zeroable workspace.
extern "C" __global__ void acoustic_rs_ts_v841__ad(RS_PRIMAL_ARGS,float *b_dts,
            RS_SHADOW_ARGS,AcousticNode *scratch,
            double *partial,AcousticFailure *failures) {
    constexpr int lanes=32;
    if(gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||blockDim.x!=128||blockDim.y!=1||blockDim.z!=1){
        if(!blockIdx.x&&!threadIdx.x) acoustic_fail(failures,0,0,6);return;
    }
    if(nlev<0||ncells<0||nedges<0||max_edges<0){
        if(!threadIdx.x) acoustic_fail(failures,0,0,6);return;
    }
    const int capacity=128+16*max_edges;
    const size_t sign_n=(size_t)ncells*max_edges,edge_n=nedges,area_n=ncells;
    const size_t theta_n=(size_t)nlev*ncells,interface_n=(size_t)(nlev+1)*ncells,ru_n=(size_t)nlev*nedges;
    const size_t os=1,od=os+sign_n,oa=od+edge_n,ot=oa+area_n,oz=ot+theta_n;
    const size_t oc=oz+nlev,oft=oc+nlev,oe=oft+interface_n,oru=oe+nlev+1,orw=oru+ru_n;
    const size_t stride=orw+interface_n;
    for(size_t i=threadIdx.x;i<lanes*stride;i+=blockDim.x) partial[i]=0.0;
    __syncthreads();
    if(threadIdx.x<lanes){
        int lane=threadIdx.x;double *q=partial+lane*stride;
        RS_FIELDS;f.sign.accum=q+os;f.dv.accum=q+od;f.area.accum=q+oa;f.theta.accum=q+ot;
        f.rdzw.accum=q+oz;f.cofrz.accum=q+oc;f.coftz.accum=q+oft;f.ewm.accum=q+oe;
        f.ru.accum=q+oru;f.rw.accum=q+orw;
        AcousticFailure *failure=failures;
        for(int cell=lane;cell<ncells;cell+=lanes){
            int count=n_edges_on_cell[cell];
            if(count<0||count>max_edges){acoustic_fail(failure,2,cell,5);break;}
            for(int k=0;k<nlev;++k){
                AcousticReverse o{scratch+(size_t)lane*capacity,0,capacity,failure};o.reset();
                rs_ts_element(o,k,cell,ncells,nedges,max_edges,count,edges_on_cell,cells_on_edge,dts,q,f);
            }
        }
    }
    __syncthreads();
    for(size_t j=threadIdx.x;j<stride;j+=blockDim.x){
        float *dst;
        if(j==0) dst=b_dts;
        else if(j<od) dst=d_sign+(j-os);
        else if(j<oa) dst=d_dv+(j-od);
        else if(j<ot) dst=d_area+(j-oa);
        else if(j<oz) dst=d_theta+(j-ot);
        else if(j<oc) dst=d_rdzw+(j-oz);
        else if(j<oft) dst=d_cofrz+(j-oc);
        else if(j<oe) dst=d_coftz+(j-oft);
        else if(j<oru) dst=d_ewm+(j-oe);
        else if(j<orw) dst=d_ru+(j-oru);
        else dst=d_rw+(j-orw);
        double sum=mpas_f32_to_f64_ieee(*dst);
        for(int lane=0;lane<lanes;++lane) sum+=partial[(size_t)lane*stride+j];
        if(!isfinite(sum)) acoustic_fail(failures,1,j,3);
        *dst=mpas_f64_to_f32_ieee(sum);
    }
}
