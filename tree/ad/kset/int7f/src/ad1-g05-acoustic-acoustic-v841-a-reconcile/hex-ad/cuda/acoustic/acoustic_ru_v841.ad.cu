// SPDX-License-Identifier: Apache-2.0
// Hand tangent and reverse for the v8.4.1 horizontal acoustic update.
// One reverse block computes per-edge contributions, synchronizes, and then
// gathers per-cell contributions in edge order. No unordered atomics are used.
#include "acoustic_branch_record.cuh"

// Section 35 device failure slots, indexed by source_id - 1.
typedef unsigned AdFailureRecord;
__device__ __forceinline__ void ru_fail(AdFailureRecord *f, int slot, unsigned long long field,
                                       unsigned long long index, unsigned long long source) {
    (void)slot; (void)field; (void)index;
    atomicExch(f + (source - 1), 1u);
}
__device__ __forceinline__ void ru_check(AdFailureRecord *f, int slot, unsigned long long field,
                                        unsigned long long index, float value) {
    if (!isfinite(value)) ru_fail(f,slot,field,index,1);
}
struct RuDual { float v, d; unsigned *branches; };
__device__ __forceinline__ RuDual ru_dual(unsigned *branch,float v,float d) { return {v,d,branch}; }
__device__ __forceinline__ RuDual ru_add(RuDual a,RuDual b) {
    return {abr_add(a.v,b.v,a.branches,5),mpas_add(a.d,b.d),a.branches};
}
__device__ __forceinline__ RuDual ru_sub(RuDual a,RuDual b) {
    return {abr_sub(a.v,b.v,a.branches,5),mpas_sub(a.d,b.d),a.branches};
}
__device__ __forceinline__ RuDual ru_mul(RuDual a,RuDual b) {
    return {abr_mul(a.v,b.v,a.branches,5),mpas_add(mpas_mul(a.d,b.v),mpas_mul(a.v,b.d)),a.branches};
}
__device__ __forceinline__ RuDual ru_div(RuDual a,RuDual b) {
    const float v=abr_div(a.v,b.v,a.branches,5);
    return {v,mpas_div(mpas_sub(a.d,mpas_mul(v,b.d)),b.v),a.branches};
}
#define RC(k,c,n) ((k)*(n)+(c))
#define RE(k,e,n) ((k)*(n)+(e))

extern "C" __global__ void acoustic_ru_v841__tl(
    const int nlev,const int nedges,const int ncells,const int small_step,
    const float dts,const float gravity,const float rgas,const float cp,
    const int *cells_on_edge,const float *inv_dc_edge,const float *zz,const float *exner,
    const float *cqu,const float *zxu,const float *tend_ru,const float *rho_pp,
    const float *rtheta_pp,float *ru_p,float *ru_avg,
    const float *d_dts_ptr,const float *d_gravity_ptr,const float *d_rgas_ptr,const float *d_cp_ptr,
    const float *d_inv_dc_edge,const float *d_zz,const float *d_exner,const float *d_cqu,
    const float *d_zxu,const float *d_tend_ru,const float *d_rho_pp,const float *d_rtheta_pp,
    float *d_ru_p,float *d_ru_avg,AdFailureRecord *failures,unsigned *branch_record)
{
    const int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=nedges)return;
    const float d_dts=*d_dts_ptr,d_gravity=*d_gravity_ptr,d_rgas=*d_rgas_ptr,d_cp=*d_cp_ptr;
    const int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
    if(c0<0||c0>=ncells||c1<0||c1>=ncells){ru_fail(failures,e,5,e,4);return;}
    ru_check(failures,e,1,0,dts); ru_check(failures,e,1,0,d_dts);
    ru_check(failures,e,2,0,gravity); ru_check(failures,e,2,0,d_gravity);
    ru_check(failures,e,3,0,rgas); ru_check(failures,e,3,0,d_rgas);
    ru_check(failures,e,4,0,cp); ru_check(failures,e,4,0,d_cp);
    unsigned branch_local[38]={0};unsigned *branch_counter=branch_record?branch_local:nullptr;
    const RuDual dt=ru_dual(branch_counter,dts,d_dts),g=ru_dual(branch_counter,gravity,d_gravity);
    const RuDual rg=ru_dual(branch_counter,rgas,d_rgas),c=ru_dual(branch_counter,cp,d_cp),half=ru_dual(branch_counter,0.5f,0.0f);
    const RuDual cv=ru_sub(c,rg);
    if(mpas_f32_magnitude_bits(cv.v)==0u)ru_fail(failures,e,4,0,2);
    const RuDual csq=ru_mul(c,ru_div(rg,cv));
    for(int k=0;k<nlev;++k){
        const int i=RE(k,e,nedges),i0=RC(k,c0,ncells),i1=RC(k,c1,ncells);
        ru_check(failures,e,10,i,tend_ru[i]);ru_check(failures,e,10,i,d_tend_ru[i]);
        if(small_step==1){
            const RuDual v=ru_mul(dt,ru_dual(branch_counter,tend_ru[i],d_tend_ru[i]));
            ru_check(failures,e,13,i,v.v);ru_check(failures,e,13,i,v.d);d_ru_p[i]=v.d;d_ru_avg[i]=v.d;
        }else{
            ru_check(failures,e,5,e,inv_dc_edge[e]);ru_check(failures,e,5,e,d_inv_dc_edge[e]);
            ru_check(failures,e,6,i0,zz[i0]);ru_check(failures,e,6,i1,zz[i1]);
            ru_check(failures,e,6,i0,d_zz[i0]);ru_check(failures,e,6,i1,d_zz[i1]);
            ru_check(failures,e,7,i0,exner[i0]);ru_check(failures,e,7,i1,exner[i1]);
            ru_check(failures,e,7,i0,d_exner[i0]);ru_check(failures,e,7,i1,d_exner[i1]);
            ru_check(failures,e,8,i,cqu[i]);ru_check(failures,e,8,i,d_cqu[i]);
            ru_check(failures,e,9,i,zxu[i]);ru_check(failures,e,9,i,d_zxu[i]);
            ru_check(failures,e,11,i0,rho_pp[i0]);ru_check(failures,e,11,i1,rho_pp[i1]);
            ru_check(failures,e,11,i0,d_rho_pp[i0]);ru_check(failures,e,11,i1,d_rho_pp[i1]);
            ru_check(failures,e,12,i0,rtheta_pp[i0]);ru_check(failures,e,12,i1,rtheta_pp[i1]);
            ru_check(failures,e,12,i0,d_rtheta_pp[i0]);ru_check(failures,e,12,i1,d_rtheta_pp[i1]);
            ru_check(failures,e,13,i,ru_p[i]);ru_check(failures,e,13,i,d_ru_p[i]);
            ru_check(failures,e,14,i,ru_avg[i]);ru_check(failures,e,14,i,d_ru_avg[i]);
            RuDual normal=ru_mul(ru_sub(ru_dual(branch_counter,rtheta_pp[i1],d_rtheta_pp[i1]),
                ru_dual(branch_counter,rtheta_pp[i0],d_rtheta_pp[i0])),ru_dual(branch_counter,inv_dc_edge[e],d_inv_dc_edge[e]));
            RuDual mean=ru_mul(half,ru_add(ru_dual(branch_counter,zz[i1],d_zz[i1]),ru_dual(branch_counter,zz[i0],d_zz[i0])));
            if(mpas_f32_magnitude_bits(mean.v)==0u)ru_fail(failures,e,6,i0,3);
            RuDual normalized=ru_div(normal,mean);
            RuDual pgrad=ru_mul(ru_dual(branch_counter,cqu[i],d_cqu[i]),half);
            pgrad=ru_mul(pgrad,csq);
            pgrad=ru_mul(pgrad,ru_add(ru_dual(branch_counter,exner[i0],d_exner[i0]),ru_dual(branch_counter,exner[i1],d_exner[i1])));
            pgrad=ru_mul(pgrad,normalized);
            RuDual terrain=ru_mul(ru_mul(ru_mul(half,ru_dual(branch_counter,zxu[i],d_zxu[i])),g),
                ru_add(ru_dual(branch_counter,rho_pp[i0],d_rho_pp[i0]),ru_dual(branch_counter,rho_pp[i1],d_rho_pp[i1])));
            pgrad=ru_add(pgrad,terrain);
            RuDual u=ru_add(ru_dual(branch_counter,ru_p[i],d_ru_p[i]),ru_mul(dt,ru_sub(ru_dual(branch_counter,tend_ru[i],d_tend_ru[i]),pgrad)));
            RuDual avg=ru_add(ru_dual(branch_counter,ru_avg[i],d_ru_avg[i]),u);
            ru_check(failures,e,13,i,u.v);ru_check(failures,e,13,i,u.d);
            ru_check(failures,e,14,i,avg.v);ru_check(failures,e,14,i,avg.d);
            d_ru_p[i]=u.d;d_ru_avg[i]=avg.d;
        }
    }
    if(branch_record)for(int i=0;i<38;++i)if(branch_local[i]) atomicAdd(branch_record+i,branch_local[i]);
}

// Eight per-edge f64 planes hold c0/c1 contributions. Four f64 scalar
// planes are gathered in edge order, with one rounding at each f32 output.
__device__ __forceinline__ void ru_reverse_edge(
    const int e,const int nlev,const int nedges,const int ncells,const int small_step,
    const float dts,const float gravity,const float rgas,const float cp,
    const int *cells_on_edge,const float *inv_dc_edge,const float *zz,const float *exner,
    const float *cqu,const float *zxu,const float *tend_ru,const float *rho_pp,
    const float *rtheta_pp,float *ru_p,float *ru_avg,
    float *b_inv_dc_edge,float *b_zz,float *b_exner,float *b_cqu,float *b_zxu,
    float *b_tend_ru,float *b_rho_pp,float *b_rtheta_pp,float *b_ru_p,float *b_ru_avg,
    float *seed_ru_p,float *seed_ru_avg,
    double *work_cell,double *work_scalar,AdFailureRecord *failures)
{
    (void)b_zz;(void)b_exner;(void)b_rho_pp;(void)b_rtheta_pp;
    const int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
    if(c0<0||c0>=ncells||c1<0||c1>=ncells){ru_fail(failures,e,5,e,4);return;}
    ru_check(failures,e,1,0,dts);ru_check(failures,e,2,0,gravity);
    ru_check(failures,e,3,0,rgas);ru_check(failures,e,4,0,cp);
    const float cv=mpas_sub(cp,rgas);
    if(mpas_f32_magnitude_bits(cv)==0u)ru_fail(failures,e,4,0,2);
    const float rcv=mpas_div(rgas,cv),csq=mpas_mul(cp,rcv);
    const unsigned long long plane=(unsigned long long)nlev*nedges;
    double bdts=0.0,bg=0.0,bcsq=0.0,binv=mpas_f32_to_f64_ieee(b_inv_dc_edge[e]);
    if(small_step!=1)ru_check(failures,e,5,e,(float)binv);
    for(int k=0;k<nlev;++k){
        const int i=RE(k,e,nedges),i0=RC(k,c0,ncells),i1=RC(k,c1,ncells);
        ru_check(failures,e,10,i,tend_ru[i]);ru_check(failures,e,10,i,b_tend_ru[i]);
        ru_check(failures,e,13,i,b_ru_p[i]);ru_check(failures,e,14,i,b_ru_avg[i]);
        ru_check(failures,e,13,i,seed_ru_p[i]);ru_check(failures,e,14,i,seed_ru_avg[i]);
        const double bu=mpas_f32_to_f64_ieee(seed_ru_p[i])+mpas_f32_to_f64_ieee(seed_ru_avg[i]);
        b_tend_ru[i]=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(b_tend_ru[i])+bu*mpas_f32_to_f64_ieee(dts));
        if(small_step==1){
            bdts+=bu*mpas_f32_to_f64_ieee(tend_ru[i]);
            if(b_ru_p==seed_ru_p)b_ru_p[i]=0.0f;
            if(b_ru_avg==seed_ru_avg)b_ru_avg[i]=0.0f;
        }else{
            ru_check(failures,e,5,e,inv_dc_edge[e]);
            ru_check(failures,e,6,i0,zz[i0]);ru_check(failures,e,6,i1,zz[i1]);
            ru_check(failures,e,7,i0,exner[i0]);ru_check(failures,e,7,i1,exner[i1]);
            ru_check(failures,e,8,i,cqu[i]);ru_check(failures,e,8,i,b_cqu[i]);
            ru_check(failures,e,9,i,zxu[i]);ru_check(failures,e,9,i,b_zxu[i]);
            ru_check(failures,e,11,i0,rho_pp[i0]);ru_check(failures,e,11,i1,rho_pp[i1]);
            ru_check(failures,e,12,i0,rtheta_pp[i0]);ru_check(failures,e,12,i1,rtheta_pp[i1]);
            ru_check(failures,e,13,i,ru_p[i]);ru_check(failures,e,14,i,ru_avg[i]);
            const float diff=mpas_sub(rtheta_pp[i1],rtheta_pp[i0]);
            const float normal=mpas_mul(diff,inv_dc_edge[e]);
            const float mean=mpas_mul(0.5f,mpas_add(zz[i1],zz[i0]));
            if(mpas_f32_magnitude_bits(mean)==0u)ru_fail(failures,e,6,i0,3);
            const float normalized=mpas_div(normal,mean);
            const float p0=mpas_mul(cqu[i],0.5f),p1=mpas_mul(p0,csq);
            const float exsum=mpas_add(exner[i0],exner[i1]);
            const float p2=mpas_mul(p1,exsum),p3=mpas_mul(p2,normalized);
            const float t0=mpas_mul(0.5f,zxu[i]),t1=mpas_mul(t0,gravity);
            const float rsum=mpas_add(rho_pp[i0],rho_pp[i1]),terrain=mpas_mul(t1,rsum);
            const float inner=mpas_sub(tend_ru[i],mpas_add(p3,terrain));
            bdts+=bu*mpas_f32_to_f64_ieee(inner);
            const double bp=-bu*mpas_f32_to_f64_ieee(dts);
            const double bt1=bp*mpas_f32_to_f64_ieee(rsum),brsum=bp*mpas_f32_to_f64_ieee(t1);
            b_zxu[i]=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(b_zxu[i])+bt1*mpas_f32_to_f64_ieee(gravity)*0.5);
            bg+=bt1*mpas_f32_to_f64_ieee(t0);
            const double bp2=bp*mpas_f32_to_f64_ieee(normalized),bnorm=bp*mpas_f32_to_f64_ieee(p2);
            const double bexsum=bp2*mpas_f32_to_f64_ieee(p1),bp1=bp2*mpas_f32_to_f64_ieee(exsum);
            bcsq+=bp1*mpas_f32_to_f64_ieee(p0);
            b_cqu[i]=mpas_f64_to_f32_ieee(mpas_f32_to_f64_ieee(b_cqu[i])+bp1*mpas_f32_to_f64_ieee(csq)*0.5);
            const double bnormal=bnorm/mpas_f32_to_f64_ieee(mean);
            const double bmean=-bnorm*mpas_f32_to_f64_ieee(normalized)/mpas_f32_to_f64_ieee(mean);
            binv+=bnormal*mpas_f32_to_f64_ieee(diff);
            const double brt=bnormal*mpas_f32_to_f64_ieee(inv_dc_edge[e]),bzz=0.5*bmean;
            const unsigned long long q=(unsigned long long)i;
            work_cell[0*plane+q]=bzz;work_cell[1*plane+q]=bzz;
            work_cell[2*plane+q]=bexsum;work_cell[3*plane+q]=bexsum;
            work_cell[4*plane+q]=brsum;work_cell[5*plane+q]=brsum;
            work_cell[6*plane+q]=-brt;work_cell[7*plane+q]=brt;
            const double initial_p=(b_ru_p==seed_ru_p)?0.0:mpas_f32_to_f64_ieee(b_ru_p[i]);
            const double initial_avg=(b_ru_avg==seed_ru_avg)?0.0:mpas_f32_to_f64_ieee(b_ru_avg[i]);
            const double seed_avg=mpas_f32_to_f64_ieee(seed_ru_avg[i]);
            b_ru_p[i]=mpas_f64_to_f32_ieee(initial_p+bu);
            b_ru_avg[i]=mpas_f64_to_f32_ieee(initial_avg+seed_avg);
            ru_check(failures,e,8,i,b_cqu[i]);ru_check(failures,e,9,i,b_zxu[i]);
            if(!isfinite(bzz)||!isfinite(bexsum)||!isfinite(brsum)||!isfinite(brt))ru_fail(failures,e,0,i,1);
        }
        ru_check(failures,e,10,i,b_tend_ru[i]);ru_check(failures,e,13,i,b_ru_p[i]);
        // Both outputs were overwritten on every acoustic substep.
        seed_ru_p[i]=0.0f;seed_ru_avg[i]=0.0f;
    }
    if(small_step!=1){b_inv_dc_edge[e]=mpas_f64_to_f32_ieee(binv);ru_check(failures,e,5,e,b_inv_dc_edge[e]);}
    const double brcv=bcsq*mpas_f32_to_f64_ieee(cp);
    const double bcv=-brcv*mpas_f32_to_f64_ieee(rcv)/mpas_f32_to_f64_ieee(cv);
    const double brg=brcv/mpas_f32_to_f64_ieee(cv)-bcv;
    const double bcp=bcsq*mpas_f32_to_f64_ieee(rcv)+bcv;
    work_scalar[e]=bdts;work_scalar[nedges+e]=bg;
    work_scalar[2*nedges+e]=brg;work_scalar[3*nedges+e]=bcp;
    if(!isfinite(bdts)||!isfinite(bg)||!isfinite(brg)||!isfinite(bcp))ru_fail(failures,e,0,e,1);
}

__device__ __forceinline__ void ru_reduce_cell(
    const int cell,const int nlev,const int nedges,const int ncells,const int small_step,
    const int *cells_on_edge,double *work_cell,const double *work_scalar,
    float *b_zz,float *b_exner,float *b_rho_pp,float *b_rtheta_pp,
    float *b_dts,float *b_gravity,float *b_rgas,float *b_cp,AdFailureRecord *failures)
{
    const unsigned long long plane=(unsigned long long)nlev*nedges;
    if(small_step!=1){
        const unsigned long long span=(unsigned long long)nlev*ncells;
        double *sum=work_cell+8*plane;
        for(int k=0;k<nlev;++k){int i=RC(k,cell,ncells);
            sum[i]=mpas_f32_to_f64_ieee(b_zz[i]);sum[span+i]=mpas_f32_to_f64_ieee(b_exner[i]);sum[2*span+i]=mpas_f32_to_f64_ieee(b_rho_pp[i]);sum[3*span+i]=mpas_f32_to_f64_ieee(b_rtheta_pp[i]);}
        for(int e=0;e<nedges;++e){
            const int c0=cells_on_edge[2*e],c1=cells_on_edge[2*e+1];
            if(c0!=cell&&c1!=cell)continue;
            for(int k=0;k<nlev;++k){int i=RC(k,cell,ncells),q=RE(k,e,nedges);
                if(c0==cell){sum[i]+=work_cell[0*plane+q];sum[span+i]+=work_cell[2*plane+q];sum[2*span+i]+=work_cell[4*plane+q];sum[3*span+i]+=work_cell[6*plane+q];}
                if(c1==cell){sum[i]+=work_cell[1*plane+q];sum[span+i]+=work_cell[3*plane+q];sum[2*span+i]+=work_cell[5*plane+q];sum[3*span+i]+=work_cell[7*plane+q];}
            }
        }
        for(int k=0;k<nlev;++k){int i=RC(k,cell,ncells);
            b_zz[i]=mpas_f64_to_f32_ieee(sum[i]);b_exner[i]=mpas_f64_to_f32_ieee(sum[span+i]);
            b_rho_pp[i]=mpas_f64_to_f32_ieee(sum[2*span+i]);b_rtheta_pp[i]=mpas_f64_to_f32_ieee(sum[3*span+i]);
            ru_check(failures,nedges+cell,6,i,b_zz[i]);ru_check(failures,nedges+cell,7,i,b_exner[i]);
            ru_check(failures,nedges+cell,11,i,b_rho_pp[i]);ru_check(failures,nedges+cell,12,i,b_rtheta_pp[i]);
        }
    }
    if(cell==0){
        double s0=mpas_f32_to_f64_ieee(*b_dts),s1=mpas_f32_to_f64_ieee(*b_gravity),s2=mpas_f32_to_f64_ieee(*b_rgas),s3=mpas_f32_to_f64_ieee(*b_cp);
        for(int e=0;e<nedges;++e){s0+=work_scalar[e];s1+=work_scalar[nedges+e];s2+=work_scalar[2*nedges+e];s3+=work_scalar[3*nedges+e];}
        *b_dts=mpas_f64_to_f32_ieee(s0);*b_gravity=mpas_f64_to_f32_ieee(s1);
        *b_rgas=mpas_f64_to_f32_ieee(s2);*b_cp=mpas_f64_to_f32_ieee(s3);
        ru_check(failures,nedges,1,0,*b_dts);ru_check(failures,nedges,2,0,*b_gravity);
        ru_check(failures,nedges,3,0,*b_rgas);ru_check(failures,nedges,4,0,*b_cp);
    }
}

extern "C" __global__ void acoustic_ru_v841__ad(
    const int nlev,const int nedges,const int ncells,const int small_step,
    const float dts,const float gravity,const float rgas,const float cp,
    const int *cells_on_edge,const float *inv_dc_edge,const float *zz,const float *exner,
    const float *cqu,const float *zxu,const float *tend_ru,const float *rho_pp,
    const float *rtheta_pp,float *ru_p,float *ru_avg,
    float *b_dts,float *b_gravity,float *b_rgas,float *b_cp,
    float *b_inv_dc_edge,float *b_zz,float *b_exner,float *b_cqu,float *b_zxu,
    float *b_tend_ru,float *b_rho_pp,float *b_rtheta_pp,float *b_ru_p,float *b_ru_avg,
    float *seed_ru_p,float *seed_ru_avg,
    double *work_cell,double *work_scalar,AdFailureRecord *failures)
{
    if(gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||blockDim.y!=1||blockDim.z!=1){
        if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)
            ru_fail(failures,0,1,0,5);
        return;
    }
    for(int e=threadIdx.x;e<nedges;e+=blockDim.x)
        ru_reverse_edge(e,nlev,nedges,ncells,small_step,dts,gravity,rgas,cp,
            cells_on_edge,inv_dc_edge,zz,exner,cqu,zxu,tend_ru,rho_pp,rtheta_pp,ru_p,ru_avg,
            b_inv_dc_edge,b_zz,b_exner,b_cqu,b_zxu,b_tend_ru,b_rho_pp,b_rtheta_pp,b_ru_p,b_ru_avg,
            seed_ru_p,seed_ru_avg,work_cell,work_scalar,failures);
    __syncthreads();
    for(int cell=threadIdx.x;cell<ncells;cell+=blockDim.x)
        ru_reduce_cell(cell,nlev,nedges,ncells,small_step,cells_on_edge,work_cell,work_scalar,
            b_zz,b_exner,b_rho_pp,b_rtheta_pp,b_dts,b_gravity,b_rgas,b_cp,failures);
}
