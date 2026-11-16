// SPDX-License-Identifier: Apache-2.0
// P6 (ad/phys): whole-launch construct kernel for the RUC sea-ice parent
// (oracle_ice_ruc_ice) derivative entries. Appended after the 11-02
// composite source (forward math prelude + seaice derivative + launch
// adapters). Unlike the 11-02 harness construct, the forward's own launch
// flags are passed in (cold start, frpcpn, graupel present, MYJ, isice), so
// any step of the captured forecast can be differentiated, not only step 1.
// No field arithmetic here: it wires per-column inputs and tape storage.
extern "C" __global__ void p6_construct(const RucRow *rows,const float *table,int n,
 SiColumnNode *pn,SiColumnLeaf *pl,SiColumnNode *inn,SiColumnLeaf *il,SiColumnNode *cn,SiColumnLeaf *cl,
 int *map,int *cmap,int *layer,float *flayer,int *rec,int *counts,SiRucIceInput *input,SiRucIceTape *tape,
 int isice,int cold_start,int frpcpn,int has_graupel,int myj) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const RucRow &r=rows[c];
 SiRucIceInput x={r.x,r.tso,r.zs,table,nullptr,nullptr,nullptr,9,5001,0,0,isice,
   cold_start!=0,frpcpn!=0,has_graupel!=0,myj!=0,0.f,0.f};
 x.bare_record=rec+c*64;x.snow_record=x.bare_record+32;x.bare_record_count=counts+c*2;
 x.snow_record_count=x.bare_record_count+1;x.bare_record_capacity=32;x.snow_record_capacity=32;
 x.bare_record_count[0]=-1;x.snow_record_count[0]=-1;
 SiRucIceTape t={};t.graph.node=pn+c*4096;t.graph.leaf=pl+c*2048;t.graph.capacity=4096;t.graph.leaf_capacity=2048;
 int *q=layer+c*145;t.tso=q;q+=9;t.zmain=q;q+=9;t.zhalf=q;q+=9;t.dz=q;q+=14;t.map=map+c*4096;
 float *f=flayer+c*59;t.tso_value=f;f+=9;t.zmain_value=f;f+=9;t.zhalf_value=f;f+=9;t.dz_value=f;f+=14;
 SiIceTmpTape &i=t.ice;i.graph.node=inn+c*4096;i.graph.leaf=il+c*1024;i.graph.capacity=4096;i.graph.leaf_capacity=1024;
 i.child.node=cn+c*2048;i.child.leaf=cl+c*512;i.child.capacity=2048;i.child.leaf_capacity=512;
 i.tso=q;q+=9;i.tice=q;q+=9;i.cap=q;q+=9;i.diff=q;q+=9;i.zmain=q;q+=9;i.zhalf=q;q+=9;i.dz=q;q+=14;
 i.bare_tso=q;q+=9;i.child_c=q;q+=9;i.child_r=q;q+=9;i.child_tso=q;i.child_map=cmap+c*2048;
 i.cap_value=f;f+=9;i.diff_value=f;input[c]=x;tape[c]=t;
}
// Graph sizes actually used per column, for the tape-bytes receipt.
extern "C" __global__ void p6_tape_counts(const SiRucIceTape *t,int n,int *out) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 out[c*6+0]=t[c].graph.count;out[c*6+1]=t[c].graph.nleaf;
 out[c*6+2]=t[c].ice.graph.count;out[c*6+3]=t[c].ice.graph.nleaf;
 out[c*6+4]=t[c].ice.child.count;out[c*6+5]=t[c].ice.child.nleaf;
}
