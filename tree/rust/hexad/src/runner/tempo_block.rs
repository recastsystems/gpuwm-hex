// SPDX-License-Identifier: Apache-2.0
//! Racer C ABI2 bounded GPU stage composition. Qualification is external.
//! This adapter preserves the existing forward/TL route. A selected native
//! image either executes every required stage or returns a defect.
use super::*;
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Ctx { dt:f32, idt:f32, flags:[i32;13] }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Rates { blob:u64, ctx:Ctx, nz:i32, columns:i32, x:u64,yb:u64,rb:u64,xb:u64,scratch:u64,status:u64 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Column { nz:i32,columns:i32,ncols:i32,presence:i32,dt:f32,first:i32,blob:u64,f:[u64;18],
 scratch:u64,reference:u64,rates:u64,out:u64,status:u64,faults:u64,
 aero:i32,hail:i32,melt:i32,radii:i32,substep:i32,steps:i32,packed:i32,column_offset:i32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Driver { column:u64,steps:i32,substep:i32,offset:i32,total_columns:i32,field_mask:u32,scalar_mask:u32,
 f:[u64;18],scalars:[u64;10],launch:u64,checkpoints:u64,masks:u64,yout:u64,ybar:u64,xbar:u64,carry:u64,scalar_bar:u64 }
pub(super) struct BlockNative { functions:HashMap<&'static str,*mut c_void>,sizes:[usize;5], rate_sizes:[usize;5],sha:String,t4_slices:u32,fwd_threads:u32,
 /// megakernel: mk_tempo_rev (one launch per tile for the whole reverse), selected by AD_TEMPO_MEGA=1.
 mega:Option<*mut c_void>,
 /// global-checkpoint images: device address of mk_ckpt_base (T8 checkpoints in global memory, 34,816 B per column).
 ckpt_global:Option<u64>,
 /// tempo: AD_TEMPO_KTIME=1 records a CUDA event after every launch and prints per-stage kernel ms per apply
 /// (diagnostic runs only: one event wait per apply).
 kt_on:bool, kt:std::cell::RefCell<Vec<(&'static str,*mut c_void)>>,
 /// tempo: images that export td_init and td_chain keep the stage checkpoints of EVERY substep of a tile (one scratch per
 /// substep), so the primal is recomputed once per substep (tf_forward) instead of twice (td_prepare + tf_forward).
 /// AD_TEMPO_BLOCK_STOREALL=0 keeps the td_prepare path. Not used with identity (td_prepare owns the compared y) or the megakernel.
 chain:bool,
 /// tempo: AD_TEMPO_BLOCK_REV_CUBIN=<image built from the SAME source> supplies the 14 reverse-only stage kernels (for a
 /// fast-math build of the derivative arithmetic); the primal recompute (td_prepare, tf_forward) stays in the strict image.
 rev_sha:Option<String>, ckpt_global2:Option<u64>,
 /// tempo, diagnostic images only (a histogram build): device address of f52_hist[2][513], the count of sedimentation
 /// sub-steps N per (column, TEMPO substep) for rain and graupel; printed and cleared after every apply.
 hist:Option<u64>,
 /// tempo: stream mode. AD_TEMPO_BLOCK_STREAMS=M (default 8; 1 = the sequential tile path): the arena's columns are cut
 /// into M slots that run their whole launch sequence at once, each on its own non-blocking stream. The kernels and the
 /// words are the same; only the idle tail of every launch is filled. Not used with identity, the megakernel or images
 /// with global T8 checkpoints (one mk_ckpt_base per image).
 streams:Vec<*mut c_void> }
#[derive(Clone,Copy)]
struct Slot{r:Rates,a:Column,d:Driver,rp:u64,aarr:u64,darr:u64,staging:u64}
/// tempo: AD_TEMPO_BLOCK_DUMP=<dir> writes the first reverse apply's launch inputs (f*.bin, s*.bin, blob.bin), its incoming
/// adjoint (ybar.bin) and its result (xbar.bin) for the off-box kernel harness (diagnostic runs only).
static DUMPED:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
const RATE_NAMES:[&str;5]=["tc_rates_process","tc_rates_t6","tc_rates_t5","tc_rates_t4","tc_rates_t3"];
const REVERSE:[&str;14]=["tf_diag","tf_final","tf_t9","tf_graupel","tf_rain","tf_t7","tf_t2_rates",
 "tc_rates_process","tc_rates_t6","tc_rates_t5","tc_rates_t4","tc_rates_t3","tf_initial","tf_aerosol_prefix"];
impl BlockNative {
 pub(super) unsafe fn load(cuda:&Cuda,path:&str)->Result<Self,String>{
  let mut image=std::fs::read(path).map_err(|e|format!("DEFECT TEMPO block image: {e}"))?;
  let sha={use sha2::Digest;format!("{:x}",sha2::Sha256::digest(&image))};image.push(0);
  let mut module=ptr::null_mut();checked("TEMPO block module",(cuda.module_load)(&mut module,image.as_ptr().cast()))?;
  let mut functions=HashMap::new();
  for name in REVERSE.iter().copied().chain(["td_sizes","tf_sizes","tc_rates_sizes","td_prepare","td_load","tf_forward","td_seed","td_seed_column","td_save"]){
   let mut f=ptr::null_mut();let s=CString::new(name).unwrap();checked(name,(cuda.module_function)(&mut f,module,s.as_ptr()))?;functions.insert(name,f);
  }
  let mega=if std::env::var("AD_TEMPO_MEGA").map(|v|v!="0").unwrap_or(false){let mut f=ptr::null_mut();let s=CString::new("mk_tempo_rev").unwrap();
   checked("DEFECT AD_TEMPO_MEGA=1 but image has no mk_tempo_rev",(cuda.module_function)(&mut f,module,s.as_ptr()))?;Some(f)}else{None};
  let ckpt_global={let (mut g,mut gn)=(0u64,0usize);let c=CString::new("mk_ckpt_base").unwrap();if (cuda.module_global)(&mut g,&mut gn,module,c.as_ptr())==0&&gn==8{Some(g)}else{None}};
  let mut this=Self{functions,sizes:[0;5],rate_sizes:[0;5],sha,t4_slices:23,fwd_threads:64,mega,ckpt_global,kt_on:false,kt:std::cell::RefCell::new(vec![]),chain:false,rev_sha:None,ckpt_global2:None,hist:None,streams:vec![]};let mut sz=0;
  checked("TEMPO block size buffer",(cuda.allocate)(&mut sz,40))?;
  this.launch(cuda,"tf_sizes",1,sz)?;dev::sync(cuda)?;
  let v=get::<u64>(cuda,sz,5)?;for (i,x) in v.iter().enumerate(){this.sizes[i]=*x as usize;}
  this.launch(cuda,"tc_rates_sizes",1,sz)?;dev::sync(cuda)?;
  let v=get::<u64>(cuda,sz,5)?;for (i,x) in v.iter().enumerate(){this.rate_sizes[i]=*x as usize;}
  // Optional: images with K-direction T4 replays export their slice count; older images use 23 slices.
  {let mut f=ptr::null_mut();let s=CString::new("tc_rates_t4_meta").unwrap();
   if (cuda.module_function)(&mut f,module,s.as_ptr())==0{this.functions.insert("tc_rates_t4_meta",f);this.launch(cuda,"tc_rates_t4_meta",1,sz)?;dev::sync(cuda)?;
    dev::zero(cuda,sz,40)?;dev::sync(cuda)?;this.launch(cuda,"tc_rates_t4_meta",1,sz)?;dev::sync(cuda)?;let v=get::<u64>(cuda,sz,3)?;if v[0]==0||v[0]>23{return Err(format!("DEFECT TEMPO block t4 slices {}",v[0]));}this.t4_slices=v[0] as u32;
    if v[2]==32{this.fwd_threads=32;}}}
  {let mut have=0;for name in ["td_init","td_chain"]{let mut f=ptr::null_mut();let s=CString::new(name).unwrap();
    if (cuda.module_function)(&mut f,module,s.as_ptr())==0{this.functions.insert(name,f);have+=1;}}
   this.chain=have==2&&std::env::var("AD_TEMPO_BLOCK_STOREALL").as_deref()!=Ok("0");}
  {let m=match std::env::var("AD_TEMPO_BLOCK_STREAMS"){Ok(v)=>v.parse::<usize>().map_err(|_|"DEFECT AD_TEMPO_BLOCK_STREAMS must be an integer")?,Err(_)=>8};
   if m>64{return Err("DEFECT AD_TEMPO_BLOCK_STREAMS above 64".into());}
   if m>1&&this.mega.is_none()&&this.ckpt_global.is_none(){for _ in 0..m{let mut st=ptr::null_mut();checked("TEMPO block stream",(cuda.stream_create)(&mut st,1))?;this.streams.push(st);}}}
  this.hist={let (mut g,mut gn)=(0u64,0usize);let c=CString::new("f52_hist").unwrap();if (cuda.module_global)(&mut g,&mut gn,module,c.as_ptr())==0&&gn==2*513*8{Some(g)}else{None}};
  if let Ok(p2)=std::env::var("AD_TEMPO_BLOCK_REV_CUBIN"){
   let mut im2=std::fs::read(&p2).map_err(|e|format!("DEFECT TEMPO block reverse image: {e}"))?;
   this.rev_sha=Some({use sha2::Digest;format!("{:x}",sha2::Sha256::digest(&im2))});im2.push(0);
   let mut m2=ptr::null_mut();checked("TEMPO block reverse module",(cuda.module_load)(&mut m2,im2.as_ptr().cast()))?;
   for name in REVERSE{let mut f=ptr::null_mut();let s=CString::new(name).unwrap();checked(name,(cuda.module_function)(&mut f,m2,s.as_ptr()))?;this.functions.insert(name,f);}
   this.ckpt_global2={let (mut g,mut gn)=(0u64,0usize);let c=CString::new("mk_ckpt_base").unwrap();if (cuda.module_global)(&mut g,&mut gn,m2,c.as_ptr())==0&&gn==8{Some(g)}else{None}};
   if this.ckpt_global.is_some()!=this.ckpt_global2.is_some(){return Err("DEFECT TEMPO block reverse image: checkpoint placement differs from the primary image".into());}
  }
  this.launch(cuda,"td_sizes",1,sz)?;dev::sync(cuda)?;let v=get::<u64>(cuda,sz,2)?;
  dev::zero(cuda,sz,40)?;dev::sync(cuda)?;checked("TEMPO block free size buffer",(cuda.free)(sz))?;
  if this.sizes[2]!=std::mem::size_of::<Column>()||this.rate_sizes[4]!=std::mem::size_of::<Rates>()||v[0] as usize!=std::mem::size_of::<Driver>()||v[1] as usize!=std::mem::size_of::<Column>()||(this.ckpt_global.is_none()&&this.sizes[4]>46*1024) {
   // the 46 KiB bound is the dynamic shared-memory budget; images with global checkpoints (mk_ckpt_base) are not bound by it
   return Err("DEFECT TEMPO block ABI2 size/shared-memory mismatch".into());
  }
  eprintln!("{}",json!({"event":"tempo_block_load","image_sha256":this.sha,"abi":2,"composition":"bounded multikernel","column_bytes":this.sizes[0],"rate_level_bytes":this.rate_sizes[3],"t4_slices":this.t4_slices,"fwd_threads":this.fwd_threads,"shared_bytes":this.sizes[4],"store_all":this.chain,"ckpt_global":this.ckpt_global.is_some(),"reverse_image_sha256":this.rev_sha,"streams":this.streams.len().max(1)}));this.kt_on=std::env::var("AD_TEMPO_KTIME").as_deref()==Ok("1");Ok(this)
 }
 unsafe fn launch(&self,cuda:&Cuda,name:&str,n:usize,ap:u64)->Result<(),String>{
  let mut arg=ap;let mut args=[(&mut arg as *mut u64).cast::<c_void>()];let f=*self.functions.get(name).ok_or_else(||format!("DEFECT missing TEMPO stage {name}"))?;
  let gy=if name=="tc_rates_t4"{self.t4_slices}else if name=="tf_diag"{15}else{1};
  let shared=if self.ckpt_global.is_none()&&(name=="tf_rain"||name=="tf_graupel"){self.sizes[4] as u32}else{0};
  let bx=if name=="td_prepare"||name=="tf_forward"{self.fwd_threads}else{64};
  checked(name,(cuda.launch)(f,n as u32,gy,1,bx,1,1,shared,ptr::null_mut(),args.as_mut_ptr(),ptr::null_mut()))?;
  let key=self.functions.get_key_value(name).map(|(k,_)|*k).unwrap_or("other");self.mark(cuda,key)
 }
 unsafe fn launch_on(&self,cuda:&Cuda,name:&str,n:usize,ap:u64,stream:*mut c_void)->Result<(),String>{
  let mut arg=ap;let mut args=[(&mut arg as *mut u64).cast::<c_void>()];let f=*self.functions.get(name).ok_or_else(||format!("DEFECT missing TEMPO stage {name}"))?;
  let gy=if name=="tc_rates_t4"{self.t4_slices}else if name=="tf_diag"{15}else{1};
  let shared=if self.ckpt_global.is_none()&&(name=="tf_rain"||name=="tf_graupel"){self.sizes[4] as u32}else{0};
  let bx=if name=="td_prepare"||name=="tf_forward"{self.fwd_threads}else{64};
  checked(name,(cuda.launch)(f,n as u32,gy,1,bx,1,1,shared,stream,args.as_mut_ptr(),ptr::null_mut()))
 }
 unsafe fn mark(&self,cuda:&Cuda,name:&'static str)->Result<(),String>{
  if !self.kt_on{return Ok(());}
  let mut e=ptr::null_mut();checked("cuEventCreate ktime",(cuda.event_create)(&mut e,0))?;
  checked("cuEventRecord ktime",(cuda.event_record)(e,ptr::null_mut()))?;self.kt.borrow_mut().push((name,e));Ok(())
 }
 unsafe fn kt_report(&self,cuda:&Cuda,columns:usize,substeps:usize,tiles:usize)->Result<(),String>{
  let mut ev=self.kt.borrow_mut();if !self.kt_on||ev.len()<2{return Ok(());}
  checked("cuEventSynchronize ktime",(cuda.event_synchronize)(ev.last().unwrap().1))?;
  let mut ms:std::collections::BTreeMap<&'static str,(f64,u64)>=Default::default();let mut total=0f64;
  for w in ev.windows(2){let mut t=0f32;checked("cuEventElapsedTime",(cuda.event_elapsed)(&mut t,w[0].1,w[1].1))?;let x=ms.entry(w[1].0).or_insert((0.0,0));x.0+=t as f64;x.1+=1;total+=t as f64;}
  for (_,e) in ev.iter(){(cuda.event_destroy)(*e);}ev.clear();
  let m:serde_json::Map<String,Value>=ms.iter().map(|(k,v)|(k.to_string(),json!([(v.0*1e3).round()/1e3,v.1]))).collect();
  eprintln!("{}",json!({"event":"tempo_block_ktime","columns":columns,"substeps":substeps,"tiles":tiles,"total_ms":total,"ms_launches_by_stage":m}));Ok(())
 }
 #[allow(clippy::too_many_arguments)]
 unsafe fn apply_streams(&self,cuda:&Cuda,arena:&mut Arena,s:&Snap,vd:u64,rd:u64,led:&mut dev::Ledger,storeall:bool,nq:usize,nin:usize,per:usize,cap:usize)->Result<(Option<u64>,usize,usize),String>{
  let (csz,dsz)=(std::mem::size_of::<Column>(),std::mem::size_of::<Driver>());
  let fixed=|m:usize|arena.used+(4<<20)+m*(40*256+s.steps*(csz+dsz)+std::mem::size_of::<Rates>());
  // columns in flight = what the arena holds; slots of at least 64 columns (fewer slots on a small arena)
  let mut m=self.streams.len();let mut nt=(arena.bytes.saturating_sub(fixed(m))/per).min(s.columns).min(cap);
  while m>1&&nt/m<64{m-=1;nt=(arena.bytes.saturating_sub(fixed(m))/per).min(s.columns).min(cap);}
  if nt==0{return Err(format!("DEFECT TEMPO block tile requires {per} bytes per column, arena {} used {}",arena.bytes,arena.used));}
  let ns=(nt/m).max(1);
  let mut slots:Vec<Slot>=Vec::with_capacity(m);
  for _ in 0..m{
   let mut r=Rates::default();r.blob=s.blob;r.ctx=Ctx{dt:s.dt,idt:1.0/s.dt,flags:[s.flags[0],s.flags[1],0,s.flags[3],0,1,1,s.flags[6],s.flags[6],s.flags[6],1,0,0]};r.nz=s.nz as i32;
   r.x=arena.take(self.rate_sizes[0]*64*ns)?;r.yb=arena.take(self.rate_sizes[1]*64*ns)?;r.rb=arena.take(self.rate_sizes[2]*64*ns)?;r.xb=arena.take(self.rate_sizes[1]*64*ns)?;r.scratch=arena.take(self.rate_sizes[3]*64*ns)?;r.status=arena.take(4*ns)?;
   let rp=arena.take(std::mem::size_of::<Rates>())?;let aarr=arena.take(s.steps*csz)?;let darr=arena.take(s.steps*dsz)?;let staging=arena.take(18*s.nz*ns*4)?;
   let mut a=Column::default();a.nz=s.nz as i32;a.presence=s.flags[5];a.dt=s.dt;a.first=s.flags[4];a.blob=s.blob;a.scratch=arena.take(self.sizes[0]*ns*nq)?;a.rates=rp;a.out=arena.take(24*ns)?;a.status=arena.take(16)?;a.faults=arena.take(4*ns)?;a.aero=s.flags[0];a.hail=s.flags[1];a.melt=s.flags[3];a.radii=s.flags[6];a.steps=s.steps as i32;
   let mut d=Driver::default();d.steps=s.steps as i32;d.total_columns=s.columns as i32;d.f=s.fsnap;d.scalars=s.ssnap;
   for j in 0..18{d.field_mask|=((s.fsnap[j]!=0) as u32)<<j;}for j in 0..10{d.scalar_mask|=((s.ssnap[j]!=0) as u32)<<j;}
   d.launch=arena.take(nin*ns*4)?;d.checkpoints=arena.take(s.steps*nin*ns*4)?;d.masks=arena.take(s.steps*nin*ns)?;d.yout=arena.take(nin*ns*4)?;d.ybar=arena.take(nin*ns*8)?;d.xbar=arena.take(nin*ns*8)?;d.carry=arena.take(18*s.nz*ns*8)?;d.scalar_bar=arena.take(10*ns*8)?;
   slots.push(Slot{r,a,d,rp,aarr,darr,staging});
  }
  let (mut c0,mut tiles,mut groups)=(0usize,0usize,0usize);self.mark(cuda,"t0")?;
  while c0<s.columns{
   // phase A, dispatch stream: per-substep argument arrays and the incoming adjoint of every slot of this group
   let mut grp:Vec<(usize,usize,usize)>=Vec::with_capacity(m);
   for i in 0..m{if c0>=s.columns{break;}let n=ns.min(s.columns-c0);grp.push((i,c0,n));c0+=n;}
   for &(i,c,n) in &grp{let sl=&mut slots[i];
    sl.r.columns=n as i32;sl.a.columns=n as i32;sl.a.ncols=n as i32;sl.a.column_offset=c as i32;sl.d.offset=c as i32;
    for j in 0..18{sl.a.f[j]=sl.staging+(j*s.nz*n*4) as u64;}
    dev::zero(cuda,sl.r.status,4*n)?;dev::zero(cuda,sl.a.faults,4*n)?;put(cuda,sl.rp,&[sl.r])?;
    let av:Vec<Column>=(0..s.steps).map(|sub|{let mut x=sl.a;x.substep=sub as i32;x.packed=(sub>0) as i32;if storeall{x.scratch=sl.a.scratch+(sub*self.sizes[0]*ns) as u64;}x}).collect();
    let dv:Vec<Driver>=(0..s.steps).map(|sub|{let mut x=sl.d;x.column=sl.aarr+(sub*csz) as u64;x.substep=sub as i32;x}).collect();
    put(cuda,sl.aarr,&av)?;put(cuda,sl.darr,&dv)?;
    led.tile(cuda,vd,sl.d.ybar,nin,s.columns,c,n,false)?;
   }
   dev::sync(cuda)?;self.mark(cuda,"stream_setup")?;   // phase A is complete before the non-blocking streams read it
   // phase B: every slot's whole launch sequence on its own stream
   for &(i,_,n) in &grp{let sl=&slots[i];let st=self.streams[i];
    let ap=|sub:usize|sl.aarr+(sub*csz) as u64;let dp=|sub:usize|sl.darr+(sub*dsz) as u64;
    if storeall{self.launch_on(cuda,"td_init",n,dp(0),st)?;
     for sub in 0..s.steps{self.launch_on(cuda,"td_load",n,dp(sub),st)?;self.launch_on(cuda,"tf_forward",n,ap(sub),st)?;if sub+1<s.steps{self.launch_on(cuda,"td_chain",n,dp(sub),st)?;}}
    }else{self.launch_on(cuda,"td_prepare",n,dp(0),st)?;}
    self.launch_on(cuda,"td_seed",n,dp(0),st)?;
    for sub in (0..s.steps).rev(){
     if !storeall{self.launch_on(cuda,"td_load",n,dp(sub),st)?;self.launch_on(cuda,"tf_forward",n,ap(sub),st)?;}
     self.launch_on(cuda,"td_seed_column",n,dp(sub),st)?;
     for name in REVERSE{self.launch_on(cuda,name,n,if RATE_NAMES.contains(&name){sl.rp}else{ap(sub)},st)?;}
     self.launch_on(cuda,"td_save",n,dp(sub),st)?;
    }
   }
   dev::sync(cuda)?;self.mark(cuda,"stream_group")?;   // phase C: every stream of the group is done
   for &(i,c,n) in &grp{let sl=&slots[i];
    led.first_nonzero(cuda,sl.a.faults,n,c,"DEFECT TEMPO block column fault")?;led.first_nonzero(cuda,sl.r.status,n,c,"DEFECT TEMPO block rate fault")?;
    led.tile(cuda,sl.d.xbar,rd,nin,s.columns,c,n,true)?;tiles+=1;
   }
   groups+=1;
  }
  self.mark(cuda,"led_tile")?;self.kt_report(cuda,s.columns,s.steps,tiles)?;
  eprintln!("{}",json!({"event":"tempo_block_apply","abi":2,"columns":s.columns,"substeps":s.steps,"tile_columns":ns,"tiles":tiles,"streams":m,"groups":groups,"columns_in_flight":ns*m,"bytes_per_column":per,"image_sha256":self.sha,
   "composition":if storeall{"bounded multikernel, store-all chain, streams"}else{"bounded multikernel, streams"}}));Ok((None,0,tiles))
 }
 #[allow(clippy::too_many_arguments)]
 pub(super) unsafe fn apply(&self,cuda:&Cuda,arena:&mut Arena,s:&Snap,vd:u64,rd:u64,identity:bool,led:&mut dev::Ledger)->Result<(Option<u64>,usize,usize),String>{
  // The pinned producer declares TEMPO_NZ_MAX=59; 64 is only the block width.
  if s.nz==0||s.nz>59||s.steps==0{return Err("DEFECT TEMPO block pinned image requires 1..59 levels and positive substeps".into());}
  let dump=std::env::var("AD_TEMPO_BLOCK_DUMP").ok().filter(|_|!DUMPED.swap(true,std::sync::atomic::Ordering::SeqCst));
  if let Some(dir)=&dump{let w=|n:String,b:Vec<u8>|std::fs::write(format!("{dir}/{n}"),b).map_err(|e|format!("DEFECT TEMPO block dump {n}: {e}"));
   std::fs::create_dir_all(dir).map_err(|e|format!("DEFECT TEMPO block dump dir: {e}"))?;
   for j in 0..18{if s.fsnap[j]!=0{w(format!("f{j}.bin"),get::<u8>(cuda,s.fsnap[j],4*s.nz*s.columns)?)?;}}
   for j in 0..10{if s.ssnap[j]!=0{w(format!("s{j}.bin"),get::<u8>(cuda,s.ssnap[j],4*s.columns)?)?;}}
   w("blob.bin".into(),get::<u8>(cuda,s.blob,s.blob_bytes)?)?;w("ybar.bin".into(),get::<u8>(cuda,vd,8*(18*s.nz+10)*s.columns)?)?;
   w("meta.json".into(),json!({"nz":s.nz,"columns":s.columns,"steps":s.steps,"dt_bits":s.dt.to_bits(),"dt":s.dt,"flags":s.flags,"blob_bytes":s.blob_bytes,
    "fields":(0..18).filter(|j|s.fsnap[*j]!=0).collect::<Vec<_>>(),"scalars":(0..10).filter(|j|s.ssnap[*j]!=0).collect::<Vec<_>>(),"image_sha256":self.sha,"identity":identity}).to_string().into_bytes())?;}
  let storeall=self.chain&&!identity&&self.mega.is_none();let nq=if storeall{s.steps}else{1};
  let nin=18*s.nz+10;let per=self.sizes[0]*nq+64*(self.rate_sizes[0]+2*self.rate_sizes[1]+self.rate_sizes[2]+self.rate_sizes[3])+s.steps*nin*5+nin*40+16+if self.ckpt_global.is_some(){self.sizes[4]}else{0};
  let avail=arena.bytes.saturating_sub(arena.used+32*256+(4<<20)+s.steps*(std::mem::size_of::<Column>()+std::mem::size_of::<Driver>()));// Tile width: as many columns as the arena holds (the kernels index columns by blockIdx.x and have no
  // 128-column assumption; run_driver.py passes all 2,562 columns in one tile). The former .min(128) cap
  // left the card mostly idle (128 blocks of 64 threads per launch). AD_TEMPO_BLOCK_TILE caps it for tests.
  let cap=std::env::var("AD_TEMPO_BLOCK_TILE").ok().and_then(|v|v.parse::<usize>().ok()).filter(|v|*v>0).unwrap_or(usize::MAX);
  if self.streams.len()>1&&!identity&&self.mega.is_none()&&self.ckpt_global.is_none(){
   let r=self.apply_streams(cuda,arena,s,vd,rd,led,storeall,nq,nin,per,cap)?;
   if let Some(dir)=&dump{std::fs::write(format!("{dir}/xbar.bin"),get::<u8>(cuda,rd,8*nin*s.columns)?).map_err(|e|format!("DEFECT TEMPO block dump xbar: {e}"))?;}
   return Ok(r);
  }
  let nt=(avail/per).min(s.columns).min(cap);
  if nt==0{return Err(format!("DEFECT TEMPO block tile requires {per} bytes per column, available {avail}"));}
  let mut r=Rates::default();r.blob=s.blob;r.ctx=Ctx{dt:s.dt,idt:1.0/s.dt,flags:[s.flags[0],s.flags[1],0,s.flags[3],0,1,1,s.flags[6],s.flags[6],s.flags[6],1,0,0]};r.nz=s.nz as i32;
  r.x=arena.take(self.rate_sizes[0]*64*nt)?;r.yb=arena.take(self.rate_sizes[1]*64*nt)?;r.rb=arena.take(self.rate_sizes[2]*64*nt)?;r.xb=arena.take(self.rate_sizes[1]*64*nt)?;r.scratch=arena.take(self.rate_sizes[3]*64*nt)?;r.status=arena.take(4*nt)?;
  let rp=arena.take(std::mem::size_of::<Rates>())?;let ap=arena.take(std::mem::size_of::<Column>())?;let dp=arena.take(std::mem::size_of::<Driver>())?;
  if let Some(gl)=self.ckpt_global{let ck=arena.take(self.sizes[4]*nt)?;put(cuda,gl,&[ck])?;if let Some(g2)=self.ckpt_global2{put(cuda,g2,&[ck])?;}}
  let (aarr,darr)=if self.mega.is_some(){(arena.take(s.steps*std::mem::size_of::<Column>())?,arena.take(s.steps*std::mem::size_of::<Driver>())?)}else{(0,0)};
  let staging=arena.take(18*s.nz*nt*4)?;
  let mut a=Column::default();a.nz=s.nz as i32;a.presence=s.flags[5];a.dt=s.dt;a.first=s.flags[4];a.blob=s.blob;a.scratch=arena.take(self.sizes[0]*nt*nq)?;a.rates=rp;let scratch0=a.scratch;a.out=arena.take(24*nt)?;a.status=arena.take(16)?;a.faults=arena.take(4*nt)?;a.aero=s.flags[0];a.hail=s.flags[1];a.melt=s.flags[3];a.radii=s.flags[6];a.steps=s.steps as i32;
  let mut d=Driver::default();d.column=ap;d.steps=s.steps as i32;d.total_columns=s.columns as i32;d.f=s.fsnap;d.scalars=s.ssnap;
  for j in 0..18{d.field_mask|=((s.fsnap[j]!=0) as u32)<<j;}for j in 0..10{d.scalar_mask|=((s.ssnap[j]!=0) as u32)<<j;}
  d.launch=arena.take(nin*nt*4)?;d.checkpoints=arena.take(s.steps*nin*nt*4)?;d.masks=arena.take(s.steps*nin*nt)?;d.yout=arena.take(nin*nt*4)?;d.ybar=arena.take(nin*nt*8)?;d.xbar=arena.take(nin*nt*8)?;d.carry=arena.take(18*s.nz*nt*8)?;d.scalar_bar=arena.take(10*nt*8)?;
  let count=if identity{Some(led.reserve(cuda,8,dev::LedgerItem::Count)?)}else{None};let mut compared=0;let mut tiles=0;self.mark(cuda,"t0")?;
  for c0 in (0..s.columns).step_by(nt){let n=nt.min(s.columns-c0);r.columns=n as i32;a.columns=n as i32;a.ncols=n as i32;a.column_offset=c0 as i32;d.offset=c0 as i32;
   for j in 0..18{a.f[j]=staging+(j*s.nz*n*4) as u64;}
   dev::zero(cuda,r.status,4*n)?;dev::zero(cuda,a.faults,4*n)?;put(cuda,rp,&[r])?;put(cuda,ap,&[a])?;put(cuda,dp,&[d])?;self.mark(cuda,"host_put")?;
   if storeall{
    // forward chain: launch boundary, then per substep load -> checkpointed column into its own scratch -> next boundary
    self.launch(cuda,"td_init",n,dp)?;
    for sub in 0..s.steps{d.substep=sub as i32;a.substep=sub as i32;a.packed=(sub>0) as i32;a.scratch=scratch0+(sub*self.sizes[0]*nt) as u64;put(cuda,dp,&[d])?;put(cuda,ap,&[a])?;self.mark(cuda,"host_put")?;
     self.launch(cuda,"td_load",n,dp)?;self.launch(cuda,"tf_forward",n,ap)?;if sub+1<s.steps{self.launch(cuda,"td_chain",n,dp)?;}}
   }else{self.launch(cuda,"td_prepare",n,dp)?;}
   if let Some((cw,_))=count{
    let mut cmp=|slot:usize,src:u64,rows:usize|->Result<(),String>{if src!=0{compared+=rows*n;for q in 0..rows{led.diff_words(cuda,d.yout+((slot+q)*n*4) as u64,src+((q*s.columns+c0)*4) as u64,n,4,cw)?;}}Ok(())};
    cmp(0,s.fpost[0],s.nz)?;for j in 4..17{cmp((j-3)*s.nz,s.fpost[j],s.nz)?;}for j in 0..10{cmp(14*s.nz+j,s.spost[j],1)?;}for j in 0..4{if j==0||s.flags[6]!=0{cmp(14*s.nz+10+j*s.nz,s.ppost[j],s.nz)?;}}
   }
   self.mark(cuda,"identity_cmp")?;led.tile(cuda,vd,d.ybar,nin,s.columns,c0,n,false)?;self.mark(cuda,"led_tile")?;
   if let Some(mk)=self.mega{
    let av:Vec<Column>=(0..s.steps).map(|sub|{let mut x=a;x.substep=sub as i32;x.packed=(sub>0) as i32;x}).collect();
    let dv:Vec<Driver>=(0..s.steps).map(|sub|{let mut x=d;x.column=aarr+(sub*std::mem::size_of::<Column>()) as u64;x.substep=sub as i32;x}).collect();
    put(cuda,aarr,&av)?;put(cuda,darr,&dv)?;
    let (mut p0,mut p1,mut p2,mut p3)=(darr,aarr,rp,s.steps as i32);
    let mut args=[(&mut p0 as *mut u64).cast::<c_void>(),(&mut p1 as *mut u64).cast::<c_void>(),(&mut p2 as *mut u64).cast::<c_void>(),(&mut p3 as *mut i32).cast::<c_void>()];
    checked("mk_tempo_rev",(cuda.launch)(mk,n as u32,1,1,64,1,1,if self.ckpt_global.is_some(){0}else{self.sizes[4] as u32},ptr::null_mut(),args.as_mut_ptr(),ptr::null_mut()))?;
   }else{self.launch(cuda,"td_seed",n,dp)?;}
   for sub in (0..s.steps).rev(){if self.mega.is_some(){break;}d.substep=sub as i32;a.substep=sub as i32;a.packed=(sub>0) as i32;if storeall{a.scratch=scratch0+(sub*self.sizes[0]*nt) as u64;}put(cuda,dp,&[d])?;put(cuda,ap,&[a])?;self.mark(cuda,"host_put")?;
    if !storeall{self.launch(cuda,"td_load",n,dp)?;self.launch(cuda,"tf_forward",n,ap)?;}self.launch(cuda,"td_seed_column",n,dp)?;
    for name in REVERSE{self.launch(cuda,name,n,if RATE_NAMES.contains(&name){rp}else{ap})?;}
    self.launch(cuda,"td_save",n,dp)?;
   }
   led.first_nonzero(cuda,a.faults,n,c0,"DEFECT TEMPO block column fault")?;led.first_nonzero(cuda,r.status,n,c0,"DEFECT TEMPO block rate fault")?;
   self.mark(cuda,"led_check")?;led.tile(cuda,d.xbar,rd,nin,s.columns,c0,n,true)?;self.mark(cuda,"led_tile")?;tiles+=1;
  }
  self.kt_report(cuda,s.columns,s.steps,tiles)?;
  if let Some(hp)=self.hist{let h=get::<u64>(cuda,hp,2*513)?;dev::zero(cuda,hp,2*513*8)?;
   let edges=[(1usize,4usize),(5,8),(9,16),(17,32),(33,64),(65,128),(129,512)];let mut out=serde_json::Map::new();
   for (i,nm) in ["rain","graupel"].iter().enumerate(){let c=&h[i*513..(i+1)*513];
    let tot:u64=c.iter().sum();let sum_n:u64=c.iter().enumerate().map(|(n,v)|n as u64*v).sum();let max_n=c.iter().rposition(|v|*v>0).unwrap_or(0);
    let b:Vec<Value>=edges.iter().map(|(a,z)|json!({"n":format!("{a}-{z}"),"column_substeps":c[*a..=*z].iter().sum::<u64>(),"sum_n":(*a..=*z).map(|n|n as u64*c[n]).sum::<u64>()})).collect();
    out.insert(nm.to_string(),json!({"column_substeps_with_species":tot,"of":s.columns*s.steps,"sum_n":sum_n,"max_n":max_n,"buckets":b}));}
   eprintln!("{}",json!({"event":"tempo_block_sedimentation_substeps","columns":s.columns,"substeps":s.steps,"species":out}));}
  if let Some(dir)=&dump{std::fs::write(format!("{dir}/xbar.bin"),get::<u8>(cuda,rd,8*nin*s.columns)?).map_err(|e|format!("DEFECT TEMPO block dump xbar: {e}"))?;}
  eprintln!("{}",json!({"event":"tempo_block_apply","abi":2,"columns":s.columns,"substeps":s.steps,"tile_columns":nt,"tiles":tiles,"bytes_per_column":per,"image_sha256":self.sha,"composition":if self.mega.is_some(){"megakernel"}else if storeall{"bounded multikernel, store-all chain"}else{"bounded multikernel"}}));Ok((count.map(|x|x.1),compared,tiles))
 }
}
