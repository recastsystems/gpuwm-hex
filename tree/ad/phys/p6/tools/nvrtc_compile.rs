// SPDX-License-Identifier: Apache-2.0
// CPU-only NVRTC cubin compiler used by the interim proof.
use std::{env,ffi::{c_char,c_int,c_void,CStr,CString},fs,ptr};
#[link(name="dl")]
unsafe extern "C" { fn dlopen(p:*const c_char,f:c_int)->*mut c_void; fn dlsym(h:*mut c_void,n:*const c_char)->*mut c_void; }
type Create=unsafe extern "C" fn(*mut *mut c_void,*const c_char,*const c_char,c_int,*const *const c_char,*const *const c_char)->c_int;
type Compile=unsafe extern "C" fn(*mut c_void,c_int,*const *const c_char)->c_int;
type Size=unsafe extern "C" fn(*mut c_void,*mut usize)->c_int;
type Get=unsafe extern "C" fn(*mut c_void,*mut c_char)->c_int;
type Destroy=unsafe extern "C" fn(*mut *mut c_void)->c_int;
unsafe fn sym<T:Copy>(h:*mut c_void,n:&CStr)->T { let p=unsafe{dlsym(h,n.as_ptr())}; assert!(!p.is_null(),"missing {n:?}"); unsafe{std::mem::transmute_copy(&p)} }
fn main() {
    let args:Vec<String>=env::args().collect(); assert_eq!(args.len(),3,"SOURCE OUTPUT");
    let library=CString::new(env::var("AD_NVRTC_LIB").unwrap()).unwrap();
    let h=unsafe{dlopen(library.as_ptr(),2)}; assert!(!h.is_null(),"NVRTC dlopen");
    let want_ptx=env::var("AD_ARCH").unwrap_or_default().contains("compute_");
    let (create,compile,size,log_size,log,get,destroy):(Create,Compile,Size,Size,Get,Get,Destroy)=unsafe{
        (sym(h,c"nvrtcCreateProgram"),sym(h,c"nvrtcCompileProgram"),if want_ptx {sym(h,c"nvrtcGetPTXSize")} else {sym(h,c"nvrtcGetCUBINSize")},sym(h,c"nvrtcGetProgramLogSize"),sym(h,c"nvrtcGetProgramLog"),if want_ptx {sym(h,c"nvrtcGetPTX")} else {sym(h,c"nvrtcGetCUBIN")},sym(h,c"nvrtcDestroyProgram"))};
    let src=CString::new(fs::read(&args[1]).unwrap()).unwrap(); let mut prog=ptr::null_mut();
    assert_eq!(unsafe{create(&mut prog,src.as_ptr(),c"proof.cu".as_ptr(),0,ptr::null(),ptr::null())},0);
    let arch=CString::new(env::var("AD_ARCH").unwrap_or("--gpu-architecture=sm_120".into())).unwrap();
    let option=env::var("AD_OPTION").ok().map(|s|CString::new(s).unwrap());
    let mut options=vec![arch.as_ptr(),c"--std=c++17".as_ptr(),c"--fmad=false".as_ptr(),c"--ftz=false".as_ptr()];
    if let Some(ref flag)=option {options.push(c"-DSI_COMPILE_ONE".as_ptr()); options.push(flag.as_ptr());}
    let rc=unsafe{compile(prog,options.len() as i32,options.as_ptr())}; let mut n=0;
    assert_eq!(unsafe{log_size(prog,&mut n)},0); let mut msg=vec![0u8;n]; if n>0 {assert_eq!(unsafe{log(prog,msg.as_mut_ptr().cast())},0)}
    assert_eq!(rc,0,"NVRTC failed: {}",String::from_utf8_lossy(&msg));
    assert_eq!(unsafe{size(prog,&mut n)},0); let mut cubin=vec![0u8;n]; assert_eq!(unsafe{get(prog,cubin.as_mut_ptr().cast())},0);
    assert_eq!(unsafe{destroy(&mut prog)},0); fs::write(&args[2],&cubin).unwrap(); println!("compiled {} bytes={n}",&args[2]);
}
