// SPDX-License-Identifier: Apache-2.0
use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("could not start C toolchain");
    assert!(status.success(), "C toolchain failed: {command:?}");
}

fn main() {
    let digest = Command::new("sha256sum").arg("step_submit.c").output()
        .expect("could not hash pinned submitter");
    assert!(digest.status.success() && digest.stdout.starts_with(
        b"befd3d1c2f000b6596ad5b06ca51a50264e5ceda1288923882b8ce46df582e49 "),
        "step_submit.c differs from pinned hex source");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let object = out.join("step_submit.o");
    let cub_object = out.join("native_cub_bridge.o");
    let thrust_object = out.join("native_thrust_bridge.o");
    let archive = out.join("libbinder_pinned_submit.a");
    run(Command::new("cc").args(["-std=c11", "-O2", "-fPIC", "-c", "step_submit.c", "-o"])
        .arg(&object));
    run(Command::new("c++").args(["-std=c++17", "-O2", "-fPIC", "-c",
        "../../../../runtime/src/binder/phys/native_cub_bridge.cc", "-o"]).arg(&cub_object));
    run(Command::new("c++").args(["-std=c++17", "-O2", "-fPIC", "-c",
        "../../../../runtime/src/binder/phys/native_thrust_bridge.cc", "-o"]).arg(&thrust_object));
    run(Command::new("ar").arg("crs").arg(&archive).arg(&object).arg(&cub_object).arg(&thrust_object));
    // Physics-on window stage A: the ad/e2e support kernels (fills, fixed-order dot reductions), only when
    // NVCC is given; otherwise an empty image and AD_DOT_* refuses at load (the plain replay never loads it).
    let support = out.join("e2e_support.fatbin");
    match env::var("NVCC") {
        Ok(nvcc) => run(Command::new(nvcc).args(["-fatbin", "-O3", "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
            "-gencode", "arch=compute_120,code=sm_120", "-gencode", "arch=compute_89,code=sm_89",
            "-gencode", "arch=compute_80,code=compute_80", "../../../../../../e2e/cuda/e2e_support.cu", "-o"]).arg(&support)),
        Err(_) => std::fs::write(&support, b"").expect("support placeholder"),
    }
    // Physics-on window lane L1 (ad-binder window): P8 rule kernels and the window support kernels, only with NVCC.
    // An optional extension driver over the window (ext_driver.rs + window/cuda/ext_driver.cu) is not part of
    // the release. It is compiled in exactly when its CUDA source is present (cfg hexad_ext_driver); without it the
    // crate builds and `--ext-driver` refuses by name.
    println!("cargo:rustc-check-cfg=cfg(hexad_ext_driver)");
    let ext_driver = "../../../../../window/cuda/ext_driver.cu";
    println!("cargo:rerun-if-changed={ext_driver}");
    let mut images = vec![("../../../../../p8/cuda/p8.cu", "p8_rules.fatbin"), ("../../../../../window/cuda/window.cu", "window_support.fatbin"), ("../../../../../window/cuda/hexad_t.cu", "hexad_t.fatbin")];
    if std::path::Path::new(ext_driver).exists() {
        println!("cargo:rustc-cfg=hexad_ext_driver");
        images.push((ext_driver, "ext_driver.fatbin"));
    }
    for (src, name) in images {
        let image = out.join(name);
        match env::var("NVCC") {
            Ok(nvcc) => run(Command::new(nvcc).args(["-fatbin", "-O3", "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
                "-gencode", "arch=compute_120,code=sm_120", "-gencode", "arch=compute_89,code=sm_89",
                "-gencode", "arch=compute_80,code=compute_80", src, "-o"]).arg(&image)),
            Err(_) => std::fs::write(&image, b"").expect("window placeholder"),
        }
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-changed=../../../../../../e2e/cuda/e2e_support.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-changed=step_submit.c");
    println!("cargo:rerun-if-changed=../../../../runtime/src/binder/phys/native_cub_bridge.cc");
    println!("cargo:rerun-if-changed=../../../../runtime/src/binder/phys/native_thrust_bridge.cc");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=binder_pinned_submit");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
}
