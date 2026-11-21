// SPDX-License-Identifier: Apache-2.0
//! Build script of the hexad crate (ad-binder + libhexad_phys).
//!
//! * The pinned submitter: `step_submit.c` is the forecast engine's own plain-stream submitter
//!   (src/hexcore/native/step_submit.c). The replay must submit exactly as the forecast does, so the build
//!   refuses unless that file has the pinned sha256 (the AD program's pin).
//! * The CUB / Thrust host bridges (src/runner/native_*_bridge.cc), compiled with the system C++ compiler.
//! * The rule and support fatbins the window embeds (cuda/*.cu), sm_120 + sm_89 SASS and compute_80 PTX, with
//!   strict numerics (no FMA contraction, IEEE division and square root), exactly the AD program's flags:
//!     - `NVCC` names an nvcc: they are compiled (a compiler failure stops the build);
//!     - else `HEXAD_FATBIN_DIR` names a directory holding the four prebuilt images: they are embedded;
//!     - else the images are empty: the strict replay works, and `window` / the C ABI refuse at module load by
//!       name ("CUDA_ERROR_INVALID_IMAGE" on the empty image). `ad-binder --version` states which it is.
use sha2::{Digest, Sha256};
use std::{env, fs, path::{Path, PathBuf}, process::Command};

const SUBMITTER: &str = "../../src/hexcore/native/step_submit.c";
const SUBMITTER_SHA256: &str = "befd3d1c2f000b6596ad5b06ca51a50264e5ceda1288923882b8ce46df582e49";
// An optional extension driver over the window (cfg hexad_ext_driver) is not part of this crate: the cfg is
// declared and never set, so `ad-binder window --ext-driver` refuses by name.
const IMAGES: [(&str, &str); 5] = [
    ("cuda/e2e_support.cu", "e2e_support.fatbin"),
    ("cuda/p8.cu", "p8_rules.fatbin"),
    ("cuda/window.cu", "window_support.fatbin"),
    ("cuda/hexad_t.cu", "hexad_t.fatbin"),
    // inner-model (0.6.2): the simplified linear model's boundary layer (read only under AD_IM_BL=fixedk).
    ("cuda/inner_model.cu", "inner_model.fatbin"),
];
const NVCC_FLAGS: [&str; 11] = ["-fatbin", "-O3", "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
    "-gencode", "arch=compute_120,code=sm_120", "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_80,code=compute_80"];

fn run(command: &mut Command) {
    let status = command.status().unwrap_or_else(|e| panic!("could not start {command:?}: {e}"));
    assert!(status.success(), "toolchain step failed: {command:?}");
}

fn source_commit() -> String {
    let sync = fs::read_to_string("SYNC.json").expect("SYNC.json (tree/ad/tools/sync_hexad_public.py writes it)");
    let key = "\"source_commit\": \"";
    let at = sync.find(key).expect("SYNC.json has no source_commit") + key.len();
    let full = &sync[at..at + sync[at..].find('"').expect("unterminated source_commit")];
    // 12 hex digits, plus "-dirty" when the synced files differed from that commit.
    format!("{}{}", &full[..full.len().min(12)], if full.ends_with("-dirty") { "-dirty" } else { "" })
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(hexad_ext_driver)");
    let digest = format!("{:x}", Sha256::digest(fs::read(SUBMITTER).expect("read the pinned submitter")));
    assert_eq!(digest, SUBMITTER_SHA256, "{SUBMITTER} differs from the pinned hex submitter");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".into());
    let (object, cub, thrust) = (out.join("step_submit.o"), out.join("native_cub_bridge.o"), out.join("native_thrust_bridge.o"));
    run(Command::new(&cc).args(["-std=c11", "-O2", "-fPIC", "-c", SUBMITTER, "-o"]).arg(&object));
    run(Command::new(&cxx).args(["-std=c++17", "-O2", "-fPIC", "-c", "src/runner/native_cub_bridge.cc", "-o"]).arg(&cub));
    run(Command::new(&cxx).args(["-std=c++17", "-O2", "-fPIC", "-c", "src/runner/native_thrust_bridge.cc", "-o"]).arg(&thrust));
    run(Command::new(env::var("AR").unwrap_or_else(|_| "ar".into())).arg("crs").arg(out.join("libbinder_pinned_submit.a"))
        .arg(&object).arg(&cub).arg(&thrust));

    let state = match (env::var("NVCC"), env::var_os("HEXAD_FATBIN_DIR")) {
        (Ok(nvcc), _) => {
            for (src, name) in IMAGES {
                run(Command::new(&nvcc).args(NVCC_FLAGS).arg(src).arg("-o").arg(out.join(name)));
            }
            let version = Command::new(&nvcc).arg("--version").output().ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.lines().filter_map(|l| l.split(", V").nth(1)).next().map(str::to_owned))
                .unwrap_or_else(|| "unknown".into());
            format!("built by nvcc {version}")
        }
        (Err(_), Some(dir)) => {
            for (_, name) in IMAGES {
                let image = Path::new(&dir).join(name);
                let bytes = fs::read(&image).unwrap_or_else(|e| panic!("HEXAD_FATBIN_DIR: {}: {e}", image.display()));
                assert!(!bytes.is_empty(), "HEXAD_FATBIN_DIR: {} is empty", image.display());
                fs::write(out.join(name), bytes).expect("copy prebuilt fatbin");
            }
            "prebuilt".into()
        }
        (Err(_), None) => {
            println!("cargo:warning=hexad: NVCC not set and no HEXAD_FATBIN_DIR: rule fatbins are EMPTY; \
                      `ad-binder window` and the C ABI will refuse at module load (the strict replay still works)");
            for (_, name) in IMAGES { fs::write(out.join(name), b"").expect("placeholder image"); }
            "absent".into()
        }
    };
    println!("cargo:rustc-env=HEXAD_FATBINS={state}");
    println!("cargo:rustc-env=HEXAD_SOURCE_COMMIT={}", source_commit());
    for (src, _) in IMAGES { println!("cargo:rerun-if-changed={src}"); }
    for path in [SUBMITTER, "src/runner/native_cub_bridge.cc", "src/runner/native_thrust_bridge.cc", "SYNC.json"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for var in ["NVCC", "HEXAD_FATBIN_DIR", "CC", "CXX", "AR"] { println!("cargo:rerun-if-env-changed={var}"); }
    // The library target links the archive through rustc-link-lib; cargo passes -l only to a package's library
    // target, so the executable (built from the same root file, not through the rlib) names it as a link argument.
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=binder_pinned_submit");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg-bins={}", out.join("libbinder_pinned_submit.a").display());
    println!("cargo:rustc-link-arg-bins=-lstdc++");
    println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
}
