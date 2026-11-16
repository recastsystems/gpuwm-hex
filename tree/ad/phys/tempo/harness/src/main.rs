// SPDX-License-Identifier: Apache-2.0
//! tempo-ad-proof: whole-launch proof of the generated TEMPO tangent/adjoint
//! (hrrrv5 tempo_warp_steps) on captured hex launches. All scoring is here in
//! Rust; CUDA does the column arithmetic.
//!
//!   tempo-ad-proof build --cu tempo_ad.cu --nvrtc libnvrtc.so.13 --out tempo_ad.cubin
//!   tempo-ad-proof prove --cubin tempo_ad.cubin --capture LAUNCH_DIR [--capture ...]
//!                        --columns 135,973,1 --tile N --out receipt.json
mod gpu;
mod prove;
mod small;

pub const STRICT: &[&str] = &["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
                              "-arch=sm_120", "--device-as-default-execution-space"];

fn arg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("build") => {
            let cu = std::fs::read_to_string(arg(&a, "--cu").expect("--cu")).unwrap();
            let nv = gpu::Nvrtc::open(&arg(&a, "--nvrtc").expect("--nvrtc"));
            let mut opts: Vec<String> = STRICT.iter().map(|s| s.to_string()).collect();
            if let Some(x) = arg(&a, "--arch") { opts.retain(|o| !o.starts_with("-arch")); opts.push(format!("-arch={x}")); }
            let t = std::time::Instant::now();
            match nv.build(&cu, &opts) {
                Ok((img, log)) => {
                    std::fs::write(arg(&a, "--out").expect("--out"), &img).unwrap();
                    println!("nvrtc {}.{} built {} bytes in {:.1} s, options {:?}", nv.version.0, nv.version.1, img.len(), t.elapsed().as_secs_f64(), opts);
                    if !log.trim().is_empty() { println!("log:\n{}", log.lines().take(60).collect::<Vec<_>>().join("\n")); }
                }
                Err(e) => { eprintln!("{}", e.lines().take(120).collect::<Vec<_>>().join("\n")); std::process::exit(1); }
            }
        }
        Some("prove") => prove::run(&a),
        Some("promoted") => prove::promoted(&a),
        Some("small") => small::run(&a),
        Some("warp") => prove::warp::run(&a),
        _ => { eprintln!("usage: tempo-ad-proof build|prove ..."); std::process::exit(2); }
    }
}
