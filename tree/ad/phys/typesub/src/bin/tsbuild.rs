// SPDX-License-Identifier: Apache-2.0
//! tsbuild: compile a generated derivative translation unit with the captured forward's NVRTC
//! options (no GPU needed). Prints the NVRTC log, the build time and the cubin size; writes the
//! cubin when --out is given.
#[path = "../gpu.rs"]
mod gpu;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned());
    let (Some(src), Some(nvrtc)) = (get("--src"), get("--nvrtc")) else {
        eprintln!("usage: tsbuild --src FILE --nvrtc LIB [--arch sm_120] [--include DIR] [--define X] [--out CUBIN]");
        std::process::exit(2)
    };
    let arch = get("--arch").unwrap_or_else(|| "sm_120".into());
    let mut opts: Vec<String> = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
        "--device-as-default-execution-space"].iter().map(|s| s.to_string()).collect();
    opts.push(format!("-arch={arch}"));
    if let Some(i) = get("--include") { opts.push(format!("--include-path={i}")); }
    if let Some(d) = get("--define") { opts.push(format!("-D{d}")); }
    let text = std::fs::read_to_string(&src).expect("read source");
    let nv = gpu::Nvrtc::new(&nvrtc).expect("nvrtc");
    let t = Instant::now();
    match nv.build(&text, &src, &opts) {
        Ok((cubin, log)) => {
            let errs = log.lines().filter(|l| l.contains("error")).count();
            println!("tsbuild: ok {:.1}s cubin {} bytes, {} log lines, {errs} error lines", t.elapsed().as_secs_f64(), cubin.len(), log.lines().count());
            if let Some(o) = get("--out") { std::fs::write(o, cubin).expect("write cubin"); }
        }
        Err(e) => {
            let lines: Vec<&str> = e.lines().filter(|l| l.contains("error")).collect();
            println!("tsbuild: FAILED {:.1}s, {} error lines", t.elapsed().as_secs_f64(), lines.len());
            for l in e.lines().filter(|l| l.contains("error")).take(60) { println!("{l}"); }
            std::process::exit(1);
        }
    }
}
