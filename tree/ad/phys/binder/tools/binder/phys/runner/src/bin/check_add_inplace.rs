// SPDX-License-Identifier: Apache-2.0
//! Check the native frame against independent same-run array views.
use serde::Deserialize;
use std::io::{self, BufRead};
#[path = "../../../../../../runtime/src/binder/phys/add_inplace.rs"]
mod add_inplace;
#[derive(Deserialize)]
struct Case { target:[i64;2], increment:[i64;2], expected:[i32;2], grid:Vec<u32>, block:Vec<u32> }
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut calls,mut scalar_differences,mut grid_differences)=(0,0,0);
    for line in io::stdin().lock().lines() {
        let c:Case=serde_json::from_str(&line?)?;
        let frame=add_inplace::prepare(c.target,c.increment)?;
        scalar_differences += usize::from(frame.nlevels!=c.expected[0]);
        scalar_differences += usize::from(frame.nowners!=c.expected[1]);
        grid_differences += usize::from(c.grid!=[frame.grid[0]] || c.block!=[128]);
        calls+=1;
    }
    println!("{}",serde_json::json!({"complete":false,"calls_checked":calls,
        "scalars_checked":2*calls,"differing_scalars":scalar_differences,
        "differing_geometry":grid_differences,"native_physics_replay":false}));
    if calls==0 || scalar_differences!=0 || grid_differences!=0 {std::process::exit(1);}
    Ok(())
}
