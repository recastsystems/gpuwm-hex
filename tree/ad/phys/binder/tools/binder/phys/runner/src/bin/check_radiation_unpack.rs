// SPDX-License-Identifier: Apache-2.0
//! Check native unpack arguments against same-run view observations.
use serde::Deserialize;
use std::{collections::HashMap, io::{self, BufRead}};
mod plan {
    use serde::Deserialize;
    #[derive(Clone, Copy, Debug, Deserialize)]
    pub struct Slice { pub allocation:u64, pub offset:usize }
}
mod submit {
    #[derive(Clone, Copy, Debug)] pub struct Allocation { pub id:u64, pub address:u64, pub bytes:usize }
    #[derive(Clone, Copy, Debug)] pub struct Slice { pub allocation:u64, pub offset:usize }
    #[derive(Clone, Debug)] pub enum Argument { Bytes(Vec<u8>), Device(Slice) }
}
#[path = "../../../../../../runtime/src/binder/phys/radiation_unpack.rs"]
mod radiation_unpack;
#[derive(Deserialize)] struct View { target:Option<plan::Slice>, shape:Vec<i64>, strides:Vec<i64> }
#[derive(Deserialize)] struct Row { dst:View, src:Option<View>, fill:u32 }
#[derive(Deserialize)] struct Alloc { id:u64, address:u64, bytes:usize }
#[derive(Deserialize)] struct Case { rows:Vec<Row>, allocations:Vec<Alloc>, expected:Vec<String>, grid:Vec<u32>, block:Vec<u32> }
fn view(v:View) -> radiation_unpack::View { radiation_unpack::View { target:v.target,shape:v.shape,strides:v.strides } }
fn hex(bytes:&[u8]) -> String { bytes.iter().map(|b|format!("{b:02x}")).collect() }
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let (mut count,mut rows,mut words,mut different,mut grid_errors)=(0,0,0,0,0);
    for line in io::stdin().lock().lines() {
        let c:Case=serde_json::from_str(&line?)?;
        let allocations:HashMap<_,_>=c.allocations.into_iter().map(|a| (a.id,submit::Allocation {
            id:a.id,address:a.address,bytes:a.bytes })).collect();
        let list:Vec<_>=c.rows.into_iter().map(|r|radiation_unpack::Row {
            dst:view(r.dst),src:r.src.map(view),fill:r.fill }).collect();
        let frame=radiation_unpack::prepare(&list,&allocations)?.ok_or("empty observed launch")?;
        if frame.arguments.len()!=c.expected.len() {return Err("argument count mismatch".into());}
        if c.grid!=[frame.grid[0],frame.grid[1]] || c.block!=[256] {grid_errors+=1;}
        for (arg, expected) in frame.arguments.iter().zip(&c.expected) {
            let value=match arg {
                submit::Argument::Bytes(value)=>value.clone(),
                submit::Argument::Device(s)=>allocations.get(&s.allocation).ok_or("missing allocation")?
                    .address.checked_add(s.offset as u64).ok_or("address overflow")?.to_le_bytes().to_vec(),
            };
            if hex(&value)!=*expected {different+=1;}
            words+=1;
        }
        count+=1; rows+=list.len();
    }
    println!("{}",serde_json::json!({"complete":false,"launches_checked":count,"rows_checked":rows,
        "arguments_checked":words,"differing_arguments":different,"differing_geometry":grid_errors,
        "view_ownership_proved":false,"native_physics_replay":false}));
    if count==0 || different!=0 || grid_errors!=0 {std::process::exit(1);}
    Ok(())
}
