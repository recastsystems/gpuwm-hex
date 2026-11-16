// SPDX-License-Identifier: Apache-2.0
//! An ordinary-forward process can lend live device allocations via CUDA IPC.
//! The sender must wait for each acknowledgement before mutating or freeing
//! them. Host-state bytes, when present, stream solely for comparison and are
//! never saved or used to initialize replay. Correct allocation identity,
//! the full census, host state, and synchronization
//! with the ordinary forecast must be established separately by its producer.
use super::{cuda::IpcMemoryHandle, decode_hex, host_state::{self, HostState}, submit};
use serde::Deserialize;
use std::{collections::{HashMap, HashSet}, io::{BufRead, BufReader, Read, Write},
          os::unix::net::UnixStream, path::Path, time::Duration};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message { event: String, step: usize, allocations: Vec<Allocation> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Allocation { id: u64, bytes: usize, offset: u64, handle_hex: String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostBegin { event: String, step: usize, count: usize }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HostBuffer { event: String, step: usize, id: u64, bytes: usize }
pub struct ReferenceAllocation { pub id: u64, pub bytes: usize, pub offset: u64, pub handle: IpcMemoryHandle }
pub struct WordReference { io: BufReader<UnixStream> }
impl WordReference {
    fn read_host_header<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, String> {
        let mut data = Vec::new();
        let read = (&mut self.io).take(65537).read_until(b'\n', &mut data)
            .map_err(|e| format!("cannot read host reference header: {e}"))?;
        if read == 0 || read > 65536 || data.last() != Some(&b'\n') {
            return Err("missing, unterminated, or oversized host reference header".into());
        }
        serde_json::from_slice(&data).map_err(|e| format!("invalid host reference header: {e}"))
    }
    pub fn connect(path: &Path) -> Result<Self, String> {
        let socket = UnixStream::connect(path).map_err(|e| format!("cannot connect reference socket: {e}"))?;
        socket.set_read_timeout(Some(Duration::from_secs(1800))).map_err(|e| e.to_string())?;
        socket.set_write_timeout(Some(Duration::from_secs(1800))).map_err(|e| e.to_string())?;
        Ok(Self { io: BufReader::new(socket) })
    }
    pub fn ready(&mut self) -> Result<(), String> {
        self.io.get_mut().write_all(b"{\"event\":\"ready\",\"schema\":\"ad.binder.word-reference/v1\"}\n")
            .map_err(|e| format!("cannot send reference readiness: {e}"))
    }
    pub fn next(&mut self, step: usize, carried: &[u64],
                actual: &HashMap<u64, submit::Allocation>) -> Result<Vec<ReferenceAllocation>, String> {
        let mut data = Vec::new();
        let read = (&mut self.io).take(32 * 1024 * 1024 + 1).read_until(b'\n', &mut data)
            .map_err(|e| format!("cannot read reference handles: {e}"))?;
        if read == 0 || read > 32 * 1024 * 1024 || data.last() != Some(&b'\n') {
            return Err("missing, unterminated, or oversized reference handle message".into());
        }
        let message: Message = serde_json::from_slice(&data).map_err(|e| format!("invalid reference handles: {e}"))?;
        if message.event != "state_handles" || message.step != step || message.allocations.len() != carried.len() {
            return Err("reference handle step or allocation count differs".into());
        }
        let expected: HashSet<_> = carried.iter().copied().collect();
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(carried.len());
        for value in message.allocations {
            if !expected.contains(&value.id) || !seen.insert(value.id) {
                return Err(format!("unknown or repeated reference allocation {}", value.id));
            }
            let live = actual.get(&value.id).ok_or("missing actual carried allocation")?;
            if value.bytes != live.bytes || value.bytes == 0 { return Err("reference allocation size differs".into()); }
            let bytes = decode_hex(&value.handle_hex)?;
            let reserved: [u8; 64] = bytes.try_into().map_err(|_| "CUDA IPC handle must contain 64 bytes")?;
            result.push(ReferenceAllocation { id: value.id, bytes: value.bytes, offset: value.offset,
                                               handle: IpcMemoryHandle { reserved } });
        }
        Ok(result)
    }
    pub fn acknowledge(&mut self, step: usize, differing: u64) -> Result<(), String> {
        serde_json::to_writer(self.io.get_mut(), &serde_json::json!({"event":"ack","step":step,
                                                                     "differing_words":differing}))
            .map_err(|e| format!("cannot encode reference acknowledgement: {e}"))?;
        self.io.get_mut().write_all(b"\n").map_err(|e| format!("cannot acknowledge reference: {e}"))
    }
    pub fn compare_host(&mut self, step: usize, carried: &[u64], actual: &HostState) -> Result<u64, String> {
        if carried.is_empty() { return Ok(0); }
        let begin: HostBegin = self.read_host_header()?;
        if begin.event != "host_begin" || begin.step != step || begin.count != carried.len() {
            return Err("host reference step or allocation count differs".into());
        }
        let expected: HashSet<_> = carried.iter().copied().collect();
        if expected.len() != carried.len() { return Err("duplicate carried host allocation".into()); }
        let mut seen = HashSet::new();
        let mut total = 0_u64;
        let mut chunk = vec![0_u8; 65536];
        for _ in 0..begin.count {
            let row: HostBuffer = self.read_host_header()?;
            if row.event != "host_buffer" || row.step != step || !expected.contains(&row.id) || !seen.insert(row.id) {
                return Err("unknown, repeated, or invalid host reference allocation".into());
            }
            let live = actual.get(row.id)?;
            if live.len() != row.bytes || row.bytes == 0 { return Err("host reference allocation size differs".into()); }
            let mut count = 0_u64;
            for bytes in live.chunks(chunk.len()) {
                self.io.read_exact(&mut chunk[..bytes.len()]).map_err(|e| format!("truncated host reference bytes: {e}"))?;
                count = count.checked_add(host_state::differing_words(bytes, &chunk[..bytes.len()])?)
                    .ok_or("host differing word count overflow")?;
            }
            total = total.checked_add(count).ok_or("host differing word total overflow")?;
            println!("{}", serde_json::json!({"event":"state_differing_words","storage":"host",
                                             "step":step,"allocation":row.id,"bytes":row.bytes,"count":count}));
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn actual() -> HashMap<u64, submit::Allocation> {
        HashMap::from([(7, submit::Allocation { id: 7, address: 0x1000, bytes: 16 }),
                       (9, submit::Allocation { id: 9, address: 0x2000, bytes: 8 })])
    }
    fn item(id: u64, bytes: usize) -> serde_json::Value {
        serde_json::json!({"id":id,"bytes":bytes,"offset":0,"handle_hex":"ab".repeat(64)})
    }
    fn receive(message: serde_json::Value) -> Result<Vec<ReferenceAllocation>, String> {
        let (client, mut server) = UnixStream::pair().unwrap();
        server.write_all(format!("{message}\n").as_bytes()).unwrap();
        let mut reference = WordReference { io: BufReader::new(client) };
        reference.next(1, &[7, 9], &actual())
    }
    #[test]
    fn accepts_each_live_carried_allocation_once() {
        let received = receive(serde_json::json!({"event":"state_handles","step":1,
                                                    "allocations":[item(9, 8),item(7, 16)]})).unwrap();
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].handle.reserved, [0xab; 64]);
    }
    #[test]
    fn rejects_missing_duplicate_wrong_step_and_wrong_size() {
        assert!(receive(serde_json::json!({"event":"state_handles","step":1,"allocations":[item(7, 16)]})).is_err());
        assert!(receive(serde_json::json!({"event":"state_handles","step":1,"allocations":[item(7, 16),item(7, 16)]})).is_err());
        assert!(receive(serde_json::json!({"event":"state_handles","step":2,"allocations":[item(7, 16),item(9, 8)]})).is_err());
        assert!(receive(serde_json::json!({"event":"state_handles","step":1,"allocations":[item(7, 15),item(9, 8)]})).is_err());
    }

    #[test]
    fn streams_host_words_across_chunks_and_rejects_bad_metadata() {
        let mut state = HostState::default();
        state.insert(20, vec![0; 65539]).unwrap();
        let (client, mut server) = UnixStream::pair().unwrap();
        let writer = std::thread::spawn(move || {
            server.write_all(b"{\"event\":\"host_begin\",\"step\":1,\"count\":1}\n").unwrap();
            server.write_all(b"{\"event\":\"host_buffer\",\"step\":1,\"id\":20,\"bytes\":65539}\n").unwrap();
            let mut reference = vec![0_u8; 65539];
            reference[65535] = 1;
            reference[65538] = 2;
            server.write_all(&reference).unwrap();
        });
        let mut reference = WordReference { io: BufReader::new(client) };
        assert_eq!(reference.compare_host(1, &[20], &state).unwrap(), 2);
        writer.join().unwrap();

        for message in [
            b"{\"event\":\"host_begin\",\"step\":2,\"count\":1}\n".as_slice(),
            b"{\"event\":\"host_begin\",\"step\":1,\"count\":2}\n".as_slice(),
            b"{\"event\":\"host_begin\",\"step\":1,\"count\":1}\n{\"event\":\"host_buffer\",\"step\":1,\"id\":20,\"bytes\":1}\n".as_slice(),
            b"{\"event\":\"host_begin\",\"step\":1,\"count\":1}\n{\"event\":\"host_buffer\",\"step\":1,\"id\":99,\"bytes\":65539}\n".as_slice(),
        ] {
            let (client, mut server) = UnixStream::pair().unwrap();
            server.write_all(message).unwrap();
            let mut reference = WordReference { io: BufReader::new(client) };
            assert!(reference.compare_host(1, &[20], &state).is_err());
        }
    }
}
