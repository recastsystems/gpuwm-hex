// SPDX-License-Identifier: Apache-2.0
//! Deterministic one-stream ordering for an observed CUDA submission trace.
//!
//! This orders the supplied graph; it does not prove that the recorder saw
//! every operation, that an event handle was not missed, or that stream races
//! cannot change results. A producer must establish those facts separately.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Work,
    EventCreate { event: u64 },
    EventDestroy { event: u64 },
    EventRecord { event: u64 },
    EventWait { event: u64 },
    ContextBarrier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub issue: u64,
    pub stream: Option<u64>,
    pub kind: Kind,
}

#[derive(Debug)]
pub struct Ordering {
    /// Input indices, including event and barrier nodes, in execution order.
    pub nodes: Vec<usize>,
    pub stream_edges: usize,
    pub event_edges: usize,
    pub barrier_edges: usize,
}

pub fn linearize(nodes: &[Node]) -> Result<Ordering, String> {
    let mut issue_order: Vec<usize> = (0..nodes.len()).collect();
    issue_order.sort_by_key(|&i| nodes[i].issue);
    if issue_order.windows(2).any(|p| nodes[p[0]].issue == nodes[p[1]].issue) {
        return Err("duplicate observed issue order".into());
    }
    let mut successors = vec![Vec::<usize>::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    let mut edges = HashSet::<(usize, usize)>::new();
    let mut stream_last = HashMap::<u64, usize>::new();
    // A newly created, unrecorded CUDA event makes a wait a no-op. For an
    // unknown event the trace is insufficient and must be rejected.
    let mut events = HashMap::<u64, Option<usize>>::new();
    let mut last_barrier = None;
    let (mut stream_edges, mut event_edges, mut barrier_edges) = (0, 0, 0);
    let mut edge = |from: usize, to: usize, count: &mut usize| {
        if from != to && edges.insert((from, to)) {
            successors[from].push(to);
            indegree[to] += 1;
            *count += 1;
        }
    };
    for &i in &issue_order {
        let node = &nodes[i];
        if let Some(stream) = node.stream {
            if let Some(&previous) = stream_last.get(&stream) {
                edge(previous, i, &mut stream_edges);
            }
            if let Some(previous) = last_barrier {
                edge(previous, i, &mut barrier_edges);
            }
            stream_last.insert(stream, i);
        }
        match node.kind {
            Kind::Work => {
                if node.stream.is_none() { return Err(format!("work {} has no stream", node.issue)); }
            },
            Kind::EventCreate { event } => {
                if events.insert(event, None).is_some() {
                    return Err(format!("event {event} created while live"));
                }
            },
            Kind::EventDestroy { event } => {
                if events.remove(&event).is_none() {
                    return Err(format!("destroy of unknown event {event}"));
                }
            },
            Kind::EventRecord { event } => {
                if node.stream.is_none() { return Err(format!("event record {} has no stream", node.issue)); }
                let value = events.get_mut(&event).ok_or_else(|| format!("record of unknown event {event}"))?;
                *value = Some(i);
            },
            Kind::EventWait { event } => {
                if node.stream.is_none() { return Err(format!("event wait {} has no stream", node.issue)); }
                if let Some(record) = events.get(&event).ok_or_else(|| format!("wait for unknown event {event}"))? {
                    edge(*record, i, &mut event_edges);
                }
            },
            Kind::ContextBarrier => {
                for previous in stream_last.values() {
                    edge(*previous, i, &mut barrier_edges);
                }
                if let Some(previous) = last_barrier { edge(previous, i, &mut barrier_edges); }
                last_barrier = Some(i);
            },
        }
    }
    let mut ready = BinaryHeap::new();
    for (i, &degree) in indegree.iter().enumerate() {
        if degree == 0 { ready.push(Reverse((nodes[i].issue, i))); }
    }
    let mut ordered = Vec::with_capacity(nodes.len());
    while let Some(Reverse((_, i))) = ready.pop() {
        ordered.push(i);
        for &next in &successors[i] {
            indegree[next] -= 1;
            if indegree[next] == 0 { ready.push(Reverse((nodes[next].issue, next))); }
        }
    }
    if ordered.len() != nodes.len() { return Err("cycle in happens-before graph".into()); }
    Ok(Ordering { nodes: ordered, stream_edges, event_edges, barrier_edges })
}
