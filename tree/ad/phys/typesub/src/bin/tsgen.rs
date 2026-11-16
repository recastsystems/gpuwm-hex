// SPDX-License-Identifier: Apache-2.0
//! tsgen: type-substitution derivative generator (P2 of the private autodiff program).
//!
//! Reads a captured physics source (the exact text the forward image was compiled from), finds
//! the functions reachable from the named roots, and re-emits them twice inside namespaces whose
//! `TF`/`TD` are the tangent and tape value types of tsad.cuh, with every `float` token replaced
//! by `TF` and every `double` by `TD`.  The captured text itself is copied unchanged ahead of the
//! generated copies (optionally without its `extern "C" __global__` kernels), so the original
//! functions stay in the same translation unit for identity checks.  Leaf functions (the source's
//! maths wrappers) are not copied: style rules (tsad_<style>.cuh) give their derivatives.
//! Each copied item keeps its top-level preprocessor context (#if/#else chain), re-evaluated at
//! the generated site, so shape macros redefined by the rules header pick the right variant.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::process::exit;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind { Ws, Comment, Pre, Ident, Num, Str, Punct }

#[derive(Clone, Debug)]
struct Tok { k: Kind, s: String }

fn lex(src: &str) -> Vec<Tok> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut out = Vec::new();
    let mut line_start = true;
    while i < n {
        let c = b[i];
        let st = i;
        let kind;
        if c == b'\n' || c == b' ' || c == b'\t' || c == b'\r' {
            while i < n && (b[i] == b' ' || b[i] == b'\t' || b[i] == b'\r' || b[i] == b'\n') {
                if b[i] == b'\n' { line_start = true; }
                i += 1;
            }
            out.push(Tok { k: Kind::Ws, s: src[st..i].to_string() });
            continue;
        } else if c == b'#' && line_start {
            while i < n && b[i] != b'\n' {
                if b[i] == b'\\' && i + 1 < n && b[i + 1] == b'\n' { i += 2; continue; }
                if b[i] == b'\\' && i + 2 < n && b[i + 1] == b'\r' && b[i + 2] == b'\n' { i += 3; continue; }
                if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
                    i += 2;
                    while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') { i += 1; }
                    i = (i + 2).min(n);
                    continue;
                }
                i += 1;
            }
            kind = Kind::Pre;
        } else if c == b'/' && i + 1 < n && b[i + 1] == b'/' {
            while i < n && b[i] != b'\n' { i += 1; }
            kind = Kind::Comment;
        } else if c == b'/' && i + 1 < n && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') { i += 1; }
            i = (i + 2).min(n);
            kind = Kind::Comment;
        } else if c == b'"' || c == b'\'' {
            i += 1;
            while i < n && b[i] != c {
                if b[i] == b'\\' { i += 1; }
                i += 1;
            }
            i = (i + 1).min(n);
            kind = Kind::Str;
        } else if c.is_ascii_alphabetic() || c == b'_' {
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            kind = Kind::Ident;
        } else if c.is_ascii_digit() || (c == b'.' && i + 1 < n && b[i + 1].is_ascii_digit()) {
            while i < n {
                let d = b[i];
                if d.is_ascii_alphanumeric() || d == b'.' || d == b'_' { i += 1; }
                else if (d == b'+' || d == b'-') && matches!(b[i - 1], b'e' | b'E' | b'p' | b'P') { i += 1; }
                else { break; }
            }
            kind = Kind::Num;
        } else {
            let two = if i + 1 < n { &src[i..i + 2] } else { "" };
            i += if two == "->" || two == "::" { 2 } else { 1 };
            kind = Kind::Punct;
        }
        line_start = false;
        out.push(Tok { k: kind, s: src[st..i].to_string() });
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
enum ItemKind { Func, Proto, Struct }

#[derive(Clone, Debug)]
struct Item {
    kind: ItemKind,
    name: String,
    start: usize,
    end: usize,          // exclusive token index
    template: bool,
    kernel: bool,
    cond: Vec<Vec<String>>, // enclosing top-level #if chains (directive lines so far per level)
    defs: Vec<(String, String)>, // top-level macros live at this item and named in it (for re-emission)
}

fn sig(t: &[Tok], a: usize, b: usize) -> Vec<usize> {
    (a..b).filter(|&i| !matches!(t[i].k, Kind::Ws | Kind::Comment)).collect()
}

const NOT_NAMES: &[&str] = &["decltype", "__launch_bounds__", "__attribute__", "alignas", "__align__",
    "sizeof", "if", "for", "while", "switch", "return", "static_assert"];

fn directive(s: &str) -> String {
    let t = s.trim_start_matches('#').trim_start();
    t.split(|c: char| !c.is_ascii_alphabetic()).next().unwrap_or("").to_string()
}

fn live_defs(t: &[Tok], a: usize, b: usize, macros: &HashMap<String, Vec<String>>, cur: &HashMap<String, String>) -> Vec<(String, String)> {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<String> = t[a..b].iter().filter(|x| x.k == Kind::Ident).map(|x| x.s.clone()).collect();
    for x in &t[a..b] { if x.k == Kind::Pre { for y in lex_pre(&x.s) { if y.k == Kind::Ident { stack.push(y.s); } } } }
    while let Some(s) = stack.pop() {
        if seen.insert(s.clone()) { if let Some(body) = macros.get(&s) { stack.extend(body.iter().cloned()); } }
    }
    let mut v: Vec<(String, String)> = seen.into_iter().filter_map(|s| cur.get(&s).map(|l| (s, l.clone()))).collect();
    v.sort();
    v
}

fn items(t: &[Tok]) -> (Vec<Item>, HashMap<String, Vec<String>>, BTreeSet<String>) {
    let mut out = Vec::new();
    let mut macros: HashMap<String, Vec<String>> = HashMap::new();
    let mut cur: HashMap<String, String> = HashMap::new();
    let mut cond: Vec<Vec<String>> = Vec::new();
    let mut i = 0;
    let mut start = 0;
    let mut paren = 0i32;
    while i < t.len() {
        let tk = &t[i];
        match tk.k {
            Kind::Pre => {
                let d = directive(&tk.s);
                match d.as_str() {
                    "if" | "ifdef" | "ifndef" => cond.push(vec![tk.s.trim_end().to_string()]),
                    "elif" | "else" => { if let Some(l) = cond.last_mut() { l.push(tk.s.trim_end().to_string()); } }
                    "endif" => { cond.pop(); }
                    "define" => {
                        let lt = lex(tk.s.trim_start_matches('#').trim_start().trim_start_matches("define"));
                        let ids: Vec<&Tok> = lt.iter().filter(|x| x.k == Kind::Ident).collect();
                        if let Some(first) = ids.first() {
                            macros.insert(first.s.clone(), ids[1..].iter().map(|x| x.s.clone()).collect());
                            if cond.is_empty() { cur.insert(first.s.clone(), tk.s.trim_end().to_string()); }
                        }
                    }
                    "undef" => {
                        let lt = lex(tk.s.trim_start_matches('#').trim_start().trim_start_matches("undef"));
                        if let Some(first) = lt.iter().find(|x| x.k == Kind::Ident) { cur.remove(&first.s); }
                    }
                    _ => {}
                }
                start = i + 1;
                i += 1;
                continue;
            }
            Kind::Punct if tk.s == "(" => paren += 1,
            Kind::Punct if tk.s == ")" => paren -= 1,
            Kind::Punct if tk.s == ";" && paren == 0 => {
                let s = sig(t, start, i);
                if let Some(pos) = s.iter().position(|&j| t[j].s == "(") {
                    if pos > 0 && t[s[pos - 1]].k == Kind::Ident && !NOT_NAMES.contains(&t[s[pos - 1]].s.as_str())
                        && !s.iter().any(|&j| t[j].s == "=" || t[j].s == "typedef" || t[j].s == "using")
                    {
                        out.push(Item { kind: ItemKind::Proto, name: t[s[pos - 1]].s.clone(), start, end: i + 1,
                            template: t.get(*s.first().unwrap()).map(|x| x.s == "template").unwrap_or(false),
                            kernel: s.iter().any(|&j| t[j].s == "__global__"), cond: cond.clone(), defs: live_defs(t, start, i + 1, &macros, &cur) });
                    }
                }
                start = i + 1;
            }
            Kind::Punct if tk.s == "{" && paren == 0 => {
                let s = sig(t, start, i);
                let is_ns = s.iter().any(|&j| t[j].s == "namespace") || (s.len() >= 2 && t[s[0]].s == "extern" && t[s[1]].k == Kind::Str && s.len() == 2);
                if is_ns { start = i + 1; i += 1; continue; }
                // matching brace
                let mut depth = 0i32;
                let mut j = i;
                while j < t.len() {
                    if t[j].k == Kind::Punct { if t[j].s == "{" { depth += 1; } else if t[j].s == "}" { depth -= 1; if depth == 0 { break; } } }
                    j += 1;
                }
                let mut end = j + 1;
                let first_paren = s.iter().position(|&k| t[k].s == "(");
                let st_kw = s.iter().position(|&k| matches!(t[k].s.as_str(), "struct" | "class" | "union" | "enum"));
                let template = s.first().map(|&k| t[k].s == "template").unwrap_or(false);
                let is_struct = match (st_kw, first_paren) { (Some(a), Some(b)) => a < b, (Some(_), None) => true, _ => false };
                if is_struct {
                    // through the closing ';'
                    let mut k = end;
                    while k < t.len() && matches!(t[k].k, Kind::Ws | Kind::Comment) { k += 1; }
                    if k < t.len() && t[k].s == ";" { end = k + 1; }
                    let p = st_kw.unwrap();
                    let name = s.get(p + 1).map(|&k| t[k].s.clone()).unwrap_or_default();
                    out.push(Item { kind: ItemKind::Struct, name, start, end, template, kernel: false, cond: cond.clone(), defs: live_defs(t, start, end, &macros, &cur) });
                } else if let Some(fp) = first_paren {
                    // first '(' preceded by a real name
                    let mut name = String::new();
                    for (q, &k) in s.iter().enumerate() {
                        if t[k].s == "(" && q > 0 && t[s[q - 1]].k == Kind::Ident && !NOT_NAMES.contains(&t[s[q - 1]].s.as_str()) {
                            name = t[s[q - 1]].s.clone();
                            break;
                        }
                    }
                    let _ = fp;
                    if !name.is_empty() {
                        out.push(Item { kind: ItemKind::Func, name, start, end, template,
                            kernel: s.iter().any(|&k| t[k].s == "__global__"), cond: cond.clone(), defs: live_defs(t, start, end, &macros, &cur) });
                    }
                }
                i = end;
                start = end;
                paren = 0;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    let final_defs: BTreeSet<String> = cur.keys().cloned().collect();
    (out, macros, final_defs)
}

fn has_fp(t: &[Tok], it: &Item) -> bool {
    t[it.start..it.end].iter().any(|x| (x.k == Kind::Ident && (x.s == "float" || x.s == "double"))
        || (x.k == Kind::Pre && (x.s.contains("float") || x.s.contains("double"))))
}

fn idents(t: &[Tok], it: &Item, macros: &HashMap<String, Vec<String>>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut stack: Vec<String> = t[it.start..it.end].iter().filter(|x| x.k == Kind::Ident).map(|x| x.s.clone()).collect();
    for x in &t[it.start..it.end] {
        if x.k == Kind::Pre { for y in lex_pre(&x.s) { if y.k == Kind::Ident { stack.push(y.s); } } }
    }
    while let Some(s) = stack.pop() {
        if out.insert(s.clone()) {
            if let Some(body) = macros.get(&s) { stack.extend(body.iter().cloned()); }
        }
    }
    out
}

fn substitute_word(s: &str) -> String {
    let mut r = String::new();
    let hash = s.find('#').map(|p| p + 1).unwrap_or(0);
    r.push_str(&s[..hash]);
    for y in lex(&s[hash..]) {
        if y.k == Kind::Ident && y.s == "float" { r.push_str("TF"); }
        else if y.k == Kind::Ident && y.s == "double" { r.push_str("TD"); }
        else { r.push_str(&y.s); }
    }
    r
}

fn emit_item(t: &[Tok], it: &Item, final_defs: &BTreeSet<String>) -> String {
    let mut r = String::new();
    for c in &it.cond { for line in c { r.push_str(line); r.push('\n'); } }
    // Macros the source #undef'd after this item are re-defined around the copy (scoped like the
    // item text itself, so their float/double are substituted too).
    let local: Vec<&(String, String)> = it.defs.iter().filter(|(n, _)| !final_defs.contains(n)).collect();
    for (n, line) in &local { r.push_str(&format!("#undef {n}\n{}\n", substitute_word(line))); }
    let mut prev: Vec<&str> = Vec::new();
    for x in &t[it.start..it.end] {
        match x.k {
            Kind::Ident if (x.s == "float" || x.s == "double") && !(prev.len() >= 2 && prev[prev.len() - 1] == "(" && prev[prev.len() - 2] == "sizeof")
                && !constant_decl(&prev) => {
                r.push_str(if x.s == "float" { "TF" } else { "TD" });
            }
            Kind::Pre => { r.push_str(&substitute_word(&x.s)); }
            _ => r.push_str(&x.s),
        }
        if !matches!(x.k, Kind::Ws | Kind::Comment) { prev.push(&x.s); }
    }
    r.push('\n');
    for (n, _) in &local { r.push_str(&format!("#undef {n}\n")); }
    for _ in &it.cond { r.push_str("#endif\n"); }
    r
}

fn usage() -> ! {
    eprintln!("usage: tsgen --source FILE --out FILE --rules HEADER [--root NAME]... [--leaf NAME,...] \
               [--inactive NAME,...] [--drop-kernels] [--manifest FILE]");
    exit(2)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut source, mut outp, mut rules, mut manifest) = (None, None, Vec::new(), None);
    let mut roots = Vec::new();
    let mut leaves: BTreeSet<String> = BTreeSet::new();
    let mut inactive: BTreeSet<String> = BTreeSet::new();
    let mut drop_kernels = false;
    let mut i = 0;
    while i < a.len() {
        let v = a.get(i + 1).cloned();
        match a[i].as_str() {
            "--source" => { source = v; i += 1; }
            "--out" => { outp = v; i += 1; }
            "--rules" => { rules.push(v.unwrap_or_else(|| usage())); i += 1; }
            "--manifest" => { manifest = v; i += 1; }
            "--root" => { roots.push(v.unwrap_or_else(|| usage())); i += 1; }
            "--leaf" => { leaves.extend(v.unwrap_or_else(|| usage()).split(',').map(|s| s.trim().to_string())); i += 1; }
            "--inactive" => { inactive.extend(v.unwrap_or_else(|| usage()).split(',').map(|s| s.trim().to_string())); i += 1; }
            "--drop-kernels" => drop_kernels = true,
            _ => usage(),
        }
        i += 1;
    }
    let (source, outp) = (source.unwrap_or_else(|| usage()), outp.unwrap_or_else(|| usage()));
    let src = fs::read_to_string(&source).unwrap_or_else(|e| { eprintln!("{source}: {e}"); exit(1) });
    let t = lex(&src);
    let round: String = t.iter().map(|x| x.s.as_str()).collect();
    if round != src { eprintln!("lexer round trip failed"); exit(1); }
    let (its, macros, final_defs) = items(&t);
    if std::env::var("TSGEN_LIST").is_ok() {
        let mut line = 1usize; let mut at = 0usize;
        for it in &its {
            while at < it.start { line += t[at].s.matches('\n').count(); at += 1; }
            eprintln!("{line}\t{:?}\t{}\t{}", it.kind, it.name, it.end - it.start);
        }
    }
    let mut by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (k, it) in its.iter().enumerate() { by_name.entry(it.name.clone()).or_default().push(k); }
    for r in &roots {
        if !by_name.get(r).map(|v| v.iter().any(|&k| its[k].kind == ItemKind::Func || its[k].kind == ItemKind::Struct)).unwrap_or(false) {
            eprintln!("root {r} not defined in {source}"); exit(1);
        }
    }
    // Reachability.
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut work: Vec<String> = roots.clone();
    let mut structs: BTreeSet<String> = BTreeSet::new();
    let mut funcs: BTreeSet<String> = BTreeSet::new();
    let mut reached_leaves: BTreeSet<String> = BTreeSet::new();
    while let Some(name) = work.pop() {
        if !seen.insert(name.clone()) { continue; }
        if leaves.contains(&name) { reached_leaves.insert(name); continue; }
        let Some(ks) = by_name.get(&name) else { continue };
        for &k in ks {
            let it = &its[k];
            if it.kernel || it.kind == ItemKind::Proto { continue; }
            match it.kind {
                ItemKind::Struct => {
                    if (!inactive.contains(&name) && has_fp(&t, it) && !it.template) || roots.contains(&name) { structs.insert(name.clone()); }
                    // A root struct (a scheme written as one struct of member functions, as MYNN is)
                    // is copied whole; free functions it names are followed like a function body.
                    if roots.contains(&name) { for id in idents(&t, it, &macros) { if id != name && by_name.contains_key(&id) { work.push(id); } } }
                }
                ItemKind::Func => {
                    if it.template { continue; }
                    funcs.insert(name.clone());
                    for id in idents(&t, it, &macros) {
                        if id != name && by_name.contains_key(&id) { work.push(id); }
                    }
                }
                _ => {}
            }
        }
    }
    // A function is copied if it has float/double text or names a copied struct (fixpoint not
    // needed beyond one pass: structs do not depend on functions).
    let emit: Vec<usize> = (0..its.len()).filter(|&k| {
        let it = &its[k];
        if it.kernel || (it.template && !(it.kind == ItemKind::Struct && roots.contains(&it.name))) { return false; }
        match it.kind {
            ItemKind::Struct => structs.contains(&it.name),
            ItemKind::Func | ItemKind::Proto => funcs.contains(&it.name) && (has_fp(&t, it)
                || idents(&t, it, &macros).iter().any(|s| structs.contains(s))),
        }
    }).collect();
    // Output.
    let mut o = String::new();
    o.push_str("// GENERATED by tsgen (P2, private autodiff program). Do not edit.\n");
    o.push_str(&format!("// source: {source}\n// roots: {}\n// leaves (rules): {}\n",
        roots.join(","), reached_leaves.iter().cloned().collect::<Vec<_>>().join(",")));
    o.push_str(&format!("// copied: {}\n", emit.iter().map(|&k| format!("{:?}:{}", its[k].kind, its[k].name)).collect::<Vec<_>>().join(" ")));
    o.push_str("// ---- captured source, unchanged");
    o.push_str(if drop_kernels { " except its extern \"C\" __global__ kernels ----\n" } else { " ----\n" });
    let mut cut: Vec<(usize, usize)> = Vec::new();
    if drop_kernels { for it in &its { if it.kernel && it.kind == ItemKind::Func { cut.push((it.start, it.end)); } } }
    let mut k = 0;
    for (idx, x) in t.iter().enumerate() {
        while k < cut.len() && cut[k].1 <= idx { k += 1; }
        if k < cut.len() && cut[k].0 <= idx && idx < cut[k].1 { continue; }
        o.push_str(&x.s);
    }
    o.push_str("\n// ---- generated derivative copies ----\n");
    for r in &rules { o.push_str(&format!("#include \"{r}\"\n")); }
    for (ns, kind) in [("tsad_tl", "Tan"), ("tsad_tp", "Tap")] {
        o.push_str(&format!("namespace {ns} {{\ntypedef tsad::VF<tsad::{kind}> TF;\ntypedef tsad::VD<tsad::{kind}> TD;\n"));
        for &k in &emit { o.push_str(&emit_item(&t, &its[k], &final_defs)); }
        o.push_str(&format!("}}  // namespace {ns}\n"));
    }
    fs::write(&outp, o).unwrap_or_else(|e| { eprintln!("{outp}: {e}"); exit(1) });
    if let Some(m) = manifest {
        let q = |v: Vec<String>| v.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(",");
        let j = format!("{{\"schema\":\"ad.tsgen/v1\",\"source\":\"{source}\",\"roots\":[{}],\"copied_functions\":[{}],\"copied_structs\":[{}],\"rule_leaves\":[{}],\"items_scanned\":{},\"dropped_kernels\":{}}}\n",
            q(roots.clone()), q(emit.iter().filter(|&&k| its[k].kind == ItemKind::Func).map(|&k| its[k].name.clone()).collect()), q(structs.iter().cloned().collect()),
            q(reached_leaves.iter().cloned().collect()), its.len(), cut.len());
        fs::write(&m, j).unwrap();
    }
    eprintln!("tsgen: {} items scanned, {} copied ({} functions, {} structs), {} rule leaves, {} kernels dropped",
        its.len(), emit.len(), funcs.len(), structs.len(), reached_leaves.len(), cut.len());
}

/// Lex the body of a preprocessor line (after its '#'), so identifiers inside it are visible.
fn lex_pre(s: &str) -> Vec<Tok> {
    let hash = s.find('#').map(|p| p + 1).unwrap_or(0);
    lex(&s[hash..])
}

/// `constexpr float k = ...;` (also `static constexpr`, `constexpr static`) declares a compile-time
/// constant: it carries no derivative and must stay a literal type, so it keeps its float/double.
fn constant_decl(prev: &[&str]) -> bool {
    let n = prev.len();
    (n >= 1 && prev[n - 1] == "constexpr") || (n >= 2 && prev[n - 1] == "static" && prev[n - 2] == "constexpr")
}
