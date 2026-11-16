// SPDX-License-Identifier: Apache-2.0
//! Size and geometry expressions from derivative specs: integers, named launch
//! scalars, grid[i]/block[i], + - * /, parentheses, ceil(), max(), min().
//! Every symbol must be supplied; nothing is guessed.
use std::collections::HashMap;

pub type Env = HashMap<String, f64>;

struct P<'a> { s: &'a [u8], i: usize, env: &'a Env }
impl P<'_> {
    fn ws(&mut self) { while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() { self.i += 1; } }
    fn eat(&mut self, c: u8) -> bool { self.ws(); if self.s.get(self.i) == Some(&c) { self.i += 1; true } else { false } }
    fn sum(&mut self) -> Result<f64, String> {
        let mut v = self.prod()?;
        loop {
            if self.eat(b'+') { v += self.prod()?; } else if self.eat(b'-') { v -= self.prod()?; } else { return Ok(v); }
        }
    }
    fn prod(&mut self) -> Result<f64, String> {
        let mut v = self.atom()?;
        loop {
            if self.eat(b'*') { v *= self.atom()?; }
            else if self.eat(b'/') { let d = self.atom()?; if d == 0.0 { return Err("division by zero".into()); } v /= d; }
            else { return Ok(v); }
        }
    }
    fn atom(&mut self) -> Result<f64, String> {
        self.ws();
        if self.eat(b'(') { let v = self.sum()?; if !self.eat(b')') { return Err("missing )".into()); } return Ok(v); }
        if self.eat(b'-') { return Ok(-self.atom()?); }
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_' || self.s[self.i] == b'.') { self.i += 1; }
        if start == self.i { return Err("missing operand".into()); }
        let word = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| "bad symbol")?;
        if word.as_bytes()[0].is_ascii_digit() { return word.parse::<f64>().map_err(|_| format!("bad number {word}")); }
        if matches!(word, "ceil" | "max" | "min") {
            if !self.eat(b'(') { return Err(format!("{word} needs (")); }
            let a = self.sum()?;
            let v = if word == "ceil" { a.ceil() } else {
                if !self.eat(b',') { return Err(format!("{word} needs two arguments")); }
                let b = self.sum()?;
                if word == "max" { a.max(b) } else { a.min(b) }
            };
            if !self.eat(b')') { return Err("missing )".into()); }
            return Ok(v);
        }
        let mut key = word.to_owned();
        if self.s.get(self.i) == Some(&b'[') {
            let end = self.s[self.i..].iter().position(|&c| c == b']').ok_or("missing ]")? + self.i;
            key.push_str(std::str::from_utf8(&self.s[self.i..=end]).map_err(|_| "bad index")?);
            self.i = end + 1;
        }
        self.env.get(&key).copied().ok_or_else(|| format!("missing symbol {key}"))
    }
}

pub fn eval(text: &str, env: &Env) -> Result<f64, String> {
    let mut p = P { s: text.as_bytes(), i: 0, env };
    let v = p.sum()?;
    p.ws();
    if p.i != p.s.len() { return Err(format!("trailing text in {text}")); }
    Ok(v)
}
pub fn eval_usize(v: &serde_json::Value, env: &Env) -> Result<usize, String> {
    let x = match v {
        serde_json::Value::Number(n) => n.as_f64().ok_or("bad number")?,
        serde_json::Value::String(s) => eval(s, env)?,
        _ => return Err(format!("unsupported size expression {v}")),
    };
    if !(x >= 0.0) || x.fract() != 0.0 || x > 1e15 { return Err(format!("non-integral size {x} from {v}")); }
    Ok(x as usize)
}
/// Element count of a declared shape: a JSON list, or a string "[a,b]".
pub fn shape_elements(v: &serde_json::Value, env: &Env) -> Result<usize, String> {
    let dims: Vec<serde_json::Value> = match v {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::String(s) => {
            let t = s.trim();
            let t = t.strip_prefix('[').and_then(|t| t.strip_suffix(']')).ok_or_else(|| format!("bad shape {s}"))?;
            if t.trim().is_empty() { vec![] } else {
                let mut parts = Vec::new();
                let (mut depth, mut start) = (0, 0);
                for (i, c) in t.char_indices() {
                    match c { '(' => depth += 1, ')' => depth -= 1,
                              ',' if depth == 0 => { parts.push(t[start..i].trim().to_owned()); start = i + 1; }, _ => () }
                }
                parts.push(t[start..].trim().to_owned());
                parts.into_iter().map(serde_json::Value::String).collect()
            }
        },
        _ => return Err(format!("unsupported shape {v}")),
    };
    dims.iter().try_fold(1usize, |n, d| Ok(n * eval_usize(d, env)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn has3d_conditional_extent() {
        // ad/held-mpas_sw_aerosol_columns: the 2-D aerosol scratch (spectral, zs, denom) is nc*14 / nc*nz / nc words
        // when has3d == 0 and one word when has3d == 1 (rrtmg_sw_v842.py:436-439); the spec writes it as one extent.
        let v = |h: f64| serde_json::Value::String("[has3d + (1 - has3d)*nc*14]".into());
        for (h, want) in [(1.0, 1usize), (0.0, 2562 * 14)] {
            let env: Env = [("has3d".to_owned(), h), ("nc".to_owned(), 2562.0), ("nz".to_owned(), 59.0)].into();
            assert_eq!(shape_elements(&v(h), &env).unwrap(), want);
            assert_eq!(shape_elements(&serde_json::Value::String("[has3d + (1 - has3d)*nc*nz]".into()), &env).unwrap(),
                       if h == 1.0 { 1 } else { 2562 * 59 });
            assert_eq!(shape_elements(&serde_json::Value::String("[has3d + (1 - has3d)*nc]".into()), &env).unwrap(),
                       if h == 1.0 { 1 } else { 2562 });
        }
    }
    #[test]
    fn expressions() {
        let env: Env = [("nlev".to_owned(), 55.0), ("ncells".to_owned(), 2562.0), ("grid[0]".to_owned(), 21.0)].into();
        assert_eq!(eval("144*(nlev+1)*ncells", &env).unwrap(), 144.0 * 56.0 * 2562.0);
        assert_eq!(eval("ceil(ncells/256)", &env).unwrap(), 11.0);
        assert_eq!(eval("max(nlev,ncells)-1", &env).unwrap(), 2561.0);
        assert_eq!(eval("grid[0]*2", &env).unwrap(), 42.0);
        assert!(eval("missing+1", &env).is_err());
        assert_eq!(shape_elements(&serde_json::json!("[nlev+1,ncells]"), &env).unwrap(), 56 * 2562);
        assert_eq!(shape_elements(&serde_json::json!([56, "ncells"]), &env).unwrap(), 56 * 2562);
        assert_eq!(shape_elements(&serde_json::json!("[]"), &env).unwrap(), 1);
    }
}
