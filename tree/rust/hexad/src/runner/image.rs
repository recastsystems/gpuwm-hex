// SPDX-License-Identifier: Apache-2.0
//! Prepare an intact captured module for the CUDA driver. The capture's bytes
//! and digest are verified before this helper is called.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format { Elf { pad: usize }, Ptx }

#[derive(Clone, Copy)]
enum Endian { Little, Big }

fn number(image: &[u8], offset: usize, len: usize, endian: Endian) -> Result<u64, String> {
    let end = offset.checked_add(len).ok_or("ELF offset overflow")?;
    let bytes = image.get(offset..end).ok_or("truncated ELF header")?;
    let mut value = 0_u64;
    match endian {
        Endian::Little => for &byte in bytes.iter().rev() { value = (value << 8) | u64::from(byte); },
        Endian::Big => for &byte in bytes { value = (value << 8) | u64::from(byte); },
    }
    Ok(value)
}

/// Return the bytes passed to cuModuleLoadData and what was appended.
/// ELF padding is restricted to at most eight zero bytes inside the final
/// declared program or section header table (COMMON-AD 77.2).
pub fn prepare(mut captured: Vec<u8>) -> Result<(Vec<u8>, Format), String> {
    if !captured.starts_with(b"\x7fELF") {
        captured.push(0); // cuModuleLoadData requires a NUL-terminated PTX string.
        return Ok((captured, Format::Ptx));
    }
    if captured.len() < 16 { return Err("truncated ELF identification".into()); }
    let endian = match captured[5] { 1 => Endian::Little, 2 => Endian::Big,
        _ => return Err("unsupported ELF byte order".into()) };
    let (header_size, phoff, shoff, ehsize, phentsize, phnum, shentsize, shnum, width, phmin, shmin) = match captured[4] {
        1 => (52, 28, 32, 40, 42, 44, 46, 48, 4, 32, 40),
        2 => (64, 32, 40, 52, 54, 56, 58, 60, 8, 56, 64),
        _ => return Err("unsupported ELF class".into()),
    };
    if captured.len() < header_size || number(&captured, ehsize, 2, endian)? < header_size as u64 {
        return Err("truncated or invalid ELF header".into());
    }
    let po = number(&captured, phoff, width, endian)?;
    let so = number(&captured, shoff, width, endian)?;
    let pe = number(&captured, phentsize, 2, endian)?;
    let pn = number(&captured, phnum, 2, endian)?;
    let se = number(&captured, shentsize, 2, endian)?;
    let sn = number(&captured, shnum, 2, endian)?;
    // Extended counts require interpreting section zero. Refuse rather than
    // guessing an extent from a possibly truncated record.
    if pn == 0xffff || (so != 0 && sn == 0) { return Err("extended ELF table counts are unsupported".into()); }
    let mut last: Option<(u64, u64)> = None;
    for (offset, entry, count, minimum) in [(po, pe, pn, phmin), (so, se, sn, shmin)] {
        if count == 0 { continue; }
        if offset == 0 || entry < minimum { return Err("invalid ELF header table".into()); }
        let end = entry.checked_mul(count).and_then(|n| offset.checked_add(n)).ok_or("ELF table extent overflow")?;
        if last.map_or(true, |(_, prior_end)| end > prior_end) { last = Some((offset, end)); }
    }
    let len = u64::try_from(captured.len()).map_err(|_| "ELF length overflow")?;
    let Some((start, end)) = last else { return Ok((captured, Format::Elf { pad: 0 })); };
    if end <= len { return Ok((captured, Format::Elf { pad: 0 })); }
    let gap = end - len;
    if gap > 8 || len < start { return Err(format!("captured ELF header-table gap {gap} is not padable")); }
    let pad = usize::try_from(gap).map_err(|_| "ELF gap overflow")?;
    captured.resize(captured.len().checked_add(pad).ok_or("ELF padded length overflow")?, 0);
    Ok((captured, Format::Elf { pad }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(len: usize) -> Vec<u8> {
        let mut image = vec![0; len];
        image[..4].copy_from_slice(b"\x7fELF");
        image[4] = 2; image[5] = 1; image[6] = 1;
        image[32..40].copy_from_slice(&64_u64.to_le_bytes());
        image[52..54].copy_from_slice(&64_u16.to_le_bytes());
        image[54..56].copy_from_slice(&56_u16.to_le_bytes());
        image[56..58].copy_from_slice(&2_u16.to_le_bytes());
        image
    }
    #[test]
    fn pads_only_small_final_table_tail() {
        for missing in 1..=8 {
            let input = fixture(176 - missing);
            let original = input.clone();
            let (image, format) = prepare(input).unwrap();
            assert_eq!(format, Format::Elf { pad: missing });
            assert_eq!(&image[..original.len()], original);
            assert_eq!(&image[original.len()..], vec![0; missing]);
        }
        assert!(prepare(fixture(167)).is_err());
        assert_eq!(prepare(fixture(176)).unwrap().1, Format::Elf { pad: 0 });
        assert_eq!(prepare(fixture(177)).unwrap().1, Format::Elf { pad: 0 });
    }
    #[test]
    fn rejects_truncation_before_final_table_and_unsupported_counts() {
        let mut image = fixture(64);
        image[32..40].copy_from_slice(&200_u64.to_le_bytes());
        assert!(prepare(image).is_err());
        let mut image = fixture(176);
        image[56..58].copy_from_slice(&0xffff_u16.to_le_bytes());
        assert!(prepare(image).is_err());
        let mut image = fixture(176);
        image[54..56].copy_from_slice(&1_u16.to_le_bytes());
        assert!(prepare(image).is_err());
        assert!(prepare(b"\x7fELF".to_vec()).is_err());
    }
    #[test]
    fn ptx_gets_a_terminator() {
        let (bytes, format) = prepare(b".version 8.0".to_vec()).unwrap();
        assert_eq!(format, Format::Ptx);
        assert_eq!(bytes, b".version 8.0\0");
    }
}
