//! CompareANT — Compare two vectors for similarity.
//! Architecture: [vec_a, vec_b] -> 16 -> 16 -> 1

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternary_signal::Signal;

const COMPARE_PROGRAM: &str = include_str!("../../runes/compare.rune");

pub struct CompareANT {
    inner: AtomicNeuralTransistor,
}

impl CompareANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                COMPARE_PROGRAM,
                Some(base.to_path_buf()),
            )?,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: AtomicNeuralTransistor::from_file(path)?,
        })
    }

    /// Compare two vectors — returns similarity signal.
    pub fn compare(&mut self, a: &[Signal], b: &[Signal]) -> Result<Signal> {
        let mut input = Vec::with_capacity(a.len() + b.len());
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        let output = self.inner.forward(&input)?;
        Ok(output.into_iter().next().unwrap_or(Signal::ZERO))
    }

    pub fn are_similar(&mut self, a: &[Signal], b: &[Signal]) -> Result<bool> {
        Ok(self.compare(a, b)?.current() > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_loads() {
        assert!(CompareANT::new().is_ok());
    }

    #[test]
    fn test_compare() {
        let mut cmp = CompareANT::new().unwrap();
        let a: Vec<Signal> = (0..32).map(|i| Signal::new_raw(1, i as u8 * 8, 1)).collect();
        let b: Vec<Signal> = (0..32).map(|i| Signal::new_raw(1, i as u8 * 8, 1)).collect();
        assert!(cmp.compare(&a, &b).is_ok());
    }
}
