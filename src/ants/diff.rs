//! DiffANT — Compute learned difference embedding between two vectors.
//! Architecture: [vec_a, vec_b] -> 24 -> 32

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternary_signal::Signal;

const DIFF_PROGRAM: &str = include_str!("../../runes/diff.rune");

pub struct DiffANT {
    inner: AtomicNeuralTransistor,
}

impl DiffANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                DIFF_PROGRAM,
                Some(base.to_path_buf()),
            )?,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: AtomicNeuralTransistor::from_file(path)?,
        })
    }

    /// Compute difference embedding between a and b.
    pub fn diff(&mut self, a: &[Signal], b: &[Signal]) -> Result<Vec<Signal>> {
        let mut input = Vec::with_capacity(a.len() + b.len());
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        self.inner.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_loads() {
        assert!(DiffANT::new().is_ok());
    }

    #[test]
    fn test_diff() {
        let mut diff = DiffANT::new().unwrap();
        let a: Vec<Signal> = (0..32).map(|i| Signal::new_raw(1, i as u8 * 8, 1)).collect();
        let b: Vec<Signal> = (0..32).map(|i| Signal::new_raw(-1, i as u8 * 8, 1)).collect();
        let result = diff.diff(&a, &b);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
