//! MergeANT — Merge multiple signals into one.
//! Architecture: [sig1, sig2] -> 24 -> 32

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternary_signal::Signal;

const MERGE_PROGRAM: &str = include_str!("../../runes/merge.rune");

pub struct MergeANT {
    inner: AtomicNeuralTransistor,
}

impl MergeANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                MERGE_PROGRAM,
                Some(base.to_path_buf()),
            )?,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: AtomicNeuralTransistor::from_file(path)?,
        })
    }

    /// Merge multiple signal vectors into one.
    pub fn merge(&mut self, signals: &[&[Signal]]) -> Result<Vec<Signal>> {
        let total: usize = signals.iter().map(|s| s.len()).sum();
        let mut input = Vec::with_capacity(total);
        for sig in signals {
            input.extend_from_slice(sig);
        }
        self.inner.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_loads() {
        assert!(MergeANT::new().is_ok());
    }

    #[test]
    fn test_merge() {
        let mut merge = MergeANT::new().unwrap();
        let sig1: Vec<Signal> = (0..32).map(|i| Signal::new_raw(1, i as u8 * 8, 1)).collect();
        let sig2: Vec<Signal> = (0..32).map(|i| Signal::new_raw(-1, i as u8 * 8, 1)).collect();
        let result = merge.merge(&[&sig1, &sig2]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
