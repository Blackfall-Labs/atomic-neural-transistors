//! MergeANT - Load merge.ternsig, combine signals

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::Signal;

const MERGE_PROGRAM: &str = include_str!("../../ternsig/merge.ternsig");

pub struct MergeANT(AtomicNeuralTransistor);

impl MergeANT {
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(MERGE_PROGRAM)?))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Merge multiple signals into one
    pub fn merge(&mut self, signals: &[&[Signal]]) -> Result<Vec<Signal>> {
        let total: usize = signals.iter().map(|s| s.len()).sum();
        let mut input = Vec::with_capacity(total);
        for sig in signals {
            input.extend_from_slice(sig);
        }
        self.0.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_loads() {
        let merge = MergeANT::new();
        assert!(merge.is_ok());
    }

    #[test]
    fn test_merge() {
        let mut merge = MergeANT::new().unwrap();
        let sig1: Vec<Signal> = (0..32).map(|i| Signal::positive(i as u8)).collect();
        let sig2: Vec<Signal> = (0..32).map(|i| Signal::negative(i as u8)).collect();
        let result = merge.merge(&[&sig1, &sig2]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
