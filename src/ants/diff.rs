//! DiffANT - Load diff.tisa.asm, compute difference embedding

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::TernarySignal;

const DIFF_TISA: &str = include_str!("../../tisa/diff.tisa.asm");

pub struct DiffANT(AtomicNeuralTransistor);

impl DiffANT {
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(DIFF_TISA)?))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Compute difference embedding between a and b
    pub fn diff(&mut self, a: &[TernarySignal], b: &[TernarySignal]) -> Result<Vec<TernarySignal>> {
        let mut input = Vec::with_capacity(a.len() + b.len());
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        self.0.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_loads() {
        let diff = DiffANT::new();
        assert!(diff.is_ok());
    }

    #[test]
    fn test_diff() {
        let mut diff = DiffANT::new().unwrap();
        let a: Vec<TernarySignal> = (0..32).map(|i| TernarySignal::positive(i as u8)).collect();
        let b: Vec<TernarySignal> = (0..32).map(|i| TernarySignal::negative(i as u8)).collect();
        let result = diff.diff(&a, &b);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
