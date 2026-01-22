//! CompareANT - Load compare.tisa.asm, compare two vectors

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::TernarySignal;

const COMPARE_TISA: &str = include_str!("../../tisa/compare.tisa.asm");

pub struct CompareANT(AtomicNeuralTransistor);

impl CompareANT {
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(COMPARE_TISA)?))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Compare two vectors - concatenate then forward
    pub fn compare(&mut self, a: &[TernarySignal], b: &[TernarySignal]) -> Result<TernarySignal> {
        let mut input = Vec::with_capacity(a.len() + b.len());
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        let output = self.0.forward(&input)?;
        Ok(output.into_iter().next().unwrap_or(TernarySignal::zero()))
    }

    pub fn are_similar(&mut self, a: &[TernarySignal], b: &[TernarySignal]) -> Result<bool> {
        Ok(self.compare(a, b)?.is_positive())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_loads() {
        let cmp = CompareANT::new();
        assert!(cmp.is_ok());
    }

    #[test]
    fn test_compare() {
        let mut cmp = CompareANT::new().unwrap();
        let a: Vec<TernarySignal> = (0..32).map(|i| TernarySignal::positive(i as u8)).collect();
        let b: Vec<TernarySignal> = (0..32).map(|i| TernarySignal::positive(i as u8)).collect();
        let result = cmp.compare(&a, &b);
        assert!(result.is_ok());
    }
}
