//! CompareANT - Load compare.ternsig, compare two vectors

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::Signal;

const COMPARE_PROGRAM: &str = include_str!("../../ternsig/compare.ternsig");

pub struct CompareANT(AtomicNeuralTransistor);

impl CompareANT {
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(COMPARE_PROGRAM)?))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Compare two vectors - concatenate then forward
    pub fn compare(&mut self, a: &[Signal], b: &[Signal]) -> Result<Signal> {
        let mut input = Vec::with_capacity(a.len() + b.len());
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        let output = self.0.forward(&input)?;
        Ok(output.into_iter().next().unwrap_or(Signal::ZERO))
    }

    pub fn are_similar(&mut self, a: &[Signal], b: &[Signal]) -> Result<bool> {
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
        let a: Vec<Signal> = (0..32).map(|i| Signal::positive(i as u8)).collect();
        let b: Vec<Signal> = (0..32).map(|i| Signal::positive(i as u8)).collect();
        let result = cmp.compare(&a, &b);
        assert!(result.is_ok());
    }
}
