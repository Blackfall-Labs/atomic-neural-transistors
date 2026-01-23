//! DiffANT - Load diff.ternsig, compute difference embedding

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::Signal;

const DIFF_PROGRAM: &str = include_str!("../../ternsig/diff.ternsig");

pub struct DiffANT(AtomicNeuralTransistor);

impl DiffANT {
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(DIFF_PROGRAM)?))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Compute difference embedding between a and b
    pub fn diff(&mut self, a: &[Signal], b: &[Signal]) -> Result<Vec<Signal>> {
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
        let a: Vec<Signal> = (0..32).map(|i| Signal::positive(i as u8)).collect();
        let b: Vec<Signal> = (0..32).map(|i| Signal::negative(i as u8)).collect();
        let result = diff.diff(&a, &b);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
