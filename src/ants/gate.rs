//! GateANT - Load gate.ternsig, apply learned gating

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::Signal;

const GATE_PROGRAM: &str = include_str!("../../ternsig/gate.ternsig");

pub struct GateANT(AtomicNeuralTransistor);

impl GateANT {
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(GATE_PROGRAM)?))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Gate signal based on control
    pub fn gate(&mut self, signal: &[Signal], control: &[Signal]) -> Result<Vec<Signal>> {
        let mut input = Vec::with_capacity(signal.len() + control.len());
        input.extend_from_slice(signal);
        input.extend_from_slice(control);
        self.0.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_loads() {
        let gate = GateANT::new();
        assert!(gate.is_ok());
    }

    #[test]
    fn test_gate() {
        let mut gate = GateANT::new().unwrap();
        let signal: Vec<Signal> = (0..32).map(|i| Signal::positive(i as u8)).collect();
        let control: Vec<Signal> = (0..32).map(|_| Signal::positive(128)).collect();
        let result = gate.gate(&signal, &control);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
