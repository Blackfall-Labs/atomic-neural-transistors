//! GateANT — Apply learned gating to a signal based on control.
//! Architecture: [signal, context] -> 16 -> 32 (sigmoid)

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternary_signal::Signal;

const GATE_PROGRAM: &str = include_str!("../../runes/gate.rune");

pub struct GateANT {
    inner: AtomicNeuralTransistor,
}

impl GateANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                GATE_PROGRAM,
                Some(base.to_path_buf()),
            )?,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: AtomicNeuralTransistor::from_file(path)?,
        })
    }

    /// Gate signal based on control.
    pub fn gate(&mut self, signal: &[Signal], control: &[Signal]) -> Result<Vec<Signal>> {
        let mut input = Vec::with_capacity(signal.len() + control.len());
        input.extend_from_slice(signal);
        input.extend_from_slice(control);
        self.inner.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_loads() {
        assert!(GateANT::new().is_ok());
    }

    #[test]
    fn test_gate() {
        let mut gate = GateANT::new().unwrap();
        let signal: Vec<Signal> = (0..32).map(|i| Signal::new_raw(1, i as u8 * 8, 1)).collect();
        let control: Vec<Signal> = (0..32).map(|_| Signal::new_raw(1, 128, 1)).collect();
        let result = gate.gate(&signal, &control);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
