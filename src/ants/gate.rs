//! GateANT — Apply learned gating to a signal based on control.
//! Architecture: [signal, context] -> 16 -> 32 (sigmoid)

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::{Path, PathBuf};
use ternary_signal::PackedSignal;

const GATE_PROGRAM: &str = include_str!("../../runes/gate.rune");

pub struct GateANT {
    inner: AtomicNeuralTransistor,
    thermo_path: Option<PathBuf>,
}

impl GateANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                GATE_PROGRAM,
                Some(base.to_path_buf()),
            )?,
            thermo_path: None,
        })
    }

    /// Load with Thermogram persistence at the given base directory.
    pub fn with_thermogram(base_dir: &Path) -> Result<Self> {
        let crate_base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let thermo_path = base_dir.join("gate.thermo");
        let thermo_exists = thermo_path.exists();
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_thermogram(
                GATE_PROGRAM,
                Some(crate_base.to_path_buf()),
                "gate",
                if thermo_exists { Some(&thermo_path) } else { None },
            )?,
            thermo_path: Some(thermo_path),
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: AtomicNeuralTransistor::from_file(path)?,
            thermo_path: None,
        })
    }

    /// Save Thermogram to disk.
    pub fn save(&self) -> Result<()> {
        if let Some(tp) = &self.thermo_path {
            self.inner.save_thermogram(tp)?;
        }
        Ok(())
    }

    /// Gate signal based on control.
    pub fn gate(&mut self, signal: &[PackedSignal], control: &[PackedSignal]) -> Result<Vec<PackedSignal>> {
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
        let signal: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8 * 8, 1)).collect();
        let control: Vec<PackedSignal> = (0..32).map(|_| PackedSignal::pack(1, 128, 1)).collect();
        let result = gate.gate(&signal, &control);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
