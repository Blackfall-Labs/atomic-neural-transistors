//! DiffANT — Compute learned difference embedding between two vectors.
//! Architecture: [vec_a, vec_b] -> 24 -> 32

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::{Path, PathBuf};
use ternary_signal::PackedSignal;

const DIFF_PROGRAM: &str = include_str!("../../runes/diff.rune");

pub struct DiffANT {
    inner: AtomicNeuralTransistor,
    thermo_path: Option<PathBuf>,
}

impl DiffANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                DIFF_PROGRAM,
                Some(base.to_path_buf()),
            )?,
            thermo_path: None,
        })
    }

    /// Load with Thermogram persistence at the given base directory.
    pub fn with_thermogram(base_dir: &Path) -> Result<Self> {
        let crate_base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let thermo_path = base_dir.join("diff.thermo");
        let thermo_exists = thermo_path.exists();
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_thermogram(
                DIFF_PROGRAM,
                Some(crate_base.to_path_buf()),
                "diff",
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

    /// Compute difference embedding between a and b.
    pub fn diff(&mut self, a: &[PackedSignal], b: &[PackedSignal]) -> Result<Vec<PackedSignal>> {
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
        let a: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8 * 8, 1)).collect();
        let b: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(-1, i as u8 * 8, 1)).collect();
        let result = diff.diff(&a, &b);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
