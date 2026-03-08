//! CompareANT — Compare two vectors for similarity.
//! Architecture: [vec_a, vec_b] -> 16 -> 16 -> 1

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::{Path, PathBuf};
use ternary_signal::PackedSignal;

const COMPARE_PROGRAM: &str = include_str!("../../runes/compare.rune");

pub struct CompareANT {
    inner: AtomicNeuralTransistor,
    thermo_path: Option<PathBuf>,
}

impl CompareANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                COMPARE_PROGRAM,
                Some(base.to_path_buf()),
            )?,
            thermo_path: None,
        })
    }

    /// Load with Thermogram persistence at the given base directory.
    pub fn with_thermogram(base_dir: &Path) -> Result<Self> {
        let crate_base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let thermo_path = base_dir.join("compare.thermo");
        let thermo_exists = thermo_path.exists();
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_thermogram(
                COMPARE_PROGRAM,
                Some(crate_base.to_path_buf()),
                "compare",
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

    /// Compare two vectors — returns similarity signal.
    pub fn compare(&mut self, a: &[PackedSignal], b: &[PackedSignal]) -> Result<PackedSignal> {
        let mut input = Vec::with_capacity(a.len() + b.len());
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        let output = self.inner.forward(&input)?;
        Ok(output.into_iter().next().unwrap_or(PackedSignal::ZERO))
    }

    pub fn are_similar(&mut self, a: &[PackedSignal], b: &[PackedSignal]) -> Result<bool> {
        Ok(self.compare(a, b)?.is_positive())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_loads() {
        assert!(CompareANT::new().is_ok());
    }

    #[test]
    fn test_compare() {
        let mut cmp = CompareANT::new().unwrap();
        let a: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8 * 8, 1)).collect();
        let b: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8 * 8, 1)).collect();
        assert!(cmp.compare(&a, &b).is_ok());
    }
}
