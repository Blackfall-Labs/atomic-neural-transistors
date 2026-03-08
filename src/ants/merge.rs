//! MergeANT — Merge multiple signals into one.
//! Architecture: [sig1, sig2] -> 24 -> 32

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::{Path, PathBuf};
use ternary_signal::PackedSignal;

const MERGE_PROGRAM: &str = include_str!("../../runes/merge.rune");

pub struct MergeANT {
    inner: AtomicNeuralTransistor,
    thermo_path: Option<PathBuf>,
}

impl MergeANT {
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                MERGE_PROGRAM,
                Some(base.to_path_buf()),
            )?,
            thermo_path: None,
        })
    }

    /// Load with Thermogram persistence at the given base directory.
    pub fn with_thermogram(base_dir: &Path) -> Result<Self> {
        let crate_base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let thermo_path = base_dir.join("merge.thermo");
        let thermo_exists = thermo_path.exists();
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_thermogram(
                MERGE_PROGRAM,
                Some(crate_base.to_path_buf()),
                "merge",
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

    /// Merge multiple signal vectors into one.
    pub fn merge(&mut self, signals: &[&[PackedSignal]]) -> Result<Vec<PackedSignal>> {
        let total: usize = signals.iter().map(|s| s.len()).sum();
        let mut input = Vec::with_capacity(total);
        for sig in signals {
            input.extend_from_slice(sig);
        }
        self.inner.forward(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_loads() {
        assert!(MergeANT::new().is_ok());
    }

    #[test]
    fn test_merge() {
        let mut merge = MergeANT::new().unwrap();
        let sig1: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8 * 8, 1)).collect();
        let sig2: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(-1, i as u8 * 8, 1)).collect();
        let result = merge.merge(&[&sig1, &sig2]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
