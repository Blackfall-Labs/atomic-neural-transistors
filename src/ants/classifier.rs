//! ClassifierANT — Multi-class classifier using .rune script.
//! Architecture: 32 -> 24 -> 4, 3x recurrent passes.

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternary_signal::Signal;

const CLASSIFIER_PROGRAM: &str = include_str!("../../runes/classifier.rune");

pub struct ClassifierANT {
    inner: AtomicNeuralTransistor,
}

impl ClassifierANT {
    /// Load default classifier with weights resolved relative to crate root.
    pub fn new() -> Result<Self> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            inner: AtomicNeuralTransistor::from_source_with_base(
                CLASSIFIER_PROGRAM,
                Some(base.to_path_buf()),
            )?,
        })
    }

    /// Load from custom .rune file.
    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: AtomicNeuralTransistor::from_file(path)?,
        })
    }

    /// Classify — returns class signals as Signal vector.
    pub fn classify(&mut self, input: &[Signal]) -> Result<Vec<Signal>> {
        self.inner.forward(input)
    }

    /// Predict — returns argmax class index.
    pub fn predict(&mut self, input: &[Signal]) -> Result<usize> {
        let output = self.classify(input)?;
        Ok(output
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.current())
            .map(|(i, _)| i)
            .unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_loads() {
        let c = ClassifierANT::new();
        assert!(c.is_ok());
    }

    #[test]
    fn test_classify() {
        let mut c = ClassifierANT::new().unwrap();
        let input: Vec<Signal> = (0..32)
            .map(|i| Signal::new_raw(1, i as u8 * 8, 1))
            .collect();
        let result = c.classify(&input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn test_predict() {
        let mut c = ClassifierANT::new().unwrap();
        let input: Vec<Signal> = (0..32)
            .map(|i| Signal::new_raw(1, i as u8 * 8, 1))
            .collect();
        let result = c.predict(&input);
        assert!(result.is_ok());
        assert!(result.unwrap() < 4);
    }
}
