//! ClassifierANT - Load classifier.tisa.asm, run argmax

use crate::core::AtomicNeuralTransistor;
use crate::error::Result;
use std::path::Path;
use ternsig::TernarySignal;

const CLASSIFIER_TISA: &str = include_str!("../../tisa/classifier.tisa.asm");

pub struct ClassifierANT(AtomicNeuralTransistor);

impl ClassifierANT {
    /// Load default classifier
    pub fn new() -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_source(CLASSIFIER_TISA)?))
    }

    /// Load from file
    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self(AtomicNeuralTransistor::from_file(path)?))
    }

    /// Classify - returns class logits
    pub fn classify(&mut self, input: &[TernarySignal]) -> Result<Vec<TernarySignal>> {
        self.0.forward(input)
    }

    /// Predict - returns argmax class index
    pub fn predict(&mut self, input: &[TernarySignal]) -> Result<usize> {
        let output = self.classify(input)?;
        Ok(output
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.as_signed_i32())
            .map(|(i, _)| i)
            .unwrap_or(0))
    }

    pub fn input_dim(&self) -> usize { self.0.input_dim() }
    pub fn output_dim(&self) -> usize { self.0.output_dim() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_loads() {
        let classifier = ClassifierANT::new();
        assert!(classifier.is_ok());
    }

    #[test]
    fn test_classify() {
        let mut classifier = ClassifierANT::new().unwrap();
        let input: Vec<TernarySignal> = (0..classifier.input_dim())
            .map(|i| TernarySignal::positive((i * 8) as u8))
            .collect();
        let output = classifier.classify(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().len(), classifier.output_dim());
    }

    #[test]
    fn test_predict() {
        let mut classifier = ClassifierANT::new().unwrap();
        let input: Vec<TernarySignal> = (0..classifier.input_dim())
            .map(|_| TernarySignal::positive(128))
            .collect();
        let prediction = classifier.predict(&input);
        assert!(prediction.is_ok());
        assert!(prediction.unwrap() < classifier.output_dim());
    }
}
