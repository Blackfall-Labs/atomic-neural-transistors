//! AtomicNeuralTransistor - Load and execute .ternsig files
//!
//! Dimensions come from the file. No runtime config.
//! Learning uses mastery approach. No floats.

use crate::error::{AntError, Result};
use std::path::Path;
use ternsig::{
    vm::{assemble, AssembledProgram, Interpreter},
    Signal,
};

/// Atomic Neural Transistor - loads and executes a .ternsig file
pub struct AtomicNeuralTransistor {
    interpreter: Interpreter,
    input_dim: usize,
    output_dim: usize,
}

impl AtomicNeuralTransistor {
    /// Load from .ternsig file path
    pub fn from_file(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| AntError::Io(e.to_string()))?;
        Self::from_source(&source)
    }

    /// Load from .ternsig source string
    pub fn from_source(source: &str) -> Result<Self> {
        let program = assemble(source)
            .map_err(|e| AntError::Assembly(e.to_string()))?;
        Self::from_program(&program)
    }

    /// Load from assembled program
    pub fn from_program(program: &AssembledProgram) -> Result<Self> {
        let input_dim = program.input_shape.iter().product();
        let output_dim = program.output_shape.iter().product();
        let interpreter = Interpreter::from_program(program);

        Ok(Self {
            interpreter,
            input_dim,
            output_dim,
        })
    }

    /// Forward pass - Signal in, Signal out
    pub fn forward(&mut self, input: &[Signal]) -> Result<Vec<Signal>> {
        if input.len() != self.input_dim {
            return Err(AntError::ShapeMismatch {
                expected: self.input_dim.to_string(),
                got: input.len().to_string(),
            });
        }

        self.interpreter
            .forward(input)
            .map_err(|e| AntError::Assembly(e))
    }

    /// Forward pass - i32 in, i32 out
    pub fn forward_i32(&mut self, input: &[i32]) -> Result<Vec<i32>> {
        if input.len() != self.input_dim {
            return Err(AntError::ShapeMismatch {
                expected: self.input_dim.to_string(),
                got: input.len().to_string(),
            });
        }

        self.interpreter
            .forward_i32(input)
            .map_err(|e| AntError::Assembly(e))
    }

    /// Get input dimension (from file)
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension (from file)
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Access interpreter for learning/thermogram operations
    pub fn interpreter_mut(&mut self) -> &mut Interpreter {
        &mut self.interpreter
    }

    /// Access interpreter (read-only)
    pub fn interpreter(&self) -> &Interpreter {
        &self.interpreter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TERNSIG: &str = r#"
.registers
    H0: i32[4]
    H1: i32[4]

.program
    load_input H0
    relu H1, H0
    store_output H1
    halt
"#;

    #[test]
    fn test_from_source() {
        let ant = AtomicNeuralTransistor::from_source(TEST_TERNSIG);
        assert!(ant.is_ok());
        let ant = ant.unwrap();
        assert_eq!(ant.input_dim(), 4);
        assert_eq!(ant.output_dim(), 4);
    }

    #[test]
    fn test_forward_signal() {
        let mut ant = AtomicNeuralTransistor::from_source(TEST_TERNSIG).unwrap();
        let input = vec![
            Signal::positive(100),
            Signal::negative(50),
            Signal::positive(200),
            Signal::ZERO,
        ];
        let output = ant.forward(&input);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.len(), 4);
        // ReLU: positive stays, negative becomes zero
        assert!(output[0].magnitude > 0);
        assert_eq!(output[1].magnitude, 0); // was negative
        assert!(output[2].magnitude > 0);
        assert_eq!(output[3].magnitude, 0);
    }

    #[test]
    fn test_forward_i32() {
        let mut ant = AtomicNeuralTransistor::from_source(TEST_TERNSIG).unwrap();
        let input = vec![100, -50, 200, 0];
        let output = ant.forward_i32(&input);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.len(), 4);
        assert_eq!(output[0], 100);
        assert_eq!(output[1], 0); // ReLU clamps negative
        assert_eq!(output[2], 200);
        assert_eq!(output[3], 0);
    }

    #[test]
    fn test_shape_mismatch() {
        let mut ant = AtomicNeuralTransistor::from_source(TEST_TERNSIG).unwrap();
        let input = vec![Signal::ZERO; 2]; // wrong size
        let result = ant.forward(&input);
        assert!(result.is_err());
    }
}
