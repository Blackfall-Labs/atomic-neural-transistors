//! AtomicNeuralTransistor — Load and execute .rune scripts with ant_ml module.
//!
//! Replaces the ternsig v1 assembly VM with the Runes scripting engine.
//! Dimensions and behavior come from the .rune script. No runtime config.
//! Learning uses mastery approach. No floats.

use crate::error::{AntError, Result};
use crate::modules::ant_ml::{AntMlModule, AntRuntime};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use runes_core::engine::Engine;
use runes_core::value::Value;
use runes_parser::lexer::Lexer;
use runes_parser::parser::Parser;
use runes_eval::evaluator::Evaluator;
use ternary_signal::Signal;

/// Atomic Neural Transistor — loads and executes a .rune program with ant_ml verbs.
pub struct AtomicNeuralTransistor {
    source: String,
    engine: Engine,
    runtime: Arc<Mutex<AntRuntime>>,
    /// Cached evaluator with function definitions already loaded from the .rune source.
    /// Populated on first `eval_call` — avoids re-lexing/re-parsing/re-evaluating per call.
    evaluator: Option<Evaluator>,
}

impl AtomicNeuralTransistor {
    /// Load from .rune file path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| AntError::Io(e.to_string()))?;
        let base_path = path.parent().map(|p| p.to_path_buf());
        Self::from_source_with_base(&source, base_path)
    }

    /// Load from .rune source string.
    pub fn from_source(source: &str) -> Result<Self> {
        Self::from_source_with_base(source, None)
    }

    /// Load from source with an optional base path for weight file resolution.
    pub fn from_source_with_base(source: &str, base_path: Option<PathBuf>) -> Result<Self> {
        let mut rt = AntRuntime::new();
        if let Some(bp) = base_path {
            rt = rt.with_base_path(bp);
        }
        let runtime = Arc::new(Mutex::new(rt));

        let engine = Engine::builder()
            .namespace("default")
                .module(AntMlModule::new())
            .build()
            .map_err(|e| AntError::Runes(format!("engine build: {e:?}")))?;

        Ok(Self {
            source: source.to_string(),
            engine,
            runtime,
            evaluator: None,
        })
    }

    /// Execute a named function in the .rune script with the given input.
    pub fn call(&mut self, func_name: &str, input: &[Signal]) -> Result<Vec<Signal>> {
        let input_vals: Vec<Value> = input.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();

        let result = self.eval_call(func_name, vec![Value::Array(Arc::new(input_vals))])?;

        match result {
            Value::Array(arr) => {
                arr.iter()
                    .map(|v| match v {
                        Value::Integer(n) => Ok(Signal::from_current(*n as i32)),
                        _ => Err(AntError::ShapeMismatch {
                            expected: "integer array".into(),
                            got: format!("{}", v.type_name()),
                        }),
                    })
                    .collect()
            }
            Value::Integer(n) => {
                Ok(vec![Signal::from_current(n as i32)])
            }
            _ => Err(AntError::ShapeMismatch {
                expected: "array or integer".into(),
                got: result.type_name().into(),
            }),
        }
    }

    /// Execute a named function with arbitrary `Value` arguments.
    ///
    /// Use this when you need to pass handles, integers, or mixed types
    /// (not just a single Signal array).
    pub fn call_values(&mut self, func_name: &str, args: Vec<Value>) -> Result<Value> {
        self.eval_call(func_name, args)
    }

    /// Execute the .rune script's `forward` function with the given input.
    pub fn forward(&mut self, input: &[Signal]) -> Result<Vec<Signal>> {
        self.call("forward", input)
    }

    /// Ensure the evaluator is initialized with function definitions from the .rune source.
    /// Only lexes, parses, and evaluates the full script once — subsequent calls reuse the cached state.
    fn ensure_evaluator(&mut self) -> Result<()> {
        if self.evaluator.is_some() {
            return Ok(());
        }

        let tokens = Lexer::new(&self.source).tokenize()
            .map_err(|e| AntError::Runes(format!("lex: {e:?}")))?;
        let program = Parser::new(tokens).parse_program()
            .map_err(|e| AntError::Runes(format!("parse: {e:?}")))?;

        let mut evaluator = Evaluator::new();
        evaluator.set_host(self.runtime.clone());

        // Evaluate once to register all function definitions in scope
        evaluator.eval_with_engine(&program, &self.engine)
            .map_err(|e| AntError::Runes(format!("eval: {e}")))?;

        self.evaluator = Some(evaluator);
        Ok(())
    }

    /// Call a named function in the .rune script.
    ///
    /// On first call, lexes/parses/evaluates the full .rune source to register
    /// function definitions. Subsequent calls use `Evaluator::invoke` to call
    /// the function directly — no re-parsing, no import re-resolution.
    fn eval_call(&mut self, func_name: &str, args: Vec<Value>) -> Result<Value> {
        self.ensure_evaluator()?;

        let evaluator = self.evaluator.as_mut().unwrap();
        evaluator.invoke(func_name, &args, &self.engine)
            .map_err(|e| AntError::Runes(format!("call: {e}")))
    }

    /// Access the shared runtime (for direct weight manipulation).
    pub fn runtime(&self) -> &Arc<Mutex<AntRuntime>> {
        &self.runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_RUNE: &str = r#"rune "test" do
  version 1
end
use :ant_ml

def forward(input) do
    relu(input)
end"#;

    #[test]
    fn test_from_source() {
        let ant = AtomicNeuralTransistor::from_source(TEST_RUNE);
        assert!(ant.is_ok());
    }

    #[test]
    fn test_forward_relu() {
        let mut ant = AtomicNeuralTransistor::from_source(TEST_RUNE).unwrap();
        let input = vec![
            Signal::new_raw(1, 100, 1),   // positive
            Signal::new_raw(-1, 50, 1),   // negative
            Signal::new_raw(1, 200, 1),   // positive
            Signal::ZERO,                 // zero
        ];
        let output = ant.forward(&input).unwrap();
        assert_eq!(output.len(), 4);
        assert!(output[0].current() > 0);
        assert!(output[1].current() == 0); // ReLU clamps negative
        assert!(output[2].current() > 0);
        assert!(output[3].current() == 0);
    }

    #[test]
    fn test_zeros_verb() {
        let source = r#"rune "test" do
  version 1
end
use :ant_ml

def forward(input) do
    zeros(4)
end"#;
        let mut ant = AtomicNeuralTransistor::from_source(source).unwrap();
        let output = ant.forward(&[Signal::ZERO]).unwrap();
        assert_eq!(output.len(), 4);
        for s in &output {
            assert_eq!(s.current(), 0);
        }
    }
}
