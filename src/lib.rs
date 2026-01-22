//! # Atomic Neural Transistors
//!
//! Load and execute .tisa.asm files. Dimensions from file, not config.
//! Learning via mastery approach. No floats.
//!
//! ```rust,ignore
//! use atomic_neural_transistors::AtomicNeuralTransistor;
//! use std::path::Path;
//!
//! let mut ant = AtomicNeuralTransistor::from_file(Path::new("model.tisa.asm"))?;
//! let output = ant.forward(&input)?;
//! ```

pub mod ants;
pub mod core;
pub mod error;

pub use core::AtomicNeuralTransistor;
pub use error::{AntError, Result};
pub use ants::{ClassifierANT, CompareANT, DiffANT, GateANT, MergeANT};
pub use ternsig::TernarySignal;
