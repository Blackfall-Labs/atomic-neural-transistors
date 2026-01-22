//! # Atomic Neural Transistors
//!
//! Load and execute .tisa.asm files. Dimensions from file, not config.
//! Learning via mastery approach. No floats in neural computation.
//!
//! ## Core Usage
//!
//! ```rust,ignore
//! use atomic_neural_transistors::{ClassifierANT, TernarySignal};
//!
//! let mut classifier = ClassifierANT::new()?;
//! let output = classifier.classify(&input)?;
//! ```
//!
//! ## Composition Algebra
//!
//! Higher-order operations composed from primitives:
//!
//! ```rust,ignore
//! use atomic_neural_transistors::composition::{contains, has_duplicate, PerfectEquality};
//!
//! let checker = PerfectEquality;
//! assert!(contains(&checker, 5, &[1, 2, 3, 5, 7]));
//! assert!(has_duplicate(&checker, &[1, 2, 3, 2, 5]));
//! ```

pub mod ants;
pub mod composition;
pub mod core;
pub mod error;

pub use core::AtomicNeuralTransistor;
pub use error::{AntError, Result};
pub use ants::{ClassifierANT, CompareANT, DiffANT, GateANT, MergeANT};
pub use ternsig::TernarySignal;
