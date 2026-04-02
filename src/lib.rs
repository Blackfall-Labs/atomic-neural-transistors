//! # Atomic Neural Transistors
//!
//! Ultra-small (<5K param) composable ternary neural primitives for CPU-only AI.
//! Uses Runes scripting engine with `ant_ml` module for ternary matrix operations.
//!
//! ## Core Usage
//!
//! ```rust,ignore
//! use atomic_neural_transistors::{ClassifierANT, Signal};
//!
//! let mut classifier = ClassifierANT::new()?;
//! let input: Vec<Signal> = vec![Signal::new_raw(1, 128, 1); 32];
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
pub mod encoding;
pub mod error;
pub mod learning;
pub mod thermal_mastery;
pub mod memory;
pub mod modules;
pub mod multiplex;
pub mod neuromod;
pub mod prediction;
pub mod salience;
pub mod testdata;
mod weights_init;

pub use core::{AtomicNeuralTransistor, WeightMatrix, ThermalWeight, ThermalWeightMatrix, ThermalMasteryConfig};
pub use error::{AntError, Result};
pub use ants::{ClassifierANT, CompareANT, DiffANT, GateANT, MergeANT};
pub use modules::ant_ml::{AntMlModule, AntRuntime};
pub use ternary_signal::Signal;
pub use neuromod::{Chemical, NeuromodState};
pub use prediction::{PredictionEngine, SurpriseSignal};
pub use salience::{SalienceRouter, RouteResult};
pub use multiplex::{MultiplexEncoder, AntSlot, MultiplexResult};
pub use encoding::{accumulate, encode_byte, encode_str, ENCODING_DIM};
pub use memory::{MemoryANT, PerceptionResult};
pub use runes_core::value::Value;
