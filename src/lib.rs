//! # Atomic Neural Transistors
//!
//! Ultra-small (<5K param) composable neural primitives for real-time AI.
//!
//! ## What are ANTs?
//!
//! ANTs are the transistors of neural computing - atomic units that perform
//! single operations with high precision and compose into larger systems.
//!
//! | ANT | Params | Accuracy | Purpose |
//! |-----|--------|----------|---------|
//! | AreEqual | ~1.5K | 99.5% | Compare two embeddings |
//! | IsEmpty | ~1.5K | 100% | Detect zero embeddings |
//! | Contains | ~3K | ~97% | Query in sequence |
//! | HasDuplicate | ~3K | 100% | Duplicate detection |
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use atomic_neural_transistors::{AtomicTRM, AtomicConfig};
//! use candle_core::Device;
//! use candle_nn::{VarBuilder, VarMap};
//!
//! // Create a tiny ANT
//! let device = Device::Cpu;
//! let varmap = VarMap::new();
//! let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
//!
//! let ant = AtomicTRM::new(&AtomicConfig::tiny(32, 1), vb)?;
//! println!("Parameters: {}", ant.param_count()); // ~1.5K
//! ```
//!
//! ## Composition
//!
//! Complex operations compose from primitives without training:
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
pub mod config;
pub mod core;
pub mod error;
pub mod pretrained;

// Re-export main types at crate root
pub use config::AtomicConfig;
pub use core::AtomicTRM;
pub use error::{AntError, Result};

// Re-export specialized ANTs
pub use ants::{ClassifierTRM, CompareTRM, DiffTRM, GateTRM, MergeTRM};

// Re-export composition functions
pub use composition::{
    all_unique, contains, count_occurrences, find_positions, has_duplicate, EqualityChecker,
    PerfectEquality,
};
