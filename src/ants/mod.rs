//! Specialized ANT variants
//!
//! Each ANT is a small, focused neural unit built on AtomicNeuralTransistor.

mod classifier;
mod compare;
mod diff;
mod gate;
mod merge;

pub use classifier::ClassifierANT;
pub use compare::CompareANT;
pub use diff::DiffANT;
pub use gate::GateANT;
pub use merge::MergeANT;
