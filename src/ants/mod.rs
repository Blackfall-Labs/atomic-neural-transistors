//! Specialized ANT variants
//!
//! Each ANT is a small, focused neural unit built on AtomicTRM.

mod classifier;
mod compare;
mod diff;
mod gate;
mod merge;

pub use classifier::ClassifierTRM;
pub use compare::CompareTRM;
pub use diff::DiffTRM;
pub use gate::GateTRM;
pub use merge::MergeTRM;
