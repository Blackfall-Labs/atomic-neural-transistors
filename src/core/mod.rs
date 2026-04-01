//! Core ANT implementation

mod atomic_neural_transistor;
pub mod weight_matrix;
pub mod thermal;

pub use atomic_neural_transistor::AtomicNeuralTransistor;
pub use weight_matrix::WeightMatrix;
pub use thermal::{ThermalWeight, ThermalWeightMatrix, ThermalMasteryConfig};
