//! Error types for Atomic Neural Transistors

use thiserror::Error;

/// ANT-specific errors
#[derive(Error, Debug)]
pub enum AntError {
    /// Candle tensor operation failed
    #[error("Tensor operation failed: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Model loading failed
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
}

/// Result type alias for ANT operations
pub type Result<T> = std::result::Result<T, AntError>;
