//! Error types for Atomic Neural Transistors

use thiserror::Error;

/// ANT-specific errors
#[derive(Error, Debug)]
pub enum AntError {
    /// Runes script error
    #[error("Runes error: {0}")]
    Runes(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// IO error
    #[error("IO error: {0}")]
    Io(String),

    /// Weight format error
    #[error("Weight error: {0}")]
    Weight(String),
}

/// Result type alias for ANT operations
pub type Result<T> = std::result::Result<T, AntError>;
