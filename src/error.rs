//! Error types for Atomic Neural Transistors

use thiserror::Error;

/// ANT-specific errors
#[derive(Error, Debug)]
pub enum AntError {
    /// TensorISA operation failed
    #[error("TensorISA error: {0}")]
    TensorIsa(#[from] ternsig::TernsigError),

    /// Thermogram operation failed
    #[error("Thermogram error: {0}")]
    Thermogram(String),

    /// Model loading failed
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// IO error
    #[error("IO error: {0}")]
    Io(String),

    /// Assembly error
    #[error("Assembly error: {0}")]
    Assembly(String),
}

/// Result type alias for ANT operations
pub type Result<T> = std::result::Result<T, AntError>;
