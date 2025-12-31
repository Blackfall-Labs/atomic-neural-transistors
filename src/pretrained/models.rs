//! Pretrained model constructors

use candle_core::{Device, Result};
use std::path::Path;

use crate::ants::CompareTRM;
use crate::config::AtomicConfig;
use crate::core::AtomicTRM;

use super::loader::load_safetensors;

/// Load pretrained AreEqual ANT (99.5% accuracy)
///
/// Compares two 32-dim embeddings for equality
pub fn are_equal(device: &Device) -> Result<CompareTRM> {
    let path = Path::new(super::loader::models_dir()).join("are_equal.safetensors");
    let (vb, _data) = load_safetensors(&path, device)?;
    CompareTRM::new(32, vb)
}

/// Load pretrained IsEmpty ANT (100% accuracy)
///
/// Detects zero/null embeddings
pub fn is_empty(device: &Device) -> Result<AtomicTRM> {
    let path = Path::new(super::loader::models_dir()).join("is_empty.safetensors");
    let (vb, _data) = load_safetensors(&path, device)?;
    AtomicTRM::new(&AtomicConfig::tiny(32, 1), vb)
}

/// Load pretrained Contains ANT (~97% accuracy)
///
/// Checks if query appears in sequence
pub fn contains(device: &Device) -> Result<AtomicTRM> {
    let path = Path::new(super::loader::models_dir()).join("contains.safetensors");
    let (vb, _data) = load_safetensors(&path, device)?;
    AtomicTRM::new(&AtomicConfig::small(64, 1), vb)
}

/// Load pretrained HasDuplicate ANT (100% accuracy)
///
/// Detects duplicate values in sequence
pub fn has_duplicate(device: &Device) -> Result<AtomicTRM> {
    let path = Path::new(super::loader::models_dir()).join("has_duplicate.safetensors");
    let (vb, _data) = load_safetensors(&path, device)?;
    AtomicTRM::new(&AtomicConfig::small(64, 1), vb)
}

/// Check if pretrained models are available
pub fn models_available() -> bool {
    let dir = Path::new(super::loader::models_dir());
    dir.join("are_equal.safetensors").exists()
}
