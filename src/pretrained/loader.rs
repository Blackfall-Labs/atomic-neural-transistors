//! SafeTensors model loading utilities

use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use safetensors::SafeTensors;
use std::path::Path;

/// Load a VarBuilder from a SafeTensors file
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<(VarBuilder<'static>, Vec<u8>)> {
    let data = std::fs::read(path.as_ref()).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to read file: {}", e))
    })?;

    let tensors = SafeTensors::deserialize(&data).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to deserialize SafeTensors: {}", e))
    })?;

    let mut varmap = VarMap::new();

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            _ => DType::F32,
        };

        let tensor = candle_core::Tensor::from_raw_buffer(
            view.data(),
            dtype,
            &shape,
            device,
        )?;

        varmap.set_one(&name, tensor)?;
    }

    Ok((
        VarBuilder::from_varmap(&varmap, DType::F32, device),
        data,
    ))
}

/// Get the models directory path (relative to crate root)
pub fn models_dir() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/models")
}
