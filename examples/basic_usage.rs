//! Basic ANT usage example

use atomic_neural_transistors::{AtomicConfig, AtomicTRM};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn main() -> anyhow::Result<()> {
    println!("=== Atomic Neural Transistors: Basic Usage ===\n");

    let device = Device::Cpu;

    // Create a tiny ANT (binary classifier)
    println!("Creating a tiny ANT...");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = AtomicConfig::tiny(32, 1);
    let ant = AtomicTRM::new(&config, vb)?;

    println!("  Config: {:?}", config);
    println!("  Parameters: {} (< 5K!)\n", ant.param_count());

    // Forward pass
    println!("Running inference...");
    let input = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
    let output = ant.forward(&input)?;

    println!("  Input shape:  {:?}", input.dims());
    println!("  Output shape: {:?}", output.dims());
    println!("  Output values: {:?}\n", output.to_vec2::<f32>()?);

    // Create a small ANT (more capacity)
    println!("Creating a small ANT...");
    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);

    let config2 = AtomicConfig::small(64, 10);
    let ant2 = AtomicTRM::new(&config2, vb2)?;

    println!("  Config: {:?}", config2);
    println!("  Parameters: {}\n", ant2.param_count());

    println!("ANTs are the transistors of neural computing!");
    println!("- Tiny: ~1-2K params");
    println!("- Fast: <1ms inference");
    println!("- Composable: build complex systems from simple parts");

    Ok(())
}
