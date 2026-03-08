//! Thermogram initialization — generates `.thermo` files with zero-initialized synaptic strengths.
//!
//! Run via `cargo test generate_thermo_files -- --ignored` to create thermogram files.
//! Note: ANTs now self-initialize zeros when no .thermo file exists, so this is optional.

#[cfg(test)]
mod tests {
    use thermogram::{Delta, PlasticityRule, Thermogram};
    use ternary_signal::PackedSignal;
    use std::path::Path;

    fn write_thermo(path: &str, name: &str, keys: &[(&str, usize, usize)]) {
        let p = Path::new(path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut thermo = Thermogram::new(name, PlasticityRule::stdp_like());
        for (key, rows, cols) in keys {
            let data = vec![PackedSignal::ZERO; rows * cols];
            let prev_hash = thermo.dirty_chain.head_hash.clone();
            let delta = Delta::create(*key, data, "weights_init");
            let _ = prev_hash; // chain managed by apply_delta
            thermo.apply_delta(delta).unwrap();
            println!("  {key} ({rows}x{cols} = {} synaptic strengths)", rows * cols);
        }
        thermo.save(p).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));
        println!("wrote {path}");
    }

    #[test]
    #[ignore] // Run manually: cargo test generate_thermo_files -- --ignored
    fn generate_thermo_files() {
        let base = env!("CARGO_MANIFEST_DIR");

        write_thermo(
            &format!("{base}/weights/classifier.thermo"),
            "classifier",
            &[
                ("classifier.w_in", 24, 32),
                ("classifier.w_rec", 24, 24),
                ("classifier.w_gate", 24, 24),
                ("classifier.w_out", 4, 24),
            ],
        );

        write_thermo(
            &format!("{base}/weights/compare.thermo"),
            "compare",
            &[
                ("compare.w_in", 16, 64),
                ("compare.w_hidden", 16, 16),
                ("compare.w_out", 1, 16),
            ],
        );

        write_thermo(
            &format!("{base}/weights/diff.thermo"),
            "diff",
            &[
                ("diff.w_in", 24, 64),
                ("diff.w_out", 32, 24),
            ],
        );

        write_thermo(
            &format!("{base}/weights/gate.thermo"),
            "gate",
            &[
                ("gate.w_in", 16, 64),
                ("gate.w_out", 32, 16),
            ],
        );

        write_thermo(
            &format!("{base}/weights/merge.thermo"),
            "merge",
            &[
                ("merge.w_in", 24, 64),
                ("merge.w_out", 32, 24),
            ],
        );
    }
}
