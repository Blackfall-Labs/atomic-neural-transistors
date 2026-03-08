//! `ant_ml` Runes module — ML operations for Atomic Neural Transistors.
//!
//! Provides ternary matrix operations, activation functions, weight I/O,
//! and mastery learning as Runes verbs.
//!
//! Usage in .rune scripts:
//! ```rune
//! use :ant_ml
//! w = load_weights("weights/classifier_l1.ant")
//! h = matmul(input, w, 24, 32)
//! h = relu(h)
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use runes_core::error::RuneError;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;
use ternary_signal::PackedSignal;
use thermogram::{Delta, Signal, Thermogram};

use crate::core::weight_matrix::{WeightMatrix, packed_from_current};
use crate::learning::{MasteryConfig, MasteryState};

use crate::neuromod::{Chemical, NeuromodState};
use crate::prediction::PredictionEngine;
use crate::salience::SalienceRouter;

// Handle type IDs
const HANDLE_WEIGHT_MATRIX: u32 = 1;
const HANDLE_NEUROMOD: u32 = 2;
const HANDLE_PREDICTOR: u32 = 3;
const HANDLE_SALIENCE: u32 = 4;

/// Runtime state shared between the ANT host and the Runes engine.
pub struct AntRuntime {
    /// Synaptic strength matrices keyed by handle ID.
    weights: HashMap<u64, WeightMatrix>,
    /// Mastery learning state keyed by handle ID.
    mastery: HashMap<u64, MasteryState>,
    /// Thermogram for persistent synaptic strength history.
    thermogram: Option<Thermogram>,
    /// Map from semantic key to handle ID (for load_synaptic deduplication).
    synaptic_keys: HashMap<String, u64>,
    /// Neuromodulator states keyed by handle ID.
    neuromods: HashMap<u64, NeuromodState>,
    /// Prediction engines keyed by handle ID.
    predictors: HashMap<u64, PredictionEngine>,
    /// Salience routers keyed by handle ID.
    salience: HashMap<u64, SalienceRouter>,
    /// Next handle ID.
    next_handle: u64,
    /// Base path for resolving relative weight file paths.
    pub base_path: Option<PathBuf>,
}

impl Default for AntRuntime {
    fn default() -> Self {
        Self {
            weights: HashMap::new(),
            mastery: HashMap::new(),
            neuromods: HashMap::new(),
            predictors: HashMap::new(),
            salience: HashMap::new(),
            thermogram: None,
            synaptic_keys: HashMap::new(),
            next_handle: 0,
            base_path: None,
        }
    }
}

impl AntRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base path for resolving relative weight file paths.
    pub fn with_base_path(mut self, path: PathBuf) -> Self {
        self.base_path = Some(path);
        self
    }

    fn alloc_handle(&mut self) -> u64 {
        let id = self.next_handle;
        self.next_handle += 1;
        id
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else if let Some(base) = &self.base_path {
            base.join(p)
        } else {
            p
        }
    }

    /// Insert a weight matrix, returning its handle ID.
    pub fn insert_weights(&mut self, wm: WeightMatrix) -> u64 {
        let id = self.alloc_handle();
        self.weights.insert(id, wm);
        id
    }

    /// Get a weight matrix by handle.
    pub fn get_weights(&self, handle: u64) -> Option<&WeightMatrix> {
        self.weights.get(&handle)
    }

    /// Get a mutable weight matrix by handle.
    pub fn get_weights_mut(&mut self, handle: u64) -> Option<&mut WeightMatrix> {
        self.weights.get_mut(&handle)
    }

    /// Set the Thermogram for persistent synaptic strength storage.
    pub fn with_thermogram(mut self, thermo: Thermogram) -> Self {
        self.thermogram = Some(thermo);
        self
    }

    /// Set a Thermogram on an already-constructed runtime.
    pub fn set_thermogram(&mut self, thermo: Thermogram) {
        self.thermogram = Some(thermo);
    }

    /// Get a reference to the Thermogram.
    pub fn thermogram(&self) -> Option<&Thermogram> {
        self.thermogram.as_ref()
    }

    /// Get a mutable reference to the Thermogram.
    pub fn thermogram_mut(&mut self) -> Option<&mut Thermogram> {
        self.thermogram.as_mut()
    }

    /// Number of loaded synaptic strength matrices.
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Number of active mastery states.
    pub fn mastery_count(&self) -> usize {
        self.mastery.len()
    }

    /// Get mastery state for a handle.
    pub fn get_mastery(&self, handle: u64) -> Option<&MasteryState> {
        self.mastery.get(&handle)
    }

    /// List all synaptic keys and their handle IDs.
    pub fn synaptic_key_handles(&self) -> &HashMap<String, u64> {
        &self.synaptic_keys
    }
}

/// The ant_ml Runes module.
pub struct AntMlModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl AntMlModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(MatmulVerb),
                Box::new(ReluVerb),
                Box::new(SigmoidVerb),
                Box::new(TanhActVerb),
                Box::new(SoftmaxVerb),
                Box::new(ArgmaxVerb),
                Box::new(DotVerb),
                Box::new(PackVerb),
                Box::new(UnpackVerb),
                Box::new(MulVerb),
                Box::new(AddVerb),
                Box::new(ShiftVerb),
                Box::new(SliceVerb),
                Box::new(LoadWeightsVerb),
                Box::new(SaveWeightsVerb),
                Box::new(ZerosVerb),
                Box::new(MasteryUpdateVerb),
                Box::new(MasteryStateVerb),
                Box::new(LoadSynapticVerb),
                Box::new(SaveSynapticVerb),
                Box::new(PersistThermoVerb),
                Box::new(NeuromodNewVerb),
                Box::new(NeuromodInjectVerb),
                Box::new(NeuromodTickVerb),
                Box::new(NeuromodGateVerb),
                Box::new(PredictNewVerb),
                Box::new(PredictObserveVerb),
                // Utility verbs
                Box::new(SubVerb),
                Box::new(AbsVerb),
                Box::new(NegateVerb),
                Box::new(ConcatVerb),
                Box::new(SumVerb),
                // Salience routing verbs
                Box::new(SalienceNewVerb),
                Box::new(SalienceRouteVerb),
                Box::new(SalienceTrainVerb),
                // Neuromod inspection
                Box::new(NeuromodReadVerb),
                // Stride verb
                Box::new(StrideSliceVerb),
            ],
        }
    }
}

impl Module for AntMlModule {
    fn name(&self) -> &str {
        "ant_ml"
    }

    fn version(&self) -> ModuleVersion {
        ModuleVersion::new(0, 6, 0)
    }

    fn verbs(&self) -> &[Box<dyn Verb>] {
        &self.verbs
    }
}

// ---------------------------------------------------------------------------
// Helper: extract runtime from EvalContext
// ---------------------------------------------------------------------------

fn with_runtime<F, R>(ctx: &mut EvalContext, f: F) -> Result<R, RuneError>
where
    F: FnOnce(&mut AntRuntime) -> Result<R, RuneError>,
{
    let span = ctx.span;
    let arc = ctx.host.downcast_mut::<Arc<Mutex<AntRuntime>>>()
        .ok_or_else(|| RuneError::type_error("ANT runtime not available", Some(span)))?;
    let mut guard = arc.lock().map_err(|_| RuneError::type_error("runtime lock poisoned", Some(span)))?;
    f(&mut guard)
}

// ---------------------------------------------------------------------------
// Helper: convert between Value arrays and PackedSignal vectors
// ---------------------------------------------------------------------------

fn packed_to_values(signals: &[PackedSignal]) -> Value {
    Value::Array(signals.iter().map(|s| Value::Integer(s.as_u8() as i64)).collect())
}

fn require_int(args: &[Value], idx: usize, name: &str, span: runes_core::Span) -> Result<i64, RuneError> {
    match args.get(idx) {
        Some(Value::Integer(n)) => Ok(*n),
        _ => Err(RuneError::argument(format!("{name} requires integer at position {idx}"), Some(span))),
    }
}

fn require_str(args: &[Value], idx: usize, name: &str, span: runes_core::Span) -> Result<String, RuneError> {
    match args.get(idx) {
        Some(Value::String(s)) => Ok(s.to_string()),
        _ => Err(RuneError::argument(format!("{name} requires string at position {idx}"), Some(span))),
    }
}

fn require_array(args: &[Value], idx: usize, name: &str, span: runes_core::Span) -> Result<Vec<PackedSignal>, RuneError> {
    match args.get(idx) {
        Some(Value::Array(arr)) => {
            arr.iter()
                .map(|v| match v {
                    Value::Integer(n) => Ok(PackedSignal::from_raw(*n as u8)),
                    _ => Err(RuneError::type_error(format!("{name}: array must contain integers"), Some(span))),
                })
                .collect()
        }
        _ => Err(RuneError::argument(format!("{name} requires array at position {idx}"), Some(span))),
    }
}

fn require_handle(args: &[Value], idx: usize, name: &str, expected_type: u32, span: runes_core::Span) -> Result<u64, RuneError> {
    match args.get(idx) {
        Some(Value::Handle(ht, id)) if ht.0 == expected_type => Ok(*id),
        Some(Value::Handle(..)) => Err(RuneError::type_error(format!("{name}: wrong handle type"), Some(span))),
        _ => Err(RuneError::argument(format!("{name} requires handle at position {idx}"), Some(span))),
    }
}

// ---------------------------------------------------------------------------
// Verb implementations
// ---------------------------------------------------------------------------

/// matmul(input, weights_handle, rows, cols) → Array
struct MatmulVerb;
impl Verb for MatmulVerb {
    fn name(&self) -> &str { "matmul" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let input = require_array(args, 0, "matmul", span)?;
        let handle = require_handle(args, 1, "matmul", HANDLE_WEIGHT_MATRIX, span)?;

        with_runtime(ctx, |rt| {
            let wm = rt.weights.get(&handle)
                .ok_or_else(|| RuneError::argument("matmul: invalid weight handle", Some(span)))?;
            let output = wm.matmul(&input);
            Ok(packed_to_values(&output))
        })
    }
}

/// relu(signals) → Array
struct ReluVerb;
impl Verb for ReluVerb {
    fn name(&self) -> &str { "relu" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "relu", span)?;
        let output: Vec<PackedSignal> = signals.iter().map(|s| {
            if s.is_negative() {
                PackedSignal::ZERO
            } else {
                *s
            }
        }).collect();
        Ok(packed_to_values(&output))
    }
}

/// sigmoid(signals) → Array
/// Ternary sigmoid: threshold curve via LUT mapping magnitude to sigmoid-like response.
struct SigmoidVerb;
impl Verb for SigmoidVerb {
    fn name(&self) -> &str { "sigmoid" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "sigmoid", span)?;
        // Sigmoid LUT: maps current magnitude to 0-255 range with S-curve
        // For ternary signals, we threshold: large negative → 0, zero → 128, large positive → 255
        let output: Vec<PackedSignal> = signals.iter().map(|s| {
            let c = s.current();
            let sigmoid_val = ternary_sigmoid(c);
            PackedSignal::pack(1, sigmoid_val, 1)
        }).collect();
        Ok(packed_to_values(&output))
    }
}

/// Integer sigmoid approximation: maps i32 current to 0-255 output magnitude.
fn ternary_sigmoid(current: i32) -> u8 {
    // Piecewise linear approximation of sigmoid * 255
    // Centered at 0, saturates at roughly ±512
    if current <= -512 {
        0
    } else if current >= 512 {
        255
    } else {
        // Linear region: (current + 512) * 255 / 1024
        ((current as i64 + 512) * 255 / 1024) as u8
    }
}

/// tanh_act(signals) → Array
/// Symmetric threshold via LUT.
struct TanhActVerb;
impl Verb for TanhActVerb {
    fn name(&self) -> &str { "tanh_act" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "tanh_act", span)?;
        let output: Vec<PackedSignal> = signals.iter().map(|s| {
            let c = s.current();
            let (pol, mag) = ternary_tanh(c);
            PackedSignal::pack(pol, mag, 1)
        }).collect();
        Ok(packed_to_values(&output))
    }
}

/// Integer tanh approximation: maps i32 current to (polarity, magnitude).
fn ternary_tanh(current: i32) -> (i8, u8) {
    if current == 0 {
        return (0, 0);
    }
    let pol: i8 = if current > 0 { 1 } else { -1 };
    let abs = current.unsigned_abs();
    let mag = if abs >= 512 { 255 } else { (abs * 255 / 512) as u8 };
    (pol, mag)
}

/// softmax(signals) → Array
/// Ternary softmax: normalize magnitudes to sum=255, preserve polarity.
struct SoftmaxVerb;
impl Verb for SoftmaxVerb {
    fn name(&self) -> &str { "softmax" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "softmax", span)?;
        if signals.is_empty() {
            return Ok(packed_to_values(&[]));
        }

        // Shift currents so min is 0, then normalize
        let currents: Vec<i32> = signals.iter().map(|s| s.current()).collect();
        let min_c = *currents.iter().min().unwrap();
        let shifted: Vec<u64> = currents.iter().map(|&c| (c as i64 - min_c as i64) as u64).collect();
        let total: u64 = shifted.iter().sum();

        let output: Vec<PackedSignal> = if total == 0 {
            // All equal: uniform distribution
            let uniform = (255 / signals.len() as u8).max(1);
            signals.iter().map(|_| PackedSignal::pack(1, uniform, 1)).collect()
        } else {
            shifted.iter().map(|&s| {
                let mag = ((s * 255) / total) as u8;
                PackedSignal::pack(1, mag, 1)
            }).collect()
        };
        Ok(packed_to_values(&output))
    }
}

/// argmax(signals) → Integer
struct ArgmaxVerb;
impl Verb for ArgmaxVerb {
    fn name(&self) -> &str { "argmax" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "argmax", span)?;
        let idx = signals
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.current())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(Value::Integer(idx as i64))
    }
}

/// dot(a, b) → Integer
struct DotVerb;
impl Verb for DotVerb {
    fn name(&self) -> &str { "dot" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let a = require_array(args, 0, "dot", span)?;
        let b = require_array(args, 1, "dot", span)?;
        if a.len() != b.len() {
            return Err(RuneError::argument("dot: vectors must have same length", Some(span)));
        }
        let sum: i64 = a.iter().zip(b.iter())
            .map(|(x, y)| x.current() as i64 * y.current() as i64)
            .sum();
        Ok(Value::Integer(sum))
    }
}

/// mul(a, b) → Array — element-wise multiply, raw product (no normalization)
struct MulVerb;
impl Verb for MulVerb {
    fn name(&self) -> &str { "mul" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let a = require_array(args, 0, "mul", span)?;
        let b = require_array(args, 1, "mul", span)?;
        if a.len() != b.len() {
            return Err(RuneError::argument("mul: arrays must have same length", Some(span)));
        }
        let output: Vec<PackedSignal> = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let product = x.current() as i64 * y.current() as i64;
                packed_from_current(product as i32)
            })
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// add(a, b) → Array — element-wise add
struct AddVerb;
impl Verb for AddVerb {
    fn name(&self) -> &str { "add" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let a = require_array(args, 0, "add", span)?;
        let b = require_array(args, 1, "add", span)?;
        if a.len() != b.len() {
            return Err(RuneError::argument("add: arrays must have same length", Some(span)));
        }
        let output: Vec<PackedSignal> = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let sum = x.current() as i64 + y.current() as i64;
                packed_from_current(sum as i32)
            })
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// shift(a, n) → Array — right-shift all elements by n bits
struct ShiftVerb;
impl Verb for ShiftVerb {
    fn name(&self) -> &str { "shift" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "shift", span)?;
        let n = require_int(args, 1, "shift", span)? as u32;
        let output: Vec<PackedSignal> = signals.iter()
            .map(|s| {
                let shifted = s.current() >> n;
                packed_from_current(shifted)
            })
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// slice(a, start, len) → Array — extract subarray
struct SliceVerb;
impl Verb for SliceVerb {
    fn name(&self) -> &str { "slice" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "slice", span)?;
        let start = require_int(args, 1, "slice", span)? as usize;
        let len = require_int(args, 2, "slice", span)? as usize;
        if start + len > signals.len() {
            return Err(RuneError::argument(
                format!("slice: range {}..{} out of bounds (len={})", start, start + len, signals.len()),
                Some(span),
            ));
        }
        Ok(packed_to_values(&signals[start..start + len]))
    }
}

/// pack(polarity, magnitude, multiplier) → Integer (PackedSignal byte)
struct PackVerb;
impl Verb for PackVerb {
    fn name(&self) -> &str { "pack" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let pol = require_int(args, 0, "pack", span)? as i8;
        let mag = require_int(args, 1, "pack", span)? as u8;
        let mul = require_int(args, 2, "pack", span)? as u8;
        Ok(Value::Integer(PackedSignal::pack(pol, mag, mul).as_u8() as i64))
    }
}

/// unpack(packed_byte) → Array [pol, mag, mul, current]
struct UnpackVerb;
impl Verb for UnpackVerb {
    fn name(&self) -> &str { "unpack" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let byte = require_int(args, 0, "unpack", span)? as u8;
        let ps = PackedSignal::from_raw(byte);
        Ok(Value::Array(vec![
            Value::Integer(ps.polarity() as i64),
            Value::Integer(ps.magnitude() as i64),
            Value::Integer(ps.multiplier() as i64),
            Value::Integer(ps.current() as i64),
        ]))
    }
}

/// load_weights(path) → Handle
/// Deprecated: use load_synaptic instead.
struct LoadWeightsVerb;
impl Verb for LoadWeightsVerb {
    fn name(&self) -> &str { "load_weights" }
    #[allow(deprecated)]
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let path_str = require_str(args, 0, "load_weights", span)?;

        with_runtime(ctx, |rt| {
            let resolved = rt.resolve_path(&path_str);
            let wm = WeightMatrix::load(&resolved)
                .map_err(|e| RuneError::argument(format!("load_weights: {e}"), Some(span)))?;
            let id = rt.insert_weights(wm);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), id))
        })
    }
}

/// save_weights(handle, path) → Nil
/// Deprecated: use save_synaptic instead.
struct SaveWeightsVerb;
impl Verb for SaveWeightsVerb {
    fn name(&self) -> &str { "save_weights" }
    #[allow(deprecated)]
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "save_weights", HANDLE_WEIGHT_MATRIX, span)?;
        let path_str = require_str(args, 1, "save_weights", span)?;

        with_runtime(ctx, |rt| {
            let resolved = rt.resolve_path(&path_str);
            let wm = rt.weights.get(&handle)
                .ok_or_else(|| RuneError::argument("save_weights: invalid handle", Some(span)))?;
            wm.save(&resolved)
                .map_err(|e| RuneError::argument(format!("save_weights: {e}"), Some(span)))?;
            Ok(Value::Nil)
        })
    }
}

/// zeros(size) → Array
struct ZerosVerb;
impl Verb for ZerosVerb {
    fn name(&self) -> &str { "zeros" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let size = require_int(args, 0, "zeros", span)? as usize;
        let arr: Vec<Value> = vec![Value::Integer(PackedSignal::ZERO.as_u8() as i64); size];
        Ok(Value::Array(arr))
    }
}

/// mastery_update(weights_handle, input, output, target, config_map) → Handle
/// Runs one mastery learning step. If no mastery state exists for this handle, creates one.
struct MasteryUpdateVerb;
impl Verb for MasteryUpdateVerb {
    fn name(&self) -> &str { "mastery_update" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "mastery_update", HANDLE_WEIGHT_MATRIX, span)?;
        let input = require_array(args, 1, "mastery_update", span)?;
        let output = require_array(args, 2, "mastery_update", span)?;
        let target = require_array(args, 3, "mastery_update", span)?;

        // Optional config from 5th arg as array [threshold, decay, gate]
        let config = if let Some(Value::Array(arr)) = args.get(4) {
            MasteryConfig {
                pressure_threshold: arr.first().and_then(|v| v.as_integer()).unwrap_or(3) as i32,
                decay_rate: arr.get(1).and_then(|v| v.as_integer()).unwrap_or(1) as i32,
                participation_gate: arr.get(2).and_then(|v| v.as_integer()).unwrap_or(5) as u32,
            }
        } else {
            MasteryConfig::default()
        };

        with_runtime(ctx, |rt| {
            let wm = rt.weights.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("mastery_update: invalid handle", Some(span)))?;
            let weight_count = wm.data.len();

            let state = rt.mastery.entry(handle).or_insert_with(|| {
                MasteryState::new(weight_count, config.clone())
            });

            state.update(wm, &input, &output, &target);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), handle))
        })
    }
}

/// mastery_state(handle) → Array [steps, transitions, pressure_sum, participation_sum]
struct MasteryStateVerb;
impl Verb for MasteryStateVerb {
    fn name(&self) -> &str { "mastery_state" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "mastery_state", HANDLE_WEIGHT_MATRIX, span)?;

        with_runtime(ctx, |rt| {
            let state = rt.mastery.get(&handle)
                .ok_or_else(|| RuneError::argument("mastery_state: no learning state for handle", Some(span)))?;

            Ok(Value::Array(vec![
                Value::Integer(state.steps as i64),
                Value::Integer(state.transitions as i64),
                Value::Integer(state.pressure.iter().map(|p| *p as i64).sum()),
                Value::Integer(state.participation.iter().map(|p| *p as i64).sum()),
            ]))
        })
    }
}

// ---------------------------------------------------------------------------
// Thermogram persistence verbs
// ---------------------------------------------------------------------------

/// load_synaptic(key, rows, cols) → Handle
/// Read synaptic strengths from Thermogram by semantic key. Self-inits zeros if not found.
struct LoadSynapticVerb;
impl Verb for LoadSynapticVerb {
    fn name(&self) -> &str { "load_synaptic" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let key = require_str(args, 0, "load_synaptic", span)?;
        let rows = require_int(args, 1, "load_synaptic", span)? as usize;
        let cols = require_int(args, 2, "load_synaptic", span)? as usize;

        with_runtime(ctx, |rt| {
            // Dedup: if we already loaded this key, return existing handle
            if let Some(&handle) = rt.synaptic_keys.get(&key) {
                return Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), handle));
            }

            // Try reading from Thermogram
            let wm = if let Some(thermo) = &rt.thermogram {
                if let Ok(Some(data)) = thermo.read(&key) {
                    if data.len() == rows * cols {
                        WeightMatrix::from_data(data, rows, cols)
                            .unwrap_or_else(|| WeightMatrix::zeros(rows, cols))
                    } else {
                        WeightMatrix::zeros(rows, cols)
                    }
                } else {
                    WeightMatrix::zeros(rows, cols)
                }
            } else {
                WeightMatrix::zeros(rows, cols)
            };

            let id = rt.alloc_handle();
            rt.weights.insert(id, wm);
            rt.synaptic_keys.insert(key, id);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), id))
        })
    }
}

/// save_synaptic(handle, key) → Nil
/// Write current synaptic strength matrix to Thermogram as an Update delta.
struct SaveSynapticVerb;
impl Verb for SaveSynapticVerb {
    fn name(&self) -> &str { "save_synaptic" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "save_synaptic", HANDLE_WEIGHT_MATRIX, span)?;
        let key = require_str(args, 1, "save_synaptic", span)?;

        with_runtime(ctx, |rt| {
            let wm = rt.weights.get(&handle)
                .ok_or_else(|| RuneError::argument("save_synaptic: invalid handle", Some(span)))?;

            let data: Vec<PackedSignal> = wm.data.clone();

            let thermo = rt.thermogram.as_mut()
                .ok_or_else(|| RuneError::argument("save_synaptic: no thermogram attached", Some(span)))?;

            let prev_hash = thermo.dirty_chain.head_hash.clone();

            // Use Create if key doesn't exist yet, Update otherwise
            let delta = if thermo.read(&key).ok().flatten().is_some() {
                Delta::update(
                    key,
                    data,
                    "mastery_update",
                    Signal::positive(204), // ~0.8 strength
                    prev_hash,
                )
            } else {
                Delta::create(key, data, "load_synaptic")
            };

            thermo.apply_delta(delta)
                .map_err(|e| RuneError::argument(format!("save_synaptic: {e}"), Some(span)))?;

            Ok(Value::Nil)
        })
    }
}

// ---------------------------------------------------------------------------
// Neuromodulator verbs
// ---------------------------------------------------------------------------

/// neuromod_new() → Handle
/// Create a neutral neuromodulator state (all chemicals at baseline 128).
struct NeuromodNewVerb;
impl Verb for NeuromodNewVerb {
    fn name(&self) -> &str { "neuromod_new" }
    fn call(&self, _args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        with_runtime(ctx, |rt| {
            let id = rt.alloc_handle();
            rt.neuromods.insert(id, NeuromodState::new());
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_NEUROMOD), id))
        })
    }
}

/// neuromod_inject(handle, chemical_str, amount) → Nil
/// Inject a signed chemical delta. chemical_str: "da", "ne", or "5ht".
struct NeuromodInjectVerb;
impl Verb for NeuromodInjectVerb {
    fn name(&self) -> &str { "neuromod_inject" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "neuromod_inject", HANDLE_NEUROMOD, span)?;
        let chem_str = require_str(args, 1, "neuromod_inject", span)?;
        let amount = require_int(args, 2, "neuromod_inject", span)? as i8;

        let chemical = match chem_str.as_str() {
            "da" | "dopamine" => Chemical::Dopamine,
            "ne" | "norepinephrine" => Chemical::Norepinephrine,
            "5ht" | "serotonin" => Chemical::Serotonin,
            _ => return Err(RuneError::argument(
                format!("neuromod_inject: unknown chemical '{chem_str}', use da/ne/5ht"),
                Some(span),
            )),
        };

        with_runtime(ctx, |rt| {
            let nm = rt.neuromods.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("neuromod_inject: invalid handle", Some(span)))?;
            nm.inject(chemical, amount);
            Ok(Value::Nil)
        })
    }
}

/// neuromod_tick(handle) → Nil
/// Decay all chemicals toward baseline by 1.
struct NeuromodTickVerb;
impl Verb for NeuromodTickVerb {
    fn name(&self) -> &str { "neuromod_tick" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "neuromod_tick", HANDLE_NEUROMOD, span)?;

        with_runtime(ctx, |rt| {
            let nm = rt.neuromods.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("neuromod_tick: invalid handle", Some(span)))?;
            nm.tick();
            Ok(Value::Nil)
        })
    }
}

/// neuromod_gate(handle) → Bool
/// Check if dopamine gate is open (DA > gate threshold).
struct NeuromodGateVerb;
impl Verb for NeuromodGateVerb {
    fn name(&self) -> &str { "neuromod_gate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "neuromod_gate", HANDLE_NEUROMOD, span)?;

        with_runtime(ctx, |rt| {
            let nm = rt.neuromods.get(&handle)
                .ok_or_else(|| RuneError::argument("neuromod_gate: invalid handle", Some(span)))?;
            Ok(Value::Bool(nm.plasticity_open()))
        })
    }
}

// ---------------------------------------------------------------------------
// Prediction verbs
// ---------------------------------------------------------------------------

/// predict_new(dims, shift, threshold) → Handle
/// Create a new prediction engine.
struct PredictNewVerb;
impl Verb for PredictNewVerb {
    fn name(&self) -> &str { "predict_new" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let dims = require_int(args, 0, "predict_new", span)? as usize;
        let shift = require_int(args, 1, "predict_new", span)? as u8;
        let threshold = require_int(args, 2, "predict_new", span)? as i32;

        with_runtime(ctx, |rt| {
            let id = rt.alloc_handle();
            rt.predictors.insert(id, PredictionEngine::new(dims, shift, threshold));
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_PREDICTOR), id))
        })
    }
}

/// predict_observe(handle, actual) → Array [magnitude, is_surprising, direction]
/// Observe actual output. Updates EMA and returns surprise info.
struct PredictObserveVerb;
impl Verb for PredictObserveVerb {
    fn name(&self) -> &str { "predict_observe" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "predict_observe", HANDLE_PREDICTOR, span)?;
        let actual = require_array(args, 1, "predict_observe", span)?;

        // Optional 3rd arg: target array for direction computation
        let target = if args.len() >= 3 {
            match &args[2] {
                Value::Nil => None,
                _ => Some(require_array(args, 2, "predict_observe", span)?),
            }
        } else {
            None
        };

        with_runtime(ctx, |rt| {
            let pred = rt.predictors.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("predict_observe: invalid handle", Some(span)))?;
            let surprise = pred.observe(&actual, target.as_deref());
            Ok(Value::Array(vec![
                Value::Integer(surprise.magnitude),
                Value::Integer(if surprise.is_surprising { 1 } else { 0 }),
                Value::Integer(surprise.direction as i64),
            ]))
        })
    }
}

// ---------------------------------------------------------------------------
// Thermogram persistence verbs
// ---------------------------------------------------------------------------

/// persist_thermo(path) → Nil
/// Save the Thermogram to disk as a .thermo file.
struct PersistThermoVerb;
impl Verb for PersistThermoVerb {
    fn name(&self) -> &str { "persist_thermo" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let path_str = require_str(args, 0, "persist_thermo", span)?;

        with_runtime(ctx, |rt| {
            let resolved = rt.resolve_path(&path_str);
            let thermo = rt.thermogram.as_ref()
                .ok_or_else(|| RuneError::argument("persist_thermo: no thermogram attached", Some(span)))?;

            thermo.save(&resolved)
                .map_err(|e| RuneError::argument(format!("persist_thermo: {e}"), Some(span)))?;

            Ok(Value::Nil)
        })
    }
}

// ---------------------------------------------------------------------------
// Utility verbs: sub, abs, negate, concat, sum
// ---------------------------------------------------------------------------

/// sub(a, b) → Array — element-wise subtract
struct SubVerb;
impl Verb for SubVerb {
    fn name(&self) -> &str { "sub" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let a = require_array(args, 0, "sub", span)?;
        let b = require_array(args, 1, "sub", span)?;
        if a.len() != b.len() {
            return Err(RuneError::argument("sub: arrays must have same length", Some(span)));
        }
        let output: Vec<PackedSignal> = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let diff = x.current() as i64 - y.current() as i64;
                packed_from_current(diff as i32)
            })
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// abs(signals) → Array — per-element absolute value
struct AbsVerb;
impl Verb for AbsVerb {
    fn name(&self) -> &str { "abs" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "abs", span)?;
        let output: Vec<PackedSignal> = signals.iter()
            .map(|s| packed_from_current(s.current().abs()))
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// negate(signals) → Array — flip sign of each element
struct NegateVerb;
impl Verb for NegateVerb {
    fn name(&self) -> &str { "negate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "negate", span)?;
        let output: Vec<PackedSignal> = signals.iter()
            .map(|s| packed_from_current(-s.current()))
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// concat(a, b) → Array — join two arrays
struct ConcatVerb;
impl Verb for ConcatVerb {
    fn name(&self) -> &str { "concat" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let a = require_array(args, 0, "concat", span)?;
        let b = require_array(args, 1, "concat", span)?;
        let mut output = a;
        output.extend_from_slice(&b);
        Ok(packed_to_values(&output))
    }
}

/// sum(signals) → Integer — sum of all current values
struct SumVerb;
impl Verb for SumVerb {
    fn name(&self) -> &str { "sum" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "sum", span)?;
        let total: i64 = signals.iter().map(|s| s.current() as i64).sum();
        Ok(Value::Integer(total))
    }
}

// ---------------------------------------------------------------------------
// Salience routing verbs
// ---------------------------------------------------------------------------

/// salience_new(n_sources, source_dim) → Handle
struct SalienceNewVerb;
impl Verb for SalienceNewVerb {
    fn name(&self) -> &str { "salience_new" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let n_sources = require_int(args, 0, "salience_new", span)? as usize;
        let source_dim = require_int(args, 1, "salience_new", span)? as usize;

        with_runtime(ctx, |rt| {
            let router = SalienceRouter::new(n_sources, source_dim);
            let id = rt.next_handle;
            rt.next_handle += 1;
            rt.salience.insert(id, router);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_SALIENCE), id))
        })
    }
}

/// salience_route(handle, outputs) → Array
/// Returns [routed_output..., confidence_0, ..., confidence_n, winner]
struct SalienceRouteVerb;
impl Verb for SalienceRouteVerb {
    fn name(&self) -> &str { "salience_route" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "salience_route", HANDLE_SALIENCE, span)?;
        let outputs = require_array(args, 1, "salience_route", span)?;

        with_runtime(ctx, |rt| {
            let router = rt.salience.get(&handle)
                .ok_or_else(|| RuneError::argument("salience_route: invalid handle", Some(span)))?;
            let result = router.route(&outputs);

            // Pack: [routed_output..., conf_0, ..., conf_n, winner]
            let mut values: Vec<Value> = result.output.iter()
                .map(|s| Value::Integer(s.as_u8() as i64))
                .collect();
            for c in &result.confidences {
                values.push(Value::Integer(*c));
            }
            values.push(Value::Integer(result.winner as i64));

            Ok(Value::Array(values))
        })
    }
}

/// salience_train(handle, outputs, routed, target) → Nil
struct SalienceTrainVerb;
impl Verb for SalienceTrainVerb {
    fn name(&self) -> &str { "salience_train" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "salience_train", HANDLE_SALIENCE, span)?;
        let outputs = require_array(args, 1, "salience_train", span)?;
        let routed = require_array(args, 2, "salience_train", span)?;
        let target = require_array(args, 3, "salience_train", span)?;

        with_runtime(ctx, |rt| {
            let router = rt.salience.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("salience_train: invalid handle", Some(span)))?;
            router.train_route(&outputs, &routed, &target);
            Ok(Value::Nil)
        })
    }
}

// ---------------------------------------------------------------------------
// neuromod_read verb
// ---------------------------------------------------------------------------

/// stride_slice(signals, offset, stride) → Array — pick every stride-th element starting at offset
/// E.g. stride_slice(arr, 0, 2) picks even indices, stride_slice(arr, 1, 2) picks odd indices.
struct StrideSliceVerb;
impl Verb for StrideSliceVerb {
    fn name(&self) -> &str { "stride_slice" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "stride_slice", span)?;
        let offset = require_int(args, 1, "stride_slice", span)? as usize;
        let stride = require_int(args, 2, "stride_slice", span)? as usize;
        if stride == 0 {
            return Err(RuneError::argument("stride_slice: stride must be > 0", Some(span)));
        }
        let output: Vec<PackedSignal> = signals.iter()
            .skip(offset)
            .step_by(stride)
            .cloned()
            .collect();
        Ok(packed_to_values(&output))
    }
}

/// neuromod_read(handle, chemical) → Integer — read raw chemical level (0-255)
struct NeuromodReadVerb;
impl Verb for NeuromodReadVerb {
    fn name(&self) -> &str { "neuromod_read" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "neuromod_read", HANDLE_NEUROMOD, span)?;
        let chem_str = require_str(args, 1, "neuromod_read", span)?;

        with_runtime(ctx, |rt| {
            let nm = rt.neuromods.get(&handle)
                .ok_or_else(|| RuneError::argument("neuromod_read: invalid handle", Some(span)))?;
            let level = match chem_str.as_str() {
                "da" | "dopamine" => nm.dopamine,
                "ne" | "norepinephrine" => nm.norepinephrine,
                "5ht" | "serotonin" => nm.serotonin,
                _ => return Err(RuneError::argument(
                    format!("neuromod_read: unknown chemical '{chem_str}'"),
                    Some(span),
                )),
            };
            Ok(Value::Integer(level as i64))
        })
    }
}
