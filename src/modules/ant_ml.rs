//! `ant_ml` Runes module — ML operations for Atomic Neural Transistors.
//!
//! Provides ternary matrix operations, activation functions, weight I/O,
//! and mastery learning as Runes verbs.
//!
//! Usage in .rune scripts:
//! ```rune
//! use :ant_ml
//! w = load_synaptic("classifier.w_in", 24, 32)
//! h = matmul(input, w, 24, 32)
//! h = relu(h)
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use runes_core::error::RuneError;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;
use ternary_signal::Signal;

use crate::core::weight_matrix::WeightMatrix;
use crate::core::thermal::{ThermalWeightMatrix, ThermalMasteryConfig};
use crate::learning::{MasteryConfig, MasteryState};
use crate::thermal_mastery::ThermalMasteryState;

use crate::neuromod::{Chemical, NeuromodState};
use crate::prediction::PredictionEngine;
use crate::salience::SalienceRouter;

// Handle type IDs
const HANDLE_WEIGHT_MATRIX: u32 = 1;
const HANDLE_NEUROMOD: u32 = 2;
const HANDLE_PREDICTOR: u32 = 3;
const HANDLE_SALIENCE: u32 = 4;
const HANDLE_THERMAL_MATRIX: u32 = 5;

/// Runtime state shared between the ANT host and the Runes engine.
pub struct AntRuntime {
    /// Synaptic strength matrices keyed by handle ID.
    weights: HashMap<u64, WeightMatrix>,
    /// Mastery learning state keyed by handle ID.
    mastery: HashMap<u64, MasteryState>,
    /// Map from semantic key to handle ID (for load_synaptic deduplication).
    synaptic_keys: HashMap<String, u64>,
    /// Neuromodulator states keyed by handle ID.
    neuromods: HashMap<u64, NeuromodState>,
    /// Prediction engines keyed by handle ID.
    predictors: HashMap<u64, PredictionEngine>,
    /// Salience routers keyed by handle ID.
    salience: HashMap<u64, SalienceRouter>,
    /// Thermal synaptic strength matrices keyed by handle ID.
    thermal_weights: HashMap<u64, ThermalWeightMatrix>,
    /// Thermal mastery learning state keyed by handle ID.
    thermal_mastery: HashMap<u64, ThermalMasteryState>,
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
            thermal_weights: HashMap::new(),
            thermal_mastery: HashMap::new(),
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

    /// Get thermal mastery state for a handle.
    pub fn get_thermal_mastery(&self, handle: u64) -> Option<&ThermalMasteryState> {
        self.thermal_mastery.get(&handle)
    }

    /// List all synaptic keys and their handle IDs.
    pub fn synaptic_key_handles(&self) -> &HashMap<String, u64> {
        &self.synaptic_keys
    }

    /// Look up a weight matrix by its synaptic key.
    pub fn weights_by_key(&self, key: &str) -> Option<&WeightMatrix> {
        let handle = self.synaptic_keys.get(key)?;
        self.weights.get(handle)
    }

    /// Look up a mutable weight matrix by its synaptic key.
    pub fn weights_by_key_mut(&mut self, key: &str) -> Option<&mut WeightMatrix> {
        let handle = *self.synaptic_keys.get(key)?;
        self.weights.get_mut(&handle)
    }

    /// Look up a thermal weight matrix by its synaptic key.
    pub fn thermal_weights_by_key(&self, key: &str) -> Option<&ThermalWeightMatrix> {
        let handle = self.synaptic_keys.get(key)?;
        self.thermal_weights.get(handle)
    }

    /// Get or create a mastery state for a synaptic key.
    pub fn ensure_mastery_for_key(&mut self, key: &str) -> Option<(&mut MasteryState, &mut WeightMatrix)> {
        let handle = *self.synaptic_keys.get(key)?;
        let wm = self.weights.get(&handle)?;
        let weight_count = wm.data.len();
        self.mastery.entry(handle).or_insert_with(|| {
            MasteryState::new(weight_count, MasteryConfig::default())
        });
        let mastery = self.mastery.get_mut(&handle).unwrap();
        let wm = self.weights.get_mut(&handle).unwrap();
        Some((mastery, wm))
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
                Box::new(NormalizeVerb),
                Box::new(SigmoidVerb),
                Box::new(TanhActVerb),
                Box::new(SoftmaxVerb),
                Box::new(ArgmaxVerb),
                Box::new(DotVerb),
                Box::new(SignalVerb),
                Box::new(MulVerb),
                Box::new(AddVerb),
                Box::new(ShiftVerb),
                Box::new(SliceVerb),
                Box::new(ZerosVerb),
                Box::new(MasteryUpdateVerb),
                Box::new(MasteryStateVerb),
                Box::new(LoadSynapticVerb),
                Box::new(LoadSynapticFrozenVerb),
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
                // Mastery utilities
                Box::new(MasteryDecayVerb),
                Box::new(ClampVerb),
                Box::new(OnehotVerb),
                // Thermal verbs
                Box::new(ThermalLoadSynapticVerb),
                Box::new(ThermalMatmulVerb),
                Box::new(ThermalMasteryUpdateVerb),
                Box::new(ThermalDecayVerb),
                Box::new(ThermalSummaryVerb),
                Box::new(ThermalImprintVerb),
                Box::new(ThermalNormalizeImprintVerb),
                Box::new(ThermalReadVerb),
            ],
        }
    }
}

impl Module for AntMlModule {
    fn name(&self) -> &str {
        "ant_ml"
    }

    fn version(&self) -> ModuleVersion {
        ModuleVersion::new(0, 7, 0)
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
    let arc = ctx.runtime_mut::<Arc<Mutex<AntRuntime>>>()?;
    let mut guard = arc.lock().map_err(|_| RuneError::type_error("runtime lock poisoned", Some(span)))?;
    f(&mut guard)
}

// ---------------------------------------------------------------------------
// Helper: convert between Value arrays and Signal vectors
// ---------------------------------------------------------------------------

/// Convert Signal slice to Runes Value array (each signal → its current i32 value).
fn signals_to_values(signals: &[Signal]) -> Value {
    Value::Array(Arc::new(signals.iter().map(|s| Value::Integer(s.current() as i64)).collect()))
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

/// Extract a Signal array from Runes Value. Each integer is interpreted as a current value.
fn require_array(args: &[Value], idx: usize, name: &str, span: runes_core::Span) -> Result<Vec<Signal>, RuneError> {
    match args.get(idx) {
        Some(Value::Array(arr)) => {
            arr.iter()
                .map(|v| match v {
                    Value::Integer(n) => Ok(Signal::from_current(*n as i32)),
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

/// matmul(input, weights_handle) → Array
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
            Ok(signals_to_values(&output))
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
        let output: Vec<Signal> = signals.iter().map(|s| {
            if s.current() < 0 { Signal::ZERO } else { *s }
        }).collect();
        Ok(signals_to_values(&output))
    }
}

/// normalize(signals) → Array
/// Scale all values so max magnitude = 127, preserving polarity.
struct NormalizeVerb;
impl Verb for NormalizeVerb {
    fn name(&self) -> &str { "normalize" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "normalize", span)?;
        if signals.is_empty() {
            return Ok(signals_to_values(&[]));
        }

        let max_abs = signals.iter()
            .map(|s| s.current().unsigned_abs())
            .max()
            .unwrap_or(1)
            .max(1) as i64;

        let output: Vec<Signal> = signals.iter().map(|s| {
            let c = s.current() as i64;
            let scaled = (c * 127) / max_abs;
            Signal::from_current(scaled as i32)
        }).collect();
        Ok(signals_to_values(&output))
    }
}

/// sigmoid(signals) → Array
/// Ternary sigmoid: threshold curve mapping to 0-255 range.
struct SigmoidVerb;
impl Verb for SigmoidVerb {
    fn name(&self) -> &str { "sigmoid" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "sigmoid", span)?;
        let output: Vec<Signal> = signals.iter().map(|s| {
            let c = s.current();
            let sigmoid_val = ternary_sigmoid(c);
            Signal::new_raw(1, sigmoid_val, 1)
        }).collect();
        Ok(signals_to_values(&output))
    }
}

/// Integer sigmoid approximation: maps i32 current to 0-255 output magnitude.
fn ternary_sigmoid(current: i32) -> u8 {
    if current <= -512 {
        0
    } else if current >= 512 {
        255
    } else {
        ((current as i64 + 512) * 255 / 1024) as u8
    }
}

/// tanh_act(signals) → Array
/// Symmetric threshold.
struct TanhActVerb;
impl Verb for TanhActVerb {
    fn name(&self) -> &str { "tanh_act" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "tanh_act", span)?;
        let output: Vec<Signal> = signals.iter().map(|s| {
            let c = s.current();
            let (pol, mag) = ternary_tanh(c);
            Signal::new_raw(pol, mag, 1)
        }).collect();
        Ok(signals_to_values(&output))
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
/// Normalize magnitudes to sum=255, positive polarity.
struct SoftmaxVerb;
impl Verb for SoftmaxVerb {
    fn name(&self) -> &str { "softmax" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "softmax", span)?;
        if signals.is_empty() {
            return Ok(signals_to_values(&[]));
        }

        let currents: Vec<i32> = signals.iter().map(|s| s.current()).collect();
        let min_c = *currents.iter().min().unwrap();
        let shifted: Vec<u64> = currents.iter().map(|&c| (c as i64 - min_c as i64) as u64).collect();
        let total: u64 = shifted.iter().sum();

        let output: Vec<Signal> = if total == 0 {
            let uniform = (255 / signals.len() as u8).max(1);
            signals.iter().map(|_| Signal::new_raw(1, uniform, 1)).collect()
        } else {
            shifted.iter().map(|&s| {
                let mag = ((s * 255) / total) as u8;
                Signal::new_raw(1, mag, 1)
            }).collect()
        };
        Ok(signals_to_values(&output))
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

/// signal(polarity, magnitude, multiplier) → Integer (current value)
/// Construct a signal and return its current value for use in arrays.
struct SignalVerb;
impl Verb for SignalVerb {
    fn name(&self) -> &str { "signal" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let pol = require_int(args, 0, "signal", span)? as i8;
        let mag = require_int(args, 1, "signal", span)? as u8;
        let mul = require_int(args, 2, "signal", span)? as u8;
        let s = Signal::new_raw(pol, mag, mul);
        Ok(Value::Integer(s.current() as i64))
    }
}

/// mul(a, b) → Array — element-wise multiply
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
        let output: Vec<Signal> = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let product = x.current() as i64 * y.current() as i64;
                Signal::from_current(product as i32)
            })
            .collect();
        Ok(signals_to_values(&output))
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
        let output: Vec<Signal> = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let sum = x.current() as i64 + y.current() as i64;
                Signal::from_current(sum as i32)
            })
            .collect();
        Ok(signals_to_values(&output))
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
        let output: Vec<Signal> = signals.iter()
            .map(|s| {
                let shifted = s.current() >> n;
                Signal::from_current(shifted)
            })
            .collect();
        Ok(signals_to_values(&output))
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
        Ok(signals_to_values(&signals[start..start + len]))
    }
}

/// zeros(size) → Array
struct ZerosVerb;
impl Verb for ZerosVerb {
    fn name(&self) -> &str { "zeros" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let size = require_int(args, 0, "zeros", span)? as usize;
        let arr: Vec<Value> = vec![Value::Integer(0); size];
        Ok(Value::Array(Arc::new(arr)))
    }
}

/// mastery_update(weights_handle, input, output, target, config_map) → Handle
struct MasteryUpdateVerb;
impl Verb for MasteryUpdateVerb {
    fn name(&self) -> &str { "mastery_update" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "mastery_update", HANDLE_WEIGHT_MATRIX, span)?;
        let input = require_array(args, 1, "mastery_update", span)?;
        let output = require_array(args, 2, "mastery_update", span)?;
        let target = require_array(args, 3, "mastery_update", span)?;

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

            Ok(Value::Array(Arc::new(vec![
                Value::Integer(state.steps as i64),
                Value::Integer(state.transitions as i64),
                Value::Integer(state.pressure.iter().map(|p| *p as i64).sum()),
                Value::Integer(state.participation.iter().map(|p| *p as i64).sum()),
            ])))
        })
    }
}

/// load_synaptic(key, rows, cols) → Handle
/// Create or retrieve a synaptic strength matrix by semantic key.
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

            let wm = WeightMatrix::zeros(rows, cols);
            let id = rt.alloc_handle();
            rt.weights.insert(id, wm);
            rt.synaptic_keys.insert(key, id);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), id))
        })
    }
}

/// load_synaptic_frozen(key, rows, cols, seed) → Handle
/// Create a frozen random projection matrix.
struct LoadSynapticFrozenVerb;
impl Verb for LoadSynapticFrozenVerb {
    fn name(&self) -> &str { "load_synaptic_frozen" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let key = require_str(args, 0, "load_synaptic_frozen", span)?;
        let rows = require_int(args, 1, "load_synaptic_frozen", span)? as usize;
        let cols = require_int(args, 2, "load_synaptic_frozen", span)? as usize;
        let seed = if args.len() > 3 {
            require_int(args, 3, "load_synaptic_frozen", span)? as u64
        } else {
            0xC0DE_CAFE_1234_5678
        };

        with_runtime(ctx, |rt| {
            if let Some(&handle) = rt.synaptic_keys.get(&key) {
                return Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), handle));
            }

            let wm = WeightMatrix::random_frozen(rows, cols, seed);
            let id = rt.alloc_handle();
            rt.weights.insert(id, wm);
            rt.synaptic_keys.insert(key, id);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_WEIGHT_MATRIX), id))
        })
    }
}

// ---------------------------------------------------------------------------
// Neuromodulator verbs
// ---------------------------------------------------------------------------

/// neuromod_new() → Handle
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

/// predict_observe(handle, actual, [target]) → Array [magnitude, is_surprising, direction]
struct PredictObserveVerb;
impl Verb for PredictObserveVerb {
    fn name(&self) -> &str { "predict_observe" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "predict_observe", HANDLE_PREDICTOR, span)?;
        let actual = require_array(args, 1, "predict_observe", span)?;

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
            Ok(Value::Array(Arc::new(vec![
                Value::Integer(surprise.magnitude),
                Value::Integer(if surprise.is_surprising { 1 } else { 0 }),
                Value::Integer(surprise.direction as i64),
            ])))
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
        let output: Vec<Signal> = a.iter().zip(b.iter())
            .map(|(x, y)| {
                let diff = x.current() as i64 - y.current() as i64;
                Signal::from_current(diff as i32)
            })
            .collect();
        Ok(signals_to_values(&output))
    }
}

/// abs(signals) → Array — per-element absolute value
struct AbsVerb;
impl Verb for AbsVerb {
    fn name(&self) -> &str { "abs" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "abs", span)?;
        let output: Vec<Signal> = signals.iter()
            .map(|s| Signal::from_current(s.current().abs()))
            .collect();
        Ok(signals_to_values(&output))
    }
}

/// negate(signals) → Array — flip sign of each element
struct NegateVerb;
impl Verb for NegateVerb {
    fn name(&self) -> &str { "negate" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "negate", span)?;
        let output: Vec<Signal> = signals.iter()
            .map(|s| Signal::from_current(-s.current()))
            .collect();
        Ok(signals_to_values(&output))
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
        Ok(signals_to_values(&output))
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

            let mut values: Vec<Value> = result.output.iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            for c in &result.confidences {
                values.push(Value::Integer(*c));
            }
            values.push(Value::Integer(result.winner as i64));

            Ok(Value::Array(Arc::new(values)))
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

/// stride_slice(signals, offset, stride) → Array
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
        let output: Vec<Signal> = signals.iter()
            .skip(offset)
            .step_by(stride)
            .cloned()
            .collect();
        Ok(signals_to_values(&output))
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

// ---------------------------------------------------------------------------
// Mastery utility verbs
// ---------------------------------------------------------------------------

/// mastery_decay(w_handle) → Nil
/// Apply per-cycle pressure decay on the MasteryState for a weight matrix.
struct MasteryDecayVerb;
impl Verb for MasteryDecayVerb {
    fn name(&self) -> &str { "mastery_decay" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "mastery_decay", HANDLE_WEIGHT_MATRIX, span)?;

        with_runtime(ctx, |rt| {
            // No-op if no mastery state exists (e.g. frozen matrices)
            if let Some(state) = rt.mastery.get_mut(&handle) {
                state.decay();
            }
            Ok(Value::Nil)
        })
    }
}

/// clamp(signals, lo, hi) → Array
/// Clamp each signal's current value to [lo, hi].
struct ClampVerb;
impl Verb for ClampVerb {
    fn name(&self) -> &str { "clamp" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let signals = require_array(args, 0, "clamp", span)?;
        let lo = require_int(args, 1, "clamp", span)? as i32;
        let hi = require_int(args, 2, "clamp", span)? as i32;
        let output: Vec<Signal> = signals.iter()
            .map(|s| Signal::from_current(s.current().clamp(lo, hi)))
            .collect();
        Ok(signals_to_values(&output))
    }
}

/// onehot(class_idx, n_classes, pos_mag, neg_mag) → Array
/// Generate a one-hot target vector. class_idx gets +pos_mag, others get -neg_mag.
struct OnehotVerb;
impl Verb for OnehotVerb {
    fn name(&self) -> &str { "onehot" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let class = require_int(args, 0, "onehot", span)? as usize;
        let n_classes = require_int(args, 1, "onehot", span)? as usize;
        let pos = require_int(args, 2, "onehot", span)? as i32;
        let neg = require_int(args, 3, "onehot", span)? as i32;

        let output: Vec<Signal> = (0..n_classes)
            .map(|i| {
                if i == class {
                    Signal::from_current(pos)
                } else {
                    Signal::from_current(-neg)
                }
            })
            .collect();
        Ok(signals_to_values(&output))
    }
}

// ---------------------------------------------------------------------------
// Thermal verbs
// ---------------------------------------------------------------------------

/// thermal_load_synaptic(key, rows, cols) → Handle
/// Create or retrieve a ThermalWeightMatrix by semantic key. Initializes as zeros (all HOT).
struct ThermalLoadSynapticVerb;
impl Verb for ThermalLoadSynapticVerb {
    fn name(&self) -> &str { "thermal_load_synaptic" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let key = require_str(args, 0, "thermal_load_synaptic", span)?;
        let rows = require_int(args, 1, "thermal_load_synaptic", span)? as usize;
        let cols = require_int(args, 2, "thermal_load_synaptic", span)? as usize;

        with_runtime(ctx, |rt| {
            // Dedup by key
            if let Some(&handle) = rt.synaptic_keys.get(&key) {
                return Ok(Value::Handle(runes_core::value::HandleType(HANDLE_THERMAL_MATRIX), handle));
            }

            // Optional seed for random init (4th arg)
            let twm = if args.len() > 3 {
                let seed = require_int(args, 3, "thermal_load_synaptic", span)? as u64;
                ThermalWeightMatrix::random_hot(rows, cols, seed)
            } else {
                ThermalWeightMatrix::zeros(rows, cols)
            };

            let id = rt.alloc_handle();
            rt.thermal_weights.insert(id, twm);
            rt.synaptic_keys.insert(key, id);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_THERMAL_MATRIX), id))
        })
    }
}

/// thermal_matmul(input, w_handle) → Array
/// Matrix-vector multiply through a ThermalWeightMatrix (temperature doesn't affect computation).
struct ThermalMatmulVerb;
impl Verb for ThermalMatmulVerb {
    fn name(&self) -> &str { "thermal_matmul" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let input = require_array(args, 0, "thermal_matmul", span)?;
        let handle = require_handle(args, 1, "thermal_matmul", HANDLE_THERMAL_MATRIX, span)?;

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get(&handle)
                .ok_or_else(|| RuneError::argument("thermal_matmul: invalid handle", Some(span)))?;
            let output = twm.matmul(&input);
            Ok(signals_to_values(&output))
        })
    }
}

/// thermal_mastery_update(w_handle, input, output, target, correct, [config]) → Handle
/// Run one thermal mastery learning step. Config is optional array [threshold, decay, gate, cooling, warming_step, warming_threshold].
struct ThermalMasteryUpdateVerb;
impl Verb for ThermalMasteryUpdateVerb {
    fn name(&self) -> &str { "thermal_mastery_update" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "thermal_mastery_update", HANDLE_THERMAL_MATRIX, span)?;
        let input = require_array(args, 1, "thermal_mastery_update", span)?;
        let output = require_array(args, 2, "thermal_mastery_update", span)?;
        let target = require_array(args, 3, "thermal_mastery_update", span)?;
        let correct = match args.get(4) {
            Some(Value::Bool(b)) => *b,
            Some(Value::Integer(n)) => *n != 0,
            _ => return Err(RuneError::argument("thermal_mastery_update requires bool or integer at position 4", Some(span))),
        };

        let config = if let Some(Value::Array(arr)) = args.get(5) {
            ThermalMasteryConfig {
                pressure_threshold: arr.first().and_then(|v| v.as_integer()).unwrap_or(3) as i32,
                decay_rate: arr.get(1).and_then(|v| v.as_integer()).unwrap_or(1) as i32,
                participation_gate: arr.get(2).and_then(|v| v.as_integer()).unwrap_or(5) as u32,
                cooling_rate: arr.get(3).and_then(|v| v.as_integer()).unwrap_or(100) as u16,
                warming_step: arr.get(4).and_then(|v| v.as_integer()).unwrap_or(10) as u8,
                warming_threshold: arr.get(5).and_then(|v| v.as_integer()).unwrap_or(20) as i32,
            }
        } else {
            ThermalMasteryConfig::default()
        };

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("thermal_mastery_update: invalid handle", Some(span)))?;

            let state = rt.thermal_mastery.entry(handle).or_insert_with(|| {
                ThermalMasteryState::new(config.clone())
            });

            state.update(twm, &input, &output, &target, correct);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_THERMAL_MATRIX), handle))
        })
    }
}

/// thermal_decay(w_handle) → Nil
/// Apply per-cycle pressure decay with temperature-gated cooling on a ThermalWeightMatrix.
struct ThermalDecayVerb;
impl Verb for ThermalDecayVerb {
    fn name(&self) -> &str { "thermal_decay" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "thermal_decay", HANDLE_THERMAL_MATRIX, span)?;

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("thermal_decay: invalid handle", Some(span)))?;
            let state = rt.thermal_mastery.get(&handle)
                .ok_or_else(|| RuneError::argument("thermal_decay: no mastery state for handle", Some(span)))?;
            state.decay(twm);
            Ok(Value::Nil)
        })
    }
}

/// thermal_summary(w_handle) → Array [hot, warm, cool, cold]
/// Return the temperature band distribution for a ThermalWeightMatrix.
struct ThermalSummaryVerb;
impl Verb for ThermalSummaryVerb {
    fn name(&self) -> &str { "thermal_summary" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "thermal_summary", HANDLE_THERMAL_MATRIX, span)?;

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get(&handle)
                .ok_or_else(|| RuneError::argument("thermal_summary: invalid handle", Some(span)))?;
            let (h, w, c, d) = twm.temp_summary();
            Ok(Value::Array(Arc::new(vec![
                Value::Integer(h as i64),
                Value::Integer(w as i64),
                Value::Integer(c as i64),
                Value::Integer(d as i64),
            ])))
        })
    }
}

/// thermal_imprint(w_handle, input, [row]) → Nil
/// Additively absorb an input pattern into a ThermalWeightMatrix.
/// If row is provided, imprints into that specific row only.
/// If omitted, broadcasts to all rows.
/// No pressure, no thresholds — direct write.
struct ThermalImprintVerb;
impl Verb for ThermalImprintVerb {
    fn name(&self) -> &str { "thermal_imprint" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "thermal_imprint", HANDLE_THERMAL_MATRIX, span)?;
        let input = require_array(args, 1, "thermal_imprint", span)?;
        let row = if args.len() > 2 {
            Some(require_int(args, 2, "thermal_imprint", span)? as usize)
        } else {
            None
        };

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("thermal_imprint: invalid handle", Some(span)))?;
            if let Some(r) = row {
                twm.imprint_row(r, &input);
            } else {
                twm.imprint(&input);
            }
            Ok(Value::Nil)
        })
    }
}

/// thermal_normalize_imprint(w_handle, count, [row]) → Nil
/// Divide imprinted weights by count. If row given, normalize that row only.
struct ThermalNormalizeImprintVerb;
impl Verb for ThermalNormalizeImprintVerb {
    fn name(&self) -> &str { "thermal_normalize_imprint" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "thermal_normalize_imprint", HANDLE_THERMAL_MATRIX, span)?;
        let count = require_int(args, 1, "thermal_normalize_imprint", span)? as usize;
        let row = if args.len() > 2 {
            Some(require_int(args, 2, "thermal_normalize_imprint", span)? as usize)
        } else {
            None
        };

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get_mut(&handle)
                .ok_or_else(|| RuneError::argument("thermal_normalize_imprint: invalid handle", Some(span)))?;
            if let Some(r) = row {
                twm.normalize_row(r, count);
            } else {
                twm.normalize_imprint(count);
            }
            Ok(Value::Nil)
        })
    }
}

/// thermal_read(w_handle, [row]) → Array
/// Read the signal values from a ThermalWeightMatrix as an array of current values.
/// If row is given, reads that row only. Otherwise reads row 0.
/// Each value preserves the full s = p × m × k computation.
struct ThermalReadVerb;
impl Verb for ThermalReadVerb {
    fn name(&self) -> &str { "thermal_read" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "thermal_read", HANDLE_THERMAL_MATRIX, span)?;
        let row = if args.len() > 1 {
            require_int(args, 1, "thermal_read", span)? as usize
        } else {
            0
        };

        with_runtime(ctx, |rt| {
            let twm = rt.thermal_weights.get(&handle)
                .ok_or_else(|| RuneError::argument("thermal_read: invalid handle", Some(span)))?;
            if row >= twm.rows {
                return Err(RuneError::argument(
                    format!("thermal_read: row {} out of bounds (rows={})", row, twm.rows),
                    Some(span),
                ));
            }
            let start = row * twm.cols;
            let end = start + twm.cols;
            let values: Vec<Value> = twm.data[start..end].iter()
                .map(|tw| Value::Integer(tw.current() as i64))
                .collect();
            Ok(Value::Array(Arc::new(values)))
        })
    }
}
