//! Memory ANT — stores/retrieves pattern signals via databank-rs.
//!
//! Surprise-gated writes, similarity-based reads. The Memory ANT perceives
//! incoming signals, recalls known patterns, and stores novel ones when the
//! dopamine gate is open.
//!
//! Core operations:
//! - `perceive(signal)` — recall if known, store if novel + DA gate open
//! - `recall(cue, top_k)` — query without side effects
//! - `store(signal, temp)` — force-store with explicit temperature
//! - `associate(from, to, edge_type)` — link patterns
//! - `save(path)` / `load(path)` — persistence

use std::path::Path;

use databank_rs::{
    BankCluster, BankConfig, BankId, DataBank, Edge, EdgeType, EntryId, QueryResult, Temperature,
};
use ternary_signal::Signal;

use crate::neuromod::{Chemical, NeuromodState};
use crate::prediction::{PredictionEngine, SurpriseSignal};

/// Result of perceiving a signal.
#[derive(Debug, Clone)]
pub struct PerceptionResult {
    /// Best match from the store (None if store is empty).
    pub best_match: Option<QueryResult>,
    /// Whether the signal was novel (no close match found).
    pub is_novel: bool,
    /// If stored, the new entry's ID.
    pub stored_as: Option<EntryId>,
    /// Surprise signal from the predictor.
    pub surprise: SurpriseSignal,
}

/// Memory ANT — stores and retrieves pattern signals via databank-rs.
pub struct MemoryANT {
    cluster: BankCluster,
    bank_id: BankId,
    neuromod: NeuromodState,
    predictor: PredictionEngine,
    /// Similarity score threshold for "known" (0-256 range).
    /// Scores above this mean the pattern is recognized.
    recall_threshold: i32,
    /// Must match encoding dim (e.g. 64).
    vector_width: u16,
    /// DA injection amount on novelty detection.
    novelty_da_boost: i8,
}

impl MemoryANT {
    /// Create a new Memory ANT with an empty store.
    pub fn new(name: &str, vector_width: u16) -> Self {
        let mut cluster = BankCluster::new();
        let bank_id = BankId::new(name, 0);
        let config = BankConfig {
            vector_width,
            ..BankConfig::default()
        };
        cluster.get_or_create(bank_id, name.to_string(), config);

        Self {
            cluster,
            bank_id,
            neuromod: NeuromodState::new(),
            predictor: PredictionEngine::new(vector_width as usize, 3, 40),
            recall_threshold: 180,
            vector_width,
            novelty_da_boost: 30,
        }
    }

    /// Create with custom recall threshold and DA gate.
    pub fn with_config(
        name: &str,
        vector_width: u16,
        recall_threshold: i32,
        dopamine_gate: u8,
    ) -> Self {
        let mut ant = Self::new(name, vector_width);
        ant.recall_threshold = recall_threshold;
        ant.neuromod = NeuromodState::with_gate(dopamine_gate);
        ant
    }

    /// Perceive an incoming signal pattern.
    ///
    /// 1. Recall best match from store
    /// 2. High match score → known pattern, return it
    /// 3. No match → predictor observes → surprise?
    /// 4. Surprising + DA gate open → store the pattern
    /// 5. Inject DA for novelty
    pub fn perceive(&mut self, signal: &[Signal]) -> PerceptionResult {
        // Step 1: recall
        let best_match = self.recall_best(signal);

        // Step 2: check if known
        let is_known = best_match
            .as_ref()
            .map_or(false, |m| m.score >= self.recall_threshold);

        // Step 3: observe for surprise
        let surprise = self.predictor.observe(signal, None);

        // Step 4: store if novel + DA gate open
        let stored_as = if !is_known && surprise.is_surprising && self.neuromod.plasticity_open() {
            // Novel and surprising — store it
            let bank = self.cluster.get_mut(self.bank_id).unwrap();
            match bank.insert(signal.to_vec(), Temperature::Hot, 0) {
                Ok(id) => {
                    // Step 5: inject DA for novelty
                    self.neuromod.inject(Chemical::Dopamine, self.novelty_da_boost);
                    Some(id)
                }
                Err(_) => None,
            }
        } else if !is_known && !self.predictor.is_warm() && self.neuromod.plasticity_open() {
            // During warmup, store everything (building initial vocabulary)
            let bank = self.cluster.get_mut(self.bank_id).unwrap();
            match bank.insert(signal.to_vec(), Temperature::Hot, 0) {
                Ok(id) => Some(id),
                Err(_) => None,
            }
        } else {
            None
        };

        PerceptionResult {
            best_match,
            is_novel: !is_known,
            stored_as,
            surprise,
        }
    }

    /// Force-store a pattern with explicit temperature.
    pub fn store(&mut self, signal: &[Signal], temperature: Temperature) -> Option<EntryId> {
        let bank = self.cluster.get_mut(self.bank_id)?;
        bank.insert(signal.to_vec(), temperature, 0).ok()
    }

    /// Query without side effects — pure recall.
    pub fn recall(&self, cue: &[Signal], top_k: usize) -> Vec<QueryResult> {
        let bank = match self.cluster.get(self.bank_id) {
            Some(b) => b,
            None => return vec![],
        };
        bank.query_sparse(cue, top_k)
    }

    /// Associate two entries with a typed edge.
    pub fn associate(
        &mut self,
        from: EntryId,
        to: EntryId,
        edge_type: EdgeType,
        strength: u8,
    ) -> Result<(), databank_rs::DataBankError> {
        let target_ref = databank_rs::BankRef {
            bank: self.bank_id,
            entry: to,
        };
        let edge = Edge {
            edge_type,
            target: target_ref,
            weight: strength,
            created_tick: 0,
        };
        let bank = self
            .cluster
            .get_mut(self.bank_id)
            .ok_or(databank_rs::DataBankError::BankNotFound { id: self.bank_id })?;
        bank.add_edge(from, edge)
    }

    /// Save the store to disk.
    pub fn save(&self, path: &Path) -> Result<(), databank_rs::DataBankError> {
        if let Some(bank) = self.cluster.get(self.bank_id) {
            let file_path = path.join(format!("{}.bank", bank.name));
            databank_rs::codec::save_atomic(bank, &file_path)?;
        }
        Ok(())
    }

    /// Load a previously saved store from disk.
    pub fn load(path: &Path, name: &str, vector_width: u16) -> Result<Self, databank_rs::DataBankError> {
        let cluster = BankCluster::load_all(path)?;
        // Resolve the bank ID from the loaded cluster by name
        let bank_id = cluster
            .get_by_name(name)
            .map(|b| b.id)
            .unwrap_or_else(|| BankId::new(name, 0));

        Ok(Self {
            cluster,
            bank_id,
            neuromod: NeuromodState::new(),
            predictor: PredictionEngine::new(vector_width as usize, 3, 40),
            recall_threshold: 180,
            vector_width,
            novelty_da_boost: 30,
        })
    }

    /// Inject a neuromodulator chemical.
    pub fn inject_da(&mut self, amount: i8) {
        self.neuromod.inject(Chemical::Dopamine, amount);
    }

    /// Tick the neuromodulator state (decay toward baseline).
    pub fn tick(&mut self) {
        self.neuromod.tick();
    }

    /// Current neuromodulator state.
    pub fn neuromod(&self) -> &NeuromodState {
        &self.neuromod
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.cluster
            .get(self.bank_id)
            .map_or(0, |b| b.len())
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Access the underlying bank for direct operations.
    pub fn bank(&self) -> Option<&DataBank> {
        self.cluster.get(self.bank_id)
    }

    // ─── Internal ─────────────────────────────────────────────────────

    fn recall_best(&self, signal: &[Signal]) -> Option<QueryResult> {
        let results = self.recall(signal, 1);
        results.into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{accumulate, ENCODING_DIM};

    fn make_pattern(values: &[i32]) -> Vec<Signal> {
        values
            .iter()
            .map(|&v| {
                if v == 0 {
                    Signal::ZERO
                } else {
                    Signal::from_current(v)
                }
            })
            .collect()
    }

    fn pattern_64(base: i32) -> Vec<Signal> {
        let mut vals = vec![0i32; 64];
        for i in 0..8 {
            vals[i] = base + (i as i32 * 10);
        }
        make_pattern(&vals)
    }

    #[test]
    fn store_and_recall_roundtrip() {
        let mut mem = MemoryANT::new("test.memory", ENCODING_DIM as u16);
        let pattern = pattern_64(100);

        let id = mem.store(&pattern, Temperature::Hot).unwrap();
        assert_eq!(mem.len(), 1);

        let results = mem.recall(&pattern, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry_id, id);
        assert!(results[0].score > 200, "exact recall should have high score: {}", results[0].score);
    }

    #[test]
    fn partial_cue_recall() {
        let mut mem = MemoryANT::new("test.partial", ENCODING_DIM as u16);
        let full = pattern_64(100);
        mem.store(&full, Temperature::Hot).unwrap();

        // Partial cue: only first 4 dims active, rest zero
        let mut cue = vec![Signal::ZERO; ENCODING_DIM];
        for i in 0..4 {
            cue[i] = full[i];
        }
        let results = mem.recall(&cue, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0, "partial cue should still recall");
    }

    #[test]
    fn novelty_gating_da_below_threshold() {
        let mut mem = MemoryANT::with_config("test.gate", ENCODING_DIM as u16, 180, 200);
        // Suppress DA below gate
        mem.neuromod.dopamine = 50;

        let pattern = pattern_64(100);
        let result = mem.perceive(&pattern);
        assert!(result.is_novel);
        assert!(result.stored_as.is_none(), "should NOT store when DA gate is closed");
        assert_eq!(mem.len(), 0);
    }

    #[test]
    fn novelty_stores_when_da_open() {
        let mut mem = MemoryANT::new("test.store", ENCODING_DIM as u16);
        // Ensure DA is above gate
        mem.neuromod.dopamine = 200;

        let pattern = pattern_64(100);
        let result = mem.perceive(&pattern);
        assert!(result.is_novel);
        // During warmup, novel patterns are stored if DA gate is open
        assert!(result.stored_as.is_some(), "should store when DA gate open");
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn known_pattern_not_restored() {
        let mut mem = MemoryANT::new("test.known", ENCODING_DIM as u16);
        let pattern = pattern_64(100);

        // Force-store first
        mem.store(&pattern, Temperature::Hot).unwrap();
        assert_eq!(mem.len(), 1);

        // Perceive the same pattern — should be recognized, not stored again
        mem.neuromod.dopamine = 200;
        let result = mem.perceive(&pattern);
        assert!(!result.is_novel, "stored pattern should be recognized");
        assert!(result.stored_as.is_none(), "known pattern should not be re-stored");
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn association_between_entries() {
        let mut mem = MemoryANT::new("test.assoc", ENCODING_DIM as u16);
        let p1 = pattern_64(100);
        let p2 = pattern_64(-100);

        let id1 = mem.store(&p1, Temperature::Hot).unwrap();
        let id2 = mem.store(&p2, Temperature::Hot).unwrap();

        mem.associate(id1, id2, EdgeType::RelatedTo, 200).unwrap();

        let bank = mem.bank().unwrap();
        let edges = bank.edges_from(id1);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target.entry, id2);
        assert_eq!(edges[0].edge_type, EdgeType::RelatedTo);
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let pattern = pattern_64(100);

        // Create and store
        {
            let mut mem = MemoryANT::new("test.persist", ENCODING_DIM as u16);
            mem.store(&pattern, Temperature::Hot).unwrap();
            mem.save(dir.path()).unwrap();
        }

        // Load and verify
        let mem2 = MemoryANT::load(dir.path(), "test.persist", ENCODING_DIM as u16).unwrap();
        assert_eq!(mem2.len(), 1);
        let results = mem2.recall(&pattern, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 200, "should recall stored pattern after reload");
    }

    #[test]
    fn utf8_encoding_integration() {
        let mut mem = MemoryANT::new("test.utf8", ENCODING_DIM as u16);

        // Store accumulated UTF-8 patterns
        let void_sig = accumulate(b"void").to_vec();
        let int_sig = accumulate(b"int").to_vec();

        let id_void = mem.store(&void_sig, Temperature::Hot).unwrap();
        let id_int = mem.store(&int_sig, Temperature::Hot).unwrap();

        // Recall "void" — should find itself
        let results = mem.recall(&void_sig, 2);
        assert_eq!(results[0].entry_id, id_void);

        // Recall "int" — should find itself
        let results = mem.recall(&int_sig, 2);
        assert_eq!(results[0].entry_id, id_int);
    }

    #[test]
    fn tick_decays_neuromod() {
        let mut mem = MemoryANT::new("test.tick", ENCODING_DIM as u16);
        mem.inject_da(50);
        let da_before = mem.neuromod().dopamine;
        mem.tick();
        let da_after = mem.neuromod().dopamine;
        // DA should decay toward baseline (128) after tick
        assert!(
            (da_after as i16 - 128).abs() <= (da_before as i16 - 128).abs(),
            "DA should decay toward baseline: before={}, after={}",
            da_before,
            da_after
        );
    }
}
