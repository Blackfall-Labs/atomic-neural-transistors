//! Sequence operations composed from ANT primitives
//!
//! These operations don't require additional training - they compose
//! the learned equality primitive into higher-order functions.

use super::traits::EqualityChecker;

/// Check if query appears in sequence
///
/// Composed as: OR(AreEqual(query, seq[i]) for all i)
pub fn contains<E: EqualityChecker>(checker: &E, query: u32, sequence: &[u32]) -> bool {
    let max_prob: f32 = sequence
        .iter()
        .map(|&val| checker.check_equal(query, val))
        .fold(0.0, f32::max);

    max_prob > 0.5
}

/// Check if sequence has any duplicate values
///
/// Composed as: OR(AreEqual(seq[i], seq[j]) for all i < j)
pub fn has_duplicate<E: EqualityChecker>(checker: &E, sequence: &[u32]) -> bool {
    let n = sequence.len();
    let mut max_prob: f32 = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let prob = checker.check_equal(sequence[i], sequence[j]);
            if prob > max_prob {
                max_prob = prob;
            }
            // Early exit optimization
            if max_prob > 0.9 {
                return true;
            }
        }
    }

    max_prob > 0.5
}

/// Check if all values in sequence are unique
///
/// Composed as: NOT(has_duplicate)
pub fn all_unique<E: EqualityChecker>(checker: &E, sequence: &[u32]) -> bool {
    !has_duplicate(checker, sequence)
}

/// Count occurrences of value in sequence
///
/// Composed as: SUM(AreEqual(query, seq[i]) > 0.5 for all i)
pub fn count_occurrences<E: EqualityChecker>(checker: &E, query: u32, sequence: &[u32]) -> usize {
    sequence
        .iter()
        .filter(|&&val| checker.check_equal(query, val) > 0.5)
        .count()
}

/// Find all positions where value appears
pub fn find_positions<E: EqualityChecker>(
    checker: &E,
    query: u32,
    sequence: &[u32],
) -> Vec<usize> {
    sequence
        .iter()
        .enumerate()
        .filter(|(_, &val)| checker.check_equal(query, val) > 0.5)
        .map(|(i, _)| i)
        .collect()
}

/// Check if value after MARK token equals query
pub fn is_marked_equal<E: EqualityChecker>(
    checker: &E,
    query: u32,
    sequence: &[u32],
    mark_token: u32,
) -> Option<bool> {
    let mark_pos = sequence.iter().position(|&v| v == mark_token)?;
    if mark_pos + 1 >= sequence.len() {
        return None;
    }
    let target = sequence[mark_pos + 1];
    Some(checker.check_equal(query, target) > 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::PerfectEquality;

    #[test]
    fn test_contains() {
        let checker = PerfectEquality;
        assert!(contains(&checker, 5, &[1, 2, 3, 5, 7]));
        assert!(!contains(&checker, 9, &[1, 2, 3, 5, 7]));
        assert!(!contains(&checker, 5, &[]));
    }

    #[test]
    fn test_has_duplicate() {
        let checker = PerfectEquality;
        assert!(has_duplicate(&checker, &[1, 2, 3, 2, 5]));
        assert!(!has_duplicate(&checker, &[1, 2, 3, 4, 5]));
        assert!(!has_duplicate(&checker, &[]));
        assert!(!has_duplicate(&checker, &[1]));
        assert!(has_duplicate(&checker, &[1, 1]));
    }

    #[test]
    fn test_all_unique() {
        let checker = PerfectEquality;
        assert!(all_unique(&checker, &[1, 2, 3, 4, 5]));
        assert!(!all_unique(&checker, &[1, 2, 3, 2, 5]));
    }

    #[test]
    fn test_count_occurrences() {
        let checker = PerfectEquality;
        assert_eq!(count_occurrences(&checker, 2, &[1, 2, 3, 2, 5, 2]), 3);
        assert_eq!(count_occurrences(&checker, 9, &[1, 2, 3, 4, 5]), 0);
    }
}
