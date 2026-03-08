//! UTF-8 Signal Encoding
//!
//! Deterministic, structured mapping from each byte (0-255) to a 64-dim
//! PackedSignal vector. Fixed lookup table — the structure carries semantic
//! information about character class, not learned features.
//!
//! ## Dimension Layout (64 dims)
//!
//! | Dims  | Purpose              |
//! |-------|----------------------|
//! | 0-7   | Character identity   |
//! | 8-15  | Class membership     |
//! | 16-23 | Subclass             |
//! | 24-31 | Lexical role         |
//! | 32-39 | Structural weight    |
//! | 40-47 | Frequency/commonality|
//! | 48-55 | Relational           |
//! | 56-63 | Reserved (zero)      |

use ternary_signal::PackedSignal;

/// Total dimensions per byte encoding.
pub const ENCODING_DIM: usize = 64;

/// The full lookup table: 256 entries × 64 dims.
/// Built once at compile time via `const` evaluation.
static UTF8_SIGNALS: [[PackedSignal; ENCODING_DIM]; 256] = build_table();

/// Get the 64-dim encoding for a single byte.
#[inline]
pub fn encode_byte(byte: u8) -> &'static [PackedSignal; ENCODING_DIM] {
    &UTF8_SIGNALS[byte as usize]
}

/// Encode a string as a sequence of byte encodings.
pub fn encode_str(s: &str) -> Vec<&'static [PackedSignal; ENCODING_DIM]> {
    s.as_bytes().iter().map(|&b| encode_byte(b)).collect()
}

/// Fold a variable-length byte sequence into a single 64-dim pattern signal.
/// Uses running accumulation with recency weighting: newer bytes have more
/// influence than older ones. The result is clamped to PackedSignal range.
///
/// This produces a fixed-size "fingerprint" suitable for databank storage/query.
pub fn accumulate(bytes: &[u8]) -> [PackedSignal; ENCODING_DIM] {
    if bytes.is_empty() {
        return [PackedSignal::ZERO; ENCODING_DIM];
    }

    // Accumulate in i32 space, then quantize back to PackedSignal.
    // Recency factor: each new byte contributes at full strength while
    // previous accumulation is decayed by 3/4.
    let mut acc = [0i32; ENCODING_DIM];

    for &byte in bytes {
        let enc = encode_byte(byte);
        for d in 0..ENCODING_DIM {
            // Decay previous accumulation (multiply by 3/4, integer)
            acc[d] = (acc[d] * 3) / 4;
            // Add new contribution at full strength
            acc[d] += enc[d].current();
        }
    }

    // Quantize back to PackedSignal
    let mut out = [PackedSignal::ZERO; ENCODING_DIM];
    for d in 0..ENCODING_DIM {
        if acc[d] != 0 {
            out[d] = PackedSignal::from_signal(&ternary_signal::Signal::from_current(acc[d]));
        }
    }
    out
}

// ─── Table construction ───────────────────────────────────────────────────────

const fn ps(pol: i8, mag: u8, mul: u8) -> PackedSignal {
    PackedSignal::pack(pol, mag, mul)
}

const fn build_table() -> [[PackedSignal; ENCODING_DIM]; 256] {
    let mut table = [[PackedSignal::ZERO; ENCODING_DIM]; 256];
    let mut i: usize = 0;
    while i < 256 {
        table[i] = encode_byte_const(i as u8);
        i += 1;
    }
    table
}

/// Const-evaluable encoding for a single byte.
/// This is the heart of the encoding — every structural decision lives here.
const fn encode_byte_const(byte: u8) -> [PackedSignal; ENCODING_DIM] {
    let mut v = [PackedSignal::ZERO; ENCODING_DIM];

    // ─── Dims 0-7: Character identity (fingerprint) ───────────────────
    // Spread the byte value across 8 dims using bit decomposition.
    // Each bit of the byte activates one dim as excitatory or inhibitory.
    let mut bit = 0;
    while bit < 8 {
        let b = (byte >> bit) & 1;
        if b == 1 {
            v[bit] = ps(1, 200, 1);
        } else {
            v[bit] = ps(-1, 100, 1);
        }
        bit += 1;
    }

    // ─── Dims 8-15: Class membership ──────────────────────────────────
    // letter, digit, operator, bracket, whitespace, punctuation, control, high-byte
    v[8] = if is_letter(byte) { ps(1, 200, 1) } else { ps(-1, 50, 1) };
    v[9] = if is_digit(byte) { ps(1, 200, 1) } else { ps(-1, 50, 1) };
    v[10] = if is_operator(byte) { ps(1, 200, 1) } else { ps(-1, 50, 1) };
    v[11] = if is_bracket(byte) { ps(1, 200, 1) } else { ps(-1, 50, 1) };
    v[12] = if is_whitespace(byte) { ps(1, 200, 1) } else { ps(-1, 50, 1) };
    v[13] = if is_punctuation(byte) { ps(1, 200, 1) } else { ps(-1, 50, 1) };
    v[14] = if byte < 32 || byte == 127 { ps(1, 200, 1) } else { ps(-1, 50, 1) }; // control
    v[15] = if byte >= 128 { ps(1, 200, 1) } else { ps(-1, 50, 1) }; // high-byte (UTF-8 continuation)

    // ─── Dims 16-23: Subclass ─────────────────────────────────────────
    v[16] = if is_uppercase(byte) { ps(1, 200, 1) } else if is_lowercase(byte) { ps(-1, 200, 1) } else { PackedSignal::ZERO };
    v[17] = if is_open_bracket(byte) { ps(1, 200, 1) } else if is_close_bracket(byte) { ps(-1, 200, 1) } else { PackedSignal::ZERO };
    v[18] = if is_arithmetic_op(byte) { ps(1, 200, 1) } else { PackedSignal::ZERO };
    v[19] = if is_comparison_op(byte) { ps(1, 200, 1) } else { PackedSignal::ZERO };
    v[20] = if is_bitwise_op(byte) { ps(1, 200, 1) } else { PackedSignal::ZERO };
    v[21] = if is_hex_digit(byte) { ps(1, 150, 1) } else { PackedSignal::ZERO };
    v[22] = if is_octal_digit(byte) { ps(1, 150, 1) } else { PackedSignal::ZERO };
    v[23] = if byte == b'_' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // underscore (identifier joiner)

    // ─── Dims 24-31: Lexical role ─────────────────────────────────────
    v[24] = if is_keyword_starter(byte) { ps(1, 180, 1) } else { PackedSignal::ZERO };
    v[25] = if is_identifier_valid(byte) { ps(1, 200, 1) } else { ps(-1, 100, 1) };
    v[26] = if is_string_delimiter(byte) { ps(1, 200, 1) } else { PackedSignal::ZERO };
    v[27] = if byte == b'#' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // preprocessor
    v[28] = if byte == b'\\' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // escape
    v[29] = if byte == b'0' { ps(1, 180, 1) } else if is_digit(byte) { ps(1, 100, 1) } else { PackedSignal::ZERO }; // numeric literal start
    v[30] = if byte == b'.' { ps(1, 180, 1) } else { PackedSignal::ZERO }; // member access / float
    v[31] = if byte == b'@' || byte == b'$' { ps(1, 150, 1) } else { PackedSignal::ZERO }; // sigil

    // ─── Dims 32-39: Structural weight ────────────────────────────────
    v[32] = if byte == b'{' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // scope opener
    v[33] = if byte == b'}' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // scope closer
    v[34] = if byte == b';' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // statement terminator
    v[35] = if byte == b',' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // separator
    v[36] = if byte == b':' { ps(1, 180, 1) } else { PackedSignal::ZERO }; // label / ternary / slice
    v[37] = if byte == b'\n' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // line break
    v[38] = if byte == b'=' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // assignment
    v[39] = if byte == b'*' || byte == b'&' { ps(1, 180, 1) } else { PackedSignal::ZERO }; // pointer/reference

    // ─── Dims 40-47: Frequency / commonality ──────────────────────────
    // How common this byte is in source code (approximate tiers).
    let freq = code_frequency_tier(byte);
    v[40] = if freq >= 4 { ps(1, 200, 1) } else { PackedSignal::ZERO }; // very common
    v[41] = if freq >= 3 { ps(1, 160, 1) } else { PackedSignal::ZERO }; // common
    v[42] = if freq >= 2 { ps(1, 120, 1) } else { PackedSignal::ZERO }; // moderate
    v[43] = if freq >= 1 { ps(1, 80, 1) } else { PackedSignal::ZERO };  // uncommon
    v[44] = if is_identifier_valid(byte) { ps(1, 180, 1) } else { PackedSignal::ZERO }; // expected in identifiers
    v[45] = if is_digit(byte) || byte == b'x' || byte == b'X' { ps(1, 150, 1) } else { PackedSignal::ZERO }; // expected in numeric literals
    v[46] = if byte == b' ' || byte == b'\t' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // indentation chars
    v[47] = if byte == b'/' || byte == b'*' { ps(1, 150, 1) } else { PackedSignal::ZERO }; // comment chars

    // ─── Dims 48-55: Relational ───────────────────────────────────────
    // Encodes pairing / association relationships.
    v[48] = bracket_pair_signal(byte);  // bracket pairing identity
    v[49] = if byte == b'"' || byte == b'\'' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // string pair
    v[50] = if is_letter(byte) || byte == b'_' || is_digit(byte) { ps(1, 180, 1) } else { PackedSignal::ZERO }; // identifier cohort
    v[51] = if byte == b'+' || byte == b'-' { ps(1, 150, 1) } else if byte == b'*' || byte == b'/' { ps(-1, 150, 1) } else { PackedSignal::ZERO }; // arithmetic pairing
    v[52] = if byte == b'<' || byte == b'>' { ps(1, 180, 1) } else { PackedSignal::ZERO }; // angle pair / comparison
    v[53] = if byte == b'&' || byte == b'|' { ps(1, 180, 1) } else { PackedSignal::ZERO }; // logical/bitwise pair
    v[54] = if byte == b'!' || byte == b'?' { ps(1, 150, 1) } else { PackedSignal::ZERO }; // assertion/query
    v[55] = if byte == b'-' && false { PackedSignal::ZERO } else if byte == b'>' { ps(1, 200, 1) } else { PackedSignal::ZERO }; // arrow component (>)

    // ─── Dims 56-63: Reserved (zero-filled) ──────────────────────────
    // Already initialized to ZERO.

    v
}

// ─── Character classification helpers (all const) ─────────────────────────────

const fn is_letter(b: u8) -> bool {
    (b >= b'A' && b <= b'Z') || (b >= b'a' && b <= b'z')
}

const fn is_uppercase(b: u8) -> bool {
    b >= b'A' && b <= b'Z'
}

const fn is_lowercase(b: u8) -> bool {
    b >= b'a' && b <= b'z'
}

const fn is_digit(b: u8) -> bool {
    b >= b'0' && b <= b'9'
}

const fn is_hex_digit(b: u8) -> bool {
    is_digit(b) || (b >= b'a' && b <= b'f') || (b >= b'A' && b <= b'F')
}

const fn is_octal_digit(b: u8) -> bool {
    b >= b'0' && b <= b'7'
}

const fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
}

const fn is_operator(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'*' | b'/' | b'%' | b'=' | b'!' | b'<' | b'>' | b'&' | b'|' | b'^' | b'~' | b'?')
}

const fn is_bracket(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'<' | b'>')
}

const fn is_open_bracket(b: u8) -> bool {
    matches!(b, b'(' | b'[' | b'{')
}

const fn is_close_bracket(b: u8) -> bool {
    matches!(b, b')' | b']' | b'}')
}

const fn is_punctuation(b: u8) -> bool {
    matches!(b, b'.' | b',' | b';' | b':' | b'\'' | b'"' | b'`' | b'#' | b'@' | b'$' | b'\\')
}

const fn is_arithmetic_op(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'*' | b'/' | b'%')
}

const fn is_comparison_op(b: u8) -> bool {
    matches!(b, b'<' | b'>' | b'=' | b'!')
}

const fn is_bitwise_op(b: u8) -> bool {
    matches!(b, b'&' | b'|' | b'^' | b'~')
}

const fn is_keyword_starter(b: u8) -> bool {
    // Letters that commonly start C/Rust keywords
    matches!(b, b'a' | b'b' | b'c' | b'd' | b'e' | b'f' | b'g' | b'i' | b'l' | b'm'
              | b'n' | b'p' | b'r' | b's' | b't' | b'u' | b'v' | b'w')
}

const fn is_identifier_valid(b: u8) -> bool {
    is_letter(b) || is_digit(b) || b == b'_'
}

const fn is_string_delimiter(b: u8) -> bool {
    b == b'"' || b == b'\'' || b == b'`'
}

/// Source code frequency tier (0=rare, 4=very common).
const fn code_frequency_tier(b: u8) -> u8 {
    match b {
        // Very common in code
        b' ' | b'e' | b't' | b'a' | b'i' | b'n' | b'o' | b's' | b'r' => 4,
        // Common
        b'l' | b'c' | b'u' | b'd' | b'p' | b'm' | b'_' | b'\n' | b'(' | b')' | b';' => 3,
        // Moderate
        b'f' | b'g' | b'h' | b'b' | b'v' | b'w' | b'x' | b'y' | b'=' | b',' | b'{' | b'}' | b'"' | b'0' | b'1' => 2,
        // Uncommon
        b'j' | b'k' | b'q' | b'z' | b'[' | b']' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/' | b'&' | b'|'
        | b'A'..=b'Z' | b'2'..=b'9' | b'.' | b':' | b'!' | b'#' | b'\\' | b'\'' | b'\t' => 1,
        // Rare
        _ => 0,
    }
}

/// Bracket pairing signal — matched pairs get the same magnitude but
/// openers are excitatory and closers are inhibitory.
const fn bracket_pair_signal(b: u8) -> PackedSignal {
    match b {
        b'(' => ps(1, 200, 1),
        b')' => ps(-1, 200, 1),
        b'[' => ps(1, 180, 1),
        b']' => ps(-1, 180, 1),
        b'{' => ps(1, 160, 1),
        b'}' => ps(-1, 160, 1),
        b'<' => ps(1, 140, 1),
        b'>' => ps(-1, 140, 1),
        _ => PackedSignal::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        // Same byte always produces the same encoding.
        let a1 = encode_byte(b'a');
        let a2 = encode_byte(b'a');
        for d in 0..ENCODING_DIM {
            assert_eq!(a1[d], a2[d], "dim {} mismatch for 'a'", d);
        }
    }

    #[test]
    fn distinct_bytes_differ() {
        let a = encode_byte(b'a');
        let b_enc = encode_byte(b'{');
        // Identity dims (0-7) must differ between 'a' (0x61) and '{' (0x7B)
        let mut diffs = 0;
        for d in 0..8 {
            if a[d] != b_enc[d] {
                diffs += 1;
            }
        }
        assert!(diffs > 0, "'a' and '{{' should differ in identity dims");
    }

    #[test]
    fn similar_chars_share_class() {
        let a = encode_byte(b'a');
        let b_enc = encode_byte(b'b');
        // Both are letters: dim 8 should be the same (excitatory)
        assert_eq!(a[8].polarity(), b_enc[8].polarity(), "'a' and 'b' should share letter class");
        // Both are lowercase: dim 16 should be inhibitory (lowercase branch)
        assert_eq!(a[16].polarity(), b_enc[16].polarity(), "'a' and 'b' should share lowercase subclass");
    }

    #[test]
    fn letter_vs_bracket_differ_in_class() {
        let a = encode_byte(b'a');
        let brace = encode_byte(b'{');
        // Dim 8: letter class — 'a' excitatory, '{' inhibitory
        assert_eq!(a[8].polarity(), 1);
        assert_eq!(brace[8].polarity(), -1);
        // Dim 11: bracket class — '{' excitatory, 'a' inhibitory
        assert_eq!(brace[11].polarity(), 1);
        assert_eq!(a[11].polarity(), -1);
    }

    #[test]
    fn digit_class_correct() {
        let five = encode_byte(b'5');
        assert_eq!(five[9].polarity(), 1, "'5' should be digit class");
        assert_eq!(five[8].polarity(), -1, "'5' should not be letter class");
    }

    #[test]
    fn accumulate_deterministic() {
        let r1 = accumulate(b"void");
        let r2 = accumulate(b"void");
        for d in 0..ENCODING_DIM {
            assert_eq!(r1[d], r2[d], "accumulate(\"void\") dim {} not deterministic", d);
        }
    }

    #[test]
    fn accumulate_distinct_words() {
        let void_sig = accumulate(b"void");
        let int_sig = accumulate(b"int");
        // Should differ in at least some dims
        let mut diffs = 0;
        for d in 0..ENCODING_DIM {
            if void_sig[d] != int_sig[d] {
                diffs += 1;
            }
        }
        assert!(diffs >= 4, "\"void\" and \"int\" should differ in multiple dims, got {}", diffs);
    }

    #[test]
    fn accumulate_empty_is_zero() {
        let empty = accumulate(b"");
        for d in 0..ENCODING_DIM {
            assert_eq!(empty[d], PackedSignal::ZERO, "empty accumulation should be all zero");
        }
    }

    #[test]
    fn bracket_pairing() {
        let open = encode_byte(b'(');
        let close = encode_byte(b')');
        // Dim 48: bracket pair — same magnitude, opposite polarity
        assert_eq!(open[48].polarity(), 1, "'(' should be excitatory in pair dim");
        assert_eq!(close[48].polarity(), -1, "')' should be inhibitory in pair dim");
    }

    #[test]
    fn structural_dims() {
        let brace = encode_byte(b'{');
        assert_eq!(brace[32].polarity(), 1, "'{{' should activate scope opener dim");

        let semi = encode_byte(b';');
        assert_eq!(semi[34].polarity(), 1, "';' should activate terminator dim");
    }

    #[test]
    fn all_256_bytes_encoded() {
        // Every byte should produce a non-zero encoding (at minimum identity dims are set).
        for byte in 0u8..=255 {
            let enc = encode_byte(byte);
            let mut has_nonzero = false;
            for d in 0..ENCODING_DIM {
                if enc[d] != PackedSignal::ZERO {
                    has_nonzero = true;
                    break;
                }
            }
            assert!(has_nonzero, "byte {} should have non-zero encoding", byte);
        }
    }

    #[test]
    fn encode_str_matches_bytes() {
        let s = "fn";
        let encoded = encode_str(s);
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0] as *const _, encode_byte(b'f') as *const _, "should return same static ref");
        assert_eq!(encoded[1] as *const _, encode_byte(b'n') as *const _, "should return same static ref");
    }

    #[test]
    fn similarity_structure() {
        // 'a' and 'b' should be more similar to each other than 'a' and '{'
        // Using simple dot product of current values as a similarity proxy
        let a = encode_byte(b'a');
        let b_enc = encode_byte(b'b');
        let brace = encode_byte(b'{');

        let sim_ab: i64 = (0..ENCODING_DIM)
            .map(|d| a[d].current() as i64 * b_enc[d].current() as i64)
            .sum();
        let sim_a_brace: i64 = (0..ENCODING_DIM)
            .map(|d| a[d].current() as i64 * brace[d].current() as i64)
            .sum();

        assert!(sim_ab > sim_a_brace, "sim(a,b)={} should > sim(a,{{}})={}", sim_ab, sim_a_brace);
    }
}
