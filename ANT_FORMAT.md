# .ant Binary Format — v3

Persistence format for ThermalWeightMatrix. Binary, no JSON, no serde overhead.

## Changes from v2

- **v2**: Signal stored as PackedSignal (1 byte, lossy). 6 bytes per synaptic strength.
- **v3**: Signal stored as Polarity + magnitude + multiplier (3 bytes, lossless). 8 bytes per synaptic strength.
- **v2 files are incompatible.** Old .ant files trained with PackedSignal precision loss are not worth preserving.

## Layout

```
File Header (16 bytes):
  magic:      [u8; 4]    = b"ANT\x03"
  n_layers:   u16 (LE)   — number of synaptic matrices
  flags:      u16 (LE)   — reserved (0)
  checksum:   u32 (LE)   — CRC32 of everything after this header
  reserved:   u32        — (0)

Per-Layer Header (8 bytes):
  rows:       u16 (LE)
  cols:       u16 (LE)
  reserved:   u32        — (0)

Per Synaptic Strength (12 bytes):
  polarity:    i8         — -1 (inhibitory), 0 (silent), +1 (excitatory)
  magnitude:   u8         — base intensity (0-255)
  multiplier:  u8         — contextual scaling (0-255)
  temperature: u8         — 255=HOT → 0=COLD
  hits:        u16 (LE)   — reinforcement count
  pressure:    i16 (LE)   — accumulated mastery pressure
  streak:      u16 (LE)   — consecutive correct participations (resets on miss)
  confidence:  u16 (LE)   — accumulated prediction margin from correct participations
```

## Size Estimates

| Architecture | Strengths | File Size |
|-------------|-----------|-----------|
| 48×80 + 1×48 (phoneme detector) | 3,888 | ~31 KB |
| 784×300 + 300×100 + 100×10 (MNIST) | 266,200 | ~2.1 MB |
| 39 phoneme detectors | ~151,632 | ~1.2 MB |

## Serialization Rules

- **Polarity**: Written as raw `i8`. On load, values outside {-1, 0, 1} are a format error — reject the file.
- **Byte order**: All multi-byte integers are little-endian.
- **Checksum**: CRC32 of all bytes after the 16-byte file header (all layer headers + all synaptic strength data).
- **No compression**: Files are small enough that compression isn't worth the complexity.

## Round-Trip Guarantee

Save → load must produce byte-identical synaptic matrices. No precision loss. This is why PackedSignal was removed — it destroyed the round-trip guarantee for magnitude × multiplier products.
