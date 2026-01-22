//! Composition algebra for ANTs
//!
//! Complex operations composed from primitives without additional training.

mod grid;
mod sequence;
mod traits;

pub use grid::{
    find_all_objects, find_connected_component, grid_count_value, grid_find_value, grids_equal,
    Region,
};
pub use sequence::{
    all_unique, contains, count_occurrences, find_positions, has_duplicate, is_marked_equal,
};
pub use traits::{EqualityChecker, PerfectEquality};
