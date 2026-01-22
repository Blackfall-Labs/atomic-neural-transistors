//! Grid operations composed from ANT primitives

use super::traits::EqualityChecker;

/// Check if two grids are equal cell-by-cell
pub fn grids_equal<E: EqualityChecker>(checker: &E, a: &[Vec<u32>], b: &[Vec<u32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a.is_empty() {
        return true;
    }
    if a[0].len() != b[0].len() {
        return false;
    }

    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (&va, &vb) in row_a.iter().zip(row_b.iter()) {
            if checker.check_equal(va, vb) < 0.5 {
                return false;
            }
        }
    }
    true
}

/// Find all positions of a value in a grid
pub fn grid_find_value<E: EqualityChecker>(
    checker: &E,
    grid: &[Vec<u32>],
    value: u32,
) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    for (r, row) in grid.iter().enumerate() {
        for (c, &cell) in row.iter().enumerate() {
            if checker.check_equal(cell, value) > 0.5 {
                positions.push((r, c));
            }
        }
    }
    positions
}

/// Count occurrences of a value in a grid
pub fn grid_count_value<E: EqualityChecker>(checker: &E, grid: &[Vec<u32>], value: u32) -> usize {
    grid_find_value(checker, grid, value).len()
}

/// A connected region in a grid
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// Positions (row, col) in this region
    pub positions: Vec<(usize, usize)>,
    /// The value of cells in this region
    pub value: u32,
}

impl Region {
    /// Get bounding box: (min_row, min_col, max_row, max_col)
    pub fn bounding_box(&self) -> (usize, usize, usize, usize) {
        if self.positions.is_empty() {
            return (0, 0, 0, 0);
        }
        let min_r = self.positions.iter().map(|(r, _)| *r).min().unwrap();
        let max_r = self.positions.iter().map(|(r, _)| *r).max().unwrap();
        let min_c = self.positions.iter().map(|(_, c)| *c).min().unwrap();
        let max_c = self.positions.iter().map(|(_, c)| *c).max().unwrap();
        (min_r, min_c, max_r, max_c)
    }

    /// Get dimensions (height, width)
    pub fn dimensions(&self) -> (usize, usize) {
        let (min_r, min_c, max_r, max_c) = self.bounding_box();
        (max_r - min_r + 1, max_c - min_c + 1)
    }

    /// Number of cells in this region
    pub fn size(&self) -> usize {
        self.positions.len()
    }

    /// Convert region to a grid with background fill
    /// Returns a grid sized to the bounding box with the region's value at each position
    pub fn as_grid(&self, background: u32) -> Vec<Vec<u32>> {
        if self.positions.is_empty() {
            return vec![];
        }

        let (min_r, min_c, max_r, max_c) = self.bounding_box();
        let h = max_r - min_r + 1;
        let w = max_c - min_c + 1;

        // Fill with background
        let mut grid = vec![vec![background; w]; h];

        // Place region cells
        for &(r, c) in &self.positions {
            let local_r = r - min_r;
            let local_c = c - min_c;
            grid[local_r][local_c] = self.value;
        }

        grid
    }
}

/// Find connected component using flood-fill
pub fn find_connected_component<E: EqualityChecker>(
    checker: &E,
    grid: &[Vec<u32>],
    start_row: usize,
    start_col: usize,
    visited: &mut [Vec<bool>],
) -> Region {
    let h = grid.len();
    let w = if h > 0 { grid[0].len() } else { 0 };

    if start_row >= h || start_col >= w || visited[start_row][start_col] {
        return Region { positions: vec![], value: 0 };
    }

    let target = grid[start_row][start_col];
    let mut positions = Vec::new();
    let mut stack = vec![(start_row, start_col)];

    while let Some((r, c)) = stack.pop() {
        if r >= h || c >= w || visited[r][c] {
            continue;
        }
        if checker.check_equal(grid[r][c], target) < 0.5 {
            continue;
        }

        visited[r][c] = true;
        positions.push((r, c));

        // 4-connected neighbors
        if r > 0 { stack.push((r - 1, c)); }
        if r + 1 < h { stack.push((r + 1, c)); }
        if c > 0 { stack.push((r, c - 1)); }
        if c + 1 < w { stack.push((r, c + 1)); }
    }

    Region { positions, value: target }
}

/// Find all distinct objects (non-background connected regions)
pub fn find_all_objects<E: EqualityChecker>(
    checker: &E,
    grid: &[Vec<u32>],
    background: u32,
) -> Vec<Region> {
    let h = grid.len();
    let w = if h > 0 { grid[0].len() } else { 0 };
    let mut visited = vec![vec![false; w]; h];
    let mut objects = Vec::new();

    for r in 0..h {
        for c in 0..w {
            if visited[r][c] {
                continue;
            }
            if checker.check_equal(grid[r][c], background) > 0.5 {
                visited[r][c] = true;
                continue;
            }
            let region = find_connected_component(checker, grid, r, c, &mut visited);
            if !region.positions.is_empty() {
                objects.push(region);
            }
        }
    }

    objects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::PerfectEquality;

    #[test]
    fn test_grids_equal() {
        let checker = PerfectEquality;
        let a = vec![vec![1, 2], vec![3, 4]];
        let b = vec![vec![1, 2], vec![3, 4]];
        let c = vec![vec![1, 2], vec![3, 5]];

        assert!(grids_equal(&checker, &a, &b));
        assert!(!grids_equal(&checker, &a, &c));
    }

    #[test]
    fn test_find_all_objects() {
        let checker = PerfectEquality;
        let grid = vec![
            vec![1, 1, 0, 2, 2],
            vec![1, 0, 0, 0, 2],
            vec![0, 0, 3, 0, 0],
        ];

        let objects = find_all_objects(&checker, &grid, 0);
        assert_eq!(objects.len(), 3);
    }
}
