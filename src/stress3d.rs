//! 3D stress analysis of a part on a voxel grid: linear elastic, eight-node brick per voxel.
//!
//! The part's closed (or nearly closed) triangle mesh is filled with cubes: a voxel is material
//! when rays along at least two of the three axes say its centre is inside (counting crossings),
//! so a small gap in the mesh cannot flip a whole row of voxels. Only material voxels take part;
//! nodes touching no material are left out, and material not connected to the supports is
//! dropped with a warning (it would make the system singular).
//!
//! The system K u = f is solved matrix-free with preconditioned conjugate gradients: a geometric
//! multigrid V-cycle (`mg3d`) on grids big enough to need it, plain Jacobi otherwise.
//! Stresses are taken at voxel centres (von Mises). Units: mm, N, MPa.

use std::collections::BTreeSet;

pub type P3 = [f64; 3];
pub type Mat24 = [[f64; 24]; 24];

fn c3(p: &P3, k: usize) -> f64 {
    p.get(k).copied().unwrap_or(0.0)
}

fn n3(n: &[usize; 3], k: usize) -> usize {
    n.get(k).copied().unwrap_or(0)
}

/// Node offsets of a brick in local order: bottom face (z = 0) counter-clockwise, then top.
pub(crate) const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

/// Stiffness of a unit cube with E = 1 (2x2x2 Gauss). A cube of edge h has h times this.
pub fn brick_stiffness(nu: f64) -> Box<Mat24> {
    let c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let (d1, d2, d3) = (c * (1.0 - nu), c * nu, c * (1.0 - 2.0 * nu) / 2.0);
    let d = [
        [d1, d2, d2, 0.0, 0.0, 0.0],
        [d2, d1, d2, 0.0, 0.0, 0.0],
        [d2, d2, d1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, d3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, d3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, d3],
    ];
    let g = 1.0 / 3f64.sqrt();
    let mut k = Box::new([[0.0; 24]; 24]);
    for gx in [-g, g] {
        for gy in [-g, g] {
            for gz in [-g, g] {
                let b = strain_matrix(gx, gy, gz, 1.0);
                // k += B^T D B * det J (1/8), weights 1.
                let mut db = [[0.0; 24]; 6];
                for (dbrow, drow) in db.iter_mut().zip(&d) {
                    for (j, out) in dbrow.iter_mut().enumerate() {
                        *out = drow.iter().zip(&b).map(|(dv, brow)| dv * brow.get(j).copied().unwrap_or(0.0)).sum();
                    }
                }
                for (i, krow) in k.iter_mut().enumerate() {
                    for (j, kv) in krow.iter_mut().enumerate() {
                        let s: f64 = b.iter().zip(&db).map(|(brow, dbrow)| brow.get(i).copied().unwrap_or(0.0) * dbrow.get(j).copied().unwrap_or(0.0)).sum();
                        *kv += s * 0.125;
                    }
                }
            }
        }
    }
    k
}

/// Strain-displacement matrix (exx, eyy, ezz, gxy, gyz, gzx) of a cube of edge h at natural
/// coordinates (xi, eta, zeta).
fn strain_matrix(xi: f64, eta: f64, zeta: f64, h: f64) -> [[f64; 24]; 6] {
    let mut b = [[0.0; 24]; 6];
    for (n, corner) in CORNERS.iter().enumerate() {
        let s = corner.map(|c| if c == 0 { -1.0 } else { 1.0 });
        let [sx, sy, sz] = s;
        // dN/dxi = sx (1 + sy eta)(1 + sz zeta) / 8; d/dx = (2 / h) d/dxi.
        let dx = 2.0 / h * sx * (1.0 + sy * eta) * (1.0 + sz * zeta) / 8.0;
        let dy = 2.0 / h * sy * (1.0 + sx * xi) * (1.0 + sz * zeta) / 8.0;
        let dz = 2.0 / h * sz * (1.0 + sx * xi) * (1.0 + sy * eta) / 8.0;
        let col = 3 * n;
        let set = |b: &mut [[f64; 24]; 6], r: usize, c: usize, v: f64| {
            if let Some(slot) = b.get_mut(r).and_then(|row| row.get_mut(c)) {
                *slot = v;
            }
        };
        set(&mut b, 0, col, dx);
        set(&mut b, 1, col + 1, dy);
        set(&mut b, 2, col + 2, dz);
        set(&mut b, 3, col, dy);
        set(&mut b, 3, col + 1, dx);
        set(&mut b, 4, col + 1, dz);
        set(&mut b, 4, col + 2, dy);
        set(&mut b, 5, col, dz);
        set(&mut b, 5, col + 2, dx);
    }
    b
}

/// A voxel grid over a box: `n` cells per axis of edge `h` from `origin`.
#[derive(Clone, Debug)]
pub struct Grid {
    pub origin: P3,
    pub h: f64,
    pub n: [usize; 3],
}

impl Grid {
    pub fn cells(&self) -> usize {
        self.n.iter().product()
    }
    pub fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.n[1] + j) * self.n[2] + k
    }
    pub fn cell_ijk(&self, c: usize) -> [usize; 3] {
        let k = c % self.n[2];
        let j = (c / self.n[2]) % self.n[1];
        let i = c / (self.n[2] * self.n[1]);
        [i, j, k]
    }
    pub fn node_index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * (self.n[1] + 1) + j) * (self.n[2] + 1) + k
    }
    pub fn nodes(&self) -> usize {
        self.n.iter().map(|v| v + 1).product()
    }
    pub fn node_pos(&self, node: usize) -> P3 {
        let nz = self.n[2] + 1;
        let ny = self.n[1] + 1;
        let k = node % nz;
        let j = (node / nz) % ny;
        let i = node / (nz * ny);
        [self.origin[0] + i as f64 * self.h, self.origin[1] + j as f64 * self.h, self.origin[2] + k as f64 * self.h]
    }
    pub fn cell_centre(&self, c: usize) -> P3 {
        let [i, j, k] = self.cell_ijk(c);
        [
            self.origin[0] + (i as f64 + 0.5) * self.h,
            self.origin[1] + (j as f64 + 0.5) * self.h,
            self.origin[2] + (k as f64 + 0.5) * self.h,
        ]
    }
    /// The 8 nodes of cell (i, j, k), in brick order.
    pub fn cell_nodes(&self, c: usize) -> [usize; 8] {
        let [i, j, k] = self.cell_ijk(c);
        CORNERS.map(|[a, b, d]| self.node_index(i + a, j + b, k + d))
    }
}

/// Fills the mesh with voxels of edge `h`. Returns the grid and a material flag per cell.
pub fn voxelize(positions: &[P3], triangles: &[[u32; 3]], h: f64) -> Result<(Grid, Vec<bool>), String> {
    if !(h.is_finite() && h > 0.0) {
        return Err("voxel size must be positive".into());
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            let v = c3(p, k);
            if !v.is_finite() {
                return Err("the mesh has a non-finite point".into());
            }
            if let (Some(l), Some(u)) = (lo.get_mut(k), hi.get_mut(k)) {
                *l = l.min(v);
                *u = u.max(v);
            }
        }
    }
    if !lo.iter().zip(&hi).all(|(l, u)| u > l) {
        return Err("the mesh is empty or flat".into());
    }
    let n = [0, 1, 2].map(|k| (((c3(&hi, k) - c3(&lo, k)) / h).ceil() as usize).max(1));
    if n.iter().product::<usize>() > 20_000_000 {
        return Err("too many voxels; use a larger voxel size".into());
    }
    let grid = Grid { origin: lo, h, n };
    let mut votes = vec![0u8; grid.cells()];
    // Small irrational shifts so no ray runs exactly along a mesh edge.
    let jitter = [h * 1.234_567e-4, h * 2.718_281e-4, h * 1.414_213e-4];
    for axis in 0..3 {
        let (p, q) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let (np, nq, nd) = (n3(&n, p), n3(&n, q), n3(&n, axis));
        let mut hits: Vec<Vec<f64>> = vec![Vec::new(); np * nq];
        for tri in triangles {
            let [Some(a), Some(b), Some(c)] = tri.map(|i| positions.get(i as usize)) else {
                return Err("the mesh has a triangle with a bad vertex index".into());
            };
            let (ap, aq, ad) = (c3(a, p), c3(a, q), c3(a, axis));
            let (bp, bq, bd) = (c3(b, p), c3(b, q), c3(b, axis));
            let (cp, cq, cd) = (c3(c, p), c3(c, q), c3(c, axis));
            let det = (bp - ap) * (cq - aq) - (cp - ap) * (bq - aq);
            if det.abs() < 1e-14 * h * h {
                continue;
            }
            let lo_p = (((ap.min(bp).min(cp) - c3(&lo, p)) / h - 0.5).floor().max(0.0)) as usize;
            let hi_p = ((((ap.max(bp).max(cp) - c3(&lo, p)) / h - 0.5).ceil().max(0.0)) as usize).min(np.saturating_sub(1));
            let lo_q = (((aq.min(bq).min(cq) - c3(&lo, q)) / h - 0.5).floor().max(0.0)) as usize;
            let hi_q = ((((aq.max(bq).max(cq) - c3(&lo, q)) / h - 0.5).ceil().max(0.0)) as usize).min(nq.saturating_sub(1));
            for ip in lo_p..=hi_p {
                for iq in lo_q..=hi_q {
                    let xp = c3(&lo, p) + (ip as f64 + 0.5) * h + c3(&jitter, p);
                    let xq = c3(&lo, q) + (iq as f64 + 0.5) * h + c3(&jitter, q);
                    let s = ((xp - ap) * (cq - aq) - (cp - ap) * (xq - aq)) / det;
                    let t = ((bp - ap) * (xq - aq) - (xp - ap) * (bq - aq)) / det;
                    if s < 0.0 || t < 0.0 || s + t > 1.0 {
                        continue;
                    }
                    if let Some(list) = hits.get_mut(ip * nq + iq) {
                        list.push(ad + s * (bd - ad) + t * (cd - ad));
                    }
                }
            }
        }
        for ip in 0..np {
            for iq in 0..nq {
                let Some(list) = hits.get_mut(ip * nq + iq) else { continue };
                list.sort_by(f64::total_cmp);
                let mut crossings = 0usize;
                let mut next = list.iter().peekable();
                for id in 0..nd {
                    let centre = c3(&lo, axis) + (id as f64 + 0.5) * h + c3(&jitter, axis);
                    while next.peek().is_some_and(|z| **z < centre) {
                        next.next();
                        crossings += 1;
                    }
                    if crossings % 2 == 1 {
                        let mut ijk = [0usize; 3];
                        for (ax, v) in [(p, ip), (q, iq), (axis, id)] {
                            if let Some(slot) = ijk.get_mut(ax) {
                                *slot = v;
                            }
                        }
                        let [i, j, k] = ijk;
                        if let Some(vote) = votes.get_mut(grid.cell_index(i, j, k)) {
                            *vote += 1;
                        }
                    }
                }
            }
        }
    }
    let solid = votes.iter().map(|v| *v >= 2).collect();
    Ok((grid, solid))
}

/// Loads and supports.
pub enum Region {
    /// Material nodes within `depth` of a side of the bounding box: axis 0..3, max side or not.
    Side { axis: usize, max: bool, depth: f64 },
    /// Material nodes whose distance to a line (point, direction) lies in [r_in, r_out].
    Ring { point: P3, dir: P3, r_in: f64, r_out: f64 },
    /// Material nodes within `r` of the axis from `point` along `dir`, between 0 and `len` along it
    /// (the material a screw clamps).
    Cylinder { point: P3, dir: P3, r: f64, len: f64 },
    /// Material nodes in any of these.
    Union(Vec<Region>),
}

/// Distance along the unit direction of `dir` and distance from that axis, of p seen from `point`.
fn axial(p: &P3, point: &P3, dir: &P3) -> (f64, f64) {
    let len = (dir.iter().map(|v| v * v).sum::<f64>()).sqrt().max(f64::MIN_POSITIVE);
    let d = dir.map(|v| v / len);
    let r = [p[0] - point[0], p[1] - point[1], p[2] - point[2]];
    let along: f64 = r.iter().zip(&d).map(|(a, b)| a * b).sum();
    (along, (r.iter().map(|v| v * v).sum::<f64>() - along * along).max(0.0).sqrt())
}

fn inside(region: &Region, p: &P3, lo: &P3, hi: &P3) -> bool {
    match region {
        Region::Side { axis, max, depth } => {
            let v = c3(p, *axis);
            if *max { v >= c3(hi, *axis) - depth } else { v <= c3(lo, *axis) + depth }
        }
        Region::Ring { point, dir, r_in, r_out } => {
            let perp = axial(p, point, dir).1;
            perp >= *r_in && perp <= *r_out
        }
        Region::Cylinder { point, dir, r, len } => {
            let (along, perp) = axial(p, point, dir);
            perp <= *r && (0.0..=*len).contains(&along)
        }
        Region::Union(parts) => parts.iter().any(|q| inside(q, p, lo, hi)),
    }
}

pub struct Case {
    pub name: String,
    /// Total force (N) spread evenly over the load nodes.
    pub force: P3,
    /// A couple (N mm) about the load nodes' centroid, spread as a rigid rotation would.
    pub moment: P3,
}

pub struct Setup<'a> {
    pub grid: &'a Grid,
    pub solid: &'a [bool],
    pub e_mpa: f64,
    pub nu: f64,
    pub fixed: &'a Region,
    pub load: &'a Region,
    pub cg_tol: f64,
    pub cg_max: usize,
    /// Voxel layers around the load and support points left out of the "away" peak.
    pub exclude_layers: usize,
    /// Relative stiffness per grid cell (1.0 = the full material); None = all cells 1.0. Lets a
    /// printed part be modelled as a solid shell around a lighter infill core.
    pub cell_scale: Option<&'a [f64]>,
    /// Precondition with multigrid (grids of a few thousand free DOFs or more); false = Jacobi.
    pub multigrid: bool,
    /// Nodes resting on a rigid surface (a wall): held along `axis` only while the surface pushes in the
    /// `push` direction (+1 or -1), free to lift off, no friction. Settled per case by re-solving.
    pub contact: Option<Contact<'a>>,
}

pub struct Contact<'a> {
    pub region: &'a Region,
    pub axis: usize,
    pub push: f64,
}

pub struct CaseResult {
    pub name: String,
    pub vm: Vec<f64>, // per cell, NAN for void
    /// Stress per cell (sx, sy, sz, txy, tyz, tzx) in MPa, as the cell's own stiffness gives it; NAN for void.
    pub stress: Vec<[f64; 6]>,
    pub max_vm: f64,
    pub max_vm_at: P3,
    pub p99_vm: f64,
    /// The peak outside the zones where the load goes in and the supports hold (those peaks
    /// depend on how the load is applied and grow with finer grids).
    pub max_vm_away: f64,
    pub max_vm_away_at: P3,
    pub max_disp: f64,
    pub max_disp_at: P3,
    pub cg_iterations: usize,
    pub converged: bool,
    /// Contact nodes the surface pushes on at the end, and the solves it took to settle (1 = no contact).
    pub contact_nodes: usize,
    pub contact_rounds: usize,
    /// Per cell: left out of the "away" peak (near the load, the supports and where the wall pushes).
    pub near: Vec<bool>,
}

/// Von Mises stress of (sx, sy, sz, txy, tyz, tzx).
pub fn von_mises(st: &[f64; 6]) -> f64 {
    let [sx, sy, sz, txy, tyz, tzx] = *st;
    (0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2)) + 3.0 * (txy * txy + tyz * tyz + tzx * tzx)).sqrt()
}

/// What pulls a printed part's layers apart when they stack along `axis` (0 x, 1 y, 2 z): the tension across
/// the layers and the two shears on the layer plane, combined (as in stress3d_printed). Pressure counts nothing.
pub fn across_layers(st: &[f64; 6], axis: usize) -> f64 {
    let [sx, sy, sz, txy, tyz, tzx] = *st;
    let (n, a, b) = match axis {
        0 => (sx, txy, tzx),
        1 => (sy, txy, tyz),
        _ => (sz, tyz, tzx),
    };
    (n.max(0.0).powi(2) + a * a + b * b).sqrt()
}

/// The stress a printed cell is judged by, against the in-layer strength: von Mises, or the across-layer demand
/// scaled up by in-layer / across-layer strength (`ratio`), whichever is larger.
pub fn printed_stress(st: &[f64; 6], axis: usize, ratio: f64) -> f64 {
    von_mises(st).max(across_layers(st, axis) * ratio)
}

impl CaseResult {
    /// A printed part, layers stacked along `axis`: every cell's `vm` becomes [`printed_stress`], and the
    /// peaks (all, away from load and supports, 99th percentile) are taken again. Stresses are unchanged.
    pub fn with_layers(&mut self, grid: &Grid, axis: usize, ratio: f64) {
        for (v, st) in self.vm.iter_mut().zip(&self.stress) {
            if v.is_finite() {
                *v = printed_stress(st, axis, ratio);
            }
        }
        let (mut max, mut max_c, mut away, mut away_c) = (0.0, 0, 0.0, 0);
        let mut sorted = Vec::new();
        for (c, v) in self.vm.iter().enumerate().filter(|(_, v)| v.is_finite()) {
            sorted.push(*v);
            if *v > max {
                (max, max_c) = (*v, c);
            }
            if *v > away && !self.near.get(c).copied().unwrap_or(true) {
                (away, away_c) = (*v, c);
            }
        }
        sorted.sort_by(f64::total_cmp);
        self.p99_vm = sorted.get(((sorted.len() as f64 * 0.99) as usize).min(sorted.len().saturating_sub(1))).copied().unwrap_or(0.0);
        (self.max_vm, self.max_vm_at, self.max_vm_away, self.max_vm_away_at) = (max, grid.cell_centre(max_c), away, grid.cell_centre(away_c));
    }
}

pub struct Analysis {
    pub cells_material: usize,
    pub cells_dropped: usize,
    pub fixed_nodes: usize,
    pub load_nodes: usize,
    pub cases: Vec<CaseResult>,
    pub warnings: Vec<String>,
}

fn select(region: &Region, grid: &Grid, in_material: &[bool], lo: &P3, hi: &P3) -> Vec<usize> {
    (0..grid.nodes())
        .filter(|n| in_material.get(*n).copied().unwrap_or(false))
        .filter(|n| inside(region, &grid.node_pos(*n), lo, hi))
        .collect()
}

/// Runs every case on the same supports and load region.
pub fn analyse(s: &Setup, cases: &[Case], mut progress: impl FnMut(&str)) -> Result<Analysis, String> {
    let grid = s.grid;
    if !(s.e_mpa > 0.0 && s.e_mpa.is_finite() && s.nu >= 0.0 && s.nu < 0.5) {
        return Err("material needs E > 0 and 0 <= nu < 0.5".into());
    }
    let mut solid = s.solid.to_vec();
    let mut warnings = Vec::new();
    let mut in_material = vec![false; grid.nodes()];
    for (c, m) in solid.iter().enumerate() {
        if *m {
            for n in grid.cell_nodes(c) {
                if let Some(slot) = in_material.get_mut(n) {
                    *slot = true;
                }
            }
        }
    }
    // The material's true extent (the grid box is rounded up).
    let mut mlo = [f64::INFINITY; 3];
    let mut mhi = [f64::NEG_INFINITY; 3];
    for (n, m) in in_material.iter().enumerate() {
        if *m {
            let p = grid.node_pos(n);
            for ((l, u), v) in mlo.iter_mut().zip(mhi.iter_mut()).zip(p) {
                *l = l.min(v);
                *u = u.max(v);
            }
        }
    }
    let fixed_nodes = select(s.fixed, grid, &in_material, &mlo, &mhi);
    let load_nodes = select(s.load, grid, &in_material, &mlo, &mhi);
    if fixed_nodes.is_empty() {
        return Err("the support region touches no material".into());
    }
    if load_nodes.is_empty() {
        return Err("the load region touches no material".into());
    }

    // Keep only material connected to the supports (cells sharing a node are connected).
    let fixed_set: BTreeSet<usize> = fixed_nodes.iter().copied().collect();
    let mut node_cells: Vec<Vec<usize>> = vec![Vec::new(); grid.nodes()];
    for (c, m) in solid.iter().enumerate() {
        if *m {
            for n in grid.cell_nodes(c) {
                if let Some(list) = node_cells.get_mut(n) {
                    list.push(c);
                }
            }
        }
    }
    let mut reached = vec![false; grid.cells()];
    let mut stack: Vec<usize> = fixed_nodes.iter().flat_map(|n| node_cells.get(*n).cloned().unwrap_or_default()).collect();
    while let Some(c) = stack.pop() {
        if reached.get(c).copied().unwrap_or(true) {
            continue;
        }
        if let Some(r) = reached.get_mut(c) {
            *r = true;
        }
        for n in grid.cell_nodes(c) {
            for next in node_cells.get(n).map(Vec::as_slice).unwrap_or(&[]) {
                if !reached.get(*next).copied().unwrap_or(true) {
                    stack.push(*next);
                }
            }
        }
    }
    let mut dropped = 0;
    for (m, r) in solid.iter_mut().zip(&reached) {
        if *m && !*r {
            *m = false;
            dropped += 1;
        }
    }
    if dropped > 0 {
        warnings.push(format!("{dropped} voxels are not connected to the supports and were left out"));
    }
    let cells: Vec<usize> = solid.iter().enumerate().filter(|(_, m)| **m).map(|(c, _)| c).collect();
    let mut active = vec![false; grid.nodes()];
    for c in &cells {
        for n in grid.cell_nodes(*c) {
            if let Some(slot) = active.get_mut(n) {
                *slot = true;
            }
        }
    }
    let load_nodes: Vec<usize> = load_nodes.into_iter().filter(|n| active.get(*n).copied().unwrap_or(false) && !fixed_set.contains(n)).collect();
    if load_nodes.is_empty() {
        return Err("the load region lies on the supports or on material not connected to them".into());
    }

    // Cells near the load and support points: those touching them, grown by `exclude_layers`.
    // cells touching the `touch` nodes, grown by `exclude_layers`
    let grow = |touch: &[bool]| {
        let mut near: Vec<bool> = (0..grid.cells()).map(|c| grid.cell_nodes(c).iter().any(|n| touch.get(*n).copied().unwrap_or(false))).collect();
        for _ in 0..s.exclude_layers {
            let before = near.clone();
            for (c, slot) in near.iter_mut().enumerate() {
                if *slot {
                    continue;
                }
                let [i, j, k] = grid.cell_ijk(c);
                let mut hit = false;
                for di in -1i64..=1 {
                    for dj in -1i64..=1 {
                        for dk in -1i64..=1 {
                            let (a, b, d) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                            if a >= 0 && b >= 0 && d >= 0 && (a as usize) < grid.n[0] && (b as usize) < grid.n[1] && (d as usize) < grid.n[2] {
                                hit |= before.get(grid.cell_index(a as usize, b as usize, d as usize)).copied().unwrap_or(false);
                            }
                        }
                    }
                }
                *slot = hit;
            }
        }
        near
    };
    let mut touch = vec![false; grid.nodes()];
    for n in load_nodes.iter().chain(&fixed_nodes) {
        if let Some(t) = touch.get_mut(*n) {
            *t = true;
        }
    }
    let near = grow(&touch);

    let ndof = 3 * grid.nodes();
    let mut fixed = vec![false; ndof];
    for (n, a) in active.iter().enumerate() {
        let hold = !*a || fixed_set.contains(&n);
        if hold {
            for d in 0..3 {
                if let Some(slot) = fixed.get_mut(3 * n + d) {
                    *slot = true;
                }
            }
        }
    }
    let ke = brick_stiffness(s.nu);
    let scale = s.e_mpa * grid.h;
    let cell_rel = |c: usize| s.cell_scale.and_then(|v| v.get(c).copied()).unwrap_or(1.0);
    let cell_k: Vec<f64> = cells.iter().map(|c| scale * cell_rel(*c)).collect();
    let dofs: Vec<[usize; 24]> = cells
        .iter()
        .map(|c| {
            let nodes = grid.cell_nodes(*c);
            std::array::from_fn(|i| 3 * nodes.get(i / 3).copied().unwrap_or(0) + i % 3)
        })
        .collect();
    // Contact DOFs: on material, not already held, not loaded.
    let load_set: BTreeSet<usize> = load_nodes.iter().copied().collect();
    let contact_dofs: Vec<usize> = match &s.contact {
        Some(c) if c.axis < 3 => select(c.region, grid, &active, &mlo, &mhi).into_iter().filter(|n| !fixed_set.contains(n) && !load_set.contains(n)).map(|n| 3 * n + c.axis).collect(),
        Some(_) => return Err("contact axis must be 0, 1 or 2".into()),
        None => Vec::new(),
    };
    let push = s.contact.as_ref().map_or(1.0, |c| c.push.signum());
    let base_fixed = fixed;
    // K u; DOFs in `fixed` give 0 (an empty mask gives the full product).
    let apply_with = |fixed: &[bool], u: &[f64], y: &mut [f64]| {
        y.iter_mut().for_each(|v| *v = 0.0);
        for (d, ck) in dofs.iter().zip(&cell_k) {
            let ue = d.map(|i| u.get(i).copied().unwrap_or(0.0));
            let mut ye = [0.0f64; 24];
            for (row, v) in ke.iter().zip(ue) {
                for (out, kv) in ye.iter_mut().zip(row) {
                    *out += kv * v;
                }
            }
            for (i, v) in d.iter().zip(ye) {
                if let Some(slot) = y.get_mut(*i) {
                    *slot += ck * v;
                }
            }
        }
        for (v, f) in y.iter_mut().zip(fixed) {
            if *f {
                *v = 0.0;
            }
        }
    };
    let mut diag = vec![0.0; ndof];
    for (d, ck) in dofs.iter().zip(&cell_k) {
        for (k, i) in d.iter().enumerate() {
            if let Some(slot) = diag.get_mut(*i) {
                *slot += ck * ke.get(k).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
            }
        }
    }
    let mut stiffness = vec![0.0; grid.cells()];
    for (c, ck) in cells.iter().zip(&cell_k) {
        if let Some(slot) = stiffness.get_mut(*c) {
            *slot = *ck;
        }
    }
    let mut base_support = vec![false; ndof];
    for n in &fixed_set {
        for d in 0..3 {
            if let Some(slot) = base_support.get_mut(3 * n + d) {
                *slot = true;
            }
        }
    }
    // Inverse diagonal and multigrid levels for one set of held DOFs; rebuilt only when the contact set changes,
    // so without contact they are built once and serve every load case.
    let mut built: Option<(Vec<bool>, Vec<f64>, Option<crate::mg3d::Hierarchy3>)> = None;
    let build = |fixed: &[bool], warnings: &mut Vec<String>, progress: &mut dyn FnMut(&str)| {
        let inv: Vec<f64> = diag.iter().zip(fixed).map(|(d, f)| if *f || *d <= 0.0 { 1.0 } else { 1.0 / d }).collect();
        let hierarchy = if s.multigrid {
            let support: Vec<bool> = fixed.iter().zip(&base_support).zip(&base_fixed).map(|((f, b), was)| *b || (*f && !*was)).collect();
            let system = crate::mg3d::FineSystem { n: grid.n, ke: &ke, cell_k: &stiffness, fixed, support: &support };
            progress("building the multigrid levels");
            let apply = |u: &[f64], y: &mut [f64]| apply_with(fixed, u, y);
            match crate::mg3d::Hierarchy3::build(&system, crate::mg3d::Settings::default(), &crate::mg3d::FineOps { apply: &apply, inv: &inv }) {
                Ok(h) => Some(h),
                Err(reason) => {
                    let w = format!("multigrid not used ({reason}); falling back to the plain solver");
                    if !reason.contains("small enough") && !warnings.contains(&w) {
                        warnings.push(w);
                    }
                    None
                }
            }
        } else {
            None
        };
        (fixed.to_vec(), inv, hierarchy)
    };

    // Load distribution over the load nodes: even force, couple as a rigid rotation.
    let pts: Vec<P3> = load_nodes.iter().map(|n| grid.node_pos(*n)).collect();
    let count = pts.len() as f64;
    let centroid = [0, 1, 2].map(|k| pts.iter().map(|p| c3(p, k)).sum::<f64>() / count);
    let rel: Vec<P3> = pts.iter().map(|p| [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]]).collect();
    let mut inertia = [[0.0; 3]; 3];
    for r in &rel {
        let r2 = r.iter().map(|v| v * v).sum::<f64>();
        for (i, row) in inertia.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v += if i == j { r2 } else { 0.0 } - c3(r, i) * c3(r, j);
            }
        }
    }

    let mut results = Vec::new();
    for case in cases {
        progress(&format!("solving {}", case.name));
        let mut f = vec![0.0; ndof];
        let w = if case.moment.iter().any(|m| *m != 0.0) {
            solve3(&inertia, &case.moment).ok_or("the load region cannot take a couple about that axis (its nodes lie on a line)")?
        } else {
            [0.0; 3]
        };
        for (n, r) in load_nodes.iter().zip(&rel) {
            let rot = [w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2], w[0] * r[1] - w[1] * r[0]];
            for (k, (fk, rk)) in case.force.iter().zip(rot).enumerate() {
                if let Some(slot) = f.get_mut(3 * n + k) {
                    *slot += fk / count + rk;
                }
            }
        }
        let mut u = vec![0.0; ndof];
        // Start with the surface pushing on every contact node; drop the ones it would have to pull, add back
        // the ones that would sink into it, until no node changes.
        let mut pressed = vec![true; contact_dofs.len()];
        // a node at the edge of the pressed area can flip back and forth; after two switches it stays pressed
        // (it carries next to nothing either way)
        let mut switches = vec![0u8; contact_dofs.len()];
        let (mut it, mut ok, mut rounds) = (0, true, 0);
        let fnorm = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        // settle on loose solves (each warm-starts the next), then one solve at the full tolerance
        let mut settled = contact_dofs.is_empty();
        loop {
            let tol = if settled { s.cg_tol } else { s.cg_tol.max(1e-5) };
            rounds += 1;
            let mut held = base_fixed.clone();
            for (d, p) in contact_dofs.iter().zip(&pressed) {
                if *p && let Some(slot) = held.get_mut(*d) {
                    *slot = true;
                    if let Some(ud) = u.get_mut(*d) {
                        *ud = 0.0;
                    }
                }
            }
            if built.as_ref().is_none_or(|b| b.0 != held) {
                built = Some(build(&held, &mut warnings, &mut progress));
            }
            let Some((_, inv, hierarchy)) = built.as_ref() else { unreachable!() };
            let apply = |x: &[f64], y: &mut [f64]| apply_with(&held, x, y);
            let precondition = |r: &[f64], z: &mut [f64]| match hierarchy {
                Some(h) => z.copy_from_slice(&h.precondition(&crate::mg3d::FineOps { apply: &apply, inv }, r)),
                None => {
                    for ((zi, ri), di) in z.iter_mut().zip(r).zip(inv) {
                        *zi = ri * di;
                    }
                }
            };
            let (i, o) = pcg(&apply, &precondition, &f, &mut u, tol, s.cg_max);
            it += i;
            ok &= o;
            if settled {
                break;
            }
            let mut ku = vec![0.0; ndof];
            apply_with(&[], &u, &mut ku);
            let umax = u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let mut changed = 0;
            for ((d, p), n) in contact_dofs.iter().zip(pressed.iter_mut()).zip(switches.iter_mut()) {
                if *n >= 2 {
                    continue;
                }
                let reaction = push * (ku.get(*d).copied().unwrap_or(0.0) - f.get(*d).copied().unwrap_or(0.0));
                let sink = push * u.get(*d).copied().unwrap_or(0.0);
                let now = if *p { reaction >= -10.0 * tol * fnorm } else { sink < -10.0 * tol * umax };
                if now != *p {
                    *p = now;
                    *n += 1;
                    changed += 1;
                }
            }
            if changed == 0 || rounds >= 30 {
                if changed > 0 {
                    warnings.push(format!("{}: the wall contact did not settle in 30 solves ({changed} nodes still switching)", case.name));
                }
                settled = true;
                if tol <= s.cg_tol {
                    break;
                }
                continue;
            }
            progress(&format!("{}: wall contact, {changed} nodes switched", case.name));
        }
        let contact_nodes = pressed.iter().filter(|p| **p).count();
        // where the rigid wall pushes is a support too: its edge peaks are left out of "away" as well
        let near_case = if contact_nodes > 0 {
            let mut t = touch.clone();
            for (d, p) in contact_dofs.iter().zip(&pressed) {
                if *p && let Some(slot) = t.get_mut(d / 3) {
                    *slot = true;
                }
            }
            grow(&t)
        } else {
            near.clone()
        };
        // Stresses at cell centres.
        let b0 = strain_matrix(0.0, 0.0, 0.0, grid.h);
        let lam = s.e_mpa * s.nu / ((1.0 + s.nu) * (1.0 - 2.0 * s.nu));
        let mu = s.e_mpa / (2.0 * (1.0 + s.nu));
        let mut vm = vec![f64::NAN; grid.cells()];
        let mut stress = vec![[f64::NAN; 6]; grid.cells()];
        for (c, d) in cells.iter().zip(&dofs) {
            let (lam, mu) = (lam * cell_rel(*c), mu * cell_rel(*c));
            let ue = d.map(|i| u.get(i).copied().unwrap_or(0.0));
            let e: [f64; 6] = std::array::from_fn(|r| b0.get(r).map(|row| row.iter().zip(&ue).map(|(a, b)| a * b).sum()).unwrap_or(0.0));
            let tr = e[0] + e[1] + e[2];
            let (sx, sy, sz) = (lam * tr + 2.0 * mu * e[0], lam * tr + 2.0 * mu * e[1], lam * tr + 2.0 * mu * e[2]);
            let (txy, tyz, tzx) = (mu * e[3], mu * e[4], mu * e[5]);
            let v = (0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2)) + 3.0 * (txy * txy + tyz * tyz + tzx * tzx)).sqrt();
            if let Some(slot) = vm.get_mut(*c) {
                *slot = v;
            }
            if let Some(slot) = stress.get_mut(*c) {
                *slot = [sx, sy, sz, txy, tyz, tzx];
            }
        }
        let (mut max_vm, mut max_c) = (0.0, 0);
        for c in &cells {
            let v = vm.get(*c).copied().unwrap_or(0.0);
            if v > max_vm {
                max_vm = v;
                max_c = *c;
            }
        }
        let (mut max_away, mut away_c) = (0.0, 0);
        for c in &cells {
            let v = vm.get(*c).copied().unwrap_or(0.0);
            if v > max_away && !near_case.get(*c).copied().unwrap_or(true) {
                max_away = v;
                away_c = *c;
            }
        }
        let mut sorted: Vec<f64> = cells.iter().filter_map(|c| vm.get(*c).copied()).collect();
        sorted.sort_by(f64::total_cmp);
        let p99 = sorted.get(((sorted.len() as f64 * 0.99) as usize).min(sorted.len().saturating_sub(1))).copied().unwrap_or(0.0);
        let (mut max_disp, mut max_n) = (0.0, 0);
        for (n, a) in active.iter().enumerate() {
            if *a {
                let d = (0..3).map(|k| u.get(3 * n + k).copied().unwrap_or(0.0).powi(2)).sum::<f64>().sqrt();
                if d > max_disp {
                    max_disp = d;
                    max_n = n;
                }
            }
        }
        if !ok {
            warnings.push(format!("{}: the solver stopped before full convergence", case.name));
        }
        results.push(CaseResult {
            name: case.name.clone(),
            vm,
            stress,
            max_vm,
            max_vm_at: grid.cell_centre(max_c),
            p99_vm: p99,
            max_vm_away: max_away,
            max_vm_away_at: grid.cell_centre(away_c),
            max_disp,
            max_disp_at: grid.node_pos(max_n),
            cg_iterations: it,
            converged: ok,
            contact_nodes,
            contact_rounds: rounds,
            near: near_case,
        });
    }
    Ok(Analysis { cells_material: cells.len(), cells_dropped: dropped, fixed_nodes: fixed_nodes.len(), load_nodes: load_nodes.len(), cases: results, warnings })
}

fn solve3(a: &[[f64; 3]; 3], b: &P3) -> Option<P3> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let scale = a.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max).powi(3);
    if det.abs() <= 1e-9 * scale || !det.is_finite() {
        return None;
    }
    let col = |k: usize| {
        let mut m = *a;
        for (row, v) in m.iter_mut().zip(b) {
            if let Some(slot) = row.get_mut(k) {
                *slot = *v;
            }
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    Some([col(0) / det, col(1) / det, col(2) / det])
}

fn pcg(apply: &dyn Fn(&[f64], &mut [f64]), precondition: &dyn Fn(&[f64], &mut [f64]), f: &[f64], u: &mut [f64], tol: f64, max: usize) -> (usize, bool) {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let n = f.len();
    let mut r = vec![0.0; n];
    apply(u, &mut r);
    for (ri, fi) in r.iter_mut().zip(f) {
        *ri = fi - *ri;
    }
    let fnorm = dot(f, f).sqrt().max(f64::MIN_POSITIVE);
    let mut z = vec![0.0; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut q = vec![0.0; n];
    for it in 0..max {
        if dot(&r, &r).sqrt() <= tol * fnorm {
            return (it, true);
        }
        apply(&p, &mut q);
        let pq = dot(&p, &q);
        if !(pq.is_finite() && pq > 0.0) {
            return (it, false);
        }
        let alpha = rz / pq;
        for ((ui, pi), (ri, qi)) in u.iter_mut().zip(&p).zip(r.iter_mut().zip(&q)) {
            *ui += alpha * pi;
            *ri -= alpha * qi;
        }
        precondition(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for (pi, zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
    }
    (max, dot(&r, &r).sqrt() <= tol * fnorm)
}

/// Reads an STL file (binary or ASCII) into positions and triangles (three points per triangle).
pub fn read_stl(bytes: &[u8]) -> Result<(Vec<P3>, Vec<[u32; 3]>), String> {
    let mut positions = Vec::new();
    if bytes.len() >= 84 {
        let count = bytes.get(80..84).and_then(|s| <[u8; 4]>::try_from(s).ok()).map_or(0, u32::from_le_bytes) as usize;
        if count.checked_mul(50).and_then(|v| v.checked_add(84)) == Some(bytes.len()) {
            for t in 0..count {
                for v in 0..3 {
                    let at = 84 + 50 * t + 12 + 12 * v;
                    let f = |o: usize| bytes.get(at + o..at + o + 4).and_then(|s| <[u8; 4]>::try_from(s).ok()).map(f32::from_le_bytes).map(f64::from);
                    positions.push([f(0).ok_or("short STL")?, f(4).ok_or("short STL")?, f(8).ok_or("short STL")?]);
                }
            }
            let triangles = (0..count as u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            return Ok((positions, triangles));
        }
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "not a binary STL of the right size, and not ASCII".to_string())?;
    for line in text.lines() {
        let mut words = line.split_whitespace();
        if words.next() == Some("vertex") {
            let v: Vec<f64> = words.filter_map(|w| w.parse().ok()).collect();
            if let [x, y, z] = v.as_slice() {
                positions.push([*x, *y, *z]);
            } else {
                return Err("bad vertex line in ASCII STL".into());
            }
        }
    }
    if positions.is_empty() || positions.len() % 3 != 0 {
        return Err("no triangles in the STL".into());
    }
    let triangles = (0..(positions.len() / 3) as u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
    Ok((positions, triangles))
}

/// A heat map of the largest stress along the line of sight, looking along `axis`, as RGB rows
/// (top row first): image x = the next axis, image y = the one after, upwards.
pub fn projection(grid: &Grid, vm: &[f64], axis: usize, max: f64, scale: usize) -> (usize, usize, Vec<u8>) {
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (nu, nv) = (n3(&grid.n, u), n3(&grid.n, v));
    let (w, h) = (nu * scale, nv * scale);
    let mut rgb = vec![0u8; w * h * 3];
    for iu in 0..nu {
        for iv in 0..nv {
            let mut best: Option<f64> = None;
            for id in 0..n3(&grid.n, axis) {
                let mut ijk = [0usize; 3];
                for (ax, val) in [(axis, id), (u, iu), (v, iv)] {
                    if let Some(slot) = ijk.get_mut(ax) {
                        *slot = val;
                    }
                }
                let [i, j, k] = ijk;
                if let Some(s) = vm.get(grid.cell_index(i, j, k)).filter(|s| s.is_finite()) {
                    best = Some(best.map_or(*s, |b: f64| b.max(*s)));
                }
            }
            let colour = match best {
                None => [0, 0, 0],
                Some(s) => heat(if max > 0.0 { s / max } else { 0.0 }),
            };
            for dx in 0..scale {
                for dy in 0..scale {
                    let x = iu * scale + dx;
                    let y = (nv - 1 - iv) * scale + dy;
                    let at = 3 * (y * w + x);
                    if let Some(px) = rgb.get_mut(at..at + 3) {
                        px.copy_from_slice(&colour);
                    }
                }
            }
        }
    }
    (w, h, rgb)
}

/// Blue (low) through green and yellow to red (high).
pub fn heat(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let stops = [(0.0, [40.0, 60.0, 200.0]), (0.33, [40.0, 190.0, 90.0]), (0.66, [240.0, 220.0, 40.0]), (1.0, [230.0, 40.0, 30.0])];
    let mut out = [0u8; 3];
    for pair in stops.windows(2) {
        if let [(t0, c0), (t1, c1)] = pair
            && t >= *t0
            && t <= *t1
        {
            let f = (t - t0) / (t1 - t0);
            for (o, (a, b)) in out.iter_mut().zip(c0.iter().zip(c1)) {
                *o = (a + f * (b - a)).round() as u8;
            }
        }
    }
    out
}

/// An uncompressed (stored) PNG, enough for small pictures with no image library.
pub fn png(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
    fn crc32(data: &[u8]) -> u32 {
        let mut c = 0xffff_ffffu32;
        for b in data {
            c ^= u32::from(*b);
            for _ in 0..8 {
                c = if c & 1 == 1 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }
    let mut raw = Vec::with_capacity(h * (1 + 3 * w));
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(rgb.get(3 * w * y..3 * w * (y + 1)).unwrap_or(&[]));
    }
    // zlib stream of stored blocks.
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, block) in blocks.iter().enumerate() {
        z.push(u8::from(i + 1 == blocks.len()));
        let len = block.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for byte in &raw {
        a = (a + u32::from(*byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

/// A closed box mesh from `lo` to `hi`.
pub fn box_stl(lo: P3, hi: P3) -> (Vec<P3>, Vec<[u32; 3]>) {
    let [x0, y0, z0] = lo;
    let [x1, y1, z1] = hi;
    let p = vec![[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    let t = vec![[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]];
    (p, t)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn cylinder_and_union_regions() {
        let (lo, hi) = ([0.0; 3], [10.0; 3]);
        let screw = |z: f64| Region::Cylinder { point: [0.0, 0.0, z], dir: [1.0, 0.0, 0.0], r: 2.0, len: 5.0 };
        let both = Region::Union(vec![screw(3.0), screw(-3.0)]);
        assert!(inside(&both, &[4.0, 1.0, 3.5], &lo, &hi) && inside(&both, &[0.0, 0.0, -3.0], &lo, &hi));
        assert!(!inside(&both, &[6.0, 0.0, 3.0], &lo, &hi), "past the plate");
        assert!(!inside(&both, &[1.0, 0.0, 0.0], &lo, &hi), "between the screws");
    }

    /// Beam on a wall, one screw near the top, tip loaded down: the bottom presses on the wall, the top lifts
    /// off. With the screw in the middle, pulling straight off the wall leaves no contact, so it matches the
    /// screw-only hold exactly.
    #[test]
    fn wall_contact_pushes_only() {
        let (p, t) = box_stl([0.0, -5.0, -10.0], [40.0, 5.0, 10.0]);
        let (grid, solid) = voxelize(&p, &t, 1.0).unwrap();
        let screw = |z: f64| Region::Cylinder { point: [0.0, 0.0, z], dir: [1.0, 0.0, 0.0], r: 2.5, len: 3.0 };
        let glued = Region::Side { axis: 0, max: false, depth: 0.01 };
        let load = Region::Side { axis: 0, max: true, depth: 0.01 };
        let run = |fixed: &Region, contact: bool, force: P3| {
            let contact = contact.then_some(Contact { region: &glued, axis: 0, push: 1.0 });
            let setup = Setup { grid: &grid, solid: &solid, e_mpa: 1000.0, nu: 0.3, fixed, load: &load, cg_tol: 1e-9, cg_max: 20_000, exclude_layers: 2, cell_scale: None, multigrid: true, contact };
            let a = analyse(&setup, &[Case { name: "c".into(), force, moment: [0.0; 3] }], |_| {}).unwrap();
            assert!(a.cases[0].converged && a.warnings.is_empty(), "{:?}", a.warnings);
            a.cases.into_iter().next().unwrap()
        };
        let down = [0.0, 0.0, -20.0];
        let (free, wall, stuck) = (run(&screw(6.0), false, down), run(&screw(6.0), true, down), run(&glued, false, down));
        let face = 11 * 21; // wall-face nodes
        assert!(wall.contact_nodes > 10 && wall.contact_nodes < face / 2, "bottom part only: {}", wall.contact_nodes);
        assert!(wall.contact_rounds > 1);
        let count = |r: &CaseResult| r.near.iter().filter(|n| **n).count();
        assert!(count(&wall) > count(&free), "where the wall pushes is left out of the away peak");
        assert!(stuck.max_disp < wall.max_disp && wall.max_disp < 0.8 * free.max_disp, "{} {} {}", stuck.max_disp, wall.max_disp, free.max_disp);
        let pull = [20.0, 0.0, 0.0];
        let (free, wall) = (run(&screw(0.0), false, pull), run(&screw(0.0), true, pull));
        assert_eq!(wall.contact_nodes, 0);
        assert!((wall.max_disp - free.max_disp).abs() < 1e-6 * wall.max_disp);
    }

    #[test]
    fn brick_is_symmetric_with_six_rigid_modes() {
        let k = brick_stiffness(0.3);
        for (i, row) in k.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!((v - k[j][i]).abs() < 1e-12);
            }
        }
        let pos: Vec<[f64; 3]> = CORNERS.iter().map(|c| c.map(|v| v as f64)).collect();
        let modes: Vec<[f64; 24]> = vec![
            std::array::from_fn(|i| if i % 3 == 0 { 1.0 } else { 0.0 }),
            std::array::from_fn(|i| if i % 3 == 1 { 1.0 } else { 0.0 }),
            std::array::from_fn(|i| if i % 3 == 2 { 1.0 } else { 0.0 }),
            std::array::from_fn(|i| match i % 3 { 0 => -pos[i / 3][1], 1 => pos[i / 3][0], _ => 0.0 }),
            std::array::from_fn(|i| match i % 3 { 1 => -pos[i / 3][2], 2 => pos[i / 3][1], _ => 0.0 }),
            std::array::from_fn(|i| match i % 3 { 2 => -pos[i / 3][0], 0 => pos[i / 3][2], _ => 0.0 }),
        ];
        for m in &modes {
            for row in k.iter() {
                assert!(row.iter().zip(m).map(|(a, b)| a * b).sum::<f64>().abs() < 1e-12);
            }
        }
    }

    #[test]
    fn voxelize_fills_a_box() {
        let (p, t) = box_stl([0.0, 0.0, 0.0], [10.0, 6.0, 4.0]);
        let (grid, solid) = voxelize(&p, &t, 1.0).unwrap();
        assert_eq!(grid.n, [10, 6, 4]);
        assert_eq!(solid.iter().filter(|s| **s).count(), 240);
    }

    #[test]
    fn a_gap_in_the_mesh_does_not_flip_columns() {
        let (p, mut t) = box_stl([0.0, 0.0, 0.0], [10.0, 6.0, 4.0]);
        t.remove(2); // open half of the top face
        let (_, solid) = voxelize(&p, &t, 1.0).unwrap();
        assert_eq!(solid.iter().filter(|s| **s).count(), 240);
    }

    /// Cantilever 100 x 10 x 10 mm, E = 1000 MPa, 10 N down at the free end, 1 mm voxels.
    /// Beam theory: tip = F L^3 / (3 E I) + F L / (k G A) = 4.0 + 0.026 mm; root stress
    /// 6 F L / (b h^2) = 6 MPa. Linear bricks are too stiff in bending, so allow 10%.
    #[test]
    fn cantilever_matches_beam_theory() {
        let (p, t) = box_stl([0.0, 0.0, 0.0], [100.0, 10.0, 10.0]);
        let (grid, solid) = voxelize(&p, &t, 1.0).unwrap();
        let fixed = Region::Side { axis: 0, max: false, depth: 0.01 };
        let load = Region::Side { axis: 0, max: true, depth: 0.01 };
        let setup = Setup { grid: &grid, solid: &solid, e_mpa: 1000.0, nu: 0.3, fixed: &fixed, load: &load, cg_tol: 1e-8, cg_max: 20_000, exclude_layers: 2, cell_scale: None, multigrid: true, contact: None };
        let cases = [Case { name: "tip".into(), force: [0.0, 0.0, -10.0], moment: [0.0; 3] }];
        let a = analyse(&setup, &cases, |_| {}).unwrap();
        let r = &a.cases[0];
        assert!(r.converged);
        let i = 10.0 * 1000.0 / 12.0;
        let g = 1000.0 / 2.6;
        let tip = 10.0 * 1e6 / (3.0 * 1000.0 * i) + 10.0 * 100.0 / (5.0 / 6.0 * g * 100.0);
        assert!((0.9..=1.02).contains(&(r.max_disp / tip)), "tip {} vs {tip}", r.max_disp);
        assert!((0.8..=1.1).contains(&(r.p99_vm / 6.0)), "p99 {}", r.p99_vm);
    }

    /// A rod pulled along x at 1 MPa: printed with the layers stacked along x the pull goes straight across
    /// them (judged at 1 MPa × ratio); stacked along y it runs along the layers (plain von Mises).
    #[test]
    fn layers_across_the_pull_count_by_the_ratio() {
        assert!((across_layers(&[2.0, -5.0, 0.0, 0.0, 0.0, 0.0], 0) - 2.0).abs() < 1e-12 && across_layers(&[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0) == 0.0, "pressure counts nothing");
        assert!((across_layers(&[0.0, 0.0, 0.0, 3.0, 0.0, 4.0], 0) - 5.0).abs() < 1e-12 && across_layers(&[0.0, 0.0, 0.0, 3.0, 0.0, 4.0], 1) == 3.0);
        let (p, t) = box_stl([0.0, 0.0, 0.0], [40.0, 10.0, 10.0]);
        let (grid, solid) = voxelize(&p, &t, 1.0).unwrap();
        let fixed = Region::Side { axis: 0, max: false, depth: 0.01 };
        let load = Region::Side { axis: 0, max: true, depth: 0.01 };
        let setup = Setup { grid: &grid, solid: &solid, e_mpa: 1000.0, nu: 0.3, fixed: &fixed, load: &load, cg_tol: 1e-9, cg_max: 20_000, exclude_layers: 4, cell_scale: None, multigrid: true, contact: None };
        let a = analyse(&setup, &[Case { name: "pull".into(), force: [100.0, 0.0, 0.0], moment: [0.0; 3] }], |_| {}).unwrap();
        let mut across = a.cases.into_iter().next().unwrap();
        let mut along = CaseResult { vm: across.vm.clone(), near: across.near.clone(), stress: across.stress.clone(), name: "along".into(), ..across };
        let ratio = 47.0 / 30.0;
        across.with_layers(&grid, 0, ratio);
        along.with_layers(&grid, 1, ratio);
        let mid = grid.cell_index(20, 5, 5); // away from the clamped and the loaded end
        assert!((across.vm[mid] / ratio - 1.0).abs() < 0.05, "across {}", across.vm[mid]);
        assert!((along.vm[mid] - 1.0).abs() < 0.05, "along {}", along.vm[mid]);
        assert!(across.max_vm_away > along.max_vm_away * 1.4);
    }

    /// Multigrid and plain Jacobi give the same answer, multigrid in a fraction of the steps.
    #[test]
    fn multigrid_gives_the_same_answer_in_fewer_steps() {
        let (p, t) = box_stl([0.0, 0.0, 0.0], [60.0, 12.0, 12.0]);
        let (grid, solid) = voxelize(&p, &t, 1.0).unwrap();
        let fixed = Region::Side { axis: 0, max: false, depth: 0.01 };
        let load = Region::Side { axis: 0, max: true, depth: 0.01 };
        let cases = [Case { name: "tip".into(), force: [0.0, 0.0, -10.0], moment: [0.0; 3] }];
        let run = |multigrid: bool| {
            let setup = Setup { grid: &grid, solid: &solid, e_mpa: 1000.0, nu: 0.3, fixed: &fixed, load: &load, cg_tol: 1e-9, cg_max: 20_000, exclude_layers: 2, cell_scale: None, multigrid, contact: None };
            analyse(&setup, &cases, |_| {}).unwrap()
        };
        let (plain, mg) = (run(false), run(true));
        let (a, b) = (&plain.cases[0], &mg.cases[0]);
        assert!(a.converged && b.converged);
        assert!((a.max_disp - b.max_disp).abs() < 1e-6 * a.max_disp, "{} vs {}", a.max_disp, b.max_disp);
        assert!((a.p99_vm - b.p99_vm).abs() < 1e-5 * a.p99_vm, "{} vs {}", a.p99_vm, b.p99_vm);
        assert!(b.cg_iterations * 3 < a.cg_iterations, "jacobi {} steps, multigrid {}", a.cg_iterations, b.cg_iterations);
    }

    #[test]
    fn a_couple_load_solves() {
        let (p, t) = box_stl([0.0, 0.0, 0.0], [20.0, 4.0, 4.0]);
        let (grid, solid) = voxelize(&p, &t, 1.0).unwrap();
        let fixed = Region::Side { axis: 1, max: false, depth: 0.01 };
        let load = Region::Side { axis: 1, max: true, depth: 0.01 };
        let setup = Setup { grid: &grid, solid: &solid, e_mpa: 1000.0, nu: 0.3, fixed: &fixed, load: &load, cg_tol: 1e-6, cg_max: 5000, exclude_layers: 2, cell_scale: None, multigrid: true, contact: None };
        let cases = [Case { name: "twist".into(), force: [0.0; 3], moment: [0.0, 100.0, 0.0] }];
        assert!(analyse(&setup, &cases, |_| {}).is_ok());
    }

    #[test]
    fn png_is_well_formed() {
        let img = png(2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(&img[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(&img[img.len() - 8..img.len() - 4], b"IEND");
    }

    #[test]
    fn stl_round_trip_binary() {
        let mut bytes = vec![0u8; 80];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 12]);
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in v {
                bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&[0, 0]);
        let (p, t) = read_stl(&bytes).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(t, vec![[0, 1, 2]]);
        assert!(read_stl(b"garbage").is_err());
    }
}
