//! 3D "take out what is not needed": start from a solid pyramid (frustum) on the wall and repeatedly
//! remove the voxels that carry the least stress (hard-kill ESO), with the real 3D solver in stress3d.
//!
//! eso3d [key=value ...]   (all lengths mm)
//!   reach=150 base_y=80 base_z=100 tip=20   frustum: wall face x=0, tip face x=reach, tip x tip
//!   voxel=2.5  force=50 side=15            tip load: force N down (-Z) and side N along Y
//!   print=x|y|z [z_mpa=30]                 3D printed with the layers stacked along this axis (x: wall plate on the bed,
//!                                          y: on its side, z: upright): a cell counts the larger of von Mises and the
//!                                          across-layer demand x strength / z_mpa (z_mpa: across-layer strength, from the
//!                                          data sheet; known for the printed library plastics). The bars report also
//!                                          compares all three orientations (safety factor, bars that need support).
//!   creep=1 creep_strength=K creep_modulus=M  one more case, "constant": the down force alone (no knock, no side), as it
//!                                          hangs for years: its stress counts / K (long-term strength = K x short-term)
//!                                          and its tip sag x 1/M (long-term modulus = M x short-term). Both from the
//!                                          filament's creep data; there are no defaults (no open data was found)
//!   loads=split knock=1                    split: two cases, down alone and side alone (+Y); real: down and side together,
//!                                          the side load both ways (two cases); knock multiplies every force (a margin for knocks)
//!   back=6 pad=8                           kept solid: wall plate thickness (screws), load pad depth
//!   min_member=5                           no member thinner than this (opening after every round; 0 = off)
//!   holes=y:z,y:z hole=4.5 sink=8.4        countersunk screw holes (diameters, 90 deg countersink) through the wall plate, in the smooth STL
//!   from=DIR blur=2                        only re-smooth DIR/mask.bin (no ESO run); blur passes
//!   target=0.12 step=0.04 sf=4             stop at this share of the frustum, remove 4 % per round, keep SF
//!   material=petg-printed out=DIR
//!   check=FILE.stl [check_voxel=1.25]      only check an STL (same supports, loads, material) on a finer grid, heat maps in out/
//!   bars=1 [blend=3 bar_res=0.75 rounds=4] every pillar of the design (also with from=DIR) becomes a round bar of the same
//!                                          volume between its two ends; checked on check_voxel, bars below sf grow
//!                                          (at most `rounds` times); fillets of `blend` mm, mesh grid bar_res mm
//!   taper=1                                size every point along a bar on its own (thick roots, thin middles), smoothed
//!   size=1                                 bars also get thinner where SF > 1.3 x sf (not below min_member); keeps the lightest round that passes
//!   plate=frame [strip=12] hold=screws     wall plate as pads round screws and bar roots joined by strips; held only at the
//!                                          screws (default with a frame; hold=wall glues the whole wall face as before;
//!                                          hold=contact: screws, and the wall face pushes back where the plate presses)
//! check and bars also write CASE-stress.ply (colour per point, for Blender) and CASE-stress-{iso,side,top}.png.
//! Stops at the target share or just before the safety factor (away from the load/support zones) drops
//! below sf, and writes DIR/bracket.stl (voxel faces, wall at x=0, Z up), report.json and heat maps.

#![allow(clippy::indexing_slicing, clippy::cast_precision_loss)] // command-line tool: indices come from the grid itself

use generative_design_engine::library::Library;
use generative_design_engine::stress3d::{Analysis, Case, Contact, Grid, Region, Setup, analyse, heat, png, printed_stress, projection, read_stl, voxelize};
use std::collections::HashMap;
use std::time::Instant;

fn stl(grid: &Grid, solid: &[bool]) -> Vec<u8> {
    let at = |i: i64, j: i64, k: i64| {
        let ok = i >= 0 && j >= 0 && k >= 0 && (i as usize) < grid.n[0] && (j as usize) < grid.n[1] && (k as usize) < grid.n[2];
        ok && solid[grid.cell_index(i as usize, j as usize, k as usize)]
    };
    // (axis, side, other two axes in the order that winds outward)
    let faces = [(0, 1, 1, 2), (0, -1, 2, 1), (1, 1, 2, 0), (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 1, 0)];
    let mut tris: Vec<[[f32; 3]; 4]> = Vec::new(); // normal + 3 corners
    for c in (0..grid.cells()).filter(|c| solid[*c]) {
        let [i, j, k] = grid.cell_ijk(c);
        let ijk = [i as i64, j as i64, k as i64];
        for (ax, side, a, b) in faces {
            let mut n = ijk;
            n[ax] += side;
            if at(n[0], n[1], n[2]) {
                continue;
            }
            let base = |du: usize, dv: usize| {
                let mut p = [ijk[0] as f64, ijk[1] as f64, ijk[2] as f64];
                p[ax] += if side > 0 { 1.0 } else { 0.0 };
                p[a] += du as f64;
                p[b] += dv as f64;
                [0, 1, 2].map(|d| (grid.origin[d] + p[d] * grid.h) as f32)
            };
            let mut nor = [0.0f32; 3];
            nor[ax] = side as f32;
            let (p00, p10, p11, p01) = (base(0, 0), base(1, 0), base(1, 1), base(0, 1));
            tris.push([nor, p00, p10, p11]);
            tris.push([nor, p00, p11, p01]);
        }
    }
    let mut out = vec![0u8; 80];
    out.extend((tris.len() as u32).to_le_bytes());
    for t in &tris {
        for v in t {
            for x in v {
                out.extend(x.to_le_bytes());
            }
        }
        out.extend([0u8; 2]);
    }
    out
}


/// The same volume with a round form: the arms (the voxels that are not kept solid) are blurred and their
/// surface cut where the whole shape gives back exactly the voxel volume; the kept wall plate and tip pad stay exact boxes (marching tetrahedra on a grid twice as fine). The wall
/// face (x = 0) and the tip face stay flat: the design is continued one voxel past them, and that skin is
/// pressed flat. Returns triangles, the level used and the volume of the result in mm3.
/// The user's standard screw (the default): countersunk, hole 4.5 mm, countersink 8.4 mm at 90 degrees.
const HOLE_R: f64 = 2.25;
const SINK_R: f64 = 4.2;

/// Signed distance (approximate inside the cone, exact on the cylinder) to the screw holes through the wall
/// plate (x = 0 to xb, countersink on the front face x = xb); negative inside a hole. Beyond x = xb + lim
/// there is no hole, so the cone never cuts an arm.
fn hole_dist(p: [f64; 3], holes: &[[f64; 2]], [hr, sr]: [f64; 2], xb: f64, lim: f64) -> f64 {
    if p[0] > xb + lim {
        return f64::INFINITY;
    }
    holes.iter().map(|[y, z]| {
        let rho = ((p[1] - y).powi(2) + (p[2] - z).powi(2)).sqrt();
        (rho - hr).min((rho - sr + xb - p[0]) / std::f64::consts::SQRT_2)
    }).fold(f64::INFINITY, f64::min)
}

/// Volume of one countersunk hole through a plate xb thick.
fn hole_volume(xb: f64, [hr, sr]: [f64; 2]) -> f64 {
    let hc = sr - hr; // 90 degrees: as deep as the radius grows
    std::f64::consts::PI * (hr * hr * (xb - hc) + hc / 3.0 * (sr * sr + sr * hr + hr * hr))
}

fn smooth(grid: &Grid, solid: &[bool], keep: &[bool], passes: usize, holes: &[[f64; 2]], r: [f64; 2], xb: f64) -> (Vec<[[f64; 3]; 3]>, f64, f64) {
    const M: usize = 4; // empty margin in voxels
    let [nx, ny, nz] = grid.n;
    let d = [nx + 2 * M, ny + 2 * M, nz + 2 * M];
    let at = |i: usize, j: usize, k: usize| d[1] * d[2] * i + d[2] * j + k;
    // x: one voxel of the design continues past each end, then nothing
    let fill = |on: &dyn Fn(usize) -> bool| {
        let mut v = vec![0.0f64; d[0] * d[1] * d[2]];
        for i in 0..d[0] {
            let si = i as i64 - M as i64;
            if si < -1 || si > nx as i64 {
                continue;
            }
            let si = si.clamp(0, nx as i64 - 1) as usize;
            for j in M..M + ny {
                for k in M..M + nz {
                    v[at(i, j, k)] = f64::from(u8::from(on(grid.cell_index(si, j - M, k - M))));
                }
            }
        }
        v
    };
    // Where an arm meets the plate or the pad, the kept cells next to it join the blur, so the arm keeps its
    // full thickness into the plate (without this it fades out one voxel short and the part falls in two).
    let arm = |c: usize| solid[c] && !keep[c];
    let root: Vec<bool> = (0..grid.cells())
        .map(|c| {
            let [i, j, k] = grid.cell_ijk(c);
            keep[c]
                && (0..27).any(|q| {
                    let (a, b, e) = (i as i64 + q / 9 - 1, j as i64 + (q / 3) % 3 - 1, k as i64 + q % 3 - 1);
                    a >= 0 && b >= 0 && e >= 0 && (a as usize) < nx && (b as usize) < ny && (e as usize) < nz && arm(grid.cell_index(a as usize, b as usize, e as usize))
                })
        })
        .collect();
    let mut f = fill(&|c| arm(c) || root[c]);
    let kp = fill(&|c| keep[c]);
    for _ in 0..passes {
        for ax in 0..3 {
            let src = f.clone();
            let stride = [d[1] * d[2], d[2], 1][ax];
            for (idx, v) in f.iter_mut().enumerate() {
                let pos = (idx / stride) % d[ax];
                let lo = if pos > 0 { src[idx - stride] } else { src[idx] };
                let hi = if pos + 1 < d[ax] { src[idx + stride] } else { src[idx] };
                *v = (lo + src[idx] + hi) / 3.0;
            }
        }
    }
    let hf = grid.h / 2.0;
    let fd = [0, 1, 2].map(|a| 2 * (grid.n[a] + 2 * (M - 1)) + 1);
    let o = [0, 1, 2].map(|a| grid.origin[a] - (M - 1) as f64 * grid.h);
    let sample = |arr: &[f64], a: usize, b: usize, c: usize| {
        let w = [a, b, c].map(|v| v as f64 * 0.5 + 0.5); // coarse index space: fine node 0 sits half a cell inside the array
        let i0 = [0, 1, 2].map(|q| (w[q].floor() as usize).min(d[q] - 2));
        let t = [0, 1, 2].map(|q| w[q] - i0[q] as f64);
        let mut acc = 0.0;
        for corner in 0..8 {
            let (dx, dy, dz) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let wt = [dx, dy, dz].iter().zip(&t).fold(1.0, |acc, (bit, tt)| acc * if *bit == 1 { *tt } else { 1.0 - tt });
            acc += wt * arr[at(i0[0] + dx, i0[1] + dy, i0[2] + dz)];
        }
        acc
    };
    let nidx = |a: usize, b: usize, c: usize| (a * fd[1] + b) * fd[2] + c;
    // fa: the blurred arms; fp: the kept plate and pad, 0.5 exactly on their faces
    let (mut fa, mut fp) = (vec![0.0f64; fd[0] * fd[1] * fd[2]], vec![0.0f64; fd[0] * fd[1] * fd[2]]);
    for a in 0..fd[0] {
        for b in 0..fd[1] {
            for c in 0..fd[2] {
                fa[nidx(a, b, c)] = sample(&f, a, b, c);
                fp[nidx(a, b, c)] = sample(&kp, a, b, c);
            }
        }
    }
    let pos = |a: usize, b: usize, c: usize| [o[0] + a as f64 * hf, o[1] + b as f64 * hf, o[2] + c as f64 * hf];
    let mut hd = vec![f64::INFINITY; fa.len()];
    if !holes.is_empty() {
        for a in 0..fd[0] {
            for b in 0..fd[1] {
                for c in 0..fd[2] {
                    hd[nidx(a, b, c)] = hole_dist(pos(a, b, c), holes, r, xb, hf);
                }
            }
        }
    }
    let reach = nx as f64 * grid.h;
    let want = solid.iter().filter(|v| **v).count() as f64 * grid.h.powi(3) - holes.len() as f64 * hole_volume(xb, r);
    let inside_nodes = |thr: f64| {
        let mut n = 0usize;
        for a in 0..fd[0] {
            let x = o[0] + a as f64 * hf;
            if x < 0.0 || x > reach {
                continue;
            }
            // the layers on the wall and tip faces count half: a node there owns half a cell
            let r = a * fd[1] * fd[2]..(a + 1) * fd[1] * fd[2];
            let layer = fa[r.clone()].iter().zip(&fp[r.clone()]).zip(&hd[r]).filter(|((av, pv), d)| (**av >= thr || **pv >= 0.5) && **d >= 0.0).count();
            n += if x < 0.5 * hf || x > reach - 0.5 * hf { layer } else { 2 * layer };
        }
        0.5 * n as f64 * hf.powi(3)
    };
    // Counting nodes only estimates the volume of the cut surface, so the target is corrected by the real volume a few times.
    let build = |want: f64| {
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if inside_nodes(mid) > want { lo = mid } else { hi = mid }
    }
    let thr = 0.5 * (lo + hi);
    // The plate field is bent so that its 0.5 face lands exactly on the level.
    // The holes: linear in their distance with the level on the hole wall, so the cut lands on the true circle.
    let field: Vec<f64> = fa.iter().zip(&fp).zip(&hd).map(|((av, pv), d)| av.max(if *pv >= 0.5 { thr + (pv - 0.5) * 2.0 * (1.0 - thr) } else { pv * 2.0 * thr }).min(thr + d / hf)).collect();
    let mut tris = march(&field, fd, &pos, thr);
    for p in tris.iter_mut().flatten() {
        p[0] = p[0].clamp(0.0, reach);
    }
    finish(&mut tris, holes, r, xb);
    let vol = volume(&tris);
    (tris, thr, vol)
    };
    let (mut target, mut res) = (want, build(want));
    for _ in 0..3 {
        target *= want / res.2;
        res = build(target);
    }
    res
}

/// Marching tetrahedra over a node field on a regular grid (`fd` nodes per axis, node position from `pos`):
/// the surface where the field crosses `thr`, wound outward (inside = field >= thr).
fn march(field: &[f64], fd: [usize; 3], pos: &dyn Fn(usize, usize, usize) -> [f64; 3], thr: f64) -> Vec<[[f64; 3]; 3]> {
    let nidx = |a: usize, b: usize, c: usize| (a * fd[1] + b) * fd[2] + c;
    // Six tetrahedra per cell around the diagonal 0-7.
    let tets = [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]];
    let mut tris = Vec::new();
    for a in 0..fd[0] - 1 {
        for b in 0..fd[1] - 1 {
            for c in 0..fd[2] - 1 {
                let corner: [(usize, usize, usize); 8] = std::array::from_fn(|q| (a + (q & 1), b + ((q >> 1) & 1), c + ((q >> 2) & 1)));
                for tet in tets {
                    let v: [(usize, usize, usize); 4] = tet.map(|q| corner[q]);
                    let inn: Vec<bool> = v.iter().map(|p| field[nidx(p.0, p.1, p.2)] >= thr).collect();
                    let k = inn.iter().filter(|x| **x).count();
                    if k == 0 || k == 4 {
                        continue;
                    }
                    // The crossing on the edge between two nodes; the same two nodes always give the same point.
                    let edge = |p: (usize, usize, usize), q: (usize, usize, usize)| {
                        let (p, q) = if nidx(p.0, p.1, p.2) < nidx(q.0, q.1, q.2) { (p, q) } else { (q, p) };
                        let (fp, fq) = (field[nidx(p.0, p.1, p.2)], field[nidx(q.0, q.1, q.2)]);
                        let t = (fp - thr) / (fp - fq);
                        let (pp, pq) = (pos(p.0, p.1, p.2), pos(q.0, q.1, q.2));
                        [0, 1, 2].map(|x| pp[x] + t * (pq[x] - pp[x]))
                    };
                    let ins: Vec<usize> = (0..4).filter(|q| inn[*q]).collect();
                    let out: Vec<usize> = (0..4).filter(|q| !inn[*q]).collect();
                    let centroid = |ix: &[usize]| {
                        let mut c3 = [0.0; 3];
                        for q in ix {
                            let p = pos(v[*q].0, v[*q].1, v[*q].2);
                            for x in 0..3 {
                                c3[x] += p[x] / ix.len() as f64;
                            }
                        }
                        c3
                    };
                    let dir = { let (ci, co) = (centroid(&ins), centroid(&out)); [0, 1, 2].map(|x| co[x] - ci[x]) };
                    let polys: Vec<[[f64; 3]; 3]> = match k {
                        1 | 3 => {
                            let (one, rest) = if k == 1 { (ins[0], &out) } else { (out[0], &ins) };
                            vec![[edge(v[one], v[rest[0]]), edge(v[one], v[rest[1]]), edge(v[one], v[rest[2]])]]
                        }
                        _ => {
                            let (e0, e1, e2, e3) = (edge(v[ins[0]], v[out[0]]), edge(v[ins[0]], v[out[1]]), edge(v[ins[1]], v[out[1]]), edge(v[ins[1]], v[out[0]]));
                            vec![[e0, e1, e2], [e0, e2, e3]]
                        }
                    };
                    for mut t in polys {
                        let (u, w) = ([0, 1, 2].map(|x| t[1][x] - t[0][x]), [0, 1, 2].map(|x| t[2][x] - t[0][x]));
                        let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
                        if n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] < 0.0 {
                            t.swap(1, 2);
                        }
                        tris.push(t);
                    }
                }
            }
        }
    }
    tris
}

/// Puts corners that fell inside a screw hole back on its wall and drops triangles pressed flat to nothing.
fn finish(tris: &mut Vec<[[f64; 3]; 3]>, holes: &[[f64; 2]], r: [f64; 2], xb: f64) {
    // Linear crossings fall a little inside a round hole (about 0.1 mm here): put those corners on the hole wall.
    for p in tris.iter_mut().flatten().filter(|p| p[0] <= xb + 1e-9) {
        for [y, z] in holes {
            let (dy, dz) = (p[1] - y, p[2] - z);
            let (rho, want) = (dy.hypot(dz), r[0].max(r[1] - (xb - p[0])));
            if rho > 1e-9 && rho < want {
                (p[1], p[2]) = (y + dy * want / rho, z + dz * want / rho);
            }
        }
    }
    // Triangles pressed flat to nothing (the skin past the wall and tip faces) are dropped.
    tris.retain(|t| {
        let (u, w) = ([0, 1, 2].map(|x| t[1][x] - t[0][x]), [0, 1, 2].map(|x| t[2][x] - t[0][x]));
        let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        n.iter().map(|v| v * v).sum::<f64>() > 1e-12
    });
}

fn volume(tris: &[[[f64; 3]; 3]]) -> f64 {
    tris.iter().map(|t| (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])) / 6.0).sum::<f64>()
}

// ---- Round bars: every pillar of the voxel design becomes a round bar of the same volume ----

/// The 26 neighbour offsets.
fn n26() -> impl Iterator<Item = [i64; 3]> {
    (0..27i64).filter(|q| *q != 13).map(|q| [q / 9 - 1, (q / 3) % 3 - 1, q % 3 - 1])
}

fn step(grid: &Grid, c: usize, d: [i64; 3]) -> Option<usize> {
    let ijk = grid.cell_ijk(c);
    let mut o = [0usize; 3];
    for a in 0..3 {
        let v = ijk[a] as i64 + d[a];
        if v < 0 || v as usize >= grid.n[a] {
            return None;
        }
        o[a] = v as usize;
    }
    Some(grid.cell_index(o[0], o[1], o[2]))
}

/// Removing the centre of this 3x3x3 block (index 9a + 3b + c) changes no topology: the solid around it is one
/// 26-connected piece and the empty space next to it one 6-connected piece (Bertrand and Malandain's test).
fn simple(nb: &[bool; 27]) -> bool {
    let d = |q: usize| [q / 9, (q / 3) % 3, q % 3].map(|v| v as i64 - 1);
    let far = |q: usize| d(q).iter().map(|v| v.abs()).sum::<i64>();
    let gap = |q: usize, r: usize| { let (a, b) = (d(q), d(r)); [0, 1, 2].map(|i| (a[i] - b[i]).abs()) };
    // pieces of `member` (cells joined by `touch`) that hold at least one cell where `counts`
    let pieces = |member: &dyn Fn(usize) -> bool, touch: &dyn Fn(usize, usize) -> bool, counts: &dyn Fn(usize) -> bool| {
        let (mut seen, mut n) = ([false; 27], 0);
        for s in 0..27 {
            if !member(s) || seen[s] {
                continue;
            }
            seen[s] = true;
            let (mut stack, mut hit) = (vec![s], false);
            while let Some(q) = stack.pop() {
                hit |= counts(q);
                let next: Vec<usize> = (0..27).filter(|r| member(*r) && !seen[*r] && touch(q, *r)).collect();
                for r in next {
                    seen[r] = true;
                    stack.push(r);
                }
            }
            n += usize::from(hit);
        }
        n
    };
    pieces(&|q| q != 13 && nb[q], &|q, r| gap(q, r).iter().all(|v| *v <= 1), &|_| true) == 1
        && pieces(&|q| q != 13 && !nb[q] && far(q) <= 2, &|q, r| gap(q, r).iter().sum::<i64>() == 1, &|q| far(q) == 1) == 1
}

/// Thins the arms (solid, not kept) to lines, one layer from each of the six sides in turn, taking only voxels
/// whose removal changes no topology. No line ends are kept, so what stays is what holds the kept plate and pad
/// together (the load paths and the loops between them); loose ends shrink away.
fn skeleton(grid: &Grid, solid: &[bool], keep: &[bool]) -> Vec<bool> {
    let mut s = solid.to_vec();
    loop {
        let mut changed = false;
        for dir in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] {
            let border: Vec<usize> = (0..grid.cells()).filter(|c| s[*c] && !keep[*c] && !step(grid, *c, dir).is_some_and(|n| s[n])).collect();
            for c in border {
                let nb: [bool; 27] = std::array::from_fn(|q| {
                    let q = q as i64;
                    q == 13 || step(grid, c, [q / 9 - 1, (q / 3) % 3 - 1, q % 3 - 1]).is_some_and(|n| s[n])
                });
                if simple(&nb) {
                    s[c] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            return s;
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Distance from p to the segment a-b.
fn seg_dist(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let (ab, ap) = (sub(b, a), sub(p, a));
    let t = if dot(ab, ab) > 0.0 { (dot(ap, ab) / dot(ab, ab)).clamp(0.0, 1.0) } else { 0.0 };
    let q = sub(ap, ab.map(|v| v * t));
    dot(q, q).sqrt()
}

/// Douglas-Peucker: the fewest points that stay within `tol` of the line.
fn simplify(p: &[[f64; 3]], tol: f64) -> Vec<[f64; 3]> {
    let (Some(a), Some(b)) = (p.first(), p.last()) else { return Vec::new() };
    let worst = (1..p.len().saturating_sub(1)).map(|i| (i, seg_dist(p[i], *a, *b))).max_by(|x, y| x.1.total_cmp(&y.1));
    match worst {
        Some((k, d)) if d > tol => {
            let mut left = simplify(&p[..=k], tol);
            left.pop();
            left.extend(simplify(&p[k..], tol));
            left
        }
        _ if p.len() == 1 => vec![*a],
        _ => vec![*a, *b],
    }
}

/// Chaikin corner cutting, `passes` times: every bend becomes a smooth arc, the two ends stay where they are.
fn round_bends(p: &[[f64; 3]], passes: usize) -> Vec<[f64; 3]> {
    let mut p = p.to_vec();
    for _ in 0..passes {
        if p.len() < 3 {
            break;
        }
        let mut q = vec![p[0]];
        for w in p.windows(2) {
            let at = |t: f64| [0, 1, 2].map(|k| w[0][k] + t * (w[1][k] - w[0][k]));
            q.extend([at(0.25), at(0.75)]);
        }
        // the cut next to each end would move the end: keep the true ends instead
        q.remove(1);
        q.pop();
        q.extend(p.last().copied());
        p = q;
    }
    p
}

/// One round bar: straight pieces through `pts`, radius `r`, carrying the volume `vol` (mm3) of its pillar.
#[derive(Clone, Debug)]
struct Bar {
    pts: Vec<[f64; 3]>,
    vol: f64,
    /// radius at every point of `pts` (all the same until the bar is tapered)
    r: Vec<f64>,
}

/// Signed distance to a cone with round ends: sphere r1 at a, sphere r2 at b, their common tangent cone
/// between (Inigo Quilez' round cone). One sphere inside the other: the bigger sphere.
fn round_cone(p: [f64; 3], a: [f64; 3], b: [f64; 3], r1: f64, r2: f64) -> f64 {
    let ba = sub(b, a);
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let pa = sub(p, a);
    if a2 <= 1e-12 {
        let pb = sub(p, b);
        return (dot(pa, pa).sqrt() - r1).min(dot(pb, pb).sqrt() - r2);
    }
    let y = dot(pa, ba);
    let z = y - l2;
    let q = sub(pa.map(|v| v * l2), ba.map(|v| v * y));
    let x2 = dot(q, q);
    let (y2, z2) = (y * y * l2, z * z * l2);
    let k = rr.signum() * rr * rr * x2;
    if z.signum() * a2 * z2 > k {
        return (x2 + z2).sqrt() / l2 - r2;
    }
    if y.signum() * a2 * y2 < k {
        return (x2 + y2).sqrt() / l2 - r1;
    }
    ((x2 * a2 / l2).sqrt() + y * rr) / l2 - r1
}

impl Bar {
    fn length(&self) -> f64 {
        self.pts.windows(2).map(|w| dot(sub(w[1], w[0]), sub(w[1], w[0])).sqrt()).sum()
    }
    /// Signed distance to the bar's surface, and the point of `pts` nearest to where it is measured.
    fn near(&self, p: [f64; 3]) -> (f64, usize) {
        let mut best = (f64::INFINITY, 0);
        for i in 0..self.pts.len().saturating_sub(1) {
            let d = round_cone(p, self.pts[i], self.pts[i + 1], self.r[i], self.r[i + 1]);
            if d < best.0 {
                let ab = sub(self.pts[i + 1], self.pts[i]);
                let t = dot(sub(p, self.pts[i]), ab) / dot(ab, ab).max(1e-12);
                best = (d, if t < 0.5 { i } else { i + 1 });
            }
        }
        if self.pts.len() == 1 {
            let q = sub(p, self.pts[0]);
            best = (dot(q, q).sqrt() - self.r[0], 0);
        }
        best
    }
    fn rmax(&self) -> f64 {
        self.r.iter().copied().fold(0.0, f64::max)
    }
    /// Volume of the cone pieces (overlaps at the bends counted twice, a little).
    fn volume(&self) -> f64 {
        (0..self.pts.len().saturating_sub(1)).map(|i| {
            let l = dot(sub(self.pts[i + 1], self.pts[i]), sub(self.pts[i + 1], self.pts[i])).sqrt();
            std::f64::consts::PI * l * (self.r[i] * self.r[i] + self.r[i] * self.r[i + 1] + self.r[i + 1] * self.r[i + 1]) / 3.0
        }).sum()
    }
}

/// New radii from the stress at every point: thicker where SF < sf, thinner (two_way) where SF > 1.3 sf, never
/// below r_min; `taper`: every point on its own, then smoothed so the bar has no bulges and no cone steeper than
/// half its length; else the whole bar by its worst point. Points with no stress measured take their neighbour's.
fn resize(bar: &Bar, peak: &[f64], sf: f64, strength: f64, two_way: bool, taper: bool, r_min: f64) -> Vec<f64> {
    let factor = |pk: f64| {
        let s_now = strength / pk.max(1e-12);
        // bending: stress goes with 1/r^3; a thinner bar aims a little above the target SF
        if s_now < sf {
            (sf * pk / strength).cbrt().clamp(1.05, 1.5)
        } else if two_way && s_now > 1.3 * sf {
            (1.1 * sf * pk / strength).cbrt().clamp(0.7, 0.95)
        } else {
            1.0
        }
    };
    let n = bar.r.len();
    let mut pk = peak.to_vec();
    if pk.iter().all(|v| *v <= 0.0) {
        return bar.r.clone();
    }
    for (i, v) in pk.iter_mut().enumerate() {
        if *v <= 0.0 {
            let from = (0..n).filter(|j| peak[*j] > 0.0).min_by_key(|j| j.abs_diff(i)).unwrap_or(i);
            *v = peak[from];
        }
    }
    if !taper {
        let f = factor(pk.iter().copied().fold(0.0, f64::max));
        return bar.r.iter().map(|r| (r * f).max(r_min)).collect();
    }
    let own: Vec<f64> = bar.r.iter().zip(&pk).map(|(r, p)| (r * factor(*p)).max(r_min)).collect();
    let mut r = own.clone();
    // smoothing only fills dips: no point ends up thinner than its own sizing asks
    for _ in 0..3 {
        let o = r.clone();
        for i in 0..n {
            let (a, b) = (o[i.saturating_sub(1)], o[(i + 1).min(n - 1)]);
            r[i] = (0.25 * a + 0.5 * o[i] + 0.25 * b).max(own[i]);
        }
    }
    // no cone steeper than half its length, both ways
    for i in 1..n {
        let l = dot(sub(bar.pts[i], bar.pts[i - 1]), sub(bar.pts[i], bar.pts[i - 1])).sqrt();
        r[i] = r[i].max(r[i - 1] - 0.5 * l);
    }
    for i in (0..n.saturating_sub(1)).rev() {
        let l = dot(sub(bar.pts[i + 1], bar.pts[i]), sub(bar.pts[i + 1], bar.pts[i])).sqrt();
        r[i] = r[i].max(r[i + 1] - 0.5 * l);
    }
    r
}

/// Cuts the arms of the voxel design into pillars: the thinned lines between junctions (and the kept plate and
/// pad); every arm voxel belongs to the line it reaches first through the material, a junction's voxels are
/// shared by its pillars. Each pillar becomes a bar of the same volume along its line, ends that touch the
/// plate or pad run on into it. Returns the bars and the arm volume that belongs to no pillar.
fn bars(grid: &Grid, solid: &[bool], keep: &[bool]) -> (Vec<Bar>, f64) {
    let skel = skeleton(grid, solid, keep);
    let line: Vec<bool> = (0..grid.cells()).map(|c| skel[c] && !keep[c]).collect();
    let around = |c: usize, f: &dyn Fn(usize) -> bool| n26().filter_map(|d| step(grid, c, d)).filter(|n| f(*n)).collect::<Vec<usize>>();
    let node: Vec<bool> = (0..grid.cells()).map(|c| line[c] && (around(c, &|n| line[n]).len() != 2 || !around(c, &|n| keep[n]).is_empty())).collect();
    // junction voxels that touch are one junction: its centre, and where it runs into the plate or pad
    let mut node_of = vec![usize::MAX; grid.cells()];
    let mut nodes: Vec<([f64; 3], Option<[f64; 3]>)> = Vec::new();
    for c in 0..grid.cells() {
        if !node[c] || node_of[c] != usize::MAX {
            continue;
        }
        let id = nodes.len();
        node_of[c] = id;
        let (mut stack, mut members) = (vec![c], Vec::new());
        while let Some(q) = stack.pop() {
            members.push(q);
            for n in around(q, &|n| node[n] && node_of[n] == usize::MAX) {
                node_of[n] = id;
                stack.push(n);
            }
        }
        let mean = |cells: &[usize]| {
            let mut m = [0.0; 3];
            for q in cells {
                let p = grid.cell_centre(*q);
                for a in 0..3 {
                    m[a] += p[a] / cells.len() as f64;
                }
            }
            m
        };
        let mut touch: Vec<usize> = members.iter().flat_map(|q| around(*q, &|n| keep[n])).collect();
        touch.sort_unstable();
        touch.dedup();
        nodes.push((mean(&members), (!touch.is_empty()).then(|| mean(&touch))));
    }
    // the lines between junctions
    let mut edge_of = vec![usize::MAX; grid.cells()];
    let mut edges: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for c in (0..grid.cells()).filter(|c| node[*c]) {
        for first in around(c, &|n| line[n] && !node[n]) {
            if edge_of[first] != usize::MAX {
                continue;
            }
            let (mut prev, mut cur, mut chain, mut end) = (c, first, Vec::new(), None);
            loop {
                if node[cur] {
                    end = Some(node_of[cur]);
                    break;
                }
                if edge_of[cur] != usize::MAX {
                    break; // ran into its own start: a loop with no junction
                }
                edge_of[cur] = edges.len();
                chain.push(cur);
                let Some(next) = around(cur, &|n| line[n] && n != prev).first().copied() else { break };
                (prev, cur) = (cur, next);
            }
            let a = node_of[c];
            edges.push((a, end.unwrap_or(a), chain));
        }
    }
    // every arm voxel to the nearest line voxel through the material
    let e = edges.len();
    let mut owner = vec![usize::MAX; grid.cells()];
    let mut queue = std::collections::VecDeque::new();
    for c in (0..grid.cells()).filter(|c| line[*c]) {
        owner[c] = if node[c] { e + node_of[c] } else { edge_of[c] };
        queue.push_back(c);
    }
    while let Some(c) = queue.pop_front() {
        for n in around(c, &|n| solid[n] && !keep[n] && owner[n] == usize::MAX) {
            owner[n] = owner[c];
            queue.push_back(n);
        }
    }
    let mut count = vec![0.0f64; e + nodes.len()];
    let mut lost = 0.0;
    for c in (0..grid.cells()).filter(|c| solid[*c] && !keep[*c]) {
        match count.get_mut(owner[c]) {
            Some(v) => *v += 1.0,
            None => lost += 1.0,
        }
    }
    let mut degree = vec![0.0f64; nodes.len()];
    for (a, b, _) in &edges {
        degree[*a] += 1.0;
        degree[*b] += 1.0;
    }
    for k in 0..nodes.len() {
        if degree[k] == 0.0 {
            lost += count[e + k];
        }
    }
    let cell = grid.h.powi(3);
    let out = edges.iter().enumerate().map(|(i, (a, b, chain))| {
        let mut pts = Vec::new();
        pts.extend(nodes[*a].1);
        pts.push(nodes[*a].0);
        pts.extend(chain.iter().map(|c| grid.cell_centre(*c)));
        pts.push(nodes[*b].0);
        pts.extend(nodes[*b].1);
        let mut bar = Bar { pts: round_bends(&simplify(&pts, grid.h), 3), vol: cell * (count[i] + count[e + a] / degree[*a] + count[e + b] / degree[*b]), r: Vec::new() };
        bar.r = vec![(bar.vol / (std::f64::consts::PI * bar.length().max(grid.h))).sqrt(); bar.pts.len()];
        bar
    }).collect();
    (out, lost * cell)
}

/// The frame plate as flat capsules (y:z end points, radius); empty when `strip` <= 0.
fn plate_frame(bars: &[Bar], holes: &[[f64; 2]], r: [f64; 2], xb: f64, strip: f64) -> Vec<([f64; 2], [f64; 2], f64)> {
    let mut caps = Vec::new();
    if strip <= 0.0 {
        return caps;
    }
    let mut screws = holes.to_vec();
    screws.sort_by(|a, b| a[1].total_cmp(&b[1]));
    for w in screws.windows(2) {
        caps.push((w[0], w[1], 0.5 * strip));
    }
    for sc in &screws {
        caps.push((*sc, *sc, r[1] + 4.0)); // round the countersink, room for the head
    }
    for b in bars {
        for (end, re) in [(b.pts.first(), b.r.first()), (b.pts.last(), b.r.last())].into_iter().filter_map(|(e, r)| Some((e?, *r?))).filter(|(e, _)| e[0] < xb) {
            let root = [end[1], end[2]];
            caps.push((root, root, re + 3.0));
            let near = screws.iter().min_by(|a, c| (a[0] - root[0]).hypot(a[1] - root[1]).total_cmp(&(c[0] - root[0]).hypot(c[1] - root[1])));
            if let Some(sc) = near {
                caps.push((root, *sc, 0.5 * strip));
            }
        }
    }
    caps
}

/// Smooth minimum: the union of two shapes with a round fillet of about `k` mm where they meet.
fn smin(a: f64, b: f64, k: f64) -> f64 {
    let t = if k > 0.0 { (k - (a - b).abs()).max(0.0) / k } else { 0.0 };
    a.min(b) - t * t * k / 4.0
}

/// The bar design as a closed mesh: the bars, the kept plate and pad as boxes around their voxels, joined with
/// fillets of `blend` mm, wall and tip faces flat, screw holes cut; marching tetrahedra on a `res` mm grid.
#[allow(clippy::too_many_arguments)]
/// `strip` > 0: the wall plate is a frame instead of the full box: pads round the screws and the bar roots,
/// strips `strip` mm wide from every bar root to its nearest screw and from screw to screw, `xb` thick.
fn bar_mesh(grid: &Grid, keep: &[bool], bars: &[Bar], blend: f64, res: f64, holes: &[[f64; 2]], r: [f64; 2], xb: f64, strip: f64) -> Vec<[[f64; 3]; 3]> {
    let reach = grid.n[0] as f64 * grid.h;
    let boxes: Vec<([f64; 3], [f64; 3])> = [false, true].iter().filter_map(|tip| {
        let cells: Vec<usize> = (0..grid.cells()).filter(|c| keep[*c] && (grid.cell_centre(*c)[0] > 0.5 * reach) == *tip).collect();
        let lo = [0, 1, 2].map(|a| cells.iter().map(|c| grid.cell_centre(*c)[a] - 0.5 * grid.h).fold(f64::INFINITY, f64::min));
        let hi = [0, 1, 2].map(|a| cells.iter().map(|c| grid.cell_centre(*c)[a] + 0.5 * grid.h).fold(f64::NEG_INFINITY, f64::max));
        (!cells.is_empty() && (*tip || strip <= 0.0)).then_some((lo, hi))
    }).collect();
    let frame = plate_frame(bars, holes, r, xb, strip);
    let reach_box: Vec<([f64; 3], [f64; 3])> = bars.iter().map(|b| {
        ([0, 1, 2].map(|a| b.pts.iter().map(|q| q[a]).fold(f64::INFINITY, f64::min)), [0, 1, 2].map(|a| b.pts.iter().map(|q| q[a]).fold(f64::NEG_INFINITY, f64::max)))
    }).collect();
    let sdf = |p: [f64; 3]| {
        let mut d = f64::INFINITY;
        for (lo, hi) in &boxes {
            let q: [f64; 3] = [0, 1, 2].map(|a| (p[a] - 0.5 * (lo[a] + hi[a])).abs() - 0.5 * (hi[a] - lo[a]));
            let out = q.map(|v| v.max(0.0));
            d = smin(d, dot(out, out).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0), blend);
        }
        for (a, b, rad) in &frame {
            let flat = seg_dist([0.0, p[1], p[2]], [0.0, a[0], a[1]], [0.0, b[0], b[1]]) - rad;
            // small round on the frame's edges, as on the hole rims
            d = smin(d, -smin(-flat, p[0].min(xb - p[0]), 1.0), blend); // inside the strip and between the faces
        }
        for (b, (blo, bhi)) in bars.iter().zip(&reach_box) {
            // a bar whose box is further than the fillet reaches cannot change d
            let out: [f64; 3] = [0, 1, 2].map(|a| (blo[a] - p[a]).max(p[a] - bhi[a]).max(0.0));
            if dot(out, out).sqrt() - b.rmax() > d + blend {
                continue;
            }
            d = smin(d, b.near(p).0, blend);
        }
        // hole edges get a small round (smooth cut): a sharp rim comes out of the mesher as a saw edge
        -smin(-d.max(-p[0]).max(p[0] - reach), hole_dist(p, holes, r, xb, res), 1.0)
    };
    // room round the design for the thickest bar and the frame's widest part (pads and strips reach past the voxels)
    let reach_out = frame.iter().map(|c| c.2).fold(bars.iter().map(Bar::rmax).fold(0.0, f64::max), f64::max);
    let pad = reach_out + blend + 2.0 * res;
    // half a step off in x, so no node lies on the wall or tip face (the crossing then lands exactly on it)
    let lo = [-1.5 * res, grid.origin[1] - pad, grid.origin[2] - pad];
    let hi = [reach + 1.5 * res, grid.origin[1] + grid.n[1] as f64 * grid.h + pad, grid.origin[2] + grid.n[2] as f64 * grid.h + pad];
    let fd = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / res).ceil() as usize + 1);
    let pos = |a: usize, b: usize, c: usize| [lo[0] + a as f64 * res, lo[1] + b as f64 * res, lo[2] + c as f64 * res];
    let mut field = vec![0.0; fd[0] * fd[1] * fd[2]];
    for a in 0..fd[0] {
        for b in 0..fd[1] {
            for c in 0..fd[2] {
                // A node on (or almost on) the surface gives slivers and flat triangles, which viewers shade
                // as black specks: move it off the surface by a tenth of a step (at most 0.1 res of shape change).
                let v = -sdf(pos(a, b, c));
                field[(a * fd[1] + b) * fd[2] + c] = if v.abs() >= 0.1 * res { v } else if v >= 0.0 { 0.1 * res } else { -0.1 * res };
            }
        }
    }
    let mut tris = march(&field, fd, &pos, 0.0);
    finish(&mut tris, holes, r, xb);
    tris
}

// ---- Stress colours: the mesh coloured by the stress in the material under it ----

/// One point per corner position (the mesher gives a shared edge the same point both times).
fn index(tris: &[[[f64; 3]; 3]]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut at: HashMap<[u64; 3], u32> = HashMap::new();
    let mut pos = Vec::new();
    let tri = tris.iter().map(|t| t.map(|p| *at.entry(p.map(f64::to_bits)).or_insert_with(|| { pos.push(p); pos.len() as u32 - 1 }))).collect();
    (pos, tri)
}

/// The stress at a surface point: the mean of the material cells around it (the nearest one further in if
/// none touches it); 0 if there is no material near.
fn stress_at(g: &Grid, vm: &[f64], p: [f64; 3]) -> f64 {
    let f = [0, 1, 2].map(|a| (p[a] - g.origin[a]) / g.h);
    let pick = |base: [i64; 3], span: i64| {
        let mut v = Vec::new();
        for q in 0..span.pow(3) {
            let ijk = [base[0] + q / (span * span), base[1] + (q / span) % span, base[2] + q % span];
            if (0..3).all(|a| ijk[a] >= 0 && (ijk[a] as usize) < g.n[a]) {
                let s = vm[g.cell_index(ijk[0] as usize, ijk[1] as usize, ijk[2] as usize)];
                if s.is_finite() {
                    v.push(s);
                }
            }
        }
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    pick(f.map(|v| (v - 0.5).floor() as i64), 2).or_else(|| pick(f.map(|v| v.floor() as i64 - 1), 3)).unwrap_or(0.0)
}

/// Binary PLY with a colour per point (Blender: Import PLY, then show the "Col" colour attribute).
fn ply(pos: &[[f64; 3]], tri: &[[u32; 3]], rgb: &[[u8; 3]]) -> Vec<u8> {
    let mut out = format!("ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n", pos.len(), tri.len()).into_bytes();
    for (p, c) in pos.iter().zip(rgb) {
        for v in p {
            out.extend((*v as f32).to_le_bytes());
        }
        out.extend(c);
    }
    for t in tri {
        out.push(3);
        for v in t {
            out.extend(v.to_le_bytes());
        }
    }
    out
}

/// A picture of the coloured mesh looking along `look` (orthographic, lit from the eye, z-buffered), with the
/// colour scale (blue = no stress, red = the peak away from load and supports) along the bottom.
fn render(pos: &[[f64; 3]], tri: &[[u32; 3]], rgb: &[[u8; 3]], look: [f64; 3], w: usize, h: usize) -> Vec<u8> {
    let unit = |v: [f64; 3]| { let l = dot(v, v).sqrt(); v.map(|x| x / l) };
    let f = unit(look);
    let right = unit(cross(f, if f[2].abs() > 0.99 { [0.0, 1.0, 0.0] } else { [0.0, 0.0, 1.0] }));
    let up = cross(right, f);
    let s: Vec<[f64; 3]> = pos.iter().map(|p| [dot(*p, right), dot(*p, up), dot(*p, f)]).collect();
    let (bar, m) = (16usize, 20.0);
    let lo = [0, 1].map(|a| s.iter().map(|v| v[a]).fold(f64::INFINITY, f64::min));
    let hi = [0, 1].map(|a| s.iter().map(|v| v[a]).fold(f64::NEG_INFINITY, f64::max));
    let scale = ((w as f64 - 2.0 * m) / (hi[0] - lo[0])).min((h as f64 - 3.0 * m - bar as f64) / (hi[1] - lo[1]));
    let px = |v: [f64; 3]| [m + (v[0] - lo[0]) * scale, h as f64 - 2.0 * m - bar as f64 - (v[1] - lo[1]) * scale, v[2]];
    let mut img = vec![16u8; w * h * 3];
    let mut depth = vec![f64::INFINITY; w * h];
    for t in tri {
        let ix = t.map(|i| i as usize);
        let [a, b, c] = ix.map(|i| px(s[i]));
        let shade = 0.3 + 0.7 * dot(unit(cross(sub(pos[ix[1]], pos[ix[0]]), sub(pos[ix[2]], pos[ix[0]]))), f).abs();
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if !shade.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
        let x1 = (a[0].max(b[0]).max(c[0]).ceil() as usize).min(w - 1);
        let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
        let y1 = (a[1].max(b[1]).max(c[1]).ceil() as usize).min(h - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (qx, qy) = (x as f64 + 0.5, y as f64 + 0.5);
                let wb = ((qx - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (qy - a[1])) / area;
                let wc = ((b[0] - a[0]) * (qy - a[1]) - (qx - a[0]) * (b[1] - a[1])) / area;
                let wa = 1.0 - wb - wc;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                let z = wa * a[2] + wb * b[2] + wc * c[2];
                if z >= depth[y * w + x] {
                    continue;
                }
                depth[y * w + x] = z;
                for k in 0..3 {
                    let col = wa * f64::from(rgb[ix[0]][k]) + wb * f64::from(rgb[ix[1]][k]) + wc * f64::from(rgb[ix[2]][k]);
                    img[3 * (y * w + x) + k] = (col * shade).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    for y in h - m as usize - bar..h - m as usize {
        for x in m as usize..w - m as usize {
            let c = heat((x as f64 - m) / (w as f64 - 2.0 * m));
            img[3 * (y * w + x)..3 * (y * w + x) + 3].copy_from_slice(&c);
        }
    }
    img
}

/// Writes CASE-stress.ply and CASE-stress-{iso,side,top}.png for every case; colour = stress / that case's peak
/// away from the load and supports (so blue to red, red = the peak; the load and support spots may go past it).
fn stress_pictures(out: &str, pos: &[[f64; 3]], tri: &[[u32; 3]], g: &Grid, an: &Analysis) -> Result<String, String> {
    let mut legend = String::new();
    for c in &an.cases {
        let rgb: Vec<[u8; 3]> = pos.iter().map(|p| heat(stress_at(g, &c.vm, *p) / c.max_vm_away.max(1e-12))).collect();
        std::fs::write(format!("{out}/{}-stress.ply", c.name), ply(pos, tri, &rgb)).map_err(|e| e.to_string())?;
        for (view, look) in [("iso", [-1.0, 0.7, -0.6]), ("side", [0.0, 1.0, 0.0]), ("top", [0.0, 0.0, -1.0])] {
            std::fs::write(format!("{out}/{}-stress-{view}.png", c.name), png(1200, 900, &render(pos, tri, &rgb, look, 1200, 900))).map_err(|e| e.to_string())?;
        }
        legend += &format!("{}: blue 0 MPa -> green -> yellow -> red {:.2} MPa (peak away from load and supports)\n", c.name, c.max_vm_away);
    }
    Ok(legend)
}

fn tris_stl(tris: &[[[f64; 3]; 3]]) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out.extend((tris.len() as u32).to_le_bytes());
    for t in tris {
        let (u, w) = ([0, 1, 2].map(|x| t[1][x] - t[0][x]), [0, 1, 2].map(|x| t[2][x] - t[0][x]));
        let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        let len = n.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        for v in [n.map(|x| x / len), t[0], t[1], t[2]] {
            for x in v {
                out.extend((x as f32).to_le_bytes());
            }
        }
        out.extend([0u8; 2]);
    }
    out
}

/// Opening with a k x k x k block: what no such block fits into (members thinner than k voxels) goes.
fn open(grid: &Grid, solid: &[bool], k: usize) -> Vec<bool> {
    let [nx, ny, nz] = grid.n;
    let block = |c: usize, back: bool, test: &dyn Fn(usize) -> bool, all: bool| {
        let [i, j, l] = grid.cell_ijk(c);
        let mut hit = false; // all: every cell must pass; any: one must
        for a in 0..k {
            for b in 0..k {
                for d in 0..k {
                    let off = |v: usize, e: usize, n: usize| if back { v.checked_sub(e).filter(|x| *x < n) } else { Some(v + e).filter(|x| *x < n) };
                    let ok = match (off(i, a, nx), off(j, b, ny), off(l, d, nz)) {
                        (Some(x), Some(y), Some(z)) => test(grid.cell_index(x, y, z)),
                        _ => false,
                    };
                    if all && !ok {
                        return false;
                    }
                    if !all && ok {
                        hit = true;
                    }
                }
            }
        }
        all || hit
    };
    let eroded: Vec<bool> = (0..grid.cells()).map(|c| block(c, false, &|n| solid[n], true)).collect();
    (0..grid.cells()).map(|c| block(c, true, &|n| eroded[n], false)).collect()
}

/// Screw tension (N) per hole for the down and the side case: the plate is taken as rigid and tips about
/// the plate edge that presses on the wall (bottom edge for a down load, the side edge the load points to
/// for a side load); each screw pulls in proportion to its distance from that edge. Load at the tip face.
fn screw_tension(holes: &[[f64; 2]], half_y: f64, half_z: f64, reach: f64, down: f64, side: f64) -> Vec<[f64; 2]> {
    let share = |d: &[f64], m: f64| {
        let s: f64 = d.iter().map(|v| v * v).sum();
        d.iter().map(|v| if s > 0.0 { m * v / s } else { f64::INFINITY }).collect::<Vec<f64>>()
    };
    let dz: Vec<f64> = holes.iter().map(|h| h[1] + half_z).collect();
    let dy: Vec<f64> = holes.iter().map(|h| (half_y * side.signum() - h[0]).abs()).collect();
    let (t_down, t_side) = (share(&dz, down.abs() * reach), share(&dy, side.abs() * reach));
    t_down.iter().zip(&t_side).map(|(a, b)| [*a, *b]).collect()
}

/// Across-layer (Z) tensile strength of the printed library plastics, MPa, from the same data sheets as the
/// library (the library cards hold the in-layer values).
fn across_strength(id: &str) -> Option<f64> {
    Some(match id {
        "pla-printed" => 38.0,
        "petg-printed" => 30.0,
        "abs-printed" => 32.0,
        "asa-printed" => 31.0,
        "paht-cf-printed" => 47.0,
        "pa12-sls" => 42.0,
        _ => return None,
    })
}

fn creep_line(ks: f64, km: f64) -> String {
    format!("Creep: the \"constant\" case is the down force alone, as it hangs for years; its stress is shown / {ks} (long-term strength = {ks} x short-term) and its tip sag / {km} (long-term modulus = {km} x short-term); both factors given by hand\n")
}

const AXES: [&str; 3] = ["x (wall plate on the bed)", "y (on its side)", "z (upright, as mounted)"];

fn print_line(print: Option<usize>, strength: f64, z_mpa: f64) -> String {
    match print {
        Some(a) => format!("Printed: layers stacked along {}; {strength} MPa along the layers, {z_mpa} MPa across them (the stress shown and sized is the larger of von Mises and the across-layer demand x {:.2})", AXES[a], strength / z_mpa),
        None => "Material taken as equally strong in every direction (print= not set)".into(),
    }
}

/// Bar length (mm) that overhangs by more than 45 degrees when the layers stack along `axis`: the parts of the
/// bar that run within 45 degrees of the bed need support (or bridge).
fn overhang_mm(bar: &Bar, axis: usize) -> f64 {
    bar.pts.windows(2).map(|w| {
        let d = sub(w[1], w[0]);
        let l = dot(d, d).sqrt();
        if l > 0.0 && d[axis].abs() < l * std::f64::consts::FRAC_1_SQRT_2 { l } else { 0.0 }
    }).sum()
}

/// One solve serves every print direction (the material model is the same in all directions): the safety
/// factor if the layers stacked along x, y or z, and the bars that would need support.
fn orientations(an: &Analysis, bars: &[Bar], strength: f64, ratio: f64, chosen: Option<usize>) -> String {
    let mut t = String::from("Print orientation (same design, same solve; peak away from load and supports):\n");
    for (axis, name) in AXES.iter().enumerate() {
        let peak = an.cases.iter().flat_map(|c| c.stress.iter().zip(&c.near).filter(|(st, near)| st[0].is_finite() && !**near).map(|(st, _)| printed_stress(st, axis, ratio))).fold(0.0, f64::max);
        let over: Vec<(usize, f64)> = bars.iter().enumerate().map(|(i, b)| (i, overhang_mm(b, axis))).filter(|(_, l)| *l > 1.0).collect();
        let list = over.iter().map(|(i, l)| format!("{i} ({l:.0} mm)")).collect::<Vec<_>>().join(", ");
        t += &format!("  layers along {name}: peak {peak:.2} MPa, SF {:.1}; {} of {} bars need support{}{}\n", strength / peak.max(1e-12), over.len(), bars.len(), if list.is_empty() { String::new() } else { format!(": {list}") }, if chosen == Some(axis) { "  <- this run" } else { "" });
    }
    t
}

/// The tip load cases. split: down alone and side alone (+Y), as runs 1-10. real: down and side together, the
/// side load both ways; down alone is left out because (linear) von Mises of a+b or of a-b is never below that
/// of a. knock scales every force.
fn load_cases(kind: &str, force: f64, side: f64, knock: f64) -> Result<Vec<Case>, String> {
    if !(1.0..=10.0).contains(&knock) || !force.is_finite() || !side.is_finite() || force < 0.0 || side < 0.0 {
        return Err("check force, side (0 or more) and knock (1-10)".into());
    }
    let (d, s) = (-force * knock, side * knock);
    let case = |name: &str, f: [f64; 3]| Case { name: name.into(), force: f, moment: [0.0; 3] };
    match kind {
        "split" => Ok(vec![case("down", [0.0, 0.0, d]), case("side", [0.0, s, 0.0])]),
        "real" => Ok(vec![case("down+side", [0.0, s, d]), case("down-side", [0.0, -s, d])]),
        _ => Err(format!("loads={kind}: use split or real")),
    }
}

/// "50 N down / 15 N side" style summary of the cases, for the reports.
fn case_list(cases: &[Case]) -> String {
    cases.iter().map(|c| format!("{} ({:.0} N down, {:+.0} N side)", c.name, -c.force[2], c.force[1])).collect::<Vec<_>>().join(", ")
}

/// "y:z,y:z" (mm on the wall face) -> hole centres; empty -> none.
fn parse_holes(s: &str) -> Result<Vec<[f64; 2]>, String> {
    s.split(',').filter(|t| !t.is_empty()).map(|t| {
        let (y, z) = t.split_once(':').ok_or(format!("holes: {t}: use y:z"))?;
        let n = |v: &str| v.parse::<f64>().ok().filter(|x| x.is_finite()).ok_or(format!("holes: {t}: numbers"));
        Ok([n(y)?, n(z)?])
    }).collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("eso3d: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut o: HashMap<String, String> = HashMap::new();
    for a in std::env::args().skip(1) {
        let (k, v) = a.split_once('=').ok_or(format!("{a}: use key=value"))?;
        o.insert(k.into(), v.into());
    }
    let num = |k: &str, d: f64| -> Result<f64, String> { o.get(k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("{k}: a number"))) };
    let (reach, by, bz, tip, h) = (num("reach", 150.0)?, num("base_y", 80.0)?, num("base_z", 100.0)?, num("tip", 20.0)?, num("voxel", 2.5)?);
    let (force, side, back, pad) = (num("force", 50.0)?, num("side", 15.0)?, num("back", 6.0)?, num("pad", 8.0)?);
    let (target, step, sf) = (num("target", 0.12)?, num("step", 0.04)?, num("sf", 4.0)?);
    let kmin = (num("min_member", 5.0)? / h).round().max(1.0) as usize;
    if !(0.5..=10.0).contains(&h) || !(0.0..1.0).contains(&target) || !(0.005..=0.3).contains(&step) || tip > by.min(bz) {
        return Err("check voxel (0.5-10), target (0-1), step (0.005-0.3), tip <= base".into());
    }
    let out = o.get("out").cloned().unwrap_or_else(|| "eso3d-out".into());
    let m = Library::builtin()?.find(o.get("material").map_or("petg-printed", String::as_str)).ok_or("material not in the library")?.clone();
    let s = m.strengths();
    let (e_gpa, nu) = (s.young_modulus_gpa.ok_or("no stiffness")?, s.poisson_ratio.ok_or("no Poisson ratio")?);
    let strength = s.yield_mpa.or(s.tensile_mpa).ok_or("no strength")?;

    let n = [(reach / h).round() as usize, (by / h).round() as usize, (bz / h).round() as usize];
    let grid = Grid { origin: [0.0, -0.5 * n[1] as f64 * h, -0.5 * n[2] as f64 * h], h, n };
    let mut solid = vec![false; grid.cells()];
    let mut keep = vec![false; grid.cells()];
    for c in 0..grid.cells() {
        let p = grid.cell_centre(c);
        let t = p[0] / (n[0] as f64 * h); // 0 at the wall, 1 at the tip
        let (hy, hz) = ((by + (tip - by) * t) / 2.0, (bz + (tip - bz) * t) / 2.0);
        solid[c] = p[1].abs() <= hy && p[2].abs() <= hz;
        keep[c] = solid[c] && (p[0] < back || p[0] > n[0] as f64 * h - pad);
    }
    let start = solid.iter().filter(|v| **v).count();
    let cell_cm3 = h.powi(3) / 1000.0;
    eprintln!("frustum {}x{}x{} voxels of {h} mm: {start} cells = {:.0} cm3, kept solid {}", n[0], n[1], n[2], start as f64 * cell_cm3, keep.iter().filter(|v| **v).count());

    let fixed = Region::Side { axis: 0, max: false, depth: 0.5 * h };
    let load = Region::Side { axis: 0, max: true, depth: 0.5 * h };
    let mut cases = load_cases(o.get("loads").map_or("split", String::as_str), force, side, num("knock", 1.0)?)?;
    let creep = o.get("creep").is_some_and(|v| v != "0");
    // shortcut: no built-in creep data (the open PETG creep papers could not be read, 21 Nov 2026); add defaults per material once sourced
    let (ks, km) = if creep {
        let need = |k: &str| if o.contains_key(k) { num(k, 0.0) } else { Err(format!("creep=1 needs {k}= (0-1) from the creep data of {}", m.id)) };
        (need("creep_strength")?, need("creep_modulus")?)
    } else {
        (1.0, 1.0)
    };
    if !(ks > 0.0 && ks <= 1.0 && km > 0.0 && km <= 1.0) {
        return Err("creep_strength and creep_modulus: above 0, at most 1".into());
    }
    if creep {
        cases.push(Case { name: "constant".into(), force: [0.0, 0.0, -force], moment: [0.0; 3] });
    }
    let print = match o.get("print").map(String::as_str) {
        None | Some("none") => None,
        Some(a) => Some(["x", "y", "z"].iter().position(|v| *v == a).ok_or(format!("print={a}: use x, y or z"))?),
    };
    let z_mpa = match o.get("z_mpa") {
        Some(_) => num("z_mpa", 0.0)?,
        None => across_strength(&m.id).unwrap_or(f64::NAN),
    };
    if (print.is_some() || o.contains_key("z_mpa")) && !(z_mpa > 0.0 && z_mpa <= strength) {
        return Err(format!("z_mpa: the across-layer strength of {} (above 0, at most {strength} MPa); give it from the data sheet", m.id));
    }
    let ratio = strength / z_mpa;
    // every solve is judged the same way: printed layers, then the constant case on its long-term values
    let layered = |mut an: Analysis, g: &Grid| {
        if let Some(ax) = print {
            an.cases.iter_mut().for_each(|c| c.with_layers(g, ax, ratio));
        }
        for c in an.cases.iter_mut().filter(|c| c.name == "constant") {
            c.vm.iter_mut().for_each(|v| *v /= ks);
            (c.max_vm, c.max_vm_away, c.p99_vm, c.max_disp) = (c.max_vm / ks, c.max_vm_away / ks, c.p99_vm / ks, c.max_disp / km);
        }
        an
    };
    let cv = num("check_voxel", 1.25)?;
    let wall_face = Region::Side { axis: 0, max: false, depth: 0.5 * cv };
    // wall = the wall face pushes back where the plate presses on it (lifts off where it pulls away)
    let analyse_mesh = |pos: &[[f64; 3]], tri: &[[u32; 3]], fixed: &Region, wall: bool| -> Result<(Grid, Vec<bool>, Analysis), String> {
        let (g, sol) = voxelize(pos, tri, cv)?;
        let setup = Setup { grid: &g, solid: &sol, e_mpa: e_gpa * 1000.0, nu, fixed, load: &Region::Side { axis: 0, max: true, depth: 0.5 * cv }, cg_tol: 1e-8, cg_max: 50_000, exclude_layers: 2, cell_scale: None, multigrid: true, contact: wall.then_some(Contact { region: &wall_face, axis: 0, push: 1.0 }) };
        let an = layered(analyse(&setup, &cases, |m| if m.contains("wall contact") { eprintln!("  {m}") })?, &g);
        Ok((g, sol, an))
    };
    if let Some(file) = o.get("check") {
        println!("loads at the tip: {}", case_list(&cases));
        println!("{}", print_line(print, strength, z_mpa));
        if creep {
            print!("{}", creep_line(ks, km));
        }
        let (pos, tri) = read_stl(&std::fs::read(file).map_err(|e| format!("{file}: {e}"))?)?;
        let (g, sol, fin) = analyse_mesh(&pos, &tri, &fixed, false)?;
        let cells = sol.iter().filter(|v| **v).count();
        eprintln!("{file}: {}x{}x{} voxels of {cv} mm, {cells} cells = {:.1} cm3", g.n[0], g.n[1], g.n[2], cells as f64 * cv.powi(3) / 1000.0);
        {
            // how much of the material is joined to the wall (cells touching by a corner count as joined)
            let mut seen = vec![false; g.cells()];
            let mut stack: Vec<usize> = (0..g.cells()).filter(|c| sol[*c] && g.cell_ijk(*c)[0] == 0).collect();
            for c in &stack {
                seen[*c] = true;
            }
            let (mut reached, mut tip) = (0usize, 0usize);
            while let Some(c) = stack.pop() {
                reached += 1;
                let [i, j, k] = g.cell_ijk(c);
                if i + 1 == g.n[0] {
                    tip += 1;
                }
                for (di, dj, dk) in (0..27).map(|q| (q / 9 - 1i64, (q / 3) % 3 - 1i64, q % 3 - 1i64)) {
                    let (a, b, d) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                    if a < 0 || b < 0 || d < 0 || a as usize >= g.n[0] || b as usize >= g.n[1] || d as usize >= g.n[2] {
                        continue;
                    }
                    let n = g.cell_index(a as usize, b as usize, d as usize);
                    if sol[n] && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
            eprintln!("joined to the wall: {reached} of {cells} cells, {tip} cells of the tip face");
        }
        std::fs::create_dir_all(&out).map_err(|e| format!("{out}: {e}"))?;
        for c in &fin.cases {
            for (ax, name) in [(0, "x"), (1, "y"), (2, "z")] {
                let (w, hh, rgb) = projection(&g, &c.vm, ax, c.max_vm_away, 4);
                std::fs::write(format!("{out}/{}-along-{name}.png", c.name), png(w, hh, &rgb)).map_err(|e| e.to_string())?;
            }
            println!("{}: peak {:.2} MPa, away from load/support {:.2} MPa (SF {:.1} on {strength} MPa), p99 {:.2}, tip moves {:.3} mm, CG {} converged {}", c.name, c.max_vm, c.max_vm_away, strength / c.max_vm_away.max(1e-12), c.p99_vm, c.max_disp, c.cg_iterations, c.converged);
        }
        for w in &fin.warnings {
            println!("warning: {w}");
        }
        print!("{}", stress_pictures(&out, &pos, &tri, &g, &fin)?);
        return Ok(());
    }
    let t0 = Instant::now();
    let mut prev = solid.clone(); // the design of the round before, for the fall-back
    let mut round = 0;
    let mut last_count = usize::MAX;
    let from = o.get("from").cloned();
    let reason = if let Some(dir) = &from {
        let bytes = std::fs::read(format!("{dir}/mask.bin")).map_err(|e| format!("{dir}/mask.bin: {e}"))?;
        if bytes.len() != grid.cells() {
            return Err("mask.bin does not fit this grid (same reach/base/tip/voxel needed)".into());
        }
        solid = bytes.iter().map(|b| *b != 0).collect();
        "smoothing only".to_string()
    } else { loop {
        let setup = Setup { grid: &grid, solid: &solid, e_mpa: e_gpa * 1000.0, nu, fixed: &fixed, load: &load, cg_tol: 1e-6, cg_max: 20_000, exclude_layers: 2, cell_scale: None, multigrid: true, contact: None };
        let r = layered(analyse(&setup, &cases, |_| {})?, &grid);
        let away = r.cases.iter().map(|c| c.max_vm_away).fold(0.0, f64::max);
        let count = r.cases[0].vm.iter().filter(|v| !v.is_nan()).count();
        let found = if away > 0.0 { strength / away } else { f64::INFINITY };
        let disp = r.cases.iter().map(|c| c.max_disp).fold(0.0, f64::max);
        eprintln!("round {round:>3}: {:.1} cm3 ({:.0} %), away peak {away:.2} MPa (SF {found:.1}), tip {disp:.3} mm, CG {}, {:.0} s", count as f64 * cell_cm3, 100.0 * count as f64 / start as f64, r.cases[0].cg_iterations, t0.elapsed().as_secs_f64());
        if found < sf && round > 0 {
            solid = prev;
            break format!("stopped: this round had SF {found:.1} < {sf}, kept the one before");
        }
        // What the solver kept (loose voxels are NaN) is the new design; this round's result is its analysis.
        for (c, v) in solid.iter_mut().enumerate() {
            *v = !r.cases[0].vm[c].is_nan();
        }
        if round > 0 && count >= last_count {
            break "no progress: every member is at the minimum size".to_string();
        }
        last_count = count;
        if count as f64 / start as f64 <= target {
            break format!("reached the target share {target}");
        }
        let score: Vec<f64> = (0..grid.cells()).map(|c| r.cases.iter().map(|k| k.vm[c]).filter(|v| !v.is_nan()).fold(0.0, f64::max)).collect();
        let mut cand: Vec<usize> = (0..grid.cells()).filter(|c| solid[*c] && !keep[*c]).collect();
        cand.sort_by(|a, b| score[*a].total_cmp(&score[*b]));
        let remove = ((count as f64 * step).ceil() as usize).min(((count as f64 - target * start as f64).max(1.0)) as usize).min(cand.len());
        if remove == 0 {
            break "nothing left to remove".into();
        }
        prev = solid.clone();
        for c in &cand[..remove] {
            solid[*c] = false;
        }
        if kmin > 1 {
            solid = open(&grid, &solid, kmin);
            for (v, kp) in solid.iter_mut().zip(&keep) {
                *v |= *kp;
            }
        }
        round += 1;
        if round > 200 {
            break "round limit".into();
        }
    } };
    let count = solid.iter().filter(|v| **v).count();
    std::fs::create_dir_all(&out).map_err(|e| format!("{out}: {e}"))?;
    std::fs::write(format!("{out}/bracket.stl"), stl(&grid, &solid)).map_err(|e| e.to_string())?;
    std::fs::write(format!("{out}/mask.bin"), solid.iter().map(|v| u8::from(*v)).collect::<Vec<u8>>()).map_err(|e| e.to_string())?;
    // the kept plate is whole voxels: every column whose centre lies within `back`
    let xb = (back / h - 0.5).ceil().max(0.0) * h;
    let holes = parse_holes(o.get("holes").map_or("", String::as_str))?;
    let (hr, sr) = (num("hole", 2.0 * HOLE_R)? / 2.0, num("sink", 2.0 * SINK_R)? / 2.0);
    if !(hr > 0.5 && sr > hr) {
        return Err("hole (mm) above 1 and sink (countersink, mm) above hole".into());
    }
    for [y, z] in &holes {
        if xb < sr - hr + 1.0 || y.abs() + sr > 0.5 * by || z.abs() + sr > 0.5 * bz {
            return Err(format!("hole {y}:{z}: needs a plate of at least {:.2} mm (it is {xb}) and the countersink inside the plate", sr - hr + 1.0));
        }
        // the screwdriver goes in along x: no arm within the countersink radius plus 2 mm
        let blocked = (0..grid.cells()).filter(|c| solid[*c] && !keep[*c]).any(|c| {
            let p = grid.cell_centre(c);
            (p[1] - y).hypot(p[2] - z) < sr + 2.0 + 0.5 * h
        });
        if blocked {
            eprintln!("warning: hole {y}:{z}: an arm is in the screwdriver's way");
        }
    }
    if !holes.is_empty() {
        let mut txt = format!("Screw pull-out (rigid plate tipping about the edge that presses on the wall; load {} at x = {:.0} mm):\n", case_list(&cases), n[0] as f64 * h);
        // split cases are added as if they came together (on the safe side); real cases already are together
        let split = cases.iter().any(|c| c.name == "side");
        let (fd, fs) = (cases.iter().map(|c| -c.force[2]).fold(0.0, f64::max), cases.iter().map(|c| c.force[1].abs()).fold(0.0, f64::max));
        let shear = (fd * fd + fs * fs).sqrt() / holes.len() as f64;
        let per_case: Vec<Vec<[f64; 2]>> = if split {
            vec![screw_tension(&holes, 0.5 * n[1] as f64 * h, 0.5 * n[2] as f64 * h, n[0] as f64 * h, fd, fs)]
        } else {
            cases.iter().map(|c| screw_tension(&holes, 0.5 * n[1] as f64 * h, 0.5 * n[2] as f64 * h, n[0] as f64 * h, -c.force[2], c.force[1])).collect()
        };
        let worst: Vec<[f64; 2]> = (0..holes.len()).map(|i| per_case.iter().map(|t| t[i]).max_by(|a, b| (a[0] + a[1]).total_cmp(&(b[0] + b[1]))).unwrap_or([0.0; 2])).collect();
        for (t, [y, z]) in worst.iter().zip(&holes) {
            let both = t[0] + t[1];
            // the head bears on the countersink cone; it would punch a plug its own size through the plate
            let (bear, punch) = (both / (std::f64::consts::PI * (sr * sr - hr * hr)), both / (std::f64::consts::PI * 2.0 * sr * xb));
            txt += &format!("  hole {y}:{z}: pull {:.1} N from the down load, {:.1} N from the side load, {both:.1} N together = {:.1} kg on the wall plug; shear {shear:.1} N; head on the plate: bearing {bear:.1} MPa, punching shear {punch:.2} MPa (SF {:.0} / {:.0} on {strength} MPa)\n", t[0], t[1], both / 9.81, strength / bear, 0.577 * strength / punch);
        }
        print!("{txt}");
        std::fs::write(format!("{out}/screws.txt"), &txt).map_err(|e| e.to_string())?;
    }
    let (tris, thr, vol) = smooth(&grid, &solid, &keep, num("blur", 2.0)? as usize, &holes, [hr, sr], xb);
    std::fs::write(format!("{out}/bracket-smooth.stl"), tris_stl(&tris)).map_err(|e| e.to_string())?;
    println!("smooth: {} triangles, level {thr:.3}, {:.1} cm3 against {:.1} cm3 as voxels", tris.len(), vol / 1000.0, count as f64 * cell_cm3);
    if o.get("bars").is_some_and(|v| v != "0") {
        let reach = n[0] as f64 * h;
        let (mut list, lost) = bars(&grid, &solid, &keep);
        let arms = (0..grid.cells()).filter(|c| solid[*c] && !keep[*c]).count() as f64 * h.powi(3);
        eprintln!("{} pillars from {:.1} cm3 of arms ({:.2} cm3 belongs to none)", list.len(), arms / 1000.0, lost / 1000.0);
        let r0: Vec<f64> = list.iter().map(|b| b.r.first().copied().unwrap_or(0.0)).collect();
        let taper = o.get("taper").is_some_and(|v| v != "0");
        let (blend, res, rounds) = (num("blend", 3.0)?, num("bar_res", 0.75)?, num("rounds", 4.0)? as usize);
        if !(0.2..=h).contains(&res) || !(0.0..=20.0).contains(&blend) {
            return Err(format!("bar_res 0.2-{h} (the voxel), blend 0-20 mm"));
        }
        // the kept pad starts here; plate and pad are not bars, their peaks are where the load and supports go in
        let pad_from = (0..grid.cells()).filter(|c| keep[*c] && grid.cell_centre(*c)[0] > 0.5 * reach).map(|c| grid.cell_centre(c)[0] - 0.5 * h).fold(reach, f64::min);
        let two_way = o.get("size").is_some_and(|v| v != "0");
        let r_min = 0.5 * num("min_member", 5.0)?;
        let frame = o.get("plate").map_or("box", String::as_str);
        let mut strip = match frame {
            "box" => 0.0,
            "frame" => num("strip", 12.0)?,
            _ => return Err("plate: box or frame".into()),
        };
        let hold = o.get("hold").map_or(if strip > 0.0 { "screws" } else { "wall" }, String::as_str);
        let fixed_bars = match hold {
            "wall" => Region::Side { axis: 0, max: false, depth: 0.5 * h }, // as the earlier runs
            // the material each screw clamps: round its axis out to the countersink plus 1.5 mm, through the plate
            "screws" | "contact" if !holes.is_empty() => Region::Union(holes.iter().map(|[y, z]| Region::Cylinder { point: [0.0, *y, *z], dir: [1.0, 0.0, 0.0], r: sr + 1.5, len: xb }).collect()),
            _ => return Err("hold: wall, screws or contact (both need holes=)".into()),
        };
        if strip > 0.0 && (holes.is_empty() || !(4.0..=40.0).contains(&strip)) {
            return Err("plate=frame needs holes= and strip 4-40 mm".into());
        }
        let mut round = 0;
        type State = (Vec<[[f64; 3]; 3]>, Vec<[f64; 3]>, Vec<[u32; 3]>, Grid, Analysis, Vec<f64>, f64, Vec<Bar>, usize, f64);
        let (mut best, mut last): (Option<State>, Option<State>) = (None, None);
        loop {
            let tris = bar_mesh(&grid, &keep, &list, blend, res, &holes, [hr, sr], xb, strip);
            let (pos, tri) = index(&tris);
            let (g, sol, an) = analyse_mesh(&pos, &tri, &fixed_bars, hold == "contact")?;
            // each bar's peak: the material cells nearest to it, outside the kept plate and pad; the plate's own
            // peak away from where it is held (the wall face, or round the screws)
            let mut station: Vec<Vec<f64>> = list.iter().map(|b| vec![0.0; b.pts.len()]).collect();
            let mut plate_peak = 0.0f64;
            for c in (0..g.cells()).filter(|c| sol[*c]) {
                let p = g.cell_centre(c);
                let worst = an.cases.iter().map(|k| k.vm[c]).filter(|v| v.is_finite()).fold(0.0, f64::max);
                if p[0] < xb {
                    // with wall contact, the cells where the wall pushes (and its edge) are held as well, per case
                    if hold == "contact" {
                        let free = an.cases.iter().filter(|k| !k.near[c]).map(|k| k.vm[c]).filter(|v| v.is_finite()).fold(0.0, f64::max);
                        if !holes.iter().any(|[y, z]| (p[1] - y).hypot(p[2] - z) < sr + 1.5 + 2.0 * cv) {
                            plate_peak = plate_peak.max(free);
                        }
                        continue;
                    }
                    let held = if hold == "wall" { p[0] < 2.0 * cv } else { holes.iter().any(|[y, z]| (p[1] - y).hypot(p[2] - z) < sr + 1.5 + 2.0 * cv) };
                    if !held {
                        plate_peak = plate_peak.max(worst);
                    }
                    continue;
                }
                if p[0] > pad_from {
                    continue;
                }
                let near = list.iter().enumerate().map(|(i, b)| (i, b.near(p))).min_by(|a, b| a.1.0.total_cmp(&b.1.0));
                if let Some((i, (d, k))) = near && d < cv {
                    // the same cells as the away peak: next to a screw or where the wall pushes, the peak is the
                    // clamping spike (it grows with every finer grid), and growing the bar cannot lower it
                    let free = an.cases.iter().filter(|k| !k.near[c]).map(|k| k.vm[c]).filter(|v| v.is_finite()).fold(0.0, f64::max);
                    station[i][k] = station[i][k].max(free);
                }
            }
            let peak: Vec<f64> = station.iter().map(|v| v.iter().copied().fold(0.0, f64::max)).collect();
            let sf_of = |v: f64| strength / v.max(1e-12);
            let sfs: Vec<String> = an.cases.iter().map(|k| format!("{} SF {:.1}, tip {:.2} mm", k.name, sf_of(k.max_vm_away), k.max_disp)).collect();
            let weak = (0..list.len()).filter(|i| sf_of(peak[*i]) < sf).count();
            // well above: the worst point (whole bar), or any point (tapered) with room to get thinner
            let slack = (0..list.len()).filter(|i| {
                let b = &list[*i];
                if taper {
                    station[*i].iter().zip(&b.r).any(|(pk, r)| *pk > 0.0 && sf_of(*pk) > 1.3 * sf && *r > r_min + 1e-9)
                } else {
                    sf_of(peak[*i]) > 1.3 * sf && b.rmax() > r_min + 1e-9
                }
            }).count();
            let vol = volume(&tris);
            eprintln!("bars round {round}: {:.1} cm3, {}; plate SF {:.1} (strips {strip:.1} mm); {weak} bars below SF {sf}, {slack} well above, {:.0} s", vol / 1000.0, sfs.join("; "), sf_of(plate_peak), t0.elapsed().as_secs_f64());
            // the frame's strips are sized too: plate bending stress goes with 1 / width
            let plate_sf = sf_of(plate_peak);
            let plate_off = strip > 0.0 && (plate_sf < sf || (two_way && plate_sf > 1.3 * sf && strip > 8.0));
            let stop = round >= rounds || (weak == 0 && !plate_off && (!two_way || slack == 0));
            let next_strip = if plate_off { (strip * (1.1 * sf / plate_sf).clamp(0.8, 1.6)).clamp(8.0, 40.0) } else { strip };
            let next: Vec<Vec<f64>> = list.iter().zip(&station).map(|(b, st)| resize(b, st, sf, strength, two_way, taper, r_min)).collect();
            let state = (tris, pos, tri, g, an, peak, plate_peak, list.clone(), round, strip);
            if weak == 0 && sf_of(plate_peak) >= sf && best.as_ref().is_none_or(|b| vol < volume(&b.0)) {
                best = Some(state);
            } else {
                last = Some(state);
            }
            if stop {
                break;
            }
            strip = next_strip;
            for (b, r) in list.iter_mut().zip(next) {
                b.r = r;
            }
            // a step that changes nothing would repeat the same round
            if round > 0 && last.as_ref().is_some_and(|l| l.7.iter().zip(&list).all(|(a, b)| a.r == b.r)) && strip == last.as_ref().map_or(strip, |l| l.9) {
                break;
            }
            round += 1;
        }
        let passed = best.is_some();
        let (tris, pos, tri, g, an, peak, plate_peak, list, used, strip) = best.or(last).ok_or("no bar round ran")?;
        std::fs::write(format!("{out}/bars.stl"), tris_stl(&tris)).map_err(|e| e.to_string())?;
        let v = volume(&tris);
        let mut rep = format!("Round bars from {}: {} pillars, each a round bar of its own voxel volume between its ends (fillets {blend} mm); sizing {} to SF {sf}, {round} rounds, kept round {used}{}\n", from.as_deref().unwrap_or("this run"), list.len(), match (two_way, taper) { (true, true) => "both ways, point by point along each bar (tapered)", (true, false) => "both ways (thinner and thicker)", (false, true) => "only thicker, point by point (tapered)", _ => "only thicker" }, if passed { " (the lightest that passes)" } else { " (NONE PASSED: the last one)" });
        rep += &format!("Wall plate: {}, {xb} mm thick; held {}\n", if strip > 0.0 { format!("frame (pads round screws and bar roots, {strip:.1} mm strips)") } else { "full box".into() }, match hold { "wall" => "on the whole wall face", "contact" => "at the screws (the material each screw clamps), and the wall pushes back where the plate presses on it (no friction)", _ => "only at the screws (the material each screw clamps; no wall contact, so on the safe side)" });
        rep += &format!("Loads at the tip: {}\n{}\n", case_list(&cases), print_line(print, strength, z_mpa));
        if creep {
            rep += &creep_line(ks, km);
        }
        for (i, b) in list.iter().enumerate() {
            let (a, z) = (b.pts.first().copied().unwrap_or_default(), b.pts.last().copied().unwrap_or_default());
            rep += &format!("  bar {i:>2}: ({:.0}, {:.0}, {:.0}) -> ({:.0}, {:.0}, {:.0}), {:.0} mm long, pillar {:.2} cm3 -> diameter {:.1} mm; now {:.1} mm at the start, {:.1} at the end, {:.1} thinnest ({:.2} cm3), peak {:.2} MPa (SF {:.1})\n", a[0], a[1], a[2], z[0], z[1], z[2], b.length(), b.vol / 1000.0, 2.0 * r0[i], 2.0 * b.r.first().copied().unwrap_or(0.0), 2.0 * b.r.last().copied().unwrap_or(0.0), 2.0 * b.r.iter().copied().fold(f64::INFINITY, f64::min), b.volume() / 1000.0, peak[i], strength / peak[i].max(1e-12));
        }
        rep += &format!("  plate: peak {plate_peak:.2} MPa (SF {:.1}) away from where it is held\n", strength / plate_peak.max(1e-12));
        for c in &an.cases {
            rep += &format!("{}: away peak {:.2} MPa (SF {:.1} on {strength} MPa), tip moves {:.3} mm (check on {cv} mm voxels){}\n", c.name, c.max_vm_away, strength / c.max_vm_away.max(1e-12), c.max_disp, if hold == "contact" { format!("; wall pushes on {} nodes (settled in {} solves)", c.contact_nodes, c.contact_rounds) } else { String::new() });
        }
        rep += "  (a bar's peak counts only material outside the plate and the tip pad; the away peak also counts inside them)\n";
        rep += &format!("bars.stl: {} triangles, {:.1} cm3 = {:.1} g (voxel design {:.1} cm3; arms {:.1} cm3, of which {:.2} in no pillar)\n", tris.len(), v / 1000.0, v / 1000.0 * m.strengths().density_kg_m3.unwrap_or(1270.0) / 1000.0, count as f64 * cell_cm3, arms / 1000.0, lost / 1000.0);
        if z_mpa.is_finite() {
            rep += &orientations(&an, &list, strength, ratio, print);
        }
        rep += &stress_pictures(&out, &pos, &tri, &g, &an)?;
        std::fs::write(format!("{out}/bars-report.txt"), &rep).map_err(|e| e.to_string())?;
        print!("{rep}");
        return Ok(());
    }
    if from.is_some() {
        return Ok(());
    }
    // final analysis of exactly what was written
    let setup = Setup { grid: &grid, solid: &solid, e_mpa: e_gpa * 1000.0, nu, fixed: &fixed, load: &load, cg_tol: 1e-8, cg_max: 50_000, exclude_layers: 2, cell_scale: None, multigrid: true, contact: None };
    let fin = layered(analyse(&setup, &cases, |_| {})?, &grid);
    let mut rep = String::new();
    for c in &fin.cases {
        for (ax, name) in [(0, "x"), (1, "y"), (2, "z")] {
            let (w, hh, rgb) = projection(&grid, &c.vm, ax, c.max_vm_away, 4);
            std::fs::write(format!("{out}/{}-along-{name}.png", c.name), png(w, hh, &rgb)).map_err(|e| e.to_string())?;
        }
        rep += &format!("{}: away peak {:.2} MPa (SF {:.1} on {strength} MPa), p99 {:.2}, tip moves {:.3} mm\n", c.name, c.max_vm_away, strength / c.max_vm_away.max(1e-12), c.p99_vm, c.max_disp);
    }
    rep += &format!("{} cells = {:.1} cm3 = {:.1} g ({:.0} % of the frustum), {reason}, {:.0} s\n", count, count as f64 * cell_cm3, count as f64 * cell_cm3 * m.strengths().density_kg_m3.unwrap_or(1270.0) / 1000.0, 100.0 * count as f64 / start as f64, t0.elapsed().as_secs_f64());
    std::fs::write(format!("{out}/report.txt"), &rep).map_err(|e| e.to_string())?;
    print!("{rep}");
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn one_cell_has_twelve_triangles() {
        let g = Grid { origin: [0.0; 3], h: 1.0, n: [1, 1, 1] };
        let b = stl(&g, &[true]);
        assert_eq!(u32::from_le_bytes(b[80..84].try_into().unwrap()), 12);
        assert_eq!(b.len(), 84 + 12 * 50);
    }

    #[test]
    fn opening_removes_thin_members_and_keeps_thick_ones() {
        let g = Grid { origin: [0.0; 3], h: 1.0, n: [8, 4, 4] };
        let mut s = vec![false; 128];
        for i in 0..8 {
            s[g.cell_index(i, 0, 0)] = true; // a 1-voxel line
        }
        for i in 0..2 {
            for j in 2..4 {
                for k in 2..4 {
                    s[g.cell_index(i, j, k)] = true; // a 2 x 2 x 2 block
                }
            }
        }
        let o = open(&g, &s, 2);
        assert!(!o[g.cell_index(3, 0, 0)] && o[g.cell_index(1, 3, 3)]);
        assert_eq!(o.iter().filter(|v| **v).count(), 8);
    }

    #[test]
    fn smooth_keeps_the_volume_and_the_flat_faces() {
        let g = Grid { origin: [0.0, -3.0, -3.0], h: 1.0, n: [6, 6, 6] };
        let (tris, _, vol) = smooth(&g, &vec![true; 216], &vec![false; 216], 2, &[], [HOLE_R, SINK_R], 0.0);
        assert!((vol - 216.0).abs() < 0.05 * 216.0, "volume {vol}");
        assert!(tris.iter().flatten().all(|p| (0.0..=6.0).contains(&p[0])));
        // a kept block stays an exact block
        let (tris, _, vol) = smooth(&g, &vec![true; 216], &vec![true; 216], 2, &[], [HOLE_R, SINK_R], 0.0);
        assert!((vol - 216.0).abs() < 0.03 * 216.0, "kept volume {vol}"); // the edges lose a little
        assert!(tris.iter().flatten().all(|p| p[1].abs() <= 3.001 && p[2].abs() <= 3.001));
    }

    #[test]
    fn bends_become_arcs_and_the_ends_stay() {
        let p = round_bends(&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], 3);
        assert_eq!((p[0], p[p.len() - 1]), ([0.0; 3], [10.0, 10.0, 0.0]));
        let turn = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| { let (u, v) = (sub(b, a), sub(c, b)); (dot(u, v) / (dot(u, u) * dot(v, v)).sqrt()).acos().to_degrees() };
        let sharpest = p.windows(3).map(|w| turn(w[0], w[1], w[2])).fold(0.0, f64::max);
        assert!(sharpest < 20.0, "a 90 degree bend in steps of at most {sharpest:.1} degrees");
        assert_eq!(round_bends(&[[0.0; 3], [1.0, 0.0, 0.0]], 3).len(), 2);
    }

    #[test]
    fn frame_plate_joins_every_bar_root_to_a_screw() {
        let bar = Bar { pts: vec![[2.0, 30.0, 40.0], [100.0, 0.0, 0.0]], vol: 1.0, r: vec![3.0; 2] };
        let caps = plate_frame(&[bar], &[[0.0, 41.0], [0.0, -41.0]], [2.75, 5.2], 5.0, 12.0);
        // spine, two screw pads, root pad, strip root -> top screw
        assert_eq!(caps.len(), 5, "{caps:?}");
        assert!(caps.contains(&([30.0, 40.0], [0.0, 41.0], 6.0)) && caps.contains(&([30.0, 40.0], [30.0, 40.0], 6.0)));
        assert!(plate_frame(&[], &[[0.0, 1.0]], [2.75, 5.2], 5.0, 0.0).is_empty());
        // as a mesh: only inside the plate, about the area of the pads and strips times its thickness
        let g = Grid { origin: [0.0, -10.0, -10.0], h: 1.0, n: [10, 20, 20] };
        let keep = vec![false; g.cells()];
        let caps = plate_frame(&[], &[[0.0, 6.0], [0.0, -6.0]], [1.0, 2.0], 3.0, 4.0);
        let tris = bar_mesh(&g, &keep, &[], 0.0, 0.25, &[[0.0, 6.0], [0.0, -6.0]], [1.0, 2.0], 3.0, 4.0);
        assert!(tris.iter().flatten().all(|p| p[0] >= -1e-9 && p[0] <= 3.0 + 1e-9), "plate stays between x 0 and 3");
        let area = 12.0 * 4.0 + 2.0 * std::f64::consts::PI * 36.0; // spine 12 x 4 + two pads of radius 6, less overlap
        assert!(caps.len() == 3 && volume(&tris) < 3.0 * area && volume(&tris) > 0.5 * 3.0 * area, "{}", volume(&tris));
        // strips wider than the grid's own margin: the mesh must still be closed
        let wide = bar_mesh(&g, &keep, &[], 0.0, 0.25, &[[0.0, 9.0], [0.0, -9.0]], [1.0, 2.0], 3.0, 16.0);
        let (_, tri) = index(&wide);
        let edges: std::collections::HashSet<(u32, u32)> = tri.iter().flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]).collect();
        assert!(edges.iter().all(|(a, b)| edges.contains(&(*b, *a))), "the wide frame has an open edge");
    }

    #[test]
    fn round_cone_distances() {
        let (a, b) = ([0.0; 3], [10.0, 0.0, 0.0]);
        // equal radii: a capsule
        assert!((round_cone([5.0, 3.0, 0.0], a, b, 1.0, 1.0) - 2.0).abs() < 1e-9);
        assert!((round_cone([-2.0, 0.0, 0.0], a, b, 1.0, 1.0) - 1.0).abs() < 1e-9);
        // tapered 2 -> 1: past the ends the spheres, beside it the cone (slant 0.1 per unit)
        assert!((round_cone([-3.0, 0.0, 0.0], a, b, 2.0, 1.0) - 1.0).abs() < 1e-9);
        assert!((round_cone([12.0, 0.0, 0.0], a, b, 2.0, 1.0) - 1.0).abs() < 1e-9);
        let s = 0.1f64.atan();
        let want = 5.0 * s.cos() - (2.0 - 0.1 * 5.0 + 5.0 * 0.1 * 0.0) * 1.0 / s.cos() * s.cos() * s.cos() - 0.0;
        let got = round_cone([5.0, 5.0, 0.0], a, b, 2.0, 1.0);
        assert!(got > 3.4 && got < 3.6, "{got} (about {want})");
        assert!(round_cone([5.0, 0.0, 0.0], a, b, 2.0, 1.0) < -1.4, "inside");
        // one sphere swallows the other
        assert!((round_cone([0.0, 0.0, 5.0], a, [0.5, 0.0, 0.0], 3.0, 1.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tapered_sizing_keeps_a_thick_root_and_a_thin_middle() {
        let pts: Vec<[f64; 3]> = (0..11).map(|i| [10.0 * f64::from(i), 0.0, 0.0]).collect();
        let bar = Bar { pts, vol: 0.0, r: vec![4.0; 11] };
        // stress high at the root, low in the middle, unmeasured at the far end
        let peak = [14.0, 12.0, 6.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0];
        let r = resize(&bar, &peak, 4.0, 47.0, true, true, 2.5);
        assert!(r[0] > 4.0 && r[5] < 4.0 && r.iter().all(|v| *v >= 2.5), "{r:?}");
        // SF 4.7 at a point already in the band: it keeps its size, smoothing does not thin it
        let mid = resize(&bar, &[10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 4.0, 47.0, true, true, 2.5);
        assert!(mid[0] >= 4.0 - 1e-12, "{mid:?}");
        assert!(r.windows(2).all(|w| (w[0] - w[1]).abs() <= 5.0 + 1e-9));
        assert_eq!(r[9], r[10], "the unmeasured end takes its neighbour's size");
        // not tapered: the whole bar by its worst point
        let u = resize(&bar, &peak, 4.0, 47.0, true, false, 2.5);
        assert!(u.iter().all(|v| (v - u[0]).abs() < 1e-12) && u[0] > 4.0);
    }

    #[test]
    fn simple_points() {
        let mut nb = [false; 27];
        nb[13] = true;
        assert!(!simple(&nb), "a lone voxel is a piece of its own");
        nb[4] = true;
        assert!(simple(&nb), "the end of a line");
        nb[22] = true;
        assert!(!simple(&nb), "the middle of a line holds it together");
        assert!(!simple(&[true; 27]), "inside a block: a hole would open");
    }

    #[test]
    fn one_pillar_becomes_one_bar_of_its_volume() {
        // plate x 0-2, pad x 18-20, a 4 x 4 arm between them, on a 1 mm grid
        let g = Grid { origin: [0.0, -5.0, -5.0], h: 1.0, n: [20, 10, 10] };
        let at = |c: usize| { let p = g.cell_centre(c); (p[0], p[1].abs(), p[2].abs()) };
        let keep: Vec<bool> = (0..g.cells()).map(|c| { let (x, y, z) = at(c); (x < 2.0 || x > 18.0) && y < 3.0 && z < 3.0 }).collect();
        let solid: Vec<bool> = (0..g.cells()).map(|c| { let (_, y, z) = at(c); keep[c] || (y < 2.0 && z < 2.0) }).collect();
        let (list, lost) = bars(&g, &solid, &keep);
        assert_eq!(list.len(), 1, "{list:?}");
        assert!(lost.abs() < 1e-9);
        let b = &list[0];
        assert!((b.vol - 16.0 * 16.0).abs() < 1e-9, "the arm is 16 x 4 x 4: {}", b.vol);
        let ends = [b.pts[0][0], b.pts[b.pts.len() - 1][0]];
        assert!(ends.iter().any(|x| *x < 2.0) && ends.iter().any(|x| *x > 18.0), "runs into plate and pad: {:?}", b.pts);
        assert!(b.pts.iter().all(|p| p[1].abs() < 1.01 && p[2].abs() < 1.01), "on the arm's axis: {:?}", b.pts);
        // the mesh holds the plate, the pad and the bar
        let tris = bar_mesh(&g, &keep, &list, 0.0, 0.25, &[], [HOLE_R, SINK_R], 0.0, 0.0);
        let want = 2.0 * 2.0 * 6.0 * 6.0 + std::f64::consts::PI * b.r[0] * b.r[0] * 16.0;
        assert!((volume(&tris) - want).abs() < 0.05 * want, "{} against {want}", volume(&tris));
        let (pos, tri) = index(&tris);
        assert!(pos.len() < tris.len() && tri.len() == tris.len());
        // closed: every edge once each way; no slivers
        let edges: std::collections::HashSet<(u32, u32)> = tri.iter().flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]).collect();
        assert_eq!(edges.len(), 3 * tri.len(), "an edge is used twice the same way");
        assert!(edges.iter().all(|(a, b)| edges.contains(&(*b, *a))), "the mesh has an open edge");
        let tiny = tris.iter().filter(|t| { let c = cross(sub(t[1], t[0]), sub(t[2], t[0])); dot(c, c).sqrt() / 2.0 < 1e-4 }).count();
        assert_eq!(tiny, 0, "slivers");
    }

    #[test]
    fn overhang_counts_the_flat_parts_only() {
        let bar = |pts: Vec<[f64; 3]>| Bar { r: vec![2.5; pts.len()], pts, vol: 0.0 };
        let flat_then_steep = bar(vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 20.0]]);
        assert!((overhang_mm(&flat_then_steep, 2) - 10.0).abs() < 1e-9, "upright: the flat 10 mm need support");
        assert!((overhang_mm(&flat_then_steep, 0) - 20.0).abs() < 1e-9, "on the plate: now the 20 mm run is flat");
        assert!(overhang_mm(&bar(vec![[0.0; 3], [10.0, 0.0, 11.0]]), 2) == 0.0, "steeper than 45 degrees prints");
        assert_eq!(across_strength("petg-printed"), Some(30.0));
        assert_eq!(across_strength("steel-s235jr"), None);
    }

    #[test]
    fn load_cases_split_real_and_knock() {
        let split = load_cases("split", 50.0, 15.0, 1.0).unwrap();
        assert_eq!(split.iter().map(|c| (c.name.as_str(), c.force)).collect::<Vec<_>>(), vec![("down", [0.0, 0.0, -50.0]), ("side", [0.0, 15.0, 0.0])]);
        let real = load_cases("real", 50.0, 15.0, 2.0).unwrap();
        assert_eq!(real.iter().map(|c| (c.name.as_str(), c.force)).collect::<Vec<_>>(), vec![("down+side", [0.0, 30.0, -100.0]), ("down-side", [0.0, -30.0, -100.0])]);
        assert!(load_cases("both", 50.0, 15.0, 1.0).is_err() && load_cases("real", 50.0, 15.0, 0.5).is_err() && load_cases("real", -1.0, 15.0, 1.0).is_err());
    }

    #[test]
    fn screw_tension_matches_the_lever_by_hand() {
        let t = screw_tension(&[[-6.5, 41.0], [-6.5, -41.0]], 40.0, 50.0, 150.0, 50.0, 15.0);
        let near = |a: f64, b: f64| (a - b).abs() < 0.01;
        assert!(near(t[0][0], 7500.0 * 91.0 / (91.0 * 91.0 + 81.0)) && near(t[1][0], 7500.0 * 9.0 / (91.0 * 91.0 + 81.0)), "{t:?}");
        assert!(near(t[0][1], 2250.0 / 93.0) && near(t[1][1], 2250.0 / 93.0), "{t:?}");
        // a side load the other way tips about the other edge: 33.5 mm instead of 46.5
        assert!(near(screw_tension(&[[-6.5, 0.0]], 40.0, 50.0, 150.0, 0.0, -15.0)[0][1], 2250.0 / 33.5));
    }

    #[test]
    fn screw_hole_is_round_and_takes_its_own_volume() {
        // a 5 mm wall plate (kept) on a 10 mm grid: its front face is free, as on the bracket
        let g = Grid { origin: [0.0, -6.0, -6.0], h: 1.0, n: [10, 12, 12] };
        let plate: Vec<bool> = (0..g.cells()).map(|c| g.cell_ijk(c)[0] < 5).collect();
        let (_, _, plain) = smooth(&g, &plate, &plate, 2, &[], [HOLE_R, SINK_R], 5.0);
        let (tris, _, holed) = smooth(&g, &plate, &plate, 2, &[[0.0, 0.0]], [HOLE_R, SINK_R], 5.0);
        let hv = hole_volume(5.0, [HOLE_R, SINK_R]);
        assert!(((plain - holed) - hv).abs() < 0.08 * hv, "removed {} against {hv}", plain - holed);
        // on the straight part of the hole every corner lies on the 4.5 mm circle
        let wall: Vec<f64> = tris.iter().flatten().filter(|p| p[0] > 0.5 && p[0] < 2.5).map(|p| p[1].hypot(p[2])).filter(|r| *r < 3.5).collect();
        assert!(!wall.is_empty() && wall.iter().all(|r| (r - HOLE_R).abs() < 0.05), "{wall:?}");
        // and no corner anywhere inside the hole, the wall face included
        let inside: Vec<&[f64; 3]> = tris.iter().flatten().filter(|p| p[1].hypot(p[2]) < HOLE_R - 1e-9).collect();
        assert!(inside.is_empty(), "{inside:?}");
        assert_eq!(parse_holes("-6.5:41,-6.5:-41").unwrap(), vec![[-6.5, 41.0], [-6.5, -41.0]]);
        assert!(parse_holes("1:x").is_err() && parse_holes("3").is_err() && parse_holes("").unwrap().is_empty());
    }
}
