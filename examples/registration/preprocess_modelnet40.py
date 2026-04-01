#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Preprocess ModelNet40 .off meshes → dense point clouds (.pt).

Reads every .off file under ModelNet40/, samples N_dense points
from the mesh surface (area-weighted), normalizes (mean-center +
unit-sphere), computes face normals, and saves as .pt tensors.

Usage:
    python examples/registration/preprocess_modelnet40.py \\
        --root data/ModelNet40 \\
        --n_points 4096 \\
        --output data/modelnet40_4096.pt

Output format (.pt):
    {
        'points':  [M, N_dense, 3]   float32  — surface samples
        'normals': [M, N_dense, 3]   float32  — per-point face normals
        'labels':  [M]               int64    — category label
        'splits':  [M]               int64    — 0=train, 1=test
        'categories': list[str]               — category name list
    }

Typical timing: ~3-5 min for 12311 meshes × 4096 points.
Output size: ~600MB for 4096 points.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def parse_off(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an OFF mesh file.

    Handles both 'OFF\\n' and 'OFF<counts>' header formats.

    Args:
        filepath: Path to .off file

    Returns:
        vertices: [V, 3] float32
        faces: [F, 3] int64 (triangle indices only)
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        if header == 'OFF':
            counts = f.readline().strip().split()
        elif header.startswith('OFF'):
            counts = header[3:].strip().split()
            if len(counts) < 3:
                counts = f.readline().strip().split()
        else:
            raise ValueError(f"Invalid OFF: {filepath}, header='{header}'")

        n_verts = int(counts[0])
        n_faces = int(counts[1])

        vertices = np.zeros((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            line = f.readline().strip().split()
            vertices[i] = [float(line[0]), float(line[1]), float(line[2])]

        faces_list = []
        for i in range(n_faces):
            line = f.readline().strip().split()
            n = int(line[0])
            if n == 3:
                faces_list.append([int(line[1]), int(line[2]), int(line[3])])

    faces = np.array(faces_list, dtype=np.int64) if faces_list else np.zeros((0, 3), dtype=np.int64)
    return vertices, faces


def sample_with_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points + normals uniformly from mesh surface.

    Handles CAD meshes with large coordinates (e.g. mm units, values ~10000)
    by pre-normalizing vertices and computing areas in float64 to prevent
    cross-product overflow in float32 (max ~3.4e38).

    Args:
        vertices: [V, 3] raw mesh vertices (any scale)
        faces: [F, 3] triangle indices
        n_points: Target number of points
        rng: NumPy random generator

    Returns:
        points: [n_points, 3] surface samples (in original vertex scale)
        normals: [n_points, 3] unit face normals
    """
    if len(faces) == 0:
        # Degenerate mesh: fall back to vertices
        idx = rng.choice(len(vertices), size=n_points, replace=True)
        pts = vertices[idx].copy()
        norms = np.zeros_like(pts)
        norms[:, 2] = 1.0  # dummy normal
        return pts, norms

    # ── Pre-normalize to prevent float32 overflow in cross product ──
    # CAD meshes can have coords ~10000 → cross² ~ 10^16 → sum ~ 10^16
    # × 3 axes ~ 3e16, still fits float32, but some ModelNet40 meshes
    # have coords ~ 1e4 with faces spanning the full range → cross ~ 1e8
    # → cross² ~ 1e16 → sum ~ 3e16: still ok. BUT if coords ~ 1e5
    # (rare but exists) → cross ~ 1e10 → cross² ~ 1e20 → overflow.
    # Safe fix: shift to float64 for area computation.
    v0 = vertices[faces[:, 0]].astype(np.float64)  # [F, 3]
    v1 = vertices[faces[:, 1]].astype(np.float64)
    v2 = vertices[faces[:, 2]].astype(np.float64)

    e1 = v1 - v0  # [F, 3]
    e2 = v2 - v0

    # Cross product in float64 → no overflow
    cross = np.cross(e1, e2)  # [F, 3] float64
    cross_norms = np.sqrt((cross ** 2).sum(axis=1, keepdims=True))  # [F, 1]
    cross_norms = np.maximum(cross_norms, 1e-30)

    face_normals = (cross / cross_norms).astype(np.float32)  # [F, 3] unit normals
    areas = (0.5 * cross_norms.squeeze())  # [F] float64

    total_area = areas.sum()
    if total_area < 1e-30:
        idx = rng.choice(len(vertices), size=n_points, replace=True)
        return vertices[idx].copy(), np.tile([0, 0, 1], (n_points, 1)).astype(np.float32)

    # Area-weighted face sampling (probs in float64 for precision)
    probs = areas / total_area  # float64, no NaN
    face_idx = rng.choice(len(faces), size=n_points, p=probs, replace=True)

    # Barycentric interpolation (in original float32 vertex coords)
    r1 = rng.random(n_points).astype(np.float32)
    r2 = rng.random(n_points).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)

    a = 1.0 - sqrt_r1
    b = sqrt_r1 * (1.0 - r2)
    c = sqrt_r1 * r2

    points = (a[:, None] * vertices[faces[face_idx, 0]] +
              b[:, None] * vertices[faces[face_idx, 1]] +
              c[:, None] * vertices[faces[face_idx, 2]])

    normals = face_normals[face_idx]  # [n_points, 3]

    return points.astype(np.float32), normals.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Preprocess ModelNet40 .off → .pt')
    parser.add_argument('--root', type=str, default='data/ModelNet40',
                        help='Path to ModelNet40/ directory')
    parser.add_argument('--n_points', type=int, default=4096,
                        help='Dense points per shape (subsample at train time)')
    parser.add_argument('--output', type=str, default='data/modelnet40_4096.pt',
                        help='Output .pt file path')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} not found")
        sys.exit(1)

    # Discover all categories (sorted for deterministic label IDs)
    cat_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    categories = [d.name for d in cat_dirs]
    print(f"Found {len(categories)} categories: {categories[:5]}... ({len(categories)} total)")

    all_points: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    all_labels: List[int] = []
    all_splits: List[int] = []  # 0=train, 1=test

    rng = np.random.default_rng(42)
    n_errors = 0
    t0 = time.time()

    for cat_id, cat_dir in enumerate(cat_dirs):
        for split_id, split_name in enumerate(['train', 'test']):
            split_dir = cat_dir / split_name
            if not split_dir.is_dir():
                continue
            off_files = sorted(split_dir.glob('*.off'))
            for off_file in off_files:
                try:
                    verts, faces = parse_off(str(off_file))

                    # Sample dense point cloud from mesh surface
                    pts, norms = sample_with_normals(verts, faces, args.n_points, rng)

                    # Normalize AFTER sampling (critical: mean of surface points,
                    # not mean of raw vertices)
                    center = pts.mean(axis=0, keepdims=True)
                    pts = pts - center

                    # Unit sphere scaling: max(||p_i||) = 1.0
                    max_norm = np.sqrt((pts ** 2).sum(axis=1).max())
                    if max_norm > 1e-8:
                        pts = pts / max_norm

                    all_points.append(pts)
                    all_normals.append(norms)
                    all_labels.append(cat_id)
                    all_splits.append(split_id)

                except Exception as e:
                    n_errors += 1
                    if n_errors <= 5:
                        print(f"  Warning: {off_file}: {e}")

        elapsed = time.time() - t0
        n_done = len(all_points)
        print(f"  [{cat_id+1:>2}/{len(categories)}] {cat_dir.name:<15} "
              f"total={n_done:>6} shapes  ({elapsed:.0f}s)", end='\r')

    print()

    # Stack into tensors
    points_tensor = torch.from_numpy(np.stack(all_points))   # [M, N, 3]
    normals_tensor = torch.from_numpy(np.stack(all_normals))  # [M, N, 3]
    labels_tensor = torch.tensor(all_labels, dtype=torch.int64)
    splits_tensor = torch.tensor(all_splits, dtype=torch.int64)

    data = {
        'points': points_tensor,
        'normals': normals_tensor,
        'labels': labels_tensor,
        'splits': splits_tensor,
        'categories': categories,
    }

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1024**2

    n_train = (splits_tensor == 0).sum().item()
    n_test = (splits_tensor == 1).sum().item()

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Shapes: {len(all_points)} ({n_train} train + {n_test} test)")
    print(f"  Points/shape: {args.n_points}")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {args.output} ({size_mb:.1f} MB)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
