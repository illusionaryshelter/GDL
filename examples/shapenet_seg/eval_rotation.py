#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Zero-shot SO(3) rotation equivariance test for part segmentation.

This script proves the architectural guarantee of SE(3) equivariance:
    A model trained on CANONICAL poses should produce IDENTICAL predictions
    under ARBITRARY rotations — with ZERO retraining or augmentation.

This test would CATASTROPHICALLY FAIL on:
    - PointNet, PointNet++, DGCNN, PointNeXt (without SO(3) augmentation)
    - Any method using spatial coordinates as scalar features

Protocol:
    1. Load trained model
    2. For each test shape:
       a. Predict with original pose → pred_original
       b. Apply random SO(3) rotation R to pos AND normals
          ⚠ LANDMINE: normals MUST be co-rotated! Rotating pos without
          rotating normals creates physically impossible geometry.
       c. Predict with rotated pose → pred_rotated
       d. Compare: pred_rotated should == pred_original (exact match)
    3. Report:
       - Per-shape rotation consistency (% exact match)
       - mIoU original vs mIoU rotated
       - mIoU drop (should be < 0.5%)

Usage:
    python examples/shapenet_seg/eval_rotation.py \\
        --checkpoint checkpoints/shapenet_seg/best_model.pt \\
        --data_root data/ShapeNetPart \\
        --num_rotations 10
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch import Tensor

from examples.shapenet_seg.dataset import (
    ShapeNetPartDataset,
    collate_fn,
    SEG_CLASSES,
    NUM_CATEGORIES,
)
from examples.shapenet_seg.model import SE3PartSegNet
from examples.shapenet_seg.train import compute_part_miou


def random_rotation_matrix(device: torch.device) -> Tensor:
    """Generate a random SO(3) rotation matrix via QR decomposition.

    Returns:
        R: [3, 3] float32, proper rotation (det = +1)
    """
    # Generate in float64 for numerical precision, convert to float32
    M = torch.randn(3, 3, dtype=torch.float64)
    Q, R_diag = torch.linalg.qr(M)
    signs = torch.sign(torch.diag(R_diag))
    Q = Q * signs.unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.float().to(device)


@torch.no_grad()
def predict_batch(
    model: SE3PartSegNet,
    pos: Tensor,
    normals: Tensor,
    ptr: Tensor,
    cat_indices: Tensor,
) -> Tensor:
    """Run model and return per-point predictions.

    Returns:
        preds: [N_total] int64 — predicted part labels
    """
    logits = model(pos, ptr, cat_indices, normals=normals)
    return logits.argmax(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Zero-shot SO(3) rotation test for part segmentation'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='data/ShapeNetPart')
    parser.add_argument('--num_rotations', type=int, default=10,
                        help='Number of random rotations to test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Zero-Shot SO(3) Rotation Equivariance Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Rotations per shape: {args.num_rotations}")
    print()

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt['args']

    model = SE3PartSegNet(
        in_channels=model_args.get('in_channels', 1),
        hidden_scalar=model_args['hidden_scalar'],
        hidden_vector=model_args['hidden_vector'],
        num_stages=model_args['num_stages'],
        layers_per_stage=model_args['layers_per_stage'],
        pool_ratio=model_args['pool_ratio'],
        use_normals=not model_args.get('no_normals', False),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {ckpt['epoch']}")
    print(f"Checkpoint inst_mIoU: {ckpt['inst_miou']:.4f}")
    print()

    # ── Dataset ──
    test_dataset = ShapeNetPartDataset(
        args.data_root, split='test', normalize=True, download=False
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline (canonical pose)
    # ═══════════════════════════════════════════════════════════════
    print("Phase 1: Baseline predictions (canonical pose)...")

    all_preds_orig: List[Tensor] = []
    all_labels: List[Tensor] = []
    all_cats: List[int] = []

    for pos, normals, labels, ptr, cat_indices in test_loader:
        pos = pos.to(device)
        normals = normals.to(device)
        labels = labels.to(device)
        ptr = ptr.to(device)
        cat_indices = cat_indices.to(device)

        preds = predict_batch(model, pos, normals, ptr, cat_indices)

        B = ptr.shape[0] - 1
        for b in range(B):
            s, e = ptr[b].item(), ptr[b + 1].item()
            all_preds_orig.append(preds[s:e].cpu())
            all_labels.append(labels[s:e].cpu())
            all_cats.append(cat_indices[b].item())

    cat_miou_orig, inst_miou_orig, per_cat_orig = compute_part_miou(
        all_preds_orig, all_labels, all_cats
    )
    print(f"  Baseline cat mIoU:  {cat_miou_orig:.4f}")
    print(f"  Baseline inst mIoU: {inst_miou_orig:.4f}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Random SO(3) rotations
    # ═══════════════════════════════════════════════════════════════
    print(f"Phase 2: Testing {args.num_rotations} random SO(3) rotations...")

    rotation_results = []

    for r_idx in range(args.num_rotations):
        R = random_rotation_matrix(device)

        all_preds_rot: List[Tensor] = []
        n_exact_match = 0
        n_total_points = 0

        shape_idx = 0
        for pos, normals, labels, ptr, cat_indices in test_loader:
            pos = pos.to(device)
            normals = normals.to(device)
            ptr = ptr.to(device)
            cat_indices = cat_indices.to(device)

            # ⚠ CRITICAL: Rotate BOTH pos AND normals!
            # Failing to rotate normals creates physically impossible
            # geometry and will cause prediction collapse.
            pos_rot = pos @ R.T         # [N, 3] @ [3, 3] = [N, 3]
            normals_rot = normals @ R.T  # Normals are type-1 vectors!

            preds_rot = predict_batch(
                model, pos_rot, normals_rot, ptr, cat_indices
            )

            B = ptr.shape[0] - 1
            for b in range(B):
                s, e = ptr[b].item(), ptr[b + 1].item()
                pred_rot_b = preds_rot[s:e].cpu()
                all_preds_rot.append(pred_rot_b)

                # Compare with original prediction
                pred_orig_b = all_preds_orig[shape_idx]
                n_exact_match += (pred_rot_b == pred_orig_b).sum().item()
                n_total_points += pred_rot_b.shape[0]
                shape_idx += 1

        # mIoU under rotation
        cat_miou_rot, inst_miou_rot, _ = compute_part_miou(
            all_preds_rot, all_labels, all_cats
        )

        consistency = n_exact_match / n_total_points * 100
        miou_drop = (inst_miou_orig - inst_miou_rot) * 100  # in percentage points

        rotation_results.append({
            'cat_miou': cat_miou_rot,
            'inst_miou': inst_miou_rot,
            'consistency': consistency,
            'miou_drop': miou_drop,
        })

        print(f"  Rotation {r_idx+1:2d}: inst_mIoU={inst_miou_rot:.4f} "
              f"(drop={miou_drop:+.2f}pp) "
              f"consistency={consistency:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    avg_miou_rot = sum(r['inst_miou'] for r in rotation_results) / len(rotation_results)
    avg_drop = sum(r['miou_drop'] for r in rotation_results) / len(rotation_results)
    avg_consistency = sum(r['consistency'] for r in rotation_results) / len(rotation_results)
    max_drop = max(r['miou_drop'] for r in rotation_results)

    print(f"  Baseline inst mIoU:     {inst_miou_orig:.4f}")
    print(f"  Avg rotated inst mIoU:  {avg_miou_rot:.4f}")
    print(f"  Avg mIoU drop:          {avg_drop:+.4f} pp")
    print(f"  Max mIoU drop:          {max_drop:+.4f} pp")
    print(f"  Avg prediction match:   {avg_consistency:.2f}%")
    print()

    # Verdict
    if abs(avg_drop) < 0.5:
        print("  ✅ PASSED: Model is rotationally invariant!")
        print("     (mIoU drop < 0.5 percentage points under random SO(3))")
    elif abs(avg_drop) < 2.0:
        print("  ⚠️  MARGINAL: Small mIoU drop detected.")
        print("     This may be due to FP32 precision in multi-stage pipeline.")
    else:
        print("  ❌ FAILED: Significant mIoU drop under rotation!")
        print("     This indicates a bug in equivariance implementation.")

    # Per-category breakdown for best and worst
    print()
    print("Per-category baseline mIoU:")
    sorted_cats = sorted(per_cat_orig.items(), key=lambda x: x[1], reverse=True)
    for cat_name, miou in sorted_cats:
        n_parts = len(SEG_CLASSES[cat_name])
        bar = "█" * int(miou * 40)
        print(f"  {cat_name:15s} ({n_parts} parts): {miou:.4f} {bar}")


if __name__ == '__main__':
    main()
