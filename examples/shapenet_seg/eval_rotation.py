#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Zero-shot SO(3) rotation equivariance + mIoU evaluation.

This script serves two purposes:
    1. **Performance**: Evaluate trained model's mIoU on the test set
    2. **Equivariance proof**: Verify predictions are invariant under
       random SO(3) rotations — the core architectural guarantee

This test would CATASTROPHICALLY FAIL on:
    - PointNet, PointNet++, DGCNN, PointNeXt (without SO(3) augmentation)
    - Any method using spatial coordinates as scalar features

Protocol:
    1. Load trained model (all constructor args from checkpoint)
    2. Phase 1: Evaluate canonical pose → baseline mIoU
    3. Phase 2: For each random SO(3) rotation R:
       a. Rotate BOTH pos AND normals: pos' = pos @ R^T, n' = n @ R^T
          ⚠ LANDMINE: normals MUST be co-rotated! Rotating pos without
          rotating normals creates physically impossible geometry.
       b. Predict on rotated input → compare with canonical predictions
       c. Report: consistency %, mIoU drop
    4. Per-category breakdown

Usage:
    python examples/shapenet_seg/eval_rotation.py \\
        --checkpoint checkpoints/shapenet_seg/best_model.pt \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --num_rotations 10

    # Quick sanity (fewer rotations):
    python examples/shapenet_seg/eval_rotation.py \\
        --checkpoint checkpoints/shapenet_seg/best_model.pt \\
        --data_root ../data/shapenetpart_hdf5_2048 \\
        --num_rotations 3 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

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

    Uses float64 for numerical precision, returns float32.

    Returns:
        R: [3, 3] float32, proper rotation (det = +1)
    """
    M = torch.randn(3, 3, dtype=torch.float64)
    Q, R_diag = torch.linalg.qr(M)
    signs = torch.sign(torch.diag(R_diag))
    Q = Q * signs.unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.float().to(device)


def build_model_from_checkpoint(
    ckpt: dict,
    device: torch.device,
) -> SE3PartSegNet:
    """Reconstruct model from checkpoint, reading ALL constructor args.

    The checkpoint stores ``vars(args)`` from train.py. This function
    maps those CLI arg names to ``SE3PartSegNet.__init__`` parameters,
    handling naming differences (e.g. ``--no_normals`` → ``use_normals``).

    Args:
        ckpt: Loaded checkpoint dict with 'args' and 'model_state_dict'
        device: Target device

    Returns:
        Model with loaded weights in eval mode
    """
    a = ckpt['args']

    model = SE3PartSegNet(
        in_channels=a.get('in_channels', 1),
        hidden_scalar=a['hidden_scalar'],
        hidden_vector=a['hidden_vector'],
        hidden_type2=a.get('hidden_type2', 0),
        num_stages=a.get('num_stages', 3),
        layers_per_stage=a.get('layers_per_stage', 2),
        pool_ratio=a.get('pool_ratio', 0.25),
        use_normals=not a.get('no_normals', False),
        head_hidden=a.get('head_hidden', 128),
        gate_mode=a.get('gate_mode', 'scalar'),
        use_self_tp=a.get('use_self_tp', False),
        use_bottleneck_attn=a.get('use_bottleneck_attn', False),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def predict_batch(
    model: SE3PartSegNet,
    pos: Tensor,
    normals: Tensor,
    ptr: Tensor,
    cat_indices: Tensor,
) -> Tensor:
    """Run model and return per-point predictions.

    Args:
        model: SE3PartSegNet in eval mode
        pos: [N_total, 3] packed positions
        normals: [N_total, 3] packed normals (l=1 vector features)
        ptr: [B+1] CSR batch offsets
        cat_indices: [B] category indices

    Returns:
        preds: [N_total] int64 — predicted part labels
    """
    logits = model(pos, ptr, cat_indices, normals=normals)
    return logits.argmax(dim=1)


@torch.no_grad()
def evaluate_miou(
    model: SE3PartSegNet,
    loader: DataLoader,
    device: torch.device,
    rotate: bool = False,
    R: Tensor | None = None,
) -> Tuple[float, float, Dict[str, float], List[Tensor], List[Tensor], List[int]]:
    """Evaluate mIoU, optionally under SO(3) rotation.

    Args:
        model: SE3PartSegNet in eval mode
        loader: Test DataLoader
        device: Target device
        rotate: If True, apply rotation R to pos and normals
        R: [3, 3] rotation matrix (required if rotate=True)

    Returns:
        cat_miou, inst_miou, per_cat_miou,
        all_preds (list of [N] tensors),
        all_labels (list of [N] tensors),
        all_cats (list of int)
    """
    all_preds: List[Tensor] = []
    all_labels: List[Tensor] = []
    all_cats: List[int] = []
    total_loss = 0.0
    n_batches = 0

    for pos, normals, labels, ptr, cat_indices in loader:
        pos = pos.to(device, non_blocking=True)
        normals = normals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ptr = ptr.to(device, non_blocking=True)
        cat_indices = cat_indices.to(device, non_blocking=True)

        if rotate and R is not None:
            # ⚠ CRITICAL: Rotate BOTH pos AND normals!
            # Normals are type-1 vectors: n' = R @ n
            pos = pos @ R.T         # [N, 3] @ [3, 3] = [N, 3]
            normals = normals @ R.T  # Co-rotate!

        logits = model(pos, ptr, cat_indices, normals=normals)
        loss = nn.functional.cross_entropy(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        B = ptr.shape[0] - 1
        for b in range(B):
            s, e = ptr[b].item(), ptr[b + 1].item()
            all_preds.append(preds[s:e].cpu())
            all_labels.append(labels[s:e].cpu())
            all_cats.append(cat_indices[b].item())

    cat_miou, inst_miou, per_cat = compute_part_miou(
        all_preds, all_labels, all_cats
    )
    return cat_miou, inst_miou, per_cat, all_preds, all_labels, all_cats


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Zero-shot SO(3) rotation test + mIoU evaluation'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to shapenetpart_hdf5_2048 or ShapeNetPart')
    parser.add_argument('--num_rotations', type=int, default=10,
                        help='Number of random SO(3) rotations to test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu'
    )

    print("=" * 70)
    print("  SE3PartSegNet — Evaluation & SO(3) Equivariance Test")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data root: {args.data_root}")
    print(f"  Rotations: {args.num_rotations}")
    print()

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")
    print(f"  Trained epoch: {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint inst_mIoU: {ckpt.get('inst_miou', 0):.4f}")
    print(f"  Checkpoint cat_mIoU:  {ckpt.get('cat_miou', 0):.4f}")

    # Print key model config from checkpoint
    a = ckpt['args']
    print(f"  Config: scalar={a['hidden_scalar']} vector={a['hidden_vector']} "
          f"type2={a.get('hidden_type2', 0)} stages={a.get('num_stages', 3)} "
          f"gate={a.get('gate_mode', 'scalar')} self_tp={a.get('use_self_tp', False)}")
    print()

    # ── Dataset ──
    test_dataset = ShapeNetPartDataset(
        args.data_root, split='test', normalize=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  Test set: {len(test_dataset)} shapes, "
          f"{len(test_loader)} batches")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline (canonical pose)
    # ═══════════════════════════════════════════════════════════════
    print("Phase 1: Baseline evaluation (canonical pose)")
    print("-" * 70)

    t0 = time.time()
    cat_miou_orig, inst_miou_orig, per_cat_orig, preds_orig, labels_all, cats_all = \
        evaluate_miou(model, test_loader, device)
    dt_baseline = time.time() - t0

    print(f"  Category mIoU:  {cat_miou_orig:.4f}")
    print(f"  Instance mIoU:  {inst_miou_orig:.4f}")
    print(f"  Time: {dt_baseline:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Random SO(3) rotations
    # ═══════════════════════════════════════════════════════════════
    print(f"Phase 2: Testing {args.num_rotations} random SO(3) rotations")
    print("-" * 70)

    rotation_results = []

    for r_idx in range(args.num_rotations):
        R = random_rotation_matrix(device)

        t0 = time.time()
        cat_miou_rot, inst_miou_rot, _, preds_rot, _, _ = \
            evaluate_miou(model, test_loader, device, rotate=True, R=R)
        dt_rot = time.time() - t0

        # Prediction consistency: exact label match across all points
        n_exact_match = 0
        n_total_points = 0
        for pred_orig, pred_rot in zip(preds_orig, preds_rot):
            n_exact_match += (pred_orig == pred_rot).sum().item()
            n_total_points += pred_orig.shape[0]

        consistency = n_exact_match / n_total_points * 100
        miou_drop = (inst_miou_orig - inst_miou_rot) * 100  # percentage points

        rotation_results.append({
            'cat_miou': cat_miou_rot,
            'inst_miou': inst_miou_rot,
            'consistency': consistency,
            'miou_drop_pp': miou_drop,
        })

        print(f"  R{r_idx+1:2d}: inst_mIoU={inst_miou_rot:.4f} "
              f"(Δ={miou_drop:+.2f}pp) "
              f"consistency={consistency:.2f}% "
              f"({dt_rot:.1f}s)")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)

    avg_miou_rot = sum(r['inst_miou'] for r in rotation_results) / len(rotation_results)
    avg_drop = sum(r['miou_drop_pp'] for r in rotation_results) / len(rotation_results)
    avg_consistency = sum(r['consistency'] for r in rotation_results) / len(rotation_results)
    max_drop = max(r['miou_drop_pp'] for r in rotation_results)
    min_consistency = min(r['consistency'] for r in rotation_results)

    print(f"  Baseline inst mIoU:     {inst_miou_orig:.4f}")
    print(f"  Baseline cat mIoU:      {cat_miou_orig:.4f}")
    print(f"  Avg rotated inst mIoU:  {avg_miou_rot:.4f}")
    print(f"  Avg mIoU drop:          {avg_drop:+.4f} pp")
    print(f"  Max mIoU drop:          {max_drop:+.4f} pp")
    print(f"  Avg prediction match:   {avg_consistency:.2f}%")
    print(f"  Min prediction match:   {min_consistency:.2f}%")
    print()

    # ── Verdict ──
    if abs(avg_drop) < 0.5:
        print("  ✅ PASSED: Model is rotationally invariant!")
        print("     (mIoU drop < 0.5 percentage points under random SO(3))")
    elif abs(avg_drop) < 2.0:
        print("  ⚠️  MARGINAL: Small mIoU drop detected.")
        print("     May be FP32 precision in multi-stage pipeline.")
        print("     Consider testing with --batch_size 1 to rule out "
              "batch-norm effects.")
    else:
        print("  ❌ FAILED: Significant mIoU drop under rotation!")
        print("     This indicates a bug in equivariance implementation.")

    # ── Per-category breakdown ──
    print()
    print("  Per-category baseline mIoU:")
    print("  " + "-" * 55)
    sorted_cats = sorted(
        per_cat_orig.items(), key=lambda x: x[1], reverse=True
    )
    for cat_name, miou in sorted_cats:
        n_parts = len(SEG_CLASSES[cat_name])
        bar = "█" * int(miou * 40)
        print(f"    {cat_name:15s} ({n_parts} parts): {miou:.4f} {bar}")

    # ── Skip scale diagnostics ──
    print()
    print("  Skip scale diagnostics (learnable residual):")
    bb = model.backbone
    for si in range(bb.num_stages):
        for bi, blk in enumerate(bb.encoder_stages[si].blocks):
            d = blk.get_skip_diagnostics()
            if d:
                print(f"    e{si}b{bi}: "
                      f"s={d['skip_s_mean']:.4f} "
                      f"v={d.get('skip_v_mean', 0):.4f} "
                      f"t2={d.get('skip_t2_mean', 0):.4f}")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
