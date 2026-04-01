#!/usr/bin/env python3
"""Robustness evaluation for the GeoEmbodied registration model.

Evaluates a trained checkpoint under two challenging scenarios:

  Task A — Symmetric category generalization:
    Tests on the 8 excluded symmetric categories (bottle, bowl, etc.).
    Since RRE is ambiguous for rotationally symmetric objects, we report
    Chamfer Distance as the primary metric. A low CD proves the model
    achieves correct geometric alignment despite rotational ambiguity.

  Task B — Low-overlap stress test:
    Evaluates at partial_ratio=0.5 (25% true overlap after bilateral crop)
    and optionally with increased noise. Tests the robustness boundary of
    the Sinkhorn dustbin mechanism under extreme conditions.

Usage:
    python examples/registration/evaluate_robustness.py \
        --checkpoint checkpoints/registration_best.pt \
        --cache data/modelnet40_4096.pt

    # With extra noise stress:
    python examples/registration/evaluate_robustness.py \
        --checkpoint checkpoints/registration_best.pt \
        --cache data/modelnet40_4096.pt \
        --noise_std 0.05
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from dataset_modelnet40 import ModelNet40Registration, SYMMETRIC_CATEGORIES
from train_registration import registration_collate_fn

from geoembodied.nn import SE3Net
from geoembodied.nn.registration_model import GeoRegistrationModel
from geoembodied.functional.numeric_safe import lie_group_precision
from geoembodied.functional.chamfer import chamfer_distance
from geoembodied.data.batch import to_dense_batch


# ═══════════════════════════════════════════════════════════════════
# Geodesic rotation error (per-sample, in degrees)
# ═══════════════════════════════════════════════════════════════════

def rotation_error_deg(R_pred: Tensor, R_gt: Tensor) -> Tensor:
    """Per-sample geodesic rotation error in degrees.

    Args:
        R_pred: [B, 3, 3], representation: SO(3)
        R_gt: [B, 3, 3], representation: SO(3)

    Returns:
        errors: [B] rotation errors in degrees
    """
    # dR = R_pred @ R_gt^T
    dR = torch.bmm(R_pred, R_gt.transpose(1, 2))
    # trace(dR) = 1 + 2*cos(θ) → θ = acos((tr-1)/2)
    trace = dR.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B]
    cos_angle = ((trace - 1) / 2).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle_rad = torch.acos(cos_angle)  # [B]
    return angle_rad * (180.0 / math.pi)


def translation_error(t_pred: Tensor, t_gt: Tensor) -> Tensor:
    """Per-sample translation error (L2 norm).

    Args:
        t_pred: [B, 3]
        t_gt: [B, 3]

    Returns:
        errors: [B]
    """
    return (t_pred - t_gt).norm(dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_dataset(
    model: GeoRegistrationModel,
    dataset: ModelNet40Registration,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 2,
    tag: str = "",
) -> dict:
    """Evaluate model on a dataset, returning per-category metrics.

    Returns:
        Dict with:
            'overall': {metric_name: value}
            'per_category': {cat_name: {metric_name: value}}
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=registration_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Accumulate per-category metrics
    cat_rot_errors: dict[int, list[float]] = defaultdict(list)
    cat_trans_errors: dict[int, list[float]] = defaultdict(list)
    cat_cd_values: dict[int, list[float]] = defaultdict(list)

    n_batches = len(loader)
    for i, (batch_src, batch_tgt, gt) in enumerate(loader):
        batch_src = batch_src.to(device, non_blocking=True)
        batch_tgt = batch_tgt.to(device, non_blocking=True)
        R_gt = gt["R"].to(device, non_blocking=True)
        t_gt = gt["t"].to(device, non_blocking=True)
        labels = gt["labels"]  # [B], int, stays on CPU

        with lie_group_precision():
            out = model(batch_src, batch_tgt)

        B = R_gt.shape[0]

        # Per-sample rotation and translation errors
        rot_errs = rotation_error_deg(out["R"], R_gt)  # [B]
        trans_errs = translation_error(out["t"], t_gt)  # [B]

        # Chamfer distance per sample
        pos_src_d = out["pos_src"]  # [B, N, 3]
        pos_tgt_d = out["pos_tgt"]  # [B, M, 3]
        mask_src = out["mask_src"]  # [B, N]

        # Align source with predicted pose
        aligned_src = torch.bmm(
            out["R"], pos_src_d.transpose(1, 2),
        ).transpose(1, 2) + out["t"].unsqueeze(1)  # [B, N, 3]

        # Per-sample CD (bidirectional for fairness)
        for b in range(B):
            src_b = aligned_src[b][mask_src[b]]  # [N_valid, 3]
            tgt_b = pos_tgt_d[b]  # [M, 3] — may have padding
            if out.get("mask_tgt") is not None:
                tgt_b = tgt_b[out["mask_tgt"][b]]
            cd_val = chamfer_distance(
                src_b, tgt_b, bidirectional=True,
            ).item()

            label = labels[b].item()
            cat_rot_errors[label].append(rot_errs[b].item())
            cat_trans_errors[label].append(trans_errs[b].item())
            cat_cd_values[label].append(cd_val)

        if (i + 1) % 20 == 0:
            print(f"  [{tag}] {i+1}/{n_batches} batches processed", end="\r")

    print(f"  [{tag}] {n_batches}/{n_batches} batches processed")

    # Aggregate
    cat_names = dataset.category_names
    per_cat = {}
    all_rot = []
    all_trans = []
    all_cd = []

    for label_id in sorted(cat_rot_errors.keys()):
        cat_name = cat_names[label_id]
        rots = cat_rot_errors[label_id]
        trans = cat_trans_errors[label_id]
        cds = cat_cd_values[label_id]

        all_rot.extend(rots)
        all_trans.extend(trans)
        all_cd.extend(cds)

        per_cat[cat_name] = {
            "RRE_mean": sum(rots) / len(rots),
            "RRE_median": sorted(rots)[len(rots) // 2],
            "TE_mean": sum(trans) / len(trans),
            "CD_mean": sum(cds) / len(cds),
            "count": len(rots),
        }

    overall = {
        "RRE_mean": sum(all_rot) / len(all_rot) if all_rot else 0,
        "RRE_median": sorted(all_rot)[len(all_rot) // 2] if all_rot else 0,
        "TE_mean": sum(all_trans) / len(all_trans) if all_trans else 0,
        "CD_mean": sum(all_cd) / len(all_cd) if all_cd else 0,
        "count": len(all_rot),
    }

    return {"overall": overall, "per_category": per_cat}


def print_results(results: dict, title: str, is_symmetric: bool = False):
    """Pretty-print evaluation results."""
    print()
    print(f"{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

    overall = results["overall"]
    per_cat = results["per_category"]

    # Overall summary
    print(f"\n  Overall ({overall['count']} samples):")
    if not is_symmetric:
        print(f"    RRE (mean):   {overall['RRE_mean']:.2f}°")
        print(f"    RRE (median): {overall['RRE_median']:.2f}°")
        print(f"    TE (mean):    {overall['TE_mean']:.4f}")
    print(f"    CD (mean):    {overall['CD_mean']:.6f}")

    # Per-category table
    print(f"\n  {'Category':<16s} {'N':>5s}", end="")
    if not is_symmetric:
        print(f" {'RRE°':>8s} {'RRE_med':>8s} {'TE':>8s}", end="")
    print(f" {'CD':>10s}")
    print(f"  {'─' * 16} {'─' * 5}", end="")
    if not is_symmetric:
        print(f" {'─' * 8} {'─' * 8} {'─' * 8}", end="")
    print(f" {'─' * 10}")

    for cat_name in sorted(per_cat.keys()):
        m = per_cat[cat_name]
        print(f"  {cat_name:<16s} {m['count']:>5d}", end="")
        if not is_symmetric:
            print(f" {m['RRE_mean']:>8.2f} {m['RRE_median']:>8.2f}"
                  f" {m['TE_mean']:>8.4f}", end="")
        print(f" {m['CD_mean']:>10.6f}")

    # Success rate (if not symmetric)
    if not is_symmetric:
        success_2 = sum(1 for r in _flatten_rot(results) if r < 2.0)
        success_5 = sum(1 for r in _flatten_rot(results) if r < 5.0)
        total = overall["count"]
        print(f"\n  Success rates:")
        print(f"    RRE < 2°:  {success_2}/{total} ({success_2/total*100:.1f}%)")
        print(f"    RRE < 5°:  {success_5}/{total} ({success_5/total*100:.1f}%)")


def _flatten_rot(results: dict) -> list:
    """Flatten all per-category RRE values."""
    # Reconstruct from per_cat
    all_rre = []
    for m in results["per_category"].values():
        all_rre.extend([m["RRE_mean"]] * m["count"])
    return all_rre


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> GeoRegistrationModel:
    """Load model from checkpoint.

    The checkpoint contains 'args' dict with all model hyperparameters.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])

    backbone = SE3Net(
        hidden_scalar=args.hidden_scalar,
        hidden_vector=args.hidden_vector,
        num_layers=args.num_layers,
        radius=args.radius,
        max_num_neighbors=args.max_neighbors,
    )
    model = GeoRegistrationModel(
        backbone=backbone,
        cross_attention_layers=args.cross_attn_layers,
        descriptor_dim=args.descriptor_dim,
        sinkhorn_iters=args.sinkhorn_iters,
        share_backbone=True,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val = ckpt.get("val_metrics", {})
    val_rot = val.get("loss/rotation_deg", "?")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {epoch}, val_rot: {val_rot}°")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    return model


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Robustness evaluation: symmetric objects + low overlap",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/registration_best.pt")
    parser.add_argument("--cache", type=str,
                        default="data/modelnet40_4096.pt")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    # Task B: stress test parameters
    parser.add_argument("--stress_ratios", type=float, nargs="+",
                        default=[0.7, 0.5, 0.3],
                        help="Partial overlap ratios to test")
    parser.add_argument("--noise_std", type=float, default=0.01,
                        help="Gaussian noise stddev on target cloud")

    # Skip flags
    parser.add_argument("--skip_symmetric", action="store_true",
                        help="Skip Task A (symmetric categories)")
    parser.add_argument("--skip_stress", action="store_true",
                        help="Skip Task B (overlap stress test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # ══════════════════════════════════════════════════════════════
    # Task A: Symmetric category evaluation
    # ══════════════════════════════════════════════════════════════
    if not args.skip_symmetric:
        print("\n" + "=" * 70)
        print("  TASK A: Symmetric Category Generalization")
        print("  Categories: " + ", ".join(SYMMETRIC_CATEGORIES))
        print("=" * 70)

        sym_ds = ModelNet40Registration(
            cache_path=args.cache,
            num_points=args.num_points,
            split="test",
            partial_ratio=0.7,  # Same as training
            noise_std=0.01,
            exclude_symmetric=False,
            categories=SYMMETRIC_CATEGORIES,  # ONLY symmetric!
        )

        if len(sym_ds) > 0:
            sym_results = evaluate_dataset(
                model, sym_ds, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tag="SYM",
            )
            print_results(
                sym_results,
                "Task A: Symmetric Objects (CD is the valid metric, NOT RRE)",
                is_symmetric=True,
            )

            # Also evaluate the NORMAL (non-symmetric) categories for comparison
            normal_ds = ModelNet40Registration(
                cache_path=args.cache,
                num_points=args.num_points,
                split="test",
                partial_ratio=0.7,
                noise_std=0.01,
                exclude_symmetric=True,  # only asymmetric
            )
            normal_results = evaluate_dataset(
                model, normal_ds, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tag="ASYM",
            )
            print_results(
                normal_results,
                "Task A (Reference): Non-Symmetric Objects",
                is_symmetric=False,
            )

            # Compare CDs
            sym_cd = sym_results["overall"]["CD_mean"]
            asym_cd = normal_results["overall"]["CD_mean"]
            print(f"\n  ── Symmetry verdict ──")
            print(f"  Symmetric CD:     {sym_cd:.6f}")
            print(f"  Non-symmetric CD: {asym_cd:.6f}")
            cd_ratio = sym_cd / max(asym_cd, 1e-9)
            if cd_ratio < 3.0:
                print(f"  Ratio: {cd_ratio:.1f}x → ✓ Model handles symmetric "
                      f"objects well (geometric alignment preserved)")
            else:
                print(f"  Ratio: {cd_ratio:.1f}x → ⚠ Symmetric objects have "
                      f"significantly worse alignment")
        else:
            print("  ⚠ No symmetric shapes found in test split!")

    # ══════════════════════════════════════════════════════════════
    # Task B: Overlap stress test
    # ══════════════════════════════════════════════════════════════
    if not args.skip_stress:
        print("\n" + "=" * 70)
        print("  TASK B: Overlap Robustness Stress Test")
        print(f"  Overlap ratios: {args.stress_ratios}")
        print(f"  Noise σ: {args.noise_std}")
        print("=" * 70)

        stress_summary = []

        for ratio in args.stress_ratios:
            eff_overlap = ratio ** 2  # bilateral cropping
            print(f"\n  ── partial_ratio={ratio:.1f} "
                  f"(effective overlap ≈ {eff_overlap:.0%}) ──")

            stress_ds = ModelNet40Registration(
                cache_path=args.cache,
                num_points=args.num_points,
                split="test",
                partial_ratio=ratio,
                noise_std=args.noise_std,
                exclude_symmetric=True,  # fair comparison
            )

            results = evaluate_dataset(
                model, stress_ds, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tag=f"R={ratio}",
            )
            print_results(
                results,
                f"Task B: partial_ratio={ratio:.1f}, "
                f"noise_σ={args.noise_std}, "
                f"overlap≈{eff_overlap:.0%}",
                is_symmetric=False,
            )
            stress_summary.append({
                "ratio": ratio,
                "overlap": eff_overlap,
                "RRE": results["overall"]["RRE_mean"],
                "RRE_med": results["overall"]["RRE_median"],
                "TE": results["overall"]["TE_mean"],
                "CD": results["overall"]["CD_mean"],
            })

        # Summary table
        print(f"\n{'═' * 70}")
        print(f"  STRESS TEST SUMMARY")
        print(f"{'═' * 70}")
        print(f"  {'Ratio':>6s} {'Overlap':>8s} {'RRE°':>8s} "
              f"{'RRE_med':>8s} {'TE':>8s} {'CD':>10s}")
        print(f"  {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")
        for s in stress_summary:
            print(f"  {s['ratio']:>6.1f} {s['overlap']:>7.0%} "
                  f"{s['RRE']:>8.2f} {s['RRE_med']:>8.2f} "
                  f"{s['TE']:>8.4f} {s['CD']:>10.6f}")

        # Verdict
        print()
        best = stress_summary[0]  # 0.7 ratio
        worst = stress_summary[-1]  # lowest ratio
        if worst["RRE"] < 2.0:
            print(f"  ✓ ENGINEERING COMPLETE: {worst['RRE']:.2f}° at "
                  f"{worst['ratio']:.0%} overlap → model is production-ready")
        elif worst["RRE"] < 5.0:
            print(f"  ◐ ACCEPTABLE: {worst['RRE']:.2f}° at "
                  f"{worst['ratio']:.0%} overlap → good robustness, "
                  f"may benefit from overlap-aware fine-tuning")
        else:
            print(f"  ⚠ DEGRADED: {worst['RRE']:.2f}° at "
                  f"{worst['ratio']:.0%} overlap → "
                  f"model reaches its capacity limit")
            # Find the boundary
            for s in stress_summary:
                if s["RRE"] < 5.0:
                    print(f"  → Robustness boundary: ~{s['ratio']:.0%} "
                          f"partial_ratio ({s['overlap']:.0%} overlap)")
                    break

    print(f"\n{'═' * 70}")
    print(f"  Evaluation complete.")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
