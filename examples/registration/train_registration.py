#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Training script for GeoRegistrationModel.

Usage:
    # Synthetic data (quick validation, no download needed):
    python examples/registration/train_registration.py \
        --dataset synthetic --epochs 50 --batch_size 4

    # ModelNet40 (requires preprocessing first):
    python examples/registration/train_registration.py \
        --dataset modelnet40 --cache data/modelnet40_4096.pt \
        --epochs 100 --batch_size 8

    # With PyTorch Profiler → TensorBoard:
    python examples/registration/train_registration.py \
        --dataset synthetic --profile --profile_epochs 1

    # View profiler output:
    tensorboard --logdir runs/

Features:
    - SE(3)-equivariant registration with shared SE3Net backbone
    - Geodesic rotation loss + translation L2 loss
    - PyTorch Profiler + TensorBoard integration (--profile)
    - TensorBoard scalar logging (loss, rotation error, translation error)
    - Automatic TF32 guard for Lie group precision
    - Gradient clipping (prevents SVD backward explosion)
    - Mixed precision training (AMP) with fp32 overrides for Lie ops
    - Warmup + cosine LR schedule
"""

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from geoembodied.data.batch import (
    PointCloudBatch,
    collate_point_clouds,
    to_dense_batch,
)
from geoembodied.nn import SE3Net
from geoembodied.nn.registration_model import GeoRegistrationModel
from geoembodied.functional.numeric_safe import lie_group_precision
from geoembodied.functional.chamfer import chamfer_distance


# ═══════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════


def geodesic_rotation_loss(R_pred: Tensor, R_gt: Tensor) -> Tensor:
    """Geodesic distance on SO(3): d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2).

    Args:
        R_pred: Predicted rotation [B, 3, 3], representation: SO(3)
        R_gt: Ground truth rotation [B, 3, 3], representation: SO(3)

    Returns:
        Mean geodesic angle error in radians, scalar
    """
    eps = 1e-7
    # R_diff = R_gt^T @ R_pred
    R_diff = torch.bmm(R_gt.transpose(1, 2), R_pred)  # [B, 3, 3]
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B]

    # Clamp for numerical safety (Rule 4)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos_angle)  # [B], radians

    return angle.mean()


def registration_loss(
    R_pred: Tensor,
    t_pred: Tensor,
    R_gt: Tensor,
    t_gt: Tensor,
    pos_src: Optional[Tensor] = None,
    pos_tgt: Optional[Tensor] = None,
    mask_src: Optional[Tensor] = None,
    assignment: Optional[Tensor] = None,
    correspondences: Optional[Tensor] = None,
    w_rot: float = 1.0,
    w_trans: float = 1.0,
    w_cd: float = 0.0,
    w_corr: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combined registration loss with optional CD and correspondence supervision.

    Performance note: all metrics are collected as detached GPU tensors
    and batch-converted to Python floats via a single ``.item()`` sync
    at the end. This eliminates mid-computation GPU pipeline stalls
    (previously 5-6 syncs per step → ~30ms overhead on RTX 4080).

    Args:
        R_pred: Predicted rotation [B, 3, 3]
        t_pred: Predicted translation [B, 3]
        R_gt: Ground truth rotation [B, 3, 3]
        t_gt: Ground truth translation [B, 3]
        pos_src: Source positions for CD [B, N, 3] (needed if w_cd > 0)
        pos_tgt: Target positions for CD [B, M, 3] (needed if w_cd > 0)
        mask_src: Source mask [B, N] (for partial overlap CD/corr)
        assignment: Sinkhorn assignment [B, N, M+1] with dustbin column
            (needed if w_corr > 0)
        correspondences: GT source→target index [B, N], -1 for outliers
            (needed if w_corr > 0)
        w_rot: Rotation loss weight
        w_trans: Translation loss weight
        w_cd: Chamfer Distance loss weight (0 = disabled)
        w_corr: Correspondence supervision weight (0 = disabled)

    Returns:
        loss: Weighted combined loss (scalar)
        metrics: Dict with individual detached loss tensors (GPU, no sync).
            Call ``.item()`` on values in the caller after ``backward()``.
    """
    l_rot = geodesic_rotation_loss(R_pred, R_gt)
    l_trans = F.mse_loss(t_pred, t_gt)

    loss = w_rot * l_rot + w_trans * l_trans

    # Collect metrics as detached tensors — ZERO GPU syncs here.
    # Caller converts to Python floats AFTER loss.backward().
    _m: dict[str, Tensor] = {
        "loss/rotation_rad": l_rot.detach(),
        "loss/translation_mse": l_trans.detach(),
    }

    # Optional Chamfer Distance auxiliary loss.
    if w_cd > 0 and pos_src is not None and pos_tgt is not None:
        aligned_src = torch.bmm(
            R_pred, pos_src.transpose(1, 2),
        ).transpose(1, 2) + t_pred.unsqueeze(1)  # [B, N, 3]
        l_cd = chamfer_distance(
            aligned_src, pos_tgt,
            mask_a=mask_src, bidirectional=False,
        )
        loss = loss + w_cd * l_cd
        _m["loss/chamfer"] = l_cd.detach()

    # Optional Correspondence supervision.
    if w_corr > 0 and assignment is not None and correspondences is not None:
        if mask_src is not None:
            corr_mask = mask_src
        else:
            corr_mask = torch.ones(
                assignment.shape[:2], dtype=torch.bool,
                device=assignment.device,
            )
        l_corr = correspondence_loss(
            assignment, correspondences, corr_mask,
            has_dustbin=True,
        )
        loss = loss + w_corr * l_corr
        _m["loss/correspondence"] = l_corr.detach()

    _m["loss/total"] = loss.detach()

    return loss, _m


def correspondence_loss(
    assignment: Tensor,
    correspondences: Tensor,
    mask: Tensor,
    has_dustbin: bool = True,
) -> Tensor:
    """Cross-entropy supervision on Sinkhorn assignment with dustbin routing.

    For each valid source point, encourages the assignment matrix to place
    high probability at the ground-truth target index. Outlier source points
    (``correspondences=-1``) are routed to the dustbin column, teaching
    the network to reject non-overlapping regions.

    Designed for future compatibility with real datasets where GT
    correspondences are generated from $R_{gt}, t_{gt}$ + nearest-neighbor
    thresholding (no manual annotation needed).

    Args:
        assignment: Sinkhorn output [B, N, M+1] (with dustbin) or [B, N, M].
            Last column is the dustbin when has_dustbin=True.
        correspondences: GT target index per source point [B, N].
            Valid indices in [0, M). Value -1 marks outliers that should
            be routed to the dustbin column.
        mask: Boolean mask [B, N]. True = real source point, False = padding.
        has_dustbin: Whether assignment has a dustbin column.

    Returns:
        Negative log-likelihood loss (scalar).
    """
    M_plus = assignment.shape[-1]  # M+1 if dustbin, M otherwise

    # Build GT target indices, routing outliers to dustbin
    gt_idx = correspondences.clone()  # [B, N]
    outlier_mask = gt_idx < 0

    if has_dustbin:
        dustbin_col = M_plus - 1
        gt_idx[outlier_mask] = dustbin_col
    else:
        # Without dustbin, exclude outliers from loss entirely
        gt_idx[outlier_mask] = 0  # Dummy index (masked out below)
        mask = mask & ~outlier_mask

    # Negative log-likelihood with proper row normalization.
    #
    # CRITICAL: Sinkhorn's last iteration is column normalization, so
    # assignment rows do NOT sum to 1.0 (typically ~1.43 for 358×512).
    # Raw log(assignment) is NOT a proper log-probability!
    #
    # Fix: row-normalize in log-space to get proper categorical distribution:
    #   log_prob[i, :] = log(A[i, :]) - logsumexp(log(A[i, :]))
    # This is equivalent to softmax in log-space, numerically stable.
    log_A = torch.log(assignment.clamp(min=1e-8))  # [B, N, M+1]
    log_prob = log_A - torch.logsumexp(log_A, dim=-1, keepdim=True)  # [B, N, M+1]

    corr_log_prob = torch.gather(
        log_prob, dim=2, index=gt_idx.unsqueeze(-1),
    ).squeeze(-1)  # [B, N]

    valid = mask.float()
    loss = -(corr_log_prob * valid).sum() / valid.sum().clamp(min=1)
    return loss


# ═══════════════════════════════════════════════════════════════════
# Collate function
# ═══════════════════════════════════════════════════════════════════


def registration_collate_fn(
    samples: list[dict],
) -> tuple[PointCloudBatch, PointCloudBatch, dict[str, Tensor]]:
    """Collate registration samples into paired PointCloudBatch.

    Bridges dataset.__getitem__ (raw dicts) to model.forward (PointCloudBatch).
    Graph construction happens inside SE3Net, not here.

    Args:
        samples: List of dicts from dataset.__getitem__

    Returns:
        batch_src: Source PointCloudBatch
        batch_tgt: Target PointCloudBatch
        gt: Dict with ground truth tensors (batched)
    """
    src_clouds = [{"pos": s["source"]} for s in samples]
    tgt_clouds = [{"pos": s["target"]} for s in samples]

    batch_src = collate_point_clouds(src_clouds)
    batch_tgt = collate_point_clouds(tgt_clouds)

    # Pad correspondences to max source size.
    # Padding positions get -1 (will be masked out or routed to dustbin).
    max_n_src = max(s["correspondences"].shape[0] for s in samples)
    B = len(samples)
    corr = torch.full((B, max_n_src), -1, dtype=torch.long)
    for b, s in enumerate(samples):
        n = s["correspondences"].shape[0]
        corr[b, :n] = s["correspondences"]

    gt = {
        "R": torch.stack([s["gt_rotation"] for s in samples]),     # [B, 3, 3]
        "t": torch.stack([s["gt_translation"] for s in samples]),  # [B, 3]
        "correspondences": corr,  # [B, N_max_src]
        "labels": torch.tensor(
            [s.get("label", -1) for s in samples], dtype=torch.long,
        ),  # [B]
    }

    return batch_src, batch_tgt, gt


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: GeoRegistrationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    writer = None,
    grad_clip: float = 1.0,
    w_cd: float = 0.0,
    w_corr: float = 0.0,
    profiler_ctx=None,
) -> dict[str, float]:
    """Train one epoch.

    Args:
        model: GeoRegistrationModel
        loader: DataLoader yielding (batch_src, batch_tgt, gt)
        optimizer: Optimizer (Adam/AdamW)
        scheduler: LR scheduler (optional)
        device: CUDA/CPU device
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter (optional)
        grad_clip: Max gradient norm
        profiler_ctx: PyTorch profiler context (optional)

    Returns:
        Dict of averaged metrics over the epoch
    """
    model.train()
    epoch_metrics: dict[str, list[float]] = {}
    global_step_base = epoch * len(loader)
    data_time_total = 0.0
    compute_time_total = 0.0

    t_data_start = time.time()
    for step, (batch_src, batch_tgt, gt) in enumerate(loader):
        data_time_total += time.time() - t_data_start
        t_compute_start = time.time()

        global_step = global_step_base + step

        # Move to device (non_blocking for overlap with pin_memory)
        batch_src = batch_src.to(device, non_blocking=True)
        batch_tgt = batch_tgt.to(device, non_blocking=True)
        R_gt = gt["R"].to(device, non_blocking=True)
        t_gt = gt["t"].to(device, non_blocking=True)
        corr_gt = gt["correspondences"].to(device, non_blocking=True)

        # Forward — TF32 guard active for Lie group ops
        with lie_group_precision():
            out = model(batch_src, batch_tgt)

        # Loss — returns detached tensor metrics (ZERO GPU syncs)
        loss, metrics_tensors = registration_loss(
            out["R"], out["t"], R_gt, t_gt,
            pos_src=out["pos_src"],
            pos_tgt=out["pos_tgt"],
            mask_src=out["mask_src"],
            assignment=out["assignment"],
            correspondences=corr_gt,
            w_cd=w_cd,
            w_corr=w_corr,
        )

        # Backward + gradient clip + optimizer
        # All GPU work queued asynchronously — no sync until metrics read.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        compute_time_total += time.time() - t_compute_start

        # ── Metrics sync: convert GPU tensors → Python floats ──
        # This is the ONLY GPU sync per step. It happens AFTER all
        # GPU work (fwd+bwd+step) is queued, so the sync cost is
        # hidden behind the GPU pipeline drain.
        metrics: dict[str, float] = {
            k: v.item() for k, v in metrics_tensors.items()
        }
        metrics["loss/rotation_deg"] = math.degrees(
            metrics["loss/rotation_rad"]
        )

        # Log to TensorBoard
        if writer is not None:
            for key, val in metrics.items():
                writer.add_scalar(f"train/{key}", val, global_step)
            writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], global_step,
            )
            # CLIP-style learnable temperature monitoring
            if hasattr(model.reg_head, "logit_scale"):
                ls = model.reg_head.logit_scale.exp().item()
                writer.add_scalar("train/logit_scale", ls, global_step)
                writer.add_scalar(
                    "train/effective_temperature", 1.0 / ls, global_step,
                )
            if hasattr(model.reg_head, "dustbin_score"):
                writer.add_scalar(
                    "train/dustbin_score",
                    model.reg_head.dustbin_score.item(), global_step,
                )

        # Accumulate for epoch average
        for key, val in metrics.items():
            epoch_metrics.setdefault(key, []).append(val)

        # Profiler step (if active)
        if profiler_ctx is not None:
            profiler_ctx.step()

        t_data_start = time.time()

    # Epoch averages
    avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    # Timing breakdown (first 3 epochs for diagnostics)
    if epoch < 3:
        n_steps = len(loader)
        data_ms = data_time_total / max(n_steps, 1) * 1000
        compute_ms = compute_time_total / max(n_steps, 1) * 1000
        total_ms = data_ms + compute_ms
        print(
            f"  ⏱ Timing: data={data_ms:.0f}ms/step "
            f"compute={compute_ms:.0f}ms/step "
            f"({data_time_total/(data_time_total+compute_time_total)*100:.0f}% data)"
        )

    return avg_metrics


@torch.no_grad()
def evaluate(
    model: GeoRegistrationModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set.

    Returns:
        Dict of averaged metrics
    """
    model.eval()
    all_metrics: dict[str, list[float]] = {}

    for batch_src, batch_tgt, gt in loader:
        batch_src = batch_src.to(device, non_blocking=True)
        batch_tgt = batch_tgt.to(device, non_blocking=True)
        R_gt = gt["R"].to(device, non_blocking=True)
        t_gt = gt["t"].to(device, non_blocking=True)
        corr_gt = gt["correspondences"].to(device, non_blocking=True)

        with lie_group_precision():
            out = model(batch_src, batch_tgt)

        # Always compute CD + Corr as evaluation metrics
        _, metrics_tensors = registration_loss(
            out["R"], out["t"], R_gt, t_gt,
            pos_src=out["pos_src"],
            pos_tgt=out["pos_tgt"],
            mask_src=out["mask_src"],
            assignment=out["assignment"],
            correspondences=corr_gt,
            w_cd=1.0,
            w_corr=1.0,
        )

        # Convert GPU tensors → Python floats (single sync)
        metrics = {k: v.item() for k, v in metrics_tensors.items()}
        metrics["loss/rotation_deg"] = math.degrees(
            metrics["loss/rotation_rad"]
        )

        for key, val in metrics.items():
            all_metrics.setdefault(key, []).append(val)

    return {k: sum(v) / len(v) for k, v in all_metrics.items()}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def build_model(args: argparse.Namespace) -> GeoRegistrationModel:
    """Build GeoRegistrationModel from CLI args."""
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
    return model


def build_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders."""
    if args.dataset == "synthetic":
        try:
            from .dataset import SyntheticRegistrationDataset
        except ImportError:
            from dataset import SyntheticRegistrationDataset
        train_ds = SyntheticRegistrationDataset(
            num_samples=args.num_train,
            num_points=args.num_points,
            rotation_range=args.rotation_range,
            translation_range=0.5,
            noise_std=0.01,
            partial_ratio=args.partial_ratio,
            seed=42,
        )
        val_ds = SyntheticRegistrationDataset(
            num_samples=max(100, args.num_train // 10),
            num_points=args.num_points,
            rotation_range=args.rotation_range,
            translation_range=0.5,
            noise_std=0.01,
            partial_ratio=args.partial_ratio,
            seed=999,
        )
    elif args.dataset == "modelnet40":
        try:
            from .dataset_modelnet40 import ModelNet40Registration
        except ImportError:
            from dataset_modelnet40 import ModelNet40Registration
        train_ds = ModelNet40Registration(
            cache_path=args.cache,
            num_points=args.num_points,
            split="train",
            rotation_range=args.rotation_range,
            partial_ratio=args.partial_ratio,
            inlier_threshold=args.inlier_threshold,
            exclude_symmetric=not args.include_symmetric,
        )
        val_ds = ModelNet40Registration(
            cache_path=args.cache,
            num_points=args.num_points,
            split="test",
            rotation_range=args.rotation_range,
            partial_ratio=args.partial_ratio,
            inlier_threshold=args.inlier_threshold,
            exclude_symmetric=not args.include_symmetric,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Data loading performance:
    # - num_workers>0: prefetch batches in parallel processes
    # - persistent_workers: avoid fork overhead per epoch
    # - prefetch_factor: each worker pre-loads N batches ahead
    # - pin_memory: enable async GPU transfer with non_blocking=True
    use_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=registration_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=registration_collate_fn,
        pin_memory=True,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    return train_loader, val_loader


def main():
    # Lazy imports — require tensorboard only at runtime
    from torch.utils.tensorboard import SummaryWriter
    from torch.profiler import (
        profile,
        ProfilerActivity,
        schedule as profiler_schedule,
        tensorboard_trace_handler,
    )

    parser = argparse.ArgumentParser(
        description="Train GeoRegistrationModel — SE(3)-equivariant "
                    "point cloud registration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "modelnet40"],
        help="Dataset to use",
    )
    parser.add_argument("--cache", type=str, default="data/modelnet40_4096.pt",
                        help="ModelNet40 cache path")
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of synthetic training samples")
    parser.add_argument("--num_points", type=int, default=512,
                        help="Points per cloud")
    parser.add_argument("--rotation_range", type=float, default=180.0,
                        help="Max rotation angle in degrees")
    parser.add_argument("--partial_ratio", type=float, default=1.0,
                        help="Source visibility fraction")
    parser.add_argument("--inlier_threshold", type=float, default=0.05,
                        help="Max warp→NN distance for GT correspondence "
                             "(larger = outlier → dustbin)")
    parser.add_argument("--include_symmetric", action="store_true",
                        help="Include rotationally symmetric categories "
                             "(bottle, bowl, cup, etc.) — NOT recommended")

    # Model
    parser.add_argument("--hidden_scalar", type=int, default=32,
                        help="SE3Net scalar channels")
    parser.add_argument("--hidden_vector", type=int, default=8,
                        help="SE3Net vector channels")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="SE3Net depth")
    parser.add_argument("--radius", type=float, default=0.3,
                        help="SE3Net neighbor radius")
    parser.add_argument("--max_neighbors", type=int, default=32,
                        help="Max neighbors per node")
    parser.add_argument("--cross_attn_layers", type=int, default=2,
                        help="Cross-attention rounds")
    parser.add_argument("--descriptor_dim", type=int, default=32,
                        help="Descriptor dimension")
    parser.add_argument("--sinkhorn_iters", type=int, default=10,
                        help="Sinkhorn OT iterations")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clipping")
    parser.add_argument("--loss_cd_weight", type=float, default=0.0,
                        help="Chamfer Distance auxiliary loss weight "
                             "(0=disabled, recommended 0.1 for partial overlap)")
    parser.add_argument("--loss_corr_weight", type=float, default=0.0,
                        help="Correspondence supervision weight "
                             "(recommended 1.0 for partial overlap training)")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (2=recommended for in-memory .pt dataset)")

    # Logging
    parser.add_argument("--log_dir", type=str, default="runs/registration",
                        help="TensorBoard log directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Profiler
    parser.add_argument("--profile", action="store_true",
                        help="Enable PyTorch Profiler + TensorBoard trace")
    parser.add_argument("--profile_epochs", type=int, default=1,
                        help="Number of epochs to profile")
    parser.add_argument("--profile_wait", type=int, default=1,
                        help="Profiler schedule: wait steps")
    parser.add_argument("--profile_warmup", type=int, default=1,
                        help="Profiler schedule: warmup steps")
    parser.add_argument("--profile_active", type=int, default=3,
                        help="Profiler schedule: active tracing steps")
    parser.add_argument("--profile_repeat", type=int, default=1,
                        help="Profiler schedule: repeat cycles")

    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        print(f"  VRAM: {vram / 1e9:.1f} GB")
        # Enable cuDNN benchmark for conv kernel autotuning
        torch.backends.cudnn.benchmark = True

    # ── Model ──
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"  Backbone: hidden_scalar={args.hidden_scalar}, "
          f"hidden_vector={args.hidden_vector}, "
          f"layers={args.num_layers}, radius={args.radius}")

    # ── Data ──
    train_loader, val_loader = build_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Warmup + Cosine schedule (per step)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── TensorBoard ──
    run_name = (
        f"{args.dataset}_s{args.hidden_scalar}_v{args.hidden_vector}_"
        f"L{args.num_layers}_r{args.radius}_"
        f"bs{args.batch_size}_lr{args.lr}"
    )
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}")

    # Log hyperparameters
    writer.add_text("hparams", str(vars(args)))

    # ── Profiler (optional) ──
    profiler_ctx = None
    if args.profile:
        profiler_dir = os.path.join(log_dir, "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        print(f"Profiler output: {profiler_dir}")
        print(f"  Schedule: wait={args.profile_wait}, "
              f"warmup={args.profile_warmup}, "
              f"active={args.profile_active}, "
              f"repeat={args.profile_repeat}")

    # ── Save dir ──
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training ──
    best_val_rot = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Set up profiler for this epoch (if enabled and within range)
        if args.profile and epoch < args.profile_epochs:
            profiler_ctx = profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=profiler_schedule(
                    wait=args.profile_wait,
                    warmup=args.profile_warmup,
                    active=args.profile_active,
                    repeat=args.profile_repeat,
                ),
                on_trace_ready=tensorboard_trace_handler(
                    os.path.join(log_dir, "profiler"),
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler_ctx.__enter__()
        else:
            profiler_ctx = None

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, writer, args.grad_clip,
            w_cd=args.loss_cd_weight,
            w_corr=args.loss_corr_weight,
            profiler_ctx=profiler_ctx,
        )

        # Close profiler
        if profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)
            profiler_ctx = None

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        t1 = time.time()

        # Log validation
        global_step = (epoch + 1) * len(train_loader)
        for key, val in val_metrics.items():
            writer.add_scalar(f"val/{key}", val, global_step)

        # Print epoch summary
        cd_str = ""
        if "loss/chamfer" in val_metrics:
            cd_str = f" val_cd={val_metrics['loss/chamfer']:.4f}"
        corr_str = ""
        if "loss/correspondence" in val_metrics:
            corr_str = f" val_corr={val_metrics['loss/correspondence']:.3f}"
        # Get learnable temperature info
        tau_str = ""
        if hasattr(model.reg_head, "logit_scale"):
            ls = model.reg_head.logit_scale.exp().clamp(min=1.0, max=100.0).item()
            tau_str = f" τ={1.0/ls:.4f}"
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"[{t1-t0:.1f}s] "
            f"train_rot={train_metrics['loss/rotation_deg']:.2f}° "
            f"train_t={train_metrics['loss/translation_mse']:.4f} "
            f"val_rot={val_metrics['loss/rotation_deg']:.2f}° "
            f"val_t={val_metrics['loss/translation_mse']:.4f}"
            f"{cd_str}{corr_str}{tau_str} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save checkpoint
        is_best = val_metrics["loss/rotation_deg"] < best_val_rot
        if is_best:
            best_val_rot = val_metrics["loss/rotation_deg"]

        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            path = os.path.join(
                args.save_dir, f"registration_epoch{epoch+1}.pt",
            )
            torch.save(ckpt, path)
            if is_best:
                best_path = os.path.join(args.save_dir, "registration_best.pt")
                torch.save(ckpt, best_path)
                print(f"  ★ New best: {best_val_rot:.2f}° → {best_path}")

    writer.close()
    print(f"\nTraining complete. Best val rotation error: {best_val_rot:.2f}°")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
