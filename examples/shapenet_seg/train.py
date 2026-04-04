#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Training script for SE(3)-equivariant ShapeNet Part Segmentation.

Usage:
    python examples/shapenet_seg/train.py --data_root data/ShapeNetPart
    python examples/shapenet_seg/train.py \
    --data_root data/shapenetpart_hdf5_2048 \
    --batch_size 64 \
    --hidden_scalar 64 --hidden_vector 16 \
    --epochs 200 \
    --lr 3e-3 \
    --eval_every 10


Key features:
    - AMP with FP32 protection for geometric ops (SH, tensor products)
    - Mandatory category masking (invalid part logits → -inf)
    - Part mIoU metric (per-category, then averaged)
    - Gradient accumulation for effective large batch sizes
    - Best model checkpointing based on validation mIoU
    - Normal vectors as l=1 equivariant features

⚠ AMP Policy: torch.cuda.amp.autocast wraps ONLY the forward pass.
  The SE3Conv's spherical harmonics and tensor products internally
  use FP32 element-wise ops (x*y, z*z). With max_l=2, the values
  stay within FP16 range (~65500), but accumulated products across
  message passing can cause subtle precision loss. We use
  GradScaler with conservative settings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

# Reduce CUDA caching allocator fragmentation (must be set before CUDA init).
# expandable_segments: use virtual memory instead of large contiguous blocks.
# Typically cuts reserved/allocated ratio from ~2x to ~1.3x.
import os as _os
_alloc_conf = _os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
if 'expandable_segments' not in _alloc_conf:
    _new = 'expandable_segments:True'
    if _alloc_conf:
        _new = _alloc_conf + ',' + _new
    _os.environ['PYTORCH_CUDA_ALLOC_CONF'] = _new

from examples.shapenet_seg.dataset import (
    ShapeNetPartDataset,
    collate_fn,
    SEG_CLASSES,
    CATEGORY_TO_IDX,
    NUM_PARTS,
    NUM_CATEGORIES,
)
from examples.shapenet_seg.model import SE3PartSegNet
from geoembodied.nn.losses import lovasz_softmax


# ═══════════════════════════════════════════════════════════════════
# Part mIoU — the gold standard metric
# ═══════════════════════════════════════════════════════════════════

def compute_part_miou(
    all_preds: List[torch.Tensor],     # list of [N] int64 per shape
    all_labels: List[torch.Tensor],    # list of [N] int64 per shape
    all_cats: List[int],               # category index per shape
) -> Tuple[float, float, Dict[str, float]]:
    """Compute ShapeNet Part mIoU.

    Standard protocol:
        1. For each shape, compute IoU per valid part label
        2. Average valid part IoUs → shape IoU
        3. Category mIoU = mean of shape IoUs per category
        4. Instance mIoU = mean of all shape IoUs

    If a part label is absent in both prediction and ground truth,
    its IoU is defined as 1.0 (perfect for absent parts).

    Args:
        all_preds: Per-shape predictions (argmax of logits)
        all_labels: Per-shape ground truth labels
        all_cats: Per-shape category indices

    Returns:
        cat_miou: Category-averaged mIoU (mean of per-category means)
        inst_miou: Instance-averaged mIoU (mean of all shape IoUs)
        per_cat: Per-category mIoU dict
    """
    # Collect shape IoUs per category
    cat_shape_ious: Dict[int, List[float]] = {
        i: [] for i in range(NUM_CATEGORIES)
    }

    for pred, label, cat_idx in zip(all_preds, all_labels, all_cats):
        # Get valid part labels for this category
        cat_name = list(SEG_CLASSES.keys())[cat_idx]
        valid_parts = SEG_CLASSES[cat_name]

        shape_iou_parts = []
        for part_id in valid_parts:
            pred_mask = (pred == part_id)
            label_mask = (label == part_id)

            intersection = (pred_mask & label_mask).sum().item()
            union = (pred_mask | label_mask).sum().item()

            if union == 0:
                # Part absent in both pred and GT → perfect IoU
                iou = 1.0
            else:
                iou = intersection / union

            shape_iou_parts.append(iou)

        shape_iou = sum(shape_iou_parts) / len(shape_iou_parts)
        cat_shape_ious[cat_idx].append(shape_iou)

    # Per-category mIoU
    per_cat: Dict[str, float] = {}
    cat_mious = []
    for cat_idx in range(NUM_CATEGORIES):
        cat_name = list(SEG_CLASSES.keys())[cat_idx]
        if len(cat_shape_ious[cat_idx]) > 0:
            cat_miou_val = sum(cat_shape_ious[cat_idx]) / len(cat_shape_ious[cat_idx])
        else:
            cat_miou_val = 0.0
        per_cat[cat_name] = cat_miou_val
        if len(cat_shape_ious[cat_idx]) > 0:
            cat_mious.append(cat_miou_val)

    # Averages
    cat_miou = sum(cat_mious) / len(cat_mious) if cat_mious else 0.0

    all_shape_ious = []
    for ious in cat_shape_ious.values():
        all_shape_ious.extend(ious)
    inst_miou = sum(all_shape_ious) / len(all_shape_ious) if all_shape_ious else 0.0

    return cat_miou, inst_miou, per_cat


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: SE3PartSegNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accum_steps: int = 1,
    use_amp: bool = True,
    diag_every: int = 50,
    grad_clip: float = 1.0,
    amp_max_scale: float = 2**30,  # effectively disabled — model runs FP32
    label_smoothing: float = 0.0,
    lovasz_weight: float = 0.0,
) -> Tuple[float, float, dict]:
    """Train for one epoch.

    Returns:
        loss: Average cross-entropy loss over the epoch
        acc: Average per-point accuracy
        diag: Dict with v_norm_mean, v_norm_std, attn_entropy_mean,
              attn_max_mean, attn_uniform_ratio (averaged over sampled steps)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    n_batches = 0

    # Diagnostic accumulators (sampled every diag_every steps)
    # Use defaultdict so new keys from forward_with_diagnostics
    # (e.g. enc0_s_norm, t2_norm_mean) are added automatically.
    from collections import defaultdict
    diag_accum: Dict[str, List[float]] = defaultdict(list)

    optimizer.zero_grad(set_to_none=True)

    for step, (pos, normals, labels, ptr, cat_indices) in enumerate(loader):
        pos = pos.to(device, non_blocking=True)
        normals = normals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ptr = ptr.to(device, non_blocking=True)
        cat_indices = cat_indices.to(device, non_blocking=True)

        # Forward with AMP
        with autocast('cuda', enabled=use_amp):
            # Collect diagnostics periodically (adds ~2% overhead)
            if step % diag_every == 0:
                logits, diag = model.forward_with_diagnostics(
                    pos, ptr, cat_indices, normals=normals
                )
                for k, v in diag.items():
                    diag_accum[k].append(v)
            else:
                logits = model(pos, ptr, cat_indices, normals=normals)

            # ── Loss computation ──
            # Category masking sets invalid part logits to -inf.
            # Standard F.cross_entropy(label_smoothing=s) distributes
            # s/C probability to ALL classes including -inf ones,
            # causing log_softmax(-inf) = -inf → loss = inf.
            # Fix: compute label smoothing manually over valid classes.
            if label_smoothing > 0:
                log_p = nn.functional.log_softmax(logits, dim=1)  # [N, C]
                # Hard target (NLL) — always finite since labels are valid
                nll = nn.functional.nll_loss(log_p, labels)
                # Smooth target — only over unmasked (non -inf) classes
                valid = logits > -1e30  # [N, C] bool
                n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
                # CRITICAL: -inf * 0 = NaN (IEEE 754), so use where not mul
                log_p_safe = torch.where(valid, log_p, torch.zeros_like(log_p))
                smooth = -log_p_safe.sum(dim=1) / n_valid.squeeze(1)
                ce_loss = (1 - label_smoothing) * nll + label_smoothing * smooth.mean()
            else:
                ce_loss = nn.functional.cross_entropy(logits, labels)

            loss = ce_loss
            if lovasz_weight > 0:
                loss = loss + lovasz_weight * lovasz_softmax(logits, labels)
            loss = loss / accum_steps

        # Backward with scaler
        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            # Cap GradScaler to prevent FP16 gradient overflow.
            # At scale S, any FP32 gradient > 65504/S overflows to Inf
            # in FP16, causing silent step-skipping. With cap=32768,
            # gradients up to 2.0 are safe (65504/32768 ≈ 2.0).
            if scaler.is_enabled() and scaler.get_scale() > amp_max_scale:
                scaler._scale.fill_(amp_max_scale)

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        n_batches += 1

        # Accuracy (cheap: argmax + compare, fully on GPU)
        preds = logits.detach().argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.shape[0]

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_correct / max(total_points, 1)

    # Average diagnostics
    avg_diag = {}
    for k, vals in diag_accum.items():
        avg_diag[k] = sum(vals) / len(vals) if vals else 0.0

    # Track scaler state for monitoring
    scaler_scale = scaler.get_scale() if scaler.is_enabled() else 0.0

    return avg_loss, avg_acc, avg_diag, scaler_scale


@torch.no_grad()
def evaluate(
    model: SE3PartSegNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Evaluate on validation/test set.

    Returns:
        loss, cat_miou, inst_miou, per_cat_miou
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_cats: List[int] = []

    for pos, normals, labels, ptr, cat_indices in loader:
        pos = pos.to(device, non_blocking=True)
        normals = normals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ptr = ptr.to(device, non_blocking=True)
        cat_indices = cat_indices.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            logits = model(pos, ptr, cat_indices, normals=normals)
            loss = nn.functional.cross_entropy(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        # Per-shape predictions
        preds = logits.argmax(dim=1)  # [N_total]
        B = ptr.shape[0] - 1
        for b in range(B):
            start = ptr[b].item()
            end = ptr[b + 1].item()
            all_preds.append(preds[start:end].cpu())
            all_labels.append(labels[start:end].cpu())
            all_cats.append(cat_indices[b].item())

    avg_loss = total_loss / max(n_batches, 1)
    cat_miou, inst_miou, per_cat = compute_part_miou(
        all_preds, all_labels, all_cats
    )
    return avg_loss, cat_miou, inst_miou, per_cat


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='SE(3)-Equivariant ShapeNet Part Segmentation'
    )
    parser.add_argument('--data_root', type=str, default='data/ShapeNetPart',
                        help='Root dir containing shapenetcore_partanno_...')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--hidden_scalar', type=int, default=32)
    parser.add_argument('--hidden_vector', type=int, default=8)
    parser.add_argument('--num_stages', type=int, default=2)
    parser.add_argument('--layers_per_stage', type=int, default=1)
    parser.add_argument('--pool_ratio', type=float, default=0.25)
    parser.add_argument('--no_normals', action='store_true',
                        help='Disable normal vector input (ablation)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--save_dir', type=str, default='checkpoints/shapenet_seg')
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--profile', action='store_true',
                        help='Profile first 5 batches and exit')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--gate_mode', type=str, default='norm',
                        choices=['scalar', 'norm'],
                        help='Vector gate mode: scalar (sigmoid) or norm (norm-aware)')
    parser.add_argument('--use_self_tp', action='store_true', default=True,
                        help='Enable self-interaction TP (v·v → scalar)')
    parser.add_argument('--no_self_tp', action='store_true',
                        help='Disable self-interaction TP')
    parser.add_argument('--use_bottleneck_attn', action='store_true', default=True,
                        help='Enable bottleneck geometric attention')
    parser.add_argument('--no_bottleneck_attn', action='store_true',
                        help='Disable bottleneck geometric attention')
    parser.add_argument('--hidden_type2', type=int, default=0,
                        help='Type-2 (l=2) channels. 0=disabled, 4=recommended')
    parser.add_argument('--head_hidden', type=int, default=128,
                        help='Classification head hidden dim')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile(dynamic=True) for kernel fusion')
    # ── Training recipe (anti-overfitting) ──
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for CE loss (0.1 recommended)')
    parser.add_argument('--lovasz_weight', type=float, default=0.0,
                        help='Weight for Lovász-Softmax loss (1.0 recommended). '
                             'Directly optimizes mIoU.')

    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Linear LR warmup epochs (5 recommended)')
    args = parser.parse_args()
    if args.no_self_tp:
        args.use_self_tp = False
    if args.no_bottleneck_attn:
        args.use_bottleneck_attn = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and device.type == 'cuda'
    use_normals = not args.no_normals

    if device.type == 'cuda':
        # Enable TF32 on Ampere+ GPUs: ~3x faster FP32 matmuls with negligible
        # precision loss (mantissa 10 bits vs 23). Safe for equivariant TP paths.
        torch.set_float32_matmul_precision('high')

    print("=" * 60)
    print("SE(3)-Equivariant ShapeNet Part Segmentation")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {'ON' if use_amp else 'OFF'}")
    print(f"Normals: {'ON (l=1 vector features)' if use_normals else 'OFF (pos only)'}")
    print(f"Gate mode: {args.gate_mode}")
    print(f"Self-TP: {'ON (ν=2)' if args.use_self_tp else 'OFF (ν=1)'}")
    print(f"Bottleneck attn: {'ON' if args.use_bottleneck_attn else 'OFF'}")
    print(f"Type-2 (l=2):  {args.hidden_type2} channels {'(disabled)' if args.hidden_type2 == 0 else ''}")
    print(f"torch.compile: {'ON (dynamic=True)' if args.compile else 'OFF'}")
    print(f"Batch: {args.batch_size} × {args.accum_steps} accum = "
          f"{args.batch_size * args.accum_steps} effective")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Lovász weight: {args.lovasz_weight} {'(OFF)' if args.lovasz_weight == 0 else '(IoU-direct)'}")

    print(f"Warmup: {args.warmup_epochs} epochs {'(OFF)' if args.warmup_epochs == 0 else '(linear)'}")
    print()

    # ── Datasets ──
    train_dataset = ShapeNetPartDataset(
        args.data_root, split='trainval', normalize=True,
    )
    test_dataset = ShapeNetPartDataset(
        args.data_root, split='test', normalize=True,
    )

    nw = args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
    )

    print(f"Train: {len(train_dataset)} shapes")
    print(f"Test:  {len(test_dataset)} shapes")

    # ── Model ──
    model = SE3PartSegNet(
        in_channels=1,
        hidden_scalar=args.hidden_scalar,
        hidden_vector=args.hidden_vector,
        hidden_type2=args.hidden_type2,
        num_stages=args.num_stages,
        layers_per_stage=args.layers_per_stage,
        pool_ratio=args.pool_ratio,
        use_normals=use_normals,
        head_hidden=args.head_hidden,
        gate_mode=args.gate_mode,
        use_self_tp=args.use_self_tp,
        use_bottleneck_attn=args.use_bottleneck_attn,

    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # torch.compile: automatic kernel fusion for _compute_messages
    # NOTE: We compile ONLY the inner TP function, NOT the whole model.
    # The outer model contains untraceable ops (CUDA KNN, radius_graph,
    # .item() in adaptive radius, checkpoint with free variables) that
    # cause graph breaks or crashes. _compute_messages is verified
    # zero-graph-break and contains the actual TP compute hotpath.
    if args.compile:
        import geoembodied.nn.modules.se3_conv as _se3_conv_mod
        _se3_conv_mod._compute_messages = torch.compile(
            _se3_conv_mod._compute_messages, dynamic=True,
        )
        print("  torch.compile(dynamic=True) applied to _compute_messages")
    print()

    # ── Optimizer + Scheduler ──
    # Standard practice (BERT/GPT/ViT): exclude 1D params (biases, norm
    # weights, scales) from weight decay. Weight decay on these pushes
    # them toward 0, corrupting normalization scale and gate behavior.
    decay_params = [p for p in model.parameters()
                    if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters()
                       if p.requires_grad and p.ndim < 2]
    print(f"  Param groups: {len(decay_params)} decay, "
          f"{len(no_decay_params)} no-decay")

    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Linear warmup: general optimizer best practice (NeurIPS 2024).
    # Lets Adam's adaptive moment estimates converge before applying
    # full lr. Especially important for GDL where SE3Conv gradient
    # variance at initialization is high.
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-2, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    # AMP GradScaler: The SE3PartSegNet model disables autocast
    # internally (all geometric ops + nn.Linear run in FP32), so the
    # scaler effectively becomes a no-op.  We keep it for API
    # compatibility with the autocast() wrapper in train_one_epoch.
    # init_scale can be high since FP32 gradients never overflow.
    scaler = GradScaler('cuda', enabled=use_amp, init_scale=2**16)

    # ── Profiling mode ──
    if args.profile:
        _profile_training(model, train_loader, optimizer, scaler,
                          device, use_amp)
        return

    # ── Checkpointing ──
    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0.0

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, diag, scaler_scale = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            accum_steps=args.accum_steps, use_amp=use_amp,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            lovasz_weight=args.lovasz_weight,
        )
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # Main log line
        log = (f"Epoch {epoch:3d}/{args.epochs} | "
               f"loss={train_loss:.4f} acc={train_acc:.4f} | "
               f"lr={lr:.6f} | {dt:.1f}s")

        # Diagnostic line (vector health + attention learning)
        v_norm = diag.get('v_norm_mean', 0)
        v_std = diag.get('v_norm_std', 0)
        attn_ent = diag.get('attn_entropy_mean', 0)
        attn_max = diag.get('attn_max_mean', 0)
        attn_uni = diag.get('attn_uniform_ratio', 1)
        diag_log = (f"  ├─ v_norm={v_norm:.4f}±{v_std:.4f} | "
                    f"attn: entropy={attn_ent:.3f} max={attn_max:.3f} "
                    f"uniform={attn_uni:.2f}")

        # Per-stage encoder scalar norms (multi-scale health)
        enc_norms = []
        for i in range(args.num_stages):
            key = f'enc{i}_s_norm'
            if key in diag:
                enc_norms.append(f"s{i}={diag[key]:.3f}")
        if enc_norms:
            diag_log += f" | enc:[{','.join(enc_norms)}]"

        # Type-2 norm diagnostics
        t2_norm = diag.get('t2_norm_mean', None)
        if t2_norm is not None:
            t2_std = diag.get('t2_norm_std', 0)
            diag_log += f" | t2={t2_norm:.4f}±{t2_std:.4f}"

        # Vector → head invariant diagnostics (Component 2)
        v_inv_norm = diag.get('v_inv_norm', None)
        if v_inv_norm is not None:
            v_inv_std = diag.get('v_inv_std', 0)
            diag_log += f" | v_inv={v_inv_norm:.4f}±{v_inv_std:.4f}"

        # Type-2 → head invariant diagnostics
        t2_inv_norm = diag.get('t2_inv_norm', None)
        if t2_inv_norm is not None:
            t2_inv_std = diag.get('t2_inv_std', 0)
            diag_log += f" | t2_inv={t2_inv_norm:.4f}±{t2_inv_std:.4f}"

        # Per-pool attention diagnostics (Component 1)
        # Show per-pool temperature and uniformity to detect deep/shallow divergence
        pool_temps = []
        pool_unis = []
        for i in range(args.num_stages - 1):
            t_key = f'pool_T{i}'
            u_key = f'pool_u{i}'
            if t_key in diag:
                pool_temps.append(f"T{i}={diag[t_key]:.3f}")
            if u_key in diag:
                pool_unis.append(f"u{i}={diag[u_key]:.2f}")
        if pool_temps:
            diag_log += f" | pool:[{','.join(pool_temps)}]"
        if pool_unis:
            diag_log += f" [{','.join(pool_unis)}]"

        # ── Skip scale diagnostics (per-block learnable residual) ──
        skip_parts = []
        bb = model.backbone
        for si in range(bb.num_stages):
            for bi, blk in enumerate(bb.encoder_stages[si].blocks):
                d = blk.get_skip_diagnostics()
                if d:
                    tag = f'e{si}b{bi}'
                    skip_parts.append(
                        f'{tag}[s={d["skip_s_mean"]:.3f}'
                        f',v={d.get("skip_v_mean", 0):.3f}'
                        f',t2={d.get("skip_t2_mean", 0):.3f}]'
                    )
        if skip_parts:
            diag_log += '\n  ├─ skip: ' + ' '.join(skip_parts)

        if scaler_scale > 0:
            diag_log += f" | amp_scale={scaler_scale:.0f}"

        # VRAM tracking
        if device.type == 'cuda':
            vram_alloc = torch.cuda.max_memory_allocated() / 1024**2
            vram_reserved = torch.cuda.max_memory_reserved() / 1024**2
            diag_log += f" | vram={vram_alloc:.0f}MB(alloc) {vram_reserved:.0f}MB(reserved)"

        # Evaluate periodically
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss, cat_miou, inst_miou, per_cat = evaluate(
                model, test_loader, device, use_amp=use_amp,
            )
            log += (f" | val_loss={val_loss:.4f} "
                    f"cat_mIoU={cat_miou:.4f} inst_mIoU={inst_miou:.4f}")

            # Save best
            if inst_miou > best_miou:
                best_miou = inst_miou
                ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'inst_miou': inst_miou,
                    'cat_miou': cat_miou,
                    'args': vars(args),
                }, ckpt_path)
                log += " ★ best"

        print(log)
        print(diag_log)

    # ── Final summary ──
    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Best instance mIoU: {best_miou:.4f}")
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pt')}")
    print("=" * 60)

    # Final detailed evaluation
    print("\nFinal per-category mIoU:")
    _, _, _, per_cat = evaluate(model, test_loader, device, use_amp=use_amp)
    for cat_name, miou in sorted(per_cat.items()):
        n_parts = len(SEG_CLASSES[cat_name])
        print(f"  {cat_name:15s}: {miou:.4f} ({n_parts} parts)")


def _profile_training(
    model: SE3PartSegNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    n_batches: int = 5,
) -> None:
    """Profile first N training batches with CUDA event timing.

    Breaks down per-step time into:
        - data_load: DataLoader → CPU tensors
        - to_device: CPU → GPU transfer
        - forward: model forward pass
        - loss: CrossEntropy computation
        - backward: gradient computation
        - optim: optimizer step
    """
    print()
    print("=" * 60)
    print("PROFILING MODE — timing first", n_batches, "batches")
    print("=" * 60)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Warmup: 1 batch
    print("Warmup...")
    for pos, normals, labels, ptr, cat_indices in loader:
        pos = pos.to(device)
        normals = normals.to(device)
        labels = labels.to(device)
        ptr = ptr.to(device)
        cat_indices = cat_indices.to(device)
        with autocast('cuda', enabled=use_amp):
            logits = model(pos, ptr, cat_indices, normals=normals)
            loss = nn.functional.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        break
    torch.cuda.synchronize()
    print("Warmup done.")

    # Check which radius_graph backend is active
    print()
    try:
        from geoembodied.csrc import get_radius_graph_backend
        backend = get_radius_graph_backend()
    except Exception:
        backend = "brute-force (import failed)"
    print(f"  radius_graph backend: {backend}")
    if backend != "cuda":
        print("  ⚠ Using O(N²) brute-force! Compile CUDA kernel for 5-10x speedup")
    print()

    # Profile
    timings = {k: [] for k in [
        'data_load', 'to_device', 'forward', 'loss', 'backward', 'optim', 'total'
    ]}

    data_iter = iter(loader)
    for step in range(n_batches):
        torch.cuda.synchronize()

        # ── Data load ──
        t0 = time.time()
        batch = next(data_iter)
        pos, normals, labels, ptr, cat_indices = batch
        t_data = time.time() - t0

        # ── To device ──
        torch.cuda.synchronize()
        t0 = time.time()
        pos = pos.to(device)
        normals = normals.to(device)
        labels = labels.to(device)
        ptr = ptr.to(device)
        cat_indices = cat_indices.to(device)
        torch.cuda.synchronize()
        t_device = time.time() - t0

        # ── Forward ──
        torch.cuda.synchronize()
        t0 = time.time()
        with autocast('cuda', enabled=use_amp):
            logits = model(pos, ptr, cat_indices, normals=normals)
        torch.cuda.synchronize()
        t_fwd = time.time() - t0

        # ── Loss ──
        torch.cuda.synchronize()
        t0 = time.time()
        with autocast('cuda', enabled=use_amp):
            loss = nn.functional.cross_entropy(logits, labels)
        torch.cuda.synchronize()
        t_loss = time.time() - t0

        # ── Backward ──
        torch.cuda.synchronize()
        t0 = time.time()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t0

        # ── Optimizer ──
        torch.cuda.synchronize()
        t0 = time.time()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_opt = time.time() - t0

        t_total = t_data + t_device + t_fwd + t_loss + t_bwd + t_opt

        timings['data_load'].append(t_data)
        timings['to_device'].append(t_device)
        timings['forward'].append(t_fwd)
        timings['loss'].append(t_loss)
        timings['backward'].append(t_bwd)
        timings['optim'].append(t_opt)
        timings['total'].append(t_total)

        N_pts = pos.shape[0]
        print(f"  Step {step}: {t_total*1000:.0f}ms total | "
              f"data={t_data*1000:.0f}ms fwd={t_fwd*1000:.0f}ms "
              f"bwd={t_bwd*1000:.0f}ms | {N_pts} pts")

    # Summary — exclude Step 0 (warmup / torch.compile overhead)
    print()
    print("=" * 60)
    if n_batches > 1:
        steady_range = range(1, n_batches)
        n_steady = n_batches - 1
        print(f"Profile Summary (ms, Steps 1-{n_batches-1} steady-state)")
        print(f"  Step 0 excluded: {timings['total'][0]*1000:.0f}ms "
              f"(warmup/compile)")
    else:
        steady_range = range(n_batches)
        n_steady = n_batches
        print(f"Profile Summary (ms, {n_batches} step)")
    print("=" * 60)

    total_avg = sum(timings['total'][i] for i in steady_range) / n_steady * 1000

    for key in ['data_load', 'to_device', 'forward', 'loss', 'backward', 'optim']:
        avg_ms = sum(timings[key][i] for i in steady_range) / n_steady * 1000
        pct = avg_ms / total_avg * 100 if total_avg > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {key:12s}: {avg_ms:8.1f} ms ({pct:5.1f}%) {bar}")

    print(f"  {'TOTAL':12s}: {total_avg:8.1f} ms")
    N_pts = pos.shape[0]
    pts_per_sec = N_pts / (total_avg / 1000)
    print(f"  Throughput:   {pts_per_sec:,.0f} pts/sec")
    print(f"  Est. epoch:   {total_avg * len(loader) / 1000:.0f}s "
          f"({total_avg * len(loader) / 60000:.1f} min)")

    if device.type == 'cuda':
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"  Peak VRAM:    {peak_alloc:.0f} MB (alloc) / {peak_reserved:.0f} MB (reserved)")

    # Identify bottleneck
    bottleneck = max(
        ['data_load', 'to_device', 'forward', 'backward'],
        key=lambda k: sum(timings[k][i] for i in steady_range)
    )
    print(f"\n  ⚠ Bottleneck: {bottleneck}")
    if bottleneck == 'data_load':
        print("    → Increase num_workers or use persistent_workers")
    elif bottleneck == 'forward':
        print("    → Graph construction + SE3Conv message passing")
        print("    → Try: reduce num_stages, layers_per_stage, or hidden channels")
    elif bottleneck == 'backward':
        print("    → Large graph + autograd overhead")


if __name__ == '__main__':
    main()
