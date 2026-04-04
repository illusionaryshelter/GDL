# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Lovász-Softmax loss — directly optimizes the mean IoU metric.

.. math::
    \\mathcal{L}_{\\text{Lovász}} = \\frac{1}{|\\mathcal{C}|}
    \\sum_{c \\in \\mathcal{C}} \\overline{\\Delta_{J_c}}(\\mathbf{e}(c))

where :math:`\\overline{\\Delta_{J_c}}` is the convex (Lovász) extension of
the Jaccard loss for class *c*, and :math:`\\mathbf{e}(c)` is the vector of
per-point prediction errors for that class.

Why this belongs in GDL rather than in task-specific code
=========================================================

Cross-entropy treats every point independently; it is dominated by
whichever class has the most points.  For any segmentation task —
point cloud parts, semantic scenes, or medical structures — the metric
that actually matters is **mean Intersection-over-Union** (mIoU), which
weights every class equally regardless of spatial extent.

Lovász-Softmax is the only known **convex, differentiable surrogate**
that directly optimizes IoU.  It is task-agnostic: it takes ``[N, C]``
logits and ``[N]`` labels — the same interface as ``F.cross_entropy``.
Placing it alongside the library's equivariant layers ensures that any
downstream segmentation head can use it without copy-pasting code.

Combined usage (recommended)::

    loss = F.cross_entropy(logits, labels, label_smoothing=0.1) \\
         + λ * lovasz_softmax(logits, labels)

The CE term provides stable per-pixel probability gradients; the Lovász
term steers optimization toward the actual evaluation metric.

Reference:
    Berman, Triki & Blaschko, "The Lovász-Softmax loss: A tractable
    surrogate for the optimization of the intersection-over-union
    measure in neural networks", CVPR 2018.

NOTE: Pure PyTorch — no scipy or external dependencies (AGENTS.md Rule 7).
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = ["lovasz_softmax"]


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Compute gradient of the Lovász extension w.r.t. sorted errors.

    Args:
        gt_sorted: [P] binary ground truth sorted by decreasing error.
            1 = foreground, 0 = background for this class.

    Returns:
        grad: [P] Lovász gradient weights.
            These transform sorted per-pixel errors into an IoU-aware loss.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)  # [P]
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]  # finite differences
    return jaccard


def _lovasz_softmax_flat(
    probas: Tensor,   # [P, C] class probabilities
    labels: Tensor,   # [P] int64 ground truth
    classes: str = 'present',  # 'all' or 'present'
) -> Tensor:
    """Multi-class Lovász-Softmax loss, flat (no batch dim).

    Args:
        probas: [P, C] softmax probabilities per point per class.
        labels: [P] ground truth class labels (int64).
        classes: 'present' = only classes in ground truth (avoids division
            by zero for absent classes). 'all' = all C classes.

    Returns:
        Scalar loss.
    """
    C = probas.shape[1]

    losses = []
    for c in range(C):
        fg = (labels == c).float()  # [P] binary foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]  # [P] predicted prob for class c

        errors = (fg - class_pred).abs()  # [P]
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = _lovasz_grad(fg_sorted)  # [P]
        losses.append(torch.dot(errors_sorted, grad))

    if not losses:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


def lovasz_softmax(
    logits: Tensor,
    labels: Tensor,
    classes: str = 'present',
) -> Tensor:
    """Lovász-Softmax loss for multi-class segmentation.

    Directly optimizes the mean IoU metric via the Lovász extension.
    Takes raw logits (applies softmax internally) — same interface as
    ``F.cross_entropy``.

    Args:
        logits: Raw model output (un-softmaxed).
            shape: [N, C], N = total points in batch, C = number of classes.
        labels: Ground truth class labels.
            shape: [N], int64, values in ``[0, C-1]``.
        classes: Which classes to average over:
            - ``'present'`` (default): only classes with ≥1 ground-truth point.
              Avoids wasting gradient on absent classes.
            - ``'all'``: every class contributes, even if absent.

    Returns:
        Scalar loss (differentiable).

    Example::

        >>> logits = model(batch)              # [N, 50]
        >>> labels = batch['label']            # [N]
        >>> ce  = F.cross_entropy(logits, labels, label_smoothing=0.1)
        >>> lov = lovasz_softmax(logits, labels)
        >>> loss = ce + lov                    # combined
    """
    probas = F.softmax(logits, dim=1)  # [N, C]
    return _lovasz_softmax_flat(probas, labels, classes=classes)
