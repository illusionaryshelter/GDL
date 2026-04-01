# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Chamfer Distance — mask-aware, supports uni/bidirectional.

Used as:
  1. Symmetry-safe evaluation metric for registration
     (CD stays low even when R_pred ≠ R_gt due to symmetry)
  2. Auxiliary training loss for partial overlap scenarios

Memory note: Uses torch.cdist → O(N*M) intermediate.
  If N > 10,000, this will OOM.  Switch to KeOps or CUDA KNN.

All inputs follow [B, N, 3] (batched) or [N, 3] (unbatched) convention.
"""

import torch
from torch import Tensor
from typing import Optional


def chamfer_distance(
    cloud_a: Tensor,
    cloud_b: Tensor,
    mask_a: Optional[Tensor] = None,
    mask_b: Optional[Tensor] = None,
    bidirectional: bool = True,
) -> Tensor:
    """Compute Chamfer Distance between two point clouds.

    When bidirectional=True (default):
        CD(A,B) = mean_a(min_b ||a - b||²) + mean_b(min_a ||b - a||²)

    When bidirectional=False (for partial overlap):
        CD(A→B) = mean_a(min_b ||a - b||²)
        Only measures how well A's points align to B.
        Essential for partial overlap where A ⊂ B.

    Mask support ensures padding positions (from ``to_dense_batch``)
    are excluded from both the min-search and the mean.

    Args:
        cloud_a: Point cloud A, shape [N, 3] or [B, N, 3]
        cloud_b: Point cloud B, shape [M, 3] or [B, M, 3]
        mask_a: Boolean mask for A [N] or [B, N]. True = real point.
            If None, all points are considered real.
        mask_b: Boolean mask for B [M] or [B, M]. True = real point.
            If None, all points are considered real.
        bidirectional: If True, compute symmetric CD.
            If False, compute only A→B direction.

    Returns:
        Scalar CD value (mean over batch if batched).

    Raises:
        ValueError: If cloud shapes are incompatible.
    """
    # Handle unbatched input
    unbatched = cloud_a.dim() == 2
    if unbatched:
        cloud_a = cloud_a.unsqueeze(0)
        cloud_b = cloud_b.unsqueeze(0)
        if mask_a is not None:
            mask_a = mask_a.unsqueeze(0)
        if mask_b is not None:
            mask_b = mask_b.unsqueeze(0)

    B, N, _ = cloud_a.shape
    _, M, _ = cloud_b.shape

    # Pairwise squared distances [B, N, M]
    # torch.cdist returns Euclidean distances; square them.
    dist = torch.cdist(cloud_a.float(), cloud_b.float()).pow(2)  # [B, N, M]

    # A→B: for each point in A, find nearest in B
    dist_a2b = dist  # [B, N, M]
    if mask_b is not None:
        # Mask out padding in B → they should NOT be nearest neighbours
        dist_a2b = dist_a2b.masked_fill(
            ~mask_b.unsqueeze(1), float("inf"),
        )  # [B, N, M]

    min_a2b = dist_a2b.min(dim=2).values  # [B, N]

    if mask_a is not None:
        # Zero out padding in A → don't count in mean
        min_a2b = min_a2b * mask_a.float()
        count_a = mask_a.float().sum(dim=1).clamp(min=1.0)  # [B]
    else:
        count_a = torch.tensor(N, dtype=cloud_a.dtype, device=cloud_a.device)

    cd_a2b = min_a2b.sum(dim=1) / count_a  # [B]

    if not bidirectional:
        return cd_a2b.mean()

    # B→A: for each point in B, find nearest in A
    dist_b2a = dist.transpose(1, 2)  # [B, M, N]
    if mask_a is not None:
        dist_b2a = dist_b2a.masked_fill(
            ~mask_a.unsqueeze(1), float("inf"),
        )  # [B, M, N]

    min_b2a = dist_b2a.min(dim=2).values  # [B, M]

    if mask_b is not None:
        min_b2a = min_b2a * mask_b.float()
        count_b = mask_b.float().sum(dim=1).clamp(min=1.0)  # [B]
    else:
        count_b = torch.tensor(M, dtype=cloud_b.dtype, device=cloud_b.device)

    cd_b2a = min_b2a.sum(dim=1) / count_b  # [B]

    return (cd_a2b + cd_b2a).mean()
