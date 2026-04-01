# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Sinkhorn optimal transport — differentiable matching solver.

A universal module for computing soft correspondence between two sets
of descriptors. Used in:
- Point cloud registration (matching source ↔ target features)
- Vision-Language-Action (aligning visual features to action tokens)
- Reinforcement learning (feature alignment across modalities)

Operates in **log domain** for numerical stability with dust-bin
support for partial overlap/outlier rejection.

No learnable parameters — this is a pure mathematical solver.
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def sinkhorn_log_domain(
    similarity: Tensor,
    num_iters: int = 10,
    temperature: float = 0.05,
    dust_bin: float = -5.0,
    mask_row: Optional[Tensor] = None,
    mask_col: Optional[Tensor] = None,
    return_full: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Sinkhorn optimal transport in log domain for numerical stability.

    Supports both single [N, M] and batched [B, N, M] inputs.
    Mask support ensures padding positions are excluded from transport.

    Args:
        similarity: Raw similarity matrix [N, M] or [B, N, M]
        num_iters: Number of Sinkhorn iterations
        temperature: Softmax temperature (lower = sharper)
        dust_bin: Log-space score for the unmatched bin
        mask_row: Boolean mask for rows [B, N] or [N].
            True = real point, False = padding. If provided,
            padding rows get -inf before normalization.
        mask_col: Boolean mask for columns [B, M] or [M].
            True = real point, False = padding. If provided,
            padding columns get -inf before normalization.

        return_full: If True, also return the full assignment matrix
            including the dustbin column [N, M+1] or [B, N, M+1].
            Needed for correspondence supervision with outlier→dustbin routing.

    Returns:
        If return_full=False: Assignment matrix [N, M] or [B, N, M].
        If return_full=True: Tuple of (assignment, assignment_full) where
            assignment_full has shape [N, M+1] or [B, N, M+1].
    """
    batched = similarity.dim() == 3
    if not batched:
        similarity = similarity.unsqueeze(0)  # [1, N, M]
        if mask_row is not None:
            mask_row = mask_row.unsqueeze(0)
        if mask_col is not None:
            mask_col = mask_col.unsqueeze(0)

    B, N, M = similarity.shape
    device = similarity.device

    # Mask padding positions with -inf BEFORE temperature scaling.
    # This ensures padding descriptors never receive transport mass.
    if mask_col is not None:
        # mask_col: [B, M] → [B, 1, M] — broadcast over rows
        similarity = similarity.masked_fill(
            ~mask_col.unsqueeze(1), float("-inf"),
        )
    if mask_row is not None:
        # mask_row: [B, N] → [B, N, 1] — broadcast over cols
        similarity = similarity.masked_fill(
            ~mask_row.unsqueeze(2), float("-inf"),
        )

    # Scale by temperature
    log_alpha = similarity / temperature  # [B, N, M]

    # Add dust bin row and column for partial overlap.
    # Use expand (not torch.full) to support nn.Parameter dust_bin
    # while keeping the autograd graph connected.
    if isinstance(dust_bin, Tensor):
        # nn.Parameter or Tensor: broadcast via expand for grad flow
        dust_val = dust_bin.reshape(1, 1, 1).expand(B, 1, M)
        dust_row = dust_val  # [B, 1, M]
        dust_col = dust_bin.reshape(1, 1, 1).expand(B, N + 1, 1)
    else:
        # Plain float: use torch.full (no grad needed)
        dust_row = torch.full((B, 1, M), dust_bin, device=device)
        dust_col = torch.full((B, N + 1, 1), dust_bin, device=device)
    log_alpha = torch.cat([log_alpha, dust_row], dim=1)  # [B, N+1, M]
    log_alpha = torch.cat([log_alpha, dust_col], dim=2)  # [B, N+1, M+1]

    # Sinkhorn iterations in log domain (batched).
    # Each iteration alternates row and column normalization.
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

    # Final row normalization to ensure each source point's assignment
    # is a proper probability distribution over target points + dustbin.
    # Without this, the last step (col norm) leaves row sums ≈ N/(M+1) ≠ 1.
    log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)

    # Convert back to probability, discard dust bin
    assignment = log_alpha[:, :N, :M].exp()

    if return_full:
        # Include dustbin column: [B, N, M+1]
        # Source rows × (target cols + dustbin).
        # For correspondence loss, outlier source points should have
        # high probability at column M (the dustbin column).
        assignment_full = log_alpha[:, :N, :].exp()  # [B, N, M+1]
        if not batched:
            return assignment.squeeze(0), assignment_full.squeeze(0)
        return assignment, assignment_full

    if not batched:
        return assignment.squeeze(0)
    return assignment
