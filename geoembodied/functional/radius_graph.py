# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Radius graph construction — multi-backend dispatcher.

Backend selection (automatic):
    1. CUDA hash-grid: O(N log N) — for N > ~2K on GPU, requires JIT compilation
    2. PyTorch brute-force: O(N²) — fallback for CPU or small N

The public interface is stable; only the backend changes.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def radius_graph(
    x: Tensor,
    r: float,
    *,
    max_num_neighbors: int = 32,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    mask: Optional[Tensor] = None,
    num_batch_elements: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Build radius graph: find all pairs (i, j) with ‖xᵢ - xⱼ‖ < r.

    Returns edges in COO format (source, target) for message passing.
    Automatically dispatches to CUDA hash-grid or PyTorch brute-force.

    Args:
        x: Point positions
            shape: [N, 3]
        r: Radius cutoff (Euclidean distance)
        max_num_neighbors: Maximum neighbors per node (caps output size)
        batch: Batch vector for multi-graph batching
            shape: [N], dtype: int64, values in [0, B-1]
            If None, treats all points as one graph.
        loop: Whether to include self-loops (i == j)
        mask: Boolean mask for valid points
            shape: [N], dtype: bool
            If provided, pad points (mask=False) are excluded from both
            sending and receiving edges via algebraic masking (dist=inf).

    Returns:
        Tuple of (row, col) tensors, each shape: [E]
        row[k] → source node, col[k] → target node
        Satisfying ‖x[row[k]] - x[col[k]]‖ < r

    Example::

        >>> x = torch.randn(100, 3)
        >>> row, col = radius_graph(x, r=0.5, max_num_neighbors=16)
        >>> # row, col: edge indices for message passing
    """
    # Try CUDA backend for GPU tensors
    if x.is_cuda:
        try:
            from geoembodied.csrc import get_radius_graph_backend, radius_graph_cuda

            if get_radius_graph_backend() == "cuda":
                # Determine batch layout
                B = num_batch_elements
                if B is None and batch is not None:
                    B = int(batch.max().item()) + 1

                if B is not None and B > 1:
                    # CUDA kernel requires equal-size batch elements.
                    # For variable-size batches (diagonal concatenation),
                    # we MUST fall through to the brute-force backend
                    # which handles per-element processing correctly.
                    counts = torch.bincount(batch, minlength=B)
                    if counts.min() == counts.max():
                        N_per = x.shape[0] // B
                        return radius_graph_cuda(
                            x, r,
                            max_num_neighbors=max_num_neighbors,
                            loop=loop,
                            mask=mask,
                            batch_size=B,
                            points_per_batch=N_per,
                        )
                    # else: variable-size → fall through to brute-force
                else:
                    return radius_graph_cuda(
                        x, r,
                        max_num_neighbors=max_num_neighbors,
                        loop=loop,
                        mask=mask,
                    )
        except Exception:
            pass  # Fall through to brute-force

    # Brute-force fallback (CPU or CUDA without compiled kernel)
    return _radius_graph_bruteforce(
        x, r,
        max_num_neighbors=max_num_neighbors,
        batch=batch,
        loop=loop,
        mask=mask,
        num_batch_elements=num_batch_elements,
    )


def _radius_graph_bruteforce(
    x: Tensor,
    r: float,
    *,
    max_num_neighbors: int = 32,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    mask: Optional[Tensor] = None,
    num_batch_elements: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """O(N²) brute-force radius graph — Phase 1 backend.

    Computes pairwise distances and filters by radius. Supports two
    execution paths:

    1. **Batched fast path** (equal-size elements): ONE batched cdist
       [B, N, N] → topk → global edges.  10-30x faster than looping.
    2. **Loop fallback** (unequal elements): per-element cdist.

    When mask is provided, pad points (mask=False) are algebraically
    excluded: their distances are set to inf so they never appear as
    source or target in any edge.

    Args:
        x: Point positions, shape: [N_total, 3] (may contain multiple batches)
        r: Radius cutoff
        max_num_neighbors: Max neighbors per node (KNN safety cap)
        batch: Batch assignment, shape: [N_total], values in [0, B-1]
        loop: Include self-loops
        mask: Boolean mask [N_total], True = real, False = pad
        num_batch_elements: If provided, skip GPU-sync detection and go
            directly to the batched fast path with B = num_batch_elements.

    Returns:
        (row, col) edge index tensors in global indices
    """
    N = x.shape[0]
    device = x.device

    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)

    # Caller-provided B → skip all GPU-sync detection
    if num_batch_elements is not None:
        B = num_batch_elements
        if B == 1:
            return _radius_graph_single(
                x, r, max_num_neighbors=max_num_neighbors,
                loop=loop, mask=mask, offset=0,
            )
        N_per = N // B
        return _radius_graph_batched(
            x, r, B, N_per,
            max_num_neighbors=max_num_neighbors,
            loop=loop, mask=mask,
        )

    # Auto-detect: requires GPU sync (unique, bincount)
    batch_ids = batch.unique()
    if batch_ids.numel() == 1:
        return _radius_graph_single(
            x, r, max_num_neighbors=max_num_neighbors,
            loop=loop, mask=mask, offset=0,
        )

    # Check if all batch elements have equal size (common in training)
    B = batch_ids.numel()
    counts = torch.bincount(batch, minlength=B)
    if counts.min() == counts.max():
        # ── Batched fast path: one cdist [B, N, N] ──
        N_per = N // B
        return _radius_graph_batched(
            x, r, B, N_per,
            max_num_neighbors=max_num_neighbors,
            loop=loop, mask=mask,
        )

    # Multi-batch with unequal sizes: process each batch element separately
    all_rows, all_cols = [], []
    for bid in batch_ids:
        sel = (batch == bid).nonzero(as_tuple=True)[0]  # global indices
        x_b = x[sel]  # [N_b, 3]
        mask_b = mask[sel] if mask is not None else None

        row_local, col_local = _radius_graph_single(
            x_b, r, max_num_neighbors=max_num_neighbors,
            loop=loop, mask=mask_b, offset=0,
        )

        # Remap local → global indices
        all_rows.append(sel[row_local])
        all_cols.append(sel[col_local])

    return torch.cat(all_rows), torch.cat(all_cols)


def _radius_graph_batched(
    x: Tensor,
    r: float,
    B: int,
    N: int,
    *,
    max_num_neighbors: int = 32,
    loop: bool = False,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Batched radius graph for equal-size point clouds.

    Uses ONE batched cdist [B, N, N] instead of B sequential calls.
    10-30x faster on GPU for typical training batch sizes.

    Args:
        x: Flattened positions [B*N, 3]
        r: Radius cutoff
        B: Batch size
        N: Points per batch element
        max_num_neighbors: KNN cap
        loop: Include self-loops
        mask: [B*N] bool, True = real

    Returns:
        (row, col) global edge indices
    """
    device = x.device
    r_sq = r * r
    K = min(max_num_neighbors, N)

    # Reshape to [B, N, 3] for batched cdist
    x_3d = x.reshape(B, N, 3)
    dist_sq = torch.cdist(x_3d, x_3d, p=2.0).pow(2)  # [B, N, N]

    # Algebraic mask: pad points → distance = inf
    if mask is not None:
        mask_2d = mask.reshape(B, N)
        invalid = ~mask_2d
        dist_sq.masked_fill_(
            invalid.unsqueeze(2) | invalid.unsqueeze(1), float('inf'),
        )

    # Set out-of-radius and self-loops to inf (in-place, fused kernel)
    dist_sq.masked_fill_(dist_sq >= r_sq, float('inf'))

    if not loop:
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        dist_sq.masked_fill_(eye, float('inf'))

    # topk: find K nearest along dim=1 (source dimension)
    topk_vals, topk_indices = dist_sq.topk(
        K, dim=1, largest=False,
    )  # [B, K, N]

    # Build edges — vectorized across batch
    valid = topk_vals < float('inf')  # [B, K, N]

    # Global offsets per batch element
    offsets = torch.arange(B, device=device).view(B, 1, 1) * N  # [B, 1, 1]

    # Source indices (from topk_indices) + offset
    global_src = topk_indices + offsets  # [B, K, N]

    # Target indices: column grid + offset
    col_grid = torch.arange(N, device=device).view(1, 1, N).expand(B, K, N)
    global_tgt = col_grid + offsets  # [B, K, N]

    row = global_src[valid]
    col = global_tgt[valid]

    return row, col


def _radius_graph_single(
    x: Tensor,
    r: float,
    *,
    max_num_neighbors: int = 32,
    loop: bool = False,
    mask: Optional[Tensor] = None,
    offset: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Radius graph for a single point cloud (no batch dimension).

    Args:
        x: [N, 3]
        r: Radius cutoff
        max_num_neighbors: KNN cap
        loop: Include self-loops
        mask: [N] bool, True = real
        offset: Index offset for global indexing (unused, kept for API)

    Returns:
        (row, col) — local indices
    """
    N = x.shape[0]
    device = x.device
    r_sq = r * r

    # Pairwise squared distance: [N, N]
    dist_sq = torch.cdist(x, x, p=2.0).pow(2)

    # Algebraic mask: pad points → distance = inf
    if mask is not None:
        invalid = ~mask
        dist_sq.masked_fill_(
            invalid.unsqueeze(1) | invalid.unsqueeze(0), float('inf'),
        )

    # Set out-of-radius and self-loops to inf (in-place, fused kernel)
    dist_sq.masked_fill_(dist_sq >= r_sq, float('inf'))

    if not loop:
        eye = torch.eye(N, dtype=torch.bool, device=device)
        dist_sq.masked_fill_(eye, float('inf'))

    K = min(max_num_neighbors, N)

    # Vectorized top-k per column (target)
    topk_vals, topk_indices = dist_sq.topk(
        K, dim=0, largest=False,
    )  # [K, N]

    # Build edges fully vectorized
    valid = topk_vals < float('inf')
    col_grid = torch.arange(N, device=device).unsqueeze(0).expand(K, N)

    row = topk_indices[valid]
    col = col_grid[valid]

    return row, col


def compute_edge_vectors(
    x: Tensor,
    row: Tensor,
    col: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute edge vectors, distances, and unit directions.

    Args:
        x: Point positions, shape: [N, 3]
        row: Source indices, shape: [E]
        col: Target indices, shape: [E]

    Returns:
        diff: Edge vectors (x[col] - x[row]), shape: [E, 3]
        dist: Edge lengths ‖diff‖, shape: [E]
        direction: Unit edge vectors diff/‖diff‖, shape: [E, 3]
    """
    diff = x[col] - x[row]  # [E, 3]
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]
    direction = diff / dist  # [E, 3]
    return diff, dist.squeeze(-1), direction
