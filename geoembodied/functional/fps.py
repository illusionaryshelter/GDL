# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Farthest Point Sampling (FPS) for point cloud keypoint extraction.

FPS selects a subset of points maximally spread across the cloud.
This is critical for robustness: instead of using ALL points
(which are sensitive to local topology changes from noise/occlusion),
we extract a set of stable, globally distributed keypoints.

Backends:
    - CUDA kernel (default on GPU): O(N × K), zero GPU→CPU sync in loop,
      1 block per batch element, warp+block argmax reduction.
    - PyTorch fallback (CPU or compile failure): O(N × K), iterative with
      GPU→CPU syncs per iteration (slow but correct).

Complexity: O(N × K) where K = number of keypoints selected.

CRITICAL: FPS uses relative distances (L2²), which are SE(3)-invariant.
  Grid voxel downsampling is ABSOLUTELY FORBIDDEN — it destroys continuous
  rotational equivariance by introducing axis-aligned discretization.
"""

import torch
from torch import Tensor
from typing import Optional


def farthest_point_sampling(
    pos: Tensor,
    num_samples: int,
    batch: Tensor = None,
    ptr: Tensor = None,
) -> Tensor:
    """Select num_samples points via farthest point sampling.

    Algorithm:
        1. Pick a starting point (first point for determinism)
        2. Repeat K-1 times:
           - Compute distance from all points to nearest selected point
           - Select the point with maximum nearest-selected distance

    This guarantees the selected subset maximally covers the point cloud,
    producing stable keypoints robust to local perturbations.

    Args:
        pos: Point positions
            shape: [N, 3]
        num_samples: Number of keypoints to select (K)
        batch: Optional batch indices (legacy API)
            shape: [N]
        ptr: Optional CSR offsets (preferred API)
            shape: [B+1], int64

    Returns:
        Indices of selected keypoints
            shape: [K] (or [B*K] if batched)
    """
    N = pos.shape[0]

    # ptr takes precedence over batch
    if ptr is not None:
        return _fps_batched_ptr(pos, num_samples, ptr)

    if batch is None:
        return _fps_single(pos, num_samples)

    # Legacy batched FPS via batch vector
    device = pos.device
    batch_size = int(batch.max().item()) + 1
    indices_list = []

    for b in range(batch_size):
        mask = batch == b
        local_pos = pos[mask]
        global_indices = torch.where(mask)[0]

        local_selection = _fps_single(local_pos, min(num_samples, local_pos.shape[0]))
        indices_list.append(global_indices[local_selection])

    return torch.cat(indices_list)


def _fps_batched_ptr(
    pos: Tensor,
    num_samples: int,
    ptr: Tensor,
) -> Tensor:
    """FPS with ptr-based batch segmentation.

    Dispatches to CUDA kernel when available (zero GPU→CPU sync in loop).
    Falls back to iterative PyTorch otherwise.

    Args:
        pos: [N_total, 3] packed point cloud
        ptr: [B+1] CSR offsets
        num_samples: K per batch element

    Returns:
        Global indices of selected points, shape: [N_out_total]
    """
    from geoembodied.csrc import fps_available, fps_cuda

    B = ptr.shape[0] - 1
    device = pos.device

    # Compute per-batch sample counts on GPU (no CPU sync)
    sizes = ptr[1:] - ptr[:-1]  # [B], on device
    k_per_batch = sizes.clamp(max=num_samples)  # [B], min(n_b, K)

    if fps_available() and pos.is_cuda:
        # CUDA kernel: 1 GPU→CPU sync total (for output allocation)
        # FPS loop itself: ZERO sync
        return fps_cuda(pos, ptr, k_per_batch)

    # Fallback: iterative PyTorch (slow but correct)
    from geoembodied.csrc import _fps_iterative_fallback
    return _fps_iterative_fallback(pos, ptr, k_per_batch)


def _fps_single(pos: Tensor, num_samples: int) -> Tensor:
    """FPS for a single (non-batched) point cloud.

    Args:
        pos: [N, 3]
        num_samples: K

    Returns:
        Selected indices, shape: [K]
    """
    N = pos.shape[0]
    device = pos.device

    if num_samples >= N:
        return torch.arange(N, device=device)

    # Try CUDA kernel for single batch
    from geoembodied.csrc import fps_available, fps_cuda

    if fps_available() and pos.is_cuda:
        ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
        k_per = torch.tensor([num_samples], dtype=torch.int64, device=device)
        return fps_cuda(pos, ptr, k_per)

    # PyTorch fallback
    selected = torch.zeros(num_samples, dtype=torch.long, device=device)
    selected[0] = 0  # deterministic start

    # Distance from each point to nearest selected point
    min_distances = torch.full((N,), float('inf'), device=device)

    for i in range(num_samples):
        # Update distances: min(current, dist to newly selected)
        current_point = pos[selected[i]]  # [3]
        dist_to_current = (pos - current_point.unsqueeze(0)).pow(2).sum(dim=-1)  # [N]
        min_distances = torch.minimum(min_distances, dist_to_current)

        if i + 1 < num_samples:
            # Select the point farthest from all selected points
            selected[i + 1] = min_distances.argmax()

    return selected
