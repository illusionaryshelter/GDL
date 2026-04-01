# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""K-Nearest Neighbor search for packed point clouds.

Public API for KNN that automatically dispatches to CUDA or CPU:
- CUDA path: JIT-compiled brute-force with MinK heap (see csrc/knn_kernel.cu)
- CPU path: PyTorch torch.cdist + topk fallback

All functions use **ptr** (CSR offsets, [B+1]) for batch segmentation,
consistent with the PointCloudBatch standard.

Usage:
    from geoembodied.functional.knn import knn, knn_self

    # Asymmetric KNN (query ≠ source)
    indices, dists = knn(query_pos, source_pos, ptr_q, ptr_s, k=16)

    # Self-KNN (includes self as nearest, slice [:, 1:] to exclude)
    indices, dists = knn_self(pos, ptr, k=8)
"""

from typing import Tuple

import torch
from torch import Tensor

from geoembodied.csrc import knn_cuda as _knn_dispatch


def knn(
    query: Tensor,
    source: Tensor,
    ptr_q: Tensor,
    ptr_s: Tensor,
    k: int,
) -> Tuple[Tensor, Tensor]:
    """K-Nearest Neighbor search with ptr-based batch segmentation.

    Automatically dispatches to CUDA or CPU fallback.

    Args:
        query: Query points
            shape: [N_q, 3], float32
        source: Source points
            shape: [N_s, 3], float32
        ptr_q: CSR offsets for query batches
            shape: [B+1], int64
        ptr_s: CSR offsets for source batches
            shape: [B+1], int64
        k: Number of nearest neighbors

    Returns:
        indices: Global indices into source
            shape: [N_q, k], int64
            -1 for positions where fewer than k neighbors exist
        dists: Squared L2 distances
            shape: [N_q, k], float32
            1e30 for invalid positions
    """
    return _knn_dispatch(query, source, ptr_q, ptr_s, k)


def knn_self(
    pos: Tensor,
    ptr: Tensor,
    k: int,
) -> Tuple[Tensor, Tensor]:
    """Self-KNN: find k nearest neighbors within each batch element.

    The first neighbor ([:, 0]) is always self (distance ≈ 0).
    Use k+1 and slice [:, 1:] to exclude self-loops.

    Args:
        pos: Packed point positions
            shape: [N_total, 3], float32
        ptr: CSR offsets
            shape: [B+1], int64
        k: Number of neighbors (including self)

    Returns:
        indices: [N_total, k] int64
        dists: [N_total, k] float32
    """
    return _knn_dispatch(pos, pos, ptr, ptr, k)
