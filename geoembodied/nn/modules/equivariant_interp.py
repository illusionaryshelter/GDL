# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Equivariant interpolation (upsampling) for multi-scale SE(3) networks.

Implements KNN distance-weighted interpolation to upsample features
from a coarse point cloud back to a dense point cloud. This is the
decoder counterpart to EquivariantPool.

The interpolation is purely geometric — no learnable parameters.
Weights are computed from inverse-distance weighting (IDW), which
is SO(3)-invariant by construction.

Architecture:
    1. For each dense point, find K nearest neighbors in coarse cloud
    2. Compute inverse-distance weights (1/d normalized)
    3. Weighted sum of coarse features → dense features
    4. Optionally add skip-connection features from encoder

Usage:
    interp = EquivariantInterpolate(k_neighbors=3)
    s_dense, v_dense = interp(
        dense_pos, coarse_pos, s_coarse, v_coarse,
        ptr_dense, ptr_coarse, s_skip=s_enc, v_skip=v_enc
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from geoembodied.functional.knn import knn


class EquivariantInterpolate(nn.Module):
    """KNN inverse-distance weighted upsampling for equivariant features.

    No learnable parameters — purely geometric interpolation.
    Equivariance is guaranteed: IDW weights depend only on distances
    (SO(3)-invariant), and weighted-sum preserves vector transformation.

    Args:
        k_neighbors: Number of nearest neighbors for interpolation (typically 3)
    """

    def __init__(self, k_neighbors: int = 3) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors

    def forward(
        self,
        dense_pos: Tensor,
        coarse_pos: Tensor,
        s_coarse: Tensor,
        v_coarse: Tensor,
        ptr_dense: Tensor,
        ptr_coarse: Tensor,
        s_skip: Optional[Tensor] = None,
        v_skip: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Interpolate features from coarse to dense resolution.

        Args:
            dense_pos: Dense (target) point positions
                shape: [N_dense, 3]
            coarse_pos: Coarse (source) point positions
                shape: [N_coarse, 3]
            s_coarse: Coarse scalar features
                shape: [N_coarse, C_s]
            v_coarse: Coarse vector features
                shape: [N_coarse, C_v, 3], representation: SO(3) type-1
            ptr_dense: CSR offsets for dense batch
                shape: [B+1], int64
            ptr_coarse: CSR offsets for coarse batch
                shape: [B+1], int64
            s_skip: Optional skip-connection scalar features (from encoder)
                shape: [N_dense, C_s_skip]
            v_skip: Optional skip-connection vector features (from encoder)
                shape: [N_dense, C_v_skip, 3], representation: SO(3) type-1

        Returns:
            s_out: Interpolated (+ skip) scalar features
                shape: [N_dense, C_s] or [N_dense, C_s + C_s_skip]
            v_out: Interpolated (+ skip) vector features
                shape: [N_dense, C_v, 3] or [N_dense, C_v + C_v_skip, 3]
        """
        # ── 1. KNN: find nearest coarse neighbors for each dense point ──
        knn_idx, knn_dists = knn(
            dense_pos, coarse_pos, ptr_dense, ptr_coarse, self.k_neighbors
        )  # [N_dense, K], [N_dense, K]

        # ── 2. Compute inverse-distance weights ──
        valid_mask = knn_idx >= 0  # [N_dense, K]

        # Inverse distance with robust clamp (Rule 4)
        # knn_dists = squared distances.  When d² < 1e-4 (d < 0.01),
        # the point is nearly co-located — FP32 distance noise (~4e-6)
        # would cause huge weight oscillation through 1/d. Clamping d²
        # at 1e-4 bounds relative weight error to < 0.01%.
        inv_dist = 1.0 / knn_dists.clamp(min=1e-4).sqrt()  # [N_dense, K]

        # Mask invalid neighbors
        inv_dist = inv_dist.masked_fill(~valid_mask, 0.0)

        # Normalize weights per query point
        weight_sum = inv_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = inv_dist / weight_sum  # [N_dense, K]

        # ── 3. Gather and aggregate ──
        safe_idx = knn_idx.clamp(min=0)

        # Scalar interpolation: [N_dense, K, C_s] → [N_dense, C_s]
        nbr_scalars = s_coarse[safe_idx]
        s_interp = (weights.unsqueeze(-1) * nbr_scalars).sum(dim=1)

        # Vector interpolation: [N_dense, K, C_v, 3] → [N_dense, C_v, 3]
        nbr_vectors = v_coarse[safe_idx]
        v_interp = (weights.unsqueeze(-1).unsqueeze(-1) * nbr_vectors).sum(dim=1)

        # Handle fully invalid rows (should not happen if data is correct)
        all_invalid = ~valid_mask.any(dim=1)
        if all_invalid.any():
            s_interp[all_invalid] = 0.0
            v_interp[all_invalid] = 0.0

        # ── 4. Skip connection (concatenate) ──
        if s_skip is not None:
            s_interp = torch.cat([s_interp, s_skip], dim=-1)  # [N_dense, C_s + C_s_skip]

        if v_skip is not None:
            v_interp = torch.cat([v_interp, v_skip], dim=1)  # [N_dense, C_v + C_v_skip, 3]

        return s_interp, v_interp

    def extra_repr(self) -> str:
        return f"k={self.k_neighbors}"
