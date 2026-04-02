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
    # With type-2:
    s_dense, v_dense, t2_dense = interp(
        dense_pos, coarse_pos, s_coarse, v_coarse,
        ptr_dense, ptr_coarse, s_skip=s_enc, v_skip=v_enc,
        t2_coarse=t2_c, t2_skip=t2_s
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
    (SO(3)-invariant), and weighted-sum preserves transformation for any l.

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
        t2_coarse: Optional[Tensor] = None,
        t2_skip: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        """Interpolate features from coarse to dense resolution.

        Args:
            dense_pos: [N_dense, 3]
            coarse_pos: [N_coarse, 3]
            s_coarse: [N_coarse, C_s]
            v_coarse: [N_coarse, C_v, 3], representation: SO(3) type-1
            ptr_dense: [B+1], int64
            ptr_coarse: [B+1], int64
            s_skip: Optional [N_dense, C_s_skip]
            v_skip: Optional [N_dense, C_v_skip, 3], representation: SO(3) type-1
            t2_coarse: Optional [N_coarse, C_t2, 5], representation: SO(3) type-2
            t2_skip: Optional [N_dense, C_t2_skip, 5], representation: SO(3) type-2

        Returns:
            Without type-2: (s_out, v_out)
            With type-2: (s_out, v_out, t2_out)
        """
        # ── 1. KNN ──
        knn_idx, knn_dists = knn(
            dense_pos, coarse_pos, ptr_dense, ptr_coarse, self.k_neighbors
        )

        # ── 2. IDW weights ──
        valid_mask = knn_idx >= 0
        inv_dist = 1.0 / knn_dists.clamp(min=1e-4).sqrt()
        inv_dist = inv_dist.masked_fill(~valid_mask, 0.0)
        weight_sum = inv_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = inv_dist / weight_sum

        # ── 3. Gather and aggregate ──
        safe_idx = knn_idx.clamp(min=0)

        # Scalar interpolation
        nbr_scalars = s_coarse[safe_idx]
        s_interp = (weights.unsqueeze(-1) * nbr_scalars).sum(dim=1)

        # Vector interpolation
        nbr_vectors = v_coarse[safe_idx]
        v_interp = (weights.unsqueeze(-1).unsqueeze(-1) * nbr_vectors).sum(dim=1)

        # Type-2 interpolation
        t2_interp = None
        if t2_coarse is not None:
            nbr_t2 = t2_coarse[safe_idx]
            t2_interp = (weights.unsqueeze(-1).unsqueeze(-1) * nbr_t2).sum(dim=1)

        # Handle invalid rows
        all_invalid = ~valid_mask.any(dim=1)
        if all_invalid.any():
            s_interp[all_invalid] = 0.0
            v_interp[all_invalid] = 0.0
            if t2_interp is not None:
                t2_interp[all_invalid] = 0.0

        # ── 4. Skip connections (concatenate) ──
        if s_skip is not None:
            s_interp = torch.cat([s_interp, s_skip], dim=-1)

        if v_skip is not None:
            v_interp = torch.cat([v_interp, v_skip], dim=1)

        if t2_interp is not None:
            if t2_skip is not None:
                t2_interp = torch.cat([t2_interp, t2_skip], dim=1)
            return s_interp, v_interp, t2_interp

        return s_interp, v_interp

    def extra_repr(self) -> str:
        return f"k={self.k_neighbors}"
