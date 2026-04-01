# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Inlier prediction and robust registration head.

Two key robustness modules:

1. InlierPredictor: Uses SE(3)-invariant features (scalars + vector norms)
   to predict per-point inlier probability W ∈ [0, 1]. Points with low
   confidence (noise, outliers, occluded boundary) get downweighted.

2. RobustRegistrationHead: Wraps Sinkhorn + Weighted SVD solvers with
   learnable temperature and dustbin parameters for robust registration.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geoembodied.nn.solvers.sinkhorn import sinkhorn_log_domain
from geoembodied.nn.solvers.weighted_svd import (
    weighted_svd,
    weighted_svd_batched,
)


class InlierPredictor(nn.Module):
    """Predict per-point inlier probability from equivariant features.

    Takes scalar features (l=0, invariant) and vector feature norms
    (‖l=1‖², also invariant) and outputs W ∈ [0, 1] per point.

    Architecture:
        [scalars | vector_norms] → MLP → sigmoid → W

    Why this is equivariant-safe:
        - Scalar features are already SO(3)-invariant
        - ‖v‖² is rotationally invariant by construction
        - The output W is a scalar → invariant
        - No equivariance violation: we only use invariant quantities

    Args:
        scalar_channels: Number of scalar feature channels
        vector_channels: Number of vector feature channels
        hidden_dim: MLP hidden dimension
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        in_dim = scalar_channels + vector_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
    ) -> Tensor:
        """Predict inlier probability per point.

        Args:
            scalars: Scalar features (l=0)
                shape: [N, C_s], representation: SO(3) invariant
            vectors: Vector features (l=1)
                shape: [N, C_v, 3], representation: SO(3) equivariant type-1

        Returns:
            Inlier probability W ∈ [0, 1]
                shape: [N, 1]
        """
        # Vector norms: ||v_c||^2 for each channel → [N, C_v]
        # This is SO(3)-invariant (Rule 3 compliant)
        vector_norms = vectors.pow(2).sum(dim=-1)  # [N, C_v]

        # Concatenate invariant features
        features = torch.cat([scalars, vector_norms], dim=-1)  # [N, C_s + C_v]

        # MLP → sigmoid for probability
        logits = self.mlp(features)  # [N, 1]
        return torch.sigmoid(logits)


class RobustRegistrationHead(nn.Module):
    """Robust registration with Sinkhorn matching and inlier weighting.

    Wraps the parameter-free Sinkhorn and WeightedSVD solvers with
    learnable temperature (logit_scale) and dustbin parameters.

    Supports both single-pair and batched operation.

    Args:
        descriptor_dim: Feature descriptor dimension
        temperature: Sinkhorn temperature
        sinkhorn_iters: Sinkhorn iteration count
        use_dust_bin: Enable dust bin for partial overlap
    """

    def __init__(
        self,
        descriptor_dim: int = 32,
        temperature: float = 0.05,
        sinkhorn_iters: int = 10,
        use_dust_bin: bool = True,
    ) -> None:
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.use_dust_bin = use_dust_bin

        # CLIP-style learnable logit scale.
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature))
        )

        # Learnable dustbin score (SuperGlue-style, init=1.0).
        self.dustbin_score = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        desc_src: Tensor,
        desc_tgt: Tensor,
        pos_src: Tensor,
        pos_tgt: Tensor,
        weights_src: Optional[Tensor] = None,
        weights_tgt: Optional[Tensor] = None,
        mask_src: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Estimate rigid transform with robust matching.

        Supports both single [N, D] and batched [B, N, D] inputs.

        Args:
            desc_src: Source descriptors [N, D] or [B, N, D]
            desc_tgt: Target descriptors [M, D] or [B, M, D]
            pos_src: Source positions [N, 3] or [B, N, 3]
            pos_tgt: Target positions [M, 3] or [B, M, 3]
            weights_src: Source inlier weights [N, 1] or [B, N, 1] (optional)
            weights_tgt: Target inlier weights [M, 1] or [B, M, 1] (optional)
            mask_src: Source valid mask [N] or [B, N] bool (optional)
            mask_tgt: Target valid mask [M] or [B, M] bool (optional)

        Returns:
            R_pred: Predicted rotation [3, 3] or [B, 3, 3]
            t_pred: Predicted translation [3] or [B, 3]
            assignment_full: Sinkhorn assignment [N, M+1] or [B, N, M+1]
        """
        batched = desc_src.dim() == 3

        # 1. L2-normalized cosine similarity
        desc_src_n = F.normalize(desc_src, dim=-1)
        desc_tgt_n = F.normalize(desc_tgt, dim=-1)

        if batched:
            similarity = torch.bmm(desc_src_n, desc_tgt_n.transpose(1, 2))
        else:
            similarity = desc_src_n @ desc_tgt_n.T  # [N, M]

        # 2. CLIP-style learnable logit scaling
        logit_scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        similarity_scaled = similarity * logit_scale

        # 3. Sinkhorn optimal transport (using solver)
        if self.use_dust_bin:
            assignment, assignment_full = sinkhorn_log_domain(
                similarity_scaled,
                num_iters=self.sinkhorn_iters,
                temperature=1.0,
                dust_bin=self.dustbin_score,
                mask_row=mask_src,
                mask_col=mask_tgt,
                return_full=True,
            )
        else:
            sim_t = similarity_scaled
            if mask_tgt is not None:
                pad_mask = ~mask_tgt
                if batched:
                    sim_t = sim_t.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
                else:
                    sim_t = sim_t.masked_fill(pad_mask.unsqueeze(0), float("-inf"))
            assignment = F.softmax(sim_t, dim=-1)
            assignment_full = assignment

        # 4. Compute weighted target positions
        if batched:
            pos_tgt_weighted = torch.bmm(assignment, pos_tgt)
        else:
            pos_tgt_weighted = assignment @ pos_tgt

        # 5. Per-point confidence = assignment_max × inlier_weight
        match_confidence = assignment.max(dim=-1).values

        if weights_src is not None:
            match_confidence = match_confidence * weights_src.squeeze(-1)

        # 6. Zero out padding confidence
        if mask_src is not None:
            match_confidence = match_confidence * mask_src.float()

        # 7. Weighted SVD (using solver)
        if batched:
            R_pred, t_pred = weighted_svd_batched(
                pos_src, pos_tgt_weighted, match_confidence,
            )
        else:
            R_pred, t_pred = weighted_svd(
                pos_src, pos_tgt_weighted, match_confidence,
            )

        return R_pred, t_pred, assignment_full
