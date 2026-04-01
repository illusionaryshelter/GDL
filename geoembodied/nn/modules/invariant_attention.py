# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3)-invariant attention for geometric feature aggregation.

Uses SE(3)-invariant quantities (distances, relative angles) as attention
biases while preserving equivariance of value aggregation.

The attention weights are invariant: attn(R⊳cloud) = attn(cloud)
The value aggregation preserves equivariance for vector features.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geoembodied.functional.radius_graph import radius_graph, compute_edge_vectors


class InvariantAttention(nn.Module):
    """SE(3)-invariant self-attention with geometric bias.

    Computes attention using invariant features (distances, angles),
    then aggregates both scalar and vector features.

    Attention mechanism:
        attn_ij = softmax_j( (Q_i · K_j) / √d + bias(d_ij) )

    where:
        - Q, K are derived from scalar features (invariant)
        - bias(d_ij) encodes the Euclidean distance (invariant)
        - Aggregation: out_j = Σ_i attn_ij * V_i

    For vector features, values are linearly mixed but aggregated
    with the same (invariant) attention weights, preserving equivariance.

    Args:
        scalar_channels: Dimension of scalar features
        vector_channels: Dimension of vector features
        num_heads: Number of attention heads
        radius: Neighborhood radius for sparse attention
        max_num_neighbors: Max neighbors per node
        num_distance_basis: Number of RBF basis for distance encoding

    Example::

        >>> attn = InvariantAttention(
        ...     scalar_channels=64, vector_channels=16,
        ...     num_heads=4, radius=2.0,
        ... )
        >>> pos = torch.randn(100, 3)
        >>> s, v = torch.randn(100, 64), torch.randn(100, 16, 3)
        >>> s_out, v_out = attn(pos, s, v)
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int = 0,
        num_heads: int = 4,
        radius: float = 5.0,
        max_num_neighbors: int = 32,
        num_distance_basis: int = 16,
    ) -> None:
        super().__init__()
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.num_heads = num_heads
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        assert scalar_channels % num_heads == 0, \
            f"scalar_channels ({scalar_channels}) must be divisible by num_heads ({num_heads})"
        self.head_dim = scalar_channels // num_heads

        # Q, K, V projections for scalar features
        self.q_proj = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.k_proj = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.v_proj = nn.Linear(scalar_channels, scalar_channels, bias=False)
        self.o_proj = nn.Linear(scalar_channels, scalar_channels, bias=True)

        # Vector value projection (if vector features exist)
        if vector_channels > 0:
            self.v_vec_proj = nn.Linear(vector_channels, vector_channels, bias=False)
            self.o_vec_proj = nn.Linear(vector_channels, vector_channels, bias=False)
        else:
            self.v_vec_proj = None
            self.o_vec_proj = None

        # Distance bias: RBF encoding → per-head bias
        self.dist_basis_freqs = nn.Parameter(
            torch.linspace(0.0, radius, num_distance_basis).unsqueeze(0),
            requires_grad=False,
        )
        self.dist_basis_sigma = radius / num_distance_basis
        self.dist_mlp = nn.Sequential(
            nn.Linear(num_distance_basis, num_heads),
        )

    def forward(
        self,
        pos: Tensor,
        scalars: Tensor,
        vectors: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply invariant attention.

        Args:
            pos: Node positions, shape: [N, 3]
            scalars: Scalar features, shape: [N, C_s]
            vectors: Vector features, shape: [N, C_v, 3]
            batch: Batch assignment, shape: [N], optional

        Returns:
            Tuple of (scalar_out, vector_out)
        """
        N = pos.shape[0]
        device = pos.device
        H = self.num_heads
        D = self.head_dim

        # 1. Build sparse attention graph
        row, col = radius_graph(
            pos, self.radius,
            max_num_neighbors=self.max_num_neighbors,
            batch=batch,
        )
        E = row.shape[0]

        if E == 0:
            return scalars, vectors

        # 2. Q, K, V projections
        Q = self.q_proj(scalars).reshape(N, H, D)  # [N, H, D]
        K = self.k_proj(scalars).reshape(N, H, D)
        V = self.v_proj(scalars).reshape(N, H, D)

        # 3. Compute attention scores (only for edges)
        Q_edge = Q[col]  # [E, H, D] — query from target
        K_edge = K[row]  # [E, H, D] — key from source

        # Scaled dot product: [E, H]
        attn_scores = (Q_edge * K_edge).sum(dim=-1) / math.sqrt(D)

        # 4. Distance bias (SE(3)-invariant)
        diff, dist, direction = compute_edge_vectors(pos, row, col)

        # Gaussian RBF: [E, B]
        dist_expanded = dist.unsqueeze(-1)  # [E, 1]
        rbf = torch.exp(
            -0.5 * ((dist_expanded - self.dist_basis_freqs) / self.dist_basis_sigma) ** 2
        )  # [E, B]

        dist_bias = self.dist_mlp(rbf)  # [E, H]
        attn_scores = attn_scores + dist_bias

        # 5. Sparse softmax (per-target normalization)
        # For each target node j, softmax over its neighbors
        attn_weights = _sparse_softmax(attn_scores, col, N)  # [E, H]

        # 6. Aggregate scalar values
        V_edge = V[row]  # [E, H, D] — values from source
        weighted_V = attn_weights.unsqueeze(-1) * V_edge  # [E, H, D]
        weighted_V_flat = weighted_V.reshape(E, -1)  # [E, H*D]

        s_out = torch.zeros(N, H * D, device=device, dtype=scalars.dtype)
        s_out.scatter_add_(
            0,
            col.unsqueeze(-1).expand_as(weighted_V_flat),
            weighted_V_flat,
        )
        s_out = self.o_proj(s_out)

        # 7. Aggregate vector values (with same invariant attention weights)
        if self.v_vec_proj is not None and vectors.shape[-2] > 0:
            V_vec = self.v_vec_proj(
                vectors[row].transpose(-1, -2)
            ).transpose(-1, -2)  # [E, C_v, 3]

            # Use mean attention weight across heads for vector aggregation
            attn_vec = attn_weights.mean(dim=-1, keepdim=True)  # [E, 1]
            weighted_V_vec = attn_vec.unsqueeze(-1) * V_vec  # [E, C_v, 3]

            v_out = torch.zeros(N, self.vector_channels, 3, device=device, dtype=vectors.dtype)
            v_out.scatter_add_(
                0,
                col.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_V_vec),
                weighted_V_vec,
            )
            v_out_mixed = self.o_vec_proj(v_out.transpose(-1, -2)).transpose(-1, -2)
        else:
            v_out_mixed = vectors

        return s_out, v_out_mixed


def _sparse_softmax(scores: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Compute softmax over sparse edges grouped by target node.

    Args:
        scores: Attention scores, shape: [E, H]
        index: Target node indices, shape: [E]
        num_nodes: Total number of nodes N

    Returns:
        Softmax weights, shape: [E, H]
    """
    # Numerical stability: subtract max per target
    max_scores = torch.zeros(num_nodes, scores.shape[-1], device=scores.device)
    max_scores.scatter_reduce_(
        0,
        index.unsqueeze(-1).expand_as(scores),
        scores,
        reduce='amax',
        include_self=False,
    )
    scores = scores - max_scores[index]

    exp_scores = scores.exp()

    # Sum per target node
    sum_exp = torch.zeros(num_nodes, scores.shape[-1], device=scores.device)
    sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)

    return exp_scores / sum_exp[index].clamp(min=1e-8)
