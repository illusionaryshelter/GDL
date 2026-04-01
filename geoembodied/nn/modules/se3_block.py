# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3)-equivariant interaction block.

Single interaction layer: Conv → Norm → Gate → Scaled Residual.

Post-Norm is REQUIRED for equivariant GNNs (NequIP/MACE/Allegro standard):
    Gate mechanism uses sigmoid(Linear(scalars)) to control vectors.
    The gate needs normalized scalar input to produce meaningful gates.
    Pre-Norm leaves conv output unnormalized → gate sees random-scale
    input → sigmoid ≈ 0.5 → vectors halved every layer → signal collapse.

Residual scaling by 1/√2 prevents linear norm growth:
    Without scaling: ||v_L|| ≈ ||v_0|| + L·c  (linear explosion)
    With 1/√2 scaling: fixed point v* = c/(√2-1) ≈ 2.41c (bounded)

The residual connection is algebraically safe for equivariance:
    R(v₁ + v₂) = Rv₁ + Rv₂

All internal components (SE3Conv, EquivariantLayerNorm,
GatedNonlinearity) maintain SO(3)/SE(3) equivariance by construction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from geoembodied.nn.modules.se3_conv import SE3Conv
from geoembodied.nn.modules.equivariant_norm import EquivariantLayerNorm
from geoembodied.nn.modules.gated_nonlinearity import GatedNonlinearity

if TYPE_CHECKING:
    from geoembodied.nn.modules.spatial_graph import SpatialGraph


class SE3NetBlock(nn.Module):
    """Single SE(3)-equivariant interaction block.

    Architecture::

        (s, v) ──→ SE3Conv ──→ EquivariantLayerNorm ──→ GatedNonlinearity
                                                               │
                                                     [+ skip] × 1/√2
                                                               │
                                                          (s_out, v_out)

    Post-Norm ensures the gate sees normalized input, which is critical
    for the sigmoid gate to produce meaningful (non-0.5) gate values.

    Residual scaling by 1/√2 prevents vector norm O(L) growth while
    preserving gradient flow through the identity path.

    Args:
        channels_scalar: Scalar feature channels (in = out)
            type: int
        channels_vector: Vector feature channels (in = out)
            type: int
        radius: Spatial graph radius for SE3Conv
            type: float
        max_num_neighbors: Max edges per node for SE3Conv
            type: int
        use_residual: Enable skip connection
            type: bool, default True
    """

    def __init__(
        self,
        channels_scalar: int,
        channels_vector: int,
        radius: float,
        max_num_neighbors: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.channels_scalar = channels_scalar
        self.channels_vector = channels_vector
        self.use_residual = use_residual

        # Residual scaling: 1/√2 prevents linear norm growth
        # through residual accumulation across L layers.
        # Fixed point: v* = c/(√2-1) ≈ 2.41c (bounded)
        self._rsqrt2 = 1.0 / math.sqrt(2.0)

        self.conv = SE3Conv(
            in_scalar_channels=channels_scalar,
            in_vector_channels=channels_vector,
            out_scalar_channels=channels_scalar,
            out_vector_channels=channels_vector,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
        )
        # Post-Norm: normalize AFTER conv, BEFORE gate
        # Gate requires normalized input for sigmoid to produce
        # discriminative values (not degenerate 0.5)
        self.norm = EquivariantLayerNorm(
            num_scalars=channels_scalar,
            num_vectors=channels_vector,
        )
        self.gate = GatedNonlinearity(
            num_scalars=channels_scalar,
            num_vectors=channels_vector,
        )

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        graph: SpatialGraph,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: Conv → Norm → Gate → Scaled Residual.

        Args:
            scalars: Scalar features
                shape: [N, channels_scalar], representation: SO(3) type-0
            vectors: Vector features
                shape: [N, channels_vector, 3], representation: SO(3) type-1
            graph: Pre-built spatial graph (shared across layers)

        Returns:
            (scalars_out, vectors_out) with same shapes as input
        """
        s_new, v_new = self.conv(scalars, vectors, graph)
        s_new, v_new = self.norm(s_new, v_new)
        s_new, v_new = self.gate(s_new, v_new)

        if self.use_residual:
            # Scaled residual: (conv_path + identity) / √2
            # Prevents ||v|| from growing O(L) through residual accumulation
            s_new = (s_new + scalars) * self._rsqrt2
            v_new = (v_new + vectors) * self._rsqrt2

        return s_new, v_new

    def extra_repr(self) -> str:
        return (
            f"scalar={self.channels_scalar}, "
            f"vector={self.channels_vector}, "
            f"residual={self.use_residual}"
        )
