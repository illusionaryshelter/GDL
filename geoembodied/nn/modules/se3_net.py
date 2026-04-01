# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE3Net — Multi-layer SE(3)-equivariant backbone for point clouds.

Pure feature extractor following the timm/torchvision pattern.
Task-specific heads (classification, registration, segmentation)
are separate modules that wrap this backbone.

Architecture::

    pos → Input Embedding → [SE3NetBlock × L] → (scalars, vectors)
                                  ↑
                       SpatialGraph.build (shared, built ONCE)

Initial vector features are ZERO — not a learnable projection of
positions. This is mathematically required because:

1. Positions transform as pos → R·pos + t (translation-dependent)
2. Type-1 features must be translation-invariant: v → R·v
3. Linear(pos) would inject translation into vectors, breaking SE(3)

The first SE3NetBlock generates vector features "from nothing" via
Path 1 (scalar × edge_direction → vector), which is equivariant
because edge directions (x_i - x_j) are inherently translation-invariant.

torch.compile integration::

    # Option A: Simple (includes graph build, minor graph breaks)
    model = SE3Net(...)
    s, v = model(pos, batch=batch)

    # Option B: Optimal (zero graph breaks in compiled region)
    model = SE3Net(...)
    model.forward_with_graph = torch.compile(
        model.forward_with_graph, dynamic=True
    )
    graph = SpatialGraph.build(pos, radius, batch=batch)  # outside compile
    s, v = model.forward_with_graph(pos, graph)           # 0 graph breaks

    # For 16GB+ VRAM servers, increase:
    # hidden_scalar=128, hidden_vector=32
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from geoembodied.nn.modules.se3_block import SE3NetBlock
from geoembodied.nn.modules.spatial_graph import SpatialGraph

if TYPE_CHECKING:
    pass


class SE3Net(nn.Module):
    """Multi-layer SE(3)-equivariant backbone for point clouds.

    Pure feature extractor — no task-specific output heads.
    Outputs per-node scalar and vector features that can be consumed
    by downstream task modules.

    Two forward paths:

    1. ``forward(pos, features, batch)`` — convenience API, builds graph
       internally. Has unavoidable graph breaks from dynamic edge count.

    2. ``forward_with_graph(pos, graph, features)`` — compile-friendly
       API. Graph is pre-built externally, all ops inside are fusible
       by Inductor with **zero graph breaks**.

    Args:
        in_channels: Raw input scalar feature dimension.
            Default 1 (ones vector, for coordinate-only input).
            type: int
        hidden_scalar: Hidden scalar channels per block.
            Default 32 (safe for 4GB VRAM).
            For 16GB+ VRAM, use 128 for best accuracy.
            type: int
        hidden_vector: Hidden vector channels per block.
            Default 8 (safe for 4GB VRAM).
            For 16GB+ VRAM, use 32 for best accuracy.
            type: int
        num_layers: Number of SE3NetBlock interaction layers.
            type: int
        radius: Radius for spatial graph construction.
            type: float
        max_num_neighbors: Max edges per node.
            type: int
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_scalar: int = 32,
        hidden_vector: int = 8,
        num_layers: int = 3,
        radius: float = 2.0,
        max_num_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.num_layers = num_layers
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        # Input embedding: raw scalars → hidden_scalar
        self.embed_scalar = nn.Linear(in_channels, hidden_scalar)

        # Interaction blocks (all share the same graph)
        self.blocks = nn.ModuleList([
            SE3NetBlock(
                channels_scalar=hidden_scalar,
                channels_vector=hidden_vector,
                radius=radius,
                max_num_neighbors=max_num_neighbors,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        pos: Tensor,
        features: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        num_batch_elements: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """Extract SE(3)-equivariant features (convenience API).

        Builds graph internally. For torch.compile optimization, use
        :meth:`forward_with_graph` instead.

        Args:
            pos: Point positions
                shape: [N_total, 3], float32
            features: Raw scalar features per node (optional).
                If None, uses all-ones [N_total, 1].
                shape: [N_total, in_channels]
            batch: Graph assignment per node (for batched input).
                If None, treats all nodes as single graph.
                shape: [N_total], int64
            num_batch_elements: Number of graphs in batch (optional).
                When provided, skips GPU-sync ``batch.max()`` detection
                and enables the batched fast path in radius_graph.
                type: int or None

        Returns:
            scalars: Per-node scalar (type-0) features
                shape: [N_total, hidden_scalar]
            vectors: Per-node vector (type-1) features
                shape: [N_total, hidden_vector, 3]
                representation: SO(3) type-1 Cartesian vectors
        """
        # Build spatial graph (causes graph breaks under torch.compile)
        graph = SpatialGraph.build(
            pos, self.radius, max_num_neighbors=self.max_num_neighbors,
            batch=batch, num_batch_elements=num_batch_elements,
        )
        return self.forward_with_graph(pos, graph, features)

    def forward_with_graph(
        self,
        pos: Tensor,
        graph: SpatialGraph,
        features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Extract SE(3)-equivariant features (compile-friendly API).

        All operations inside this method are torch.compile-safe:
        cuBLAS matmuls, element-wise ops, and custom_op scatter.
        **Zero graph breaks** when compiled with ``dynamic=True``.

        Usage with torch.compile::

            model = SE3Net(...)
            compiled_forward = torch.compile(
                model.forward_with_graph, dynamic=True
            )
            graph = SpatialGraph.build(pos, radius, batch=batch)
            s, v = compiled_forward(pos, graph)

        Args:
            pos: Point positions
                shape: [N_total, 3], float32
            graph: Pre-built SpatialGraph (from SpatialGraph.build)
            features: Raw scalar features per node (optional).
                If None, uses all-ones [N_total, 1].
                shape: [N_total, in_channels]

        Returns:
            scalars: [N_total, hidden_scalar] — type-0 features
            vectors: [N_total, hidden_vector, 3] — type-1 features
        """
        N = pos.shape[0]

        # 1. Embed scalar features
        if features is None:
            features = pos.new_ones(N, self.in_channels)
        s = self.embed_scalar(features)  # [N, hidden_scalar]

        # 2. Initialize vector features as ZERO
        # First layer generates vectors via Path 1:
        #   scalar × edge_direction → vector (0 → 1 tensor product)
        v = pos.new_zeros(N, self.hidden_vector, 3)  # [N, hidden_vector, 3]

        # 3. Message passing layers (all share pre-built graph)
        for block in self.blocks:
            s, v = block(s, v, graph)

        return s, v

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, "
            f"hidden_s={self.hidden_scalar}, hidden_v={self.hidden_vector}, "
            f"layers={self.num_layers}, r={self.radius}"
        )


def global_mean_pool(
    x: Tensor,
    batch: Tensor,
    num_graphs: int,
) -> Tensor:
    """Deterministic scatter-mean pooling across nodes per graph.

    Equivariant for both scalar and vector features:
    - Scalar: invariant (mean is invariant)
    - Vector: R·mean(v_i) = mean(R·v_i) (mean commutes with linear R)

    Uses scatter_add + divide (no atomics, deterministic).

    Args:
        x: Node features to pool.
            shape: [N_total, C] for scalars, or [N_total, C, 3] for vectors
        batch: Graph assignment per node
            shape: [N_total], int64
        num_graphs: Total number of graphs in the batch (B)

    Returns:
        Pooled features
            shape: [B, C] for scalars, or [B, C, 3] for vectors
    """
    if x.dim() == 2:
        # Scalar: [N, C] → [B, C]
        out = x.new_zeros(num_graphs, x.shape[1])
        idx = batch.unsqueeze(1).expand_as(x)
        out.scatter_add_(0, idx, x)
    elif x.dim() == 3:
        # Vector: [N, C, 3] → [B, C, 3]
        out = x.new_zeros(num_graphs, x.shape[1], x.shape[2])
        idx = batch.unsqueeze(1).unsqueeze(2).expand_as(x)
        out.scatter_add_(0, idx, x)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")

    # Divide by count per graph (deterministic mean)
    count = torch.bincount(batch, minlength=num_graphs).clamp(min=1)
    if x.dim() == 2:
        out = out / count.unsqueeze(1)
    else:
        out = out / count.unsqueeze(1).unsqueeze(2)

    return out
