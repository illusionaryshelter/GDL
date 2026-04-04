# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SpatialGraph — Immutable spatial topology for equivariant message passing.

This is the single source of truth for graph structure in the GeoEmbodied
library. Built ONCE per point cloud per forward pass, then shared across
all SE3Conv layers that operate on the same point cloud.

Design principles:
    1. **Immutable**: No in-place mutations after construction
    2. **Topology decoupled from operators**: SE3Conv is stateless
    3. **CSR cached**: Sorted edges + CSR offsets enable deterministic
       segment reduce (no atomic contention)
    4. **SH cached**: Spherical harmonics computed once, reused by all layers

Usage::

    # At SE3Net / Stage level:
    graph = SpatialGraph.build(pos, radius=0.15, max_num_neighbors=32)

    # Shared by all SE3Conv layers:
    s1, v1 = conv1(scalars, vectors, graph)
    s2, v2 = conv2(s1, v1, graph)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from geoembodied.functional.radius_graph import radius_graph, compute_edge_vectors
from geoembodied.functional.segment_reduce import sort_and_build_csr
from geoembodied.kernels.triton_sph_harm import spherical_harmonics


@dataclass(frozen=True)
class SpatialGraph:
    """Immutable spatial topology + cached geometry for message passing.

    All tensors are on the same device and contiguous.

    Attributes:
        row: [E] int64 — source node indices (COO, original order)
        col: [E] int64 — target node indices (COO, original order)
        direction: [E, 3] float32 — unit direction vectors (target-to-source: r̂_ij)
        dist: [E] float32 — edge distances |r_ij|
        col_sorted: [E] int32 — target indices sorted for segment reduce
        perm: [E] int32 — permutation: sorted → original edge order
        node_start: [N] int32 — CSR start offsets per node
        node_end: [N] int32 — CSR end offsets per node
        Y: [E, num_sh] float32 — spherical harmonics of direction vectors
        N: int — number of nodes
        E: int — number of edges
    """
    # ── COO edges (original order) ──
    row: Tensor          # [E] int64 — source node indices
    col: Tensor          # [E] int64 — target node indices

    # ── Edge geometry ──
    direction: Tensor    # [E, 3] float32 — unit direction vectors
    dist: Tensor         # [E] float32 — edge distances

    # ── CSR structure (sorted by col for segment reduce) ──
    col_sorted: Tensor   # [E] int32 — target indices, sorted
    perm: Tensor         # [E] int32 — permutation: sorted → original
    node_start: Tensor   # [N] int32 — CSR row pointer start
    node_end: Tensor     # [N] int32 — CSR row pointer end

    # ── Cached filter basis ──
    Y: Tensor            # [E, num_sh] float32 — spherical harmonics (l≤2)

    # ── Metadata ──
    N: int               # number of nodes
    E: int               # number of edges
    avg_degree: Optional[Tensor] = None  # [N] float32 — per-node avg degree of its cloud

    @staticmethod
    def build(
        pos: Tensor,
        radius: float,
        max_num_neighbors: int = 32,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        max_l: int = 2,
        num_batch_elements: Optional[int] = None,
    ) -> SpatialGraph:
        """Build spatial graph from point positions.

        This is the ONLY place where ``radius_graph`` is called.
        All downstream layers receive the pre-built graph.

        Args:
            pos: [N, 3] float32 — point positions
            radius: Maximum edge distance
            max_num_neighbors: K for KNN within radius
            mask: [N] bool — True for valid points (optional)
            batch: [N] int64 — batch assignment (optional)
            max_l: Maximum SH degree (default 2 → 9 coefficients)
            num_batch_elements: Number of batch elements (for batched dispatch)

        Returns:
            Fully initialized SpatialGraph
        """
        N = pos.shape[0]
        device = pos.device

        # 1. Build radius graph (CUDA or brute-force)
        row, col = radius_graph(
            pos, radius,
            max_num_neighbors=max_num_neighbors,
            batch=batch,
            mask=mask,
            num_batch_elements=num_batch_elements,
        )

        E_count = row.shape[0]

        if E_count == 0:
            return SpatialGraph._empty(N, device, max_l)

        # 2. Compute edge vectors
        diff, dist, direction = compute_edge_vectors(pos, row, col)

        # 3. Spherical harmonics
        Y = spherical_harmonics(direction, max_l=max_l, normalize=False)

        # 4. Sort by target + build CSR for segment reduce
        col_sorted, perm, node_start, node_end = sort_and_build_csr(col, N)

        # 5. Per-cloud average degree (batch-isolation safe)
        #    Each node stores the average degree of its cloud, so the
        #    normalization divisor is identical whether the cloud is
        #    processed alone or in a batch.  shape: [N]
        degree = (node_end - node_start).float()  # [N]
        if batch is not None:
            # Use num_batch_elements to avoid GPU→CPU sync
            if num_batch_elements is not None:
                B_count = num_batch_elements
            else:
                B_count = int(batch.max().item()) + 1
            cloud_sum = torch.zeros(B_count, device=device)
            cloud_cnt = torch.zeros(B_count, device=device)
            cloud_sum.scatter_add_(0, batch, degree)
            cloud_cnt.scatter_add_(0, batch, torch.ones_like(degree))
            cloud_avg = cloud_sum / cloud_cnt.clamp(min=1)  # [B]
            per_node_avg = cloud_avg[batch]  # [N]
        else:
            per_node_avg = degree.mean().expand(N)  # single cloud
        per_node_avg = per_node_avg.clamp(min=1.0)  # safety

        return SpatialGraph(
            row=row,
            col=col,
            direction=direction,
            dist=dist,
            col_sorted=col_sorted,
            perm=perm,
            node_start=node_start,
            node_end=node_end,
            Y=Y,
            N=N,
            E=E_count,
            avg_degree=per_node_avg,
        )

    @staticmethod
    def _empty(N: int, device: torch.device, max_l: int = 2) -> SpatialGraph:
        """Create an empty SpatialGraph (zero edges)."""
        num_sh = (max_l + 1) ** 2
        return SpatialGraph(
            row=torch.empty(0, dtype=torch.long, device=device),
            col=torch.empty(0, dtype=torch.long, device=device),
            direction=torch.empty(0, 3, device=device),
            dist=torch.empty(0, device=device),
            col_sorted=torch.empty(0, dtype=torch.int32, device=device),
            perm=torch.empty(0, dtype=torch.int32, device=device),
            node_start=torch.zeros(N, dtype=torch.int32, device=device),
            node_end=torch.zeros(N, dtype=torch.int32, device=device),
            Y=torch.empty(0, num_sh, device=device),
            N=N,
            E=0,
            avg_degree=torch.ones(N, device=device),  # safe: no edges → /1
        )

    @staticmethod
    def from_edge_index(
        row: Tensor,
        col: Tensor,
        pos: Tensor,
        N: int,
        max_l: int = 2,
        batch: Optional[Tensor] = None,
        num_batch_elements: Optional[int] = None,
    ) -> SpatialGraph:
        """Build SpatialGraph from pre-computed edge indices.

        Useful for converting existing (row, col) pairs to the
        SpatialGraph format without recomputing radius_graph.

        Args:
            row: [E] int64 — source indices
            col: [E] int64 — target indices
            pos: [N, 3] — point positions
            N: Number of nodes
            max_l: Maximum SH degree
            batch: [N] int64 — batch assignment (optional, for per-cloud avg)
            num_batch_elements: Number of clouds (avoids GPU sync if provided)

        Returns:
            SpatialGraph
        """
        device = pos.device
        E_count = row.shape[0]

        if E_count == 0:
            return SpatialGraph._empty(N, device, max_l)

        diff, dist, direction = compute_edge_vectors(pos, row, col)
        Y = spherical_harmonics(direction, max_l=max_l, normalize=False)
        col_sorted, perm, node_start, node_end = sort_and_build_csr(col, N)

        # Per-cloud average degree (same logic as build())
        degree = (node_end - node_start).float()
        if batch is not None:
            if num_batch_elements is not None:
                B_count = num_batch_elements
            else:
                B_count = int(batch.max().item()) + 1
            cloud_sum = torch.zeros(B_count, device=device)
            cloud_cnt = torch.zeros(B_count, device=device)
            cloud_sum.scatter_add_(0, batch, degree)
            cloud_cnt.scatter_add_(0, batch, torch.ones_like(degree))
            cloud_avg = cloud_sum / cloud_cnt.clamp(min=1)
            per_node_avg = cloud_avg[batch]
        else:
            per_node_avg = degree.mean().expand(N)
        per_node_avg = per_node_avg.clamp(min=1.0)

        return SpatialGraph(
            row=row, col=col,
            direction=direction, dist=dist,
            col_sorted=col_sorted, perm=perm,
            node_start=node_start, node_end=node_end,
            Y=Y, N=N, E=E_count,
            avg_degree=per_node_avg,
        )


