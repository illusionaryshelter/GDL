# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3)-equivariant continuous convolution on point clouds.

This is the core neural network layer of GeoEmbodied. It implements
continuous convolution using spherical harmonic filters and tensor
products, achieving exact SE(3) equivariance by construction.

Architecture:
    For each edge (i → j) in the radius graph:
    1. Compute relative direction r̂_ij and distance d_ij
    2. Evaluate spherical harmonics Y_l(r̂_ij) → angular filter basis
    3. Evaluate radial basis R(d_ij) → learned radial weight
    4. Combine: filter_l = R(d_ij) * Y_l(r̂_ij)
    5. Tensor product: message = Σ_paths W_path ⊗ (filter, feature)
    6. Aggregate: node_j_features = Σ_{i∈N(j)} message_ij

Feature representation:
    - Scalar (l=0): shape [N, C_scalar]
    - Vector (l=1): shape [N, C_vector, 3]

The equivariance guarantee:
    f(Rp + t) = R ⊳ f(p)  for all R ∈ SO(3), t ∈ R³
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from geoembodied.functional.radius_graph import radius_graph, compute_edge_vectors
from geoembodied.functional.segment_reduce import segment_reduce
from geoembodied.kernels.triton_sph_harm import spherical_harmonics

if TYPE_CHECKING:
    from geoembodied.nn.modules.spatial_graph import SpatialGraph


# ═══════════════════════════════════════════════════════════════════
# Pure message computation — inner loop
# ═══════════════════════════════════════════════════════════════════

# NOTE: torch.compile is INTENTIONALLY NOT USED here.
# Our edge count E varies every batch (different point clouds → different
# graph topologies). torch.compile(dynamic=True) caches compiled kernels
# for each distinct E, causing continuous GPU memory growth — a known
# PyTorch bug (issues #174468, #128424, #119607, #177869).
# The tensor products below are already memory-bandwidth-bound (gather +
# matmul + scatter), so compile offers <5% speedup at the cost of ~15GB
# VRAM leak over a training run.

def _compute_messages(
    s_src: Tensor,          # [E, C_s_in]  source scalar features
    v_src: Tensor,          # [E, C_v_in, 3]  source vector features
    direction: Tensor,      # [E, 3]  unit direction vectors
    Y_0: Tensor,            # [E, 1]  l=0 spherical harmonic
    R: Tensor,              # [E, P]  radial weights per path
    w_ss: Optional[Tensor],       # [C_s_out, C_s_in]  or None
    w_sv: Optional[Tensor],       # [C_v_out, C_s_in]  or None
    w_vv_scalar: Optional[Tensor],# [C_v_out, C_v_in]  or None
    w_vs: Optional[Tensor],       # [C_s_out, C_v_in]  or None
    w_vv_cross: Optional[Tensor], # [C_v_out, C_v_in]  or None
    Cs_out: int,
    Cv_out: int,
) -> Tuple[Tensor, Tensor]:
    """Compute all 5 tensor-product path messages for SE3Conv.

    This is a **pure function** operating on per-edge tensors only.
    No graph topology, no dynamic shapes, no side effects.
    All operations are cuBLAS matmuls or element-wise — ideal for
    ``torch.compile`` fusion by the caller (SE3Net/user code).

    TP Paths (filter_l × feature_l → output_l):
        Path 0: Y_0 × f_scalar → f_scalar     (scalar pass-through)
        Path 1: dir × f_scalar → f_vector      (scalar → vector promotion)
        Path 2: Y_0 × f_vector → f_vector      (vector pass-through)
        Path 3: dir · f_vector → f_scalar      (dot product, invariant)
        Path 4: dir × f_vector → f_vector      (cross product, equivariant)

    Args:
        s_src: Source scalar features, shape: [E, C_s_in]
        v_src: Source vector features, shape: [E, C_v_in, 3]
            representation: SO(3) type-1 Cartesian vectors
        direction: Unit direction vectors, shape: [E, 3]
        Y_0: l=0 spherical harmonic coefficient, shape: [E, 1]
        R: Radial weights per path, shape: [E, num_active_paths]
        w_ss..w_vv_cross: Weight matrices for each path (transposed
            from nn.Linear convention: [C_out, C_in]), or None if inactive
        Cs_out: Output scalar channels
        Cv_out: Output vector channels

    Returns:
        (scalar_msg, vector_msg):
            scalar_msg: [E, C_s_out] — per-edge scalar messages
            vector_msg: [E, C_v_out, 3] — per-edge vector messages
    """
    # ⚠ AMP PROTECTION (Landmine 3): Geometric tensor product paths
    # require at least FP32. torch.linalg.cross does not support FP16,
    # and direction/SH multiplications lose critical precision in half.
    # Strategy: upcast FP16/BF16 → FP32, but PRESERVE FP64 for
    # double-precision equivariance tests.
    E = s_src.shape[0]
    device = s_src.device

    # Determine compute dtype: at least float32, preserve float64
    input_dtype = s_src.dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype  # float32 or float64, keep as-is

    s_src = s_src.to(compute_dtype)
    v_src = v_src.to(compute_dtype)
    direction = direction.to(compute_dtype)
    Y_0 = Y_0.to(compute_dtype)
    R = R.to(compute_dtype)

    # Also cast weight matrices to match compute dtype
    if w_ss is not None:
        w_ss = w_ss.to(compute_dtype)
    if w_sv is not None:
        w_sv = w_sv.to(compute_dtype)
    if w_vv_scalar is not None:
        w_vv_scalar = w_vv_scalar.to(compute_dtype)
    if w_vs is not None:
        w_vs = w_vs.to(compute_dtype)
    if w_vv_cross is not None:
        w_vv_cross = w_vv_cross.to(compute_dtype)

    total_scalar_msg = s_src.new_zeros(E, Cs_out)
    total_vector_msg = v_src.new_zeros(E, Cv_out, 3)

    path_idx = 0

    # Path 0: Y_0 × f_s → f_s  (cuBLAS matmul)
    if w_ss is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        # nn.Linear stores weight as [C_out, C_in], matmul: s_src @ W^T
        total_scalar_msg = total_scalar_msg + r_weight * Y_0 * (s_src @ w_ss.t())
        path_idx += 1

    # Path 1: direction × f_s → f_v
    if w_sv is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        mixed = r_weight * (s_src @ w_sv.t())  # [E, C_v_out]
        total_vector_msg = total_vector_msg + mixed.unsqueeze(-1) * direction.unsqueeze(-2)
        path_idx += 1

    # Path 2: Y_0 × f_v → f_v
    if w_vv_scalar is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        # v_src: [E, C_v_in, 3], weight: [C_v_out, C_v_in]
        # → (E, 3, C_v_in) @ (C_v_in, C_v_out) → (E, 3, C_v_out) → transpose
        mixed = (v_src.transpose(-1, -2) @ w_vv_scalar.t()).transpose(-1, -2)
        total_vector_msg = total_vector_msg + r_weight.unsqueeze(-1) * Y_0.unsqueeze(-1) * mixed
        path_idx += 1

    # Path 3: direction · f_v → f_s (invariant dot product)
    if w_vs is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        dot = (direction.unsqueeze(-2) * v_src).sum(dim=-1)  # [E, C_v_in]
        total_scalar_msg = total_scalar_msg + r_weight * (dot @ w_vs.t())
        path_idx += 1

    # Path 4: direction × f_v → f_v (equivariant cross product)
    if w_vv_cross is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        cross = torch.linalg.cross(
            direction.unsqueeze(-2).expand_as(v_src),
            v_src, dim=-1,
        )  # [E, C_v_in, 3]
        mixed = (cross.transpose(-1, -2) @ w_vv_cross.t()).transpose(-1, -2)
        total_vector_msg = total_vector_msg + r_weight.unsqueeze(-1) * mixed
        path_idx += 1

    return total_scalar_msg, total_vector_msg


class RadialBasis(nn.Module):
    """Learnable radial basis function for distance encoding.

    Uses sinusoidal basis + MLP to encode edge distances.
    The radial part is rotationally invariant by construction.

    Args:
        num_basis: Number of radial basis functions
        cutoff: Maximum distance (radius)
        num_hidden: MLP hidden dimension
        num_output: Output dimension (= number of TP paths)
    """

    def __init__(
        self,
        num_basis: int = 16,
        cutoff: float = 5.0,
        num_hidden: int = 64,
        num_output: int = 1,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis

        # Sinusoidal basis frequencies
        freqs = torch.arange(1, num_basis + 1).float() * math.pi / cutoff
        self.register_buffer('freqs', freqs)

        # MLP: basis → weights
        self.mlp = nn.Sequential(
            nn.Linear(num_basis, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_output),
        )

    def forward(self, dist: Tensor) -> Tensor:
        """Compute radial weights from distances.

        Args:
            dist: Edge distances, shape: [E]

        Returns:
            Radial weights, shape: [E, num_output]
        """
        # Sinusoidal basis: [E, num_basis]
        d = dist.unsqueeze(-1)  # [E, 1]
        basis = torch.sin(self.freqs * d) / d.clamp(min=1e-8)  # [E, B]

        # Cosine cutoff envelope (smooth to zero at boundary)
        envelope = 0.5 * (1.0 + torch.cos(math.pi * dist / self.cutoff))
        envelope = envelope.clamp(min=0.0).unsqueeze(-1)  # [E, 1]

        return self.mlp(basis * envelope)  # [E, num_output]


class SE3Conv(nn.Module):
    """SE(3)-equivariant continuous convolution on point clouds.

    Processes both scalar (l=0) and vector (l=1) features through
    tensor product with spherical harmonic filters.

    Tensor product paths (filter_l × feature_l → output_l):
        Path 0: Y_0 × f_scalar → f_scalar     (scalar filter × scalar)
        Path 1: Y_1 × f_scalar → f_vector      (vector filter → vector out)
        Path 2: Y_0 × f_vector → f_vector      (scalar filter × vector, keep)
        Path 3: Y_1 × f_vector → f_scalar      (dot product: vector→scalar)
        Path 4: Y_1 × f_vector → f_vector      (cross product: vector→vector)

    Each path has a learnable radial weight R_path(d).

    Args:
        in_scalar_channels: Input scalar feature dimension
        in_vector_channels: Input vector feature dimension
        out_scalar_channels: Output scalar feature dimension
        out_vector_channels: Output vector feature dimension
        radius: Neighborhood radius
        max_num_neighbors: Maximum neighbors per node
        num_radial_basis: Number of radial basis functions

    Example::

        >>> conv = SE3Conv(
        ...     in_scalar_channels=64, in_vector_channels=16,
        ...     out_scalar_channels=64, out_vector_channels=16,
        ...     radius=2.0,
        ... )
        >>> pos = torch.randn(100, 3)
        >>> s_in = torch.randn(100, 64)
        >>> v_in = torch.randn(100, 16, 3)
        >>> s_out, v_out = conv(pos, s_in, v_in)

    .. note:: **torch.compile integration:**
        This layer is designed for ``torch.compile`` at the **SE3Net or user
        level**, not per-layer. The inner message computation
        (``_compute_messages``) is a pure function of per-edge tensors
        (fixed shapes), and ``segment_reduce`` is registered as a
        ``torch.library.custom_op`` — both are compile-transparent.
        The dynamic-shape gather (``scalars[graph.row]``) is left outside
        the compiled region automatically by the Dynamo tracer.
    """

    def __init__(
        self,
        in_scalar_channels: int,
        in_vector_channels: int,
        out_scalar_channels: int,
        out_vector_channels: int,
        radius: float = 2.0,
        max_num_neighbors: int = 32,
        num_radial_basis: int = 16,
    ) -> None:
        super().__init__()
        self.in_scalar_channels = in_scalar_channels
        self.in_vector_channels = in_vector_channels
        self.out_scalar_channels = out_scalar_channels
        self.out_vector_channels = out_vector_channels
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        # Count active tensor product paths
        self._num_paths = 0

        # Path 0: Y_0 × f_s → f_s (scalar × scalar → scalar)
        if in_scalar_channels > 0 and out_scalar_channels > 0:
            self.w_ss = nn.Linear(in_scalar_channels, out_scalar_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_ss = None

        # Path 1: Y_1 × f_s → f_v (vector filter × scalar → vector)
        if in_scalar_channels > 0 and out_vector_channels > 0:
            self.w_sv = nn.Linear(in_scalar_channels, out_vector_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_sv = None

        # Path 2: Y_0 × f_v → f_v (scalar filter × vector → vector, keep)
        if in_vector_channels > 0 and out_vector_channels > 0:
            self.w_vv_scalar = nn.Linear(in_vector_channels, out_vector_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_vv_scalar = None

        # Path 3: Y_1 · f_v → f_s (dot product: vector → scalar, invariant)
        if in_vector_channels > 0 and out_scalar_channels > 0:
            self.w_vs = nn.Linear(in_vector_channels, out_scalar_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_vs = None

        # Path 4: Y_1 × f_v → f_v (cross product: vector → vector)
        if in_vector_channels > 0 and out_vector_channels > 0:
            self.w_vv_cross = nn.Linear(in_vector_channels, out_vector_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_vv_cross = None

        # Radial basis for distance-dependent weighting
        self.radial_basis = RadialBasis(
            num_basis=num_radial_basis,
            cutoff=radius,
            num_hidden=64,
            num_output=max(self._num_paths, 1),
        )

        # Output bias (scalars only; vectors have no bias for equivariance)
        if out_scalar_channels > 0:
            self.scalar_bias = nn.Parameter(torch.zeros(out_scalar_channels))
        else:
            self.scalar_bias = None

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        graph: 'SpatialGraph',
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """SE(3)-equivariant convolution — pure stateless operator.

        Topology is provided by the caller via ``graph``.
        This enables edge reuse across layers, CSR caching for
        deterministic segment reduce, and SH caching.

        The inner message computation is factored into
        ``_compute_messages`` — a pure function of per-edge tensors
        with fixed shapes, suitable for ``torch.compile`` fusion
        by the caller (SE3Net or user code).

        Args:
            scalars: Scalar features
                shape: [N, C_s_in]
            vectors: Vector features
                shape: [N, C_v_in, 3]
                representation: SO(3) type-1 Cartesian vectors
            graph: Pre-built SpatialGraph containing edges, directions,
                SH coefficients, and CSR structure.
            mask: Boolean mask for valid (non-pad) points
                shape: [N], dtype: bool, optional

        Returns:
            Tuple of (scalar_out, vector_out):
                scalar_out: [N, C_s_out]
                vector_out: [N, C_v_out, 3]
        """
        N = graph.N
        device = scalars.device
        input_dtype = scalars.dtype

        if graph.E == 0:
            # Keep autograd graph alive: inputs contribute zero, but grad flows
            s_out = scalars.new_zeros(N, self.out_scalar_channels)
            v_out = vectors.new_zeros(N, self.out_vector_channels, 3)
            # Connect input to graph: s_in * 0 so grad passes through
            s_out = s_out + (scalars.sum() * 0).unsqueeze(0)
            v_out = v_out + (vectors.sum() * 0).unsqueeze(0).unsqueeze(0)
            if self.scalar_bias is not None:
                s_out = s_out + self.scalar_bias
            return s_out, v_out

        # ── AMP GUARD ──────────────────────────────────────────────
        # Disable autocast for the ENTIRE convolution body.
        #
        # Why: autocast silently demotes nn.Linear and @ (matmul) to
        # FP16, including RadialBasis.mlp and the tensor-product
        # matmuls in _compute_messages (s_src @ w_ss.t() etc.).
        # In the backward pass these FP16 matmul gradients, when
        # multiplied by GradScaler's scale factor, overflow FP16 max
        # (65504) → Inf → GradScaler halves scale every epoch,
        # eventually reaching scale=1 and causing gradient underflow.
        #
        # By disabling autocast here, ALL operations (radial MLP,
        # tensor products, segment reduce) run in genuine FP32,
        # both forward AND backward.  This is correct: geometric
        # tensor products require FP32 precision (Rule 4).
        # ──────────────────────────────────────────────────────────
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type, enabled=False):
            # Upcast half → FP32; preserve FP64 for equivariance tests
            compute_dtype = torch.float32 if scalars.dtype in (torch.float16, torch.bfloat16) else scalars.dtype
            scalars_f = scalars.to(compute_dtype)
            vectors_f = vectors.to(compute_dtype)

            # ── 1. Gather source features ──
            s_src = scalars_f[graph.row]       # [E, C_s_in]
            v_src = vectors_f[graph.row]       # [E, C_v_in, 3]

            # ── 2. Compute radial weights (MLP runs in ≥FP32) ──
            R = self.radial_basis(graph.dist.to(compute_dtype))  # [E, P]

            # ── 3. Compute messages (all matmuls in genuine FP32) ──
            total_scalar_msg, total_vector_msg = _compute_messages(
                s_src, v_src,
                graph.direction.to(compute_dtype), graph.Y[..., 0:1].to(compute_dtype), R,
                self.w_ss.weight if self.w_ss is not None else None,
                self.w_sv.weight if self.w_sv is not None else None,
                self.w_vv_scalar.weight if self.w_vv_scalar is not None else None,
                self.w_vs.weight if self.w_vs is not None else None,
                self.w_vv_cross.weight if self.w_vv_cross is not None else None,
                self.out_scalar_channels,
                self.out_vector_channels,
            )

            # ── 4. Aggregate: deterministic segment reduce ──
            E = graph.E
            s_out = segment_reduce(
                total_scalar_msg,
                graph.col_sorted, graph.perm,
                graph.node_start, graph.node_end,
                N,
            )

            # Vectors: [E, Cv, 3] → [E, Cv*3], reduce, reshape back
            v_msg_flat = total_vector_msg.reshape(E, -1)  # [E, Cv*3]
            v_out_flat = segment_reduce(
                v_msg_flat,
                graph.col_sorted, graph.perm,
                graph.node_start, graph.node_end,
                N,
            )
            v_out = v_out_flat.reshape(N, self.out_vector_channels, 3)

            # 5. Add scalar bias
            if self.scalar_bias is not None:
                s_out = s_out + self.scalar_bias

            # 6. Zero out pad point features (algebraic mask enforcement)
            if mask is not None:
                pad_mask = (~mask).unsqueeze(-1)  # [N, 1]
                s_out = s_out.masked_fill(pad_mask, 0.0)
                v_out = v_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Cast output back to input dtype for downstream autocast layers
        return s_out.to(input_dtype), v_out.to(input_dtype)

    def forward_legacy(
        self,
        pos: Tensor,
        scalars: Tensor,
        vectors: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        edge_index: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Legacy forward: auto-builds SpatialGraph from pos.

        Maintained for backward compatibility with existing code.
        For new code, prefer building SpatialGraph explicitly and
        calling ``forward(scalars, vectors, graph)``.

        Args:
            pos: [N, 3] point positions
            scalars: [N, C_s_in]
            vectors: [N, C_v_in, 3], representation: SO(3) type-1
            batch: [N] optional batch assignment
            mask: [N] bool optional
            edge_index: (row, col) optional pre-computed edges

        Returns:
            (scalar_out, vector_out)
        """
        from geoembodied.nn.modules.spatial_graph import SpatialGraph

        if edge_index is not None:
            row, col = edge_index
            graph = SpatialGraph.from_edge_index(row, col, pos, pos.shape[0])
        else:
            graph = SpatialGraph.build(
                pos, self.radius,
                max_num_neighbors=self.max_num_neighbors,
                batch=batch, mask=mask,
            )

        return self.forward(scalars, vectors, graph, mask=mask)

    def extra_repr(self) -> str:
        return (
            f"scalar: {self.in_scalar_channels}→{self.out_scalar_channels}, "
            f"vector: {self.in_vector_channels}→{self.out_vector_channels}, "
            f"radius={self.radius}, paths={self._num_paths}"
        )
