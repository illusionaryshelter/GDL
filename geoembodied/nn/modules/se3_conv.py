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
    - Type-2 (l=2): shape [N, C_type2, 5]

The equivariance guarantee:
    f(Rp + t) = D^l(R) ⊳ f(p)  for all R ∈ SO(3), t ∈ R³
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
from geoembodied.functional.tensor_product import get_cg_matrix
from geoembodied.kernels.triton_sph_harm import spherical_harmonics

if TYPE_CHECKING:
    from geoembodied.nn.modules.spatial_graph import SpatialGraph


# ═══════════════════════════════════════════════════════════════════
# Pure message computation — inner loop
# ═══════════════════════════════════════════════════════════════════

# NOTE on torch.compile compatibility:
# This function is FULLY torch.compile transparent (zero graph breaks).
# The segment_reduce custom_op and all TP paths trace without issues.
# As of PyTorch ≥2.10, dynamic=True handles variable E without memory leaks.
#
# IMPORTANT: Compile THIS FUNCTION directly, NOT the whole model.
# The outer model (graph construction, KNN, .item(), checkpoint) contains
# untraceable ops that crash or produce graph breaks.
# Usage:  _compute_messages = torch.compile(_compute_messages, dynamic=True)

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
    # ── Type-2 paths (optional) ──
    t2_src: Optional[Tensor] = None,   # [E, C_t2_in, 5]
    Y_2: Optional[Tensor] = None,      # [E, 5]
    w_st2: Optional[Tensor] = None,    # [C_t2_out, C_s_in]
    w_t2t2: Optional[Tensor] = None,   # [C_t2_out, C_t2_in]
    w_vt2: Optional[Tensor] = None,    # [C_t2_out, C_v_in]
    w_t2s: Optional[Tensor] = None,    # [C_s_out, C_t2_in]
    cg_11_2: Optional[Tensor] = None,  # [9, 5] CG matrix
    Ct2_out: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Compute all tensor-product path messages for SE3Conv.

    TP Paths (filter_l × feature_l → output_l):
        Path 0: Y_0 × f_scalar → f_scalar     (scalar pass-through)
        Path 1: dir × f_scalar → f_vector      (scalar → vector promotion)
        Path 2: Y_0 × f_vector → f_vector      (vector pass-through)
        Path 3: dir · f_vector → f_scalar      (dot product, invariant)
        Path 4: dir × f_vector → f_vector      (cross product, equivariant)
        Path 5: Y_2 × f_scalar → f_type2       (SH filter → type-2)
        Path 6: Y_0 × f_type2 → f_type2        (type-2 pass-through)
        Path 7: dir ⊗ f_vector → f_type2       (CG l=1⊗l=1→l=2)
        Path 8: ||f_type2||² → f_scalar        (invariant contraction)

    Returns:
        (scalar_msg, vector_msg, type2_msg)
    """
    E = s_src.shape[0]
    device = s_src.device

    # Determine compute dtype: at least float32, preserve float64
    input_dtype = s_src.dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    s_src = s_src.to(compute_dtype)
    v_src = v_src.to(compute_dtype)
    direction = direction.to(compute_dtype)
    Y_0 = Y_0.to(compute_dtype)
    R = R.to(compute_dtype)

    # Cast weight matrices
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

    # Type-2 casts
    if t2_src is not None:
        t2_src = t2_src.to(compute_dtype)
    if Y_2 is not None:
        Y_2 = Y_2.to(compute_dtype)
    if w_st2 is not None:
        w_st2 = w_st2.to(compute_dtype)
    if w_t2t2 is not None:
        w_t2t2 = w_t2t2.to(compute_dtype)
    if w_vt2 is not None:
        w_vt2 = w_vt2.to(compute_dtype)
    if w_t2s is not None:
        w_t2s = w_t2s.to(compute_dtype)
    if cg_11_2 is not None:
        cg_11_2 = cg_11_2.to(compute_dtype)

    total_scalar_msg = s_src.new_zeros(E, Cs_out)
    total_vector_msg = v_src.new_zeros(E, Cv_out, 3)
    total_type2_msg = None
    if Ct2_out > 0:
        total_type2_msg = s_src.new_zeros(E, Ct2_out, 5)

    path_idx = 0

    # Path 0: Y_0 × f_s → f_s  (cuBLAS matmul)
    if w_ss is not None:
        r_weight = R[:, path_idx:path_idx+1]
        total_scalar_msg = total_scalar_msg + r_weight * Y_0 * (s_src @ w_ss.t())
        path_idx += 1

    # Path 1: direction × f_s → f_v
    if w_sv is not None:
        r_weight = R[:, path_idx:path_idx+1]
        mixed = r_weight * (s_src @ w_sv.t())  # [E, C_v_out]
        total_vector_msg = total_vector_msg + mixed.unsqueeze(-1) * direction.unsqueeze(-2)
        path_idx += 1

    # Path 2: Y_0 × f_v → f_v
    if w_vv_scalar is not None:
        r_weight = R[:, path_idx:path_idx+1]
        mixed = (v_src.transpose(-1, -2) @ w_vv_scalar.t()).transpose(-1, -2)
        total_vector_msg = total_vector_msg + r_weight.unsqueeze(-1) * Y_0.unsqueeze(-1) * mixed
        path_idx += 1

    # Path 3: direction · f_v → f_s (invariant dot product)
    if w_vs is not None:
        r_weight = R[:, path_idx:path_idx+1]
        dot = (direction.unsqueeze(-2) * v_src).sum(dim=-1)  # [E, C_v_in]
        total_scalar_msg = total_scalar_msg + r_weight * (dot @ w_vs.t())
        path_idx += 1

    # Path 4: direction × f_v → f_v (equivariant cross product)
    if w_vv_cross is not None:
        r_weight = R[:, path_idx:path_idx+1]
        cross = torch.linalg.cross(
            direction.unsqueeze(-2).expand_as(v_src),
            v_src, dim=-1,
        )
        mixed = (cross.transpose(-1, -2) @ w_vv_cross.t()).transpose(-1, -2)
        total_vector_msg = total_vector_msg + r_weight.unsqueeze(-1) * mixed
        path_idx += 1

    # ── Type-2 paths ──

    # Path 5: Y_2 × f_s → f_t2  (SH filter → type-2)
    # Y_2: [E, 5], s_src: [E, C_s_in], w_st2: [C_t2_out, C_s_in]
    # Result: [E, C_t2_out, 5]
    if w_st2 is not None and Y_2 is not None and total_type2_msg is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        mixed = r_weight * (s_src @ w_st2.t())  # [E, C_t2_out]
        total_type2_msg = total_type2_msg + mixed.unsqueeze(-1) * Y_2.unsqueeze(-2)
        path_idx += 1

    # Path 6: Y_0 × f_t2 → f_t2  (type-2 pass-through)
    if w_t2t2 is not None and t2_src is not None and total_type2_msg is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        # t2_src: [E, C_t2_in, 5], w_t2t2: [C_t2_out, C_t2_in]
        mixed = (t2_src.transpose(-1, -2) @ w_t2t2.t()).transpose(-1, -2)
        total_type2_msg = total_type2_msg + r_weight.unsqueeze(-1) * Y_0.unsqueeze(-1) * mixed
        path_idx += 1

    # Path 7: dir ⊗ f_v → f_t2 via CG(1,1,2)
    # CG is pre-permuted to Cartesian (x,y,z) order at init time,
    # so NO runtime permutation of direction/v_src is needed.
    # cg_11_2 shape: [3, 3, 5] (Cartesian order) → reshaped to [9, 5] for matmul.
    if w_vt2 is not None and cg_11_2 is not None and total_type2_msg is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        # Outer product: dir[E,3] ⊗ v[E,Cv,3] → [E,Cv,3,3]
        dir_exp = direction.unsqueeze(1).unsqueeze(-1)   # [E, 1, 3, 1]
        v_exp = v_src.unsqueeze(-2)                       # [E, Cv, 1, 3]
        outer = dir_exp * v_exp                           # [E, Cv, 3, 3]
        # [E, Cv, 3, 3] → [E, Cv, 9] @ CG[9, 5] → [E, Cv, 5]
        outer_flat = outer.reshape(E, v_src.shape[1], 9)
        tp_result = outer_flat @ cg_11_2.reshape(9, 5)
        # Channel mixing: [E, 5, C_v_in] @ [C_v_in, C_t2_out] → [E, 5, C_t2_out] → transpose
        mixed = (tp_result.transpose(-1, -2) @ w_vt2.t()).transpose(-1, -2)  # [E, C_t2_out, 5]
        total_type2_msg = total_type2_msg + r_weight.unsqueeze(-1) * mixed
        path_idx += 1

    # Path 8: ||f_t2||² → f_s  (invariant contraction)
    # t2_src: [E, C_t2_in, 5] → sum_m t2²[c,m] → [E, C_t2_in]
    if w_t2s is not None and t2_src is not None:
        r_weight = R[:, path_idx:path_idx+1]  # [E, 1]
        t2_norm_sq = (t2_src * t2_src).sum(dim=-1)  # [E, C_t2_in]
        total_scalar_msg = total_scalar_msg + r_weight * (t2_norm_sq @ w_t2s.t())
        path_idx += 1

    return total_scalar_msg, total_vector_msg, total_type2_msg


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

        freqs = torch.arange(1, num_basis + 1).float() * math.pi / cutoff
        self.register_buffer('freqs', freqs)

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
        d = dist.unsqueeze(-1)
        basis = torch.sin(self.freqs * d) / d.clamp(min=1e-8)
        envelope = 0.5 * (1.0 + torch.cos(math.pi * dist / self.cutoff))
        envelope = envelope.clamp(min=0.0).unsqueeze(-1)
        return self.mlp(basis * envelope)


class SE3Conv(nn.Module):
    """SE(3)-equivariant continuous convolution on point clouds.

    Processes scalar (l=0), vector (l=1), and optionally type-2 (l=2)
    features through tensor product with spherical harmonic filters.

    Tensor product paths (filter_l × feature_l → output_l):
        Path 0: Y_0 × f_scalar → f_scalar     (scalar filter × scalar)
        Path 1: Y_1 × f_scalar → f_vector      (vector filter → vector out)
        Path 2: Y_0 × f_vector → f_vector      (scalar filter × vector)
        Path 3: Y_1 × f_vector → f_scalar      (dot product: vector→scalar)
        Path 4: Y_1 × f_vector → f_vector      (cross product: vector→vector)
        Path 5: Y_2 × f_scalar → f_type2       (SH filter → type-2)
        Path 6: Y_0 × f_type2 → f_type2        (type-2 pass-through)
        Path 7: Y_1 ⊗ f_vector → f_type2       (CG l=1⊗l=1→l=2)
        Path 8: ||f_type2||² → f_scalar        (invariant contraction)

    Args:
        in_scalar_channels: Input scalar feature dimension
        in_vector_channels: Input vector feature dimension
        out_scalar_channels: Output scalar feature dimension
        out_vector_channels: Output vector feature dimension
        in_type2_channels: Input type-2 feature dimension (0 to disable)
        out_type2_channels: Output type-2 feature dimension (0 to disable)
        radius: Neighborhood radius
        max_num_neighbors: Maximum neighbors per node
        num_radial_basis: Number of radial basis functions
    """

    def __init__(
        self,
        in_scalar_channels: int,
        in_vector_channels: int,
        out_scalar_channels: int,
        out_vector_channels: int,
        in_type2_channels: int = 0,
        out_type2_channels: int = 0,
        radius: float = 2.0,
        max_num_neighbors: int = 32,
        num_radial_basis: int = 16,
    ) -> None:
        super().__init__()
        self.in_scalar_channels = in_scalar_channels
        self.in_vector_channels = in_vector_channels
        self.out_scalar_channels = out_scalar_channels
        self.out_vector_channels = out_vector_channels
        self.in_type2_channels = in_type2_channels
        self.out_type2_channels = out_type2_channels
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        # Count active tensor product paths
        self._num_paths = 0

        # Path 0: Y_0 × f_s → f_s
        if in_scalar_channels > 0 and out_scalar_channels > 0:
            self.w_ss = nn.Linear(in_scalar_channels, out_scalar_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_ss = None

        # Path 1: Y_1 × f_s → f_v
        if in_scalar_channels > 0 and out_vector_channels > 0:
            self.w_sv = nn.Linear(in_scalar_channels, out_vector_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_sv = None

        # Path 2: Y_0 × f_v → f_v
        if in_vector_channels > 0 and out_vector_channels > 0:
            self.w_vv_scalar = nn.Linear(in_vector_channels, out_vector_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_vv_scalar = None

        # Path 3: Y_1 · f_v → f_s (invariant)
        if in_vector_channels > 0 and out_scalar_channels > 0:
            self.w_vs = nn.Linear(in_vector_channels, out_scalar_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_vs = None

        # Path 4: Y_1 × f_v → f_v (cross product)
        if in_vector_channels > 0 and out_vector_channels > 0:
            self.w_vv_cross = nn.Linear(in_vector_channels, out_vector_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_vv_cross = None

        # ── Type-2 paths ──

        # Path 5: Y_2 × f_s → f_t2
        if in_scalar_channels > 0 and out_type2_channels > 0:
            self.w_st2 = nn.Linear(in_scalar_channels, out_type2_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_st2 = None

        # Path 6: Y_0 × f_t2 → f_t2
        if in_type2_channels > 0 and out_type2_channels > 0:
            self.w_t2t2 = nn.Linear(in_type2_channels, out_type2_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_t2t2 = None

        # Path 7: dir ⊗ f_v → f_t2 (CG l=1⊗l=1→l=2)
        if in_vector_channels > 0 and out_type2_channels > 0:
            self.w_vt2 = nn.Linear(in_vector_channels, out_type2_channels, bias=False)
            self._num_paths += 1
            # Pre-permute CG matrix from SH order (y,z,x) to Cartesian (x,y,z).
            # This eliminates runtime direction[:,[1,2,0]] index-select copies.
            # SH→Cartesian permutation: [2, 0, 1]
            cg_sh = get_cg_matrix(1, 1, 2)  # [9, 5] in SH order
            cg_3d = cg_sh.reshape(3, 3, 5)
            P = torch.tensor([2, 0, 1])  # SH→Cartesian
            cg_cart = cg_3d[P][:, P]  # [3, 3, 5] in Cartesian order
            self.register_buffer('cg_11_2', cg_cart)  # [3, 3, 5]
        else:
            self.w_vt2 = None
            self.register_buffer('cg_11_2', None)

        # Path 8: ||f_t2||² → f_s (invariant contraction)
        if in_type2_channels > 0 and out_scalar_channels > 0:
            self.w_t2s = nn.Linear(in_type2_channels, out_scalar_channels, bias=False)
            self._num_paths += 1
        else:
            self.w_t2s = None

        # Radial basis
        self.radial_basis = RadialBasis(
            num_basis=num_radial_basis,
            cutoff=radius,
            num_hidden=64,
            num_output=max(self._num_paths, 1),
        )

        # Output bias (scalars only)
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
        type2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        """SE(3)-equivariant convolution.

        Args:
            scalars: [N, C_s_in]
            vectors: [N, C_v_in, 3], representation: SO(3) type-1
            graph: Pre-built SpatialGraph
            mask: [N] bool, optional
            type2: [N, C_t2_in, 5], representation: SO(3) type-2, optional

        Returns:
            If type2 is None: (scalar_out, vector_out)
            If type2 is given: (scalar_out, vector_out, type2_out)
        """
        N = graph.N
        device = scalars.device
        input_dtype = scalars.dtype
        has_type2 = type2 is not None and self.out_type2_channels > 0

        if graph.E == 0:
            s_out = scalars.new_zeros(N, self.out_scalar_channels)
            v_out = vectors.new_zeros(N, self.out_vector_channels, 3)
            s_out = s_out + (scalars.sum() * 0).unsqueeze(0)
            v_out = v_out + (vectors.sum() * 0).unsqueeze(0).unsqueeze(0)
            if self.scalar_bias is not None:
                s_out = s_out + self.scalar_bias
            if has_type2:
                t2_out = scalars.new_zeros(N, self.out_type2_channels, 5)
                return s_out, v_out, t2_out
            return s_out, v_out

        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type, enabled=False):
            compute_dtype = torch.float32 if scalars.dtype in (torch.float16, torch.bfloat16) else scalars.dtype
            scalars_f = scalars.to(compute_dtype)
            vectors_f = vectors.to(compute_dtype)

            # ── 1. Gather source features ──
            s_src = scalars_f[graph.row]
            v_src = vectors_f[graph.row].contiguous()

            # Type-2 gather
            t2_src = None
            if type2 is not None and self.in_type2_channels > 0:
                t2_src = type2.to(compute_dtype)[graph.row].contiguous()

            # ── 2. Compute radial weights ──
            R = self.radial_basis(graph.dist.to(compute_dtype))

            # ── 3. Compute messages ──
            Y_2_data = None
            if self.out_type2_channels > 0 and graph.Y.shape[-1] >= 9:
                Y_2_data = graph.Y[..., 4:9]  # l=2 SH: channels 4-8

            total_scalar_msg, total_vector_msg, total_type2_msg = _compute_messages(
                s_src, v_src,
                graph.direction.to(compute_dtype), graph.Y[..., 0:1].to(compute_dtype), R,
                self.w_ss.weight if self.w_ss is not None else None,
                self.w_sv.weight if self.w_sv is not None else None,
                self.w_vv_scalar.weight if self.w_vv_scalar is not None else None,
                self.w_vs.weight if self.w_vs is not None else None,
                self.w_vv_cross.weight if self.w_vv_cross is not None else None,
                self.out_scalar_channels,
                self.out_vector_channels,
                # Type-2 args
                t2_src=t2_src,
                Y_2=Y_2_data,
                w_st2=self.w_st2.weight if self.w_st2 is not None else None,
                w_t2t2=self.w_t2t2.weight if self.w_t2t2 is not None else None,
                w_vt2=self.w_vt2.weight if self.w_vt2 is not None else None,
                w_t2s=self.w_t2s.weight if self.w_t2s is not None else None,
                cg_11_2=self.cg_11_2,
                Ct2_out=self.out_type2_channels,
            )

            # ── 4. Aggregate (fused into single segment_reduce call) ──
            E = graph.E
            Cv3 = self.out_vector_channels * 3
            v_msg_flat = total_vector_msg.reshape(E, Cv3)

            if total_type2_msg is not None:
                # Fuse all 3 feature types into one reduce kernel launch
                Ct5 = self.out_type2_channels * 5
                t2_msg_flat = total_type2_msg.reshape(E, Ct5)
                all_msg = torch.cat([total_scalar_msg, v_msg_flat, t2_msg_flat], dim=-1)
                all_out = segment_reduce(
                    all_msg,
                    graph.col_sorted, graph.perm,
                    graph.node_start, graph.node_end, N,
                )
            else:
                # No type-2: fuse scalar + vector only (2→1 reduce)
                all_msg = torch.cat([total_scalar_msg, v_msg_flat], dim=-1)
                all_out = segment_reduce(
                    all_msg,
                    graph.col_sorted, graph.perm,
                    graph.node_start, graph.node_end, N,
                )

            # ── 4b. Batch-average degree normalization ──
            # segment_reduce uses SUM. In multi-scale architectures,
            # deeper stages have fewer points → each node has more
            # neighbors within radius → SUM grows with depth.
            #
            # MACE divides by a global avg_num_neighbors constant.
            # We use per-batch-average degree: same divisor for all
            # nodes within a conv call, so within-stage density
            # variation (boundary vs interior) is preserved.
            # This is a HARD divisor (not learnable).
            degree = (graph.node_end - graph.node_start).float()
            avg_degree = degree.clamp(min=1).mean()
            all_out = all_out / avg_degree

            Cs = self.out_scalar_channels
            if total_type2_msg is not None:
                s_out = all_out[:, :Cs]
                v_out = all_out[:, Cs:Cs+Cv3].reshape(N, self.out_vector_channels, 3)
                t2_out = all_out[:, Cs+Cv3:].reshape(N, self.out_type2_channels, 5)
            else:
                s_out = all_out[:, :Cs]
                v_out = all_out[:, Cs:].reshape(N, self.out_vector_channels, 3)
                t2_out = None

            # 5. Scalar bias
            if self.scalar_bias is not None:
                s_out = s_out + self.scalar_bias

            # 6. Mask
            if mask is not None:
                pad_mask = (~mask).unsqueeze(-1)
                s_out = s_out.masked_fill(pad_mask, 0.0)
                v_out = v_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                if t2_out is not None:
                    t2_out = t2_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if has_type2 and t2_out is not None:
            return s_out.to(input_dtype), v_out.to(input_dtype), t2_out.to(input_dtype)
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
        """Legacy forward: auto-builds SpatialGraph from pos."""
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
            f"type2: {self.in_type2_channels}→{self.out_type2_channels}, "
            f"radius={self.radius}, paths={self._num_paths}"
        )
