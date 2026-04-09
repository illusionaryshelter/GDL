# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Equivariant cross-scale skip fusion for multi-scale SE(3) decoders.

Replaces the naive ``cat(interp, skip) → Linear`` skip connections with
direction-aware tensor product fusion.  For each fine-resolution point,
messages from its K nearest coarse-resolution neighbors are computed via
the **same TP mechanism** used by SE3Conv — but the graph edges now span
coarse→fine rather than within a single resolution.

This module solves the multi-scale information bottleneck identified by
SVD analysis of the trained head: the old linear projections completely
discard the coarse→fine displacement direction, forcing the classification
head to compensate by directly ingesting raw multi-scale encoder globals.

Architecture::

    coarse features  →  TP(Y_l(d̂), feature)  →  aggregate  →  norm  →  gate
                                                      ↕
    skip features  ────────────────────────────→  gated add  →  output

Mathematical guarantee:
    All operations use CG tensor products with SH filters and invariant
    radial weights.  Equivariance follows from the Wigner-Eckart theorem,
    exactly as for SE3Conv.

K-neighbor selection:
    We use K=6 as default (vs K=3 for the old IDW interpolation) because
    TP messages require angular diversity — with only 3 neighbors the SH
    filter can't capture rich directional gradients.  K=6 gives ~2x the
    solid-angle coverage at modest cost (+50% edges vs K=3).  The radial
    basis provides a soft distance cutoff, so distant neighbors are
    automatically down-weighted.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from geoembodied.functional.knn import knn
from geoembodied.functional.tensor_product import get_cg_matrix
from geoembodied.kernels.triton_sph_harm import spherical_harmonics
from geoembodied.nn.modules.se3_conv import RadialBasis, _compute_messages
from geoembodied.nn.modules.equivariant_norm import EquivariantLayerNorm
from geoembodied.nn.modules.gated_nonlinearity import GatedNonlinearity


class EquivariantSkipFusion(nn.Module):
    """Cross-scale TP fusion replacing concat+Linear skip connections.

    For each fine point i, gathers features from K nearest coarse points,
    computes TP messages using the coarse→fine displacement direction as
    the SH filter, aggregates, normalizes, gates, then fuses with the
    encoder skip features via learned scalar gates.

    Args:
        scalar_channels: l=0 feature channels (same in/out)
        vector_channels: l=1 feature channels (same in/out)
        type2_channels: l=2 feature channels (same in/out), 0 to disable
        k_neighbors: Number of coarse neighbors per fine point (default 6)
        num_radial_basis: Number of radial basis functions
        gate_mode: Gated nonlinearity mode ('norm' or 'scalar')
        cutoff_multiplier: Cutoff = median_coarse_spacing * multiplier

    Example::

        >>> fusion = EquivariantSkipFusion(96, 32, 16)
        >>> s, v, t2 = fusion(
        ...     fine_pos, coarse_pos,
        ...     s_coarse, v_coarse, ptr_fine, ptr_coarse,
        ...     s_skip=s_enc, v_skip=v_enc,
        ...     t2_coarse=t2_c, t2_skip=t2_enc,
        ... )
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        type2_channels: int = 0,
        k_neighbors: int = 6,
        num_radial_basis: int = 16,
        gate_mode: str = 'norm',
        cutoff_multiplier: float = 6.0,
    ) -> None:
        super().__init__()
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.type2_channels = type2_channels
        self.k_neighbors = k_neighbors
        self.cutoff_multiplier = cutoff_multiplier

        C_s = scalar_channels
        C_v = vector_channels
        C_t2 = type2_channels

        # ── TP path weights (same structure as SE3Conv) ──
        self._num_paths = 0

        # Path 0: Y_0 × s_coarse → s_out
        self.w_ss = nn.Linear(C_s, C_s, bias=False)
        self._num_paths += 1

        # Path 1: dir × s_coarse → v_out
        if C_v > 0:
            self.w_sv = nn.Linear(C_s, C_v, bias=False)
            self._num_paths += 1
        else:
            self.w_sv = None

        # Path 2: Y_0 × v_coarse → v_out
        if C_v > 0:
            self.w_vv_scalar = nn.Linear(C_v, C_v, bias=False)
            self._num_paths += 1
        else:
            self.w_vv_scalar = None

        # Path 3: dir · v_coarse → s_out (direction-aware invariant)
        if C_v > 0:
            self.w_vs = nn.Linear(C_v, C_s, bias=False)
            self._num_paths += 1
        else:
            self.w_vs = None

        # Path 4: dir × v_coarse → v_out (cross product)
        if C_v > 0:
            self.w_vv_cross = nn.Linear(C_v, C_v, bias=False)
            self._num_paths += 1
        else:
            self.w_vv_cross = None

        # Path 5: Y_2 × s_coarse → t2_out
        if C_t2 > 0:
            self.w_st2 = nn.Linear(C_s, C_t2, bias=False)
            self._num_paths += 1
        else:
            self.w_st2 = None

        # Path 6: Y_0 × t2_coarse → t2_out
        if C_t2 > 0:
            self.w_t2t2 = nn.Linear(C_t2, C_t2, bias=False)
            self._num_paths += 1
        else:
            self.w_t2t2 = None

        # Path 7: dir ⊗ v_coarse → t2_out (CG l=1⊗l=1→l=2)
        if C_v > 0 and C_t2 > 0:
            self.w_vt2 = nn.Linear(C_v, C_t2, bias=False)
            self._num_paths += 1
            # Pre-permuted CG matrix (SH→Cartesian, same as SE3Conv)
            cg_sh = get_cg_matrix(1, 1, 2)  # [9, 5]
            cg_3d = cg_sh.reshape(3, 3, 5)
            P = torch.tensor([2, 0, 1])  # SH→Cartesian
            cg_cart = cg_3d[P][:, P]
            self.register_buffer('cg_11_2', cg_cart)
        else:
            self.w_vt2 = None
            self.register_buffer('cg_11_2', None)

        # Path 8: ||t2_coarse||² → s_out (invariant contraction)
        if C_t2 > 0:
            self.w_t2s = nn.Linear(C_t2, C_s, bias=False)
            self._num_paths += 1
        else:
            self.w_t2s = None

        # Radial basis (adaptive cutoff estimated at runtime)
        # Use a generous default cutoff; the radial envelope handles decay
        self.radial_basis = RadialBasis(
            num_basis=num_radial_basis,
            cutoff=1.0,  # Will be re-scaled at runtime
            num_hidden=64,
            num_output=max(self._num_paths, 1),
        )

        # Scalar bias
        self.scalar_bias = nn.Parameter(torch.zeros(C_s))

        # ── Post-TP normalization (critical for variance stability) ──
        # TP outputs from different paths have wildly different scales.
        # EquivariantLayerNorm brings them to a common variance BEFORE gating.
        self.tp_norm = EquivariantLayerNorm(
            num_scalars=C_s, num_vectors=C_v, num_type2=C_t2,
        )

        # ── Gated nonlinearity ──
        # Applies non-linear activation to scalar channels and
        # norm-based gating to l≥1 channels.
        self.gate = GatedNonlinearity(
            num_scalars=C_s, num_vectors=C_v, num_type2=C_t2,
            gate_mode=gate_mode,
        )

        # ── Skip gate: learned scalar gates for residual fusion ──
        # gate(s_tp, s_skip) → [0, 1] per channel
        # This lets the network learn how much encoder vs decoder info to use
        gate_in = 2 * C_s  # concat(s_tp, s_skip)
        self.skip_gate_s = nn.Sequential(
            nn.Linear(gate_in, C_s),
            nn.Sigmoid(),
        )
        if C_v > 0:
            self.skip_gate_v = nn.Sequential(
                nn.Linear(gate_in, C_v),
                nn.Sigmoid(),
            )
        else:
            self.skip_gate_v = None

        if C_t2 > 0:
            self.skip_gate_t2 = nn.Sequential(
                nn.Linear(gate_in, C_t2),
                nn.Sigmoid(),
            )
        else:
            self.skip_gate_t2 = None

        # ── Output norm (stabilize before decoder stage) ──
        self.output_norm = EquivariantLayerNorm(
            num_scalars=C_s, num_vectors=C_v, num_type2=C_t2,
        )

    def _estimate_cutoff(
        self, coarse_pos: Tensor, ptr_coarse: Tensor,
    ) -> float:
        """Estimate a reasonable cutoff from coarse point spacing.

        Uses median nearest-neighbor distance × cutoff_multiplier.
        This adapts to variable point density across batches.

        Returns:
            cutoff: float, estimated adaptive cutoff distance
        """
        with torch.no_grad():
            # Use a small K to find nearest neighbor distances
            knn_idx, knn_dist_sq = knn(
                coarse_pos, coarse_pos, ptr_coarse, ptr_coarse, k=2,
            )
            # knn_dist_sq: [N_coarse, 2] — dist to self (0) and nearest neighbor
            nn_dist = knn_dist_sq[:, -1].clamp(min=1e-12).sqrt()
            median_dist = nn_dist.median().item()
            return median_dist * self.cutoff_multiplier

    def forward(
        self,
        fine_pos: Tensor,
        coarse_pos: Tensor,
        s_coarse: Tensor,
        v_coarse: Tensor,
        ptr_fine: Tensor,
        ptr_coarse: Tensor,
        s_skip: Tensor,
        v_skip: Tensor,
        t2_coarse: Optional[Tensor] = None,
        t2_skip: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        """Cross-scale TP fusion: coarse→fine with skip gating.

        Args:
            fine_pos: [N_fine, 3] — fine resolution positions
            coarse_pos: [N_coarse, 3] — coarse resolution positions
            s_coarse: [N_coarse, C_s] — coarse scalar features
            v_coarse: [N_coarse, C_v, 3] — coarse vector features,
                representation: SO(3) type-1
            ptr_fine: [B+1] int64 — batch pointers for fine resolution
            ptr_coarse: [B+1] int64 — batch pointers for coarse resolution
            s_skip: [N_fine, C_s] — encoder skip scalar features
            v_skip: [N_fine, C_v, 3] — encoder skip vector features,
                representation: SO(3) type-1
            t2_coarse: Optional [N_coarse, C_t2, 5] — coarse type-2 features,
                representation: SO(3) type-2
            t2_skip: Optional [N_fine, C_t2, 5] — encoder skip type-2 features,
                representation: SO(3) type-2

        Returns:
            Without type-2: (s_out, v_out)  — both at fine resolution
            With type-2: (s_out, v_out, t2_out)
        """
        N_fine = fine_pos.shape[0]
        device = fine_pos.device
        input_dtype = s_coarse.dtype
        has_t2 = t2_coarse is not None and self.type2_channels > 0

        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type, enabled=False):
            compute_dtype = torch.float32 if input_dtype in (
                torch.float16, torch.bfloat16,
            ) else input_dtype

            # Cast to compute dtype
            s_coarse_f = s_coarse.to(compute_dtype)
            v_coarse_f = v_coarse.to(compute_dtype)
            fine_pos_f = fine_pos.to(compute_dtype)
            coarse_pos_f = coarse_pos.to(compute_dtype)

            # ── 1. Cross-scale KNN: for each fine point, find K coarse neighbors ──
            knn_idx, knn_dist_sq = knn(
                fine_pos_f, coarse_pos_f, ptr_fine, ptr_coarse,
                k=self.k_neighbors,
            )
            # knn_idx: [N_fine, K], knn_dist_sq: [N_fine, K]

            valid_mask = knn_idx >= 0  # [N_fine, K]
            safe_idx = knn_idx.clamp(min=0)  # [N_fine, K]

            # ── 2. Compute edge vectors and SH filters ──
            # Flatten edges: [N_fine * K]
            K = self.k_neighbors
            fine_expanded = fine_pos_f.unsqueeze(1).expand(-1, K, -1)  # [N_fine, K, 3]
            coarse_gathered = coarse_pos_f[safe_idx]  # [N_fine, K, 3]
            edge_vec = fine_expanded - coarse_gathered  # [N_fine, K, 3]
            edge_dist = edge_vec.norm(dim=-1).clamp(min=1e-8)  # [N_fine, K]
            edge_dir = edge_vec / edge_dist.unsqueeze(-1)  # [N_fine, K, 3]

            # Zero out invalid edges
            edge_dir = edge_dir.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            edge_dist = edge_dist.masked_fill(~valid_mask, 0.0)

            # Flatten for TP computation
            E = N_fine * K
            edge_dir_flat = edge_dir.reshape(E, 3)  # [E, 3]
            edge_dist_flat = edge_dist.reshape(E)  # [E]

            # Spherical harmonics: Y_0 = 1/√(4π), Y_1 = dir, Y_2 = 5-component
            Y = spherical_harmonics(edge_dir_flat, max_l=2)  # [E, 9]
            Y_0 = Y[:, 0:1]  # [E, 1]
            Y_2 = Y[:, 4:9] if has_t2 else None  # [E, 5] or None

            # ── 3. Gather coarse source features ──
            s_src = s_coarse_f[safe_idx.reshape(-1)].reshape(E, -1)  # [E, C_s]
            v_src = v_coarse_f[safe_idx.reshape(-1)].reshape(E, self.vector_channels, 3)

            t2_src = None
            if has_t2:
                t2_src = t2_coarse.to(compute_dtype)[safe_idx.reshape(-1)].reshape(
                    E, self.type2_channels, 5,
                )

            # ── 4. Radial weights ──
            # Estimate cutoff adaptively
            cutoff = self._estimate_cutoff(coarse_pos_f, ptr_coarse)
            # Rescale distances to [0, 1] range for the radial basis
            dist_scaled = edge_dist_flat / max(cutoff, 1e-6)
            # Reuse the radial basis but with normalized distances
            # We need to temporarily adjust the cutoff
            old_cutoff = self.radial_basis.cutoff
            self.radial_basis.cutoff = 1.0  # distances are pre-normalized
            R = self.radial_basis(dist_scaled.clamp(max=1.0))  # [E, num_paths]
            self.radial_basis.cutoff = old_cutoff

            # Zero out invalid edges
            invalid_flat = ~valid_mask.reshape(E)
            R = R.masked_fill(invalid_flat.unsqueeze(-1), 0.0)

            # ── 5. Compute TP messages ──
            s_msg, v_msg, t2_msg = _compute_messages(
                s_src, v_src.contiguous(),
                edge_dir_flat, Y_0, R,
                self.w_ss.weight, 
                self.w_sv.weight if self.w_sv is not None else None,
                self.w_vv_scalar.weight if self.w_vv_scalar is not None else None,
                self.w_vs.weight if self.w_vs is not None else None,
                self.w_vv_cross.weight if self.w_vv_cross is not None else None,
                self.scalar_channels,
                self.vector_channels,
                t2_src=t2_src,
                Y_2=Y_2,
                w_st2=self.w_st2.weight if self.w_st2 is not None else None,
                w_t2t2=self.w_t2t2.weight if self.w_t2t2 is not None else None,
                w_vt2=self.w_vt2.weight if self.w_vt2 is not None else None,
                w_t2s=self.w_t2s.weight if self.w_t2s is not None else None,
                cg_11_2=self.cg_11_2,
                Ct2_out=self.type2_channels,
            )

            # ── 6. Aggregate: weighted mean over K neighbors ──
            # Reshape back: [N_fine, K, ...]
            s_msg = s_msg.reshape(N_fine, K, self.scalar_channels)
            v_msg = v_msg.reshape(N_fine, K, self.vector_channels, 3)

            # Inverse-distance weights for aggregation
            inv_dist = 1.0 / edge_dist.clamp(min=1e-6)  # [N_fine, K]
            inv_dist = inv_dist.masked_fill(~valid_mask, 0.0)
            weight_sum = inv_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            agg_weights = inv_dist / weight_sum  # [N_fine, K]

            # Weighted mean
            s_tp = (agg_weights.unsqueeze(-1) * s_msg).sum(dim=1)  # [N_fine, C_s]
            v_tp = (agg_weights.unsqueeze(-1).unsqueeze(-1) * v_msg).sum(dim=1)

            if has_t2 and t2_msg is not None:
                t2_msg = t2_msg.reshape(N_fine, K, self.type2_channels, 5)
                t2_tp = (agg_weights.unsqueeze(-1).unsqueeze(-1) * t2_msg).sum(dim=1)
            else:
                t2_tp = None

            # Scalar bias
            s_tp = s_tp + self.scalar_bias

            # ── 7. Post-TP normalization → gate ──
            # Critical: brings multi-path TP outputs to consistent variance
            if has_t2 and t2_tp is not None:
                s_tp, v_tp, t2_tp = self.tp_norm(s_tp, v_tp, t2_tp)
                s_tp, v_tp, t2_tp = self.gate(s_tp, v_tp, t2_tp)
            else:
                s_tp, v_tp = self.tp_norm(s_tp, v_tp)
                s_tp, v_tp = self.gate(s_tp, v_tp)

            # ── 8. Gated skip fusion ──
            s_skip_f = s_skip.to(compute_dtype)
            v_skip_f = v_skip.to(compute_dtype)

            gate_input = torch.cat([s_tp, s_skip_f], dim=-1)  # [N_fine, 2*C_s]

            # Scalar fusion: s_out = g_s * s_skip + (1-g_s) * s_tp
            g_s = self.skip_gate_s(gate_input)  # [N_fine, C_s]
            s_out = g_s * s_skip_f + (1.0 - g_s) * s_tp

            # Vector fusion: gate from scalars (equivariant)
            if self.skip_gate_v is not None:
                g_v = self.skip_gate_v(gate_input)  # [N_fine, C_v]
                v_out = g_v.unsqueeze(-1) * v_skip_f + (1.0 - g_v).unsqueeze(-1) * v_tp
            else:
                v_out = v_tp

            # Type-2 fusion
            if has_t2 and t2_tp is not None and self.skip_gate_t2 is not None:
                t2_skip_f = t2_skip.to(compute_dtype) if t2_skip is not None else torch.zeros_like(t2_tp)
                g_t2 = self.skip_gate_t2(gate_input)  # [N_fine, C_t2]
                t2_out = g_t2.unsqueeze(-1) * t2_skip_f + (1.0 - g_t2).unsqueeze(-1) * t2_tp
            else:
                t2_out = t2_tp

            # ── 9. Output normalization ──
            if has_t2 and t2_out is not None:
                s_out, v_out, t2_out = self.output_norm(s_out, v_out, t2_out)
            else:
                s_out, v_out = self.output_norm(s_out, v_out)

        # Cast back
        s_out = s_out.to(input_dtype)
        v_out = v_out.to(input_dtype)

        if has_t2 and t2_out is not None:
            return s_out, v_out, t2_out.to(input_dtype)
        return s_out, v_out

    def extra_repr(self) -> str:
        return (
            f"scalar={self.scalar_channels}, "
            f"vector={self.vector_channels}, "
            f"type2={self.type2_channels}, "
            f"k={self.k_neighbors}, "
            f"paths={self._num_paths}"
        )
