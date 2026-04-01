# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Equivariant normalization layers.

Standard BatchNorm/LayerNorm destroys equivariance because it normalizes
spatial dimensions (the xyz components of vectors). These variants are safe:

- EquivariantLayerNorm: e3nn-style equivariant batch normalization.
  - Scalars (l=0): standard LayerNorm (per-point, over channels)
  - Vectors (l=1): BatchNorm on ||v_c||² with running stats (per-channel,
    across all points in batch). Preserves direction and per-point relative
    magnitude while controlling overall scale growth.

Per AGENTS.md Rule 3: ordinary spatial BatchNorm on xyz components is FORBIDDEN.
This module normalizes by the *norm* of vectors, which is SO(3)-invariant.

Reference: e3nn.nn.BatchNorm (https://github.com/e3nn/e3nn)
"""

import torch
import torch.nn as nn
from torch import Tensor


class EquivariantLayerNorm(nn.Module):
    """SO(3)-equivariant normalization with running statistics.

    For scalar features (l=0): per-point LayerNorm over channels.
    For vector features (l=1): per-channel BatchNorm on ||v||².

    Vector normalization detail (e3nn-style):
        Training:
            1. Compute batch_var_c = mean_over_N( ||v_{i,c}||² )
            2. Update running_var_c via EMA: (1-m)*running + m*batch
            3. Normalize: v_out = v / sqrt(batch_var + eps) * weight
        Eval:
            1. Use running_var_c (frozen, tracked during training)
            2. Normalize: v_out = v / sqrt(running_var + eps) * weight

    This preserves:
        - SO(3) equivariance (dividing by invariant scalar)
        - Per-point spatial variation (all points share the same
          per-channel divisor, so relative magnitudes are preserved)
        - Scale control (running_var prevents unbounded growth)

    Args:
        num_scalars: Number of scalar (l=0) channels
        num_vectors: Number of vector (l=1) channels
        eps: Numerical epsilon for normalization stability
        momentum: EMA momentum for running stats (e3nn default: 0.1)
        affine: Whether to add learnable scale parameters

    Example::

        >>> norm = EquivariantLayerNorm(num_scalars=64, num_vectors=16)
        >>> s, v = torch.randn(N, 64), torch.randn(N, 16, 3)
        >>> s_out, v_out = norm(s, v)  # train mode: uses batch stats
        >>> norm.eval()
        >>> s_out, v_out = norm(s, v)  # eval mode: uses running stats
    """

    def __init__(
        self,
        num_scalars: int,
        num_vectors: int = 0,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        self.eps = eps
        self.momentum = momentum

        # ── Scalar: per-point LayerNorm parameters ──
        if affine and num_scalars > 0:
            self.scalar_weight = nn.Parameter(torch.ones(num_scalars))
            self.scalar_bias = nn.Parameter(torch.zeros(num_scalars))
        else:
            self.register_parameter('scalar_weight', None)
            self.register_parameter('scalar_bias', None)

        # ── Vector: BatchNorm on ||v||² ──
        if num_vectors > 0:
            # running_var tracks EMA of mean(||v_c||²) per channel
            # Initialized to 1.0 (unit norm assumption)
            self.register_buffer(
                'running_var', torch.ones(num_vectors)
            )
            self.register_buffer(
                'num_batches_tracked', torch.tensor(0, dtype=torch.long)
            )

            if affine:
                # Learnable per-channel scale (no bias for vectors!)
                self.vector_weight = nn.Parameter(torch.ones(num_vectors))
            else:
                self.register_parameter('vector_weight', None)
        else:
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
            self.register_parameter('vector_weight', None)

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply equivariant normalization.

        Args:
            scalars: Scalar features
                shape: [N, num_scalars] or [..., num_scalars]
            vectors: Vector features
                shape: [N, num_vectors, 3] or [..., num_vectors, 3]
                representation: SO(3) type-1 vectors

        Returns:
            Tuple of (normalized_scalars, normalized_vectors)
        """
        # ── Scalar normalization (per-point LayerNorm) ──
        s_mean = scalars.mean(dim=-1, keepdim=True)
        s_var = scalars.var(dim=-1, keepdim=True, unbiased=False)
        scalars_out = (scalars - s_mean) / torch.sqrt(s_var + self.eps)

        if self.scalar_weight is not None:
            scalars_out = scalars_out * self.scalar_weight + self.scalar_bias

        # ── Vector normalization (BatchNorm on ||v||²) ──
        # e3nn-style: normalize by per-channel RMS with running stats.
        #
        # Equivariance proof:
        #   var_c = mean_i(||v_{i,c}||²) is SO(3)-invariant
        #   v_out = v / sqrt(var_c) preserves equivariance
        #
        # Spatial variation:
        #   All N points share the SAME per-channel divisor sqrt(var_c),
        #   so per-point relative magnitudes are exactly preserved.
        if self.num_vectors > 0 and vectors.shape[-2] > 0:
            # Per-point, per-channel squared norms: [..., C_v]
            v_sq_norms = vectors.pow(2).sum(dim=-1)  # [..., C_v]

            if self.training:
                # ── Training: use batch statistics ──
                # Flatten to [N_total, C_v] for population mean
                flat_sq = v_sq_norms.reshape(-1, self.num_vectors)
                # Per-channel mean of ||v_c||² across all points
                batch_var = flat_sq.mean(dim=0)  # [C_v]

                # Update running stats via EMA
                with torch.no_grad():
                    self.running_var.copy_(
                        (1.0 - self.momentum) * self.running_var
                        + self.momentum * batch_var
                    )
                    self.num_batches_tracked += 1

                # Normalize using batch stats
                inv_rms = (batch_var + self.eps).pow(-0.5)  # [C_v]
            else:
                # ── Eval: use running stats (frozen) ──
                inv_rms = (self.running_var + self.eps).pow(-0.5)  # [C_v]

            # Broadcast inv_rms to match vectors shape: [..., C_v, 3]
            inv_rms_bc = inv_rms
            for _ in range(vectors.dim() - 2):
                inv_rms_bc = inv_rms_bc.unsqueeze(0)
            inv_rms_bc = inv_rms_bc.unsqueeze(-1)  # [..., C_v, 1]

            vectors_out = vectors * inv_rms_bc

            if self.vector_weight is not None:
                # Apply learnable per-channel scale
                weight_bc = self.vector_weight
                for _ in range(vectors.dim() - 2):
                    weight_bc = weight_bc.unsqueeze(0)
                weight_bc = weight_bc.unsqueeze(-1)  # [..., C_v, 1]
                vectors_out = vectors_out * weight_bc
        else:
            vectors_out = vectors

        return scalars_out, vectors_out

    def extra_repr(self) -> str:
        return (
            f"num_scalars={self.num_scalars}, "
            f"num_vectors={self.num_vectors}, "
            f"eps={self.eps}, momentum={self.momentum}"
        )
