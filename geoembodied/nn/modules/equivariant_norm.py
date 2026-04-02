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
  - Type-2 (l=2): BatchNorm on ||t2_c||² = Σ_m t2[c,m]² (5 components).
    Same math as vectors but with 5 components instead of 3.

Per AGENTS.md Rule 3: ordinary spatial BatchNorm on xyz components is FORBIDDEN.
This module normalizes by the *norm* of features, which is SO(3)-invariant.

Reference: e3nn.nn.BatchNorm (https://github.com/e3nn/e3nn)
"""

import torch
import torch.nn as nn
from torch import Tensor


class EquivariantLayerNorm(nn.Module):
    """SO(3)-equivariant normalization with running statistics.

    For scalar features (l=0): per-point LayerNorm over channels.
    For vector features (l=1): per-channel BatchNorm on ||v||².
    For type-2 features (l=2): per-channel BatchNorm on ||t2||².

    Vector/type-2 normalization detail (e3nn-style):
        Training:
            1. Compute batch_var_c = mean_over_N( ||f_{i,c}||² )
            2. Update running_var_c via EMA: (1-m)*running + m*batch
            3. Normalize: f_out = f / sqrt(batch_var + eps) * weight
        Eval:
            1. Use running_var_c (frozen, tracked during training)
            2. Normalize: f_out = f / sqrt(running_var + eps) * weight

    This preserves:
        - SO(3) equivariance (dividing by invariant scalar)
        - Per-point spatial variation (all points share the same
          per-channel divisor, so relative magnitudes are preserved)
        - Scale control (running_var prevents unbounded growth)

    Args:
        num_scalars: Number of scalar (l=0) channels
        num_vectors: Number of vector (l=1) channels
        num_type2: Number of type-2 (l=2) channels
        eps: Numerical epsilon for normalization stability
        momentum: EMA momentum for running stats (e3nn default: 0.1)
        affine: Whether to add learnable scale parameters

    Example::

        >>> norm = EquivariantLayerNorm(num_scalars=64, num_vectors=16, num_type2=4)
        >>> s, v, t2 = torch.randn(N, 64), torch.randn(N, 16, 3), torch.randn(N, 4, 5)
        >>> s_out, v_out, t2_out = norm(s, v, t2)
    """

    def __init__(
        self,
        num_scalars: int,
        num_vectors: int = 0,
        num_type2: int = 0,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        self.num_type2 = num_type2
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
            self.register_buffer('running_var', torch.ones(num_vectors))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            if affine:
                self.vector_weight = nn.Parameter(torch.ones(num_vectors))
            else:
                self.register_parameter('vector_weight', None)
        else:
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
            self.register_parameter('vector_weight', None)

        # ── Type-2: BatchNorm on ||t2||² ──
        if num_type2 > 0:
            self.register_buffer('running_var_t2', torch.ones(num_type2))
            self.register_buffer('num_batches_tracked_t2', torch.tensor(0, dtype=torch.long))
            if affine:
                self.type2_weight = nn.Parameter(torch.ones(num_type2))
            else:
                self.register_parameter('type2_weight', None)
        else:
            self.register_buffer('running_var_t2', None)
            self.register_buffer('num_batches_tracked_t2', None)
            self.register_parameter('type2_weight', None)

    def _norm_higher_order(
        self,
        features: Tensor,
        running_var: Tensor,
        weight: Tensor | None,
        num_channels: int,
    ) -> Tensor:
        """Normalize l≥1 features by per-channel RMS with running stats.

        Works for both l=1 (3 components) and l=2 (5 components).

        Equivariance proof:
            ||R·f||² = f^T R^T R f = ||f||² is SO(3)-invariant.
            Dividing by sqrt(mean(||f||²)) preserves the transformation.

        Args:
            features: shape [..., C, D] where D=3 (l=1) or D=5 (l=2)
            running_var: shape [C]
            weight: shape [C] or None
            num_channels: C

        Returns:
            Normalized features, same shape as input
        """
        # Per-point, per-channel squared norms: [..., C]
        f_sq_norms = features.pow(2).sum(dim=-1)  # [..., C]

        if self.training:
            # Flatten to [N_total, C] for population mean
            flat_sq = f_sq_norms.reshape(-1, num_channels)
            batch_var = flat_sq.mean(dim=0)  # [C]

            with torch.no_grad():
                running_var.copy_(
                    (1.0 - self.momentum) * running_var
                    + self.momentum * batch_var
                )

            inv_rms = (batch_var + self.eps).pow(-0.5)  # [C]
        else:
            inv_rms = (running_var + self.eps).pow(-0.5)  # [C]

        # Broadcast inv_rms to match features shape: [..., C, D]
        inv_rms_bc = inv_rms
        for _ in range(features.dim() - 2):
            inv_rms_bc = inv_rms_bc.unsqueeze(0)
        inv_rms_bc = inv_rms_bc.unsqueeze(-1)  # [..., C, 1]

        out = features * inv_rms_bc

        if weight is not None:
            weight_bc = weight
            for _ in range(features.dim() - 2):
                weight_bc = weight_bc.unsqueeze(0)
            weight_bc = weight_bc.unsqueeze(-1)  # [..., C, 1]
            out = out * weight_bc

        return out

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        type2: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Apply equivariant normalization.

        Args:
            scalars: Scalar features
                shape: [N, num_scalars]
            vectors: Vector features
                shape: [N, num_vectors, 3], representation: SO(3) type-1
            type2: Type-2 features (optional)
                shape: [N, num_type2, 5], representation: SO(3) type-2

        Returns:
            If type2 is None: (normalized_scalars, normalized_vectors)
            If type2 is given: (normalized_scalars, normalized_vectors, normalized_type2)
        """
        input_dtype = scalars.dtype
        scalars = scalars.float()
        vectors = vectors.float()

        # ── Scalar normalization (per-point LayerNorm) ──
        s_mean = scalars.mean(dim=-1, keepdim=True)
        s_var = scalars.var(dim=-1, keepdim=True, unbiased=False)
        scalars_out = (scalars - s_mean) / torch.sqrt(s_var + self.eps)

        if self.scalar_weight is not None:
            scalars_out = scalars_out * self.scalar_weight + self.scalar_bias

        # ── Vector normalization (BatchNorm on ||v||²) ──
        if self.num_vectors > 0 and vectors.shape[-2] > 0:
            vectors_out = self._norm_higher_order(
                vectors, self.running_var, self.vector_weight, self.num_vectors
            )
        else:
            vectors_out = vectors

        # ── Type-2 normalization (BatchNorm on ||t2||²) ──
        if type2 is not None and self.num_type2 > 0:
            type2 = type2.float()
            type2_out = self._norm_higher_order(
                type2, self.running_var_t2, self.type2_weight, self.num_type2
            )
            return (
                scalars_out.to(input_dtype),
                vectors_out.to(input_dtype),
                type2_out.to(input_dtype),
            )

        return scalars_out.to(input_dtype), vectors_out.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"num_scalars={self.num_scalars}, "
            f"num_vectors={self.num_vectors}, "
            f"num_type2={self.num_type2}, "
            f"eps={self.eps}, momentum={self.momentum}"
        )
