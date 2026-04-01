# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SO(3)-invariant self-attention with sinusoidal pairwise-distance RPE.

Digests same-cloud geometric structure so that downstream cross-attention
can perform pure semantic matching with geometry-aware features.

Design (mirrors GeoTransformer RPEConditionalTransformer):
    Self-Attention:  uses distance RPE  → geometric structure
    Cross-Attention: NO RPE at all      → pure feature matching

Mathematical guarantee:
    Pairwise distance ||p_i - p_j|| is SO(3)-invariant (rotations preserve
    distances).  The sinusoidal embedding and linear projection are
    element-wise / per-pair operations → the full RPE is SO(3)-invariant.
    Since Self-Attention only mixes l=0 scalars (already invariant),
    the output remains SO(3)-invariant.

Implementation:
    Uses ``F.scaled_dot_product_attention`` with MATH backend and float
    ``attn_mask`` as additive geometric bias.  FlashAttention is explicitly
    disabled because it silently ignores custom float attn_mask.

Architecture (Pre-LN Transformer block):
    x → LayerNorm → MultiHeadSelfAttn(+RPE) → + residual
      → LayerNorm → FFN → + residual → out
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


class SinusoidalDistanceEmbedding(nn.Module):
    """Multi-frequency sinusoidal embedding of pairwise distances.

    Encodes d / sigma_d with logarithmically spaced frequencies,
    producing 2 * n_freqs channels (sin + cos at each frequency).

    SO(3)-invariant: pairwise distances are rotation-invariant.

    Args:
        n_freqs: Number of frequency bands
        sigma_d: Distance temperature. For unit-sphere normalized
            point clouds, 0.1 gives fine-grained resolution.
    """

    def __init__(self, n_freqs: int = 8, sigma_d: float = 0.1) -> None:
        super().__init__()
        self.sigma_d = sigma_d
        self.n_freqs = n_freqs
        self.out_dim = 2 * n_freqs

        # Log-spaced frequencies: [1, 2, 4, ..., 2^(n_freqs-1)]
        freqs = torch.pow(2.0, torch.arange(n_freqs, dtype=torch.float32))
        self.register_buffer('freqs', freqs)  # [n_freqs]

    def forward(self, distances: Tensor) -> Tensor:
        """Encode pairwise distances.

        Args:
            distances: Pairwise distance matrix
                shape: [B, N, N], representation: SO(3) invariant

        Returns:
            Sinusoidal embeddings
                shape: [B, N, N, 2*n_freqs]
        """
        # Normalize by temperature
        x = distances / self.sigma_d  # [B, N, N]

        # Expand with frequencies: [B, N, N, n_freqs]
        x = x.unsqueeze(-1) * self.freqs  # broadcast

        # Sin + cos encoding
        return torch.cat([x.sin(), x.cos()], dim=-1)  # [B, N, N, 2*n_freqs]


class InvariantSelfAttention(nn.Module):
    """l=0 self-attention with sinusoidal pairwise-distance RPE.

    Pre-LN Transformer block that injects geometric structure via
    additive distance bias in the attention logits.

    Architecture::

        x ──→ LN ──→ MHSA(+RPE) ──→ (+) ──→ LN ──→ FFN ──→ (+) ──→ out
        │                             ↑      │                 ↑
        └─────────────────────────────┘      └─────────────────┘
                  residual                        residual

    Memory optimization:
        Instead of materializing [B, N, N, 2*n_freqs] sinusoidal embeddings
        (2+ GB at N=1024, B=32), we fuse the sinusoidal computation with
        the linear projection by iterating over frequency bands.  Each
        iteration only allocates [B, N, N], reducing peak memory from
        O(B·N²·2F) to O(B·N²·H).

    All operations are on l=0 scalars → SO(3)-invariant by construction.
    No dustbin needed — every point is a valid same-cloud neighbor.

    Args:
        channels: Scalar feature dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        sigma_d: Distance temperature for sinusoidal encoding
        n_freqs: Number of sinusoidal frequency bands
        ffn_ratio: FFN hidden dim = channels * ffn_ratio

    Example::

        >>> sa = InvariantSelfAttention(channels=32, num_heads=2)
        >>> s = torch.randn(4, 1024, 32)   # [B, N, C]
        >>> pos = torch.randn(4, 1024, 3)  # [B, N, 3]
        >>> out = sa(s, pos)                # [B, N, C]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 2,
        sigma_d: float = 0.1,
        n_freqs: int = 8,
        ffn_ratio: int = 1,
    ) -> None:
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.sigma_d = sigma_d
        self.n_freqs = n_freqs

        # ── Self-Attention projections (independent from CrossAttention) ──
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.o_proj = nn.Linear(channels, channels, bias=True)

        # ── Fused Distance RPE ──
        # Instead of SinusoidalDistanceEmbedding → Linear(2F, H),
        # we store the weight matrix [2*n_freqs, H] and fuse the
        # computation to avoid materializing the [B, N, N, 2F] tensor.
        # Log-spaced frequencies: [1, 2, 4, ..., 2^(n_freqs-1)]
        freqs = torch.pow(2.0, torch.arange(n_freqs, dtype=torch.float32))
        self.register_buffer('freqs', freqs)  # [n_freqs]

        # RPE weights: separate for sin and cos components
        # geo_bias[h] = Σ_k (w_sin[k,h] * sin(f_k * d/σ) + w_cos[k,h] * cos(f_k * d/σ)) + bias[h]
        self.rpe_weight_sin = nn.Parameter(torch.randn(n_freqs, num_heads) * 0.02)
        self.rpe_weight_cos = nn.Parameter(torch.randn(n_freqs, num_heads) * 0.02)
        self.rpe_bias = nn.Parameter(torch.zeros(num_heads))

        # ── Pre-LN ──
        self.norm_attn = nn.LayerNorm(channels)
        self.norm_ffn = nn.LayerNorm(channels)

        # ── FFN ──
        ffn_hidden = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, channels),
        )

    def _compute_fused_rpe(self, positions: Tensor) -> Tensor:
        """Compute per-head geometric bias with fused sinusoidal projection.

        Memory-efficient: iterates over frequency bands instead of
        materializing the full [B, N, N, 2*n_freqs] embedding tensor.

        Args:
            positions: Point positions
                shape: [B, N, 3], representation: SO(3) invariant

        Returns:
            Per-head geometric bias
                shape: [B, H, N, N], dtype: float
        """
        # Pairwise distances (SO(3)-invariant, no grad needed)
        with torch.no_grad():
            dist = torch.cdist(positions, positions)  # [B, N, N]
            scaled_dist = dist / self.sigma_d  # [B, N, N]

        # Accumulate bias per head: Σ_k (w_sin[k,h] * sin + w_cos[k,h] * cos)
        B, N, _ = positions.shape
        H = self.num_heads

        # Initialize with broadcast bias: [H] → [1, H, 1, 1]
        geo_bias = self.rpe_bias.view(1, H, 1, 1).expand(B, H, N, N).clone()

        # Fused loop over frequency bands — each iteration uses O(B*N²) memory
        for k in range(self.n_freqs):
            # [B, N, N] — single frequency band
            phase = scaled_dist * self.freqs[k]

            sin_k = phase.sin()  # [B, N, N]
            cos_k = phase.cos()  # [B, N, N]

            # w_sin[k, :] and w_cos[k, :] are [H] vectors
            # Accumulate: geo_bias[b, h, i, j] += w_sin[k,h] * sin_k[b,i,j] + w_cos[k,h] * cos_k[b,i,j]
            geo_bias += sin_k.unsqueeze(1) * self.rpe_weight_sin[k].view(1, H, 1, 1)
            geo_bias += cos_k.unsqueeze(1) * self.rpe_weight_cos[k].view(1, H, 1, 1)

        return geo_bias

    def forward(
        self,
        scalars: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Self-attention with distance RPE on l=0 scalars.

        Args:
            scalars: l=0 scalar features
                shape: [B, N, C], representation: SO(3) invariant
            positions: Point cloud coordinates
                shape: [B, N, 3]
            mask: Valid point mask [B, N] bool (True=real, False=pad)

        Returns:
            Updated scalar features
                shape: [B, N, C], representation: SO(3) invariant
        """
        B, N, C = scalars.shape
        H = self.num_heads
        D = self.head_dim

        # ── Sub-block 1: Pre-LN Self-Attention with RPE ──
        x = self.norm_attn(scalars)

        # Q, K, V projections → [B, H, N, D]
        Q = self.q_proj(x).reshape(B, N, H, D).transpose(1, 2)
        K = self.k_proj(x).reshape(B, N, H, D).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, H, D).transpose(1, 2)

        # Fused distance RPE → [B, H, N, N] (memory-efficient)
        geo_bias = self._compute_fused_rpe(positions)

        # Combine with padding mask if needed
        if mask is not None:
            # Pad points should not attend to or be attended by anything
            pad_mask = mask[:, None, None, :]  # [B, 1, 1, N] — key mask
            geo_bias = geo_bias.masked_fill(~pad_mask, float('-inf'))

        # SDPA with MATH backend (FlashAttn silently ignores float attn_mask)
        with sdpa_kernel([SDPBackend.MATH]):
            attn_out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=geo_bias,
            )  # [B, H, N, D]

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.o_proj(attn_out)

        # Residual
        scalars = scalars + attn_out

        # ── Sub-block 2: Pre-LN FFN ──
        x_ffn = self.norm_ffn(scalars)
        scalars = scalars + self.ffn(x_ffn)

        # Zero out pad positions
        if mask is not None:
            scalars = scalars.masked_fill(~mask.unsqueeze(-1), 0.0)

        return scalars
