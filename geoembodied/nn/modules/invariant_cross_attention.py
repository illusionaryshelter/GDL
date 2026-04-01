# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SO(3)-invariant cross-attention with dustbin for mid-fusion registration.

Exchanges ONLY l=0 (scalar/invariant) features between two point clouds.
Vector (l=1) features are NEVER mixed across clouds — they are modulated
indirectly via scalar gating (0⊗1→1 path).

Dustbin mechanism (inspired by SuperGlue):
    A learnable "dustbin" key/value token is appended to each K/V sequence.
    Points in occluded regions — which have NO corresponding point in the
    other cloud — can route their attention to this dustbin instead of
    being forced to match an irrelevant point. This produces a near-zero
    cross-cloud message, which the downstream ScalarGate interprets as
    "no match found → suppress l=1 modulation".

    Without dustbin: softmax forces Σ_j attn_ij = 1 over real targets.
    With dustbin:    softmax distributes weight to dustbin, leaving
                     less budget for real (potentially wrong) matches.

Mathematical guarantee:
    The dustbin is a fixed (learnable but input-independent) parameter.
    It does not depend on any point's position or orientation.
    Therefore it is trivially SO(3)-invariant, preserving all guarantees.

    The scalar gate g = σ(MLP(s ‖ M)) is also invariant, so:
        v_new = g ⊙ v_old
    preserves equivariance: R(g ⊙ v) = g ⊙ (Rv).

Implementation:
    Uses ``torch.nn.functional.scaled_dot_product_attention`` (SDPA)
    which auto-dispatches to FlashAttention / Memory-Efficient backends.
    Boolean masks ensure pad points receive zero attention weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class InvariantCrossAttention(nn.Module):
    """l=0-only cross-attention with dustbin between two point clouds.

    Computes bidirectional dense attention on scalar features:
        M_src_i = Σ_j softmax(Q(s_src_i) · K([s_tgt_j; dustbin]) / √d) · V([s_tgt_j; dustbin])
        M_tgt_j = Σ_i softmax(Q(s_tgt_j) · K([s_src_i; dustbin]) / √d) · V([s_src_i; dustbin])

    The dustbin token allows points to "refuse to match" — occluded points
    route their attention weight to the dustbin, producing a near-zero
    cross-cloud message that suppresses downstream l=1 gating.

    All computations are on l=0 scalars → SO(3)-invariant by construction.

    Args:
        channels: Scalar feature dimension (must be divisible by num_heads)
        num_heads: Number of attention heads

    Example::

        >>> cross = InvariantCrossAttention(channels=32, num_heads=2)
        >>> s_src = torch.randn(4, 1024, 32)  # [B, N, C]
        >>> s_tgt = torch.randn(4, 1024, 32)
        >>> m_src, m_tgt = cross(s_src, s_tgt)
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Q, K, V projections (shared for both directions)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)

        # Output projection
        self.o_proj = nn.Linear(channels, channels, bias=True)

        # Dustbin: learnable key/value token for "refuse to match"
        # Points with no valid match route attention here → near-zero message
        # Key initialized with small noise; Value initialized to zero so that
        # attending to dustbin produces ~zero output (= "no information")
        # Shape: [1, H, 1, D] — one token per head, broadcast across batch
        self.dustbin_k = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.dustbin_v = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))

        # LayerNorm before attention (pre-norm)
        self.norm_src = nn.LayerNorm(channels)
        self.norm_tgt = nn.LayerNorm(channels)

    def forward(
        self,
        s_src: Tensor,
        s_tgt: Tensor,
        mask_src: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Bidirectional l=0 cross-attention with dustbin.

        Args:
            s_src: Source scalar features
                shape: [B, N, C], representation: SO(3) invariant l=0
            s_tgt: Target scalar features
                shape: [B, M, C], representation: SO(3) invariant l=0
            mask_src: Source valid mask [B, N] bool (True=real, False=pad)
            mask_tgt: Target valid mask [B, M] bool (True=real, False=pad)

        Returns:
            m_src: Cross-attention message for source [B, N, C]
            m_tgt: Cross-attention message for target [B, M, C]
        """
        B, N, C = s_src.shape
        M = s_tgt.shape[1]
        H = self.num_heads
        D = self.head_dim

        # Pre-norm
        s_src_n = self.norm_src(s_src)
        s_tgt_n = self.norm_tgt(s_tgt)

        # Q, K, V projections → [B, H, N/M, D]
        Q_src = self.q_proj(s_src_n).reshape(B, N, H, D).transpose(1, 2)
        K_tgt = self.k_proj(s_tgt_n).reshape(B, M, H, D).transpose(1, 2)
        V_tgt = self.v_proj(s_tgt_n).reshape(B, M, H, D).transpose(1, 2)

        Q_tgt = self.q_proj(s_tgt_n).reshape(B, M, H, D).transpose(1, 2)
        K_src = self.k_proj(s_src_n).reshape(B, N, H, D).transpose(1, 2)
        V_src = self.v_proj(s_src_n).reshape(B, N, H, D).transpose(1, 2)

        # Append dustbin token to K/V sequences
        # dustbin_k: [1, H, 1, D] → broadcast to [B, H, 1, D]
        db_k = self.dustbin_k.expand(B, H, 1, D)
        db_v = self.dustbin_v.expand(B, H, 1, D)

        K_tgt_db = torch.cat([K_tgt, db_k], dim=2)  # [B, H, M+1, D]
        V_tgt_db = torch.cat([V_tgt, db_v], dim=2)  # [B, H, M+1, D]
        K_src_db = torch.cat([K_src, db_k], dim=2)  # [B, H, N+1, D]
        V_src_db = torch.cat([V_src, db_v], dim=2)  # [B, H, N+1, D]

        # Build SDPA-compatible boolean masks
        # Append True for dustbin column (always attendable)
        attn_mask_tgt_db = None
        attn_mask_src_db = None
        if mask_tgt is not None:
            dustbin_col = torch.ones(B, 1, dtype=torch.bool, device=s_src.device)
            mask_tgt_db = torch.cat([mask_tgt, dustbin_col], dim=1)  # [B, M+1]
            attn_mask_tgt_db = mask_tgt_db[:, None, None, :]  # [B, 1, 1, M+1]
        if mask_src is not None:
            dustbin_col = torch.ones(B, 1, dtype=torch.bool, device=s_src.device)
            mask_src_db = torch.cat([mask_src, dustbin_col], dim=1)  # [B, N+1]
            attn_mask_src_db = mask_src_db[:, None, None, :]  # [B, 1, 1, N+1]

        # Source attends to Target+Dustbin
        out_src = F.scaled_dot_product_attention(
            Q_src, K_tgt_db, V_tgt_db, attn_mask=attn_mask_tgt_db,
        )  # [B, H, N, D]

        # Target attends to Source+Dustbin
        out_tgt = F.scaled_dot_product_attention(
            Q_tgt, K_src_db, V_src_db, attn_mask=attn_mask_src_db,
        )  # [B, H, M, D]

        # Reshape and project
        m_src = out_src.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        m_tgt = out_tgt.transpose(1, 2).reshape(B, M, C)  # [B, M, C]

        m_src = self.o_proj(m_src)
        m_tgt = self.o_proj(m_tgt)

        # Zero out pad positions
        if mask_src is not None:
            m_src = m_src.masked_fill(~mask_src.unsqueeze(-1), 0.0)
        if mask_tgt is not None:
            m_tgt = m_tgt.masked_fill(~mask_tgt.unsqueeze(-1), 0.0)

        return m_src, m_tgt


class ScalarGate(nn.Module):
    """Scalar gating for cross-cloud modulation of l=1 vectors.

    Takes self-scalars + cross-attention message (both l=0) and produces:
        1. Scalar residual update: Δs = MLP(s ‖ M)
        2. Vector gate: g = σ(Linear(s ‖ M)) → v_new = g ⊙ v_old

    This is the 0⊗1→1 tensor product path: the cross-cloud scalar
    information (invariant) modulates the within-cloud vector features
    (equivariant) without breaking equivariance.

    When the cross-attention message M ≈ 0 (point attended to dustbin),
    the gate g ≈ σ(W[s; 0]) which learns to suppress the vector features
    for unmatched points.

    Args:
        scalar_channels: l=0 feature dimension
        vector_channels: l=1 feature dimension
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
    ) -> None:
        super().__init__()

        # Scalar update: concat(s, M) → Δs
        self.scalar_mlp = nn.Sequential(
            nn.Linear(2 * scalar_channels, scalar_channels),
            nn.SiLU(),
            nn.Linear(scalar_channels, scalar_channels),
        )

        # Vector gate: concat(s, M) → gate ∈ [0, 1]^{C_v}
        self.vector_gate = nn.Sequential(
            nn.Linear(2 * scalar_channels, vector_channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        cross_msg: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply scalar gating with cross-cloud message.

        Args:
            scalars: Self l=0 features [B, N, C_s]
            vectors: Self l=1 features [B, N, C_v, 3]
            cross_msg: Cross-attention message [B, N, C_s] (invariant)
            mask: Valid mask [B, N] bool

        Returns:
            scalars_new: Updated scalars [B, N, C_s]
            vectors_new: Gated vectors [B, N, C_v, 3]
        """
        # Concatenate self-scalars with cross-cloud message
        combined = torch.cat([scalars, cross_msg], dim=-1)  # [B, N, 2*C_s]

        # Scalar residual
        s_delta = self.scalar_mlp(combined)  # [B, N, C_s]
        s_new = scalars + s_delta

        # Vector gating: g ∈ [0,1]^{C_v}, v_new = g ⊙ v
        g = self.vector_gate(combined)  # [B, N, C_v]
        v_new = vectors * g.unsqueeze(-1)  # [B, N, C_v, 3]

        # Zero out pad positions
        if mask is not None:
            pad = ~mask.unsqueeze(-1)  # [B, N, 1]
            s_new = s_new.masked_fill(pad, 0.0)
            v_new = v_new.masked_fill(pad.unsqueeze(-1), 0.0)

        return s_new, v_new
