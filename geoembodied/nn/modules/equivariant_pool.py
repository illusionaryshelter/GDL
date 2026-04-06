# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Equivariant pooling (downsampling) for multi-scale SE(3) networks.

Implements FPS-based spatial downsampling with attention-weighted
invariant aggregation. The attention weights are computed ONLY from
SO(3)-invariant quantities (distances, scalar norms) — guaranteeing
that l≥1 vector features remain exactly equivariant after pooling.

Architecture:
    1. FPS selects K seed points from N input points
    2. KNN assigns each seed its nearest neighbors in the dense cloud
    3. Attention MLP computes weights from invariant features:
       [distance, scalar_norm, seed_scalar_norm] → weight
    4. Weighted mean aggregates scalar and vector features

CRITICAL: MaxPool is ABSOLUTELY FORBIDDEN for l≥1 features.
    max(g·v1, g·v2) ≠ g·max(v1, v2) — equivariance is destroyed.

Usage:
    pool = EquivariantPool(scalar_channels=64, vector_channels=16, ratio=0.25)
    seed_pos, s_out, v_out, ptr_out, fps_idx = pool(
        pos, scalars, vectors, ptr
    )
    # With type-2:
    pool_t2 = EquivariantPool(scalar_channels=64, vector_channels=16,
                               type2_channels=4, ratio=0.25)
    seed_pos, s_out, v_out, t2_out, ptr_out, fps_idx = pool_t2(
        pos, scalars, vectors, ptr, type2=type2
    )
"""

from __future__ import annotations



import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from typing import Optional, Tuple

# FPS dispatch now goes through geoembodied.csrc (CUDA kernel or fallback)
from geoembodied.functional.knn import knn


class NeighborNorm(nn.Module):
    """Per-seed-per-feature normalization for pool attention invariants.

    For each seed point i and each invariant feature f, normalize across
    the K neighbors:
        x_normed[i, k, f] = (x[i, k, f] - mean_f(i)) / (std_f(i) + eps)
    where mean_f and std_f are computed over the K dimension for seed i.

    Why not BatchNorm?
        BatchNorm normalizes each feature across ALL (seed, neighbor) pairs
        globally, destroying intra-seed variance — exactly the signal that
        attention needs to differentiate neighbors.

    Why not per-seed joint LayerNorm (across K*F)?
        Physical quantities like distance (≥0) and cosine similarity ([-1,1])
        have different scales and semantics. Joint normalization mixes them,
        suppressing features with smaller natural variance.

    This module normalizes each feature INDEPENDENTLY across K neighbors,
    maximizing each feature's discriminative power while remaining
    semantically correct and mode-invariant (no train/eval split).

    Args:
        n_features: Number of invariant features (e.g. 7 or 8)
        eps: Small constant for numerical stability

    Input:
        x: [N_out, K, F] — invariant features for all seed-neighbor pairs
    Output:
        [N_out, K, F] — per-seed-per-feature normalized, with affine transform
    """

    def __init__(self, n_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        # Learnable affine parameters: per-feature scale and shift.
        # Initialized to identity (weight=1, bias=0) so the module
        # starts as pure normalization.
        self.weight = nn.Parameter(torch.ones(n_features))   # type: ignore[arg-type]
        self.bias = nn.Parameter(torch.zeros(n_features))    # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        """Normalize invariant features per-seed, per-feature.

        Args:
            x: shape [N_out, K, F] — invariant features for pool attention.
                Each (seed, neighbor) pair has F invariant features.

        Returns:
            Normalized tensor of same shape [N_out, K, F].
        """
        # x: [N_out, K, F]
        # Compute per-seed, per-feature statistics across K neighbors
        mean = x.mean(dim=1, keepdim=True)               # [N_out, 1, F]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [N_out, 1, F]
        x_normed = (x - mean) / (var + self.eps).sqrt()   # [N_out, K, F]
        # Apply learnable affine (broadcast across N_out and K)
        return x_normed * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, eps={self.eps}"


class EquivariantPool(nn.Module):
    """FPS + attention-weighted mean pooling for equivariant features.

    Downsamples a point cloud by ratio, aggregating features from
    KNN neighborhoods using learned SO(3)-invariant attention weights.

    Args:
        scalar_channels: Number of scalar (l=0) feature channels
        vector_channels: Number of vector (l=1) feature channels
        ratio: Downsampling ratio (0 < ratio < 1), e.g. 0.25 = keep 25%
        k_neighbors: Number of KNN neighbors for aggregation
        attn_hidden: Hidden dim for attention MLP
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        type2_channels: int = 0,
        ratio: float = 0.25,
        k_neighbors: int = 16,
        attn_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.type2_channels = type2_channels
        self.ratio = ratio
        self.k_neighbors = k_neighbors

        # ═══════════════════════════════════════════════════════════════
        # Dynamic Content-Aware Attention (GATv2-style)
        # ═══════════════════════════════════════════════════════════════
        #
        # Two-part attention score:
        #   attn = content_score(seed_s, nbr_s) + geo_bias(invariants)
        #
        # Part 1 — Content attention (GATv2 pattern):
        #   W_q projects seed scalars, W_k projects neighbor scalars,
        #   then: score = a^T * SiLU(W_q * s_seed + W_k * s_nbr)
        #   This is DYNAMIC: the ranking of neighbors depends on the
        #   query (seed) node features, resolving the static attention
        #   problem identified by Brody et al. (ICLR 2022).
        #
        # Part 2 — Geometric position bias:
        #   Same 7-8 invariant features as before, but now as additive
        #   bias rather than the sole attention input.
        #
        # All inputs are l=0 (scalars/invariants) → SO(3) equivariance
        # is preserved for the weighted aggregation of l≥1 features.

        # Content attention dimension (bottleneck to save params)
        d_attn = max(scalar_channels // 4, 8)
        self.d_attn = d_attn

        # Q/K projections for content-aware attention
        self.q_proj = nn.Linear(scalar_channels, d_attn, bias=False)
        self.k_proj = nn.Linear(scalar_channels, d_attn, bias=False)
        # Attention vector: projects combined Q+K through nonlinearity → scalar
        self.attn_vec = nn.Linear(d_attn, 1, bias=False)

        # QK-Norm: L2-normalize Q and K before adding, then scale by
        # a learnable parameter. Without this, |Q| >> K_std (20x on
        # real data) → SiLU(Q+K) ≈ SiLU(Q) → no neighbor discrimination.
        # After QK-Norm: |Q_normed| = |K_normed| = qk_scale, balanced.
        # Initialized to √d_attn (same as standard transformer convention).
        self.qk_scale = nn.Parameter(torch.full((1,), d_attn ** 0.5))

        # Geometric position bias (invariant features → scalar bias)
        # Base 6 invariant features (l=0, l=1):
        #   0. dist              — Euclidean distance (≥0)
        #   1. scalar_ratio      — ‖s_nbr‖ / ‖s_seed‖ (relative, ~1.0)
        #   2. cos(v_seed, v_nbr) — cosine similarity, [-1, 1]
        #   3. cos(d̂_ij,  v_nbr) — direction-neighbor alignment, [-1, 1]
        #   4. cos(d̂_ij,  v_seed)— direction-seed alignment, [-1, 1]
        #   5. ‖v_nbr‖ - ‖v_seed‖ — relative vector magnitude diff
        # +1 if type-2 enabled:
        #   6. ‖t2_nbr‖_mean    — neighbor type-2 norm (SO(3)-invariant)
        #
        # NOTE: ‖v_seed‖ alone was REMOVED — it's the seed's own value,
        # identical for all K neighbors → zero intra-seed variance → dead
        # channel after NeighborNorm. Replaced by the relative difference
        # ‖v_nbr‖ - ‖v_seed‖ which HAS per-neighbor variance.
        n_inv_features = 6 + (1 if type2_channels > 0 else 0)
        self.attn_feat_norm = NeighborNorm(n_inv_features)

        self.geo_mlp = nn.Sequential(
            nn.Linear(n_inv_features, attn_hidden),
            nn.SiLU(),
            nn.Linear(attn_hidden, 1),
        )

        # Learnable temperature: logits are divided by exp(log_temp).
        # Initialized to -0.5 → temp ≈ 0.607 for sharper initial attention.
        # Rationale: autopsy shows geo_bias range ≈ 2.5-3.0; dividing by
        # 0.6 gives effective_range ≈ 4-5, closer to the ≥5.5 needed for
        # sharp softmax over K=16. The network can still learn to adjust.
        self.log_temperature = nn.Parameter(torch.full((1,), -0.5))

    def forward(
        self,
        pos: Tensor,
        scalars: Tensor,
        vectors: Tensor,
        ptr: Tensor,
        type2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        """Downsample point cloud with equivariant feature aggregation.

        Args:
            pos: Packed point positions
                shape: [N_total, 3]
            scalars: Scalar features
                shape: [N_total, C_s]
            vectors: Vector features
                shape: [N_total, C_v, 3], representation: SO(3) type-1
            ptr: CSR batch offsets
                shape: [B+1], int64
            type2: Optional type-2 features
                shape: [N_total, C_t2, 5]

        Returns:
            fps_idx: FPS selected indices (global, into original)
                shape: [N_out], int64
        """
        # Force ≥FP32 for geometric computations (norm, sqrt, cosine).
        # Under AMP, inputs may be FP16 but these ops need full precision.
        # CRITICAL: must also disable autocast, otherwise attn_mlp's
        # nn.Linear layers get silently demoted to FP16 by autocast,
        # causing backward gradient overflow → GradScaler instability.
        # NOTE: preserve FP64 if present (equivariance tests need it).
        input_dtype = scalars.dtype
        device = pos.device
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        compute_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype

        with autocast(device_type, enabled=False):
            pos = pos.to(compute_dtype)
            scalars = scalars.to(compute_dtype)
            vectors = vectors.to(compute_dtype)

            B = ptr.shape[0] - 1

            # ── 1. FPS: select seed points per batch ──
            # Compute per-batch sample count entirely on GPU (no CPU sync)
            sizes = ptr[1:] - ptr[:-1]  # [B], on device
            k_per_batch = (sizes.float() * self.ratio).clamp(min=1).long()  # [B]

            # Batched FPS: CUDA kernel (zero sync in loop) or PyTorch fallback
            from geoembodied.csrc import fps_available, fps_cuda, _fps_iterative_fallback
            if fps_available() and pos.is_cuda:
                fps_idx = fps_cuda(pos, ptr, k_per_batch)
            else:
                fps_idx = _fps_iterative_fallback(pos, ptr, k_per_batch)

            seed_pos = pos[fps_idx]  # [N_out, 3]

            # Build ptr_out for downsampled batch (stays on GPU)
            sizes_out = k_per_batch  # [B]
            ptr_out = torch.zeros(B + 1, dtype=torch.int64, device=device)
            ptr_out[1:] = sizes_out.cumsum(0)

            N_out = fps_idx.shape[0]

            # ── 2. KNN: find neighbors of each seed in original cloud ──
            knn_idx, knn_dists = knn(
                seed_pos, pos, ptr_out, ptr, self.k_neighbors
            )  # [N_out, K], [N_out, K]

            # ── 3. Compute attention weights from SO(3)-INVARIANT features ──
            valid_mask = knn_idx >= 0  # [N_out, K]
            safe_idx = knn_idx.clamp(min=0)  # safe indexing for gather
            K = self.k_neighbors
            eps = 1e-8

            # ── Scalar features ──
            nbr_scalars = scalars[safe_idx]                         # [N_out, K, C_s]
            seed_scalars = scalars[fps_idx]                         # [N_out, C_s]

            nbr_s_norm = nbr_scalars.norm(dim=-1)                  # [N_out, K]
            seed_s_norm = seed_scalars.norm(dim=-1, keepdim=True)   # [N_out, 1]

            # Feature 0: distance
            dist = knn_dists.clamp(min=eps).sqrt()                  # [N_out, K]

            # Feature 1: scalar norm ratio (relative, ~1.0, invariant)
            scalar_ratio = nbr_s_norm / seed_s_norm.clamp(min=eps)  # [N_out, K]

            # ── Vector features ──
            nbr_vectors = vectors[safe_idx]                         # [N_out, K, C_v, 3]
            seed_vectors = vectors[fps_idx]                         # [N_out, C_v, 3]
            seed_vectors_exp = seed_vectors.unsqueeze(1)            # [N_out, 1, C_v, 3]

            # Per-channel norms for cosine similarity
            nbr_v_norms = nbr_vectors.norm(dim=-1)                 # [N_out, K, C_v]
            seed_v_norms = seed_vectors.norm(dim=-1)                # [N_out, C_v]

            # Feature 2: cos(v_seed, v_nbr) — angular alignment, [-1, 1]
            vi_vj_dot = (seed_vectors_exp * nbr_vectors).sum(dim=-1)  # [N_out, K, C_v]
            cos_vi_vj = vi_vj_dot / (
                seed_v_norms.unsqueeze(1) * nbr_v_norms + eps
            )                                                       # [N_out, K, C_v]
            cos_vi_vj_mean = cos_vi_vj.mean(dim=-1)                # [N_out, K]

            # Feature 3 & 4: direction-vector cosine similarities
            rel_pos = pos[safe_idx] - seed_pos.unsqueeze(1)         # [N_out, K, 3]
            rel_dist = rel_pos.norm(dim=-1, keepdim=True).clamp(min=eps)
            d_hat = rel_pos / rel_dist                              # [N_out, K, 3] unit

            # cos(d̂_ij, v_nbr)
            d_vj_dot = (d_hat.unsqueeze(2) * nbr_vectors).sum(dim=-1)  # [N_out, K, C_v]
            cos_d_vj = d_vj_dot / (nbr_v_norms + eps)              # [N_out, K, C_v]
            cos_d_vj_mean = cos_d_vj.mean(dim=-1)                  # [N_out, K]

            # cos(d̂_ij, v_seed)
            d_vi_dot = (d_hat.unsqueeze(2) * seed_vectors_exp).sum(dim=-1)  # [N_out, K, C_v]
            cos_d_vi = d_vi_dot / (seed_v_norms.unsqueeze(1) + eps)  # [N_out, K, C_v]
            cos_d_vi_mean = cos_d_vi.mean(dim=-1)                  # [N_out, K]

            # Feature 5: relative vector magnitude difference
            # ‖v_nbr‖ − ‖v_seed‖: has intra-seed variance (unlike the
            # old ‖v_seed‖ which was constant → dead after NeighborNorm).
            nbr_v_norm_mean = nbr_v_norms.mean(dim=-1)             # [N_out, K]
            seed_v_norm_mean = seed_v_norms.mean(dim=-1, keepdim=True)  # [N_out, 1]

            # Stack invariant features: [N_out, K, 6 or 7]
            feat_list = [
                dist,              # ≥0, physical scale
                scalar_ratio,      # ~1.0, relative
                cos_vi_vj_mean,    # [-1, 1], angular
                cos_d_vj_mean,     # [-1, 1], angular
                cos_d_vi_mean,     # [-1, 1], angular
                nbr_v_norm_mean - seed_v_norm_mean,  # ℝ, relative magnitude
            ]

            # Type-2 invariant feature
            if type2 is not None and self.type2_channels > 0:
                type2_f = type2.to(compute_dtype)
                nbr_t2 = type2_f[safe_idx]  # [N_out, K, C_t2, 5]
                # ||t2||: invariant per-channel norm, averaged.
                # clamp before sqrt: backward of sqrt(0) = 0.5/sqrt(0) = NaN (Rule 4)
                nbr_t2_norm_mean = nbr_t2.pow(2).sum(dim=-1).clamp(min=1e-12).sqrt().mean(dim=-1)  # [N_out, K]
                feat_list.append(nbr_t2_norm_mean)

            attn_input = torch.stack(feat_list, dim=-1)

            # ── Part 2: Geometric position bias ──
            # NeighborNorm: normalize each invariant feature INDEPENDENTLY
            # across K neighbors within each seed. This preserves the
            # intra-seed ranking that drives attention to discriminate
            # neighbors, unlike BatchNorm which globally flattens variance.
            attn_normed = self.attn_feat_norm(attn_input)  # [N_out, K, F]

            # geo_bias: [N_out, K]
            geo_bias = self.geo_mlp(attn_normed).squeeze(-1)  # [N_out, K]

            # ── Part 1: Content attention (GATv2-style dynamic) ──
            # Q from seed, K from neighbor — both l=0 scalars → invariant
            Q = self.q_proj(seed_scalars)              # [N_out, d_attn]
            K_feat = self.k_proj(nbr_scalars)          # [N_out, K, d_attn]

            # QK-Norm: normalize Q and K to unit vectors, then scale.
            # Without this, |Q|/K_std ≈ 20x → SiLU(Q+K) has no K-dependent
            # variation → content attention provides zero neighbor discrimination.
            # After norm: both have magnitude ≈ qk_scale, balanced contribution.
            Q = torch.nn.functional.normalize(Q, dim=-1) * self.qk_scale
            K_feat = torch.nn.functional.normalize(K_feat, dim=-1) * self.qk_scale

            # GATv2: a^T * σ(Q_expand + K_feat) — dynamic because the
            # ranking of neighbors changes based on the seed's features
            combined = torch.nn.functional.silu(
                Q.unsqueeze(1) + K_feat                # [N_out, K, d_attn]
            )
            content_score = self.attn_vec(combined).squeeze(-1)  # [N_out, K]

            # ── Combined attention logits ──
            attn_logits = content_score + geo_bias

            # Learnable temperature
            temperature = self.log_temperature.exp().clamp(min=0.01)
            attn_logits = attn_logits / temperature

            # Mask invalid neighbors (padding) to -inf
            attn_logits = attn_logits.masked_fill(~valid_mask, float('-inf'))

            # Softmax over neighbors
            attn_weights = torch.softmax(attn_logits, dim=-1)  # [N_out, K]

            # Handle all-invalid rows (single isolated points)
            nan_mask = attn_weights.isnan()
            if nan_mask.any():
                attn_weights = attn_weights.masked_fill(nan_mask, 0.0)

            # Store for diagnostic extraction.
            # MUST use .clone() — .detach() alone shares storage with the
            # computation graph tensor, preventing the autograd graph from
            # being freed on backward(). .clone() creates independent storage.
            self._last_attn_weights = attn_weights.detach().clone()  # [N_out, K]
            self._last_valid_mask = valid_mask.detach().clone()       # [N_out, K]

            # ── 4. Weighted aggregation ──
            w = attn_weights  # [N_out, K]

            # Scalar aggregation: [N_out, K, C_s] weighted sum → [N_out, C_s]
            s_out = (w.unsqueeze(-1) * nbr_scalars).sum(dim=1)

            # Vector aggregation: [N_out, K, C_v, 3] weighted sum → [N_out, C_v, 3]
            nbr_vectors = vectors[safe_idx]  # [N_out, K, C_v, 3]
            v_out = (w.unsqueeze(-1).unsqueeze(-1) * nbr_vectors).sum(dim=1)

            # ── Norm-preserving rescaling for vectors (l=1) ──
            # Weighted average of multi-dim features causes directional
            # cancellation (50–73% norm loss). Fix: rescale so ||v_out||
            # matches the weighted average of input norms (no cancellation
            # since norms are positive scalars).
            #
            # Equivariance: target and actual are SO(3)-invariant (norms).
            #   scale = target/actual is a per-channel scalar.
            #   ρ(g)(α·v) = α·ρ(g)v  ✓
            #
            # v_out: [N_out, C_v, 3], representation: SO(3) type-1
            _eps = 1e-12
            nbr_v_norms = (nbr_vectors ** 2).sum(-1).clamp(min=_eps).sqrt()
            target_v = (w.unsqueeze(-1) * nbr_v_norms).sum(dim=1)  # [N_out, C_v]
            actual_v = (v_out ** 2).sum(-1).clamp(min=_eps).sqrt()  # [N_out, C_v]
            scale_v = torch.where(
                target_v > 1e-7,
                target_v / actual_v.clamp(min=1e-8),
                torch.ones_like(target_v),
            )
            if not getattr(self, '_disable_norm_rescale', False):
                v_out = v_out * scale_v.unsqueeze(-1)

            # Type-2 aggregation: [N_out, K, C_t2, 5] weighted sum → [N_out, C_t2, 5]
            t2_out = None
            if type2 is not None and self.type2_channels > 0:
                type2_f = type2.to(compute_dtype)
                nbr_t2_all = type2_f[safe_idx]  # [N_out, K, C_t2, 5]
                t2_out = (w.unsqueeze(-1).unsqueeze(-1) * nbr_t2_all).sum(dim=1)

                # ── Norm-preserving rescaling for type-2 (l=2) ──
                # t2_out: [N_out, C_t2, 5], representation: SO(3) type-2
                nbr_t2_norms = (nbr_t2_all ** 2).sum(-1).clamp(min=_eps).sqrt()
                target_t2 = (w.unsqueeze(-1) * nbr_t2_norms).sum(dim=1)
                actual_t2 = (t2_out ** 2).sum(-1).clamp(min=_eps).sqrt()
                scale_t2 = torch.where(
                    target_t2 > 1e-7,
                    target_t2 / actual_t2.clamp(min=1e-8),
                    torch.ones_like(target_t2),
                )
                if not getattr(self, '_disable_norm_rescale', False):
                    t2_out = t2_out * scale_t2.unsqueeze(-1)

            # Zero out features from invalid neighbors
            all_invalid = ~valid_mask.any(dim=1)  # [N_out]
            if all_invalid.any():
                s_out[all_invalid] = 0.0
                v_out[all_invalid] = 0.0
                if t2_out is not None:
                    t2_out[all_invalid] = 0.0

        if t2_out is not None:
            return seed_pos, s_out.to(input_dtype), v_out.to(input_dtype), t2_out.to(input_dtype), ptr_out, fps_idx
        return seed_pos, s_out.to(input_dtype), v_out.to(input_dtype), ptr_out, fps_idx

    def extra_repr(self) -> str:
        return (
            f"scalar={self.scalar_channels}, vector={self.vector_channels}, "
            f"ratio={self.ratio}, k={self.k_neighbors}"
        )
