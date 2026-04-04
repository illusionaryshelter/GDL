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

        # NOTE: No 1/√d_attn scaling here. Scaling is needed for
        # dot-product attention (Q·K) where variance grows with d_k.
        # Our attention is ADDITIVE (GATv2: a^T · σ(Q + K)), which
        # does not suffer from variance-grows-with-dimension.
        # The learnable temperature provides sufficient logit control.

        # Q/K projections for content-aware attention
        self.q_proj = nn.Linear(scalar_channels, d_attn, bias=False)
        self.k_proj = nn.Linear(scalar_channels, d_attn, bias=False)
        # Attention vector: projects combined Q+K through nonlinearity → scalar
        self.attn_vec = nn.Linear(d_attn, 1, bias=False)

        # Geometric position bias (invariant features → scalar bias)
        # Base 7 invariant features (l=0, l=1):
        #   0. dist              — Euclidean distance (≥0)
        #   1. scalar_ratio      — ‖s_nbr‖ / ‖s_seed‖ (relative, ~1.0)
        #   2. cos(v_seed, v_nbr) — cosine similarity, [-1, 1]
        #   3. cos(d̂_ij,  v_nbr) — direction-neighbor alignment, [-1, 1]
        #   4. cos(d̂_ij,  v_seed)— direction-seed alignment, [-1, 1]
        #   5. ‖v_nbr‖          — neighbor vector magnitude
        #   6. ‖v_seed‖         — seed vector magnitude
        # +1 if type-2 enabled:
        #   7. ‖t2_nbr‖_mean    — neighbor type-2 norm (SO(3)-invariant)
        n_inv_features = 7 + (1 if type2_channels > 0 else 0)
        self.attn_feat_norm = nn.BatchNorm1d(n_inv_features)

        self.geo_mlp = nn.Sequential(
            nn.Linear(n_inv_features, attn_hidden),
            nn.SiLU(),
            nn.Linear(attn_hidden, 1),
        )

        # Learnable temperature: logits are divided by exp(log_temp).
        # Initialized to log(1.0) = 0 (no effect). The network learns
        # to sharpen (temp < 1) or smooth (temp > 1) attention.
        self.log_temperature = nn.Parameter(torch.zeros(1))

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

            # Feature 5 & 6: vector magnitudes
            nbr_v_norm_mean = nbr_v_norms.mean(dim=-1)             # [N_out, K]
            seed_v_norm_mean = seed_v_norms.mean(dim=-1, keepdim=True).expand(-1, K)

            # Stack invariant features: [N_out, K, 7 or 8]
            feat_list = [
                dist,              # ≥0, physical scale
                scalar_ratio,      # ~1.0, relative
                cos_vi_vj_mean,    # [-1, 1], angular
                cos_d_vj_mean,     # [-1, 1], angular
                cos_d_vi_mean,     # [-1, 1], angular
                nbr_v_norm_mean,   # ≥0, magnitude
                seed_v_norm_mean,  # ≥0, magnitude
            ]

            # Type-2 invariant feature
            if type2 is not None and self.type2_channels > 0:
                type2_f = type2.to(compute_dtype)
                nbr_t2 = type2_f[safe_idx]  # [N_out, K, C_t2, 5]
                # ||t2||: invariant per-channel norm, averaged
                nbr_t2_norm_mean = nbr_t2.pow(2).sum(dim=-1).sqrt().mean(dim=-1)  # [N_out, K]
                feat_list.append(nbr_t2_norm_mean)

            attn_input = torch.stack(feat_list, dim=-1)

            # ── Part 2: Geometric position bias ──
            # Per-feature BatchNorm: [N_out, K, 7] → [N_out*K, 7] → normalize → reshape
            NK = N_out * K
            attn_flat = attn_input.reshape(NK, -1)       # [NK, 7]
            attn_normed = self.attn_feat_norm(attn_flat)  # [NK, 7]
            attn_normed = attn_normed.reshape(N_out, K, -1)  # [N_out, K, 7]

            # geo_bias: [N_out, K]
            geo_bias = self.geo_mlp(attn_normed).squeeze(-1)  # [N_out, K]

            # ── Part 1: Content attention (GATv2-style dynamic) ──
            # Q from seed, K from neighbor — both l=0 scalars → invariant
            Q = self.q_proj(seed_scalars)              # [N_out, d_attn]
            K_feat = self.k_proj(nbr_scalars)          # [N_out, K, d_attn]

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

            # Type-2 aggregation: [N_out, K, C_t2, 5] weighted sum → [N_out, C_t2, 5]
            t2_out = None
            if type2 is not None and self.type2_channels > 0:
                type2_f = type2.to(compute_dtype)
                nbr_t2_all = type2_f[safe_idx]  # [N_out, K, C_t2, 5]
                t2_out = (w.unsqueeze(-1).unsqueeze(-1) * nbr_t2_all).sum(dim=1)

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
