# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""MultiScaleSE3Net — U-Net style equivariant backbone for point clouds.

Implements a hierarchical encoder-decoder with skip connections,
operating at multiple spatial scales via FPS downsampling and
KNN-based upsampling.

Architecture::

    Encoder:  pos₀ → [SE3Conv × L] → Pool → [SE3Conv × L] → Pool → ...
    Decoder:  ... → Interp + Skip → [SE3Conv × L] → Interp + Skip → ...

Key design decisions:
    - SSG (Single-Scale Grouping): TP+SH receptive field is sufficient
    - Adaptive radius: auto-computed from avg nearest-neighbor distance
    - KNN fallback in SpatialGraph for extreme sparsity at bottleneck
    - Vector features use attention-weighted mean pool (NEVER max pool)
    - Skip connections via channel concatenation (not addition)
    - Feature projection after skip-concat to restore channel count

Usage::

    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=64, hidden_vector=16,
        num_stages=3, layers_per_stage=2,
        pool_ratio=0.25,
    )
    s_out, v_out, ptr_out = model(pos, ptr)
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from geoembodied.nn.modules.se3_block import SE3NetBlock
from geoembodied.nn.modules.spatial_graph import SpatialGraph
from geoembodied.nn.modules.equivariant_pool import EquivariantPool
from geoembodied.nn.modules.equivariant_interp import EquivariantInterpolate
from geoembodied.nn.modules.equivariant_norm import EquivariantLayerNorm
from geoembodied.nn.modules.geometric_self_attention import InvariantSelfAttention
from geoembodied.functional.knn import knn_self


class _EncoderStage(nn.Module):
    """One encoder stage: build graph → [SE3Conv blocks × L].

    Builds SpatialGraph with adaptive radius, then runs L SE3NetBlocks.

    Args:
        channels_scalar: Scalar feature channels
        channels_vector: Vector feature channels
        num_layers: Number of SE3NetBlock layers
        radius_multiplier: Multiplier for adaptive radius (avg_nn_dist * mult)
        max_num_neighbors: Max neighbors per node in radius graph
        knn_fallback_k: Min neighbors guaranteed by KNN fallback
    """

    def __init__(
        self,
        channels_scalar: int,
        channels_vector: int,
        channels_type2: int = 0,
        num_layers: int = 2,
        radius_multiplier: float = 4.0,
        max_num_neighbors: int = 32,
        knn_fallback_k: int = 3,
        gate_mode: str = 'scalar',
        use_self_tp: bool = False,
    ) -> None:
        super().__init__()
        self.channels_scalar = channels_scalar
        self.channels_vector = channels_vector
        self.channels_type2 = channels_type2
        self.radius_multiplier = radius_multiplier
        self.max_num_neighbors = max_num_neighbors
        self.knn_fallback_k = knn_fallback_k

        self.blocks = nn.ModuleList([
            SE3NetBlock(
                channels_scalar=channels_scalar,
                channels_vector=channels_vector,
                channels_type2=channels_type2,
                radius=1.0,  # placeholder, will use adaptive
                max_num_neighbors=max_num_neighbors,
                gate_mode=gate_mode,
                use_self_tp=use_self_tp,
            )
            for _ in range(num_layers)
        ])

    def _estimate_adaptive_radius(
        self, pos: Tensor, ptr: Tensor
    ) -> float:
        """Estimate radius from average nearest-neighbor distance.

        Uses KNN(k=2) where neighbor 0 is self → neighbor 1 is nearest.
        Radius = avg(nn_dist) * multiplier (typically 4.0).

        NOTE: This involves 1 GPU→CPU sync (.item()). To amortize,
        the caller should cache the result when possible.

        Args:
            pos: [N, 3] packed positions
            ptr: [B+1] CSR offsets

        Returns:
            Adaptive radius (float, on CPU)
        """
        with torch.no_grad():
            _, dists = knn_self(pos, ptr, k=2)
            # dists[:, 0] is self (≈0), dists[:, 1] is nearest neighbor
            nn_dists = dists[:, 1]
            # Compute mean without sync-heavy .any() check:
            # If all invalid, clamp makes mean ≈ sqrt(1e-8) → radius = eps*mult ≈ 0.04
            avg_nn = nn_dists.clamp(max=1e20).clamp(min=1e-8).sqrt().mean().item()
        return avg_nn * self.radius_multiplier

    def _build_hybrid_graph(
        self, pos: Tensor, radius: float,
        ptr: Tensor, batch: Optional[Tensor] = None,
    ) -> SpatialGraph:
        """Build radius graph with KNN fallback for isolated nodes.

        Layer 1: Radius graph (continuous, differentiable)
        Layer 2: For nodes with < knn_fallback_k neighbors, add KNN edges

        CRITICAL: KNN fallback edges are BIDIRECTIONAL (symmetrized)
        to prevent "feature black holes" (Pitfall 5).
        KNN uses ptr for batch isolation — no cross-graph connections.

        GPU SYNC BUDGET: This method uses 0 GPU→CPU syncs.
        All control flow uses tensor ops or CPU-side metadata.

        Args:
            pos: [N, 3]
            radius: Search radius
            ptr: [B+1] CSR offsets — REQUIRED for batch-safe KNN
            batch: [N] optional batch vector for radius_graph

        Returns:
            SpatialGraph with guaranteed min connectivity
        """
        from geoembodied.functional.radius_graph import radius_graph

        N = pos.shape[0]
        B = ptr.shape[0] - 1  # CPU metadata, no sync
        device = pos.device

        # Layer 1: Radius graph
        # Derive B from ptr (CPU metadata, no sync). Check equal sizes
        # on CPU to decide whether to use batched fast path.
        counts_cpu = (ptr[1:] - ptr[:-1]).cpu()  # tiny tensor, ~128B
        B = counts_cpu.numel()
        equal_size = (counts_cpu[0] == counts_cpu).all().item() if B > 1 else True
        num_batch = B if equal_size else None  # skip sync if equal

        row, col = radius_graph(
            pos, radius,
            max_num_neighbors=self.max_num_neighbors,
            batch=batch,
            num_batch_elements=num_batch,
        )

        # Count neighbors per node
        if row.numel() > 0:
            deg = torch.zeros(N, dtype=torch.long, device=device)
            deg.scatter_add_(0, col, torch.ones_like(col))
        else:
            deg = torch.zeros(N, dtype=torch.long, device=device)

        # Layer 2: KNN fallback for isolated nodes (FULLY VECTORIZED)
        # NO .any() sync — always run vectorized path.
        # If no isolated nodes, isolated_idx is empty → all ops are no-ops.
        isolated = deg < self.knn_fallback_k
        isolated_idx = torch.where(isolated)[0]  # [N_iso]

        if isolated_idx.numel() > 0:
            # Use ACTUAL ptr for batch-safe KNN (no cross-graph edges!)
            knn_idx, _ = knn_self(pos, ptr, k=self.knn_fallback_k + 1)
            # knn_idx[:, 0] is self, [:, 1:] are actual neighbors

            # Select isolated nodes' neighbor indices: [N_iso, K]
            iso_nbrs = knn_idx[isolated_idx, 1:]  # [N_iso, K], exclude self

            # Valid mask: neighbor index >= 0
            valid = iso_nbrs >= 0  # [N_iso, K]

            # Expand isolated_idx to match neighbor shape
            iso_node = isolated_idx.unsqueeze(1).expand_as(iso_nbrs)  # [N_iso, K]

            # Flatten and filter valid entries only
            src_flat = iso_node[valid]    # [E_fallback]
            dst_flat = iso_nbrs[valid]    # [E_fallback]

            if src_flat.numel() > 0:
                # Bidirectional: stack both directions
                extra_r = torch.cat([src_flat, dst_flat])
                extra_c = torch.cat([dst_flat, src_flat])

                row = torch.cat([row, extra_r])
                col = torch.cat([col, extra_c])

                # Deduplicate edges via hash
                edge_hash = row * N + col
                unique_edges = edge_hash.unique()
                row = unique_edges // N
                col = unique_edges % N

        return SpatialGraph.from_edge_index(row, col, pos, N)

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        pos: Tensor,
        ptr: Tensor,
        batch: Optional[Tensor] = None,
        graph: Optional[SpatialGraph] = None,
        type2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        """Run SE3Conv blocks at this scale.

        Args:
            scalars: [N, C_s]
            vectors: [N, C_v, 3]
            pos: [N, 3]
            ptr: [B+1]
            batch: [N] optional
            graph: Pre-built SpatialGraph to reuse
            type2: [N, C_t2, 5] optional type-2 tensor features

        Returns:
            Without type-2: (s_out, v_out, graph)
            With type-2: (s_out, v_out, t2_out, graph)
        """
        if graph is None:
            with torch.no_grad():
                radius = self._estimate_adaptive_radius(pos, ptr)
                graph = self._build_hybrid_graph(pos, radius, ptr=ptr, batch=batch)

        has_t2 = type2 is not None and self.channels_type2 > 0
        s, v = scalars, vectors
        t2 = type2

        for block in self.blocks:
            if has_t2:
                if self.training:
                    def _run_block_t2(_s, _v, _t2, _block=block, _g=graph):
                        return _block(_s, _v, _g, type2=_t2)
                    s, v, t2 = torch.utils.checkpoint.checkpoint(
                        _run_block_t2, s, v, t2, use_reentrant=False,
                    )
                else:
                    s, v, t2 = block(s, v, graph, type2=t2)
            else:
                if self.training:
                    def _run_block(_s, _v, _block=block, _g=graph):
                        return _block(_s, _v, _g)
                    s, v = torch.utils.checkpoint.checkpoint(
                        _run_block, s, v, use_reentrant=False,
                    )
                else:
                    s, v = block(s, v, graph)

        if has_t2:
            return s, v, t2, graph
        return s, v, graph


class MultiScaleSE3Net(nn.Module):
    """U-Net style multi-scale SE(3)-equivariant backbone.

    Encoder stages progressively downsample via FPS + Pool.
    Decoder stages upsample via KNN interpolation + skip connections.
    Each stage runs L SE3Conv blocks at its resolution.

    Args:
        in_channels: Raw input feature dimension (default 1)
        hidden_scalar: Base scalar channels
        hidden_vector: Base vector channels
        num_stages: Number of encoder/decoder stages (depth of U-Net)
        layers_per_stage: SE3Conv blocks per stage
        pool_ratio: FPS downsampling ratio per stage
        pool_k: KNN neighbors for pool aggregation
        interp_k: KNN neighbors for interpolation
        radius_multiplier: Adaptive radius = avg_nn * multiplier
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_scalar: int = 64,
        hidden_vector: int = 16,
        hidden_type2: int = 0,
        num_stages: int = 3,
        layers_per_stage: int = 2,
        pool_ratio: float = 0.25,
        pool_k: int = 16,
        interp_k: int = 3,
        radius_multiplier: float = 4.0,
        gate_mode: str = 'scalar',
        use_self_tp: bool = False,
        use_bottleneck_attn: bool = False,
        attn_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.hidden_type2 = hidden_type2
        self.num_stages = num_stages

        C_s = hidden_scalar
        C_v = hidden_vector
        C_t2 = hidden_type2

        # Input embedding
        self.embed_scalar = nn.Linear(in_channels, C_s)

        # ── Encoder ──
        self.encoder_stages = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i in range(num_stages):
            self.encoder_stages.append(_EncoderStage(
                channels_scalar=C_s,
                channels_vector=C_v,
                channels_type2=C_t2,
                num_layers=layers_per_stage,
                radius_multiplier=radius_multiplier,
                gate_mode=gate_mode,
                use_self_tp=use_self_tp,
            ))
            if i < num_stages - 1:
                self.pool_layers.append(EquivariantPool(
                    scalar_channels=C_s,
                    vector_channels=C_v,
                    type2_channels=C_t2,
                    ratio=pool_ratio,
                    k_neighbors=pool_k,
                ))

        # ── Decoder ──
        self.decoder_stages = nn.ModuleList()
        self.interp_layers = nn.ModuleList()
        self.skip_proj_s = nn.ModuleList()
        self.skip_proj_v = nn.ModuleList()
        self.skip_proj_t2 = nn.ModuleList()
        self.skip_norms = nn.ModuleList()

        for i in range(num_stages - 1):
            self.interp_layers.append(EquivariantInterpolate(k_neighbors=interp_k))

            self.skip_proj_s.append(nn.Linear(C_s * 2, C_s))
            self.skip_proj_v.append(nn.Linear(C_v * 2, C_v, bias=False))

            # Type-2 skip projection
            if C_t2 > 0:
                self.skip_proj_t2.append(nn.Linear(C_t2 * 2, C_t2, bias=False))
            else:
                self.skip_proj_t2.append(None)

            self.skip_norms.append(EquivariantLayerNorm(
                num_scalars=C_s, num_vectors=C_v, num_type2=C_t2,
            ))

            self.decoder_stages.append(_EncoderStage(
                channels_scalar=C_s,
                channels_vector=C_v,
                channels_type2=C_t2,
                num_layers=layers_per_stage,
                radius_multiplier=radius_multiplier,
                gate_mode=gate_mode,
                use_self_tp=use_self_tp,
            ))

        # ── Final Norm ──
        self.final_norm = EquivariantLayerNorm(
            num_scalars=C_s, num_vectors=C_v, num_type2=C_t2,
        )

        # ── Bottleneck attention (optional, scalar-only) ──
        self.use_bottleneck_attn = use_bottleneck_attn
        if use_bottleneck_attn:
            n_heads = min(attn_num_heads, C_s)
            while C_s % n_heads != 0:
                n_heads -= 1
            self.bottleneck_attn = InvariantSelfAttention(
                channels=C_s,
                num_heads=n_heads,
                sigma_d=0.1,
                n_freqs=8,
                ffn_ratio=2,
            )

    def forward(
        self,
        pos: Tensor,
        ptr: Tensor,
        features: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        v_init: Optional[Tensor] = None,
        return_encoder_features: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Multi-scale feature extraction.

        Args:
            pos: [N_total, 3]
            ptr: [B+1], int64
            features: Optional [N_total, in_channels]
            batch: Optional [N_total], int64
            v_init: Optional [N_total, hidden_vector, 3]
            return_encoder_features: If True, return per-stage encoder features

        Returns:
            Without type-2: (s, v, ptr) or (s, v, ptr, enc_s, enc_ptr)
            With type-2: (s, v, t2, ptr) or (s, v, t2, ptr, enc_s, enc_ptr)
        """
        N = pos.shape[0]
        C_s = self.hidden_scalar
        C_v = self.hidden_vector
        C_t2 = self.hidden_type2
        has_t2 = C_t2 > 0

        # Embed input
        if features is None:
            features = pos.new_ones(N, self.in_channels)
        s = self.embed_scalar(features)

        if v_init is not None:
            v = v_init
        else:
            v = pos.new_zeros(N, C_v, 3)

        # Type-2: zero-init (no type-2 input from raw data)
        t2 = pos.new_zeros(N, C_t2, 5) if has_t2 else None

        if batch is None:
            B = ptr.shape[0] - 1
            counts = ptr[1:] - ptr[:-1]
            batch = torch.arange(B, device=pos.device).repeat_interleave(counts)

        # ── Encoder ──
        enc_s_list: List[Tensor] = []
        enc_v_list: List[Tensor] = []
        enc_t2_list: List[Optional[Tensor]] = []
        enc_pos_list: List[Tensor] = []
        enc_ptr_list: List[Tensor] = []
        enc_batch_list: List[Tensor] = []
        enc_graph_list: List[SpatialGraph] = []

        cur_pos = pos
        cur_ptr = ptr
        cur_batch = batch

        for i in range(self.num_stages):
            if has_t2:
                s, v, t2, enc_graph = self.encoder_stages[i](
                    s, v, cur_pos, cur_ptr, cur_batch, type2=t2
                )
            else:
                s, v, enc_graph = self.encoder_stages[i](
                    s, v, cur_pos, cur_ptr, cur_batch
                )

            enc_s_list.append(s)
            enc_v_list.append(v)
            enc_t2_list.append(t2 if has_t2 else None)
            enc_pos_list.append(cur_pos)
            enc_ptr_list.append(cur_ptr)
            enc_batch_list.append(cur_batch)
            enc_graph_list.append(enc_graph)

            if i < self.num_stages - 1:
                if has_t2:
                    cur_pos, s, v, t2, cur_ptr, fps_idx = self.pool_layers[i](
                        cur_pos, s, v, cur_ptr, type2=t2
                    )
                else:
                    cur_pos, s, v, cur_ptr, fps_idx = self.pool_layers[i](
                        cur_pos, s, v, cur_ptr
                    )
                cur_batch = enc_batch_list[-1][fps_idx]

        # ── Bottleneck attention (scalar-only) ──
        if self.use_bottleneck_attn:
            B_bn = cur_ptr.shape[0] - 1
            counts_bn = cur_ptr[1:] - cur_ptr[:-1]
            N_max = counts_bn.max().item()

            s_padded = s.new_zeros(B_bn, N_max, s.shape[-1])
            pos_padded = cur_pos.new_zeros(B_bn, N_max, 3)
            mask_padded = torch.zeros(B_bn, N_max, dtype=torch.bool, device=s.device)

            for b_idx in range(B_bn):
                start = cur_ptr[b_idx].item()
                end = cur_ptr[b_idx + 1].item()
                n_b = end - start
                s_padded[b_idx, :n_b] = s[start:end]
                pos_padded[b_idx, :n_b] = cur_pos[start:end]
                mask_padded[b_idx, :n_b] = True

            s_padded = self.bottleneck_attn(s_padded, pos_padded, mask=mask_padded)

            for b_idx in range(B_bn):
                start = cur_ptr[b_idx].item()
                end = cur_ptr[b_idx + 1].item()
                n_b = end - start
                s[start:end] = s_padded[b_idx, :n_b]

        # ── Decoder ──
        for i in range(self.num_stages - 2, -1, -1):
            if has_t2:
                s_interp, v_interp, t2_interp = self.interp_layers[i](
                    enc_pos_list[i], cur_pos, s, v,
                    enc_ptr_list[i], cur_ptr,
                    s_skip=enc_s_list[i], v_skip=enc_v_list[i],
                    t2_coarse=t2, t2_skip=enc_t2_list[i],
                )
            else:
                s_interp, v_interp = self.interp_layers[i](
                    enc_pos_list[i], cur_pos, s, v,
                    enc_ptr_list[i], cur_ptr,
                    s_skip=enc_s_list[i], v_skip=enc_v_list[i],
                )

            # Project back
            s = self.skip_proj_s[i](s_interp)

            N_i = v_interp.shape[0]
            v_flat = v_interp.transpose(1, 2).reshape(N_i * 3, C_v * 2)
            v_proj = self.skip_proj_v[i](v_flat)
            v = v_proj.reshape(N_i, 3, C_v).transpose(1, 2)

            if has_t2:
                # t2_interp: [N_i, C_t2*2, 5] → project → [N_i, C_t2, 5]
                t2_flat = t2_interp.transpose(1, 2).reshape(N_i * 5, C_t2 * 2)
                t2_proj = self.skip_proj_t2[i](t2_flat)
                t2 = t2_proj.reshape(N_i, 5, C_t2).transpose(1, 2)

            # Skip norm
            if has_t2:
                s, v, t2 = self.skip_norms[i](s, v, t2)
            else:
                s, v = self.skip_norms[i](s, v)

            cur_pos = enc_pos_list[i]
            cur_ptr = enc_ptr_list[i]
            cur_batch = enc_batch_list[i]

            if has_t2:
                s, v, t2, _ = self.decoder_stages[i](
                    s, v, cur_pos, cur_ptr, cur_batch,
                    graph=enc_graph_list[i], type2=t2,
                )
            else:
                s, v, _ = self.decoder_stages[i](
                    s, v, cur_pos, cur_ptr, cur_batch,
                    graph=enc_graph_list[i],
                )

        # Final norm
        if has_t2:
            s, v, t2 = self.final_norm(s, v, t2)
        else:
            s, v = self.final_norm(s, v)

        if has_t2:
            if return_encoder_features:
                return s, v, t2, ptr, enc_s_list, enc_ptr_list
            return s, v, t2, ptr
        else:
            if return_encoder_features:
                return s, v, ptr, enc_s_list, enc_ptr_list
            return s, v, ptr

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, "
            f"hidden_s={self.hidden_scalar}, hidden_v={self.hidden_vector}, "
            f"hidden_t2={self.hidden_type2}, stages={self.num_stages}"
        )
