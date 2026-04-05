# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3)-equivariant part segmentation model.

A generic part segmentation architecture built on MultiScaleSE3Net.
Dataset-specific constants (num_categories, num_parts, part_mask)
are passed as constructor arguments — no hardcoded dataset knowledge.

Architecture::

    pos, normals → MultiScaleSE3Net (U-Net) → s_out [N, C_s]
                                               v_out [N, C_v, 3]
                                               t2_out [N, C_t2, 5]
                                              ↓
    Invariant projection:  v_inv = proj(||v_c||)   ← SO(3)-invariant
                          t2_inv = ||t2_c||²        ← SO(3)-invariant
                                              ↓
    cat_indices → one_hot [N, num_categories] ─┤
                                              ↓
                             MLP head → logits [N, num_parts]
                                              ↓
                          category mask → masked logits [N, num_parts]

Design principles:
    - Head uses SO(3)-invariant features derived from ALL representation
      types: scalars (l=0), vector norms (l=1 → l=0), type-2 norms (l=2 → l=0)
    - This follows the MACE/Vector Neurons principle: invariant readout
      from equivariant features via norm contraction.
    - Category one-hot injected at HEAD level only (backbone stays
      category-agnostic → can pretrain across datasets)
    - Category mask is MANDATORY (prevents impossible part predictions)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast


def _amp_safe_cast(x: Tensor) -> Tensor:
    """Upcast FP16/BF16 → FP32 for AMP safety, but PRESERVE FP64.

    `.float()` unconditionally casts to FP32, destroying FP64 precision
    needed for equivariance tests.  This helper only upcasts half dtypes.
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x  # FP32 or FP64 — keep as-is

from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net


class SE3PartSegNet(nn.Module):
    """SE(3)-equivariant part segmentation network.

    Generic version: dataset-specific constants are constructor arguments.

    Args:
        num_categories: Number of object categories (e.g. 16 for ShapeNet)
        num_parts: Total number of part labels (e.g. 50 for ShapeNet)
        category_part_mask: Boolean mask [num_categories, num_parts]
            indicating valid part labels per category. If None, all
            parts are valid for all categories (no masking).
        in_channels: Input scalar feature dimension (default: 1)
        hidden_scalar: Hidden scalar channels in U-Net
        hidden_vector: Hidden vector channels in U-Net
        num_stages: Number of encoder/decoder stages
        layers_per_stage: SE3Conv layers per stage
        pool_ratio: Downsampling ratio per stage
        use_normals: If True, inject normals as initial l=1 vector features
        head_hidden: Hidden dim in classification head

    Example::

        # ShapeNet Part Segmentation
        model = SE3PartSegNet(
            num_categories=16,
            num_parts=50,
            category_part_mask=shapenet_mask,  # [16, 50] bool
            hidden_scalar=96,
            hidden_vector=24,
        )

        # S3DIS Semantic Segmentation (no category conditioning)
        model = SE3PartSegNet(
            num_categories=1,
            num_parts=13,
            category_part_mask=None,
            hidden_scalar=128,
            hidden_vector=32,
        )
    """

    def __init__(
        self,
        num_categories: int,
        num_parts: int,
        category_part_mask: Optional[Tensor] = None,
        in_channels: int = 1,
        hidden_scalar: int = 64,
        hidden_vector: int = 16,
        hidden_type2: int = 0,
        num_stages: int = 3,
        layers_per_stage: int = 2,
        pool_ratio: float = 0.25,
        use_normals: bool = True,
        head_hidden: int = 128,
        gate_mode: str = 'scalar',
        use_self_tp: bool = False,
        use_bottleneck_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_parts = num_parts
        self.use_normals = use_normals
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.hidden_type2 = hidden_type2
        self.num_stages = num_stages

        # U-Net backbone
        self.backbone = MultiScaleSE3Net(
            in_channels=in_channels,
            hidden_scalar=hidden_scalar,
            hidden_vector=hidden_vector,
            hidden_type2=hidden_type2,
            num_stages=num_stages,
            layers_per_stage=layers_per_stage,
            pool_ratio=pool_ratio,
            gate_mode=gate_mode,
            use_self_tp=use_self_tp,
            use_bottleneck_attn=use_bottleneck_attn,
        )

        # Normal vector embedding: project 3D normals → C_v vector channels
        # This is an equivariant linear map: v_out = W @ v_in
        # where v_in: [N, 1, 3], v_out: [N, C_v, 3]
        if use_normals:
            self.normal_proj = nn.Linear(1, hidden_vector, bias=False)

        # ── Vector Invariant Projection ──
        # Extract SO(3)-invariant information from vector (l=1) features.
        #
        # ||v_c|| = sqrt(v_c · v_c) is SO(3)-invariant per channel.
        # This linear map learns which combinations of vector channel
        # magnitudes are informative for downstream tasks.
        #
        # Follows Vector Neurons (Deng et al., ICCV 2021) and MACE
        # readout principles: invariant contractions from equivariant reps.
        v_inv_dim = hidden_vector  # one norm per vector channel
        v_inv_out = hidden_scalar // 2
        self.v_inv_proj = nn.Linear(v_inv_dim, v_inv_out, bias=False)
        # LayerNorm: ensures v_inv has comparable magnitude to other head
        # inputs (s_out, multi-scale globals), preventing one component from
        # dominating the head's first Linear simply due to scale mismatch.
        self.v_inv_norm = nn.LayerNorm(v_inv_out)

        # ── Type-2 Invariant Projection ──
        # Extract SO(3)-invariant information from type-2 (l=2) features.
        #
        # ||t2_c||² = Σ_m |t2_{c,m}|² is the l=2 Casimir invariant,
        # guaranteed invariant under SO(3): ||D²(R) t2||² = ||t2||².
        #
        # Uses the SAME readout pattern as v_inv:
        #   invariant contraction → learned projection → LayerNorm
        # Without projection+normalization, t2_inv has activation std ≈ 0.03
        # while normalized features have std ≈ 1.0 — a 32x scale mismatch
        # that makes the head unable to effectively utilize type-2 information.
        t2_inv_out = hidden_scalar // 2 if hidden_type2 > 0 else 0
        if hidden_type2 > 0:
            self.t2_inv_proj = nn.Linear(hidden_type2, t2_inv_out, bias=False)
            self.t2_inv_norm = nn.LayerNorm(t2_inv_out)

        # ── Classification Head ──
        # Input: local scalar [N, C_s]
        #        + vector invariant [N, C_s // 2] (projected ||v_c||)
        #        + type-2 invariant [N, C_s // 2] (projected ||t2_c||)
        #        + multi-scale globals [N, num_stages * C_s]
        #        + category one-hot [N, num_categories]
        # Output: logits [N, num_parts]
        head_in = (
            hidden_scalar           # s_out
            + hidden_scalar // 2    # v_inv (projected vector norms)
            + t2_inv_out            # t2_inv (projected type-2 norms)
            + hidden_scalar * num_stages  # multi-scale globals
            + num_categories        # category one-hot
        )
        self._head_in_dim = head_in  # for diagnostics
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(head_hidden, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, num_parts),
        )

        # Register category mask as buffer (moves with model to device)
        if category_part_mask is not None:
            self.register_buffer(
                'cat_mask', category_part_mask.clone(),
                persistent=False,
            )
        else:
            # No masking — all parts valid for all categories
            self.register_buffer(
                'cat_mask',
                torch.ones(num_categories, num_parts, dtype=torch.bool),
                persistent=False,
            )

    def inject_normals(self, normals: Tensor) -> Tensor:
        """Project surface normals into vector feature space.

        Args:
            normals: Unit normals
                shape: [N, 3], representation: SO(3) type-1

        Returns:
            v_init: Vector features from normals
                shape: [N, C_v, 3], representation: SO(3) type-1
        """
        # normals: [N, 3] → [N, 1, 3] → Linear(1, C_v) → [N, C_v, 3]
        v = normals.unsqueeze(1)                   # [N, 1, 3]
        v = v.transpose(1, 2)                      # [N, 3, 1]
        v = self.normal_proj(v)                    # [N, 3, C_v]
        v = v.transpose(1, 2)                      # [N, C_v, 3]
        return v

    def forward(
        self,
        pos: Tensor,              # [N_total, 3]
        ptr: Tensor,              # [B+1] int64
        cat_indices: Tensor,      # [B] int64
        normals: Optional[Tensor] = None,  # [N_total, 3]
        features: Optional[Tensor] = None, # [N_total, in_channels]
    ) -> Tensor:
        """Forward pass.

        Args:
            pos: Packed point positions
                shape: [N_total, 3]
            ptr: CSR batch offsets
                shape: [B+1], int64
            cat_indices: Category index per shape
                shape: [B], int64, values in [0, num_categories-1]
            normals: Optional point normals (l=1 vector features)
                shape: [N_total, 3], representation: SO(3) type-1
            features: Optional scalar input features
                shape: [N_total, in_channels]

        Returns:
            logits: Per-point part logits with category masking
                shape: [N_total, num_parts]
                Invalid part logits are set to -inf
        """
        N = pos.shape[0]
        B = ptr.shape[0] - 1
        device = pos.device

        # ── AMP GUARD ──────────────────────────────────────────────
        # Disable autocast for the ENTIRE model forward pass.
        #
        # Why: CUDA autocast silently demotes ALL nn.Linear and
        # matmul (@) operations to FP16 — including geometric layers
        # (RadialBasis MLP, gate projections, skip projections,
        # normal embedding, classification head).  In the backward
        # pass, these FP16 matmul gradients × GradScaler scale
        # overflow FP16 max (65504) → Inf → GradScaler halves
        # scale every epoch, eventually reaching scale=1 and
        # causing gradient underflow + training collapse.
        #
        # This model's computation is >95% FP32 by design (SE3Conv
        # requires FP32 for geometric tensor products).  The few
        # nn.Linear layers that autocast would "optimize" to FP16
        # are too small to matter for throughput.  Disabling
        # autocast here ensures stable training with zero perf cost.
        # ──────────────────────────────────────────────────────────
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type, enabled=False):
            # Upcast half → FP32; preserve FP64 for equivariance tests
            pos = _amp_safe_cast(pos)
            if normals is not None:
                normals = _amp_safe_cast(normals)
            if features is not None:
                features = _amp_safe_cast(features)

            # ── 1. Project normals → initial vector features ──
            v_init = None
            if self.use_normals and normals is not None:
                v_init = self.inject_normals(normals)  # [N, C_v, 3]

            # ── 2. Run U-Net backbone (with encoder features for multi-scale head) ──
            backbone_out = self.backbone(
                pos, ptr, features=features, v_init=v_init,
                return_encoder_features=True,
            )
            if self.hidden_type2 > 0:
                s_out, v_out, t2_out, _, enc_s_list, enc_ptr_list = backbone_out
            else:
                s_out, v_out, _, enc_s_list, enc_ptr_list = backbone_out
                t2_out = None

            # ── 3. Build multi-scale global features ──
            global_feats = []
            for enc_s, enc_ptr in zip(enc_s_list, enc_ptr_list):
                B_enc = enc_ptr.shape[0] - 1
                enc_counts = enc_ptr[1:] - enc_ptr[:-1]
                enc_batch = torch.arange(B_enc, device=device).repeat_interleave(enc_counts)
                shape_sum = torch.zeros(B_enc, self.hidden_scalar, device=device, dtype=enc_s.dtype)
                shape_sum.scatter_add_(0, enc_batch.unsqueeze(1).expand_as(enc_s), enc_s)
                shape_mean = shape_sum / enc_counts.unsqueeze(1).clamp(min=1).to(enc_s.dtype)
                global_feats.append(shape_mean)

            # ── 4. Build per-point category one-hot ──
            sizes = ptr[1:] - ptr[:-1]
            point_cat_batch = torch.arange(B, device=device).repeat_interleave(sizes)
            point_cat = cat_indices[point_cat_batch]

            one_hot = torch.zeros(N, self.num_categories, device=device, dtype=s_out.dtype)
            one_hot.scatter_(1, point_cat.unsqueeze(1), 1.0)

            # ── 5. Multi-scale global → per-point broadcast ──
            point_globals = [g[point_cat_batch] for g in global_feats]
            multi_scale = torch.cat(point_globals, dim=-1)

            # ── 6. Invariant feature extraction ──
            # All features entering the head MUST be SO(3)-invariant.
            head_parts = [s_out]

            # Vector invariant: ||v_c|| per channel → learned projection → LayerNorm
            # ||v_c|| = sqrt(sum_d v[c,d]²) is invariant under SO(3)
            v_norms = v_out.norm(dim=-1)  # [N, C_v]
            v_inv = self.v_inv_norm(self.v_inv_proj(v_norms))  # [N, C_s // 2]
            head_parts.append(v_inv)

            # Type-2 invariant: ||t2_c|| per channel → learned projection → LayerNorm
            # ||t2_c||² is the l=2 Casimir invariant (SO(3)-invariant)
            if t2_out is not None and self.hidden_type2 > 0:
                t2_norms = t2_out.norm(dim=-1)  # [N, C_t2]
                t2_inv = self.t2_inv_norm(self.t2_inv_proj(t2_norms))  # [N, C_s // 2]
                head_parts.append(t2_inv)
            head_parts.extend([multi_scale, one_hot])

            # ── 7. Classification head ──
            head_input = torch.cat(head_parts, dim=1)
            logits = self.head(head_input)

            # ── 8. Category masking ──
            mask = self.cat_mask[point_cat]
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        return logits

    def forward_with_diagnostics(
        self,
        pos: Tensor,
        ptr: Tensor,
        cat_indices: Tensor,
        normals: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Forward pass returning diagnostics for training monitoring.

        Returns:
            logits: Same as forward()
            diagnostics: Dict with:
                v_norm_mean: Mean vector feature norm
                v_norm_std: Std of vector feature norms
                attn_entropy_mean: Mean attention entropy across pool layers
                attn_max_mean: Mean max attention weight
                attn_uniform_ratio: How close to uniform (1.0 = uniform)
        """
        N = pos.shape[0]
        B = ptr.shape[0] - 1
        device = pos.device

        # AMP guard: same as forward() — see docstring there.
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type, enabled=False):
            pos = _amp_safe_cast(pos)
            if normals is not None:
                normals = _amp_safe_cast(normals)
            if features is not None:
                features = _amp_safe_cast(features)

            v_init = None
            if self.use_normals and normals is not None:
                v_init = self.inject_normals(normals)

            backbone_out = self.backbone(
                pos, ptr, features=features, v_init=v_init,
                return_encoder_features=True,
            )
            if self.hidden_type2 > 0:
                s_out, v_out, t2_out, _, enc_s_list, enc_ptr_list = backbone_out
            else:
                s_out, v_out, _, enc_s_list, enc_ptr_list = backbone_out
                t2_out = None

            # ── Diagnostics ──
            diag: dict = {}

            # Vector feature norms (per-channel, then per-point)
            v_per_channel_norms = v_out.norm(dim=-1)  # [N, C_v]
            v_per_point = v_per_channel_norms.mean(dim=-1)  # [N]
            diag['v_norm_mean'] = v_per_point.mean().item()
            diag['v_norm_std'] = v_per_point.std().item()

            for i, enc_s in enumerate(enc_s_list):
                diag[f'enc{i}_s_norm'] = enc_s.norm(dim=-1).mean().item()

            # Decoder output scalar norm (post-final_norm)
            diag['s_out_norm'] = s_out.norm(dim=-1).mean().item()

            # Type-2 diagnostics
            if t2_out is not None:
                t2_norms = t2_out.norm(dim=-1).mean(dim=-1)  # [N]
                diag['t2_norm_mean'] = t2_norms.mean().item()
                diag['t2_norm_std'] = t2_norms.std().item()

            diag.update(self.get_pool_attn_stats())

            # ── Multi-scale Head (same logic as forward) ──
            global_feats = []
            for enc_s, enc_ptr in zip(enc_s_list, enc_ptr_list):
                B_enc = enc_ptr.shape[0] - 1
                enc_counts = enc_ptr[1:] - enc_ptr[:-1]
                enc_batch = torch.arange(B_enc, device=device).repeat_interleave(enc_counts)
                shape_sum = torch.zeros(B_enc, self.hidden_scalar, device=device, dtype=enc_s.dtype)
                shape_sum.scatter_add_(0, enc_batch.unsqueeze(1).expand_as(enc_s), enc_s)
                shape_mean = shape_sum / enc_counts.unsqueeze(1).clamp(min=1).to(enc_s.dtype)
                global_feats.append(shape_mean)

            sizes = ptr[1:] - ptr[:-1]
            point_cat_batch = torch.arange(B, device=device).repeat_interleave(sizes)
            point_cat = cat_indices[point_cat_batch]

            one_hot = torch.zeros(N, self.num_categories, device=device, dtype=s_out.dtype)
            one_hot.scatter_(1, point_cat.unsqueeze(1), 1.0)

            point_globals = [g[point_cat_batch] for g in global_feats]
            multi_scale = torch.cat(point_globals, dim=-1)

            # ── 6. Invariant feature extraction (same as forward) ──
            head_parts = [s_out]

            # Vector invariant: ||v_c|| → learned projection → LayerNorm
            v_norms = v_per_channel_norms  # reuse from diagnostics
            v_inv = self.v_inv_norm(self.v_inv_proj(v_norms))  # [N, C_s // 2]
            head_parts.append(v_inv)

            # Vector invariant diagnostics
            diag['v_inv_norm'] = v_inv.norm(dim=-1).mean().item()
            diag['v_inv_std'] = v_inv.std(dim=0).mean().item()
            # v_utilization: ratio of v_inv contribution to head input
            # Higher = vector features contributing more to predictions
            diag['head_in_dim'] = self._head_in_dim

            if t2_out is not None and self.hidden_type2 > 0:
                t2_norms = t2_out.norm(dim=-1)  # [N, C_t2]
                t2_inv = self.t2_inv_norm(self.t2_inv_proj(t2_norms))  # [N, C_s // 2]
                head_parts.append(t2_inv)
                diag['t2_inv_norm'] = t2_inv.norm(dim=-1).mean().item()
                diag['t2_inv_std'] = t2_inv.std(dim=0).mean().item()
            head_parts.extend([multi_scale, one_hot])

            head_input = torch.cat(head_parts, dim=1)
            logits = self.head(head_input)

            mask = self.cat_mask[point_cat]
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        return logits, diag

    def get_pool_attn_stats(self) -> dict:
        """Extract attention statistics from pool layers.

        Returns per-pool breakdowns in addition to averages for detailed
        monitoring (e.g., deep pool may diverge while shallow is fine).
        """
        import math
        entropies = []
        maxes = []
        temperatures = []

        for idx, layer in enumerate(self.backbone.pool_layers):
            if not hasattr(layer, '_last_attn_weights'):
                continue
            w = layer._last_attn_weights     # [N_out, K]

            # Per-seed entropy
            log_w = torch.log(w.clamp(min=1e-10))
            ent = -(w * log_w).sum(dim=-1)   # [N_out]
            entropies.append(ent.mean().item())

            # Per-seed max weight
            maxes.append(w.max(dim=-1).values.mean().item())

            # Learned temperature (< 1 = sharpening, > 1 = smoothing)
            temp = layer.log_temperature.exp().item()
            temperatures.append(temp)

        K = self.backbone.pool_layers[0].k_neighbors if self.backbone.pool_layers else 16
        max_ent = math.log(K)

        stats = {
            'attn_entropy_mean': sum(entropies) / max(len(entropies), 1),
            'attn_max_mean': sum(maxes) / max(len(maxes), 1),
            'attn_uniform_ratio': (sum(entropies) / max(len(entropies), 1)) / max_ent
                                  if max_ent > 0 else 1.0,
        }

        # Per-pool breakdowns: pool_T0, pool_T1, ...; pool_u0, pool_u1, ...
        for i, (ent, mx, temp) in enumerate(zip(entropies, maxes, temperatures)):
            stats[f'pool_T{i}'] = temp
            stats[f'pool_u{i}'] = ent / max_ent if max_ent > 0 else 1.0
            stats[f'pool_max{i}'] = mx

        if temperatures:
            stats['pool_temperature'] = sum(temperatures) / len(temperatures)
        return stats

