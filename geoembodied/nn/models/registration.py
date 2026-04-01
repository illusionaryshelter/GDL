# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3)-equivariant point cloud registration task module.

Architecture:
    PointCloudBatch → SE3Net (packed) → to_dense_batch →
    InvariantCrossAttention (dense, l=0 only) → ScalarGate →
    InlierPredictor → Sinkhorn + Weighted SVD → (R, t) ∈ SE(3)

Design principles:
    - Task module wraps backbone (timm-style): no feature extraction logic
    - API accepts PointCloudBatch: graph construction stays in DataLoader
    - Packed→Dense bridge via to_dense_batch with boolean mask
    - Mask propagates to Sinkhorn and SVD: padding never corrupts output
    - Shared backbone (default): src/tgt map to same latent space

Data flow shape annotations::

    Phase A (backbone):    [N_total, 3] → [N_total, C_s], [N_total, C_v, 3]
    Phase B (unbatch):     [N_total, C] → [B, N_max, C] + mask [B, N_max]
    Phase C (cross-attn):  [B, N, C] × [B, M, C] → fused features
    Phase D (inlier/desc): [B, N, C] → W [B, N, 1], desc [B, N, D]
    Phase E (SVD):         desc + pos + mask → R [B, 3, 3], t [B, 3]
"""

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from geoembodied.data.batch import PointCloudBatch, to_dense_batch
from geoembodied.nn.invariant_cross_attention import (
    InvariantCrossAttention,
    ScalarGate,
)
from geoembodied.nn.models.robust_registration import (
    InlierPredictor,
    RobustRegistrationHead,
)
from geoembodied.nn.se3_net import SE3Net


class GeoRegistrationModel(nn.Module):
    """SE(3)-equivariant point cloud registration.

    Wraps SE3Net backbone with cross-attention mid-fusion,
    inlier prediction, and robust SVD alignment.

    The model accepts two ``PointCloudBatch`` objects (source and target)
    and returns a dict with predicted rotation, translation, and
    auxiliary outputs for loss computation.

    Args:
        backbone: Pre-built SE3Net instance. Shared between src/tgt
            by default (``share_backbone=True``).
        cross_attention_layers: Number of cross-attention rounds.
            Each round performs bidirectional l=0 attention + scalar gating.
        descriptor_dim: Projection dimension for feature descriptors.
        sinkhorn_iters: Number of Sinkhorn OT iterations.
        share_backbone: If True, src and tgt share the same SE3Net
            (recommended for same-modality registration).
            If False, a deep copy is created for siamese operation.

    Example::

        >>> backbone = SE3Net(hidden_scalar=32, hidden_vector=8,
        ...                   num_layers=3, radius=2.5, max_num_neighbors=16)
        >>> model = GeoRegistrationModel(backbone, cross_attention_layers=2)
        >>> batch_src = collate_point_clouds([{"pos": src_pts}])
        >>> batch_tgt = collate_point_clouds([{"pos": tgt_pts}])
        >>> out = model(batch_src, batch_tgt)
        >>> out["R"].shape  # [1, 3, 3]
        >>> out["t"].shape  # [1, 3]
    """

    def __init__(
        self,
        backbone: SE3Net,
        cross_attention_layers: int = 1,
        descriptor_dim: int = 32,
        sinkhorn_iters: int = 10,
        share_backbone: bool = True,
    ) -> None:
        super().__init__()

        # --- Backbone ---
        # Shared: both references point to the same module (same params)
        # Siamese: deep copy for independent params
        self.backbone_src: SE3Net = backbone
        if share_backbone:
            self.backbone_tgt: SE3Net = backbone
        else:
            self.backbone_tgt = copy.deepcopy(backbone)

        self.share_backbone = share_backbone

        C_s: int = backbone.hidden_scalar
        C_v: int = backbone.hidden_vector

        # --- Cross-attention (l=0 scalars only, SO(3)-invariant) ---
        self.cross_attn_layers = nn.ModuleList([
            InvariantCrossAttention(channels=C_s)
            for _ in range(cross_attention_layers)
        ])
        self.scalar_gates_src = nn.ModuleList([
            ScalarGate(scalar_channels=C_s, vector_channels=C_v)
            for _ in range(cross_attention_layers)
        ])
        self.scalar_gates_tgt = nn.ModuleList([
            ScalarGate(scalar_channels=C_s, vector_channels=C_v)
            for _ in range(cross_attention_layers)
        ])

        # --- Inlier predictor ---
        self.inlier_pred = InlierPredictor(
            scalar_channels=C_s,
            vector_channels=C_v,
        )

        # --- Descriptor head (shared for src/tgt) ---
        self.desc_head = nn.Sequential(
            nn.Linear(C_s, descriptor_dim),
            nn.SiLU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

        # --- Registration head (Sinkhorn + Weighted SVD) ---
        self.reg_head = RobustRegistrationHead(
            descriptor_dim=descriptor_dim,
            sinkhorn_iters=sinkhorn_iters,
        )

    def forward(
        self,
        batch_src: PointCloudBatch,
        batch_tgt: PointCloudBatch,
    ) -> dict[str, Tensor]:
        """End-to-end registration from two packed point cloud batches.

        Args:
            batch_src: Source point cloud batch (from collate_point_clouds).
                Must contain pos, batch, sizes, num_graphs.
            batch_tgt: Target point cloud batch.
                Must have same num_graphs as batch_src.

        Returns:
            dict with keys:
                R: Predicted rotation, shape [B, 3, 3],
                    representation: SO(3)
                t: Predicted translation, shape [B, 3]
                assignment: Sinkhorn assignment,
                    shape [B, N_max, M_max] (for loss computation)
                weights_src: Inlier confidence,
                    shape [B, N_max, 1]
                mask_src: Source valid mask,
                    shape [B, N_max]
                mask_tgt: Target valid mask,
                    shape [B, M_max]
        """
        B = batch_src.num_graphs
        assert B == batch_tgt.num_graphs, (
            f"Batch size mismatch: src={B}, tgt={batch_tgt.num_graphs}"
        )

        # ══════════════════════════════════════════════════════════
        # Phase A: Backbone feature extraction (packed mode)
        # SE3Net operates on packed [N_total, 3] → [N_total, C]
        # Graph construction happens inside SE3Net.forward via
        # SpatialGraph.build (radius_graph + batch vector).
        #
        # NOTE: Do NOT pass num_batch_elements here. Registration
        # inputs have variable-size point clouds per batch element.
        # Passing B would force the equal-size batched fast path
        # which crashes on x.reshape(B, N, 3) when sizes differ.
        # ══════════════════════════════════════════════════════════

        s_src, v_src = self.backbone_src(
            batch_src.pos, batch=batch_src.batch,
        )  # s: [N_total_s, C_s], v: [N_total_s, C_v, 3]

        s_tgt, v_tgt = self.backbone_tgt(
            batch_tgt.pos, batch=batch_tgt.batch,
        )  # s: [N_total_t, C_s], v: [N_total_t, C_v, 3]

        # ══════════════════════════════════════════════════════════
        # Phase B: Unbatch — Packed → Dense
        # to_dense_batch: [N_total, *] → [B, N_max, *] + mask
        # mask is critical for all downstream dense operations.
        # ══════════════════════════════════════════════════════════

        s_src_d, mask_src = to_dense_batch(
            s_src, batch_src.batch, B,
        )  # [B, N_max, C_s], [B, N_max]
        s_tgt_d, mask_tgt = to_dense_batch(
            s_tgt, batch_tgt.batch, B,
        )  # [B, M_max, C_s], [B, M_max]

        v_src_d, _ = to_dense_batch(
            v_src, batch_src.batch, B,
        )  # [B, N_max, C_v, 3] — trailing dims preserved
        v_tgt_d, _ = to_dense_batch(
            v_tgt, batch_tgt.batch, B,
        )  # [B, M_max, C_v, 3]

        # ══════════════════════════════════════════════════════════
        # Phase C: Cross-Attention + ScalarGate (dense mode)
        # InvariantCrossAttention operates on l=0 scalars only.
        # ScalarGate modulates l=1 vectors via scalar gating (0⊗1→1).
        # Masks are forwarded to attention for pad isolation.
        # ══════════════════════════════════════════════════════════

        for cross_attn, gate_src, gate_tgt in zip(
            self.cross_attn_layers,
            self.scalar_gates_src,
            self.scalar_gates_tgt,
        ):
            # Bidirectional l=0 cross-attention
            m_src, m_tgt = cross_attn(
                s_src_d, s_tgt_d, mask_src, mask_tgt,
            )  # [B, N_max, C_s], [B, M_max, C_s]

            # Scalar gate: fuse cross message into self features
            s_src_d, v_src_d = gate_src(
                s_src_d, v_src_d, m_src, mask_src,
            )
            s_tgt_d, v_tgt_d = gate_tgt(
                s_tgt_d, v_tgt_d, m_tgt, mask_tgt,
            )

        # ══════════════════════════════════════════════════════════
        # Phase D: Inlier Prediction + Descriptor Extraction
        # InlierPredictor: [scalars | ‖vectors‖²] → W ∈ [0,1]
        # Descriptor head: scalars → desc (shared for src/tgt)
        # ══════════════════════════════════════════════════════════

        # Reshape for InlierPredictor: expects [N, C_s] and [N, C_v, 3]
        # We flatten batch dim, compute, then reshape back.
        N_max = s_src_d.shape[1]
        M_max = s_tgt_d.shape[1]

        w_src = self.inlier_pred(
            s_src_d.reshape(-1, s_src_d.shape[-1]),
            v_src_d.reshape(-1, v_src_d.shape[-2], 3),
        ).reshape(B, N_max, 1)  # [B, N, 1]

        # Zero out padding inlier weights
        w_src = w_src * mask_src.unsqueeze(-1).float()

        desc_src = self.desc_head(s_src_d)  # [B, N_max, D]
        desc_tgt = self.desc_head(s_tgt_d)  # [B, M_max, D]

        # Zero out padding descriptors (prevent similarity leakage)
        desc_src = desc_src * mask_src.unsqueeze(-1).float()
        desc_tgt = desc_tgt * mask_tgt.unsqueeze(-1).float()

        # ══════════════════════════════════════════════════════════
        # Phase E: Sinkhorn + Weighted SVD
        # Positions must also be unbatched to dense for SVD.
        # Masks propagate to Sinkhorn (padding → -inf) and SVD
        # (padding → zero confidence weight).
        # ══════════════════════════════════════════════════════════

        pos_src_d, _ = to_dense_batch(
            batch_src.pos, batch_src.batch, B,
        )  # [B, N_max, 3]
        pos_tgt_d, _ = to_dense_batch(
            batch_tgt.pos, batch_tgt.batch, B,
        )  # [B, M_max, 3]

        R_pred, t_pred, assignment_full = self.reg_head(
            desc_src, desc_tgt,
            pos_src_d, pos_tgt_d,
            weights_src=w_src,
            mask_src=mask_src,
            mask_tgt=mask_tgt,
        )

        return {
            "R": R_pred,        # [B, 3, 3], representation: SO(3)
            "t": t_pred,        # [B, 3]
            "pos_src": pos_src_d,  # [B, N_max, 3] (dense, for CD loss)
            "pos_tgt": pos_tgt_d,  # [B, M_max, 3] (dense, for CD loss)
            "assignment": assignment_full,  # [B, N, M+1] dustbin or [B, N, M]
            "weights_src": w_src,  # [B, N_max, 1]
            "mask_src": mask_src,  # [B, N_max]
            "mask_tgt": mask_tgt,  # [B, M_max]
        }
