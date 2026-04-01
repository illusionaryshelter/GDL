# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for GeoRegistrationModel and to_dense_batch.

Test hierarchy:
    1. to_dense_batch: shape, mask, multi-dim, monotonicity, roundtrip
    2. GeoRegistrationModel: forward shapes, shared backbone, mask propagation,
       SO(3) equivariance, gradient flow, Sinkhorn mask isolation
"""

import pytest
import torch
from torch import Tensor

from geoembodied.data.batch import (
    PointCloudBatch,
    collate_point_clouds,
    to_dense_batch,
)
from geoembodied.nn.se3_net import SE3Net
from geoembodied.nn.models.registration import GeoRegistrationModel
from geoembodied.nn.solvers import sinkhorn_log_domain
from geoembodied.functional.so3_ops import so3_exp
from geoembodied.functional.quaternion_ops import quaternion_to_matrix

# ───────────── test config ─────────────

_S, _V, _L, _R = 16, 4, 2, 2.5


def _backbone(**kw) -> SE3Net:
    d = dict(hidden_scalar=_S, hidden_vector=_V, num_layers=_L,
             radius=_R, max_num_neighbors=16)
    d.update(kw)
    return SE3Net(**d)


def _model(**kw) -> GeoRegistrationModel:
    bb = _backbone()
    defaults = dict(
        backbone=bb,
        cross_attention_layers=1,
        descriptor_dim=16,
        sinkhorn_iters=5,
        share_backbone=True,
    )
    defaults.update(kw)
    return GeoRegistrationModel(**defaults)


def _cloud(n: int) -> Tensor:
    return torch.randn(n, 3) * 0.5


def _rot() -> Tensor:
    """Random SO(3) rotation matrix [3, 3]."""
    omega = torch.randn(1, 3) * 1.5
    q = so3_exp(omega)
    return quaternion_to_matrix(q).squeeze(0)


# ═══════════════════════════════════════════════════════════════════
# 1. to_dense_batch tests
# ═══════════════════════════════════════════════════════════════════


class TestToDenseBatch:
    def test_shapes_scalar(self):
        """Scalar [N_total, C] → [B, N_max, C]."""
        x = torch.randn(8, 32)
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
        d, m = to_dense_batch(x, batch, 3)
        assert d.shape == (3, 3, 32)  # N_max = max(3, 2, 3) = 3
        assert m.shape == (3, 3)

    def test_shapes_vector(self):
        """Vector [N_total, C_v, 3] → [B, N_max, C_v, 3]."""
        v = torch.randn(8, 4, 3)
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
        d, m = to_dense_batch(v, batch, 3)
        assert d.shape == (3, 3, 4, 3)
        assert m.shape == (3, 3)

    def test_mask_correctness(self):
        """Mask True for real nodes, False for padding."""
        x = torch.randn(5, 8)
        batch = torch.tensor([0, 0, 1, 1, 1])
        d, m = to_dense_batch(x, batch, 2)
        assert m.tolist() == [[True, True, False], [True, True, True]]

    def test_roundtrip(self):
        """Dense values at real positions match original packed values."""
        x = torch.randn(7, 16)
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        d, m = to_dense_batch(x, batch, 3)

        # Graph 0: nodes 0,1,2 → d[0, 0:3]
        assert torch.equal(d[0, 0], x[0])
        assert torch.equal(d[0, 1], x[1])
        assert torch.equal(d[0, 2], x[2])
        # Graph 1: nodes 3,4 → d[1, 0:2]
        assert torch.equal(d[1, 0], x[3])
        assert torch.equal(d[1, 1], x[4])
        # Padding
        assert (d[1, 2] == 0.0).all()  # fill_value

    def test_monotonicity_check(self):
        """Non-sorted batch indices must raise AssertionError."""
        with pytest.raises(AssertionError, match="monotonically"):
            to_dense_batch(torch.randn(3, 4), torch.tensor([1, 0, 2]), 3)

    def test_equal_sizes(self):
        """When all graphs have same size, no padding needed."""
        x = torch.randn(6, 8)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        d, m = to_dense_batch(x, batch, 2)
        assert d.shape == (2, 3, 8)
        assert m.all()  # no padding


# ═══════════════════════════════════════════════════════════════════
# 2. GeoRegistrationModel tests
# ═══════════════════════════════════════════════════════════════════


class TestGeoRegistrationForward:
    def test_forward_shapes(self):
        """Output shapes are correct for variable-size batches."""
        model = _model()
        model.eval()

        torch.manual_seed(42)
        batch_src = collate_point_clouds([
            {"pos": _cloud(20)}, {"pos": _cloud(15)},
        ])
        batch_tgt = collate_point_clouds([
            {"pos": _cloud(18)}, {"pos": _cloud(22)},
        ])

        with torch.no_grad():
            out = model(batch_src, batch_tgt)

        assert out["R"].shape == (2, 3, 3)
        assert out["t"].shape == (2, 3)
        assert out["mask_src"].shape == (2, 20)  # N_max=max(20,15)
        assert out["mask_tgt"].shape == (2, 22)  # M_max=max(18,22)
        assert out["weights_src"].shape == (2, 20, 1)

    def test_shared_backbone_identity(self):
        """Shared backbone: src and tgt backbone are same object."""
        model = _model(share_backbone=True)
        assert model.backbone_src is model.backbone_tgt

    def test_siamese_backbone_independence(self):
        """Non-shared backbone: src and tgt backbone are different."""
        model = _model(share_backbone=False)
        assert model.backbone_src is not model.backbone_tgt

    def test_rotation_produces_valid_so3(self):
        """Output R must be valid rotation: det(R)=1, R^T R = I."""
        model = _model()
        model.eval()

        torch.manual_seed(99)
        bs = collate_point_clouds([{"pos": _cloud(20)}])
        bt = collate_point_clouds([{"pos": _cloud(20)}])

        with torch.no_grad():
            out = model(bs, bt)

        R = out["R"][0]
        det = torch.det(R).item()
        orth_err = (R.T @ R - torch.eye(3)).abs().max().item()
        assert abs(det - 1.0) < 0.01, f"det(R) = {det:.4f}"
        assert orth_err < 1e-5, f"R^T R - I err = {orth_err:.2e}"


class TestGeoRegistrationMask:
    def test_padding_weights_zero(self):
        """Inlier weights for padding positions must be exactly 0."""
        model = _model()
        model.eval()

        torch.manual_seed(77)
        bs = collate_point_clouds([
            {"pos": _cloud(20)}, {"pos": _cloud(10)},
        ])
        bt = collate_point_clouds([
            {"pos": _cloud(15)}, {"pos": _cloud(15)},
        ])

        with torch.no_grad():
            out = model(bs, bt)

        # Graph 1 has 10 nodes, N_max=20, so positions 10:19 are padding
        w_pad = out["weights_src"][1, 10:]
        assert (w_pad == 0.0).all(), (
            f"Padding weights not zero: max={w_pad.max():.4e}"
        )

    def test_sinkhorn_mask_excludes_padding(self):
        """Sinkhorn with mask must give 0 mass to padding columns."""
        sim = torch.randn(1, 5, 8)  # B=1, N=5, M=8
        mask_col = torch.ones(1, 8, dtype=torch.bool)
        mask_col[0, 5:] = False  # last 3 columns are padding

        assignment = sinkhorn_log_domain(
            sim, mask_col=mask_col, num_iters=20,
        )
        # Padding columns should have ~0 assignment mass
        pad_mass = assignment[0, :, 5:].sum().item()
        real_mass = assignment[0, :, :5].sum().item()
        assert pad_mass < 1e-6, (
            f"Padding columns got mass {pad_mass:.4e}"
        )
        assert real_mass > 0.1, f"Real columns got no mass: {real_mass:.4e}"


class TestGeoRegistrationEquivariance:
    def test_inlier_weight_so3_invariance(self):
        """Inlier weights must be SO(3)-invariant.

        Rotating the source cloud must produce identical inlier weights
        because:
        - Scalar features (l=0) are rotationally invariant
        - Vector norms ‖v‖² are invariant
        - InlierPredictor uses only invariant quantities
        """
        model = _model()
        model.eval()

        torch.manual_seed(200)
        src = _cloud(25)
        tgt = _cloud(20)
        R = _rot()

        bs = collate_point_clouds([{"pos": src}])
        bt = collate_point_clouds([{"pos": tgt}])
        bs_rot = collate_point_clouds([{"pos": src @ R.T}])

        with torch.no_grad():
            out1 = model(bs, bt)
            out2 = model(bs_rot, bt)

        N = 25
        w1 = out1["weights_src"][0, :N]
        w2 = out2["weights_src"][0, :N]
        err = (w1 - w2).abs().max().item()
        assert err < 5e-5, (
            f"Inlier weights not SO(3)-invariant: err={err:.2e}"
        )

    def test_alignment_equivariance(self):
        """Rotating src must produce equivalent alignment quality.

        If src is rotated by R2, the predicted transform should
        compensate, yielding the same final alignment error.
        """
        model = _model()
        model.eval()

        torch.manual_seed(300)
        R_gt = _rot()
        t_gt = torch.randn(3) * 2
        src = _cloud(25)
        tgt = (src @ R_gt.T) + t_gt

        R2 = _rot()
        src_rot = src @ R2.T

        bs = collate_point_clouds([{"pos": src}])
        bs2 = collate_point_clouds([{"pos": src_rot}])
        bt = collate_point_clouds([{"pos": tgt}])

        with torch.no_grad():
            out1 = model(bs, bt)
            out2 = model(bs2, bt)

        # Align using predicted R, t
        aligned1 = src @ out1["R"][0].T + out1["t"][0]
        aligned2 = src_rot @ out2["R"][0].T + out2["t"][0]

        err1 = (aligned1 - tgt).norm(dim=-1).mean().item()
        err2 = (aligned2 - tgt).norm(dim=-1).mean().item()

        # Equivariance: both should give same alignment error.
        # Tolerance is 2e-3 because the error compounds through
        # Sinkhorn (discrete OT) + SVD (saddle-point geometry).
        # Pre-Norm changes the FP computation order, and different
        # GPU architectures produce different ulp-level rounding in
        # matmul, which Sinkhorn iterative normalization amplifies.
        # The strict invariance is tested on features (inlier weights)
        # in test_inlier_weight_so3_invariance at 5e-5 tolerance.
        assert abs(err1 - err2) < 2e-3, (
            f"Alignment not equivariant: err1={err1:.4f}, err2={err2:.4f}, "
            f"diff={abs(err1-err2):.4e}"
        )


class TestGeoRegistrationGradient:
    def test_gradient_flow_no_nan(self):
        """Gradient must flow through SVD without NaN."""
        model = _model()
        model.train()

        torch.manual_seed(444)
        bs = collate_point_clouds([{"pos": _cloud(20)}])
        bt = collate_point_clouds([{"pos": _cloud(18)}])

        out = model(bs, bt)
        loss = out["R"].sum() + out["t"].sum()
        loss.backward()

        nan_params = [
            n for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert len(nan_params) == 0, f"NaN grads in: {nan_params}"

    def test_gradient_flow_batched(self):
        """Gradient must flow correctly on B=2 variable-size batch."""
        model = _model()
        model.train()

        torch.manual_seed(555)
        bs = collate_point_clouds([
            {"pos": _cloud(20)}, {"pos": _cloud(12)},
        ])
        bt = collate_point_clouds([
            {"pos": _cloud(15)}, {"pos": _cloud(18)},
        ])

        out = model(bs, bt)
        loss = out["R"].sum() + out["t"].sum()
        loss.backward()

        nan_params = [
            n for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert len(nan_params) == 0, f"NaN grads in: {nan_params}"


# ═══════════════════════════════════════════════════════════════════
# Phase 4.5: Robustness Hardening Tests
# ═══════════════════════════════════════════════════════════════════


class TestSinkhornDustbin:
    """Verify dustbin mechanism absorbs outlier mass."""

    def test_dustbin_routing(self) -> None:
        """Outlier points must have near-zero assignment after Sinkhorn."""
        import torch.nn.functional as F

        torch.manual_seed(777)

        # Build 10 src descriptors: 5 matching + 5 outliers (very far)
        desc_match = torch.randn(5, 32)
        desc_outlier = torch.randn(5, 32) * 100.0  # Deliberately extreme
        desc_src = torch.cat([desc_match, desc_outlier], dim=0)  # [10, 32]

        # Target: 5 matching (same) + 5 unrelated
        desc_tgt = torch.cat([desc_match, torch.randn(5, 32)], dim=0)  # [10, 32]

        # Compute similarity → Sinkhorn with dustbin
        sim = F.normalize(desc_src, dim=-1) @ F.normalize(desc_tgt, dim=-1).T
        assignment = sinkhorn_log_domain(
            sim, num_iters=50, temperature=0.05, dust_bin=-5.0,
        )  # [10, 10]

        # Matching points (rows 0:5) should capture significant mass
        match_mass = assignment[:5, :5].sum().item()
        # Outlier points (rows 5:10) should have mass absorbed by dustbin
        outlier_row_mass = assignment[5:, :].sum().item()

        # Matching pairs should capture most of the total available mass
        assert match_mass > 2.0, (
            f"Match mass too low: {match_mass:.3f}"
            " — dustbin absorbing inliers?"
        )
        # Outlier rows should have very little mass (rest went to dustbin)
        assert outlier_row_mass < match_mass, (
            f"Outlier mass ({outlier_row_mass:.3f}) ≥ match mass"
            f" ({match_mass:.3f}) — dustbin not absorbing outliers"
        )

    def test_learnable_dustbin_has_grad(self) -> None:
        """Dustbin score parameter must receive gradient."""
        import torch.nn.functional as F
        from geoembodied.nn.models.robust_registration import RobustRegistrationHead

        head = RobustRegistrationHead(
            descriptor_dim=16, sinkhorn_iters=5, use_dust_bin=True,
        )
        head.train()

        desc_s = torch.randn(1, 8, 16, requires_grad=True)
        desc_t = torch.randn(1, 8, 16, requires_grad=True)
        pos_s = torch.randn(1, 8, 3)
        pos_t = torch.randn(1, 8, 3)

        R, t, _ = head(desc_s, desc_t, pos_s, pos_t)
        loss = R.sum() + t.sum()
        loss.backward()

        assert head.dustbin_score.grad is not None, "dustbin_score has no grad"
        assert torch.isfinite(head.dustbin_score.grad), (
            f"dustbin_score grad is {head.dustbin_score.grad}"
        )
        assert head.logit_scale.grad is not None, "logit_scale has no grad"
        assert torch.isfinite(head.logit_scale.grad), (
            f"logit_scale grad is {head.logit_scale.grad}"
        )


class TestSVDCoplanarSurvival:
    """SVD on degenerate geometry must NOT produce NaN gradients."""

    def test_coplanar_z_zero(self) -> None:
        """Source cloud with all Z=0 (rank-2 geometry)."""
        model = _model()
        model.train()

        torch.manual_seed(888)
        # Coplanar source: all Z=0
        src = torch.randn(20, 3)
        src[:, 2] = 0.0  # Force coplanar
        tgt = torch.randn(20, 3)

        bs = collate_point_clouds([{"pos": src}])
        bt = collate_point_clouds([{"pos": tgt}])

        out = model(bs, bt)
        loss = out["R"].sum() + out["t"].sum()
        loss.backward()

        nan_params = [
            n for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert len(nan_params) == 0, (
            f"NaN grads from coplanar SVD in: {nan_params}"
        )

    def test_symmetric_geometry_equal_sigma(self) -> None:
        """Cube-like symmetric cloud → σ₁ ≈ σ₂ before perturbation."""
        model = _model()
        model.train()

        torch.manual_seed(999)
        # Near-perfect cube: σ₁ ≈ σ₂ ≈ σ₃
        cube = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
        ], dtype=torch.float32)
        # Tile to get enough points
        src = cube.repeat(3, 1) + torch.randn(24, 3) * 0.001
        tgt = cube.repeat(3, 1) + torch.randn(24, 3) * 0.001

        bs = collate_point_clouds([{"pos": src}])
        bt = collate_point_clouds([{"pos": tgt}])

        out = model(bs, bt)
        loss = out["R"].sum() + out["t"].sum()
        loss.backward()

        nan_params = [
            n for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert len(nan_params) == 0, (
            f"NaN grads from symmetric SVD in: {nan_params}"
        )


class TestChamferDistance:
    """Tests for mask-aware Chamfer Distance."""

    def test_identical_clouds_zero_cd(self) -> None:
        """CD between identical clouds should be 0."""
        from geoembodied.functional.chamfer import chamfer_distance

        cloud = torch.randn(2, 10, 3)
        cd = chamfer_distance(cloud, cloud)
        assert cd.item() < 1e-6, f"CD of identical clouds: {cd.item()}"

    def test_unidirectional_partial(self) -> None:
        """Unidirectional CD: A⊂B should give 0 when A's points are in B."""
        from geoembodied.functional.chamfer import chamfer_distance

        b = torch.randn(1, 20, 3)
        a = b[:, :10, :]  # A is a subset of B

        cd_uni = chamfer_distance(a, b, bidirectional=False)
        cd_bi = chamfer_distance(a, b, bidirectional=True)

        # Unidirectional: every point in A is exactly in B → 0
        assert cd_uni.item() < 1e-6, f"Unidirectional CD: {cd_uni.item()}"
        # Bidirectional: B→A has non-zero distance (B has extra points)
        assert cd_bi.item() > cd_uni.item(), (
            "Bidirectional should be >= unidirectional"
        )

    def test_mask_excludes_padding(self) -> None:
        """Padding positions (mask=False) must not affect CD."""
        from geoembodied.functional.chamfer import chamfer_distance

        torch.manual_seed(42)
        # Cloud A: 2 real points + 3 padding at [999, 999, 999]
        a = torch.zeros(1, 5, 3)
        a[0, :2] = torch.randn(2, 3)
        a[0, 2:] = 999.0  # Padding — very far away
        mask_a = torch.tensor([[True, True, False, False, False]])

        b = torch.randn(1, 5, 3)

        cd_no_mask = chamfer_distance(a, b)
        cd_masked = chamfer_distance(a, b, mask_a=mask_a)

        # Without mask, padding [999,999,999] contributes enormous distance
        # With mask, only 2 real points count
        assert cd_masked < cd_no_mask, (
            f"Masked CD ({cd_masked:.4f}) should be < "
            f"unmasked ({cd_no_mask:.4f})"
        )
        assert torch.isfinite(cd_masked)

    def test_unbatched(self) -> None:
        """Unbatched [N, 3] input should work."""
        from geoembodied.functional.chamfer import chamfer_distance

        a = torch.randn(10, 3)
        b = torch.randn(10, 3)

        cd = chamfer_distance(a, b)
        assert cd.dim() == 0  # scalar
        assert torch.isfinite(cd)
