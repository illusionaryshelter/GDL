# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for SO(3) and SE(3) operations not covered in test_core.py.

Covers:
    - hat/vee roundtrip
    - so3_act (known axis rotations)
    - so3_adjoint (Ad(R) = R matrix)
    - se3_adjoint compositional property
    - se3_identity
    - SE3 wrapper: rotation/translation properties, to_matrix, from_rot_trans
    - SO3 wrapper: to_matrix/from_matrix, slerp, ltype
    - Batch broadcasting
    - Double precision (float64)
"""

import math

import pytest
import torch
from torch import Tensor


class TestSO3HatVee:
    """Test so3_hat / so3_vee roundtrip."""

    def test_roundtrip(self) -> None:
        """vee(hat(ω)) = ω."""
        from geoembodied.functional.so3_ops import so3_hat, so3_vee

        omega = torch.randn(20, 3)
        skew = so3_hat(omega)
        omega_back = so3_vee(skew)
        assert torch.allclose(omega, omega_back, atol=1e-6)

    def test_skew_symmetric(self) -> None:
        """hat(ω) must be skew-symmetric: S + S^T = 0."""
        from geoembodied.functional.so3_ops import so3_hat

        omega = torch.randn(10, 3)
        S = so3_hat(omega)
        assert torch.allclose(S + S.transpose(-1, -2), torch.zeros_like(S), atol=1e-6)

    def test_hat_shape(self) -> None:
        """hat should produce [..., 3, 3]."""
        from geoembodied.functional.so3_ops import so3_hat
        result = so3_hat(torch.randn(5, 8, 3))
        assert result.shape == (5, 8, 3, 3)


class TestSO3Act:
    """Test so3_act — rotation of known vectors."""

    def test_90deg_z_rotates_x_to_y(self) -> None:
        """90° around z-axis: x̂ → ŷ."""
        from geoembodied.functional.so3_ops import so3_exp, so3_act

        omega = torch.tensor([0.0, 0.0, math.pi / 2])
        q = so3_exp(omega)
        v = torch.tensor([1.0, 0.0, 0.0])
        result = so3_act(q, v)
        assert torch.allclose(result, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

    def test_90deg_x_rotates_y_to_z(self) -> None:
        """90° around x-axis: ŷ → ẑ."""
        from geoembodied.functional.so3_ops import so3_exp, so3_act

        omega = torch.tensor([math.pi / 2, 0.0, 0.0])
        q = so3_exp(omega)
        v = torch.tensor([0.0, 1.0, 0.0])
        result = so3_act(q, v)
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


class TestSO3Adjoint:
    """Test so3_adjoint: Ad(R) = R matrix."""

    def test_adjoint_equals_matrix(self) -> None:
        """For SO(3), Ad(R) is the rotation matrix R itself."""
        from geoembodied.functional.so3_ops import so3_exp, so3_adjoint
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(42)
        omega = torch.randn(10, 3)
        q = so3_exp(omega)
        Ad = so3_adjoint(q)
        R = quaternion_to_matrix(q)
        assert torch.allclose(Ad, R, atol=1e-5)


class TestSE3Adjoint:
    """Test se3_adjoint compositional property."""

    def test_adjoint_composition(self) -> None:
        """Ad(T1 ∘ T2) = Ad(T1) @ Ad(T2)."""
        from geoembodied.functional.se3_ops import se3_exp, se3_multiply, se3_adjoint

        torch.manual_seed(42)
        xi1 = torch.randn(5, 6) * 0.3
        xi2 = torch.randn(5, 6) * 0.3
        T1 = se3_exp(xi1)
        T2 = se3_exp(xi2)
        T12 = se3_multiply(T1, T2)

        Ad_12 = se3_adjoint(T12)              # [..., 6, 6]
        Ad_1 = se3_adjoint(T1)                # [..., 6, 6]
        Ad_2 = se3_adjoint(T2)                # [..., 6, 6]
        Ad_1_Ad_2 = torch.bmm(Ad_1, Ad_2)    # [..., 6, 6]

        assert torch.allclose(Ad_12, Ad_1_Ad_2, atol=1e-4), \
            f"Ad(T1∘T2) ≠ Ad(T1)Ad(T2): max err {(Ad_12 - Ad_1_Ad_2).abs().max():.2e}"


class TestSE3Identity:
    """Test se3_identity functional."""

    def test_identity_values(self) -> None:
        """Identity should be [1,0,0,0, 0,0,0]."""
        from geoembodied.functional.se3_ops import se3_identity

        I = se3_identity()
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(I, expected)

    def test_identity_batch(self) -> None:
        """Batched identity should have correct shape."""
        from geoembodied.functional.se3_ops import se3_identity

        I = se3_identity((3, 4))
        assert I.shape == (3, 4, 7)
        assert torch.allclose(I[..., 0], torch.ones(3, 4))
        assert torch.allclose(I[..., 1:], torch.zeros(3, 4, 6))


class TestSE3LogNearZero:
    """Test SE(3) log near zero twist."""

    def test_log_of_identity(self) -> None:
        """log(I) should be zero twist."""
        from geoembodied.functional.se3_ops import se3_identity, se3_log

        I = se3_identity()
        xi = se3_log(I)
        assert torch.allclose(xi, torch.zeros(6), atol=1e-5)

    def test_pure_translation(self) -> None:
        """log of pure translation should have ω=0, v=t."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        # Pure translation twist: [0,0,0, tx,ty,tz]
        xi = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        T = se3_exp(xi)
        xi_back = se3_log(T)
        assert torch.allclose(xi, xi_back, atol=1e-4), \
            f"Pure translation roundtrip failed: {xi_back}"


class TestSO3WrapperExtras:
    """Test SO3 LieTensor wrapper methods not covered in test_core."""

    def test_ltype(self) -> None:
        """SO3.ltype should return LieType.SO3."""
        from geoembodied.lietensor import SO3
        from geoembodied.lietensor.types import LieType
        R = SO3.identity()
        assert R.ltype == LieType.SO3

    def test_group_dim(self) -> None:
        """SO3.group_dim should be 4."""
        from geoembodied.lietensor import SO3
        assert SO3.group_dim == 4

    def test_tangent_dim(self) -> None:
        """SO3.tangent_dim should be 3."""
        from geoembodied.lietensor import SO3
        assert SO3.tangent_dim == 3

    def test_to_from_matrix_roundtrip(self) -> None:
        """SO3 → matrix → SO3 roundtrip."""
        from geoembodied.lietensor import SO3
        torch.manual_seed(0)
        R = SO3.exp(torch.randn(10, 3))
        M = R.to_matrix()
        R_back = SO3.from_matrix(M)
        dot = (R.as_subclass(Tensor) * R_back.as_subclass(Tensor)).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones(10), atol=1e-4)

    def test_slerp_wrapper(self) -> None:
        """SO3 slerp wrapper should work."""
        from geoembodied.lietensor import SO3
        R0 = SO3.identity()
        R1 = SO3.exp(torch.tensor([0.0, 0.0, math.pi / 2]))
        mid = R0.slerp(R1, torch.tensor(0.5))
        assert isinstance(mid, SO3)
        assert mid.as_subclass(Tensor).norm().item() == pytest.approx(1.0, abs=1e-5)

    def test_log_wrapper(self) -> None:
        """SO3.log should return tensor (not SO3)."""
        from geoembodied.lietensor import SO3
        R = SO3.exp(torch.randn(3))
        omega = R.log()
        assert isinstance(omega, Tensor)
        assert omega.shape == (3,)

    def test_identity_batch(self) -> None:
        """SO3.identity with batch shape."""
        from geoembodied.lietensor import SO3
        R = SO3.identity((5, 3))
        assert R.shape == (5, 3, 4)
        assert torch.allclose(R.as_subclass(Tensor)[..., 0], torch.ones(5, 3))

    def test_project_(self) -> None:
        """project_ should re-normalize quaternion."""
        from geoembodied.lietensor import SO3
        R = SO3(torch.tensor([2.0, 0.0, 0.0, 0.0]))
        R.project_()
        assert R.as_subclass(Tensor).norm().item() == pytest.approx(1.0, abs=1e-6)

    def test_repr(self) -> None:
        """__repr__ should include class name."""
        from geoembodied.lietensor import SO3
        R = SO3.identity()
        r = repr(R)
        assert "SO3" in r


class TestSE3WrapperExtras:
    """Test SE3 LieTensor wrapper methods not covered in test_core."""

    def test_ltype(self) -> None:
        """SE3.ltype should return LieType.SE3."""
        from geoembodied.lietensor import SE3
        from geoembodied.lietensor.types import LieType
        T = SE3.identity()
        assert T.ltype == LieType.SE3

    def test_group_dim(self) -> None:
        from geoembodied.lietensor import SE3
        assert SE3.group_dim == 7

    def test_tangent_dim(self) -> None:
        from geoembodied.lietensor import SE3
        assert SE3.tangent_dim == 6

    def test_rotation_property(self) -> None:
        """SE3.rotation should return SO3."""
        from geoembodied.lietensor import SE3, SO3
        T = SE3.exp(torch.randn(6) * 0.3)
        R = T.rotation
        assert isinstance(R, SO3)
        assert R.shape == (4,)

    def test_translation_property(self) -> None:
        """SE3.translation should return [..., 3]."""
        from geoembodied.lietensor import SE3
        T = SE3.exp(torch.randn(5, 6) * 0.3)
        t = T.translation
        assert t.shape == (5, 3)

    def test_from_rotation_translation(self) -> None:
        """SE3.from_rotation_translation should compose correctly."""
        from geoembodied.lietensor import SE3, SO3

        R = SO3.exp(torch.tensor([0.1, -0.2, 0.3]))
        t = torch.tensor([1.0, 2.0, 3.0])
        T = SE3.from_rotation_translation(R, t)

        assert isinstance(T, SE3)
        assert T.shape == (7,)
        assert torch.allclose(T.translation, t, atol=1e-6)
        # Quaternion part should match
        dot = (T.rotation.as_subclass(Tensor) * R.as_subclass(Tensor)).sum().abs()
        assert dot.item() == pytest.approx(1.0, abs=1e-5)

    def test_to_matrix(self) -> None:
        """SE3.to_matrix should produce valid 4×4 homogeneous matrix."""
        from geoembodied.lietensor import SE3

        T = SE3.exp(torch.randn(6) * 0.3)
        M = T.to_matrix()
        assert M.shape == (4, 4)

        # Bottom row should be [0, 0, 0, 1]
        assert torch.allclose(M[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)

        # Upper 3×3 should be orthogonal
        R = M[:3, :3]
        RtR = R.T @ R
        assert torch.allclose(RtR, torch.eye(3), atol=1e-5)

    def test_to_matrix_batch(self) -> None:
        """Batched to_matrix."""
        from geoembodied.lietensor import SE3
        T = SE3.exp(torch.randn(5, 6) * 0.3)
        M = T.to_matrix()
        assert M.shape == (5, 4, 4)

    def test_to_matrix_act_consistency(self) -> None:
        """T @ points should match to_matrix() @ points_homo."""
        from geoembodied.lietensor import SE3

        torch.manual_seed(42)
        T = SE3.exp(torch.randn(6) * 0.5)
        points = torch.randn(50, 3)

        # Method 1: LieTensor act
        result1 = T @ points

        # Method 2: 4×4 matrix multiply
        M = T.to_matrix()
        points_homo = torch.cat([points, torch.ones(50, 1)], dim=-1)
        result2 = (M @ points_homo.T).T[:, :3]

        assert torch.allclose(result1, result2, atol=1e-5)

    def test_identity_batch(self) -> None:
        """SE3.identity with batch shape."""
        from geoembodied.lietensor import SE3
        T = SE3.identity((2, 3))
        assert T.shape == (2, 3, 7)

    def test_project_(self) -> None:
        """project_ should re-normalize quaternion part."""
        from geoembodied.lietensor import SE3
        data = torch.tensor([2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        T = SE3(data)
        T.project_()
        q_norm = T.as_subclass(Tensor)[:4].norm()
        assert q_norm.item() == pytest.approx(1.0, abs=1e-6)
        # Translation should be unchanged
        assert torch.allclose(T.as_subclass(Tensor)[4:], torch.tensor([1.0, 2.0, 3.0]))

    def test_repr(self) -> None:
        from geoembodied.lietensor import SE3
        T = SE3.identity()
        assert "SE3" in repr(T)

    def test_log_wrapper(self) -> None:
        """SE3.log should return 6-dim twist."""
        from geoembodied.lietensor import SE3
        T = SE3.exp(torch.randn(6) * 0.3)
        xi = T.log()
        assert isinstance(xi, Tensor)
        assert xi.shape == (6,)

    def test_adjoint_wrapper(self) -> None:
        """SE3.adjoint should return 6×6 matrix."""
        from geoembodied.lietensor import SE3
        T = SE3.exp(torch.randn(6) * 0.3)
        Ad = T.adjoint()
        assert Ad.shape == (6, 6)


class TestLieTensorInterceptionExtras:
    """Test additional illegal operations are blocked."""

    def test_so3_sub_blocked(self) -> None:
        """SO3 - SO3 must raise TypeError."""
        from geoembodied.lietensor import SO3
        R1 = SO3.identity()
        R2 = SO3.identity()
        with pytest.raises(TypeError, match="Cannot apply"):
            _ = R1 - R2

    def test_se3_sub_blocked(self) -> None:
        """SE3 - SE3 must raise TypeError."""
        from geoembodied.lietensor import SE3
        T1 = SE3.identity()
        T2 = SE3.identity()
        with pytest.raises(TypeError, match="Cannot apply"):
            _ = T1 - T2

    def test_radd_blocked(self) -> None:
        """1 + SO3 must raise TypeError."""
        from geoembodied.lietensor import SO3
        R = SO3.identity()
        with pytest.raises(TypeError):
            _ = 1 + R


class TestLieTensorParameter:
    """Test the .parameter() factory."""

    def test_is_leaf(self) -> None:
        """parameter() should create leaf tensor."""
        from geoembodied.lietensor import SO3
        R = SO3.exp(torch.randn(3)).parameter()
        assert R.is_leaf

    def test_requires_grad(self) -> None:
        """parameter() should have requires_grad=True."""
        from geoembodied.lietensor import SE3
        T = SE3.exp(torch.randn(6)).parameter()
        assert T.requires_grad

    def test_detached_from_original(self) -> None:
        """parameter() should be independent of the original."""
        from geoembodied.lietensor import SO3
        omega = torch.randn(3, requires_grad=True)
        R = SO3.exp(omega)
        R_param = R.parameter()
        # Modifying R_param should not affect omega graph
        assert R_param.grad_fn is None


class TestDoublePrecision:
    """Test float64 support."""

    def test_so3_exp_float64(self) -> None:
        """SO(3) exp should work in float64."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.randn(10, 3, dtype=torch.float64)
        q = so3_exp(omega)
        assert q.dtype == torch.float64
        omega_back = so3_log(q)
        q_back = so3_exp(omega_back)
        dot = (q * q_back).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones(10, dtype=torch.float64), atol=1e-10)

    def test_se3_exp_float64(self) -> None:
        """SE(3) exp should work in float64."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        xi = torch.randn(10, 6, dtype=torch.float64) * 0.5
        T = se3_exp(xi)
        assert T.dtype == torch.float64
        xi_back = se3_log(T)
        T_back = se3_exp(xi_back)
        assert torch.allclose(T[..., :4].abs(), T_back[..., :4].abs(), atol=1e-8)


class TestDistanceExtras:
    """Additional distance metric tests."""

    def test_se3_chordal_identity_zero(self) -> None:
        """SE(3) chordal distance to self = 0."""
        from geoembodied.functional.se3_ops import se3_exp
        from geoembodied.functional.distance import se3_chordal_distance

        T = se3_exp(torch.randn(10, 6) * 0.5)
        d = se3_chordal_distance(T, T)
        assert torch.allclose(d, torch.zeros(10), atol=1e-5)

    def test_se3_chordal_symmetry(self) -> None:
        """d(T1, T2) = d(T2, T1)."""
        from geoembodied.functional.se3_ops import se3_exp
        from geoembodied.functional.distance import se3_chordal_distance

        torch.manual_seed(0)
        T1 = se3_exp(torch.randn(10, 6) * 0.5)
        T2 = se3_exp(torch.randn(10, 6) * 0.5)
        d12 = se3_chordal_distance(T1, T2)
        d21 = se3_chordal_distance(T2, T1)
        assert torch.allclose(d12, d21, atol=1e-6)

    def test_so3_geodesic_identity_zero(self) -> None:
        """SO(3) geodesic distance to self = 0."""
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.distance import so3_geodesic_distance

        q = so3_exp(torch.randn(10, 3))
        d = so3_geodesic_distance(q, q)
        assert torch.allclose(d, torch.zeros(10), atol=5e-4)

    def test_so3_chordal_symmetry(self) -> None:
        """d(R1, R2) = d(R2, R1)."""
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.distance import so3_chordal_distance

        q1 = so3_exp(torch.randn(20, 3))
        q2 = so3_exp(torch.randn(20, 3))
        d12 = so3_chordal_distance(q1, q2)
        d21 = so3_chordal_distance(q2, q1)
        assert torch.allclose(d12, d21, atol=1e-6)

    def test_chordal_nonnegative(self) -> None:
        """Chordal distance should always be ≥ 0."""
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.distance import so3_chordal_distance

        q1 = so3_exp(torch.randn(50, 3))
        q2 = so3_exp(torch.randn(50, 3))
        d = so3_chordal_distance(q1, q2)
        assert (d >= -1e-6).all()

    def test_se3_chordal_weights(self) -> None:
        """SE(3) chordal distance weights should scale components."""
        from geoembodied.functional.se3_ops import se3_exp
        from geoembodied.functional.distance import se3_chordal_distance

        T1 = se3_exp(torch.randn(5, 6) * 0.5)
        T2 = se3_exp(torch.randn(5, 6) * 0.5)

        d_equal = se3_chordal_distance(T1, T2, w_rot=1.0, w_trans=1.0)
        d_rot_only = se3_chordal_distance(T1, T2, w_rot=1.0, w_trans=0.0)
        d_trans_only = se3_chordal_distance(T1, T2, w_rot=0.0, w_trans=1.0)

        assert torch.allclose(d_rot_only + d_trans_only, d_equal, atol=1e-5)


class TestNumericalEdgeCases:
    """Test numerical stability at extreme values."""

    def test_taylor_cos_over_theta_near_zero(self) -> None:
        """(1-cos θ)/θ² should be ≈ 0.5 near θ=0."""
        from geoembodied.functional.numeric_safe import taylor_cos_over_theta

        # Very small values should use Taylor and be ≈ 0.5
        theta_small = torch.tensor([0.0, 1e-8, 1e-6])
        result_small = taylor_cos_over_theta(theta_small)
        assert torch.allclose(result_small, 0.5 * torch.ones_like(result_small), atol=1e-5)
        assert not torch.isnan(result_small).any()

        # Larger value should use numerical path and still be finite
        theta_large = torch.tensor([1e-3, 0.1, 1.0])
        result_large = taylor_cos_over_theta(theta_large)
        assert not torch.isnan(result_large).any()
        assert (result_large > 0).all()

    def test_so3_exp_gradient_at_various_scales(self) -> None:
        """exp map gradient should be finite at all scales."""
        from geoembodied.functional.so3_ops import so3_exp

        for scale in [1e-8, 1e-5, 1e-3, 0.1, 1.0, math.pi - 0.01]:
            omega = torch.tensor([scale, 0.0, 0.0], requires_grad=True)
            q = so3_exp(omega)
            loss = q.sum()
            loss.backward()
            assert omega.grad is not None, f"No grad at scale {scale}"
            assert not torch.isnan(omega.grad).any(), f"NaN grad at scale {scale}"
            assert not torch.isinf(omega.grad).any(), f"Inf grad at scale {scale}"

    def test_se3_exp_gradient_at_various_scales(self) -> None:
        """SE(3) exp map gradient should be finite at all scales."""
        from geoembodied.functional.se3_ops import se3_exp

        for scale in [1e-8, 1e-5, 1e-3, 0.1, 1.0]:
            xi = torch.tensor([scale, 0, 0, 1.0, 0, 0], requires_grad=True)
            T = se3_exp(xi)
            loss = T.sum()
            loss.backward()
            assert not torch.isnan(xi.grad).any(), f"NaN grad at scale {scale}"

    def test_distance_gradient_at_identity(self) -> None:
        """Chordal distance gradient at d=0 (same point) should be finite."""
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.distance import so3_chordal_distance

        omega = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
        q = so3_exp(omega)
        d = so3_chordal_distance(q, q.detach())
        d.backward()
        assert not torch.isnan(omega.grad).any()


class TestBatchBroadcasting:
    """Test that all ops handle batch dimensions correctly."""

    def test_so3_exp_single(self) -> None:
        """Single element (no batch)."""
        from geoembodied.functional.so3_ops import so3_exp
        q = so3_exp(torch.randn(3))
        assert q.shape == (4,)

    def test_so3_exp_1d_batch(self) -> None:
        """1D batch."""
        from geoembodied.functional.so3_ops import so3_exp
        q = so3_exp(torch.randn(10, 3))
        assert q.shape == (10, 4)

    def test_so3_exp_2d_batch(self) -> None:
        """2D batch."""
        from geoembodied.functional.so3_ops import so3_exp
        q = so3_exp(torch.randn(5, 8, 3))
        assert q.shape == (5, 8, 4)

    def test_se3_exp_shapes(self) -> None:
        """SE(3) exp with various batch shapes."""
        from geoembodied.functional.se3_ops import se3_exp
        assert se3_exp(torch.randn(6)).shape == (7,)
        assert se3_exp(torch.randn(10, 6)).shape == (10, 7)
        assert se3_exp(torch.randn(3, 5, 6)).shape == (3, 5, 7)

    def test_se3_act_broadcasting(self) -> None:
        """SE(3) act with different batch configs."""
        from geoembodied.functional.se3_ops import se3_exp, se3_act

        T = se3_exp(torch.randn(5, 6))
        pts = torch.randn(5, 100, 3)
        result = se3_act(T, pts)
        assert result.shape == (5, 100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
