# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Core tests for SO(3) and SE(3) Lie group operations.

Verifies:
    1. exp/log round-trip consistency
    2. Group axioms (closure, associativity, identity, inverse)
    3. Rigid body transform correctness (vs matrix multiplication)
    4. Numerical safety at singularities (θ→0, θ→π)
    5. LieTensor illegal operation interception
    6. SE(3) 7-dim compact storage correctness
"""

import pytest
import torch
from torch import Tensor

# ── Layer 0: Pure functional tests ──────────────────────────────


class TestNumericSafety:
    """Test numerically safe operations at singularities."""

    def test_safe_acos_at_boundaries(self) -> None:
        """safe_acos should not produce NaN at ±1."""
        from geoembodied.functional.numeric_safe import safe_acos

        x = torch.tensor([-1.0, -0.99999, 0.0, 0.99999, 1.0])
        result = safe_acos(x)
        assert not torch.isnan(result).any(), "NaN in safe_acos output"
        assert not torch.isinf(result).any(), "Inf in safe_acos output"

    def test_safe_acos_grad_at_boundaries(self) -> None:
        """Gradient of safe_acos should not be NaN/Inf at ±1."""
        from geoembodied.functional.numeric_safe import safe_acos

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        y = safe_acos(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), "NaN gradient in safe_acos"
        assert not torch.isinf(x.grad).any(), "Inf gradient in safe_acos"

    def test_taylor_sinc_near_zero(self) -> None:
        """sin(θ)/θ should be ≈ 1 near θ=0 via Taylor expansion."""
        from geoembodied.functional.numeric_safe import taylor_sinc

        theta = torch.tensor([0.0, 1e-8, 1e-5, 1e-3])
        result = taylor_sinc(theta)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-4)
        assert not torch.isnan(result).any()

    def test_safe_sqrt_near_zero(self) -> None:
        """safe_sqrt should not produce NaN gradient at 0."""
        from geoembodied.functional.numeric_safe import safe_sqrt

        x = torch.tensor([0.0, 1e-10, 1.0], requires_grad=True)
        y = safe_sqrt(x).sum()
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestSO3Functional:
    """Test SO(3) pure functional operations."""

    def test_exp_log_roundtrip(self) -> None:
        """exp(log(R)) ≈ R for random rotations."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        torch.manual_seed(42)
        omega = torch.randn(100, 3) * 2.0  # up to ~2 radians
        q = so3_exp(omega)
        omega_recovered = so3_log(q)
        q_recovered = so3_exp(omega_recovered)

        # Compare quaternions (up to sign ambiguity)
        dot = (q * q_recovered).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5), \
            f"exp/log roundtrip failed, max error: {(1 - dot).max():.2e}"

    def test_exp_log_roundtrip_near_zero(self) -> None:
        """exp/log roundtrip for very small rotations (θ→0)."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.tensor([[1e-8, 0, 0], [0, 1e-10, 0], [0, 0, 1e-12]])
        q = so3_exp(omega)
        omega_back = so3_log(q)
        assert torch.allclose(omega, omega_back, atol=1e-6), \
            f"Near-zero roundtrip failed: {(omega - omega_back).abs().max():.2e}"

    def test_exp_log_roundtrip_near_pi(self) -> None:
        """exp/log roundtrip for near-π rotations."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log
        import math

        # Rotation close to π around x-axis
        omega = torch.tensor([[math.pi - 1e-4, 0, 0]])
        q = so3_exp(omega)
        omega_back = so3_log(q)
        q_back = so3_exp(omega_back)

        dot = (q * q_back).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones_like(dot), atol=1e-4)

    def test_group_identity(self) -> None:
        """R @ I = R and I @ R = R."""
        from geoembodied.functional.so3_ops import so3_exp, so3_multiply

        torch.manual_seed(0)
        omega = torch.randn(10, 3)
        q = so3_exp(omega)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)

        # Right identity
        q_ri = so3_multiply(q, identity)
        dot_ri = (q * q_ri).sum(dim=-1).abs()
        assert torch.allclose(dot_ri, torch.ones_like(dot_ri), atol=1e-6)

        # Left identity
        q_li = so3_multiply(identity, q)
        dot_li = (q * q_li).sum(dim=-1).abs()
        assert torch.allclose(dot_li, torch.ones_like(dot_li), atol=1e-6)

    def test_group_inverse(self) -> None:
        """R @ R⁻¹ ≈ I."""
        from geoembodied.functional.so3_ops import so3_exp, so3_multiply, so3_inverse

        torch.manual_seed(1)
        omega = torch.randn(50, 3)
        q = so3_exp(omega)
        q_inv = so3_inverse(q)
        product = so3_multiply(q, q_inv)

        # Should be close to identity [1, 0, 0, 0]
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(product)
        dot = (product * identity).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)

    def test_group_associativity(self) -> None:
        """(A @ B) @ C ≈ A @ (B @ C)."""
        from geoembodied.functional.so3_ops import so3_exp, so3_multiply

        torch.manual_seed(2)
        a = so3_exp(torch.randn(20, 3))
        b = so3_exp(torch.randn(20, 3))
        c = so3_exp(torch.randn(20, 3))

        ab_c = so3_multiply(so3_multiply(a, b), c)
        a_bc = so3_multiply(a, so3_multiply(b, c))

        dot = (ab_c * a_bc).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)


class TestSE3Functional:
    """Test SE(3) pure functional operations."""

    def test_exp_log_roundtrip(self) -> None:
        """exp(log(T)) ≈ T."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        torch.manual_seed(42)
        xi = torch.randn(100, 6) * 0.5
        T = se3_exp(xi)
        xi_back = se3_log(T)
        T_back = se3_exp(xi_back)

        # Compare rotation (quaternion dot)
        q_dot = (T[..., :4] * T_back[..., :4]).sum(dim=-1).abs()
        assert torch.allclose(q_dot, torch.ones_like(q_dot), atol=1e-4), \
            f"SE3 rotation roundtrip failed: max error {(1 - q_dot).max():.2e}"

        # Compare translation
        t_err = (T[..., 4:] - T_back[..., 4:]).norm(dim=-1)
        assert (t_err < 1e-4).all(), f"SE3 translation roundtrip failed: max error {t_err.max():.2e}"

    def test_multiply_inverse(self) -> None:
        """T @ T⁻¹ ≈ I."""
        from geoembodied.functional.se3_ops import se3_exp, se3_multiply, se3_inverse

        torch.manual_seed(3)
        xi = torch.randn(30, 6) * 0.5
        T = se3_exp(xi)
        T_inv = se3_inverse(T)
        product = se3_multiply(T, T_inv)

        # Rotation should be identity
        q_dot = product[..., 0].abs()  # qw should be ≈ 1
        assert torch.allclose(q_dot, torch.ones_like(q_dot), atol=1e-5)

        # Translation should be zero
        t_norm = product[..., 4:].norm(dim=-1)
        assert (t_norm < 1e-5).all()

    def test_act_vs_matrix(self) -> None:
        """SE3.act(points) should match 4×4 matrix multiplication."""
        from geoembodied.functional.se3_ops import se3_exp, se3_act
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(4)
        xi = torch.randn(5, 6) * 0.5
        T = se3_exp(xi)
        points = torch.randn(5, 100, 3)

        # Method 1: functional act
        result_act = se3_act(T, points)

        # Method 2: explicit matrix multiplication
        R = quaternion_to_matrix(T[..., :4])  # [5, 3, 3]
        t = T[..., 4:]                        # [5, 3]
        result_matrix = torch.einsum('bij,bnj->bni', R, points) + t.unsqueeze(1)

        assert torch.allclose(result_act, result_matrix, atol=1e-5), \
            f"act vs matrix mismatch: max error {(result_act - result_matrix).abs().max():.2e}"

    def test_7dim_storage_size(self) -> None:
        """Verify 7-dim compact form uses less memory than 4×4 matrix."""
        from geoembodied.functional.se3_ops import se3_exp

        xi = torch.randn(1000, 6)
        T_compact = se3_exp(xi)  # [1000, 7]
        T_matrix = torch.randn(1000, 4, 4)  # hypothetical matrix form

        compact_bytes = T_compact.nelement() * T_compact.element_size()
        matrix_bytes = T_matrix.nelement() * T_matrix.element_size()

        # 7/16 = 43.75% of matrix storage
        ratio = compact_bytes / matrix_bytes
        assert ratio < 0.5, f"Compact form should use <50% of matrix memory, got {ratio:.1%}"


class TestSE3Associativity:
    """Test SE(3) group associativity."""

    def test_associativity(self) -> None:
        """(A ∘ B) ∘ C ≈ A ∘ (B ∘ C)."""
        from geoembodied.functional.se3_ops import se3_exp, se3_multiply

        torch.manual_seed(5)
        a = se3_exp(torch.randn(20, 6) * 0.5)
        b = se3_exp(torch.randn(20, 6) * 0.5)
        c = se3_exp(torch.randn(20, 6) * 0.5)

        ab_c = se3_multiply(se3_multiply(a, b), c)
        a_bc = se3_multiply(a, se3_multiply(b, c))

        assert torch.allclose(ab_c, a_bc, atol=1e-4), \
            f"SE3 associativity failed: max error {(ab_c - a_bc).abs().max():.2e}"


# ── Layer 1: LieTensor wrapper tests ────────────────────────────


class TestLieTensorInterception:
    """Test that illegal operations are blocked."""

    def test_so3_add_blocked(self) -> None:
        """SO3 + SO3 must raise TypeError."""
        from geoembodied.lietensor import SO3

        R1 = SO3.identity()
        R2 = SO3.identity()

        with pytest.raises(TypeError, match="Cannot apply"):
            _ = R1 + R2

    def test_se3_add_blocked(self) -> None:
        """SE3 + SE3 must raise TypeError."""
        from geoembodied.lietensor import SE3

        T1 = SE3.identity()
        T2 = SE3.identity()

        with pytest.raises(TypeError, match="Cannot apply"):
            _ = T1 + T2


class TestLieTensorMatmul:
    """Test @ operator for group operations."""

    def test_so3_matmul_composition(self) -> None:
        """R1 @ R2 should compose rotations."""
        from geoembodied.lietensor import SO3

        torch.manual_seed(10)
        R1 = SO3.exp(torch.randn(5, 3))
        R2 = SO3.exp(torch.randn(5, 3))
        R3 = R1 @ R2
        assert isinstance(R3, SO3)
        assert R3.shape == (5, 4)

    def test_so3_matmul_points(self) -> None:
        """R @ points should rotate points."""
        from geoembodied.lietensor import SO3

        R = SO3.exp(torch.tensor([[0.0, 0.0, 3.14159 / 2]]))  # 90° around z
        points = torch.tensor([[1.0, 0.0, 0.0]])  # unit x
        rotated = R @ points
        # Should be approximately [0, 1, 0]
        assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-3)

    def test_se3_matmul_composition(self) -> None:
        """T1 @ T2 should compose transforms."""
        from geoembodied.lietensor import SE3

        torch.manual_seed(11)
        T1 = SE3.exp(torch.randn(5, 6) * 0.3)
        T2 = SE3.exp(torch.randn(5, 6) * 0.3)
        T3 = T1 @ T2
        assert isinstance(T3, SE3)
        assert T3.shape == (5, 7)

    def test_se3_matmul_points(self) -> None:
        """T @ points should rigidly transform points."""
        from geoembodied.lietensor import SE3

        # Pure translation [0,0,0, 1,2,3]
        T = SE3(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))
        points = torch.tensor([[0.0, 0.0, 0.0]])
        result = T @ points
        assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-6)

    def test_se3_inverse_roundtrip(self) -> None:
        """T @ T.inverse() ≈ Identity."""
        from geoembodied.lietensor import SE3

        torch.manual_seed(12)
        T = SE3.exp(torch.randn(10, 6) * 0.5)
        product = T @ T.inverse()

        # qw should be ≈ 1, rest ≈ 0
        assert torch.allclose(
            product.as_subclass(Tensor),
            SE3.identity((10,)).as_subclass(Tensor),
            atol=1e-5,
        )


class TestDistanceMetrics:
    """Test chordal and geodesic distance functions."""

    def test_chordal_identity_zero(self) -> None:
        """Distance to self should be zero."""
        from geoembodied.functional.distance import so3_chordal_distance
        from geoembodied.functional.so3_ops import so3_exp

        q = so3_exp(torch.randn(10, 3))
        d = so3_chordal_distance(q, q)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)

    def test_chordal_vs_geodesic_small(self) -> None:
        """For small perturbations, chordal ∝ geodesic²."""
        from geoembodied.functional.distance import so3_chordal_distance, so3_geodesic_distance
        from geoembodied.functional.so3_ops import so3_exp

        torch.manual_seed(20)
        q1 = so3_exp(torch.randn(50, 3) * 0.01)
        q2 = so3_exp(torch.randn(50, 3) * 0.01)

        d_chordal = so3_chordal_distance(q1, q2)
        d_geodesic = so3_geodesic_distance(q1, q2)

        # For small angles: chordal ≈ geodesic² (up to constant)
        # Check correlation
        correlation = torch.corrcoef(
            torch.stack([d_chordal, d_geodesic ** 2])
        )[0, 1]
        assert correlation > 0.99, f"Chordal/geodesic correlation too low: {correlation:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
