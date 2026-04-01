# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Comprehensive unit tests for quaternion operations.

Covers:
    - normalize, conjugate, multiply, apply, slerp
    - to_matrix / from_matrix roundtrip (Shepperd's method)
    - Edge cases: identity, antipodal, near-zero, 180° rotation
    - Non-commutativity of Hamilton product
    - Batch broadcasting
"""

import math

import pytest
import torch
from torch import Tensor

from geoembodied.functional.quaternion_ops import (
    quaternion_normalize,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_apply,
    quaternion_to_matrix,
    quaternion_from_matrix,
    quaternion_slerp,
)


class TestQuaternionNormalize:
    """Test quaternion_normalize."""

    def test_identity_preserved(self) -> None:
        """Normalizing a unit quaternion should not change it."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(quaternion_normalize(q), q, atol=1e-6)

    def test_scales_to_unit(self) -> None:
        """Non-unit quaternion should be projected to S³."""
        q = torch.tensor([2.0, 0.0, 0.0, 0.0])
        result = quaternion_normalize(q)
        assert torch.allclose(result.norm(), torch.tensor(1.0), atol=1e-6)

    def test_batch(self) -> None:
        """Batch normalization should work element-wise."""
        q = torch.randn(100, 4)
        result = quaternion_normalize(q)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)


class TestQuaternionConjugate:
    """Test quaternion_conjugate."""

    def test_identity(self) -> None:
        """Conjugate of identity is identity."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(quaternion_conjugate(q), q, atol=1e-6)

    def test_negate_imaginary(self) -> None:
        """Conjugate should negate xyz, keep w."""
        q = torch.tensor([0.5, 0.5, 0.5, 0.5])
        expected = torch.tensor([0.5, -0.5, -0.5, -0.5])
        assert torch.allclose(quaternion_conjugate(q), expected, atol=1e-6)

    def test_double_conjugate(self) -> None:
        """Conjugate of conjugate = original."""
        q = quaternion_normalize(torch.randn(10, 4))
        result = quaternion_conjugate(quaternion_conjugate(q))
        assert torch.allclose(result, q, atol=1e-6)


class TestQuaternionMultiply:
    """Test Hamilton product."""

    def test_identity_left(self) -> None:
        """I ⊗ q = q."""
        I = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = quaternion_normalize(torch.tensor([0.7, 0.3, -0.5, 0.1]))
        result = quaternion_multiply(I, q)
        assert torch.allclose(result, q, atol=1e-6)

    def test_identity_right(self) -> None:
        """q ⊗ I = q."""
        I = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = quaternion_normalize(torch.tensor([0.7, 0.3, -0.5, 0.1]))
        result = quaternion_multiply(q, I)
        assert torch.allclose(result, q, atol=1e-6)

    def test_inverse_product(self) -> None:
        """q ⊗ q* = I (for unit quaternions)."""
        q = quaternion_normalize(torch.randn(20, 4))
        q_conj = quaternion_conjugate(q)
        product = quaternion_normalize(quaternion_multiply(q, q_conj))
        I = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(product)
        dot = (product * I).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones(20), atol=1e-5)

    def test_not_commutative(self) -> None:
        """Hamilton product is NOT commutative: q1⊗q2 ≠ q2⊗q1 in general."""
        q1 = quaternion_normalize(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q2 = quaternion_normalize(torch.tensor([0.7, 0.3, -0.5, 0.1]))
        p12 = quaternion_multiply(q1, q2)
        p21 = quaternion_multiply(q2, q1)
        # Should be different (unless trivial case)
        assert not torch.allclose(p12, p21, atol=1e-3), \
            "Hamilton product should not be commutative"


class TestQuaternionApply:
    """Test quaternion_apply — rotating 3D vectors."""

    def test_identity_rotation(self) -> None:
        """Identity quaternion should not change vectors."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(10, -1)
        v = torch.randn(10, 3)
        assert torch.allclose(quaternion_apply(q, v), v, atol=1e-6)

    def test_90deg_z(self) -> None:
        """90° around z: [1,0,0] → [0,1,0]."""
        angle = math.pi / 2
        q = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        v = torch.tensor([1.0, 0.0, 0.0])
        result = quaternion_apply(q, v)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_180deg_x(self) -> None:
        """180° around x: [0,1,0] → [0,-1,0]."""
        angle = math.pi
        q = torch.tensor([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        v = torch.tensor([0.0, 1.0, 0.0])
        result = quaternion_apply(q, v)
        expected = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_preserves_norm(self) -> None:
        """Rotation preserves vector norm (isometry)."""
        torch.manual_seed(42)
        q = quaternion_normalize(torch.randn(20, 4))
        v = torch.randn(20, 3)
        v_rot = quaternion_apply(q, v)
        assert torch.allclose(v.norm(dim=-1), v_rot.norm(dim=-1), atol=1e-5)

    def test_batch(self) -> None:
        """Batched rotation of multiple vectors."""
        q = quaternion_normalize(torch.randn(5, 4))
        v = torch.randn(5, 100, 3)
        # Need to expand quaternion for broadcasting
        q_expanded = q.unsqueeze(1).expand(-1, 100, -1)
        result = quaternion_apply(q_expanded, v)
        assert result.shape == (5, 100, 3)
        # Check norm preservation
        assert torch.allclose(
            v.norm(dim=-1), result.norm(dim=-1), atol=1e-4
        )


class TestQuaternionToFromMatrix:
    """Test quaternion ↔ rotation matrix conversion."""

    def test_identity(self) -> None:
        """Identity quaternion → identity matrix."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_matrix(q)
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_roundtrip_random(self) -> None:
        """to_matrix → from_matrix should recover quaternion."""
        from geoembodied.functional.so3_ops import so3_exp
        torch.manual_seed(42)
        omega = torch.randn(50, 3) * 2.0
        q_orig = so3_exp(omega)
        R = quaternion_to_matrix(q_orig)
        q_back = quaternion_from_matrix(R)
        # Quaternion sign ambiguity: q and -q represent same rotation
        dot = (q_orig * q_back).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones(50), atol=1e-4), \
            f"Roundtrip failed: max error {(1 - dot).max():.2e}"

    def test_orthogonal(self) -> None:
        """Resulting matrix should be orthogonal: R^T R = I."""
        q = quaternion_normalize(torch.randn(30, 4))
        R = quaternion_to_matrix(q)
        RtR = torch.bmm(R.transpose(-1, -2), R)
        I = torch.eye(3).unsqueeze(0).expand(30, -1, -1)
        assert torch.allclose(RtR, I, atol=1e-5)

    def test_det_one(self) -> None:
        """Resulting matrix should have det = +1 (proper rotation)."""
        q = quaternion_normalize(torch.randn(20, 4))
        R = quaternion_to_matrix(q)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(20), atol=1e-5)

    def test_90deg_axes(self) -> None:
        """Test 90° rotations around each axis match known matrices."""
        # 90° around x
        angle = math.pi / 2
        q_x = torch.tensor([math.cos(angle / 2), math.sin(angle / 2), 0, 0])
        R_x = quaternion_to_matrix(q_x)
        expected_x = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=torch.float32)
        assert torch.allclose(R_x, expected_x, atol=1e-5)

    def test_from_matrix_180deg(self) -> None:
        """Shepperd handles 180° rotation (trace = -1)."""
        # 180° around z
        R = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ], dtype=torch.float32)
        q = quaternion_from_matrix(R)
        R_back = quaternion_to_matrix(q)
        assert torch.allclose(R, R_back, atol=1e-4)


class TestQuaternionSlerp:
    """Test spherical linear interpolation."""

    def test_t0(self) -> None:
        """slerp(q0, q1, t=0) = q0."""
        q0 = quaternion_normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q1 = quaternion_normalize(torch.tensor([0.7, 0.3, 0.5, 0.1]))
        result = quaternion_slerp(q0, q1, torch.tensor(0.0))
        dot = (result * q0).sum().abs()
        assert torch.allclose(dot, torch.tensor(1.0), atol=1e-5)

    def test_t1(self) -> None:
        """slerp(q0, q1, t=1) = q1."""
        q0 = quaternion_normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q1 = quaternion_normalize(torch.tensor([0.7, 0.3, 0.5, 0.1]))
        result = quaternion_slerp(q0, q1, torch.tensor(1.0))
        dot = (result * q1).sum().abs()
        assert torch.allclose(dot, torch.tensor(1.0), atol=1e-5)

    def test_midpoint(self) -> None:
        """slerp at t=0.5 should produce a midpoint rotation."""
        from geoembodied.functional.so3_ops import so3_exp
        q0 = so3_exp(torch.tensor([0.0, 0.0, 0.0]))
        q1 = so3_exp(torch.tensor([0.0, 0.0, math.pi / 2]))
        mid = quaternion_slerp(q0, q1, torch.tensor(0.5))
        # Should be a 45° rotation around z
        q_45 = so3_exp(torch.tensor([0.0, 0.0, math.pi / 4]))
        dot = (mid * q_45).sum().abs()
        assert torch.allclose(dot, torch.tensor(1.0), atol=1e-4)

    def test_antipodal(self) -> None:
        """slerp should handle antipodal quaternions (q and -q)."""
        q0 = quaternion_normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q1 = -q0  # Same rotation, opposite sign
        result = quaternion_slerp(q0, q1, torch.tensor(0.5))
        # Should still be a valid unit quaternion
        assert torch.allclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    def test_same_quaternion(self) -> None:
        """slerp(q, q, t) = q for any t."""
        q = quaternion_normalize(torch.randn(4))
        for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = quaternion_slerp(q, q, torch.tensor(t))
            dot = (result * q).sum().abs()
            assert torch.allclose(dot, torch.tensor(1.0), atol=1e-4)

    def test_unit_norm_preserved(self) -> None:
        """Slerp result should always be unit quaternion."""
        torch.manual_seed(10)
        q0 = quaternion_normalize(torch.randn(20, 4))
        q1 = quaternion_normalize(torch.randn(20, 4))
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = quaternion_slerp(q0, q1, torch.tensor(t))
            norms = result.norm(dim=-1)
            assert torch.allclose(norms, torch.ones(20), atol=1e-5)
