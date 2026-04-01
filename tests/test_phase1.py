# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Phase 1 tests: Spherical harmonics, tensor products, equivariant layers.

Test categories:
    1. Spherical harmonics correctness + equivariance
    2. Tensor product CG coefficient verification
    3. Radius graph correctness
    4. SE3Conv equivariance (AGENTS.md mandatory)
    5. InvariantAttention invariance
    6. EquivariantLayerNorm + GatedNonlinearity equivariance
    7. Gradient flow (no NaN)
"""

import math

import pytest
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# 1. Spherical Harmonics Tests
# ═══════════════════════════════════════════════════════════════════

class TestSphericalHarmonics:
    """Test spherical harmonics correctness and properties."""

    def test_output_shape_l0(self) -> None:
        """l=0: output should be 1 channel."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        Y = spherical_harmonics(torch.randn(50, 3), max_l=0)
        assert Y.shape == (50, 1)

    def test_output_shape_l1(self) -> None:
        """l=0,1: output should be 4 channels."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        Y = spherical_harmonics(torch.randn(50, 3), max_l=1)
        assert Y.shape == (50, 4)

    def test_output_shape_l2(self) -> None:
        """l=0,1,2: output should be 9 channels."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        Y = spherical_harmonics(torch.randn(50, 3), max_l=2)
        assert Y.shape == (50, 9)

    def test_l0_constant(self) -> None:
        """Y_0^0 should be constant = 1/√(4π) regardless of direction."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        C0 = 1.0 / math.sqrt(4 * math.pi)
        dirs = torch.randn(100, 3)
        Y = spherical_harmonics(dirs, max_l=0)
        assert torch.allclose(Y, C0 * torch.ones(100, 1), atol=1e-5)

    def test_l1_proportional_to_direction(self) -> None:
        """Y_1 should be proportional to the unit direction vector."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        C1 = math.sqrt(3.0 / (4 * math.pi))
        dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Y = spherical_harmonics(dirs, max_l=1, normalize=False)
        # Y_1 = [c1*y, c1*z, c1*x]
        # For [1,0,0]: Y_1 = [0, 0, c1]
        assert torch.allclose(Y[0, 1:4], torch.tensor([0.0, 0.0, C1]), atol=1e-5)
        # For [0,1,0]: Y_1 = [c1, 0, 0]
        assert torch.allclose(Y[1, 1:4], torch.tensor([C1, 0.0, 0.0]), atol=1e-5)
        # For [0,0,1]: Y_1 = [0, c1, 0]
        assert torch.allclose(Y[2, 1:4], torch.tensor([0.0, C1, 0.0]), atol=1e-5)

    def test_orthonormality_l0_l1(self) -> None:
        """SH should be approximately orthogonal on the sphere.

        ∫ Y_l1^m1(r̂) Y_l2^m2(r̂) dΩ ≈ δ_{l1l2} δ_{m1m2}
        We approximate the integral with random sampling on S².
        """
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics

        torch.manual_seed(42)
        N = 10000
        dirs = torch.randn(N, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        Y = spherical_harmonics(dirs, max_l=2, normalize=False)  # [N, 9]

        # Approximate ∫ Y_i Y_j dΩ = (4π/N) Σ Y_i Y_j
        gram = (Y.T @ Y) * (4 * math.pi / N)  # [9, 9]

        # Should be approximately identity
        # Tolerance is loose due to Monte Carlo sampling
        I = torch.eye(9)
        assert torch.allclose(gram, I, atol=0.2), \
            f"SH orthogonality check failed: max off-diag = {(gram - I).abs().max():.3f}"

    def test_equivariance_l1(self) -> None:
        """Y_1(R @ r̂) = R @ Y_1(r̂) — equivariance under rotation.

        For l=1, the Wigner-D matrix D¹(R) is simply R itself (the rotation matrix).
        So rotating the input direction should rotate the l=1 output.
        """
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        from geoembodied.functional.so3_ops import so3_exp, so3_act
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(42)
        dirs = torch.randn(100, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)

        # Random rotation
        omega = torch.randn(3) * 0.5
        q = so3_exp(omega)
        R = quaternion_to_matrix(q)  # [3, 3]

        # Y_1(r̂): l=1 part only
        Y_orig = spherical_harmonics(dirs, max_l=1, normalize=False)[:, 1:4]  # [N, 3]

        # Y_1(R @ r̂)
        dirs_rotated = (R @ dirs.T).T  # [N, 3]
        Y_rotated = spherical_harmonics(dirs_rotated, max_l=1, normalize=False)[:, 1:4]

        # D¹(R) @ Y_1(r̂) — for l=1, D¹ = R in (y,z,x) basis
        # Y_1 = [c1*y, c1*z, c1*x] → reorder to xyz before applying R
        C1 = math.sqrt(3.0 / (4 * math.pi))
        # Y_1 in (y,z,x) order. Convert to (x,y,z):
        Y_xyz = torch.stack([Y_orig[:, 2], Y_orig[:, 0], Y_orig[:, 1]], dim=-1) / C1
        # Rotate
        Y_xyz_rot = (R @ Y_xyz.T).T
        # Convert back to (y,z,x) and multiply by C1
        expected = C1 * torch.stack([Y_xyz_rot[:, 1], Y_xyz_rot[:, 2], Y_xyz_rot[:, 0]], dim=-1)

        assert torch.allclose(Y_rotated, expected, atol=1e-4), \
            f"SH l=1 equivariance failed: max err {(Y_rotated - expected).abs().max():.2e}"

    def test_gradient_no_nan(self) -> None:
        """Gradient through SH should be finite."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        dirs = torch.randn(100, 3, requires_grad=True)
        Y = spherical_harmonics(dirs, max_l=2)
        loss = Y.sum()
        loss.backward()
        assert not torch.isnan(dirs.grad).any()
        assert not torch.isinf(dirs.grad).any()

    def test_batch_dims(self) -> None:
        """SH should support batch dimensions."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        dirs = torch.randn(5, 10, 3)
        Y = spherical_harmonics(dirs, max_l=2)
        assert Y.shape == (5, 10, 9)


# ═══════════════════════════════════════════════════════════════════
# 2. Tensor Product Tests
# ═══════════════════════════════════════════════════════════════════

class TestTensorProduct:
    """Test CG-coefficient tensor products."""

    def test_scalar_scalar(self) -> None:
        """l=0 ⊗ l=0 → l=0: should be simple multiplication."""
        from geoembodied.functional.tensor_product import tensor_product
        a = torch.tensor([[3.0]])
        b = torch.tensor([[4.0]])
        c = tensor_product(a, b, 0, 0, 0)
        assert torch.allclose(c, torch.tensor([[12.0]]), atol=1e-5)

    def test_scalar_vector(self) -> None:
        """l=0 ⊗ l=1 → l=1: should scale the vector."""
        from geoembodied.functional.tensor_product import tensor_product
        s = torch.tensor([[2.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        result = tensor_product(s, v, 0, 1, 1)
        expected = torch.tensor([[2.0, 4.0, 6.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_dot_product_gives_scalar(self) -> None:
        """l=1 ⊗ l=1 → l=0: should give (scaled) dot product."""
        from geoembodied.functional.tensor_product import tensor_product
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[1.0, 0.0, 0.0]])
        result = tensor_product(v1, v2, 1, 1, 0)
        # Should be non-zero and scalar
        assert result.shape == (1, 1)
        assert result.item() > 0

    def test_cross_product_antisymmetric(self) -> None:
        """l=1 ⊗ l=1 → l=1: cross product should be antisymmetric."""
        from geoembodied.functional.tensor_product import tensor_product
        v1 = torch.randn(20, 3)
        v2 = torch.randn(20, 3)
        c12 = tensor_product(v1, v2, 1, 1, 1)
        c21 = tensor_product(v2, v1, 1, 1, 1)
        assert torch.allclose(c12, -c21, atol=1e-5)

    def test_dot_product_equivariance(self) -> None:
        """Dot product (l=1⊗l=1→l=0) should be rotation-invariant."""
        from geoembodied.functional.tensor_product import tensor_product
        from geoembodied.functional.so3_ops import so3_exp, so3_act
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(42)
        v1 = torch.randn(20, 3)
        v2 = torch.randn(20, 3)

        omega = torch.randn(3) * 0.5
        q = so3_exp(omega)
        R = quaternion_to_matrix(q)

        # Rotate both
        v1_rot = (R @ v1.T).T
        v2_rot = (R @ v2.T).T

        # Dot product should be invariant
        s_orig = tensor_product(v1, v2, 1, 1, 0)
        s_rot = tensor_product(v1_rot, v2_rot, 1, 1, 0)

        assert torch.allclose(s_orig, s_rot, atol=1e-4), \
            f"Dot product not invariant: max err {(s_orig - s_rot).abs().max():.2e}"

    def test_cross_product_equivariance(self) -> None:
        """Cross product (l=1⊗l=1→l=1) should be equivariant."""
        from geoembodied.functional.tensor_product import tensor_product
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(42)
        v1 = torch.randn(20, 3)
        v2 = torch.randn(20, 3)

        omega = torch.randn(3) * 0.5
        q = so3_exp(omega)
        R = quaternion_to_matrix(q)

        v1_rot = (R @ v1.T).T
        v2_rot = (R @ v2.T).T

        # f(Rv1, Rv2) = R f(v1, v2)
        c_orig = tensor_product(v1, v2, 1, 1, 1)
        c_rot = tensor_product(v1_rot, v2_rot, 1, 1, 1)
        c_expected = (R @ c_orig.T).T

        assert torch.allclose(c_rot, c_expected, atol=1e-4), \
            f"Cross TP not equivariant: max err {(c_rot - c_expected).abs().max():.2e}"

    def test_cg_available_keys(self) -> None:
        """All expected CG matrices should exist."""
        from geoembodied.functional.tensor_product import get_cg_matrix
        for key in [(0,0,0), (0,1,1), (1,0,1), (1,1,0), (1,1,1), (1,1,2)]:
            cg = get_cg_matrix(*key)
            assert cg is not None


# ═══════════════════════════════════════════════════════════════════
# 3. Radius Graph Tests
# ═══════════════════════════════════════════════════════════════════

class TestRadiusGraph:
    """Test brute-force radius graph construction."""

    def test_basic(self) -> None:
        """Basic radius graph should create edges."""
        from geoembodied.functional.radius_graph import radius_graph
        x = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]])
        row, col = radius_graph(x, r=1.0)
        # First two points are close, third is far
        assert row.shape[0] == 2  # Two directed edges: 0→1 and 1→0
        assert 2 not in row and 2 not in col  # Third point isolated

    def test_no_self_loops(self) -> None:
        """Default: no self-loops."""
        from geoembodied.functional.radius_graph import radius_graph
        x = torch.randn(20, 3)
        row, col = radius_graph(x, r=100.0)
        assert not (row == col).any()

    def test_with_batch(self) -> None:
        """Batch support: points in different batches should not connect."""
        from geoembodied.functional.radius_graph import radius_graph
        x = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        batch = torch.tensor([0, 1])
        row, col = radius_graph(x, r=100.0, batch=batch)
        assert row.shape[0] == 0  # No edges — different batches

    def test_edge_vectors(self) -> None:
        """compute_edge_vectors should return correct diff/dist/direction."""
        from geoembodied.functional.radius_graph import compute_edge_vectors
        x = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        row = torch.tensor([0])
        col = torch.tensor([1])
        diff, dist, direction = compute_edge_vectors(x, row, col)
        assert torch.allclose(diff, torch.tensor([[3.0, 4.0, 0.0]]))
        assert torch.allclose(dist, torch.tensor([5.0]), atol=1e-5)
        assert torch.allclose(direction, torch.tensor([[0.6, 0.8, 0.0]]), atol=1e-5)

    def test_empty_graph(self) -> None:
        """When radius is very small, should return empty edges."""
        from geoembodied.functional.radius_graph import radius_graph
        x = torch.randn(20, 3) * 10.0  # Spread out
        row, col = radius_graph(x, r=0.001)
        assert row.shape[0] == 0


# ═══════════════════════════════════════════════════════════════════
# 4. SE3Conv Equivariance Test (MANDATORY per AGENTS.md)
# ═══════════════════════════════════════════════════════════════════

class TestSE3ConvEquivariance:
    """Mandatory equivariance test: f(R⊳cloud, R⊳features) = R⊳f(cloud, features)."""

    def test_se3conv_so3_equivariance(self) -> None:
        """SE3Conv must be SO(3)-equivariant.

        f(R·pos, s, R·v) == (s', R·v') where s',v' = f(pos, s, v)

        We use a large radius relative to cloud extent so that the
        radius graph is identical before and after rotation (avoiding
        edge-set instability at the cutoff boundary).
        """
        from geoembodied.nn import SE3Conv
        from geoembodied.nn.spatial_graph import SpatialGraph
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(42)
        # Use a tight point cloud with large radius to ensure stable graph
        conv = SE3Conv(8, 4, 8, 4, radius=10.0, num_radial_basis=8)
        conv.eval()

        N = 20
        pos = torch.randn(N, 3) * 0.3  # Small cloud, large radius → everyone sees everyone

        s = torch.randn(N, 8)
        v = torch.randn(N, 4, 3)

        # Random rotation
        omega = torch.randn(3) * 0.8
        q = so3_exp(omega)
        R = quaternion_to_matrix(q)  # [3, 3]

        # Method 1: Apply convolution, then rotate output
        graph = SpatialGraph.build(pos, radius=10.0)
        with torch.no_grad():
            s_out, v_out = conv(s, v, graph)
        v_out_rotated = torch.einsum('ij,...j->...i', R, v_out)  # [N, C_v, 3]

        # Method 2: Rotate inputs, then apply convolution
        pos_rotated = (R @ pos.T).T  # [N, 3]
        v_rotated = torch.einsum('ij,...j->...i', R, v)  # [N, C_v, 3]
        graph_rot = SpatialGraph.build(pos_rotated, radius=10.0)
        with torch.no_grad():
            s_out2, v_out2 = conv(s, v_rotated, graph_rot)

        # Scalar output should be identical (rotation-invariant paths)
        assert torch.allclose(s_out, s_out2, atol=1e-2), \
            f"Scalar equivariance failed: max err {(s_out - s_out2).abs().max():.2e}"

        # Vector output should be rotated version
        assert torch.allclose(v_out_rotated, v_out2, atol=1e-2), \
            f"Vector equivariance failed: max err {(v_out_rotated - v_out2).abs().max():.2e}"

    def test_se3conv_translation_equivariance(self) -> None:
        """SE3Conv should be translation-equivariant.

        f(pos + t, s, v) == f(pos, s, v)

        (Translation only affects position, not features.)
        """
        from geoembodied.nn import SE3Conv
        from geoembodied.nn.spatial_graph import SpatialGraph

        torch.manual_seed(42)
        conv = SE3Conv(8, 4, 8, 4, radius=3.0, num_radial_basis=8)
        conv.eval()

        N = 30
        pos = torch.randn(N, 3) * 0.5
        s = torch.randn(N, 8)
        v = torch.randn(N, 4, 3)

        t = torch.randn(1, 3) * 5.0  # Large translation

        graph1 = SpatialGraph.build(pos, radius=3.0)
        graph2 = SpatialGraph.build(pos + t, radius=3.0)

        with torch.no_grad():
            s1, v1 = conv(s, v, graph1)
            s2, v2 = conv(s, v, graph2)

        assert torch.allclose(s1, s2, atol=1e-4), \
            f"Translation equivariance (scalar) failed: max err {(s1 - s2).abs().max():.2e}"
        assert torch.allclose(v1, v2, atol=1e-4), \
            f"Translation equivariance (vector) failed: max err {(v1 - v2).abs().max():.2e}"


# ═══════════════════════════════════════════════════════════════════
# 5. Invariant Attention Tests
# ═══════════════════════════════════════════════════════════════════

class TestInvariantAttention:
    """Test SE(3)-invariance of attention weights."""

    def test_attention_translation_invariance(self) -> None:
        """attn(pos + t, ...) = attn(pos, ...)"""
        from geoembodied.nn import InvariantAttention

        torch.manual_seed(42)
        attn = InvariantAttention(16, 4, num_heads=4, radius=3.0)
        attn.eval()

        N = 20
        pos = torch.randn(N, 3) * 0.5
        s = torch.randn(N, 16)
        v = torch.randn(N, 4, 3)
        t = torch.randn(1, 3) * 10.0

        with torch.no_grad():
            s1, v1 = attn(pos, s, v)
            s2, v2 = attn(pos + t, s, v)

        assert torch.allclose(s1, s2, atol=1e-4)
        assert torch.allclose(v1, v2, atol=1e-4)

    def test_forward_shape(self) -> None:
        """Output shapes should match input."""
        from geoembodied.nn import InvariantAttention

        attn = InvariantAttention(16, 8, num_heads=4, radius=3.0)
        pos = torch.randn(30, 3)
        s = torch.randn(30, 16)
        v = torch.randn(30, 8, 3)
        s_out, v_out = attn(pos, s, v)
        assert s_out.shape == s.shape
        assert v_out.shape == v.shape


# ═══════════════════════════════════════════════════════════════════
# 6. EquivariantLayerNorm + GatedNonlinearity Tests
# ═══════════════════════════════════════════════════════════════════

class TestEquivariantNorm:
    """Test equivariant layer norm."""

    def test_forward_shape(self) -> None:
        from geoembodied.nn import EquivariantLayerNorm
        norm = EquivariantLayerNorm(32, 8)
        s = torch.randn(10, 32)
        v = torch.randn(10, 8, 3)
        s_out, v_out = norm(s, v)
        assert s_out.shape == s.shape
        assert v_out.shape == v.shape

    def test_scalar_normalized(self) -> None:
        """Scalars should be approximately zero-mean, unit-var."""
        from geoembodied.nn import EquivariantLayerNorm
        norm = EquivariantLayerNorm(32, 0, affine=False)
        s = torch.randn(100, 32) * 5.0 + 10.0
        v = torch.empty(100, 0, 3)
        s_out, _ = norm(s, v)
        # Should be roughly mean=0, std=1 per row
        assert s_out.mean(dim=-1).abs().max() < 0.1
        assert (s_out.std(dim=-1) - 1.0).abs().max() < 0.2


class TestGatedNonlinearity:
    """Test gated nonlinearity."""

    def test_forward_shape(self) -> None:
        from geoembodied.nn import GatedNonlinearity
        gate = GatedNonlinearity(32, 8)
        s = torch.randn(10, 32)
        v = torch.randn(10, 8, 3)
        s_out, v_out = gate(s, v)
        assert s_out.shape == s.shape
        assert v_out.shape == v.shape

    def test_equivariance(self) -> None:
        """Gate should preserve equivariance: R(gate*v) = gate*(Rv)."""
        from geoembodied.nn import GatedNonlinearity
        from geoembodied.functional.so3_ops import so3_exp
        from geoembodied.functional.quaternion_ops import quaternion_to_matrix

        torch.manual_seed(42)
        gate = GatedNonlinearity(32, 8)
        gate.eval()

        s = torch.randn(10, 32)
        v = torch.randn(10, 8, 3)

        omega = torch.randn(3) * 0.5
        R = quaternion_to_matrix(so3_exp(omega))

        with torch.no_grad():
            s1, v1 = gate(s, v)
            v1_rot = torch.einsum('ij,...j->...i', R, v1)

            v_rot = torch.einsum('ij,...j->...i', R, v)
            s2, v2 = gate(s, v_rot)

        # Scalar output should be identical (rotation doesn't affect scalars)
        assert torch.allclose(s1, s2, atol=1e-5)
        # Vector output: f(s, Rv) = R f(s, v)
        assert torch.allclose(v1_rot, v2, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════
# 7. Gradient Flow Tests
# ═══════════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Test that gradients flow without NaN through all new layers."""

    def test_se3conv_gradient(self) -> None:
        """SE3Conv should produce finite gradients."""
        from geoembodied.nn import SE3Conv
        from geoembodied.nn.spatial_graph import SpatialGraph

        conv = SE3Conv(8, 4, 8, 4, radius=3.0)
        pos = torch.randn(20, 3)
        s = torch.randn(20, 8, requires_grad=True)
        v = torch.randn(20, 4, 3, requires_grad=True)

        graph = SpatialGraph.build(pos, radius=3.0)
        s_out, v_out = conv(s, v, graph)
        loss = s_out.sum() + v_out.sum()
        loss.backward()

        assert s.grad is not None
        assert not torch.isnan(s.grad).any()
        assert v.grad is not None
        assert not torch.isnan(v.grad).any()

    def test_attention_gradient(self) -> None:
        """InvariantAttention should produce finite gradients."""
        from geoembodied.nn import InvariantAttention

        attn = InvariantAttention(16, 4, num_heads=4, radius=3.0)
        pos = torch.randn(20, 3)
        s = torch.randn(20, 16, requires_grad=True)
        v = torch.randn(20, 4, 3, requires_grad=True)

        s_out, v_out = attn(pos, s, v)
        loss = s_out.sum() + v_out.sum()
        loss.backward()

        assert not torch.isnan(s.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_sph_harm_gradient(self) -> None:
        """Spherical harmonics backward should be finite."""
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics
        dirs = torch.randn(100, 3, requires_grad=True)
        Y = spherical_harmonics(dirs, max_l=2)
        Y.sum().backward()
        assert not torch.isnan(dirs.grad).any()

    def test_tensor_product_gradient(self) -> None:
        """Tensor product should have finite gradients."""
        from geoembodied.functional.tensor_product import tensor_product
        v1 = torch.randn(20, 3, requires_grad=True)
        v2 = torch.randn(20, 3, requires_grad=True)

        for l_out in [0, 1, 2]:
            result = tensor_product(v1, v2, 1, 1, l_out)
            result.sum().backward(retain_graph=True)
            assert not torch.isnan(v1.grad).any()
            v1.grad = None
            v2.grad = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
