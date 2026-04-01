# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Equivariance test template.

Per AGENTS.md mandatory testing rule:
    For every equivariant layer, verify:
        f(g⊳x) ≈ g⊳f(x)
    for random group element g ∈ G and random input x.

Usage pattern for new layers::

    class TestMyLayer(EquivarianceTestBase):
        def make_layer(self):
            return MyEquivariantLayer(in_features=16, out_features=32)

        def make_input(self, batch=5):
            return torch.randn(batch, 100, 16)  # [B, N, C]

        def group_action(self, g, x):
            # Apply g to both spatial and feature dimensions
            return g @ x
"""

import pytest
import torch
from torch import Tensor


class EquivarianceTestBase:
    """Base class for equivariance tests.

    Tests the fundamental property of equivariant layers:
        f(g ⊳ x) = g ⊳ f(x)

    Subclass and implement:
        - make_layer: Create the layer to test
        - make_input: Generate random input
        - group_action: Apply group element to input/output
    """

    def make_layer(self) -> torch.nn.Module:
        raise NotImplementedError

    def make_input(self, batch: int = 5) -> Tensor:
        raise NotImplementedError

    def group_action(self, g: Tensor, x: Tensor) -> Tensor:
        """Apply group element g to tensor x."""
        raise NotImplementedError

    def make_random_group_element(self) -> Tensor:
        """Generate a random group element."""
        from geoembodied.lietensor import SO3
        return SO3.exp(torch.randn(3))

    def test_equivariance(self, atol: float = 1e-5) -> None:
        """Test: f(g⊳x) ≈ g⊳f(x).

        This is the MANDATORY equivariance test per AGENTS.md Rule 3.
        """
        torch.manual_seed(42)

        layer = self.make_layer()
        layer.eval()

        x = self.make_input()
        g = self.make_random_group_element()

        # Path 1: transform input, then pass through layer
        gx = self.group_action(g, x)
        y1 = layer(gx)

        # Path 2: pass through layer, then transform output
        fx = layer(x)
        y2 = self.group_action(g, fx)

        # Relative error check
        err = (y1 - y2).norm() / (y2.norm() + 1e-8)
        assert err < atol, (
            f"Equivariance violated! Relative error: {err:.2e}\n"
            f"  ‖f(gx) - g f(x)‖ / ‖g f(x)‖ = {err:.2e}\n"
            f"  Expected < {atol:.2e}"
        )


class TestSO3ActionEquivariance(EquivarianceTestBase):
    """Verify that SO3 rotation action is self-consistent.

    The rotation action R ⊳ (R' ⊳ x) should equal (R ∘ R') ⊳ x.
    This validates the group action before we build equivariant layers.
    """

    def make_layer(self) -> torch.nn.Module:
        """Identity 'layer' — just passes through."""
        return torch.nn.Identity()

    def make_input(self, batch: int = 1) -> Tensor:
        return torch.randn(batch, 50, 3)

    def group_action(self, g: Tensor, x: Tensor) -> Tensor:
        from geoembodied.functional.quaternion_ops import quaternion_apply
        q = g.as_subclass(Tensor)
        # Expand quaternion to match input batch + point dimensions
        while q.dim() < x.dim():
            q = q.unsqueeze(0)
        q = q.expand(*x.shape[:-1], 4)
        return quaternion_apply(q, x)

    def test_composition_equivariance(self) -> None:
        """R1 ⊳ (R2 ⊳ x) = (R1 ∘ R2) ⊳ x."""
        from geoembodied.lietensor import SO3

        torch.manual_seed(0)

        x = torch.randn(20, 3)
        R1 = SO3.exp(torch.randn(3))
        R2 = SO3.exp(torch.randn(3))

        # Path 1: sequential application
        y1 = R2 @ x
        y1 = R1 @ y1

        # Path 2: composed rotation
        R12 = R1 @ R2
        y2 = R12 @ x

        assert torch.allclose(y1, y2, atol=1e-5), \
            f"Composition equivariance failed: {(y1 - y2).abs().max():.2e}"


class TestSE3ActionEquivariance(EquivarianceTestBase):
    """Verify SE3 rigid body transform composition."""

    def make_layer(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def make_input(self, batch: int = 1) -> Tensor:
        return torch.randn(batch, 50, 3)

    def group_action(self, g: Tensor, x: Tensor) -> Tensor:
        from geoembodied.functional.quaternion_ops import quaternion_apply
        t_data = g.as_subclass(Tensor)
        q = t_data[..., :4]
        t = t_data[..., 4:]
        # Expand to match batched input
        while q.dim() < x.dim():
            q = q.unsqueeze(0)
            t = t.unsqueeze(0)
        q = q.expand(*x.shape[:-1], 4)
        t = t.expand(*x.shape[:-1], 3)
        return quaternion_apply(q, x) + t

    def make_random_group_element(self) -> Tensor:
        from geoembodied.lietensor import SE3
        return SE3.exp(torch.randn(6) * 0.5)

    def test_composition_equivariance(self) -> None:
        """T1 ⊳ (T2 ⊳ x) = (T1 ∘ T2) ⊳ x."""
        from geoembodied.lietensor import SE3

        torch.manual_seed(1)

        x = torch.randn(50, 3)
        T1 = SE3.exp(torch.randn(6) * 0.5)
        T2 = SE3.exp(torch.randn(6) * 0.5)

        # Path 1: sequential
        y1 = T2 @ x
        y1 = T1 @ y1

        # Path 2: composed
        T12 = T1 @ T2
        y2 = T12 @ x

        assert torch.allclose(y1, y2, atol=1e-4), \
            f"SE3 composition equivariance failed: {(y1 - y2).abs().max():.2e}"


class TestInvariantSelfAttentionInvariance:
    """Verify InvariantSelfAttention produces SO(3)-invariant scalar outputs.

    Since the module operates only on l=0 scalars and uses pairwise distances
    (which are SO(3)-invariant), rotating ALL point positions should NOT
    change the scalar output.

    Test: f(R·pos, scalars) ≈ f(pos, scalars)  for random R ∈ SO(3)
    """

    def test_invariance_under_rotation(self) -> None:
        """Self-attention output should be invariant to SO(3) rotation of positions."""
        from geoembodied.nn.geometric_self_attention import InvariantSelfAttention
        from geoembodied.lietensor import SO3

        torch.manual_seed(42)
        B, N, C = 2, 64, 32
        H = 2

        layer = InvariantSelfAttention(
            channels=C, num_heads=H, sigma_d=0.1, n_freqs=8,
        )
        layer.eval()

        # Random input
        pos = torch.randn(B, N, 3)
        scalars = torch.randn(B, N, C)

        # Random rotation
        R_so3 = SO3.exp(torch.randn(3))
        R_mat = R_so3.to_matrix()  # [3, 3]

        # Rotated positions: pos @ R^T
        pos_rotated = pos @ R_mat.T

        # Forward both
        with torch.no_grad():
            out_orig = layer(scalars, pos)
            out_rot = layer(scalars, pos_rotated)

        err = (out_orig - out_rot).abs().max()
        rel_err = (out_orig - out_rot).norm() / (out_orig.norm() + 1e-8)
        assert rel_err < 5e-4, (
            f"InvariantSelfAttention NOT SO(3)-invariant!\n"
            f"  max |f(pos) - f(R·pos)| = {err:.2e}\n"
            f"  relative error = {rel_err:.2e}"
        )

    def test_invariance_different_rotations_per_batch(self) -> None:
        """Each batch element can have independent rotation → still invariant."""
        from geoembodied.nn.geometric_self_attention import InvariantSelfAttention
        from geoembodied.lietensor import SO3

        torch.manual_seed(123)
        B, N, C = 4, 32, 32

        layer = InvariantSelfAttention(channels=C, num_heads=2, sigma_d=0.1)
        layer.eval()

        pos = torch.randn(B, N, 3)
        scalars = torch.randn(B, N, C)

        # Different rotation per batch element
        pos_rot = pos.clone()
        for b in range(B):
            R = SO3.exp(torch.randn(3)).to_matrix()
            pos_rot[b] = pos[b] @ R.T

        with torch.no_grad():
            out_orig = layer(scalars, pos)
            out_rot = layer(scalars, pos_rot)

        err = (out_orig - out_rot).abs().max()
        rel_err = (out_orig - out_rot).norm() / (out_orig.norm() + 1e-8)
        assert rel_err < 5e-4, f"Per-batch rotation invariance failed: rel_err={rel_err:.2e}"

    def test_gradient_flows_through_rpe(self) -> None:
        """Gradients should flow through rpe_proj (the learnable part of RPE)."""
        from geoembodied.nn.geometric_self_attention import InvariantSelfAttention

        torch.manual_seed(7)
        B, N, C = 2, 32, 32

        layer = InvariantSelfAttention(channels=C, num_heads=2, sigma_d=0.1)
        layer.train()

        pos = torch.randn(B, N, 3)
        scalars = torch.randn(B, N, C, requires_grad=True)

        out = layer(scalars, pos)
        loss = out.sum()
        loss.backward()

        # rpe_weight_sin should have gradients (learnable RPE parameters)
        assert layer.rpe_weight_sin.grad is not None, \
            "rpe_weight_sin has no gradient!"
        assert layer.rpe_weight_sin.grad.abs().max() > 0, \
            "rpe_weight_sin gradient is all zeros!"
        # scalars should have gradients
        assert scalars.grad is not None, "Input scalars have no gradient!"
        assert torch.isfinite(scalars.grad).all(), "NaN in input gradients!"

    def test_mask_zeros_pad_positions(self) -> None:
        """Pad positions (mask=False) should produce zero output."""
        from geoembodied.nn.geometric_self_attention import InvariantSelfAttention

        torch.manual_seed(0)
        B, N, C = 2, 32, 32

        layer = InvariantSelfAttention(channels=C, num_heads=2, sigma_d=0.1)
        layer.eval()

        pos = torch.randn(B, N, 3)
        scalars = torch.randn(B, N, C)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -8:] = False  # last 8 points are padding

        with torch.no_grad():
            out = layer(scalars, pos, mask=mask)

        # Pad positions should be zeroed
        assert (out[:, -8:] == 0).all(), \
            "Pad positions are not zeroed in output!"

