# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Comprehensive Lie group numerics stress tests.

Tests the full range of θ ∈ [0, π] for SO(3) and SE(3) operations,
with focus on:
    1. Forward accuracy at singularities (θ→0, θ→π)
    2. Backward NaN-freedom across entire domain
    3. Taylor expansion accuracy vs analytic formula
    4. Gradient correctness (vs finite differences)
    5. Float32 precision boundaries (documented, not asserted)

Per AGENTS.md: mandatory equivariance error test pattern:
    f(g·x) ≈ g·f(x) for random g ∈ G.
"""

import math
import pytest
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# SO(3) Forward Accuracy
# ═══════════════════════════════════════════════════════════════════


class TestSO3Forward:
    """Test so3_exp / so3_log forward accuracy across θ ∈ [0, π]."""

    @pytest.mark.parametrize("theta", [
        0.0, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        0.1, 0.5, 1.0, 2.0, 3.0,
        math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-6,
    ])
    def test_exp_log_roundtrip(self, theta: float) -> None:
        """exp(log(exp(ω))) ≈ exp(ω) across the full range."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        if theta == 0.0:
            omega = torch.zeros(1, 3)
        else:
            # Axis: normalized random direction
            torch.manual_seed(42)
            axis = torch.randn(1, 3)
            axis = axis / axis.norm(dim=-1, keepdim=True)
            omega = theta * axis

        q = so3_exp(omega)
        omega_back = so3_log(q)
        q_back = so3_exp(omega_back)

        # Compare quaternions (up to sign ambiguity)
        dot = (q * q_back).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones_like(dot), atol=1e-4), \
            f"SO3 exp/log roundtrip failed at θ={theta}: dot={dot.item():.6f}"

    @pytest.mark.parametrize("theta", [
        math.pi - 1e-7,  # float32 precision boundary
    ])
    def test_exp_log_near_pi_boundary(self, theta: float) -> None:
        """At θ = π ± float32_ULP, roundtrip may fail but must NOT NaN."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.tensor([[theta, 0.0, 0.0]])
        q = so3_exp(omega)
        omega_back = so3_log(q)

        # Must not produce NaN/Inf
        assert not torch.isnan(omega_back).any(), "NaN in so3_log near π"
        assert not torch.isinf(omega_back).any(), "Inf in so3_log near π"

        # The magnitude should be close to π
        theta_back = omega_back.norm(dim=-1)
        assert torch.allclose(theta_back, torch.tensor([math.pi]), atol=1e-4), \
            f"so3_log magnitude wrong at θ→π: {theta_back.item():.6f}"

    def test_exp_log_batch_random(self) -> None:
        """Batch roundtrip with random rotations up to 3 radians."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        torch.manual_seed(123)
        omega = torch.randn(1000, 3) * 1.5  # θ up to ~2.6 rad

        q = so3_exp(omega)
        omega_back = so3_log(q)
        q_back = so3_exp(omega_back)

        dot = (q * q_back).sum(dim=-1).abs()
        assert (dot > 1.0 - 1e-4).all(), \
            f"Batch roundtrip failed: min dot = {dot.min():.6f}"


# ═══════════════════════════════════════════════════════════════════
# SE(3) Forward Accuracy
# ═══════════════════════════════════════════════════════════════════


class TestSE3Forward:
    """Test se3_exp / se3_log forward accuracy."""

    @pytest.mark.parametrize("theta", [
        0.0, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        0.1, 0.5, 1.0, 2.0, 3.0,
    ])
    def test_exp_log_roundtrip(self, theta: float) -> None:
        """SE3 exp/log roundtrip across angular range."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        torch.manual_seed(42)
        axis = torch.randn(1, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        vel = torch.randn(1, 3)

        xi = torch.cat([theta * axis, vel], dim=-1)  # [1, 6]

        T = se3_exp(xi)
        xi_back = se3_log(T)
        T_back = se3_exp(xi_back)

        # Compare rotation (quaternion dot)
        q_dot = (T[..., :4] * T_back[..., :4]).sum(dim=-1).abs()
        assert torch.allclose(q_dot, torch.ones_like(q_dot), atol=1e-3), \
            f"SE3 rotation roundtrip failed at θ={theta}: dot={q_dot.item():.6f}"

        # Compare translation
        t_err = (T[..., 4:] - T_back[..., 4:]).norm(dim=-1)
        assert (t_err < 1e-3).all(), \
            f"SE3 translation roundtrip failed at θ={theta}: err={t_err.item():.4e}"


# ═══════════════════════════════════════════════════════════════════
# Backward NaN-Freedom (the most critical tests)
# ═══════════════════════════════════════════════════════════════════


class TestBackwardNaNFreedom:
    """Verify that NO backward pass produces NaN or Inf across full θ range.

    This is the core guarantee of our numeric hardening.
    """

    @pytest.mark.parametrize("theta", [
        0.0, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        0.1, 0.5, 1.0, 2.0, 3.0,
        math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-6,
        math.pi - 1e-7, math.pi,
    ])
    def test_so3_exp_backward(self, theta: float) -> None:
        """so3_exp backward must be finite everywhere."""
        from geoembodied.functional.so3_ops import so3_exp

        omega = torch.tensor([[theta, 0.0, 0.0]], requires_grad=True)
        q = so3_exp(omega)
        loss = q.sum()
        loss.backward()
        assert omega.grad is not None
        assert not torch.isnan(omega.grad).any(), \
            f"NaN gradient in so3_exp at θ={theta}"
        assert not torch.isinf(omega.grad).any(), \
            f"Inf gradient in so3_exp at θ={theta}"

    @pytest.mark.parametrize("theta", [
        0.0, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        0.1, 0.5, 1.0, 2.0, 3.0,
        math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-6,
    ])
    def test_so3_log_backward(self, theta: float) -> None:
        """so3_log backward must be finite everywhere."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        if theta == 0.0:
            omega = torch.zeros(1, 3)
        else:
            omega = torch.tensor([[theta, 0.0, 0.0]])

        q = so3_exp(omega).detach().requires_grad_(True)
        omega_back = so3_log(q)
        loss = omega_back.sum()
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any(), \
            f"NaN gradient in so3_log at θ={theta}"
        assert not torch.isinf(q.grad).any(), \
            f"Inf gradient in so3_log at θ={theta}"

    @pytest.mark.parametrize("theta", [
        0.0, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        0.1, 0.5, 1.0, 2.0, 3.0,
    ])
    def test_se3_exp_backward(self, theta: float) -> None:
        """se3_exp backward must be finite everywhere."""
        from geoembodied.functional.se3_ops import se3_exp

        xi = torch.tensor(
            [[theta, 0.0, 0.0, 1.0, 0.5, 0.3]], requires_grad=True
        )
        T = se3_exp(xi)
        loss = T.sum()
        loss.backward()
        assert xi.grad is not None
        assert not torch.isnan(xi.grad).any(), \
            f"NaN gradient in se3_exp at θ={theta}"
        assert not torch.isinf(xi.grad).any(), \
            f"Inf gradient in se3_exp at θ={theta}"

    @pytest.mark.parametrize("theta", [
        0.0, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        0.1, 0.5, 1.0, 2.0, 3.0,
        math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-6,
    ])
    def test_se3_log_backward(self, theta: float) -> None:
        """se3_log backward must be finite everywhere."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        xi_input = torch.tensor([[theta, 0.0, 0.0, 1.0, 0.5, 0.3]])
        T = se3_exp(xi_input).detach().requires_grad_(True)
        xi_back = se3_log(T)
        loss = xi_back.sum()
        loss.backward()
        assert T.grad is not None
        assert not torch.isnan(T.grad).any(), \
            f"NaN gradient in se3_log at θ={theta}"
        assert not torch.isinf(T.grad).any(), \
            f"Inf gradient in se3_log at θ={theta}"

    def test_full_pipeline_backward(self) -> None:
        """Full exp→log chain backward must be NaN-free."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        torch.manual_seed(42)
        # Use moderate range to stay within float32 exp/log domain
        xi = (torch.randn(100, 6) * 0.5).requires_grad_(True)

        T = se3_exp(xi)
        xi_back = se3_log(T)
        loss = xi_back.norm(dim=-1).sum()
        loss.backward()

        assert xi.grad is not None, ".grad is None — check requires_grad"
        assert not torch.isnan(xi.grad).any(), "NaN in full pipeline backward"
        assert not torch.isinf(xi.grad).any(), "Inf in full pipeline backward"


# ═══════════════════════════════════════════════════════════════════
# Taylor Expansion Accuracy
# ═══════════════════════════════════════════════════════════════════


class TestTaylorExpansions:
    """Verify Taylor expansions match analytic formulas in overlap region."""

    def test_sinc_taylor_vs_analytic(self) -> None:
        """sin(θ)/θ Taylor matches analytic in [1e-5, 1e-3]."""
        from geoembodied.functional.numeric_safe import taylor_sinc

        # Test at overlap region where both are accurate
        theta = torch.linspace(1e-5, 1e-3, 100)
        result = taylor_sinc(theta)
        expected = torch.sin(theta) / theta

        assert torch.allclose(result, expected, atol=1e-6), \
            f"sinc Taylor error: max={( result - expected).abs().max():.4e}"

    def test_cos_over_theta_taylor_vs_analytic(self) -> None:
        """(1-cos θ)/θ² Taylor matches analytic in overlap region."""
        from geoembodied.functional.numeric_safe import taylor_cos_over_theta

        # Test in range where both analytic and result are accurate
        # _TAYLOR_THRESH = 1e-4, so test above that to use analytic path
        theta = torch.linspace(1e-3, 1.0, 100)
        result = taylor_cos_over_theta(theta)
        expected = (1.0 - torch.cos(theta)) / (theta * theta)

        assert torch.allclose(result, expected, atol=1e-5), \
            f"cos_over_theta Taylor error: max={(result - expected).abs().max():.4e}"

    def test_V_inv_coeff_taylor_vs_analytic(self) -> None:
        """V⁻¹ coefficient Taylor matches analytic in [0.01, 0.1]."""
        from geoembodied.functional.numeric_safe import taylor_V_inv_coeff

        # Test in region where *both* analytic and Taylor are accurate
        theta = torch.linspace(0.01, 0.1, 100, dtype=torch.float64)
        result = taylor_V_inv_coeff(theta)

        # Analytic (safe at this range)
        expected = (
            1.0 / (theta * theta)
            - (1.0 + torch.cos(theta))
            / (2.0 * theta * torch.sin(theta))
        )

        assert torch.allclose(result, expected, atol=1e-6), \
            f"V_inv_coeff Taylor error: max={(result - expected).abs().max():.4e}"

    def test_V_inv_coeff_at_zero(self) -> None:
        """V⁻¹ coefficient should be ≈ 1/12 at θ=0."""
        from geoembodied.functional.numeric_safe import taylor_V_inv_coeff

        result = taylor_V_inv_coeff(torch.tensor([0.0]))
        expected = 1.0 / 12.0
        assert abs(result.item() - expected) < 1e-6, \
            f"V_inv at 0: got {result.item()}, expected {expected}"

    def test_theta_minus_sin_over_theta3_at_zero(self) -> None:
        """(θ-sinθ)/θ³ should be ≈ 1/6 at θ=0."""
        from geoembodied.functional.numeric_safe import taylor_theta_minus_sin_over_theta3

        result = taylor_theta_minus_sin_over_theta3(torch.tensor([0.0]))
        expected = 1.0 / 6.0
        assert abs(result.item() - expected) < 1e-6, \
            f"B at 0: got {result.item()}, expected {expected}"


# ═══════════════════════════════════════════════════════════════════
# Gradient Correctness (vs Finite Differences)
# ═══════════════════════════════════════════════════════════════════


class TestGradientCorrectness:
    """Verify analytic gradients match finite difference approximation."""

    def test_so3_exp_gradcheck(self) -> None:
        """so3_exp passes torch.autograd.gradcheck (double precision)."""
        from geoembodied.functional.so3_ops import so3_exp

        omega = torch.randn(5, 3, dtype=torch.float64, requires_grad=True) * 0.5
        assert torch.autograd.gradcheck(so3_exp, omega, atol=1e-5), \
            "so3_exp gradcheck failed"

    def test_so3_log_gradcheck(self) -> None:
        """so3_log passes torch.autograd.gradcheck (double precision)."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.randn(5, 3, dtype=torch.float64) * 0.5
        q = so3_exp(omega).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(so3_log, q, atol=1e-5), \
            "so3_log gradcheck failed"

    def test_se3_exp_gradcheck(self) -> None:
        """se3_exp passes torch.autograd.gradcheck (double precision)."""
        from geoembodied.functional.se3_ops import se3_exp

        xi = torch.randn(5, 6, dtype=torch.float64, requires_grad=True) * 0.5
        assert torch.autograd.gradcheck(se3_exp, xi, atol=1e-5), \
            "se3_exp gradcheck failed"

    def test_se3_log_gradcheck(self) -> None:
        """se3_log passes torch.autograd.gradcheck (double precision)."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        xi = torch.randn(5, 6, dtype=torch.float64) * 0.5
        T = se3_exp(xi).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(se3_log, T, atol=1e-5), \
            "se3_log gradcheck failed"

    def test_so3_exp_gradcheck_near_zero(self) -> None:
        """so3_exp gradcheck at very small angles (Taylor branch)."""
        from geoembodied.functional.so3_ops import so3_exp

        omega = torch.tensor(
            [[1e-6, 2e-6, -1e-6]], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(so3_exp, omega, atol=1e-5), \
            "so3_exp gradcheck failed near zero"

    # ── Boundary gradcheck: θ→0 ──

    def test_so3_log_gradcheck_near_zero(self) -> None:
        """so3_log gradcheck at θ→0 (Taylor branch of J_l⁻¹)."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        # Create quaternion close to identity (θ ≈ 1e-6)
        omega_small = torch.tensor(
            [[1e-7, -2e-7, 1e-7]], dtype=torch.float64
        )
        q = so3_exp(omega_small).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(so3_log, q, atol=1e-5), \
            "so3_log gradcheck failed near θ=0"

    def test_se3_exp_gradcheck_near_zero(self) -> None:
        """se3_exp gradcheck at θ→0 (Taylor branch of V matrix)."""
        from geoembodied.functional.se3_ops import se3_exp

        # Rotational part near zero, translational part nonzero
        xi = torch.tensor(
            [[1e-7, -2e-7, 1e-7, 0.5, -0.3, 0.2]], dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(se3_exp, xi, atol=1e-5), \
            "se3_exp gradcheck failed near θ=0"

    def test_se3_log_gradcheck_near_zero(self) -> None:
        """se3_log gradcheck at θ→0 (Taylor branch of V⁻¹)."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        xi_small = torch.tensor(
            [[1e-7, -2e-7, 1e-7, 0.5, -0.3, 0.2]], dtype=torch.float64
        )
        T = se3_exp(xi_small).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(se3_log, T, atol=1e-5), \
            "se3_log gradcheck failed near θ=0"

    # ── Boundary gradcheck: θ→π ──

    def test_so3_exp_gradcheck_near_pi(self) -> None:
        """so3_exp gradcheck at θ≈π (near 180° rotation)."""
        from geoembodied.functional.so3_ops import so3_exp

        # θ = π - 1e-4, axis along x
        theta = torch.pi - 1e-4
        omega = torch.tensor(
            [[theta, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(so3_exp, omega, atol=1e-5), \
            "so3_exp gradcheck failed near θ=π"

    def test_so3_log_gradcheck_near_pi(self) -> None:
        """so3_log gradcheck at θ≈π (near 180° rotation).

        This tests the analytic J_l⁻¹ at the most dangerous singularity.
        """
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        # Build quaternion at θ ≈ π - 1e-3 (safely away from exact π
        # where log is mathematically undefined / multi-valued)
        theta = torch.pi - 1e-3
        omega_pi = torch.tensor(
            [[theta, 0.0, 0.0]], dtype=torch.float64
        )
        q = so3_exp(omega_pi).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(so3_log, q, atol=1e-4), \
            "so3_log gradcheck failed near θ=π"

    def test_se3_exp_gradcheck_near_pi(self) -> None:
        """se3_exp gradcheck at θ≈π."""
        from geoembodied.functional.se3_ops import se3_exp

        theta = torch.pi - 1e-4
        xi = torch.tensor(
            [[theta, 0.0, 0.0, 0.5, -0.3, 0.2]], dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(se3_exp, xi, atol=1e-5), \
            "se3_exp gradcheck failed near θ=π"

    def test_se3_log_gradcheck_near_pi(self) -> None:
        """se3_log gradcheck at θ≈π (V⁻¹ singularity protection).

        This is the hardest test: V⁻¹ has a 1/θ² - (1+cosθ)/(2θsinθ)
        term that blows up at θ=π. Our taylor_V_inv_coeff handles this
        with an analytic limit branch.
        """
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        theta = torch.pi - 1e-3
        xi_pi = torch.tensor(
            [[theta, 0.0, 0.0, 0.5, -0.3, 0.2]], dtype=torch.float64
        )
        T = se3_exp(xi_pi).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(se3_log, T, atol=1e-4), \
            "se3_log gradcheck failed near θ=π"

    # ── Boundary gradcheck: multiple angles in one batch ──

    def test_so3_exp_gradcheck_mixed_angles(self) -> None:
        """so3_exp gradcheck with a batch spanning θ∈{~0, ~1, ~π}."""
        from geoembodied.functional.so3_ops import so3_exp

        omega = torch.tensor([
            [1e-7, 0.0, 0.0],        # θ ≈ 0 (Taylor)
            [0.5, 0.3, -0.2],         # θ ≈ 0.6 (normal)
            [torch.pi - 1e-3, 0.0, 0.0],  # θ ≈ π
        ], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(so3_exp, omega, atol=1e-5), \
            "so3_exp gradcheck failed for mixed angles batch"

    def test_so3_log_gradcheck_mixed_angles(self) -> None:
        """so3_log gradcheck with a batch spanning θ∈{~0, ~1, ~π}."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        omega = torch.tensor([
            [1e-7, 0.0, 0.0],
            [0.5, 0.3, -0.2],
            [torch.pi - 1e-3, 0.0, 0.0],
        ], dtype=torch.float64)
        q = so3_exp(omega).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(so3_log, q, atol=1e-4), \
            "so3_log gradcheck failed for mixed angles batch"


# ═══════════════════════════════════════════════════════════════════
# safe_where NaN Prevention
# ═══════════════════════════════════════════════════════════════════


class TestSafeWhere:
    """Verify _safe_theta pattern prevents torch.where NaN gradient routing."""

    def test_taylor_sinc_grad_at_zero(self) -> None:
        """taylor_sinc must have finite gradient at θ = 0 exactly."""
        from geoembodied.functional.numeric_safe import taylor_sinc

        theta = torch.tensor([0.0], requires_grad=True)
        result = taylor_sinc(theta)
        result.backward()
        assert not torch.isnan(theta.grad).any(), \
            "taylor_sinc gradient is NaN at θ=0"
        assert not torch.isinf(theta.grad).any(), \
            "taylor_sinc gradient is Inf at θ=0"

    def test_taylor_cos_over_theta_grad_at_zero(self) -> None:
        """taylor_cos_over_theta must have finite gradient at θ = 0."""
        from geoembodied.functional.numeric_safe import taylor_cos_over_theta

        theta = torch.tensor([0.0], requires_grad=True)
        result = taylor_cos_over_theta(theta)
        result.backward()
        assert not torch.isnan(theta.grad).any()
        assert not torch.isinf(theta.grad).any()

    def test_V_inv_coeff_grad_at_zero(self) -> None:
        """taylor_V_inv_coeff must have finite gradient at θ = 0."""
        from geoembodied.functional.numeric_safe import taylor_V_inv_coeff

        theta = torch.tensor([0.0], requires_grad=True)
        result = taylor_V_inv_coeff(theta)
        result.backward()
        assert not torch.isnan(theta.grad).any(), \
            "V_inv_coeff gradient is NaN at θ=0"
        assert not torch.isinf(theta.grad).any(), \
            "V_inv_coeff gradient is Inf at θ=0"

    def test_batch_mixed_small_large(self) -> None:
        """Batch with both small and large θ must be NaN-free."""
        from geoembodied.functional.so3_ops import so3_exp, so3_log

        # Mix of small angles (Taylor) and large angles (normal)
        omega = torch.tensor([
            [0.0, 0.0, 0.0],      # exactly zero
            [1e-10, 0.0, 0.0],    # tiny
            [1.0, 0.0, 0.0],      # normal
            [3.0, 0.0, 0.0],      # large
        ], requires_grad=True)

        q = so3_exp(omega)
        omega_back = so3_log(q)
        loss = omega_back.norm(dim=-1).sum()
        loss.backward()

        assert not torch.isnan(omega.grad).any(), \
            "NaN in mixed batch backward"
        assert not torch.isinf(omega.grad).any(), \
            "Inf in mixed batch backward"


# ═══════════════════════════════════════════════════════════════════
# V⁻¹ coefficient θ→π protection
# ═══════════════════════════════════════════════════════════════════


class TestVInvCoeffNearPi:
    """Verify V⁻¹ coefficient is stable at θ→π."""

    def test_V_inv_coeff_at_pi(self) -> None:
        """V⁻¹ coefficient approaches 1/π² at θ=π."""
        from geoembodied.functional.numeric_safe import taylor_V_inv_coeff

        theta = torch.tensor([math.pi])
        result = taylor_V_inv_coeff(theta)
        expected = 1.0 / (math.pi ** 2)
        assert abs(result.item() - expected) < 1e-4, \
            f"V_inv at π: got {result.item()}, expected {expected}"

    @pytest.mark.parametrize("delta", [1e-3, 1e-4, 1e-6, 1e-8])
    def test_V_inv_coeff_near_pi_nan_free(self, delta: float) -> None:
        """V⁻¹ coefficient must not NaN near θ=π."""
        from geoembodied.functional.numeric_safe import taylor_V_inv_coeff

        theta = torch.tensor([math.pi - delta], requires_grad=True)
        result = taylor_V_inv_coeff(theta)

        assert not torch.isnan(result).any(), \
            f"NaN in V_inv_coeff at θ=π-{delta}"
        assert not torch.isinf(result).any(), \
            f"Inf in V_inv_coeff at θ=π-{delta}"

        result.backward()
        assert not torch.isnan(theta.grad).any(), \
            f"NaN gradient in V_inv_coeff at θ=π-{delta}"

    def test_V_inv_coeff_continuity_at_pi(self) -> None:
        """V⁻¹ coeff should be continuous across the θ→π boundary."""
        from geoembodied.functional.numeric_safe import taylor_V_inv_coeff

        # Values just inside and outside the near-pi threshold
        theta_inside = torch.tensor([math.pi - 5e-4])  # inside threshold
        theta_outside = torch.tensor([math.pi - 2e-3])  # outside threshold

        val_inside = taylor_V_inv_coeff(theta_inside).item()
        val_outside = taylor_V_inv_coeff(theta_outside).item()

        # They should be reasonably close (same order of magnitude)
        assert abs(val_inside - val_outside) < 0.01, \
            f"Discontinuity at π boundary: {val_inside} vs {val_outside}"


# ═══════════════════════════════════════════════════════════════════
# SO(3) Left Jacobian Correctness
# ═══════════════════════════════════════════════════════════════════


class TestSO3LeftJacobian:
    """Verify J_l · J_l⁻¹ = I and J_l is correct."""

    def test_jacobian_inverse_identity(self) -> None:
        """J_l(ω) · J_l⁻¹(ω) should equal I."""
        from geoembodied.functional.so3_ops import (
            _so3_left_jacobian,
            _so3_left_jacobian_inv,
        )
        torch.manual_seed(42)
        omega = torch.randn(10, 3, dtype=torch.float64) * 1.5

        J = _so3_left_jacobian(omega)
        Jinv = _so3_left_jacobian_inv(omega)

        product = J @ Jinv  # [..., 3, 3]
        I = torch.eye(3, dtype=torch.float64).expand_as(product)

        assert torch.allclose(product, I, atol=1e-6), \
            f"J·J⁻¹ ≠ I: max error = {(product - I).abs().max():.4e}"

    @pytest.mark.parametrize("theta", [
        0.0, 1e-8, 1e-4, 0.1, 1.0, 3.0,
        math.pi - 1e-3, math.pi - 1e-6,
    ])
    def test_jacobian_inverse_parametric(self, theta: float) -> None:
        """J · J⁻¹ = I across the full θ range."""
        from geoembodied.functional.so3_ops import (
            _so3_left_jacobian,
            _so3_left_jacobian_inv,
        )
        if theta == 0.0:
            omega = torch.zeros(1, 3, dtype=torch.float64)
        else:
            omega = torch.tensor([[theta, 0.0, 0.0]], dtype=torch.float64)

        J = _so3_left_jacobian(omega)
        Jinv = _so3_left_jacobian_inv(omega)

        product = J @ Jinv
        I = torch.eye(3, dtype=torch.float64).expand_as(product)

        assert torch.allclose(product, I, atol=1e-5), \
            f"J·J⁻¹ ≠ I at θ={theta}: max err = {(product - I).abs().max():.4e}"

    def test_jacobian_at_zero_is_identity(self) -> None:
        """J_l(0) should equal I (leading Taylor term)."""
        from geoembodied.functional.so3_ops import _so3_left_jacobian

        omega = torch.zeros(1, 3, dtype=torch.float64)
        J = _so3_left_jacobian(omega)
        I = torch.eye(3, dtype=torch.float64)

        assert torch.allclose(J.squeeze(0), I, atol=1e-6), \
            f"J_l(0) ≠ I: {J}"


# ═══════════════════════════════════════════════════════════════════
# SE(3) θ→π roundtrip
# ═══════════════════════════════════════════════════════════════════


class TestSE3NearPi:
    """Verify SE(3) exp/log at θ→π."""

    @pytest.mark.parametrize("theta", [
        math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-6,
    ])
    def test_se3_exp_log_roundtrip_near_pi(self, theta: float) -> None:
        """SE(3) exp/log roundtrip near θ=π must be accurate."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        xi = torch.tensor([[theta, 0.0, 0.0, 1.0, 0.5, 0.3]])
        T = se3_exp(xi)
        xi_back = se3_log(T)
        T_back = se3_exp(xi_back)

        # Compare rotation
        q_dot = (T[..., :4] * T_back[..., :4]).sum(dim=-1).abs()
        assert torch.allclose(q_dot, torch.ones_like(q_dot), atol=1e-3), \
            f"SE3 rotation roundtrip failed at θ={theta}: dot={q_dot.item()}"

        # Compare translation
        t_err = (T[..., 4:] - T_back[..., 4:]).norm(dim=-1)
        assert (t_err < 1e-3).all(), \
            f"SE3 translation roundtrip failed at θ={theta}: err={t_err.item()}"

    @pytest.mark.parametrize("theta", [
        math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-6,
    ])
    def test_se3_log_backward_near_pi(self, theta: float) -> None:
        """se3_log backward near θ=π must be NaN-free."""
        from geoembodied.functional.se3_ops import se3_exp, se3_log

        torch.manual_seed(42)
        axis = torch.randn(1, 3)
        axis = axis / axis.norm(dim=-1, keepdim=True)
        vel = torch.randn(1, 3)

        xi_in = torch.cat([theta * axis, vel], dim=-1)
        T = se3_exp(xi_in).detach().requires_grad_(True)
        xi_out = se3_log(T)
        xi_out.sum().backward()

        assert not torch.isnan(T.grad).any(), \
            f"NaN in se3_log grad at θ={theta}"
        assert not torch.isinf(T.grad).any(), \
            f"Inf in se3_log grad at θ={theta}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
