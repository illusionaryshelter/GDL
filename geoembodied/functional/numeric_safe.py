# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Numerically safe wrappers for dangerous mathematical operations.

GDL is extremely prone to gradient explosion and NaN propagation.
Every function in this module enforces clamping/Taylor-expansion
guards per AGENTS.md Rules 4-5.

All functions are pure (stateless) and operate on raw torch.Tensor.

## Critical Design: ``safe_where`` and the NaN gradient problem

``torch.where(mask, f(x), g(x))`` computes **both branches** during backward,
even for elements where only one branch was selected in forward. If the
unselected branch produces NaN (e.g. sin(θ)/θ at θ=0), the NaN gradient
propagates into the final gradient. See PyTorch issue #52248.

Fix: before computing f(x) in the "normal" branch, replace dangerous inputs
with a safe sentinel value (e.g. clamp theta away from 0). The forward output
is correct because ``torch.where`` selects the Taylor branch for those elements.
The backward gradient on the "normal" branch is now finite (though incorrect),
but it's multiplied by ``1 - mask = 0`` in the ``where`` backward, so it
doesn't affect the final gradient. This is the approach used by JAX, Sophus,
and Ceres Solver.

## TF32 Precision Guard

TF32 (TensorFloat-32) on Ampere+ GPUs truncates matmul mantissa from 23
to 10 bits. This is acceptable for neural network weight matmuls but
catastrophic for Lie group operations where Taylor expansions at θ→0
and ∞-∞ cancellation at θ→π require full fp32 precision.

Use :func:`lie_group_precision` as a context manager around Lie group
computations to ensure strict fp32 regardless of global TF32 settings.
"""

from contextlib import contextmanager

import torch
from torch import Tensor

# Global numerical epsilon — used in all safe ops
_EPS: float = 1e-7

# Threshold for switching to Taylor expansion near singularities
# Following Ceres Solver / GTSAM convention
_TAYLOR_THRESH: float = 1e-4


@contextmanager
def lie_group_precision():
    """Temporarily enforce strict fp32 for Lie group operations.

    On Ampere+ GPUs, TF32 reduces matmul mantissa from 23→10 bits.
    This is fine for neural network weight matmuls (SE3Conv messages)
    but catastrophic for:

    - Taylor expansion coefficients at θ→0 (sinc, cos/θ², etc.)
    - V⁻¹ coefficient at θ→π (∞-∞ cancellation needs full mantissa)
    - Quaternion normalization (drift accumulates with low precision)

    Usage::

        with lie_group_precision():
            q = so3_exp(omega)   # strict fp32 matmul
            omega = so3_log(q)   # strict fp32 matmul

    Outside this context, neural network matmuls retain TF32 speed.
    """
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul


def _safe_theta(theta: Tensor) -> Tensor:
    """Replace near-zero theta with a safe sentinel for the 'normal' branch.

    This ensures sin(θ)/θ, (1-cos θ)/θ², etc. never see θ=0 as input,
    so their gradients are always finite. The sentinel value (1.0) is
    arbitrary — the output of the normal branch at the sentinel is
    discarded by torch.where anyway.

    Args:
        theta: Rotation angle tensor
            shape: arbitrary

    Returns:
        theta with |θ| < _TAYLOR_THRESH replaced by 1.0
    """
    return torch.where(theta.abs() < _TAYLOR_THRESH, torch.ones_like(theta), theta)


def safe_acos(x: Tensor) -> Tensor:
    """Numerically safe arccos — clamps input away from ±1.

    At x = ±1, d(acos)/dx → -∞, causing NaN gradients in backprop.

    Args:
        x: Input tensor, expected in [-1, 1]
            shape: arbitrary

    Returns:
        acos(clamp(x)), same shape as input
    """
    return torch.acos(x.clamp(-1.0 + _EPS, 1.0 - _EPS))


def safe_sqrt(x: Tensor) -> Tensor:
    """Numerically safe sqrt — clamps input away from 0.

    At x = 0, d(sqrt)/dx → ∞, causing NaN gradients.

    Args:
        x: Input tensor, expected ≥ 0
            shape: arbitrary

    Returns:
        sqrt(clamp(x)), same shape as input
    """
    return torch.sqrt(x.clamp(min=_EPS))


def taylor_sinc(theta: Tensor) -> Tensor:
    """Computes sin(θ)/θ with Taylor expansion near θ = 0.

    Near θ = 0, the analytic form sin(θ)/θ suffers from 0/0.
    Standard fix used by Ceres Solver, GTSAM, Sophus:
        When |θ| < threshold, use Taylor: 1 - θ²/6 + θ⁴/120

    Uses _safe_theta() to prevent NaN gradient routing through torch.where.

    Args:
        theta: Rotation angle tensor
            shape: arbitrary

    Returns:
        sin(θ)/θ, stable everywhere, same shape as input
    """
    small_mask = theta.abs() < _TAYLOR_THRESH
    theta_sq = theta * theta

    # Taylor expansion: sinc(θ) ≈ 1 - θ²/6 + θ⁴/120
    taylor_result = 1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0

    # Normal computation — use _safe_theta to avoid 0/0 in backward
    theta_safe = _safe_theta(theta)
    normal_result = torch.sin(theta_safe) / theta_safe

    return torch.where(small_mask, taylor_result, normal_result)


def taylor_cos_over_theta(theta: Tensor) -> Tensor:
    """Computes (1 - cos(θ)) / θ² with Taylor expansion near θ = 0.

    Near θ = 0, this is 0/0. Taylor: 1/2 - θ²/24 + θ⁴/720

    Uses _safe_theta() to prevent NaN gradient routing through torch.where.

    Args:
        theta: Rotation angle tensor
            shape: arbitrary

    Returns:
        (1 - cos(θ)) / θ², stable everywhere, same shape as input
    """
    small_mask = theta.abs() < _TAYLOR_THRESH
    theta_sq = theta * theta

    # Taylor: (1 - cos(θ))/θ² ≈ 1/2 - θ²/24 + θ⁴/720
    taylor_result = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0

    # Normal computation — safe denominator
    theta_safe = _safe_theta(theta)
    theta_sq_safe = theta_safe * theta_safe
    normal_result = (1.0 - torch.cos(theta_safe)) / theta_sq_safe

    return torch.where(small_mask, taylor_result, normal_result)


def taylor_theta_minus_sin_over_theta3(theta: Tensor) -> Tensor:
    """Computes (θ - sin(θ)) / θ³ with Taylor expansion near θ = 0.

    Used in SE(3) exponential map for the left Jacobian V(ω).

    Near θ = 0: Taylor = 1/6 - θ²/120 + θ⁴/5040

    Args:
        theta: Rotation angle tensor
            shape: arbitrary

    Returns:
        (θ - sin θ) / θ³, stable everywhere, same shape as input
    """
    small_mask = theta.abs() < _TAYLOR_THRESH
    theta_sq = theta * theta

    # Taylor: 1/6 - θ²/120 + θ⁴/5040
    taylor_result = (
        1.0 / 6.0
        - theta_sq / 120.0
        + theta_sq * theta_sq / 5040.0
    )

    # Normal computation
    theta_safe = _safe_theta(theta)
    theta_sq_safe = theta_safe * theta_safe
    normal_result = (theta_safe - torch.sin(theta_safe)) / (theta_sq_safe * theta_safe)

    return torch.where(small_mask, taylor_result, normal_result)


def taylor_V_inv_coeff(theta: Tensor) -> Tensor:
    """Computes the V⁻¹ coefficient: 1/θ² - (1 + cos θ)/(2θ sin θ).

    Used in SE(3) logarithm map for computing V⁻¹(ω) @ t.

    This is the most dangerous function in the entire library.
    It has TWO singularities:

    1. θ → 0: ∞ - ∞ (catastrophic cancellation).
       Taylor: 1/12 + θ²/720 + θ⁴/30240

    2. θ → π: sin(θ) → 0, so (1+cosθ)/(2θsinθ) blows up.
       But at θ=π: cosθ = -1, so numerator (1+cosθ) = 0 too.
       Limit: 1/π² ≈ 0.10132 (by L'Hôpital or half-angle identity)
       Taylor around δ = π - θ:
         C(π - δ) ≈ 1/π² - (1/(6π) - 1/π³)·δ + O(δ²)

    .. warning:: **Three-way torch.where gradient routing.**
       The normal branch contains ``1 / sin(θ)`` which diverges at
       θ = π. Even when ``near_pi_mask`` selects the limit result,
       the normal branch's gradient (∞) still flows backward through
       the ``where`` Jacobian. Fix: sentinel-replace θ with 1.0
       in the normal branch for BOTH θ→0 and θ→π regimes.

    Args:
        theta: Rotation angle tensor
            shape: arbitrary

    Returns:
        V⁻¹ coefficient, stable everywhere, same shape as input
    """
    theta_sq = theta * theta

    # ── Branch 1: θ → 0 (Taylor) ──
    small_mask = theta.abs() < _TAYLOR_THRESH

    # Taylor: 1/12 + θ²/720 + θ⁴/30240
    taylor_result = (
        1.0 / 12.0
        + theta_sq / 720.0
        + theta_sq * theta_sq / 30240.0
    )

    # ── Branch 2: θ → π (limit + correction) ──
    _PI_THRESH = 1e-3
    delta = torch.pi - theta.abs()  # distance to π
    near_pi_mask = delta.abs() < _PI_THRESH

    # At θ = π: C = 1/π²
    # Half-angle form: C = 1/θ² - cot(θ/2)/(2θ)
    # Taylor in δ = π - θ:
    #   1/(π-δ)² ≈ 1/π² + 2δ/π³ + 3δ²/π⁴
    #   cot(θ/2)/(2θ) = tan(δ/2)/(2(π-δ)) ≈ δ/(4π) + δ²/(4π²)
    # C(π-δ) ≈ 1/π² + (2/π³ - 1/(4π))·δ + (3/π⁴ - 1/(4π²))·δ²
    _inv_pi_sq = 1.0 / (torch.pi * torch.pi)
    # (8 - π²) / (4π³) ≈ -0.01507
    _corr_coeff_1 = 2.0 / (torch.pi ** 3) - 1.0 / (4.0 * torch.pi)
    # (12 - π²) / (4π⁴) ≈ 0.005468
    _corr_coeff_2 = 3.0 / (torch.pi ** 4) - 1.0 / (4.0 * torch.pi ** 2)
    near_pi_result = _inv_pi_sq + _corr_coeff_1 * delta + _corr_coeff_2 * delta * delta

    # ── Branch 3: Normal range ──
    # CRITICAL: sentinel-replace theta for BOTH singularities.
    # Without this, the 1/sin(θ) gradient at θ=π leaks ∞ into backward
    # even when near_pi_mask selects the limit branch.
    safe_mask = small_mask | near_pi_mask
    theta_safe = torch.where(safe_mask, torch.ones_like(theta), theta)
    theta_sq_safe = theta_safe * theta_safe
    normal_result = (
        1.0 / theta_sq_safe
        - (1.0 + torch.cos(theta_safe))
        / (2.0 * theta_safe * torch.sin(theta_safe))
    )

    # Combine: small_mask takes priority, then near_pi_mask, then normal
    result = torch.where(near_pi_mask, near_pi_result, normal_result)
    result = torch.where(small_mask, taylor_result, result)
    return result


def taylor_half_theta_cot_half_theta(theta: Tensor) -> Tensor:
    """Computes (θ/2) * cot(θ/2) = (θ/2) * cos(θ/2) / sin(θ/2).

    Used in so3_log backward (Jacobian of the log map).

    Near θ = 0: Taylor = 1 - θ²/12 - θ⁴/720
    Near θ = π: this approaches 0, which is fine.

    Args:
        theta: Rotation angle tensor
            shape: arbitrary

    Returns:
        (θ/2) * cot(θ/2), stable everywhere, same shape as input
    """
    small_mask = theta.abs() < _TAYLOR_THRESH
    theta_sq = theta * theta

    # Taylor: 1 - θ²/12 - θ⁴/720
    taylor_result = 1.0 - theta_sq / 12.0 - theta_sq * theta_sq / 720.0

    # Normal computation
    theta_safe = _safe_theta(theta)
    half = 0.5 * theta_safe
    normal_result = half * torch.cos(half) / torch.sin(half)

    return torch.where(small_mask, taylor_result, normal_result)
