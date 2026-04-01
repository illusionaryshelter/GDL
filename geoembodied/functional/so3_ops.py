# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SO(3) Lie group operations — pure functional, on raw tensors.

SO(3) is the Special Orthogonal Group in 3D: the group of rotations.
Internal representation: unit quaternion [qw, qx, qy, qz] (wxyz convention).

Lie algebra so(3) = R³ (axis-angle form, or angular velocity).

Key maps:
    exp: so(3) → SO(3)   (Rodrigues' formula via quaternion)
    log: SO(3) → so(3)   (inverse Rodrigues)
    hat: R³ → so(3) matrix (3×3 skew-symmetric)
    vee: so(3) matrix → R³

All functions use Taylor expansion near singularities (θ→0, θ→π)
following Ceres Solver / GTSAM conventions.

Gradient safety: all torch.where calls use the _safe_theta sentinel pattern
from numeric_safe.py to prevent NaN gradient propagation through the
unused branch (PyTorch issue #52248).
"""

import torch
from torch import Tensor

from geoembodied.functional.numeric_safe import (
    safe_acos,
    safe_sqrt,
    taylor_sinc,
    taylor_cos_over_theta,
    taylor_theta_minus_sin_over_theta3,
    taylor_half_theta_cot_half_theta,
    lie_group_precision,
    _safe_theta,
    _EPS,
    _TAYLOR_THRESH,
)
from geoembodied.functional.quaternion_ops import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_apply,
    quaternion_to_matrix,
)


def so3_hat(omega: Tensor) -> Tensor:
    """Hat operator: R³ → so(3) (3×3 skew-symmetric matrix).

    [ω]× = [[  0, -ωz,  ωy],
             [ ωz,   0, -ωx],
             [-ωy,  ωx,   0]]

    Args:
        omega: Axis-angle vector
            shape: [..., 3], representation: so(3) angular velocity

    Returns:
        Skew-symmetric matrix, shape: [..., 3, 3]
    """
    zeros = torch.zeros_like(omega[..., 0])
    ox, oy, oz = omega.unbind(dim=-1)

    return torch.stack([
        torch.stack([zeros, -oz, oy], dim=-1),
        torch.stack([oz, zeros, -ox], dim=-1),
        torch.stack([-oy, ox, zeros], dim=-1),
    ], dim=-2)


def so3_vee(skew: Tensor) -> Tensor:
    """Vee operator: so(3) → R³ (extract vector from skew-symmetric).

    Args:
        skew: Skew-symmetric matrix
            shape: [..., 3, 3]

    Returns:
        Axis-angle vector, shape: [..., 3]
    """
    return torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0],
    ], dim=-1)


def so3_exp(omega: Tensor) -> Tensor:
    """Exponential map: so(3) → SO(3).

    Maps angular velocity vector to unit quaternion.
    Uses half-angle formula: q = [cos(θ/2), sin(θ/2) * ω/‖ω‖]
    where θ = ‖ω‖.

    Uses custom autograd Function (SO3ExpFunction) with analytic
    backward pass via half-angle derivatives, eliminating NaN risk
    from torch.where gradient routing through unused branches.

    TF32 is temporarily disabled via :func:`lie_group_precision` to
    ensure Taylor coefficients maintain full fp32 mantissa precision.

    Args:
        omega: Axis-angle vector (angular velocity × time)
            shape: [..., 3], representation: so(3) Lie algebra element

    Returns:
        Unit quaternion, shape: [..., 4]
        representation: SO(3) unit quaternion wxyz
    """
    with lie_group_precision():
        return SO3ExpFunction.apply(omega)


def so3_log(q: Tensor) -> Tensor:
    """Logarithm map: SO(3) → so(3).

    Maps unit quaternion back to axis-angle vector.
    q = [cos(θ/2), sin(θ/2) * n̂] → ω = θ * n̂

    Uses custom autograd Function (SO3LogFunction) with analytic
    backward pass, stable at both θ→0 and θ→π.

    TF32 is temporarily disabled via :func:`lie_group_precision` to
    ensure J_l⁻¹ computation maintains full fp32 mantissa precision.

    Args:
        q: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Axis-angle vector, shape: [..., 3]
        representation: so(3) Lie algebra element
    """
    with lie_group_precision():
        return SO3LogFunction.apply(q)


# ═══════════════════════════════════════════════════════════════════
# SO(3) Left Jacobian (for custom autograd backward)
# ═══════════════════════════════════════════════════════════════════


def _so3_left_jacobian(omega: Tensor) -> Tensor:
    """SO(3) left Jacobian J_l(ω).

    J_l = I + A * [ω]× + B * [ω]×²

    where:
        A = (1 - cos θ) / θ²    (taylor_cos_over_theta)
        B = (θ - sin θ) / θ³    (taylor_theta_minus_sin_over_theta3)

    Near θ→0: J_l ≈ I + ½[ω]× + ⅙[ω]×²  (from Taylor of A, B)

    Args:
        omega: Axis-angle vector
            shape: [..., 3], representation: so(3) Lie algebra element

    Returns:
        Left Jacobian matrix, shape: [..., 3, 3]
    """
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [..., 1]
    theta = safe_sqrt(theta_sq)

    A = taylor_cos_over_theta(theta)            # (1-cosθ)/θ²
    B = taylor_theta_minus_sin_over_theta3(theta)  # (θ-sinθ)/θ³

    omega_hat = so3_hat(omega)           # [..., 3, 3]
    omega_hat_sq = omega_hat @ omega_hat  # [..., 3, 3]

    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    # Broadcast A, B to matrix shape: [..., 1, 1]
    A = A.unsqueeze(-1)  # [..., 1, 1]
    B = B.unsqueeze(-1)  # [..., 1, 1]

    return I + A * omega_hat + B * omega_hat_sq


def _so3_left_jacobian_inv(omega: Tensor) -> Tensor:
    """SO(3) left Jacobian inverse J_l⁻¹(ω).

    J_l⁻¹ = I - ½[ω]× + C * [ω]×²

    where C = 1/θ² - (1+cosθ)/(2θsinθ)

    We use the equivalent form via half-angle cotangent:
        C = (1 - (θ/2)cot(θ/2)) / θ²

    Near θ→0: J_l⁻¹ ≈ I - ½[ω]× + 1/12 [ω]×²

    Note: at θ→π, the half-angle cot formula is naturally stable
    since cot(π/2) = 0, so the C coefficient → 1/π².

    Args:
        omega: Axis-angle vector
            shape: [..., 3], representation: so(3) Lie algebra element

    Returns:
        Inverse left Jacobian matrix, shape: [..., 3, 3]
    """
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [..., 1]
    theta = safe_sqrt(theta_sq)

    # (θ/2)cot(θ/2) — safe at both θ→0 and θ→π
    htcht = taylor_half_theta_cot_half_theta(theta)

    # C = (1 - (θ/2)cot(θ/2)) / θ²
    # Near θ→0: (1 - (1 - θ²/12)) / θ² = 1/12
    # Note: use theta_sq for mask, not theta, because safe_sqrt(0)
    # returns sqrt(eps) which may exceed _TAYLOR_THRESH
    small_mask = theta_sq < (_TAYLOR_THRESH * _TAYLOR_THRESH)
    theta_sq_safe = torch.where(
        small_mask, torch.ones_like(theta_sq), theta_sq
    )
    C_normal = (1.0 - htcht) / theta_sq_safe
    C_taylor = 1.0 / 12.0 + theta_sq / 720.0
    C = torch.where(small_mask, C_taylor, C_normal)

    omega_hat = so3_hat(omega)           # [..., 3, 3]
    omega_hat_sq = omega_hat @ omega_hat  # [..., 3, 3]

    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    C = C.unsqueeze(-1)  # [..., 1, 1]

    return I - 0.5 * omega_hat + C * omega_hat_sq


# ═══════════════════════════════════════════════════════════════════
# Custom Autograd Functions (analytic Jacobian backward)
# ═══════════════════════════════════════════════════════════════════


class SO3ExpFunction(torch.autograd.Function):
    """Custom autograd for so3_exp with analytic left Jacobian backward.

    Forward: ω ∈ so(3) → q ∈ SO(3) (standard so3_exp)
    Backward: dL/dω = dL/dq · ∂q/∂ω using right Jacobian

    The key insight: for so3_exp, the relationship between the
    tangent-space perturbation and the quaternion perturbation is
    mediated by the left Jacobian. Instead of computing the full
    ∂q/∂ω (which is 4×3), we use the chain rule through the
    rotation matrix parametrization and the left Jacobian.

    Specifically, for a loss L:
        dL/dω = J_l^T @ (R^T @ dL_matrix_dR + ...)

    However, a simpler and more robust approach is to compute:
        dL/dω_i = (dL/dq) · (∂q/∂ω_i)

    where ∂q/∂ω_i is computed analytically from the half-angle formula.
    """

    @staticmethod
    def forward(ctx, omega: Tensor) -> Tensor:
        """Forward: compute so3_exp(omega)."""
        q = _so3_exp_impl(omega)
        ctx.save_for_backward(omega, q)
        return q

    @staticmethod
    def backward(ctx, grad_q: Tensor) -> Tensor:
        """Backward: analytic gradient dL/dω via half-angle derivatives.

        q = [cos(θ/2), sin(θ/2)/θ · ω]

        ∂q/∂ω_i is computed analytically:
        - ∂qw/∂ω_i = -sin(θ/2) · ω_i / (2θ)
        - ∂qxyz/∂ω_i = (cos(θ/2)/(2θ) - sin(θ/2)/θ²) · ω_i · ω/θ
                      + sin(θ/2)/θ · e_i
        """
        omega, q = ctx.saved_tensors
        theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
        theta = safe_sqrt(theta_sq)

        small_mask = theta < _TAYLOR_THRESH
        theta_safe = _safe_theta(theta)

        half = 0.5 * theta_safe
        cos_half = torch.cos(half)
        sin_half = torch.sin(half)

        # sinc_half = sin(θ/2)/θ — Taylor safe
        sinc_half_taylor = 0.5 - theta_sq / 48.0
        sinc_half_normal = sin_half / theta_safe
        sinc_half = torch.where(small_mask, sinc_half_taylor, sinc_half_normal)

        # d_sinc_half = d/dθ[sin(θ/2)/θ] = cos(θ/2)/(2θ) - sin(θ/2)/θ²
        # Taylor: -1/24 · θ + ...
        d_sinc_half_taylor = -theta / 24.0
        d_sinc_half_normal = cos_half / (2.0 * theta_safe) - sin_half / (theta_safe * theta_safe)
        d_sinc_half = torch.where(small_mask, d_sinc_half_taylor, d_sinc_half_normal)

        # d_cos_half / dθ = -sin(θ/2)/2
        # dθ/dω_i = ω_i / θ
        # ∂qw/∂ω_i = -sin(θ/2)/2 · ω_i/θ = -sinc_half/2 · ω_i  (using sinc = sin/(2·θ/2))
        # Actually: dqw/dω_i = -sin(θ/2)/(2θ) · ω_i
        d_qw_coeff_taylor = -0.25 + theta_sq / 96.0  # -sin(θ/2)/(2θ) Taylor
        d_qw_coeff_normal = -sin_half / (2.0 * theta_safe)
        d_qw_coeff = torch.where(small_mask, d_qw_coeff_taylor, d_qw_coeff_normal)

        # ∂qxyz/∂ω_i = sinc_half · δ_ij + d_sinc_half · ω_i · ω_j / θ
        # dθ/dω_i = ω_i / θ, so chain rule:
        # ∂(sinc_half · ω_j)/∂ω_i = sinc_half·δ_{ij} + d_sinc_half·(ω_i/θ)·ω_j

        # dL/dω = grad_qw * d_qw + grad_qxyz * d_qxyz
        grad_qw = grad_q[..., 0:1]     # [..., 1]
        grad_xyz = grad_q[..., 1:4]     # [..., 3]

        # Contribution from qw: dL/dω_i += grad_qw * d_qw_coeff * ω_i
        grad_omega = d_qw_coeff * omega * grad_qw

        # Contribution from qxyz:
        # sinc_half * grad_xyz  (identity part)
        grad_omega = grad_omega + sinc_half * grad_xyz

        # d_sinc_half * (ω · grad_xyz) * ω / θ  (outer product part)
        omega_dot_grad = (omega * grad_xyz).sum(dim=-1, keepdim=True)
        theta_safe_for_div = torch.where(
            small_mask, torch.ones_like(theta), theta
        )
        grad_omega = grad_omega + d_sinc_half * omega_dot_grad * omega / theta_safe_for_div

        return grad_omega


class SO3LogFunction(torch.autograd.Function):
    """Custom autograd for so3_log with analytic inverse Jacobian backward.

    Forward: q ∈ SO(3) → ω ∈ so(3) (standard so3_log)
    Backward: dL/dq = dL/dω · J_l⁻¹ chain mapped back to quaternion space
    """

    @staticmethod
    def forward(ctx, q: Tensor) -> Tensor:
        """Forward: compute so3_log(q)."""
        omega = _so3_log_impl(q)
        ctx.save_for_backward(q, omega)
        return omega

    @staticmethod
    def backward(ctx, grad_omega: Tensor) -> Tensor:
        """Backward: analytic gradient dL/dq.

        ω = scale(q) · xyz where scale = 2·atan2(‖xyz‖, w) / ‖xyz‖

        dL/dq = dL/dω · ∂ω/∂q

        ∂ω/∂q involves the derivative of scale w.r.t. w and ‖xyz‖.
        """
        q, omega = ctx.saved_tensors
        q = quaternion_normalize(q)
        # Enforce canonical form
        q = torch.where(q[..., 0:1] < 0, -q, q)

        w = q[..., 0:1]
        xyz = q[..., 1:4]

        sin_half_sq = (xyz * xyz).sum(dim=-1, keepdim=True)
        sin_half = safe_sqrt(sin_half_sq)
        half_theta = torch.atan2(sin_half, w)

        small_mask = sin_half < _TAYLOR_THRESH

        # scale = 2 * half_theta / sin_half
        sin_half_safe = torch.where(
            small_mask, torch.ones_like(sin_half), sin_half
        )
        scale = torch.where(
            small_mask,
            2.0 + sin_half_sq / 3.0,
            2.0 * half_theta / sin_half_safe,
        )

        # ∂scale/∂(sin_half): d/ds[2·atan2(s,w)/s]
        # = 2 * (w/(s²+w²) · s - atan2(s,w)) / s²
        # = 2 * (w/(s²+w²) - atan2(s,w)/s) / s
        # Taylor near s→0: -2/3 (leading correction)
        norm_sq = sin_half_sq + w * w  # should be 1 for unit quaternion
        d_scale_d_s_normal = (
            2.0 * w / (norm_sq * sin_half_safe)
            - 2.0 * half_theta / (sin_half_safe * sin_half_safe)
        )
        d_scale_d_s_taylor = -2.0 / 3.0 * torch.ones_like(sin_half)
        d_scale_d_s = torch.where(small_mask, d_scale_d_s_taylor, d_scale_d_s_normal)

        # ∂scale/∂w = 2 * sin_half / (sin_half² + w²) / sin_half
        #           = 2 / (sin_half² + w²)
        # But ∂(atan2(s,w))/∂w = -s/(s²+w²), so:
        # ∂scale/∂w = 2 * (-sin_half/(sin_half²+w²)) / sin_half_safe = -2/(sin_half²+w²)
        d_scale_d_w = -2.0 / norm_sq

        # ω = scale * xyz
        # dL/dq_w = Σ_j (dL/dω_j * ∂ω_j/∂q_w) = Σ_j (grad_ω_j * d_scale_d_w * xyz_j)
        grad_qw = d_scale_d_w * (grad_omega * xyz).sum(dim=-1, keepdim=True)

        # dL/dq_xyz_i = grad_ω_i * scale + Σ_j (grad_ω_j * d_scale_d_s * ∂s/∂xyz_i * xyz_j)
        # ∂s/∂xyz_i = xyz_i / s
        # So: dL/dq_xyz_i = scale * grad_ω_i + d_scale_d_s / s * xyz_i * Σ_j(grad_ω_j * xyz_j)
        omega_dot_grad = (grad_omega * xyz).sum(dim=-1, keepdim=True)
        grad_qxyz = scale * grad_omega + d_scale_d_s * xyz * omega_dot_grad / sin_half_safe

        grad_q = torch.cat([grad_qw, grad_qxyz], dim=-1)
        return grad_q


# ── Internal implementations (called by custom autograd) ──

def _so3_exp_impl(omega: Tensor) -> Tensor:
    """Pure forward computation of so3_exp (no autograd hooks)."""
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    theta = safe_sqrt(theta_sq)

    small_mask = theta < _TAYLOR_THRESH

    half_theta_f64 = (0.5 * theta).double()
    w = torch.cos(half_theta_f64).to(theta.dtype)
    sin_half_f64 = torch.sin(half_theta_f64)

    sinc_half_taylor = 0.5 - theta_sq / 48.0 + theta_sq * theta_sq / 3840.0
    theta_safe = _safe_theta(theta)
    sinc_half_normal = sin_half_f64.to(theta.dtype) / theta_safe

    sinc_half = torch.where(small_mask, sinc_half_taylor, sinc_half_normal)
    xyz = sinc_half * omega

    q = torch.cat([w, xyz], dim=-1)
    return quaternion_normalize(q)


def _so3_log_impl(q: Tensor) -> Tensor:
    """Pure forward computation of so3_log (no autograd hooks)."""
    q = quaternion_normalize(q)
    q = torch.where(q[..., 0:1] < 0, -q, q)

    w = q[..., 0:1]
    xyz = q[..., 1:4]

    sin_half_sq = (xyz * xyz).sum(dim=-1, keepdim=True)
    sin_half = safe_sqrt(sin_half_sq)

    half_theta = torch.atan2(sin_half, w)

    small_mask = sin_half < _TAYLOR_THRESH
    scale_taylor = 2.0 + sin_half_sq / 3.0
    sin_half_safe = torch.where(
        small_mask, torch.ones_like(sin_half), sin_half
    )
    scale_normal = 2.0 * half_theta / sin_half_safe
    scale = torch.where(small_mask, scale_taylor, scale_normal)

    return scale * xyz


def so3_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """SO(3) group multiplication via quaternion Hamilton product.

    Args:
        q1, q2: Unit quaternions
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Product quaternion, shape: [..., 4]
    """
    return quaternion_normalize(quaternion_multiply(q1, q2))


def so3_inverse(q: Tensor) -> Tensor:
    """SO(3) group inverse (= quaternion conjugate for unit quaternions).

    Args:
        q: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Inverse quaternion, shape: [..., 4]
    """
    return quaternion_conjugate(q)


def so3_act(q: Tensor, v: Tensor) -> Tensor:
    """Apply SO(3) rotation to 3D vectors.

    Computes v' = R(q) @ v using optimized quaternion formula.

    Args:
        q: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz
        v: 3D vectors
            shape: [..., 3]

    Returns:
        Rotated vectors, shape: [..., 3]
    """
    return quaternion_apply(q, v)


def so3_adjoint(q: Tensor) -> Tensor:
    """Adjoint representation of SO(3) element.

    For SO(3), Ad(R) = R (the rotation matrix itself).

    Args:
        q: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Adjoint matrix (= rotation matrix), shape: [..., 3, 3]
    """
    return quaternion_to_matrix(q)
