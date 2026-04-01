# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3) Lie group operations — pure functional, on raw tensors.

SE(3) is the Special Euclidean Group in 3D: rigid body transformations
(rotation + translation). The group of proper isometries in R³.

Internal representation: 7-dim compact vector [qw, qx, qy, qz, tx, ty, tz]
    - Quaternion part [..., :4]: wxyz unit quaternion (SO(3) rotation)
    - Translation part [..., 4:7]: xyz translation
    - Total: 7 floats (vs 16 for 4×4 matrix → 56% memory saving)

Lie algebra se(3) = R⁶ = [ω (angular), v (linear)]
    - ω ∈ R³: angular velocity (so(3) part)
    - v ∈ R³: linear velocity

Key identity:
    T_new = exp(Δξ^) ∘ T_old   (AGENTS.md Rule 2)

Gradient safety: all torch.where calls use the _safe_theta sentinel pattern
to prevent NaN gradient propagation through unused branches.
"""

import torch
from torch import Tensor

from geoembodied.functional.numeric_safe import (
    safe_sqrt,
    taylor_sinc,
    taylor_cos_over_theta,
    taylor_theta_minus_sin_over_theta3,
    taylor_V_inv_coeff,
    lie_group_precision,
    _safe_theta,
    _EPS,
    _TAYLOR_THRESH,
)
from geoembodied.functional.so3_ops import (
    so3_exp,
    so3_log,
    so3_hat,
)
from geoembodied.functional.quaternion_ops import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_apply,
    quaternion_to_matrix,
)


# ── Helper: Extract rotation and translation from compact 7-dim ──


def _split_se3(T: Tensor) -> tuple[Tensor, Tensor]:
    """Split SE(3) compact form into quaternion and translation.

    Args:
        T: SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]

    Returns:
        q: Unit quaternion, shape: [..., 4]
        t: Translation, shape: [..., 3]
    """
    return T[..., :4], T[..., 4:7]


def _join_se3(q: Tensor, t: Tensor) -> Tensor:
    """Join quaternion and translation into SE(3) compact form.

    Args:
        q: Unit quaternion, shape: [..., 4]
        t: Translation, shape: [..., 3]

    Returns:
        SE(3) element, shape: [..., 7]
    """
    return torch.cat([quaternion_normalize(q), t], dim=-1)


# ── Core SE(3) operations ──


def se3_identity(
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Create identity SE(3) element(s).

    Args:
        batch_shape: Shape of batch dimensions
        dtype: Tensor dtype
        device: Tensor device

    Returns:
        Identity transform [1,0,0,0, 0,0,0], shape: [*batch_shape, 7]
    """
    identity = torch.zeros(*batch_shape, 7, dtype=dtype, device=device)
    identity[..., 0] = 1.0  # qw = 1
    return identity


def se3_exp(xi: Tensor) -> Tensor:
    """Exponential map: se(3) → SE(3).

    Maps twist vector ξ = [ω, v] ∈ R⁶ to SE(3) compact form.

    Uses the closed-form formula:
        R = exp(ω^)                    (SO(3) exponential map)
        t = V(ω) @ v                   (left Jacobian of SO(3))

    where V(ω) = I + ((1-cos θ)/θ²)[ω]× + ((θ - sin θ)/θ³)[ω]×²

    TF32 is temporarily disabled via :func:`lie_group_precision` to
    ensure V matrix Taylor coefficients maintain full fp32 precision.

    Args:
        xi: Twist vector [angular, linear]
            shape: [..., 6], representation: se(3) Lie algebra element

    Returns:
        SE(3) compact form, shape: [..., 7]
        representation: SE(3) compact [qw,qx,qy,qz,tx,ty,tz]
    """
    with lie_group_precision():
        return _se3_exp_impl(xi)


def _se3_exp_impl(xi: Tensor) -> Tensor:
    omega = xi[..., :3]   # [..., 3] angular part
    vel = xi[..., 3:6]    # [..., 3] linear part

    # SO(3) exponential: quaternion from axis-angle
    q = so3_exp(omega)  # [..., 4]

    # Compute left Jacobian V(ω) to get translation
    # V @ v = v + A*(ω × v) + B*(ω × (ω × v))
    # where A = (1-cos θ)/θ², B = (θ - sin θ)/θ³
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [..., 1]
    theta = safe_sqrt(theta_sq)

    # Use shared Taylor-safe functions (with _safe_theta sentinel inside)
    A = taylor_cos_over_theta(theta)           # (1-cosθ)/θ²
    B = taylor_theta_minus_sin_over_theta3(theta)  # (θ-sinθ)/θ³

    # ω × v
    omega_cross_v = torch.linalg.cross(omega, vel, dim=-1)
    # ω × (ω × v)
    omega_cross_omega_cross_v = torch.linalg.cross(omega, omega_cross_v, dim=-1)

    # t = v + A * (ω × v) + B * (ω × (ω × v))
    # This is V(ω) @ v
    t = vel + A * omega_cross_v + B * omega_cross_omega_cross_v

    return _join_se3(q, t)


def se3_log(T: Tensor) -> Tensor:
    """Logarithm map: SE(3) → se(3).

    Maps SE(3) compact form back to twist vector ξ = [ω, v].

    Uses V⁻¹(ω) to recover linear velocity from translation:
        ω = log_SO3(R)
        v = V⁻¹(ω) @ t

    The V⁻¹ coefficient (1/θ² - (1+cosθ)/(2θsinθ)) suffers from
    catastrophic ∞-∞ cancellation at θ→0. We use taylor_V_inv_coeff()
    with the _safe_theta sentinel to handle this correctly.

    TF32 is temporarily disabled via :func:`lie_group_precision` to
    ensure V⁻¹ coefficient computation maintains full fp32 precision.

    Args:
        T: SE(3) compact form
            shape: [..., 7], representation: SE(3) compact [q|t]

    Returns:
        Twist vector [angular, linear]
            shape: [..., 6], representation: se(3) Lie algebra element
    """
    with lie_group_precision():
        return _se3_log_impl(T)


def _se3_log_impl(T: Tensor) -> Tensor:
    """Implementation of se3_log (called inside lie_group_precision)."""
    q, t = _split_se3(T)

    # SO(3) logarithm
    omega = so3_log(q)  # [..., 3]

    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [..., 1]
    theta = safe_sqrt(theta_sq)

    # Compute V⁻¹(ω) @ t
    # V⁻¹ = I - ½[ω]× + C*[ω]×²
    # where C = 1/θ² - (1+cosθ)/(2θsinθ)
    # This C has catastrophic cancellation at θ→0 (∞ - ∞)!
    # taylor_V_inv_coeff uses Taylor: C ≈ 1/12 + θ²/720 + θ⁴/30240
    C = taylor_V_inv_coeff(theta)

    # ω × t
    omega_cross_t = torch.linalg.cross(omega, t, dim=-1)
    # ω × (ω × t)
    omega_cross_omega_cross_t = torch.linalg.cross(omega, omega_cross_t, dim=-1)

    # v = V⁻¹ @ t = t - 0.5 * (ω × t) + C * (ω × (ω × t))
    vel = t - 0.5 * omega_cross_t + C * omega_cross_omega_cross_t

    return torch.cat([omega, vel], dim=-1)  # [..., 6]


def se3_multiply(T1: Tensor, T2: Tensor) -> Tensor:
    """SE(3) group multiplication (composition of rigid transforms).

    (R1, t1) ∘ (R2, t2) = (R1 R2, R1 t2 + t1)

    Args:
        T1: First SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]
        T2: Second SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]

    Returns:
        Product T1 ∘ T2, shape: [..., 7]
    """
    q1, t1 = _split_se3(T1)
    q2, t2 = _split_se3(T2)

    # Rotation: R1 @ R2 via quaternion product
    q = quaternion_multiply(q1, q2)

    # Translation: R1 @ t2 + t1
    t = quaternion_apply(q1, t2) + t1

    return _join_se3(q, t)


def se3_inverse(T: Tensor) -> Tensor:
    """SE(3) group inverse.

    (R, t)⁻¹ = (R^T, -R^T t)

    Args:
        T: SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]

    Returns:
        Inverse transform, shape: [..., 7]
    """
    q, t = _split_se3(T)

    # R⁻¹ = R^T = q* (conjugate)
    q_inv = quaternion_conjugate(q)

    # t_inv = -R^T @ t = -q* ⊳ t
    t_inv = -quaternion_apply(q_inv, t)

    return _join_se3(q_inv, t_inv)


def se3_act(T: Tensor, points: Tensor) -> Tensor:
    """Apply SE(3) rigid body transform to 3D points.

    p' = R @ p + t

    Args:
        T: SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]
        points: 3D points
            shape: [..., N, 3] or [..., 3]

    Returns:
        Transformed points, shape: same as input points
    """
    q, t = _split_se3(T)

    # Handle broadcasting for batched points
    if points.dim() > q.dim():
        # points: [..., N, 3], q: [..., 4] → need q: [..., 1, 4]
        q = q.unsqueeze(-2)
        t = t.unsqueeze(-2)

    return quaternion_apply(q, points) + t


def se3_adjoint(T: Tensor) -> Tensor:
    """Adjoint representation of SE(3) element.

    Ad(T) = [[R, [t]× R],
             [0,    R   ]]  ∈ R^{6×6}

    Maps twists from one frame to another.

    Args:
        T: SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]

    Returns:
        Adjoint matrix, shape: [..., 6, 6]
    """
    q, t = _split_se3(T)

    R = quaternion_to_matrix(q)         # [..., 3, 3]
    t_hat = so3_hat(t)                  # [..., 3, 3]
    t_hat_R = torch.matmul(t_hat, R)    # [..., 3, 3]

    zeros = torch.zeros_like(R)  # [..., 3, 3]

    # Build 6×6 block matrix
    top = torch.cat([R, t_hat_R], dim=-1)      # [..., 3, 6]
    bottom = torch.cat([zeros, R], dim=-1)     # [..., 3, 6]

    return torch.cat([top, bottom], dim=-2)    # [..., 6, 6]
