# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Quaternion operations — pure functional, on raw tensors.

Convention: wxyz (Hamilton convention)
    q = [qw, qx, qy, qz] where qw is the scalar part.
    Storage shape: [..., 4]

All operations preserve autograd graph and are numerically safe.
"""

import torch
from torch import Tensor

from geoembodied.functional.numeric_safe import safe_acos, safe_sqrt, _EPS


def quaternion_normalize(q: Tensor) -> Tensor:
    """Normalize quaternion to unit norm.

    Projects floating-point-drifted quaternions back to S³.
    Per AGENTS.md Rule 5: must be called after repeated multiplications.

    Args:
        q: Quaternion tensor
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Unit quaternion, shape: [..., 4]
    """
    return q / safe_sqrt((q * q).sum(dim=-1, keepdim=True))


def quaternion_conjugate(q: Tensor) -> Tensor:
    """Quaternion conjugate (= inverse for unit quaternions).

    For unit q: q* = q⁻¹, so q ⊗ q* = [1, 0, 0, 0].

    Args:
        q: Quaternion tensor
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Conjugate quaternion, shape: [..., 4]
    """
    # Negate imaginary parts: [w, -x, -y, -z]
    signs = q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return q * signs


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternions.

    Implements q1 ⊗ q2 using the standard Hamilton product formula.
    This corresponds to rotation composition: R(q1 ⊗ q2) = R(q1) R(q2).

    Args:
        q1: First quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz
        q2: Second quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Product quaternion q1 ⊗ q2, shape: [..., 4]
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_apply(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3D vectors by unit quaternions.

    Uses the formula: v' = q ⊗ [0, v] ⊗ q*
    Optimized to avoid full quaternion multiply.

    Args:
        q: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz
        v: 3D vectors to rotate
            shape: [..., 3]

    Returns:
        Rotated vectors, shape: [..., 3]
    """
    # Extract scalar and vector parts
    q_w = q[..., 0:1]   # [..., 1]
    q_vec = q[..., 1:4]  # [..., 3]

    # t = 2 * (q_vec × v)
    t = 2.0 * torch.linalg.cross(q_vec, v, dim=-1)

    # v' = v + q_w * t + q_vec × t
    return v + q_w * t + torch.linalg.cross(q_vec, t, dim=-1)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion to 3×3 rotation matrix.

    Args:
        q: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Rotation matrix, shape: [..., 3, 3]
        representation: SO(3) rotation matrix (det=1, orthogonal)
    """
    # Ensure unit norm for numerical safety
    q = quaternion_normalize(q)

    w, x, y, z = q.unbind(dim=-1)

    # Precompute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Build rotation matrix rows
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)

    return matrix


def quaternion_from_matrix(matrix: Tensor) -> Tensor:
    """Convert 3×3 rotation matrix to unit quaternion.

    Uses Shepperd's method for numerical stability across all rotations.

    Args:
        matrix: Rotation matrix
            shape: [..., 3, 3], representation: SO(3) rotation matrix

    Returns:
        Unit quaternion, shape: [..., 4]
        representation: SO(3) unit quaternion wxyz
    """
    batch_shape = matrix.shape[:-2]

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    trace = m00 + m11 + m22

    # Shepperd's method: choose the largest diagonal element
    # to avoid division by small numbers

    # Case 1: trace > 0
    s1 = safe_sqrt(trace + 1.0) * 2.0  # s = 4w
    w1 = 0.25 * s1
    x1 = (m21 - m12) / s1
    y1 = (m02 - m20) / s1
    z1 = (m10 - m01) / s1

    # Case 2: m00 is largest diagonal
    s2 = safe_sqrt(1.0 + m00 - m11 - m22) * 2.0
    w2 = (m21 - m12) / s2
    x2 = 0.25 * s2
    y2 = (m01 + m10) / s2
    z2 = (m02 + m20) / s2

    # Case 3: m11 is largest diagonal
    s3 = safe_sqrt(1.0 + m11 - m00 - m22) * 2.0
    w3 = (m02 - m20) / s3
    x3 = (m01 + m10) / s3
    y3 = 0.25 * s3
    z3 = (m12 + m21) / s3

    # Case 4: m22 is largest diagonal
    s4 = safe_sqrt(1.0 + m22 - m00 - m11) * 2.0
    w4 = (m10 - m01) / s4
    x4 = (m02 + m20) / s4
    y4 = (m12 + m21) / s4
    z4 = 0.25 * s4

    # Select based on which case gives best numerical stability
    cond1 = trace > 0
    cond2 = (m00 > m11) & (m00 > m22) & ~cond1
    cond3 = (m11 > m22) & ~cond1 & ~cond2

    w = torch.where(cond1, w1, torch.where(cond2, w2, torch.where(cond3, w3, w4)))
    x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
    y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
    z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))

    q = torch.stack([w, x, y, z], dim=-1)

    # Enforce canonical form (w >= 0) for uniqueness
    q = torch.where(q[..., 0:1] < 0, -q, q)

    return quaternion_normalize(q)


def quaternion_slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation between two unit quaternions.

    Args:
        q0: Start quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz
        q1: End quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz
        t: Interpolation parameter in [0, 1]
            shape: [...] or scalar

    Returns:
        Interpolated quaternion, shape: [..., 4]
    """
    # Ensure shortest path
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    # Compute angle
    theta = safe_acos(dot.clamp(-1.0, 1.0))

    # Ensure t has correct shape
    if t.dim() < q0.dim():
        t = t.unsqueeze(-1)

    # Near-zero angle: linear interpolation
    sin_theta = torch.sin(theta).clamp(min=_EPS)
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    # Fallback to linear for very small angles
    small = theta.abs() < 1e-4
    s0 = torch.where(small, 1.0 - t, s0)
    s1 = torch.where(small, t, s1)

    return quaternion_normalize(s0 * q0 + s1 * q1)
