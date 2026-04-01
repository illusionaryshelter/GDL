# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Distance metrics on Lie groups — pure functional.

Provides both chordal (fast, no singularities) and geodesic
(mathematically exact, needs Taylor guards) distance metrics.

For LieGroupAttention, chordal distance is the recommended default
due to zero numerical risk and first-order equivalence to geodesic.
"""

import torch
from torch import Tensor

from geoembodied.functional.numeric_safe import safe_acos, safe_sqrt, _EPS
from geoembodied.functional.so3_ops import so3_log
from geoembodied.functional.se3_ops import se3_inverse, se3_multiply, se3_log
from geoembodied.functional.quaternion_ops import (
    quaternion_to_matrix,
    quaternion_multiply,
    quaternion_conjugate,
)


def so3_chordal_distance(q1: Tensor, q2: Tensor) -> Tensor:
    """Chordal distance between two SO(3) rotations.

    d_chordal(R1, R2) = ‖R1 - R2‖_F²

    Properties:
    - No trigonometric functions → no singularities
    - First-order equivalent to geodesic distance for small perturbations
    - Ideal for attention score computation in large-scale point clouds

    The Frobenius norm squared can be computed efficiently from quaternions:
        ‖R1 - R2‖_F² = 6 - 2 * Tr(R1^T R2)
    And Tr(R1^T R2) = 2(q1 · q2)² - 1 (via quaternion inner product).

    Args:
        q1: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz
        q2: Unit quaternion
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Squared chordal distance, shape: [...]
    """
    # Inner product of quaternions
    dot = (q1 * q2).sum(dim=-1)  # [...]

    # Tr(R1^T R2) = 2 * dot² - 1
    trace = 2.0 * dot * dot - 1.0

    # ‖R1 - R2‖_F² = 6 - 2 * Tr(R1^T R2) = 8 - 8 * dot²
    # Simplified: 4(1 - dot²) = 4 * sin²(θ/2) where θ is rotation angle
    return (1.0 - dot * dot).clamp(min=0.0) * 4.0


def so3_geodesic_distance(q1: Tensor, q2: Tensor) -> Tensor:
    """Geodesic distance on SO(3) manifold.

    d_geo(R1, R2) = ‖log(R1^T R2)‖₂ = θ (rotation angle)

    This is the unique bi-invariant Riemannian distance on SO(3).

    Warning: Uses acos internally — gradients diverge near θ=0 and θ=π.
    For attention scores in large batches, prefer so3_chordal_distance.

    Args:
        q1, q2: Unit quaternions
            shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Returns:
        Geodesic distance (rotation angle in radians), shape: [...]
    """
    # Relative rotation: q_rel = q1⁻¹ ⊗ q2
    q_rel = quaternion_multiply(quaternion_conjugate(q1), q2)

    # ω = log(q_rel)
    omega = so3_log(q_rel)  # [..., 3]

    # θ = ‖ω‖
    return safe_sqrt((omega * omega).sum(dim=-1))


def se3_chordal_distance(T1: Tensor, T2: Tensor, w_rot: float = 1.0, w_trans: float = 1.0) -> Tensor:
    """Chordal distance between two SE(3) poses.

    Combines rotational chordal distance with translational Euclidean distance:
        d(T1, T2) = w_rot * ‖R1 - R2‖_F² + w_trans * ‖t1 - t2‖²

    The weights allow balancing rotation vs translation contributions,
    which is critical in robotics where these have different physical units.

    Args:
        T1: SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]
        T2: SE(3) element
            shape: [..., 7], representation: SE(3) compact [q|t]
        w_rot: Weight for rotational component
        w_trans: Weight for translational component

    Returns:
        Weighted chordal distance, shape: [...]
    """
    q1, t1 = T1[..., :4], T1[..., 4:7]
    q2, t2 = T2[..., :4], T2[..., 4:7]

    rot_dist = so3_chordal_distance(q1, q2)  # [...]
    trans_dist = ((t1 - t2) ** 2).sum(dim=-1)  # [...]

    return w_rot * rot_dist + w_trans * trans_dist
