# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE3 — 3D rigid body transformation Lie group.

Storage: 7-dim compact vector [qw, qx, qy, qz, tx, ty, tz]
    - Quaternion [..., :4]: rotation (SO(3))
    - Translation [..., 4:]: position (R³)
    - 56% memory saving vs 4×4 matrix (7 vs 16 floats)

Lie algebra: se(3) ≅ R⁶ = [ω, v]
    - ω ∈ R³: angular velocity (so(3) part)
    - v ∈ R³: linear velocity
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import Tensor

from geoembodied.lietensor.base import LieTensor
from geoembodied.lietensor.types import LieType
from geoembodied.lietensor.so3 import SO3
from geoembodied.functional import (
    se3_exp,
    se3_log,
    se3_multiply,
    se3_inverse,
    se3_act,
    se3_adjoint,
)
from geoembodied.functional.quaternion_ops import quaternion_normalize


class SE3(LieTensor):
    """SE(3) Lie group element — 3D rigid body transformation.

    Internal storage: [qw, qx, qy, qz, tx, ty, tz]
        shape: [..., 7], representation: SE(3) compact [q|t]

    Group operations:
        - exp: se(3) → SE(3)  (twist to rigid transform)
        - log: SE(3) → se(3)  (rigid transform to twist)
        - multiply (@ operator): composition of transforms
        - inverse: (R, t)⁻¹ = (R^T, -R^T t)
        - act (@ with points): rigid body transformation of 3D points

    Example::

        >>> xi = torch.randn(10, 6)       # 10 random twists
        >>> T = SE3.exp(xi)                # → SE3 of shape [10, 7]
        >>> T_inv = T.inverse()
        >>> Identity = T @ T_inv           # → I (up to float precision)
        >>> pts = torch.randn(10, 1000, 3) # 10 batches × 1000 points
        >>> transformed = T @ pts          # → T ⊳ pts
    """

    group_dim: int = 7
    tangent_dim: int = 6

    @property
    def ltype(self) -> LieType:
        return LieType.SE3

    @property
    def rotation(self) -> SO3:
        """Extract SO(3) rotation component.

        Returns:
            SO3 quaternion, shape: [..., 4]
        """
        return SO3(self.as_subclass(Tensor)[..., :4])

    @property
    def translation(self) -> Tensor:
        """Extract translation component.

        Returns:
            Translation vector, shape: [..., 3]
        """
        return self.as_subclass(Tensor)[..., 4:7]

    @classmethod
    def exp(cls, xi: Tensor) -> 'SE3':
        """Exponential map: se(3) → SE(3).

        T = exp(ξ^) where ξ = [ω, v] ∈ R⁶

        Uses closed-form with SO(3) left Jacobian V(ω).

        Args:
            xi: Twist vector [angular, linear]
                shape: [..., 6], representation: se(3) Lie algebra element

        Returns:
            SE3 transform, shape: [..., 7]
        """
        return cls(se3_exp(xi))

    def log(self) -> Tensor:
        """Logarithm map: SE(3) → se(3).

        Returns:
            Twist vector [angular, linear]
                shape: [..., 6], representation: se(3) Lie algebra element
        """
        return se3_log(self.as_subclass(Tensor))

    def multiply(self, other: 'SE3') -> 'SE3':
        """Group multiplication: self ∘ other.

        (R1, t1) ∘ (R2, t2) = (R1 R2, R1 t2 + t1)

        Args:
            other: Another SE3 transform

        Returns:
            Composed transform
        """
        result = se3_multiply(
            self.as_subclass(Tensor),
            other.as_subclass(Tensor),
        )
        return SE3(result)

    def inverse(self) -> 'SE3':
        """Group inverse: (R, t)⁻¹ = (R^T, -R^T t).

        Returns:
            Inverse transform
        """
        return SE3(se3_inverse(self.as_subclass(Tensor)))

    @classmethod
    def identity(
        cls,
        batch_shape: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        device: Any = "cpu",
    ) -> 'SE3':
        """Identity transform (no rotation, no translation).

        Returns:
            Identity [1,0,0,0, 0,0,0], shape: [*batch_shape, 7]
        """
        T = torch.zeros(*batch_shape, 7, dtype=dtype, device=device)
        T[..., 0] = 1.0  # qw = 1
        return cls(T)

    @classmethod
    def from_rotation_translation(cls, R: SO3, t: Tensor) -> 'SE3':
        """Create SE3 from rotation and translation.

        Args:
            R: SO3 rotation, shape: [..., 4]
            t: Translation, shape: [..., 3]

        Returns:
            SE3 transform, shape: [..., 7]
        """
        q = R.as_subclass(Tensor)
        return cls(torch.cat([quaternion_normalize(q), t], dim=-1))

    def project_(self) -> 'SE3':
        """In-place projection: normalize quaternion part.

        Per AGENTS.md Rule 5: corrects floating-point drift
        in quaternion norm after repeated multiplications.
        """
        q = quaternion_normalize(self.data[..., :4])
        self.data[..., :4] = q
        return self

    def act(self, points: Tensor) -> Tensor:
        """Apply rigid body transform to 3D points.

        p' = R @ p + t

        Args:
            points: 3D points, shape: [..., 3] or [..., N, 3]

        Returns:
            Transformed points, same shape as input
        """
        return se3_act(self.as_subclass(Tensor), points)

    def adjoint(self) -> Tensor:
        """Adjoint representation (6×6 matrix).

        Maps twists between frames:
            ξ_B = Ad(T_AB) @ ξ_A

        Returns:
            Adjoint matrix, shape: [..., 6, 6]
        """
        return se3_adjoint(self.as_subclass(Tensor))

    def to_matrix(self) -> Tensor:
        """Convert to 4×4 homogeneous transformation matrix.

        Returns:
            Transformation matrix, shape: [..., 4, 4]

        Note: This allocates a 4×4 matrix. For memory-critical paths,
        use .rotation and .translation separately.
        """
        R = self.rotation.to_matrix()  # [..., 3, 3]
        t = self.translation           # [..., 3]

        batch_shape = R.shape[:-2]
        T = torch.zeros(*batch_shape, 4, 4, dtype=R.dtype, device=R.device)
        T[..., :3, :3] = R
        T[..., :3, 3] = t
        T[..., 3, 3] = 1.0

        return T
