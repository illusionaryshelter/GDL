# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SO3 — 3D rotation Lie group, backed by unit quaternion.

Storage: [..., 4] quaternion in wxyz convention
Lie algebra: so(3) ≅ R³ (axis-angle / angular velocity)
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import Tensor

from geoembodied.lietensor.base import LieTensor
from geoembodied.lietensor.types import LieType
from geoembodied.functional import (
    so3_exp,
    so3_log,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_apply,
    quaternion_normalize,
    quaternion_to_matrix,
    quaternion_from_matrix,
    quaternion_slerp,
)


class SO3(LieTensor):
    """SO(3) Lie group element — 3D rotation.

    Internal storage: unit quaternion [qw, qx, qy, qz]
        shape: [..., 4], representation: SO(3) unit quaternion wxyz

    Group operations:
        - exp: so(3) → SO(3) (axis-angle to quaternion)
        - log: SO(3) → so(3) (quaternion to axis-angle)
        - multiply (@ operator): rotation composition
        - inverse: rotation transpose
        - act (@ with points): rotate 3D vectors

    Example::

        >>> omega = torch.randn(10, 3)  # 10 random axis-angle vectors
        >>> R = SO3.exp(omega)           # → SO3 of shape [10, 4]
        >>> R_inv = R.inverse()          # → R^T
        >>> identity = R @ R_inv         # → I (up to float precision)
        >>> points = torch.randn(10, 100, 3)
        >>> rotated = R @ points         # → R ⊳ points
    """

    group_dim: int = 4
    tangent_dim: int = 3

    @property
    def ltype(self) -> LieType:
        return LieType.SO3

    @classmethod
    def exp(cls, omega: Tensor) -> 'SO3':
        """Exponential map: so(3) → SO(3).

        Args:
            omega: Axis-angle vector
                shape: [..., 3], representation: so(3) angular velocity

        Returns:
            SO3 rotation, shape: [..., 4]
        """
        return cls(so3_exp(omega))

    def log(self) -> Tensor:
        """Logarithm map: SO(3) → so(3).

        Returns:
            Axis-angle vector, shape: [..., 3]
            representation: so(3) Lie algebra element
        """
        return so3_log(self.as_subclass(Tensor))

    def multiply(self, other: 'SO3') -> 'SO3':
        """Group multiplication: self ∘ other (rotation composition).

        Args:
            other: Another SO3 rotation

        Returns:
            Composed rotation
        """
        result = quaternion_multiply(
            self.as_subclass(Tensor),
            other.as_subclass(Tensor),
        )
        return SO3(quaternion_normalize(result))

    def inverse(self) -> 'SO3':
        """Group inverse (quaternion conjugate = R^T).

        Returns:
            Inverse rotation
        """
        return SO3(quaternion_conjugate(self.as_subclass(Tensor)))

    @classmethod
    def identity(
        cls,
        batch_shape: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        device: Any = "cpu",
    ) -> 'SO3':
        """Identity rotation (no rotation).

        Returns:
            Identity quaternion [1, 0, 0, 0]
        """
        q = torch.zeros(*batch_shape, 4, dtype=dtype, device=device)
        q[..., 0] = 1.0
        return cls(q)

    def project_(self) -> 'SO3':
        """In-place normalize to unit quaternion.

        Per AGENTS.md Rule 5: corrects floating-point drift.
        """
        self.data = quaternion_normalize(self.data)
        return self

    def act(self, points: Tensor) -> Tensor:
        """Rotate 3D points/vectors.

        Args:
            points: 3D vectors, shape: [..., 3] or [..., N, 3]

        Returns:
            Rotated vectors, same shape as input
        """
        q = self.as_subclass(Tensor)
        if points.dim() > q.dim():
            q = q.unsqueeze(-2)
        return quaternion_apply(q, points)

    def to_matrix(self) -> Tensor:
        """Convert to 3×3 rotation matrix.

        Returns:
            Rotation matrix, shape: [..., 3, 3]
            representation: SO(3) rotation matrix (det=1, orthogonal)
        """
        return quaternion_to_matrix(self.as_subclass(Tensor))

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> 'SO3':
        """Create SO3 from 3×3 rotation matrix.

        Args:
            matrix: Rotation matrix, shape: [..., 3, 3]

        Returns:
            SO3 quaternion
        """
        return cls(quaternion_from_matrix(matrix))

    def slerp(self, other: 'SO3', t: Tensor) -> 'SO3':
        """Spherical linear interpolation.

        Args:
            other: Target rotation
            t: Interpolation parameter ∈ [0, 1]

        Returns:
            Interpolated rotation
        """
        return SO3(quaternion_slerp(
            self.as_subclass(Tensor),
            other.as_subclass(Tensor),
            t,
        ))
