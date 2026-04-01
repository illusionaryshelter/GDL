# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""LieTensor — torch.Tensor subclass with Lie group constraints.

Key design decisions:
    1. Subclasses torch.Tensor for ecosystem compatibility (DataLoader, DDP, etc.)
    2. Implements PyTree protocol (__tensor_flatten__ / __tensor_unflatten__) for
       torch.compile (Dynamo) compatibility — avoids graph breaks.
    3. Overrides __torch_function__ to block non-manifold operations (Rule 2, 6).
    4. Provides abstract interface for exp/log/multiply/inverse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from geoembodied.lietensor.types import LieType


# Operations that MUST NOT be applied to LieTensor
# (they break manifold constraints — AGENTS.md Rule 2)
# Include both high-level torch.* and the underlying aten ops
_BLOCKED_OP_NAMES: set[str] = {
    "add", "add_", "sub", "sub_",
    "__add__", "__iadd__", "__radd__",
    "__sub__", "__isub__", "__rsub__",
}


class LieTensor(Tensor):
    """Base class for Lie group tensors with physical constraints.

    A torch.Tensor subclass that enforces Lie group structure. Direct
    arithmetic operations (+, -) are blocked; use group operations
    (exp, log, multiply) instead.

    Subclasses must implement:
        - ltype (property): LieType enum value
        - group_dim (class attribute): dimension of group element
        - tangent_dim (class attribute): dimension of Lie algebra element
        - exp (classmethod): Lie algebra → Lie group
        - log (method): Lie group → Lie algebra
        - multiply (method): Group multiplication
        - inverse (method): Group inverse
        - identity (classmethod): Identity element
        - project_ (method): Project onto manifold (normalize)
    """

    # Subclasses must override these
    group_dim: int = NotImplemented
    tangent_dim: int = NotImplemented

    @staticmethod
    def __new__(
        cls: type,
        data: Any,
        *,
        requires_grad: bool = False,
    ) -> 'LieTensor':
        if isinstance(data, Tensor):
            instance = data.as_subclass(cls)
        else:
            instance = torch.tensor(data, dtype=torch.float32).as_subclass(cls)

        if requires_grad:
            # For optimizer-ready parameters, must create leaf tensor
            instance = instance.detach().clone().as_subclass(cls)
            instance.requires_grad_(True)

        return instance

    def parameter(self) -> 'LieTensor':
        """Create an optimizer-friendly version of this LieTensor.

        Returns a leaf tensor with requires_grad=True, suitable for
        use with ManifoldAdam/ManifoldSGD.

        Example::

            >>> T = SE3.exp(torch.randn(6))
            >>> T_param = T.parameter()  # leaf, requires_grad=True
            >>> optimizer = ManifoldAdam([T_param], lr=0.01)
        """
        return type(self)(self.data.detach().clone(), requires_grad=True)

    # ── PyTree protocol for torch.compile (Dynamo) ──────────────

    def __tensor_flatten__(self) -> Tuple[List[str], Dict[str, Any]]:
        """Flatten LieTensor for Dynamo graph tracing.

        Returns the underlying tensor data and metadata needed for
        reconstruction. This avoids graph breaks in torch.compile.
        """
        return ["_data"], {"ltype_name": type(self).__name__}

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors: Dict[str, Tensor],
        metadata: Dict[str, Any],
        outer_size: torch.Size,
        outer_stride: Tuple[int, ...],
    ) -> 'LieTensor':
        """Reconstruct LieTensor after Dynamo tracing."""
        return cls(inner_tensors["_data"])

    # ── Illegal operation interception ──────────────────────────

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Intercept torch operations to block manifold-violating ops.

        Per AGENTS.md Rule 2: T = T + delta_T is FORBIDDEN.
        Use exponential map: T_new = exp(Δξ^) ∘ T_old.
        """
        kwargs = kwargs or {}

        # Get the function name, handling both torch.add and Tensor.add_
        func_name = getattr(func, "__name__", "")

        if func_name in _BLOCKED_OP_NAMES:
            raise TypeError(
                f"Cannot apply '{func_name}' to {cls.__name__}. "
                f"Additive updates break manifold constraints (AGENTS.md Rule 2). "
                f"Use Lie algebra operations instead:\n"
                f"  T_new = T_cls.exp(delta_xi) @ T_old\n"
                f"  (where delta_xi is in the tangent space / Lie algebra)"
            )

        # For allowed ops, use default Tensor behavior
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)

        return ret

    # ── Override __add__ / __sub__ directly for Python operator dispatch ──

    def __add__(self, other: Any) -> Any:
        raise TypeError(
            f"Cannot apply '+' to {type(self).__name__}. "
            f"Additive updates break manifold constraints (AGENTS.md Rule 2). "
            f"Use: T_new = T_cls.exp(delta_xi) @ T_old"
        )

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        raise TypeError(
            f"Cannot apply '-' to {type(self).__name__}. "
            f"Additive updates break manifold constraints (AGENTS.md Rule 2). "
            f"Use: T_new = T_cls.exp(delta_xi) @ T_old"
        )

    # ── Group operation protocol (abstract) ─────────────────────

    @property
    def ltype(self) -> LieType:
        """Return the Lie group type of this tensor."""
        raise NotImplementedError

    @classmethod
    def exp(cls, tangent: Tensor) -> 'LieTensor':
        """Exponential map: Lie algebra → Lie group.

        Args:
            tangent: Tangent vector (Lie algebra element)
                shape: [..., tangent_dim]
        """
        raise NotImplementedError

    def log(self) -> Tensor:
        """Logarithm map: Lie group → Lie algebra.

        Returns:
            Tangent vector, shape: [..., tangent_dim]
        """
        raise NotImplementedError

    def multiply(self, other: 'LieTensor') -> 'LieTensor':
        """Group multiplication: self ∘ other."""
        raise NotImplementedError

    def inverse(self) -> 'LieTensor':
        """Group inverse: self⁻¹."""
        raise NotImplementedError

    @classmethod
    def identity(
        cls,
        batch_shape: Tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        device: Any = "cpu",
    ) -> 'LieTensor':
        """Create identity element(s)."""
        raise NotImplementedError

    def project_(self) -> 'LieTensor':
        """In-place projection onto manifold.

        Per AGENTS.md Rule 5: must be called periodically after
        repeated multiplications to correct floating-point drift.
        """
        raise NotImplementedError

    # ── Operator overloads (using group operations) ─────────────

    def __matmul__(self, other: Any) -> Any:
        """@ operator → group multiplication or group action.

        If other is LieTensor: group multiplication (T1 ∘ T2)
        If other is plain Tensor: group action (T ⊳ points)
        """
        if isinstance(other, LieTensor):
            return self.multiply(other)
        elif isinstance(other, Tensor):
            return self.act(other)
        return NotImplemented

    def act(self, points: Tensor) -> Tensor:
        """Apply group action to points/vectors.

        Args:
            points: Points to transform, shape: [..., 3]

        Returns:
            Transformed points, shape: [..., 3]
        """
        raise NotImplementedError

    # ── String representation ───────────────────────────────────

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        data_str = Tensor.__repr__(self)
        return f"{cls_name}({data_str})"
