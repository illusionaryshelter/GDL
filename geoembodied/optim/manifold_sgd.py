# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""ManifoldSGD — SGD with exponential map retraction for LieTensor.

The key insight for manifold optimization of quaternion/SE(3) parameters:

    1. PyTorch autograd computes the Euclidean gradient ∂L/∂q in the
       ambient R⁴ (SO3) or R⁷ (SE3) space.

    2. We project this to the Riemannian gradient on the manifold.
       For SO(3)/SE(3), the Riemannian gradient at point p is:

           rgrad = egrad - <egrad, p> * p    (project to tangent hyperplane)

       This removes the component along the constraint normal.

    3. We then retract using the exponential map:
           T_new = exp(-lr * rgrad_tangent) ∘ T_old

       Or equivalently for quaternions, use the simpler retraction:
           q_new = normalize(q - lr * rgrad)

Reference:
    - Geoopt (Kochurov et al.): manifold.egrad2rgrad + manifold.retr
    - Absil et al.: "Optimization Algorithms on Matrix Manifolds"
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from geoembodied.lietensor.base import LieTensor


def _egrad_to_rgrad_so3(point: Tensor, egrad: Tensor) -> Tensor:
    """Project Euclidean gradient to Riemannian gradient on S³.

    The unit quaternion manifold is S³ (3-sphere in R⁴).
    Tangent space at q is the hyperplane orthogonal to q:
        T_q S³ = {v ∈ R⁴ : <v, q> = 0}

    Projection: rgrad = egrad - <egrad, q> * q

    Args:
        point: Current quaternion, shape: [..., 4]
        egrad: Euclidean gradient, shape: [..., 4]

    Returns:
        Riemannian gradient on S³, shape: [..., 4]
    """
    # Project out the normal component
    inner = (egrad * point).sum(dim=-1, keepdim=True)
    return egrad - inner * point


def _egrad_to_rgrad_se3(point: Tensor, egrad: Tensor) -> Tensor:
    """Project Euclidean gradient to Riemannian gradient on SE(3).

    SE(3) = S³ × R³ (product manifold: quaternion sphere × Euclidean)
    - Quaternion part [..., :4]: project to tangent of S³
    - Translation part [..., 4:]: already in Euclidean space, no projection

    Args:
        point: SE(3) element, shape: [..., 7]
        egrad: Euclidean gradient, shape: [..., 7]

    Returns:
        Riemannian gradient, shape: [..., 7]
    """
    rgrad = egrad.clone()
    # Project quaternion part onto tangent space of S³
    q = point[..., :4]
    eq = egrad[..., :4]
    inner = (eq * q).sum(dim=-1, keepdim=True)
    rgrad[..., :4] = eq - inner * q
    # Translation part stays as-is (Euclidean)
    return rgrad


def _retract_so3(point: Tensor, direction: Tensor, lr: float) -> Tensor:
    """Retract on S³ by normalizing after step.

    q_new = normalize(q - lr * direction)

    This is the "projection retraction" which is equivalent to
    the exponential map to first order.

    Args:
        point: Current quaternion, shape: [..., 4]
        direction: Riemannian gradient, shape: [..., 4]
        lr: Learning rate

    Returns:
        New quaternion on S³, shape: [..., 4]
    """
    from geoembodied.functional.quaternion_ops import quaternion_normalize
    new_q = point - lr * direction
    return quaternion_normalize(new_q)


def _retract_se3(point: Tensor, direction: Tensor, lr: float) -> Tensor:
    """Retract on SE(3) = S³ × R³.

    - Quaternion: normalize(q - lr * dir_q)
    - Translation: t - lr * dir_t

    Args:
        point: SE(3) element, shape: [..., 7]
        direction: Riemannian gradient, shape: [..., 7]
        lr: Learning rate

    Returns:
        New SE(3) element, shape: [..., 7]
    """
    from geoembodied.functional.quaternion_ops import quaternion_normalize
    new_p = point - lr * direction
    # Re-normalize quaternion part
    new_p[..., :4] = quaternion_normalize(new_p[..., :4])
    return new_p


def _egrad2rgrad(point: Tensor, egrad: Tensor) -> Tensor:
    """Dispatch egrad → rgrad based on LieTensor type."""
    from geoembodied.lietensor.so3 import SO3
    from geoembodied.lietensor.se3 import SE3
    if isinstance(point, SO3):
        return _egrad_to_rgrad_so3(point.as_subclass(Tensor), egrad)
    elif isinstance(point, SE3):
        return _egrad_to_rgrad_se3(point.as_subclass(Tensor), egrad)
    else:
        return egrad


def _retract(point: Tensor, direction: Tensor, lr: float) -> Tensor:
    """Dispatch retraction based on LieTensor type."""
    from geoembodied.lietensor.so3 import SO3
    from geoembodied.lietensor.se3 import SE3
    if isinstance(point, SO3):
        return _retract_so3(point.as_subclass(Tensor), direction, lr)
    elif isinstance(point, SE3):
        return _retract_se3(point.as_subclass(Tensor), direction, lr)
    else:
        return point - lr * direction


class ManifoldSGD(Optimizer):
    """Stochastic Gradient Descent on Lie group manifolds.

    For LieTensor parameters, the update rule is:

        rgrad = project(egrad) onto tangent space
        v_t = momentum * v_{t-1} + rgrad
        T_{t+1} = retract(T_t, v_t, lr)                 (with re-normalization)
        T_{t+1}.project_() every N steps                 (stabilize)

    For plain Tensor parameters, standard SGD applies.

    Args:
        params: Iterable of parameters (LieTensor and/or Tensor)
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        dampening: Dampening for momentum (default: 0)
        weight_decay: Weight decay / L2 penalty (default: 0).
            Only applied to plain Tensor params, NOT LieTensor.
        stabilize_every: Re-project LieTensor params every N steps (default: 10).

    Example::

        >>> T = SE3.exp(torch.randn(6)).parameter()
        >>> optimizer = ManifoldSGD([T], lr=0.01, momentum=0.9)
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-2,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        stabilize_every: int = 10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            stabilize_every=stabilize_every,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            stabilize_every = group["stabilize_every"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1

                if isinstance(p, LieTensor):
                    # ── Manifold update ──────────────────────
                    # Convert Euclidean gradient to Riemannian
                    rgrad = _egrad2rgrad(p, grad)

                    if momentum > 0:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = rgrad.clone()
                        else:
                            buf = state["momentum_buffer"]
                            buf.mul_(momentum).add_(rgrad, alpha=1 - dampening)
                        direction = state["momentum_buffer"]
                    else:
                        direction = rgrad

                    # Retraction (normalize after step)
                    new_data = _retract(p, direction, lr)
                    p.data.copy_(new_data)

                    # Periodic stabilization (AGENTS.md Rule 5)
                    if state["step"] % stabilize_every == 0:
                        p.project_()

                else:
                    # ── Euclidean update ─────────────────────
                    if weight_decay != 0:
                        grad = grad.add(p.data, alpha=weight_decay)

                    if momentum > 0:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = grad.clone()
                        else:
                            buf = state["momentum_buffer"]
                            buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                        grad = state["momentum_buffer"]

                    p.data.add_(grad, alpha=-lr)

        return loss
