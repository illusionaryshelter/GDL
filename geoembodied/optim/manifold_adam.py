# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""ManifoldAdam — Adam with Riemannian gradient and projection retraction.

Uses the same egrad → rgrad → retraction pattern as ManifoldSGD,
with Adam-style first/second moment estimation in the ambient space.

For SO(3): gradient projected to tangent of S³, retract by normalize
For SE(3): quaternion part projected to tangent of S³, translation Euclidean

Reference:
    - Bécigneul & Ganea (2019): "Riemannian Adaptive Optimization Methods"
    - Geoopt: RiemannianAdam (simplified — no parallel transport)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from geoembodied.lietensor.base import LieTensor
from geoembodied.optim.manifold_sgd import _egrad2rgrad, _retract


class ManifoldAdam(Optimizer):
    """Adam optimizer on Lie group manifolds.

    For LieTensor params:
        rgrad = project(egrad)              (Riemannian gradient)
        m_t = β₁ m_{t-1} + (1-β₁) rgrad    (first moment in ambient)
        v_t = β₂ v_{t-1} + (1-β₂) rgrad²   (second moment, component-wise)
        direction = m̂_t / (√v̂_t + ε)        (bias-corrected)
        retract(T_t, direction, lr)          (projection retraction)

    For plain Tensor params: standard Adam.

    Args:
        params: Iterable of parameters
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Numerical stability term (default: 1e-8)
        weight_decay: L2 penalty for Euclidean params only (default: 0)
        amsgrad: Use AMSGrad variant (default: False)
        stabilize_every: Manifold projection interval (default: 10)

    Example::

        >>> T = SE3.exp(torch.randn(6)).parameter()
        >>> optimizer = ManifoldAdam([T], lr=0.01)
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        stabilize_every: int = 10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            stabilize_every = group["stabilize_every"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if isinstance(p, LieTensor):
                    # ── Manifold Adam ────────────────────────
                    # Project to Riemannian gradient
                    rgrad = _egrad2rgrad(p, grad)

                    # Update moments
                    exp_avg.mul_(beta1).add_(rgrad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(rgrad, rgrad, value=1 - beta2)

                    # Bias correction
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]

                    if amsgrad:
                        max_sq = state["max_exp_avg_sq"]
                        torch.max(max_sq, exp_avg_sq, out=max_sq)
                        denom = (max_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

                    step_size = lr / bc1
                    direction = exp_avg / denom

                    # Retraction
                    new_data = _retract(p, direction, step_size)
                    p.data.copy_(new_data)

                    # Periodic stabilization
                    if state["step"] % stabilize_every == 0:
                        p.project_()

                else:
                    # ── Standard Adam ────────────────────────
                    if weight_decay != 0:
                        grad = grad.add(p.data, alpha=weight_decay)

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]

                    if amsgrad:
                        max_sq = state["max_exp_avg_sq"]
                        torch.max(max_sq, exp_avg_sq, out=max_sq)
                        denom = (max_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

                    step_size = lr / bc1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
