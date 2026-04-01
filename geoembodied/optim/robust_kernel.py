# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Robust kernel functions for outlier-resilient optimization.

In SLAM and registration, measurements can contain outliers (wrong
loop closures, mismatched features). Standard least-squares amplifies
outlier influence quadratically. Robust kernels re-weight the residuals
to reduce outlier impact.

Given squared residual s = ‖e‖², robust kernels provide:
    - ρ(s): the robust loss value
    - ρ'(s): first derivative (weight)
    - ρ''(s): second derivative (for Hessian correction)

The total cost becomes: Σ ρ(‖e_i‖²) instead of Σ ‖e_i‖²

All kernels are differentiable for use within torch.autograd.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class RobustKernel(ABC):
    """Base class for robust loss kernels.

    Subclasses implement ρ(s) where s = ‖e‖² is the squared residual.
    """

    @abstractmethod
    def evaluate(self, squared_residual: Tensor) -> Tensor:
        """Evaluate ρ(s).

        Args:
            squared_residual: s = ‖e‖², shape: [...]

        Returns:
            Robust loss value ρ(s), shape: [...]
        """
        ...

    @abstractmethod
    def weight(self, squared_residual: Tensor) -> Tensor:
        """Evaluate ρ'(s) — the per-residual weight.

        The weighted least-squares problem uses:
            w(s) = ρ'(s) / s as the IRLS weight.

        Args:
            squared_residual: s = ‖e‖², shape: [...]

        Returns:
            Weight ρ'(s), shape: [...]
        """
        ...


class TrivialKernel(RobustKernel):
    """Identity kernel ρ(s) = s — standard least-squares."""

    def evaluate(self, squared_residual: Tensor) -> Tensor:
        return squared_residual

    def weight(self, squared_residual: Tensor) -> Tensor:
        return torch.ones_like(squared_residual)


class HuberKernel(RobustKernel):
    """Huber robust kernel — linear growth beyond threshold.

    ρ(s) = s                       if √s ≤ δ
    ρ(s) = 2δ√s - δ²              if √s > δ

    Provides a smooth transition from L2 to L1-like behavior.
    Good general-purpose robust kernel for SLAM.

    Args:
        delta: Threshold parameter (in residual units, not squared)
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta
        self.delta_sq = delta * delta

    def evaluate(self, squared_residual: Tensor) -> Tensor:
        s = squared_residual
        abs_r = torch.sqrt(s.clamp(min=1e-12))
        # s when |r|<= delta,  2*delta*|r| - delta^2 when |r| > delta
        return torch.where(
            s <= self.delta_sq,
            s,
            2.0 * self.delta * abs_r - self.delta_sq,
        )

    def weight(self, squared_residual: Tensor) -> Tensor:
        s = squared_residual
        abs_r = torch.sqrt(s.clamp(min=1e-12))
        return torch.where(
            s <= self.delta_sq,
            torch.ones_like(s),
            self.delta / abs_r,
        )


class CauchyKernel(RobustKernel):
    """Cauchy (Lorentzian) robust kernel — stronger outlier suppression.

    ρ(s) = c² · log(1 + s/c²)

    Has heavier tails than Huber, more aggressively downweights outliers.
    Good for environments with many wrong loop closures.

    Args:
        c: Scale parameter
    """

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        self.c_sq = c * c

    def evaluate(self, squared_residual: Tensor) -> Tensor:
        return self.c_sq * torch.log1p(squared_residual / self.c_sq)

    def weight(self, squared_residual: Tensor) -> Tensor:
        return 1.0 / (1.0 + squared_residual / self.c_sq)
