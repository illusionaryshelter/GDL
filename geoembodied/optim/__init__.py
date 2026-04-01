# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Manifold-aware optimizers for Lie group parameters.

These optimizers correctly handle LieTensor parameters by computing
gradients in the tangent space (Lie algebra) and retracting back to
the manifold (Lie group) via the exponential map.

For ordinary torch.Tensor parameters, they behave identically to
their standard PyTorch counterparts.
"""

from geoembodied.optim.manifold_sgd import ManifoldSGD
from geoembodied.optim.manifold_adam import ManifoldAdam
from geoembodied.optim.gauss_newton import ManifoldGaussNewton
from geoembodied.optim.factor import (
    Factor,
    BetweenFactor,
    PriorFactor,
    LandmarkFactor,
)
from geoembodied.optim.robust_kernel import (
    RobustKernel,
    TrivialKernel,
    HuberKernel,
    CauchyKernel,
)

__all__ = [
    "ManifoldSGD",
    "ManifoldAdam",
    "ManifoldGaussNewton",
    "Factor",
    "BetweenFactor",
    "PriorFactor",
    "LandmarkFactor",
    "RobustKernel",
    "TrivialKernel",
    "HuberKernel",
    "CauchyKernel",
]
