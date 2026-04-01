# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Differentiable mathematical solver layers (no learnable parameters).

These are pure optimization/matching algorithms that can be composed
into larger architectures. They do NOT contain nn.Parameter.

- ``sinkhorn_log_domain``: Optimal transport matching
- ``weighted_svd`` / ``weighted_svd_batched``: Rigid alignment via SVD
"""

from geoembodied.nn.solvers.sinkhorn import sinkhorn_log_domain
from geoembodied.nn.solvers.weighted_svd import (
    weighted_svd,
    weighted_svd_batched,
)

__all__ = [
    "sinkhorn_log_domain",
    "weighted_svd",
    "weighted_svd_batched",
]
