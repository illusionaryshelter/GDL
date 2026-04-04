# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""GeoEmbodied neural network components.

Three sub-packages:

- ``nn.modules``: Learnable SE(3)-equivariant nn.Module layers
  (convolutions, normalization, attention, backbones).

- ``nn.solvers``: Parameter-free differentiable mathematical solvers
  (Sinkhorn OT, weighted SVD).

- ``nn.models``: Pre-built task-level model compositions
  (registration, part segmentation).

Re-exports from all sub-packages for convenience::

    from geoembodied.nn import SE3Net, GeoRegistrationModel, lovasz_softmax
"""

# ── Learnable modules ──
from geoembodied.nn.modules import (
    SE3Conv,
    SE3NetBlock,
    EquivariantLayerNorm,
    GatedNonlinearity,
    InvariantAttention,
    InvariantCrossAttention,
    ScalarGate,
    InvariantSelfAttention,
    SinusoidalDistanceEmbedding,
    SE3Net,
    global_mean_pool,
    SpatialGraph,
)

# ── Parameter-free solvers ──
from geoembodied.nn.solvers import (
    sinkhorn_log_domain,
    weighted_svd,
    weighted_svd_batched,
)

# ── Pre-built task models ──
from geoembodied.nn.models import (
    GeoRegistrationModel,
    SE3PartSegNet,
)

# ── Loss functions ──
from geoembodied.nn.losses import (
    lovasz_softmax,
)

__all__ = [
    # Modules
    "SE3Conv",
    "SE3NetBlock",
    "EquivariantLayerNorm",
    "GatedNonlinearity",
    "InvariantAttention",
    "InvariantCrossAttention",
    "ScalarGate",
    "InvariantSelfAttention",
    "SinusoidalDistanceEmbedding",
    "SE3Net",
    "global_mean_pool",
    "SpatialGraph",
    # Solvers
    "sinkhorn_log_domain",
    "weighted_svd",
    "weighted_svd_batched",
    # Models
    "GeoRegistrationModel",
    "SE3PartSegNet",
    # Losses
    "lovasz_softmax",
]
