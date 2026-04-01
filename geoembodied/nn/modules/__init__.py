# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Learnable SE(3)-equivariant modules (all inherit nn.Module).

Contains all components with trainable parameters:
convolutions, normalization, attention, gating, and backbones.
"""

from geoembodied.nn.modules.se3_conv import SE3Conv
from geoembodied.nn.modules.se3_block import SE3NetBlock
from geoembodied.nn.modules.equivariant_norm import EquivariantLayerNorm
from geoembodied.nn.modules.gated_nonlinearity import GatedNonlinearity
from geoembodied.nn.modules.invariant_attention import InvariantAttention
from geoembodied.nn.modules.invariant_cross_attention import (
    InvariantCrossAttention,
    ScalarGate,
)
from geoembodied.nn.modules.geometric_self_attention import (
    InvariantSelfAttention,
    SinusoidalDistanceEmbedding,
)
from geoembodied.nn.modules.se3_net import SE3Net, global_mean_pool
from geoembodied.nn.modules.spatial_graph import SpatialGraph
from geoembodied.nn.modules.equivariant_pool import EquivariantPool
from geoembodied.nn.modules.equivariant_interp import EquivariantInterpolate
from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

__all__ = [
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
    "EquivariantPool",
    "EquivariantInterpolate",
    "MultiScaleSE3Net",
]
