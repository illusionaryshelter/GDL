# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""GeoEmbodied: Geometric Deep Learning for Robotics & Autonomous Driving.

A core library built on strict Lie group theory and differential geometry,
providing physically-constrained tensor types, equivariant neural operators,
and differentiable SLAM/planning primitives.

Architecture:
    Layer 0 — ``geoembodied.functional``: Pure math functions (exp, log, adj, ...)
    Layer 1 — ``geoembodied.lietensor``: LieTensor geometry wrappers (SO3, SE3, ...)
    Layer 2 — ``geoembodied.nn``: Equivariant neural network layers
"""

from geoembodied._version import __version__
from geoembodied.lietensor import LieTensor, SO3, SE3

__all__ = [
    "__version__",
    "LieTensor",
    "SO3",
    "SE3",
]
