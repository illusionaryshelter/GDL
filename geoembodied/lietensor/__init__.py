# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Layer 1: Geometry wrapper types (LieTensor subclasses).

Exports:
    LieTensor — base class for Lie group tensors
    SO3 — 3D rotation (unit quaternion wxyz)
    SE3 — 3D rigid transform (7-dim compact: [q|t])
"""

from geoembodied.lietensor.types import LieType
from geoembodied.lietensor.base import LieTensor
from geoembodied.lietensor.so3 import SO3
from geoembodied.lietensor.se3 import SE3

__all__ = ["LieType", "LieTensor", "SO3", "SE3"]
