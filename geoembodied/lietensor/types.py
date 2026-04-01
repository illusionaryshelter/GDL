# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Lie group type enumeration."""

from enum import Enum, auto


class LieType(Enum):
    """Enumeration of supported Lie group types."""
    SO3 = auto()   # Rotation in 3D, dim=4 (quaternion wxyz)
    SE3 = auto()   # Rigid transform in 3D, dim=7 (quaternion + translation)
    # Future: SO2, SE2, Sim3
