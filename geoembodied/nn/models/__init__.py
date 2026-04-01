# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Pre-built SE(3)-equivariant task models.

This sub-package provides ready-to-use model architectures composed
from the building blocks in ``nn.modules`` and ``nn.solvers``.

Architecture hierarchy::

    geoembodied/nn/
    ├── modules/    ← Learnable building blocks (SE3Conv, Pool, ...)
    ├── solvers/    ← Parameter-free math (SVD, Sinkhorn, ...)
    └── models/     ← 🆕 Task-level model compositions

Available models:
    - ``GeoRegistrationModel``: Point cloud registration (SE3Net +
      Cross-Attention + Sinkhorn + Weighted SVD)
    - ``SE3PartSegNet``: Part segmentation (MultiScaleSE3Net U-Net +
      category-conditioned head)

Usage::

    from geoembodied.nn.models import GeoRegistrationModel
    from geoembodied.nn.models import SE3PartSegNet
"""

from geoembodied.nn.models.registration import GeoRegistrationModel
from geoembodied.nn.models.part_segmentation import SE3PartSegNet

__all__ = [
    "GeoRegistrationModel",
    "SE3PartSegNet",
]
