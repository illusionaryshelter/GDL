# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""ShapeNet Part Segmentation model — thin wrapper.

This module re-exports the generic SE3PartSegNet from
``geoembodied.nn.models`` with ShapeNet-specific constants
pre-configured. The training script imports from here.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from geoembodied.nn.models.part_segmentation import (
    SE3PartSegNet as _SE3PartSegNet,
)
from examples.shapenet_seg.dataset import (
    CATEGORY_PART_MASK,
    NUM_PARTS,
    NUM_CATEGORIES,
)


class SE3PartSegNet(_SE3PartSegNet):
    """SE3PartSegNet pre-configured for ShapeNet Part Segmentation.

    Passes ShapeNet constants (16 categories, 50 parts, category mask)
    to the generic base model automatically.

    Args:
        in_channels: Input scalar feature dimension (default: 1)
        hidden_scalar: Hidden scalar channels in U-Net
        hidden_vector: Hidden vector channels in U-Net
        num_stages: Number of encoder/decoder stages
        layers_per_stage: SE3Conv layers per stage
        pool_ratio: Downsampling ratio per stage
        use_normals: If True, inject normals as initial l=1 vector features
        head_hidden: Hidden dim in classification head
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_scalar: int = 64,
        hidden_vector: int = 16,
        num_stages: int = 3,
        layers_per_stage: int = 2,
        pool_ratio: float = 0.25,
        use_normals: bool = True,
        head_hidden: int = 128,
    ) -> None:
        super().__init__(
            num_categories=NUM_CATEGORIES,
            num_parts=NUM_PARTS,
            category_part_mask=CATEGORY_PART_MASK,
            in_channels=in_channels,
            hidden_scalar=hidden_scalar,
            hidden_vector=hidden_vector,
            num_stages=num_stages,
            layers_per_stage=layers_per_stage,
            pool_ratio=pool_ratio,
            use_normals=use_normals,
            head_hidden=head_hidden,
        )
