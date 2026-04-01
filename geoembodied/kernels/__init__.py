# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""GPU-accelerated kernels — Triton JIT and optional CUDA.

Spherical harmonics: Triton kernel with PyTorch fallback.
"""

from geoembodied.kernels.triton_sph_harm import (
    spherical_harmonics,
    get_l_channels,
)

__all__ = [
    "spherical_harmonics",
    "get_l_channels",
]
