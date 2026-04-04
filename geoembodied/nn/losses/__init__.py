# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Loss functions for geometric deep learning.

Provides differentiable surrogates that align training objectives with
evaluation metrics commonly used in segmentation and registration tasks.

Available losses:

- :func:`lovasz_softmax` — Lovász-Softmax: directly optimizes mean IoU.
"""

from geoembodied.nn.losses.lovasz import lovasz_softmax

__all__ = ["lovasz_softmax"]
