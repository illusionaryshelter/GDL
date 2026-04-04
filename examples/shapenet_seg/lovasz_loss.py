# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Backward compatibility — loss moved to geoembodied.nn.losses.lovasz."""

from geoembodied.nn.losses.lovasz import lovasz_softmax  # noqa: F401

__all__ = ["lovasz_softmax"]
