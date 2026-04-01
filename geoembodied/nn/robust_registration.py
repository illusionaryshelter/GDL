# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Backward compatibility — moved to geoembodied.nn.models.robust_registration."""

from geoembodied.nn.models.robust_registration import (  # noqa: F401
    InlierPredictor,
    RobustRegistrationHead,
)

__all__ = ["InlierPredictor", "RobustRegistrationHead"]
