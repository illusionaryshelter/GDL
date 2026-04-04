# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""SE(3)-equivariant interaction block.

Single interaction layer:

    Conv → InnerNorm → Gate → [SelfTP] → Skip Residual → **OutputNorm**

Supports scalar (l=0), vector (l=1), and type-2 (l=2) features.

**Dual normalization (hard constraint against catastrophic blowup):**

    The **inner norm** (after conv, before gate) controls the conv
    output scale — same as NequIP/MACE.

    The **output norm** (after residual) is the critical addition.
    Without it, the skip path can inject unbounded feature norms
    into downstream blocks, causing exponential blowup in deep stages
    (observed: s2 → 3.6×10¹¹ in a single epoch at stage 2).

    This is NOT Pre-Norm (which normalises the input TO conv → starves
    conv contribution). The conv sees its RAW input, gets normalised
    by inner norm + gate, then the TOTAL output (conv + skip) is
    bounded by output norm.  This guarantees:
        - Conv always operates on its natural scale → no contribution collapse
        - Output is always bounded → no catastrophic amplification
        - Each block feeds ~O(1) features to the next block

    Mathematically: Pre-Norm = x + f(norm(x)), output never bounded.
    This design  = norm(f(x) + skip·x), output ALWAYS bounded.

**Learnable skip (MACE/NequIP pattern):**

    Per-channel skip_scale initialised to 1/√2, giving the optimiser
    fine-grained control over the residual contribution.

Self-Interaction TP (optional, use_self_tp=True):
    - v·v → scalar (ν=2 body order)
    - t2·t2 → scalar (||t2_c||² invariant contraction)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from geoembodied.nn.modules.se3_conv import SE3Conv
from geoembodied.nn.modules.equivariant_norm import EquivariantLayerNorm
from geoembodied.nn.modules.gated_nonlinearity import GatedNonlinearity

if TYPE_CHECKING:
    from geoembodied.nn.modules.spatial_graph import SpatialGraph

# Default initial value for learnable skip scales.
# 1/√2 ≈ 0.7071 — matches the old fixed residual scaling so that
# behaviour at initialisation is unchanged.
_SKIP_INIT = 1.0 / math.sqrt(2.0)


class SE3NetBlock(nn.Module):
    """Single SE(3)-equivariant interaction block.

    Architecture::

        (s, v, [t2]) → SE3Conv → InnerNorm → Gate → [SelfTP]
                                                        │
                                              [+ skip_scale · input]
                                                        │
                                                  OutputNorm
                                                        │
                                                   (s_out, v_out, [t2_out])

    The inner norm controls conv output; the output norm prevents
    catastrophic residual amplification across stacked blocks.

    The ``skip_*_scale`` parameters are **per-channel, per-block**
    (NOT shared across modules).  For l≥1 features the scale is a
    simple channel-wise scalar multiplication, which is SO(3)-equivariant:
    ``γ_c · D^l(R) f_{c} = D^l(R) (γ_c · f_{c})``.

    Args:
        channels_scalar: Scalar feature channels (in = out)
        channels_vector: Vector feature channels (in = out)
        channels_type2: Type-2 feature channels (in = out), 0 to disable
        radius: Spatial graph radius for SE3Conv
        max_num_neighbors: Max edges per node
        use_residual: Enable skip connection
        gate_mode: Gate mode ('scalar' or 'norm')
        use_self_tp: Enable self-interaction tensor product
    """

    def __init__(
        self,
        channels_scalar: int,
        channels_vector: int,
        channels_type2: int = 0,
        radius: float = 1.0,
        max_num_neighbors: int = 32,
        use_residual: bool = True,
        gate_mode: str = 'scalar',
        use_self_tp: bool = False,
    ) -> None:
        super().__init__()
        self.channels_scalar = channels_scalar
        self.channels_vector = channels_vector
        self.channels_type2 = channels_type2
        self.use_residual = use_residual
        self.use_self_tp = use_self_tp

        self.conv = SE3Conv(
            in_scalar_channels=channels_scalar,
            in_vector_channels=channels_vector,
            out_scalar_channels=channels_scalar,
            out_vector_channels=channels_vector,
            in_type2_channels=channels_type2,
            out_type2_channels=channels_type2,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
        )

        # Inner norm: controls conv output before gate
        self.norm = EquivariantLayerNorm(
            num_scalars=channels_scalar,
            num_vectors=channels_vector,
            num_type2=channels_type2,
        )

        self.gate = GatedNonlinearity(
            num_scalars=channels_scalar,
            num_vectors=channels_vector,
            num_type2=channels_type2,
            gate_mode=gate_mode,
        )

        # Output norm: hard constraint on post-residual features.
        # Prevents catastrophic norm amplification across stacked blocks.
        # Uses separate running stats from inner norm.
        self.output_norm = EquivariantLayerNorm(
            num_scalars=channels_scalar,
            num_vectors=channels_vector,
            num_type2=channels_type2,
        )

        # ── Learnable skip projections (MACE/NequIP pattern) ──
        # Per-channel scale, initialised to 1/√2.
        # Each block owns its own parameters — no cross-module sharing.
        if use_residual:
            self.skip_s_scale = nn.Parameter(
                torch.full((channels_scalar,), _SKIP_INIT)
            )
            if channels_vector > 0:
                self.skip_v_scale = nn.Parameter(
                    torch.full((channels_vector,), _SKIP_INIT)
                )
            else:
                self.register_parameter('skip_v_scale', None)

            if channels_type2 > 0:
                self.skip_t2_scale = nn.Parameter(
                    torch.full((channels_type2,), _SKIP_INIT)
                )
            else:
                self.register_parameter('skip_t2_scale', None)
        else:
            self.register_parameter('skip_s_scale', None)
            self.register_parameter('skip_v_scale', None)
            self.register_parameter('skip_t2_scale', None)

        # Self-Interaction TP: v·v + t2·t2 → scalar
        self_tp_input_dim = 0
        if use_self_tp and channels_vector > 0:
            self_tp_input_dim += channels_vector
        if use_self_tp and channels_type2 > 0:
            self_tp_input_dim += channels_type2

        if self_tp_input_dim > 0:
            self.self_tp_proj = nn.Linear(self_tp_input_dim, channels_scalar, bias=False)
            self.self_tp_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.self_tp_proj = None

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        graph: 'SpatialGraph',
        type2: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Forward pass: Conv → InnerNorm → Gate → [SelfTP] → Skip → OutputNorm.

        Args:
            scalars: [N, channels_scalar], representation: SO(3) type-0
            vectors: [N, channels_vector, 3], representation: SO(3) type-1
            graph: Pre-built spatial graph
            type2: [N, channels_type2, 5], representation: SO(3) type-2, optional

        Returns:
            If type2 is None: (scalars_out, vectors_out)
            If type2 is given: (scalars_out, vectors_out, type2_out)
        """
        has_type2 = type2 is not None and self.channels_type2 > 0

        # ── Conv ──
        if has_type2:
            s_new, v_new, t2_new = self.conv(scalars, vectors, graph, type2=type2)
        else:
            s_new, v_new = self.conv(scalars, vectors, graph)
            t2_new = None

        # ── Inner Norm (controls conv output scale) ──
        if t2_new is not None:
            s_new, v_new, t2_new = self.norm(s_new, v_new, t2_new)
        else:
            s_new, v_new = self.norm(s_new, v_new)

        # ── Gate ──
        if t2_new is not None:
            s_new, v_new, t2_new = self.gate(s_new, v_new, t2_new)
        else:
            s_new, v_new = self.gate(s_new, v_new)

        # ── Self-Interaction TP ──
        if self.self_tp_proj is not None:
            tp_inputs = []
            if self.channels_vector > 0:
                v_dot_v = (v_new * v_new).sum(dim=-1)  # [N, C_v]
                tp_inputs.append(v_dot_v)
            if t2_new is not None and self.channels_type2 > 0:
                t2_dot_t2 = (t2_new * t2_new).sum(dim=-1)  # [N, C_t2]
                tp_inputs.append(t2_dot_t2)
            if tp_inputs:
                tp_cat = torch.cat(tp_inputs, dim=-1)  # [N, C_v + C_t2]
                s_tp = self.self_tp_proj(tp_cat) * self.self_tp_scale
                s_new = s_new + s_tp

        # ── Learnable skip residual ──
        if self.use_residual:
            s_new = s_new + scalars * self.skip_s_scale
            if self.skip_v_scale is not None:
                v_new = v_new + vectors * self.skip_v_scale.unsqueeze(-1)
            if t2_new is not None and type2 is not None and self.skip_t2_scale is not None:
                t2_new = t2_new + type2 * self.skip_t2_scale.unsqueeze(-1)

        # ── Output norm (hard constraint: bounds post-residual features) ──
        # This is the critical difference from Pre-Norm:
        #   Pre-Norm:  x + f(norm(x))  → output unbounded (skip bypasses norm)
        #   This:      norm(f(x) + skip·x) → output ALWAYS bounded
        if t2_new is not None:
            s_new, v_new, t2_new = self.output_norm(s_new, v_new, t2_new)
        else:
            s_new, v_new = self.output_norm(s_new, v_new)

        if has_type2 and t2_new is not None:
            return s_new, v_new, t2_new
        return s_new, v_new

    def get_skip_diagnostics(self) -> Dict[str, float]:
        """Return skip scale statistics for training diagnostics.

        Returns:
            Dictionary with per-type skip scale mean/std/min/max.
        """
        diag: Dict[str, float] = {}
        if self.skip_s_scale is not None:
            s = self.skip_s_scale.detach()
            diag['skip_s_mean'] = s.mean().item()
            diag['skip_s_std'] = s.std().item()
            diag['skip_s_min'] = s.min().item()
            diag['skip_s_max'] = s.max().item()
        if self.skip_v_scale is not None:
            v = self.skip_v_scale.detach()
            diag['skip_v_mean'] = v.mean().item()
            diag['skip_v_std'] = v.std().item()
        if self.skip_t2_scale is not None:
            t = self.skip_t2_scale.detach()
            diag['skip_t2_mean'] = t.mean().item()
            diag['skip_t2_std'] = t.std().item()
        return diag

    def extra_repr(self) -> str:
        parts = [
            f"scalar={self.channels_scalar}",
            f"vector={self.channels_vector}",
            f"type2={self.channels_type2}",
            f"residual={self.use_residual}",
            f"gate_mode={self.gate.gate_mode}",
            f"self_tp={self.use_self_tp}",
        ]
        if self.skip_s_scale is not None:
            parts.append(f"skip_init={_SKIP_INIT:.4f}")
        return ", ".join(parts)
