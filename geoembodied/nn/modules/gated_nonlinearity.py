# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Gated nonlinearity for equivariant neural networks.

Standard nonlinearities (ReLU, GELU) cannot be applied to l≥1 features
because they break equivariance. The solution is gate higher-order features
using scalar-derived gates:

    scalar_out = σ(scalar_in)
    vector_out = gate(scalar_in) * vector_in
    type2_out  = gate(scalar_in) * type2_in

Since the gate is an SO(3)-invariant scalar, multiplying it with any
equivariant feature preserves equivariance: D^l(R)(gate * f) = gate * D^l(R)f.

**Norm mode** (gate_mode='norm'):
    Combines scalar-derived gate with norm-based activation for l≥1 features.
    f_out = activation(W_gate @ s + W_norm @ ||f||) * f / ||f||
    This allows the nonlinearity to *change* feature amplitude based on
    both scalar context AND feature magnitude, while strictly preserving
    the transformation rule (and therefore equivariance).

    Equivariance proof for any l:
        ||D^l(R)f|| = ||f|| (Wigner-D is unitary), so the gate is invariant.
        f/||f|| transforms as D^l(R)f/||D^l(R)f|| = D^l(R)(f/||f||).
        Therefore: gate(s, ||f||) * D^l(R)f/||f|| = D^l(R) * [gate(s, ||f||) * f/||f||]
"""

import torch
import torch.nn as nn
from torch import Tensor


class GatedNonlinearity(nn.Module):
    """Gated nonlinearity that preserves SO(3) equivariance.

    Applies standard nonlinearity to scalar features. For l≥1 features
    (vectors, type-2 tensors), generates scalar gates from a linear
    projection and multiplies.

    Args:
        num_scalars: Number of scalar input/output channels
        num_vectors: Number of vector input/output channels
        num_type2: Number of type-2 (l=2) input/output channels
        scalar_activation: Activation for scalar features (default: SiLU)
        gate_activation: Activation for gates (default: sigmoid)
        gate_mode: 'scalar' (default) or 'norm'
            'scalar': gate = σ(W @ s_in)
            'norm': gate = σ(W_s @ s + W_n @ ||f|| + b)
    """

    def __init__(
        self,
        num_scalars: int,
        num_vectors: int = 0,
        num_type2: int = 0,
        scalar_activation: str = "silu",
        gate_activation: str = "sigmoid",
        gate_mode: str = "scalar",
    ) -> None:
        super().__init__()
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors
        self.num_type2 = num_type2
        self.gate_mode = gate_mode

        # Scalar nonlinearity
        activations = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        self.scalar_act = activations.get(scalar_activation, nn.SiLU())

        # Gate nonlinearity
        gate_activations = {
            "sigmoid": nn.Sigmoid(),
            "tanh_abs": nn.Tanh(),
        }
        self.gate_act = gate_activations.get(gate_activation, nn.Sigmoid())

        # ── Vector gate ──
        if num_vectors > 0:
            self.gate_proj_v = nn.Linear(num_scalars, num_vectors, bias=True)
            if gate_mode == 'norm':
                self.norm_proj_v = nn.Linear(num_vectors, num_vectors, bias=False)
        else:
            self.gate_proj_v = None

        # ── Type-2 gate ──
        if num_type2 > 0:
            self.gate_proj_t2 = nn.Linear(num_scalars, num_type2, bias=True)
            if gate_mode == 'norm':
                self.norm_proj_t2 = nn.Linear(num_type2, num_type2, bias=False)
        else:
            self.gate_proj_t2 = None

    def _gate_features(
        self,
        scalars: Tensor,
        features: Tensor,
        gate_proj: nn.Linear,
        norm_proj: nn.Linear | None,
    ) -> Tensor:
        """Apply gating to l≥1 features.

        Args:
            scalars: [..., C_s]
            features: [..., C_f, D] where D=3 (l=1) or D=5 (l=2)
            gate_proj: Linear(C_s → C_f)
            norm_proj: Linear(C_f → C_f) or None (only for norm mode)

        Returns:
            Gated features, same shape as input
        """
        if self.gate_mode == 'norm' and norm_proj is not None:
            # ||f||: [..., C_f] — SO(3) invariant (Rule 4: clamp for safety)
            f_norm = features.norm(dim=-1).clamp(min=1e-8)  # [..., C_f]
            f_hat = features / f_norm.unsqueeze(-1)  # unit direction

            # Gate input: W_s @ s + W_n @ ||f||
            gate_input = gate_proj(scalars) + norm_proj(f_norm)
            gates = self.gate_act(gate_input)  # [..., C_f], in [0,1]

            # Scale = gate * original_norm
            return (gates * f_norm).unsqueeze(-1) * f_hat  # [..., C_f, D]
        else:
            # Standard scalar gate
            gates = self.gate_act(gate_proj(scalars))  # [..., C_f]
            return features * gates.unsqueeze(-1)  # [..., C_f, D]

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
        type2: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Apply gated nonlinearity.

        Args:
            scalars: [..., num_scalars]
            vectors: [..., num_vectors, 3]
            type2: [..., num_type2, 5] or None

        Returns:
            If type2 is None: (activated_scalars, gated_vectors)
            If type2 is given: (activated_scalars, gated_vectors, gated_type2)
        """
        s_out = self.scalar_act(scalars)

        # Vector gate
        if self.gate_proj_v is not None and vectors.shape[-2] > 0:
            norm_proj = getattr(self, 'norm_proj_v', None)
            v_out = self._gate_features(scalars, vectors, self.gate_proj_v, norm_proj)
        else:
            v_out = vectors

        # Type-2 gate
        if type2 is not None and self.gate_proj_t2 is not None:
            norm_proj = getattr(self, 'norm_proj_t2', None)
            t2_out = self._gate_features(scalars, type2, self.gate_proj_t2, norm_proj)
            return s_out, v_out, t2_out

        return s_out, v_out

    def extra_repr(self) -> str:
        return (
            f"num_scalars={self.num_scalars}, "
            f"num_vectors={self.num_vectors}, "
            f"num_type2={self.num_type2}, "
            f"gate_mode={self.gate_mode}"
        )
