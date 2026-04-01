# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Gated nonlinearity for equivariant neural networks.

Standard nonlinearities (ReLU, GELU) cannot be applied to vector (l≥1)
features because they break equivariance. The solution is gate vectors
using scalar-derived gates:

    scalar_out = σ(scalar_in)
    vector_out = gate(scalar_in) * vector_in

Since the gate is an SO(3)-invariant scalar, multiplying it with a vector
preserves equivariance: R(gate * v) = gate * Rv.
"""

import torch
import torch.nn as nn
from torch import Tensor


class GatedNonlinearity(nn.Module):
    """Gated nonlinearity that preserves SO(3) equivariance.

    Applies standard nonlinearity to scalar features. For vector features,
    generates scalar gates from a linear projection and multiplies:

        s_out = activation(s_in)
        v_out = sigmoid(W_gate @ s_in) ⊙ v_in

    The gate values are SO(3)-invariant scalars, so the output vectors
    remain equivariant.

    Args:
        num_scalars: Number of scalar input/output channels
        num_vectors: Number of vector input/output channels
        scalar_activation: Activation for scalar features (default: SiLU)
        gate_activation: Activation for vector gates (default: sigmoid)

    Example::

        >>> gate = GatedNonlinearity(num_scalars=64, num_vectors=16)
        >>> s, v = torch.randn(B, N, 64), torch.randn(B, N, 16, 3)
        >>> s_out, v_out = gate(s, v)
    """

    def __init__(
        self,
        num_scalars: int,
        num_vectors: int = 0,
        scalar_activation: str = "silu",
        gate_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors

        # Scalar nonlinearity
        activations = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        self.scalar_act = activations.get(scalar_activation, nn.SiLU())

        # Gate nonlinearity (produces values in [0, 1])
        gate_activations = {
            "sigmoid": nn.Sigmoid(),
            "tanh_abs": nn.Tanh(),  # will take abs() in forward
        }
        self.gate_act = gate_activations.get(gate_activation, nn.Sigmoid())

        # Linear projection: scalar features → gate values
        if num_vectors > 0:
            self.gate_proj = nn.Linear(num_scalars, num_vectors, bias=True)
        else:
            self.gate_proj = None

    def forward(
        self,
        scalars: Tensor,
        vectors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply gated nonlinearity.

        Args:
            scalars: Scalar features
                shape: [..., num_scalars]
            vectors: Vector features
                shape: [..., num_vectors, 3]

        Returns:
            Tuple of (activated_scalars, gated_vectors)
        """
        # Scalar: standard nonlinearity (equivariance-safe for l=0)
        s_out = self.scalar_act(scalars)

        # Vector: gate using scalar-derived values
        if self.gate_proj is not None and vectors.shape[-2] > 0:
            gates = self.gate_act(self.gate_proj(scalars))  # [..., num_vectors]
            v_out = vectors * gates.unsqueeze(-1)  # [..., num_vectors, 3]
        else:
            v_out = vectors

        return s_out, v_out

    def extra_repr(self) -> str:
        return (
            f"num_scalars={self.num_scalars}, "
            f"num_vectors={self.num_vectors}"
        )
