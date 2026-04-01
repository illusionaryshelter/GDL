# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Tensor product of SO(3) irreducible representations with CG coefficients.

This module implements the tensor product operation for coupling spherical
harmonic features, which is the core operation of all equivariant neural
networks in the Wigner-Eckart framework.

For two irreps of degree l₁ and l₂, the tensor product decomposes as:
    l₁ ⊗ l₂ = |l₁-l₂| ⊕ |l₁-l₂|+1 ⊕ ... ⊕ l₁+l₂

Phase 1 supports:
    - l=0 ⊗ l=0 → l=0  (scalar × scalar)
    - l=0 ⊗ l=1 → l=1  (scalar × vector = scaled vector)
    - l=1 ⊗ l=0 → l=1  (vector × scalar = scaled vector)
    - l=1 ⊗ l=1 → l=0  (dot product → scalar, invariant)
    - l=1 ⊗ l=1 → l=1  (cross product → vector, equivariant)
    - l=1 ⊗ l=1 → l=2  (symmetric traceless tensor)

CG coefficients are precomputed and stored as constants.
All operations preserve the equivariance property.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# Clebsch-Gordan Coefficients (precomputed for l ≤ 2)
# ═══════════════════════════════════════════════════════════════════

def _build_cg_coefficients() -> Dict[Tuple[int, int, int], Tensor]:
    """Build CG coefficient matrices for l₁, l₂ → l_out with l ≤ 2.

    CG coefficients C^{l_out, m_out}_{l1, m1, l2, m2} are stored as
    dense matrices of shape [(2*l1+1)*(2*l2+1), (2*l_out+1)].

    Returns:
        Dict mapping (l1, l2, l_out) → CG matrix
    """
    cg = {}

    # ── l₁=0, l₂=0 → l=0 ─────────────────────────────────────
    # Trivial: 1 × 1 → 1, coefficient = 1
    cg[(0, 0, 0)] = torch.tensor([[1.0]])

    # ── l₁=0, l₂=1 → l=1 ─────────────────────────────────────
    # Scalar × vector → vector: identity mapping
    # Shape: [1*3, 3] = [3, 3]
    cg[(0, 1, 1)] = torch.eye(3)

    # ── l₁=1, l₂=0 → l=1 ─────────────────────────────────────
    cg[(1, 0, 1)] = torch.eye(3)

    # ── l₁=1, l₂=1 → l=0 ─────────────────────────────────────
    # Dot product: v₁ · v₂ → scalar
    # CG coefficients: C^{0,0}_{1,m1,1,m2} = δ_{m1,-m2} * (-1)^m1 / √3
    # In Cartesian (m=-1,0,1 ↔ y,z,x):
    #   Y_1^{-1} · Y_1^{-1} + Y_1^0 · Y_1^0 + Y_1^1 · Y_1^1 → Y_0^0
    # Shape: [3*3, 1] = [9, 1]
    cg_11_0 = torch.zeros(9, 1)
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    # (m1=0, m2=0): C = 1/√3 → index (0*3+0, 0) in flattened
    # Using (y,z,x) ordering for m=-1,0,1:
    #   (m1=-1, m2=-1) → index 0*3+0=0: C = 1/√3
    #   (m1=0, m2=0)   → index 1*3+1=4: C = 1/√3
    #   (m1=1, m2=1)   → index 2*3+2=8: C = 1/√3
    cg_11_0[0, 0] = inv_sqrt3   # m1=-1, m2=-1
    cg_11_0[4, 0] = inv_sqrt3   # m1=0,  m2=0
    cg_11_0[8, 0] = inv_sqrt3   # m1=1,  m2=1
    cg[(1, 1, 0)] = cg_11_0

    # ── l₁=1, l₂=1 → l=1 ─────────────────────────────────────
    # Cross product: v₁ × v₂ → vector (equivariant)
    # CG coefficients encode the Levi-Civita structure
    # Shape: [3*3, 3] = [9, 3]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    cg_11_1 = torch.zeros(9, 3)
    # In (y,z,x) = (m=-1,0,1) basis:
    # Cross product ε_{ijk}: (v1)_i (v2)_j → (v1×v2)_k
    # m1=-1(y), m2=0(z) → m_out=1(x): C = 1/√2
    cg_11_1[0*3+1, 2] =  inv_sqrt2  # y×z → x
    cg_11_1[1*3+0, 2] = -inv_sqrt2  # z×y → -x
    # m1=0(z), m2=1(x) → m_out=-1(y): C = 1/√2
    cg_11_1[1*3+2, 0] =  inv_sqrt2  # z×x → y
    cg_11_1[2*3+1, 0] = -inv_sqrt2  # x×z → -y
    # m1=1(x), m2=-1(y) → m_out=0(z): C = 1/√2
    cg_11_1[2*3+0, 1] =  inv_sqrt2  # x×y → z
    cg_11_1[0*3+2, 1] = -inv_sqrt2  # y×x → -z
    cg[(1, 1, 1)] = cg_11_1

    # ── l₁=1, l₂=1 → l=2 ─────────────────────────────────────
    # Symmetric traceless outer product → quadrupole
    # Shape: [3*3, 5] = [9, 5]
    cg_11_2 = torch.zeros(9, 5)
    # m_out = -2,-1,0,1,2 for l=2
    # Using standard real CG coefficients:
    # Y_2^{-2}: xy component
    sqrt_half = math.sqrt(0.5)
    cg_11_2[0*3+2, 0] = sqrt_half   # y*x → Y2^{-2}
    cg_11_2[2*3+0, 0] = sqrt_half   # x*y → Y2^{-2}
    # Y_2^{-1}: yz component
    cg_11_2[0*3+1, 1] = sqrt_half   # y*z → Y2^{-1}
    cg_11_2[1*3+0, 1] = sqrt_half   # z*y → Y2^{-1}
    # Y_2^{0}: (2zz - xx - yy)/√6
    inv_sqrt6 = 1.0 / math.sqrt(6.0)
    cg_11_2[1*3+1, 2] =  2.0*inv_sqrt6   # z*z → Y2^0
    cg_11_2[2*3+2, 2] = -inv_sqrt6       # x*x → Y2^0
    cg_11_2[0*3+0, 2] = -inv_sqrt6       # y*y → Y2^0
    # Y_2^{1}: xz component
    cg_11_2[2*3+1, 3] = sqrt_half   # x*z → Y2^{1}
    cg_11_2[1*3+2, 3] = sqrt_half   # z*x → Y2^{1}
    # Y_2^{2}: (xx - yy)/√2
    inv_sqrt2_v = 1.0 / math.sqrt(2.0)
    cg_11_2[2*3+2, 4] =  inv_sqrt2_v   # x*x → Y2^2
    cg_11_2[0*3+0, 4] = -inv_sqrt2_v   # y*y → Y2^2
    cg[(1, 1, 2)] = cg_11_2

    return cg


# Global CG coefficient cache (computed once at import)
_CG_CACHE: Dict[Tuple[int, int, int], Tensor] = _build_cg_coefficients()


def get_cg_matrix(l1: int, l2: int, l_out: int) -> Tensor:
    """Get precomputed Clebsch-Gordan coefficient matrix.

    Args:
        l1: Degree of first input irrep
        l2: Degree of second input irrep
        l_out: Degree of output irrep

    Returns:
        CG matrix, shape: [(2*l1+1)*(2*l2+1), (2*l_out+1)]
    """
    key = (l1, l2, l_out)
    if key not in _CG_CACHE:
        raise ValueError(
            f"CG coefficients for ({l1}, {l2}) → {l_out} not precomputed. "
            f"Available: {list(_CG_CACHE.keys())}"
        )
    return _CG_CACHE[key]


# ═══════════════════════════════════════════════════════════════════
# Tensor Product Operations
# ═══════════════════════════════════════════════════════════════════

def tensor_product(
    f1: Tensor,
    f2: Tensor,
    l1: int,
    l2: int,
    l_out: int,
) -> Tensor:
    """Compute tensor product of two irrep features using CG coefficients.

    Computes: out_m = Σ_{m1,m2} C^{l_out,m}_{l1,m1,l2,m2} f1_{m1} f2_{m2}

    This is the fundamental coupling operation in equivariant networks.

    .. note:: **torch.compile safe**: This function operates on fixed-shape
        tensors only (determined by l1, l2, l_out). It is safe to decorate
        with ``@torch.compile``. The calling layer (e.g., SE3Conv) should
        NOT be compiled because ``radius_graph`` produces dynamic shapes.

    Args:
        f1: Features of type l1
            shape: [..., (2*l1+1)]
        f2: Features of type l2
            shape: [..., (2*l2+1)]
        l1: Degree of first input
        l2: Degree of second input
        l_out: Degree of output

    Returns:
        Coupled features, shape: [..., (2*l_out+1)]

    Example::

        >>> # Dot product: two vectors → scalar
        >>> v1 = torch.randn(100, 3)  # l=1 features
        >>> v2 = torch.randn(100, 3)  # l=1 features
        >>> s = tensor_product(v1, v2, 1, 1, 0)  # [100, 1]
    """
    cg = get_cg_matrix(l1, l2, l_out).to(f1.device, f1.dtype)

    batch_shape = f1.shape[:-1]
    n1 = 2 * l1 + 1
    n2 = 2 * l2 + 1

    # Outer product of features: [..., n1, n2] → [..., n1*n2]
    outer = (f1.unsqueeze(-1) * f2.unsqueeze(-2))  # [..., n1, n2]
    outer_flat = outer.reshape(*batch_shape, n1 * n2)  # [..., n1*n2]

    # Contract with CG coefficients: [..., n1*n2] @ [n1*n2, n_out]
    return outer_flat @ cg  # [..., n_out]


def tensor_product_weighted(
    f1: Tensor,
    f2: Tensor,
    l1: int,
    l2: int,
    l_out: int,
    weight: Tensor,
) -> Tensor:
    """Weighted tensor product — the learnable version.

    Computes: out = weight * CG(f1, f2)

    The weight is a per-path scalar that makes the tensor product learnable.
    This is the core of TP-based equivariant linear layers.

    Args:
        f1: Features of type l1, shape: [..., C_in1, (2*l1+1)]
        f2: Features of type l2, shape: [..., C_in2, (2*l2+1)]
        l1, l2, l_out: Irrep degrees
        weight: Learnable mixing weight, shape: [C_in1, C_in2, C_out]

    Returns:
        Output features, shape: [..., C_out, (2*l_out+1)]
    """
    cg = get_cg_matrix(l1, l2, l_out).to(f1.device, f1.dtype)

    n1 = 2 * l1 + 1
    n2 = 2 * l2 + 1
    n_out = 2 * l_out + 1
    C_in1 = f1.shape[-2]
    C_in2 = f2.shape[-2]
    C_out = weight.shape[-1]
    batch_shape = f1.shape[:-2]

    # Outer product: [..., C_in1, n1, 1] * [..., 1, C_in2, 1, n2]
    # → [..., C_in1, C_in2, n1, n2]
    outer = torch.einsum(
        '...im,...jn->...ijmn',
        f1, f2,
    )  # [..., C_in1, C_in2, n1, n2]

    # Contract with CG: [..., C_in1, C_in2, n1*n2] @ [n1*n2, n_out]
    outer_flat = outer.reshape(*batch_shape, C_in1, C_in2, n1 * n2)
    tp = outer_flat @ cg  # [..., C_in1, C_in2, n_out]

    # Mix with learned weights: [C_in1, C_in2, C_out]
    out = torch.einsum('...ijk,ijo->...ok', tp, weight)
    return out  # [..., C_out, n_out]


# ═══════════════════════════════════════════════════════════════════
# Convenience: direct physical operations
# ═══════════════════════════════════════════════════════════════════

def dot_product_l1(v1: Tensor, v2: Tensor) -> Tensor:
    """Invariant dot product of l=1 features.

    Equivalent to tensor_product(v1, v2, 1, 1, 0) but optimized.

    Args:
        v1: Vector features, shape: [..., 3]
        v2: Vector features, shape: [..., 3]

    Returns:
        Scalar features, shape: [..., 1]
    """
    return (v1 * v2).sum(dim=-1, keepdim=True) / math.sqrt(3.0)


def cross_product_l1(v1: Tensor, v2: Tensor) -> Tensor:
    """Equivariant cross product of l=1 features.

    Equivalent to tensor_product(v1, v2, 1, 1, 1) but optimized.

    Args:
        v1: Vector features, shape: [..., 3]
        v2: Vector features, shape: [..., 3]

    Returns:
        Vector features (l=1), shape: [..., 3]
    """
    return torch.linalg.cross(v1, v2, dim=-1) / math.sqrt(2.0)
