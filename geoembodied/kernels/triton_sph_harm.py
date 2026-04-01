# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Spherical harmonics Y_l^m(r̂) — Triton-accelerated or PyTorch fallback.

Implements real spherical harmonics up to l=2 (9 channels).
These are the angular basis functions used by SE3Conv for filter construction.

Supports both a Triton JIT kernel (GPU) and a pure PyTorch fallback (CPU/debug).

Channels layout (l=0..2, total 9):
    l=0: Y_0^0           (1 channel)     — isotropic / scalar
    l=1: Y_1^{-1,0,1}    (3 channels)    — dipole / vector
    l=2: Y_2^{-2,-1,0,1,2} (5 channels)  — quadrupole / tensor

Normalization: orthonormal on S², i.e. ∫ Y_l^m Y_{l'}^{m'} dΩ = δ_{ll'} δ_{mm'}

Reference:
    - e3nn spherical harmonics conventions
    - EquiTriton (Intel Labs) for Triton kernel design patterns
"""

import math
from typing import List, Optional

import torch
from torch import Tensor

# Try to import Triton; fall back to PyTorch if unavailable
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# Normalization constants for real spherical harmonics
# Y_l^m: coefficient = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
_C0 = math.sqrt(1.0 / (4.0 * math.pi))          # l=0: 1/√(4π)
_C1 = math.sqrt(3.0 / (4.0 * math.pi))           # l=1: √(3/(4π))
_C2_0 = math.sqrt(5.0 / (16.0 * math.pi))        # l=2, m=0: √(5/(16π))
_C2_1 = math.sqrt(15.0 / (4.0 * math.pi))        # l=2, |m|=1: √(15/(4π))
_C2_2 = math.sqrt(15.0 / (16.0 * math.pi))       # l=2, |m|=2: √(15/(16π))

# Channel counts per degree
_CHANNELS_PER_L = {0: 1, 1: 3, 2: 5}
_TOTAL_CHANNELS_L2 = 9  # sum of 1 + 3 + 5


# ═══════════════════════════════════════════════════════════════════
# Triton Kernels (GPU-accelerated)
# ═══════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:
    @triton.jit
    def _sph_harm_fwd_kernel(
        # Pointers
        xyz_ptr,       # [N, 3] input coordinates
        out_ptr,       # [N, 9] output spherical harmonics
        # Strides
        stride_xyz_n,  # stride for N dim in xyz
        stride_out_n,  # stride for N dim in out
        # Size
        N: tl.constexpr,
        # Constants (passed as kernel args for JIT)
        c0: tl.constexpr,
        c1: tl.constexpr,
        c2_0: tl.constexpr,
        c2_1: tl.constexpr,
        c2_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel: compute Y_l^m(x,y,z) for l=0,1,2.

        Input coordinates should be UNIT vectors (pre-normalized).
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load x, y, z components
        x = tl.load(xyz_ptr + offsets * stride_xyz_n + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + offsets * stride_xyz_n + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + offsets * stride_xyz_n + 2, mask=mask, other=0.0)

        # l=0: Y_0^0 = c0
        y00 = c0 + 0.0 * x  # broadcast to correct shape

        # l=1: Y_1^{-1} = c1*y, Y_1^0 = c1*z, Y_1^1 = c1*x
        y1m1 = c1 * y
        y10 = c1 * z
        y11 = c1 * x

        # l=2: five components
        y2m2 = c2_1 * x * y                          # Y_2^{-2} = √(15/4π) xy
        y2m1 = c2_1 * y * z                          # Y_2^{-1} = √(15/4π) yz
        y20 = c2_0 * (3.0 * z * z - 1.0)             # Y_2^0  = √(5/16π)(3z²-1)
        y21 = c2_1 * x * z                           # Y_2^1  = √(15/4π) xz
        y22 = c2_2 * (x * x - y * y)                 # Y_2^2  = √(15/16π)(x²-y²)

        # Store all 9 channels
        tl.store(out_ptr + offsets * stride_out_n + 0, y00, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 1, y1m1, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 2, y10, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 3, y11, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 4, y2m2, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 5, y2m1, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 6, y20, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 7, y21, mask=mask)
        tl.store(out_ptr + offsets * stride_out_n + 8, y22, mask=mask)

    @triton.jit
    def _sph_harm_bwd_kernel(
        # Pointers
        xyz_ptr,       # [N, 3] input coordinates
        grad_out_ptr,  # [N, 9] gradient w.r.t. output
        grad_xyz_ptr,  # [N, 3] gradient w.r.t. input (OUTPUT)
        # Strides
        stride_xyz_n,
        stride_grad_out_n,
        stride_grad_xyz_n,
        # Size
        N: tl.constexpr,
        # Constants
        c1: tl.constexpr,
        c2_0: tl.constexpr,
        c2_1: tl.constexpr,
        c2_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel: backward pass for spherical harmonics.

        Computes ∂L/∂(x,y,z) from ∂L/∂Y via chain rule.
        All derivatives are analytically hard-coded (no autograd overhead).
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load coordinates
        x = tl.load(xyz_ptr + offsets * stride_xyz_n + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + offsets * stride_xyz_n + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + offsets * stride_xyz_n + 2, mask=mask, other=0.0)

        # Load upstream gradients for all 9 channels
        g00 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 0, mask=mask, other=0.0)
        g1m1 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 1, mask=mask, other=0.0)
        g10 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 2, mask=mask, other=0.0)
        g11 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 3, mask=mask, other=0.0)
        g2m2 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 4, mask=mask, other=0.0)
        g2m1 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 5, mask=mask, other=0.0)
        g20 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 6, mask=mask, other=0.0)
        g21 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 7, mask=mask, other=0.0)
        g22 = tl.load(grad_out_ptr + offsets * stride_grad_out_n + 8, mask=mask, other=0.0)

        # Accumulate ∂L/∂x, ∂L/∂y, ∂L/∂z
        # l=0: ∂Y00/∂x = ∂Y00/∂y = ∂Y00/∂z = 0 (constant)

        # l=1: ∂Y1m1/∂y = c1, ∂Y10/∂z = c1, ∂Y11/∂x = c1
        grad_x = c1 * g11
        grad_y = c1 * g1m1
        grad_z = c1 * g10

        # l=2: chain rule for each component
        # ∂(xy)/∂x = y, ∂(xy)/∂y = x
        grad_x = grad_x + c2_1 * y * g2m2
        grad_y = grad_y + c2_1 * x * g2m2

        # ∂(yz)/∂y = z, ∂(yz)/∂z = y
        grad_y = grad_y + c2_1 * z * g2m1
        grad_z = grad_z + c2_1 * y * g2m1

        # ∂(3z²-1)/∂z = 6z
        grad_z = grad_z + c2_0 * 6.0 * z * g20

        # ∂(xz)/∂x = z, ∂(xz)/∂z = x
        grad_x = grad_x + c2_1 * z * g21
        grad_z = grad_z + c2_1 * x * g21

        # ∂(x²-y²)/∂x = 2x, ∂(x²-y²)/∂y = -2y
        grad_x = grad_x + c2_2 * 2.0 * x * g22
        grad_y = grad_y - c2_2 * 2.0 * y * g22

        # Store gradients
        tl.store(grad_xyz_ptr + offsets * stride_grad_xyz_n + 0, grad_x, mask=mask)
        tl.store(grad_xyz_ptr + offsets * stride_grad_xyz_n + 1, grad_y, mask=mask)
        tl.store(grad_xyz_ptr + offsets * stride_grad_xyz_n + 2, grad_z, mask=mask)


# ═══════════════════════════════════════════════════════════════════
# PyTorch fallback (CPU / debug / no-Triton)
# ═══════════════════════════════════════════════════════════════════

def _spherical_harmonics_pytorch(
    xyz: Tensor,
    max_l: int = 2,
) -> Tensor:
    """Pure PyTorch spherical harmonics — CPU fallback.

    Args:
        xyz: Unit direction vectors
            shape: [..., 3]
        max_l: Maximum spherical harmonic degree (0, 1, or 2)

    Returns:
        Spherical harmonics values
            shape: [..., (max_l+1)²]
    """
    x = xyz[..., 0]  # [...]
    y = xyz[..., 1]
    z = xyz[..., 2]

    outputs = []

    # l=0
    outputs.append(_C0 * torch.ones_like(x).unsqueeze(-1))  # [..., 1]

    if max_l >= 1:
        # l=1: Y_1^{-1,0,1} = c1 * (y, z, x)
        l1 = torch.stack([_C1 * y, _C1 * z, _C1 * x], dim=-1)  # [..., 3]
        outputs.append(l1)

    if max_l >= 2:
        # l=2: five components
        l2 = torch.stack([
            _C2_1 * x * y,                    # Y_2^{-2}
            _C2_1 * y * z,                    # Y_2^{-1}
            _C2_0 * (3.0 * z * z - 1.0),      # Y_2^0
            _C2_1 * x * z,                    # Y_2^1
            _C2_2 * (x * x - y * y),          # Y_2^2
        ], dim=-1)  # [..., 5]
        outputs.append(l2)

    return torch.cat(outputs, dim=-1)  # [..., (max_l+1)²]


# ═══════════════════════════════════════════════════════════════════
# Autograd Function wrapper
# ═══════════════════════════════════════════════════════════════════

class SphericalHarmonicsFunction(torch.autograd.Function):
    """Custom autograd for spherical harmonics with Triton backward.

    Forward: Triton kernel (GPU) or PyTorch (CPU)
    Backward: Triton kernel with analytically hard-coded derivatives
    """

    @staticmethod
    def forward(
        ctx,
        xyz: Tensor,
        max_l: int = 2,
    ) -> Tensor:
        """Compute Y_l(r̂) for l = 0..max_l.

        Args:
            xyz: Unit direction vectors, shape: [N, 3] (must be 2D for Triton)
            max_l: Maximum degree (0, 1, or 2)
        """
        ctx.save_for_backward(xyz)
        ctx.max_l = max_l

        if _TRITON_AVAILABLE and xyz.is_cuda and max_l == 2:
            # Use Triton kernel
            N = xyz.shape[0]
            out = torch.empty(N, _TOTAL_CHANNELS_L2, device=xyz.device, dtype=xyz.dtype)

            BLOCK_SIZE = 256
            grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

            _sph_harm_fwd_kernel[grid](
                xyz, out,
                xyz.stride(0), out.stride(0),
                N,
                _C0, _C1, _C2_0, _C2_1, _C2_2,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out
        else:
            # PyTorch fallback
            return _spherical_harmonics_pytorch(xyz, max_l)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        xyz, = ctx.saved_tensors
        max_l = ctx.max_l

        if _TRITON_AVAILABLE and xyz.is_cuda and max_l == 2:
            N = xyz.shape[0]
            grad_xyz = torch.empty_like(xyz)

            BLOCK_SIZE = 256
            grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

            _sph_harm_bwd_kernel[grid](
                xyz, grad_output.contiguous(), grad_xyz,
                xyz.stride(0), grad_output.stride(0), grad_xyz.stride(0),
                N,
                _C1, _C2_0, _C2_1, _C2_2,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return grad_xyz, None
        else:
            # PyTorch autograd fallback
            with torch.enable_grad():
                xyz_ag = xyz.detach().requires_grad_(True)
                out = _spherical_harmonics_pytorch(xyz_ag, max_l)
                out.backward(grad_output)
            return xyz_ag.grad, None


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def spherical_harmonics(
    xyz: Tensor,
    max_l: int = 2,
    normalize: bool = True,
) -> Tensor:
    """Compute real spherical harmonics Y_l^m(r̂) for l = 0..max_l.

    This is the main entry point for spherical harmonics computation.
    Automatically selects Triton (GPU) or PyTorch (CPU) backend.

    Args:
        xyz: Direction vectors
            shape: [..., 3]
            If normalize=True, vectors will be L2-normalized internally.
            If normalize=False, vectors MUST be unit vectors.
        max_l: Maximum spherical harmonic degree (0, 1, or 2)
        normalize: Whether to L2-normalize input vectors (default: True)

    Returns:
        Spherical harmonics values
            shape: [..., (max_l + 1)²]
            Channel layout: [Y_0^0 | Y_1^{-1,0,1} | Y_2^{-2..2}]

    Example::

        >>> r = torch.randn(1000, 3, requires_grad=True)
        >>> Y = spherical_harmonics(r, max_l=2)  # [1000, 9]
        >>> Y.shape
        torch.Size([1000, 9])

    Equivariance property (tested in test suite):
        Y_l(R @ r̂) = D^l(R) @ Y_l(r̂)
        where D^l is the Wigner-D matrix of degree l.
    """
    assert max_l in (0, 1, 2), f"max_l must be 0, 1, or 2, got {max_l}"

    original_shape = xyz.shape[:-1]

    if normalize:
        xyz = xyz / xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Flatten batch dims for Triton kernel compatibility
    flat_xyz = xyz.reshape(-1, 3)
    result = SphericalHarmonicsFunction.apply(flat_xyz, max_l)

    # Restore batch dimensions
    num_channels = (max_l + 1) ** 2
    return result.reshape(*original_shape, num_channels)


def get_l_channels(max_l: int) -> List[int]:
    """Get number of output channels per degree.

    Args:
        max_l: Maximum degree

    Returns:
        List of channel counts: [1, 3, 5, ...] for l=0,1,2,...
    """
    return [2 * l + 1 for l in range(max_l + 1)]
