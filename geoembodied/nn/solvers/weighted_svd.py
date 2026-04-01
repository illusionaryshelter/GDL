# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Weighted SVD — differentiable rigid alignment solver (Kabsch algorithm).

Computes the optimal SE(3) rigid transform (R, t) that minimizes:
    Σ wᵢ ‖R·src_i + t − tgt_i‖²

Uses SVD of the weighted cross-covariance matrix H = Σ wᵢ (src_i − c_s)(tgt_i − c_t)ᵀ.

Includes Rule 4 defence: asymmetric diagonal perturbation to H prevents
SVD backward NaN from degenerate (equal) singular values (coplanar or
symmetric geometry).

No learnable parameters — this is a pure mathematical solver.

Used in:
- Point cloud registration (computing SE(3) from correspondences)
- SLAM (incremental pose estimation)
- Manipulation (object pose refinement)
"""

import logging
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)
_warned_svd_fallback = False


def _safe_svd(H: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """SVD with automatic CPU fallback for CUDA cuSOLVER failures.

    cuSOLVER's batched gesvdjBatched can fail on certain driver/hardware
    combinations. Since H is always [3, 3] or [B, 3, 3], the CPU fallback
    has negligible overhead (<0.1ms).

    Args:
        H: Cross-covariance matrix [3, 3] or [B, 3, 3], float32

    Returns:
        U, S, Vh — all on the same device as input H
    """
    global _warned_svd_fallback
    try:
        return torch.linalg.svd(H)
    except RuntimeError:
        if not _warned_svd_fallback:
            logger.warning(
                "CUDA SVD failed (cuSOLVER bug), falling back to CPU. "
                "This has negligible overhead for 3×3 matrices."
            )
            _warned_svd_fallback = True
        device = H.device
        U, S, Vh = torch.linalg.svd(H.cpu())
        return U.to(device), S.to(device), Vh.to(device)


def _det3x3(M: Tensor) -> Tensor:
    """Analytical determinant for 3×3 matrices.

    Avoids torch.det which uses LU decomposition (cuBLAS batched LU can fail).
    For 3×3 matrices this is faster AND more robust.

    Args:
        M: [..., 3, 3] matrix

    Returns:
        det: [...] scalar determinant
    """
    return (
        M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
        - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
        + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0])
    )


def weighted_svd(
    src: Tensor,
    tgt: Tensor,
    weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Weighted SVD for rigid alignment (single pair).

    Args:
        src: Source points
            shape: [N, 3]
        tgt: Target points (soft correspondences)
            shape: [N, 3]
        weights: Per-point confidence
            shape: [N]

    Returns:
        R: Rotation matrix
            shape: [3, 3], representation: SO(3)
        t: Translation vector
            shape: [3]
    """
    # Normalize weights
    w = weights / weights.sum().clamp(min=1e-8)

    # Weighted centroids
    centroid_src = (w.unsqueeze(-1) * src).sum(dim=0)
    centroid_tgt = (w.unsqueeze(-1) * tgt).sum(dim=0)

    src_c = src - centroid_src.unsqueeze(0)
    tgt_c = tgt - centroid_tgt.unsqueeze(0)

    # Weighted cross-covariance
    H = (w.unsqueeze(-1) * src_c).T @ tgt_c

    # Rule 4 defence: asymmetric forward perturbation for SVD stability.
    # SVD backward contains 1/(σi²−σj²) which explodes when singular
    # values are equal (coplanar geometry) or close (symmetric shapes).
    # We inject H += diag(ε, 2ε, 3ε) scaled to trace(H) so that:
    #   (a) |σi − σj| ≥ ε guaranteeing finite backward gradients
    #   (b) perturbation is proportional to H's scale → FP32-safe
    #       (avoids 1e-6 being truncated when H~1e4)
    #   (c) asymmetric values ensure no two σ can become equal
    H_f32 = H.float()
    eps_svd = torch.clamp(
        H_f32.diag().abs().sum() * 1e-6, min=1e-7,
    )
    perturbation = torch.tensor(
        [1.0, 2.0, 3.0], device=H.device, dtype=torch.float32,
    ) * eps_svd
    H_safe = H_f32 + torch.diag(perturbation)

    # SVD — force float32, CPU fallback for cuSOLVER bugs
    U, S, Vh = _safe_svd(H_safe)

    # Correct reflection (analytical det — avoids cuBLAS LU failures)
    d = _det3x3(Vh.T @ U.T)
    sign_val = d.sign().detach()
    sign_fix = torch.diag(torch.stack([
        torch.ones(1, device=src.device).squeeze(),
        torch.ones(1, device=src.device).squeeze(),
        sign_val,
    ]))
    R = Vh.T @ sign_fix @ U.T
    t = centroid_tgt - R @ centroid_src

    return R, t


def weighted_svd_batched(
    src: Tensor,
    tgt: Tensor,
    weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Batched weighted SVD for rigid alignment (Kabsch).

    All operations are fully batched — no Python loops.
    Uses torch.linalg.svd on [B, 3, 3] cross-covariance matrices.

    Args:
        src: Source points
            shape: [B, N, 3]
        tgt: Target points (soft correspondences)
            shape: [B, N, 3]
        weights: Per-point confidence
            shape: [B, N]

    Returns:
        R: Rotation matrices
            shape: [B, 3, 3], representation: SO(3)
        t: Translation vectors
            shape: [B, 3]
    """
    B = src.shape[0]
    device = src.device

    # Normalize weights per sample
    w = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, N]

    # Weighted centroids [B, 3]
    centroid_src = (w.unsqueeze(-1) * src).sum(dim=1)  # [B, 3]
    centroid_tgt = (w.unsqueeze(-1) * tgt).sum(dim=1)  # [B, 3]

    # Center
    src_c = src - centroid_src.unsqueeze(1)  # [B, N, 3]
    tgt_c = tgt - centroid_tgt.unsqueeze(1)  # [B, N, 3]

    # Weighted cross-covariance [B, 3, 3]
    # H = (w * src_c)^T @ tgt_c = [B, 3, N] @ [B, N, 3]
    H = torch.bmm((w.unsqueeze(-1) * src_c).transpose(1, 2), tgt_c)

    # Rule 4 defence: asymmetric forward perturbation for SVD stability.
    # Dynamic eps scaled to trace(H) for FP32 safety.
    H_f32 = H.float()
    trace_H = H_f32.diagonal(dim1=-2, dim2=-1).abs().sum(dim=-1)  # [B]
    eps_svd = torch.clamp(trace_H * 1e-6, min=1e-7)  # [B]
    # Build [B, 3] perturbation → [B, 3, 3] diagonal
    scales = torch.tensor(
        [1.0, 2.0, 3.0], device=H.device, dtype=torch.float32,
    ).unsqueeze(0)  # [1, 3]
    perturbation = eps_svd.unsqueeze(-1) * scales  # [B, 3]
    H_safe = H_f32 + torch.diag_embed(perturbation)  # [B, 3, 3]

    # Batched SVD [B, 3, 3] — CPU fallback for cuSOLVER bugs
    U, S, Vh = _safe_svd(H_safe)

    # Correct reflection: analytical det (avoids cuBLAS LU failures)
    # [B, 3, 3] @ [B, 3, 3] → det → [B]
    d = _det3x3(Vh.transpose(1, 2) @ U.transpose(1, 2))
    sign_val = d.sign().detach()  # [B]

    # Build sign correction diagonal [B, 3, 3]
    ones = torch.ones(B, 2, device=device)
    sign_diag = torch.cat([ones, sign_val.unsqueeze(-1)], dim=-1)  # [B, 3]
    sign_fix = torch.diag_embed(sign_diag)  # [B, 3, 3]

    # R = Vh^T @ sign_fix @ U^T
    R = torch.bmm(torch.bmm(Vh.transpose(1, 2), sign_fix), U.transpose(1, 2))

    # t = centroid_tgt - R @ centroid_src
    t = centroid_tgt - torch.bmm(R, centroid_src.unsqueeze(-1)).squeeze(-1)

    return R, t
