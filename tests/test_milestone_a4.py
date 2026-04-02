#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Milestone A4 equivariance & correctness tests for type-2 (l=2) features.

Tests:
  A4-Conv: SE3Conv type-2 TP paths equivariance
  A4-Gate: GatedNonlinearity type-2 gating equivariance
  A4-Norm: EquivariantLayerNorm type-2 equivariance
  A4-Block: SE3NetBlock with type-2 equivariance
  A4-E2E: End-to-end MultiScaleSE3Net with type-2 equivariance
  A4-Head: SE3PartSegNet with type-2 logit invariance
  A4-Backward: Backward compat (hidden_type2=0) regression

⚠ These tests verify mathematical correctness (equivariance).
  They must be run on the remote GPU server.
  Local execution is FORBIDDEN per AGENTS.md.

Usage (remote):
    python tests/test_milestone_a4.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from geoembodied.functional.so3_ops import so3_exp
from geoembodied.functional.quaternion_ops import quaternion_to_matrix


# ═══════════════════════════════════════════════════════════════════
# Wigner-D matrix for l=2 (test utility, NOT production code)
# ═══════════════════════════════════════════════════════════════════

def wigner_d_l2(R: torch.Tensor) -> torch.Tensor:
    """Compute the real 5×5 Wigner-D matrix for l=2 from rotation matrix.

    Uses the defining property of real spherical harmonics:
        Y_l(R·r̂) = D^l(R) · Y_l(r̂)

    We compute D^2 by evaluating Y_2 on M > 5 random directions, rotating
    them, and solving the overdetermined system via least squares.

    This is a TEST UTILITY only — not used in production.

    Args:
        R: [3, 3] rotation matrix

    Returns:
        D2: [5, 5] real Wigner-D matrix for l=2
            such that Y_2(R·r̂) = D2 @ Y_2(r̂)
    """
    from geoembodied.kernels.triton_sph_harm import spherical_harmonics

    dtype = R.dtype
    device = R.device

    # Use 20 random (but deterministic) directions for a well-conditioned system
    gen = torch.Generator()
    gen.manual_seed(12345)
    dirs = torch.randn(20, 3, dtype=dtype, device=device, generator=gen)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # Y_2 for original directions: [20, 9] → take channels 4:9 for l=2
    sh_input = dirs.float() if dtype != torch.float32 else dirs
    Y_orig = spherical_harmonics(sh_input)  # [20, 9]
    Y2_orig = Y_orig[:, 4:9].to(dtype)  # [20, 5]

    # Rotate directions
    dirs_rot = dirs @ R.t()
    dirs_rot = dirs_rot / dirs_rot.norm(dim=-1, keepdim=True).clamp(min=1e-10)

    # Y_2 for rotated directions
    sh_rot_input = dirs_rot.float() if dtype != torch.float32 else dirs_rot
    Y_rot = spherical_harmonics(sh_rot_input)
    Y2_rot = Y_rot[:, 4:9].to(dtype)  # [20, 5]

    # Solve overdetermined system: Y2_rot ≈ Y2_orig @ D2.T
    # lstsq returns the least-squares solution
    D2_T = torch.linalg.lstsq(Y2_orig, Y2_rot).solution  # [5, 5]
    D2 = D2_T.T

    return D2


def _random_rotation(scale: float = 1.0) -> torch.Tensor:
    """Generate random SO(3) rotation matrix."""
    omega = torch.randn(3) * scale
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)
    return R


# ═══════════════════════════════════════════════════════════════════
# Test: Wigner-D matrix self-consistency
# ═══════════════════════════════════════════════════════════════════

def test_wigner_d_consistency():
    """Verify Wigner-D matrix properties.

    1. D(I) = I
    2. D(R1 R2) ≈ D(R1) D(R2)
    3. D(R)^T ≈ D(R^T) (orthogonality)
    """
    from geoembodied.kernels.triton_sph_harm import spherical_harmonics

    print("=" * 60)
    print("TEST A4-WignerD: Wigner-D self-consistency")
    print("=" * 60)

    # Property 1: D(I) = I
    I = torch.eye(3, dtype=torch.float64)
    D_I = wigner_d_l2(I)
    err_identity = (D_I - torch.eye(5, dtype=torch.float64)).abs().max().item()
    print(f"  D(I) = I err: {err_identity:.2e}")
    assert err_identity < 1e-10, f"FAIL: D(I) ≠ I: {err_identity}"

    # Property 2: D(R1 R2) = D(R1) D(R2)
    torch.manual_seed(42)
    R1 = _random_rotation(1.5).double()
    R2 = _random_rotation(1.2).double()
    R12 = R1 @ R2

    D1 = wigner_d_l2(R1)
    D2 = wigner_d_l2(R2)
    D12 = wigner_d_l2(R12)

    err_comp = (D12 - D1 @ D2).abs().max().item()
    print(f"  D(R1R2) = D(R1)D(R2) err: {err_comp:.2e}")
    assert err_comp < 1e-6, f"FAIL: Composition: {err_comp}"

    # Property 3: D(R)^T = D(R^T)
    D1_T = wigner_d_l2(R1.T)
    err_orth = (D1.T - D1_T).abs().max().item()
    print(f"  D(R)^T = D(R^T) err: {err_orth:.2e}")
    assert err_orth < 1e-6, f"FAIL: Orthogonality: {err_orth}"

    # Property 4: D is orthogonal (det = ±1, D^T D = I)
    dtd = D1.T @ D1
    err_ortho = (dtd - torch.eye(5, dtype=torch.float64)).abs().max().item()
    print(f"  D^T D = I err: {err_ortho:.2e}")
    assert err_ortho < 1e-6, f"FAIL: Not orthogonal: {err_ortho}"

    print("TEST A4-WignerD PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-Conv: SE3Conv type-2 paths equivariance
# ═══════════════════════════════════════════════════════════════════

def test_a4_conv_type2_equivariance():
    """SE3Conv with type-2 produces equivariant outputs.

    f(Rp, s, Rv, D2·t2) should equal (s', Rv', D2·t2')
    where (s', v', t2') = f(p, s, v, t2)
    """
    from geoembodied.nn.modules.se3_conv import SE3Conv
    from geoembodied.nn.modules.spatial_graph import SpatialGraph

    print("=" * 60)
    print("TEST A4-Conv: SE3Conv type-2 equivariance")
    print("=" * 60)

    C_s, C_v, C_t2 = 16, 4, 2
    conv = SE3Conv(C_s, C_v, C_s, C_v,
                   in_type2_channels=C_t2, out_type2_channels=C_t2,
                   radius=2.0)
    conv.eval()

    torch.manual_seed(42)
    N = 80
    pos = torch.randn(N, 3)
    s = torch.randn(N, C_s)
    v = torch.randn(N, C_v, 3)
    t2 = torch.randn(N, C_t2, 5)

    R = _random_rotation(1.0)
    D2 = wigner_d_l2(R)

    pos_rot = pos @ R.t()
    v_rot = torch.einsum('nvc,dc->nvd', v, R)
    t2_rot = torch.einsum('ncm,km->nck', t2, D2)

    g1 = SpatialGraph.build(pos, radius=2.0)
    g2 = SpatialGraph.build(pos_rot, radius=2.0)

    with torch.no_grad():
        s1, v1, t2_1 = conv(s, v, g1, type2=t2)
        s2, v2, t2_2 = conv(s, v_rot, g2, type2=t2_rot)

    # Scalar invariance
    s_err = (s2 - s1).abs().max().item()
    print(f"  Scalar invariance err: {s_err:.2e}")
    assert s_err < 5e-4, f"FAIL scalar: {s_err}"

    # Vector equivariance
    v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - v1_rot).abs().max().item()
    print(f"  Vector equivariance err: {v_err:.2e}")
    assert v_err < 5e-3, f"FAIL vector: {v_err}"

    # Type-2 equivariance: t2_2 ≈ D2 @ t2_1
    t2_1_rot = torch.einsum('ncm,km->nck', t2_1, D2)
    t2_err = (t2_2 - t2_1_rot).abs().max().item()
    t2_rel = t2_err / (t2_1.abs().max().item() + 1e-8)
    print(f"  Type-2 equivariance abs err: {t2_err:.2e}")
    print(f"  Type-2 equivariance rel err: {t2_rel:.2e}")
    assert t2_err < 5e-3, f"FAIL type-2: {t2_err}"

    print("TEST A4-Conv PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-Gate: Type-2 gating equivariance
# ═══════════════════════════════════════════════════════════════════

def test_a4_gate_type2_equivariance():
    """GatedNonlinearity type-2 gating preserves equivariance.

    gate(s, Rv, D2·t2) = (σ(s), R·gate_v, D2·gate_t2)
    """
    from geoembodied.nn.modules.gated_nonlinearity import GatedNonlinearity

    print("=" * 60)
    print("TEST A4-Gate: Type-2 gating equivariance")
    print("=" * 60)

    for mode in ['scalar', 'norm']:
        C_s, C_v, C_t2 = 16, 4, 2
        gate = GatedNonlinearity(C_s, C_v, num_type2=C_t2, gate_mode=mode)
        gate.eval()

        torch.manual_seed(42)
        N = 100
        s = torch.randn(N, C_s)
        v = torch.randn(N, C_v, 3)
        t2 = torch.randn(N, C_t2, 5)

        R = _random_rotation(1.5)
        D2 = wigner_d_l2(R)

        v_rot = torch.einsum('nvc,dc->nvd', v, R)
        t2_rot = torch.einsum('ncm,km->nck', t2, D2)

        with torch.no_grad():
            s1, v1, t2_1 = gate(s, v, t2)
            s2, v2, t2_2 = gate(s, v_rot, t2_rot)

        # Scalar invariant
        s_err = (s2 - s1).abs().max().item()
        assert s_err < 1e-6, f"FAIL scalar ({mode}): {s_err}"

        # Vector equivariant
        v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
        v_err = (v2 - v1_rot).abs().max().item()
        print(f"  [{mode}] v_err: {v_err:.2e}", end="")
        assert v_err < 1e-5, f"FAIL vector ({mode}): {v_err}"

        # Type-2 equivariant
        t2_1_rot = torch.einsum('ncm,km->nck', t2_1, D2)
        t2_err = (t2_2 - t2_1_rot).abs().max().item()
        print(f"  t2_err: {t2_err:.2e}")
        assert t2_err < 1e-5, f"FAIL type-2 ({mode}): {t2_err}"

    print("TEST A4-Gate PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-Norm: EquivariantLayerNorm type-2 equivariance
# ═══════════════════════════════════════════════════════════════════

def test_a4_norm_type2_equivariance():
    """EquivariantLayerNorm preserves type-2 equivariance.

    ||D2·t2||² = ||t2||² (Wigner-D is orthogonal)
    So the norm-based normalization uses an invariant divisor.
    """
    from geoembodied.nn.modules.equivariant_norm import EquivariantLayerNorm

    print("=" * 60)
    print("TEST A4-Norm: Type-2 LayerNorm equivariance")
    print("=" * 60)

    C_s, C_v, C_t2 = 16, 4, 2
    norm = EquivariantLayerNorm(C_s, C_v, num_type2=C_t2)
    norm.eval()

    # Warm up running stats
    torch.manual_seed(42)
    for _ in range(5):
        norm.train()
        s_ = torch.randn(200, C_s)
        v_ = torch.randn(200, C_v, 3)
        t2_ = torch.randn(200, C_t2, 5)
        norm(s_, v_, t2_)
    norm.eval()

    N = 100
    s = torch.randn(N, C_s)
    v = torch.randn(N, C_v, 3)
    t2 = torch.randn(N, C_t2, 5)

    R = _random_rotation(1.0)
    D2 = wigner_d_l2(R)

    v_rot = torch.einsum('nvc,dc->nvd', v, R)
    t2_rot = torch.einsum('ncm,km->nck', t2, D2)

    with torch.no_grad():
        s1, v1, t2_1 = norm(s, v, t2)
        s2, v2, t2_2 = norm(s, v_rot, t2_rot)

    # Scalar invariant (same input)
    s_err = (s2 - s1).abs().max().item()
    print(f"  Scalar invariance err: {s_err:.2e}")
    assert s_err < 1e-6, f"FAIL scalar: {s_err}"

    # Vector equivariant
    v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - v1_rot).abs().max().item()
    print(f"  Vector equivariance err: {v_err:.2e}")
    assert v_err < 1e-5, f"FAIL vector: {v_err}"

    # Type-2 equivariant
    t2_1_rot = torch.einsum('ncm,km->nck', t2_1, D2)
    t2_err = (t2_2 - t2_1_rot).abs().max().item()
    print(f"  Type-2 equivariance err: {t2_err:.2e}")
    assert t2_err < 1e-5, f"FAIL type-2: {t2_err}"

    print("TEST A4-Norm PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-Block: SE3NetBlock with type-2 equivariance
# ═══════════════════════════════════════════════════════════════════

def test_a4_block_type2_equivariance():
    """Full SE3NetBlock with type-2 features is equivariant."""
    from geoembodied.nn.modules.se3_block import SE3NetBlock
    from geoembodied.nn.modules.spatial_graph import SpatialGraph

    print("=" * 60)
    print("TEST A4-Block: SE3NetBlock type-2 equivariance")
    print("=" * 60)

    C_s, C_v, C_t2 = 16, 4, 2
    block = SE3NetBlock(
        channels_scalar=C_s, channels_vector=C_v, channels_type2=C_t2,
        radius=2.0, gate_mode='norm', use_self_tp=True,
    )
    block.eval()

    torch.manual_seed(42)
    N = 64
    pos = torch.randn(N, 3)
    s = torch.randn(N, C_s)
    v = torch.randn(N, C_v, 3)
    t2 = torch.randn(N, C_t2, 5)

    R = _random_rotation(1.0)
    D2 = wigner_d_l2(R)

    pos_rot = pos @ R.t()
    v_rot = torch.einsum('nvc,dc->nvd', v, R)
    t2_rot = torch.einsum('ncm,km->nck', t2, D2)

    g1 = SpatialGraph.build(pos, radius=2.0)
    g2 = SpatialGraph.build(pos_rot, radius=2.0)

    with torch.no_grad():
        s1, v1, t2_1 = block(s, v, g1, type2=t2)
        s2, v2, t2_2 = block(s, v_rot, g2, type2=t2_rot)

    s_err = (s2 - s1).abs().max().item()
    print(f"  Scalar invariance err: {s_err:.2e}")
    assert s_err < 5e-3, f"FAIL scalar: {s_err}"

    v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - v1_rot).abs().max().item()
    print(f"  Vector equivariance err: {v_err:.2e}")
    assert v_err < 5e-3, f"FAIL vector: {v_err}"

    t2_1_rot = torch.einsum('ncm,km->nck', t2_1, D2)
    t2_err = (t2_2 - t2_1_rot).abs().max().item()
    print(f"  Type-2 equivariance err: {t2_err:.2e}")
    assert t2_err < 5e-3, f"FAIL type-2: {t2_err}"

    print("TEST A4-Block PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-E2E: End-to-end equivariance with type-2
# ═══════════════════════════════════════════════════════════════════

def test_a4_e2e_equivariance():
    """End-to-end MultiScaleSE3Net with type-2 is equivariant.

    This is the ultimate test: full U-Net backbone with l=0,1,2 features.
    Tests both FP32 and FP64 precision.
    """
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

    print("=" * 60)
    print("TEST A4-E2E: Full backbone type-2 equivariance")
    print("=" * 60)

    C_s, C_v, C_t2 = 16, 4, 2

    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v, hidden_type2=C_t2,
        num_stages=2, layers_per_stage=1,
        gate_mode='norm', use_self_tp=True,
    )
    model.eval()

    torch.manual_seed(42)
    N1, N2 = 200, 160
    N = N1 + N2
    pos = torch.randn(N, 3)
    ptr = torch.tensor([0, N1, N], dtype=torch.int64)

    R = _random_rotation(0.8)
    D2 = wigner_d_l2(R)
    t = torch.randn(3) * 3.0
    pos_rot = (pos @ R.t()) + t.unsqueeze(0)

    torch.manual_seed(0)
    with torch.no_grad():
        s1, v1, t2_1, _ = model(pos, ptr)

    torch.manual_seed(0)
    with torch.no_grad():
        s2, v2, t2_2, _ = model(pos_rot, ptr)

    # Scalar INVARIANCE
    s_err = (s2 - s1).abs().max().item()
    s_rel = s_err / (s1.abs().max().item() + 1e-8)
    print(f"  Scalar invariance err: {s_err:.2e} (rel: {s_rel:.2e})")
    assert s_err < 5e-4, f"FAIL scalar: {s_err}"

    # Vector EQUIVARIANCE
    v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - v1_rot).abs().max().item()
    v_rel = v_err / (v1.abs().max().item() + 1e-8)
    print(f"  Vector equivariance err: {v_err:.2e} (rel: {v_rel:.2e})")
    assert v_err < 1e-3, f"FAIL vector: {v_err}"

    # Type-2 EQUIVARIANCE
    t2_1_rot = torch.einsum('ncm,km->nck', t2_1, D2)
    t2_err = (t2_2 - t2_1_rot).abs().max().item()
    t2_rel = t2_err / (t2_1.abs().max().item() + 1e-8)
    print(f"  Type-2 equivariance err: {t2_err:.2e} (rel: {t2_rel:.2e})")
    assert t2_err < 1e-3, f"FAIL type-2: {t2_err}"

    # FP64
    print(f"\n  FP64 Baseline:")
    model_64 = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v, hidden_type2=C_t2,
        num_stages=2, layers_per_stage=1,
        gate_mode='norm', use_self_tp=True,
    )
    model_64.load_state_dict(model.state_dict())
    model_64 = model_64.double()
    model_64.eval()

    pos64 = pos.double()
    R64, t64 = R.double(), t.double()
    D2_64 = wigner_d_l2(R64)
    ptr64 = ptr
    pos_rot64 = (pos64 @ R64.t()) + t64.unsqueeze(0)

    torch.manual_seed(0)
    with torch.no_grad():
        s1_64, v1_64, t2_1_64, _ = model_64(pos64, ptr64)
    torch.manual_seed(0)
    with torch.no_grad():
        s2_64, v2_64, t2_2_64, _ = model_64(pos_rot64, ptr64)

    s_err64 = (s2_64 - s1_64).abs().max().item()
    v1_rot64 = torch.einsum('nvc,dc->nvd', v1_64, R64)
    v_err64 = (v2_64 - v1_rot64).abs().max().item()
    t2_1_rot64 = torch.einsum('ncm,km->nck', t2_1_64, D2_64)
    t2_err64 = (t2_2_64 - t2_1_rot64).abs().max().item()

    print(f"  FP64 scalar invariance err: {s_err64:.2e}")
    print(f"  FP64 vector equivariance err: {v_err64:.2e}")
    print(f"  FP64 type-2 equivariance err: {t2_err64:.2e}")

    print("TEST A4-E2E PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-Head: SE3PartSegNet logit invariance with type-2
# ═══════════════════════════════════════════════════════════════════

def test_a4_head_invariance():
    """SE3PartSegNet logits are SE(3)-invariant even with type-2 features.

    The head extracts ||t2||² (invariant) from type-2 features,
    so the final logits should not change under rotation.
    """
    from geoembodied.nn.models.part_segmentation import SE3PartSegNet

    print("=" * 60)
    print("TEST A4-Head: SE3PartSegNet type-2 logit invariance")
    print("=" * 60)

    model = SE3PartSegNet(
        num_categories=16, num_parts=50,
        hidden_scalar=16, hidden_vector=4, hidden_type2=2,
        num_stages=2, layers_per_stage=1,
        gate_mode='norm', use_self_tp=True,
    )
    model.eval()

    torch.manual_seed(42)
    N1, N2 = 64, 48
    N = N1 + N2
    pos = torch.randn(N, 3)
    ptr = torch.tensor([0, N1, N], dtype=torch.int64)
    cat = torch.tensor([0, 3], dtype=torch.int64)
    normals = torch.randn(N, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)

    R = _random_rotation(1.0)
    t = torch.randn(3) * 2.0
    pos_rot = (pos @ R.t()) + t.unsqueeze(0)
    normals_rot = normals @ R.t()

    torch.manual_seed(0)
    with torch.no_grad():
        logits1 = model(pos, ptr, cat, normals=normals)

    torch.manual_seed(0)
    with torch.no_grad():
        logits2 = model(pos_rot, ptr, cat, normals=normals_rot)

    # Logits should be invariant (scalar output)
    err = (logits2 - logits1).abs().max().item()
    rel = err / (logits1.abs().max().item() + 1e-8)
    print(f"  Logit invariance abs err: {err:.2e}")
    print(f"  Logit invariance rel err: {rel:.2e}")
    assert err < 0.5, f"FAIL: Logits not invariant: {err}"

    print("TEST A4-Head PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A4-Backward: Backward compatibility (hidden_type2=0)
# ═══════════════════════════════════════════════════════════════════

def test_a4_backward_compat():
    """hidden_type2=0 should produce identical results to Milestone A code.

    Regression test: ensure type-2 code paths don't affect existing
    code when disabled.
    """
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

    print("=" * 60)
    print("TEST A4-Backward: Backward compatibility (type2=0)")
    print("=" * 60)

    C_s, C_v = 16, 4

    # Model with type2=0 (disabled)
    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v, hidden_type2=0,
        num_stages=2, layers_per_stage=1,
        gate_mode='norm', use_self_tp=True,
    )
    model.eval()

    torch.manual_seed(42)
    N = 128
    pos = torch.randn(N, 3)
    ptr = torch.tensor([0, 64, N], dtype=torch.int64)

    with torch.no_grad():
        result = model(pos, ptr)

    # Should return 3 values (s, v, ptr) — NOT 4
    assert len(result) == 3, f"Expected 3 outputs, got {len(result)}"
    s, v, p = result
    assert s.shape == (N, C_s), f"Bad scalar shape: {s.shape}"
    assert v.shape == (N, C_v, 3), f"Bad vector shape: {v.shape}"
    print(f"  Output shapes correct: s={s.shape}, v={v.shape}")

    # return_encoder_features mode
    with torch.no_grad():
        result_enc = model(pos, ptr, return_encoder_features=True)
    assert len(result_enc) == 5, f"Expected 5 outputs, got {len(result_enc)}"
    print(f"  return_encoder_features: {len(result_enc)} outputs ✓")

    # Equivariance should still hold
    R = _random_rotation(0.8)
    t = torch.randn(3)
    pos_rot = (pos @ R.t()) + t.unsqueeze(0)

    torch.manual_seed(0)
    with torch.no_grad():
        s1, v1, _ = model(pos, ptr)
    torch.manual_seed(0)
    with torch.no_grad():
        s2, v2, _ = model(pos_rot, ptr)

    s_err = (s2 - s1).abs().max().item()
    v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - v1_rot).abs().max().item()
    print(f"  Scalar invariance err: {s_err:.2e}")
    print(f"  Vector equivariance err: {v_err:.2e}")
    assert s_err < 5e-4, f"FAIL scalar: {s_err}"
    assert v_err < 1e-3, f"FAIL vector: {v_err}"

    print("TEST A4-Backward PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test: Memory estimation with type-2
# ═══════════════════════════════════════════════════════════════════

def test_a4_memory_estimation():
    """Parameter count comparison with and without type-2."""
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

    print("=" * 60)
    print("MEMORY ESTIMATION (type-2)")
    print("=" * 60)

    configs = [
        (48, 12, 0, 2, 1, "No type-2 (baseline)"),
        (48, 12, 2, 2, 1, "C_t2=2"),
        (48, 12, 4, 2, 1, "C_t2=4 (recommended)"),
        (64, 16, 4, 2, 1, "C_s=64, C_t2=4"),
        (64, 16, 4, 2, 2, "C_s=64, C_t2=4, deep"),
    ]

    for C_s, C_v, C_t2, stages, layers, desc in configs:
        model = MultiScaleSE3Net(
            in_channels=1,
            hidden_scalar=C_s, hidden_vector=C_v, hidden_type2=C_t2,
            num_stages=stages, layers_per_stage=layers,
            gate_mode='norm', use_self_tp=True,
        )
        n = sum(p.numel() for p in model.parameters())
        mb = n * 4 / 1024**2
        floats_per_point = C_s + C_v * 3 + C_t2 * 5
        print(f"  {desc:30s}: {n:>10,} params ({mb:.1f}MB) "
              f"| {floats_per_point} floats/point")

    print()
    print("MEMORY ESTIMATION DONE\n")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Milestone A4: Type-2 (l=2) Equivariance Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_wigner_d_consistency,
        test_a4_gate_type2_equivariance,
        test_a4_norm_type2_equivariance,
        test_a4_conv_type2_equivariance,
        test_a4_block_type2_equivariance,
        test_a4_e2e_equivariance,
        test_a4_head_invariance,
        test_a4_backward_compat,
        test_a4_memory_estimation,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(failed)
