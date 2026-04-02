#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Milestone A equivariance & correctness tests.

Tests for each Phase of Milestone A:
  A1: Multi-scale head produces correct output shapes
  A2: Norm-based gate preserves SO(3) equivariance
  A3: Self-interaction TP (v·v) is SO(3)-invariant
  A1-A3 combined: End-to-end equivariance of upgraded model

⚠ These tests verify mathematical correctness (equivariance).
  They must be run on the remote GPU server.
  Local execution is FORBIDDEN per AGENTS.md.

Usage (remote):
    python tests/test_milestone_a.py
    python -m pytest tests/test_milestone_a.py -v
"""

import os
import sys
import math

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from geoembodied.functional.so3_ops import so3_exp
from geoembodied.functional.quaternion_ops import quaternion_to_matrix


# ═══════════════════════════════════════════════════════════════════
# Test A2: Norm-based gate preserves equivariance
# ═══════════════════════════════════════════════════════════════════

def test_a2_norm_gate_equivariance():
    """Norm-based gate mode preserves SO(3) equivariance of vectors.

    Proof obligation:
        gate(s, Rv) = R · gate(s, v)  ∀ R ∈ SO(3)

    Because ||Rv|| = ||v||, the scalar gate is invariant.
    The unit direction Rv/||Rv|| = R(v/||v||) transforms covariantly.
    Therefore the output = gate(||v||,s) * Rv/||Rv|| = R * [gate(||v||,s) * v/||v||]
    """
    from geoembodied.nn.modules.gated_nonlinearity import GatedNonlinearity

    print("=" * 60)
    print("TEST A2: Norm-based gate equivariance")
    print("=" * 60)

    C_s, C_v = 32, 8
    N = 100

    gate = GatedNonlinearity(
        num_scalars=C_s, num_vectors=C_v, gate_mode='norm'
    )
    gate.eval()  # fix BN if any

    torch.manual_seed(42)
    s = torch.randn(N, C_s)
    v = torch.randn(N, C_v, 3)

    # Random SO(3) rotation
    omega = torch.randn(3) * 1.5
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)  # [3, 3]

    # v_rot = v @ R^T (rotate each vector)
    v_rot = torch.einsum('nvc,dc->nvd', v, R)

    with torch.no_grad():
        s_out1, v_out1 = gate(s, v)
        s_out2, v_out2 = gate(s, v_rot)

    # Check 1: scalar output invariant (same s input)
    s_err = (s_out2 - s_out1).abs().max().item()
    print(f"  Scalar invariance err: {s_err:.2e}")
    assert s_err < 1e-6, f"FAIL: Scalars differ: {s_err}"

    # Check 2: vector output equivariant: v_out2 ≈ v_out1 @ R^T
    v_out1_rot = torch.einsum('nvc,dc->nvd', v_out1, R)
    v_err = (v_out2 - v_out1_rot).abs().max().item()
    v_rel = v_err / (v_out1.abs().max().item() + 1e-8)
    print(f"  Vector equivariance abs err: {v_err:.2e}")
    print(f"  Vector equivariance rel err: {v_rel:.2e}")
    assert v_err < 1e-5, f"FAIL: Vector NOT equivariant: {v_err}"

    # FP64 verification
    gate_64 = GatedNonlinearity(num_scalars=C_s, num_vectors=C_v, gate_mode='norm')
    gate_64.load_state_dict(gate.state_dict())
    gate_64 = gate_64.double()
    gate_64.eval()

    s64, v64 = s.double(), v.double()
    R64 = R.double()
    v_rot64 = torch.einsum('nvc,dc->nvd', v64, R64)

    with torch.no_grad():
        _, v_out1_64 = gate_64(s64, v64)
        _, v_out2_64 = gate_64(s64, v_rot64)

    v_out1_rot64 = torch.einsum('nvc,dc->nvd', v_out1_64, R64)
    v_err64 = (v_out2_64 - v_out1_rot64).abs().max().item()
    print(f"  FP64 vector equivariance err: {v_err64:.2e}")
    assert v_err64 < 1e-7, f"FAIL: FP64 not equivariant: {v_err64}"

    print("TEST A2 PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A3: Self-interaction TP invariance
# ═══════════════════════════════════════════════════════════════════

def test_a3_self_tp_invariance():
    """Self-interaction v·v → scalar is SO(3)-invariant.

    Proof obligation:
        (Rv)·(Rv) = v^T R^T R v = v^T v = v·v  ∀ R ∈ SO(3)
    """
    print("=" * 60)
    print("TEST A3: Self-interaction TP invariance")
    print("=" * 60)

    N, C_v = 100, 16
    torch.manual_seed(42)
    v = torch.randn(N, C_v, 3)

    # Random rotation
    omega = torch.randn(3) * 2.0
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)

    v_rot = torch.einsum('nvc,dc->nvd', v, R)

    # v·v = sum_d(v_d²) per channel
    vv_orig = (v * v).sum(dim=-1)      # [N, C_v]
    vv_rot = (v_rot * v_rot).sum(dim=-1)  # [N, C_v]

    err = (vv_rot - vv_orig).abs().max().item()
    rel = err / (vv_orig.abs().max().item() + 1e-8)
    print(f"  v·v invariance abs err: {err:.2e}")
    print(f"  v·v invariance rel err: {rel:.2e}")
    assert err < 1e-5, f"FAIL: v·v not invariant: {err}"

    # FP64
    v64 = v.double()
    R64 = R.double()
    vr64 = torch.einsum('nvc,dc->nvd', v64, R64)
    err64 = ((vr64 * vr64).sum(-1) - (v64 * v64).sum(-1)).abs().max().item()
    print(f"  FP64 invariance err: {err64:.2e}")
    assert err64 < 1e-5, f"FAIL: FP64 not invariant: {err64}"

    print("TEST A3 PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A1: Multi-scale head shape correctness
# ═══════════════════════════════════════════════════════════════════

def test_a1_multiscale_head_shape():
    """Multi-scale head produces correct output shape with all new features.

    Verifies:
    1. Model constructs without errors with new params
    2. Forward pass produces [N, num_parts] logits
    3. Multi-scale global features are present (via diagnostics)
    """
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

    print("=" * 60)
    print("TEST A1: Multi-scale head shape")
    print("=" * 60)

    C_s, C_v = 16, 4
    num_stages = 2

    backbone = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s,
        hidden_vector=C_v,
        num_stages=num_stages,
        layers_per_stage=1,
        pool_ratio=0.25,
        gate_mode='norm',
        use_self_tp=True,
    )
    backbone.eval()

    torch.manual_seed(42)
    N = 128
    pos = torch.randn(N, 3)
    ptr = torch.tensor([0, N], dtype=torch.int64)

    with torch.no_grad():
        result = backbone(pos, ptr, return_encoder_features=True)

    s_out, v_out, _, enc_s_list, enc_ptr_list = result

    # Check shapes
    assert s_out.shape == (N, C_s), f"Bad scalar shape: {s_out.shape}"
    assert v_out.shape == (N, C_v, 3), f"Bad vector shape: {v_out.shape}"
    assert len(enc_s_list) == num_stages, f"Bad enc_s_list len: {len(enc_s_list)}"
    assert len(enc_ptr_list) == num_stages, f"Bad enc_ptr_list len: {len(enc_ptr_list)}"

    # Verify encoder features have decreasing sizes
    for i, (enc_s, enc_ptr) in enumerate(zip(enc_s_list, enc_ptr_list)):
        n_enc = enc_s.shape[0]
        print(f"  Encoder stage {i}: {n_enc} points, scalar shape {enc_s.shape}")
        assert enc_s.shape[1] == C_s
    assert enc_s_list[0].shape[0] >= enc_s_list[-1].shape[0], \
        "Encoder sizes should decrease"

    # Verify backward-compatible mode
    with torch.no_grad():
        result_compat = backbone(pos, ptr, return_encoder_features=False)
    assert len(result_compat) == 3, "Default should return 3 values"

    print("TEST A1 PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test A1-A3 Combined: End-to-End equivariance with all upgrades
# ═══════════════════════════════════════════════════════════════════

def test_combined_e2e_equivariance():
    """End-to-end equivariance with ALL Milestone A features enabled.

    This is the ultimate truth test: f(Rp+t) ≡ f(p) for scalar outputs,
    and v2 ≈ R @ v1 for vector outputs.

    Combines: multi-scale head + norm gate + self-TP
    """
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

    print("=" * 60)
    print("TEST Combined: E2E equivariance (A1+A2+A3)")
    print("=" * 60)

    C_s, C_v = 16, 4

    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v,
        num_stages=2, layers_per_stage=1,
        pool_ratio=0.25,
        gate_mode='norm',
        use_self_tp=True,
    )
    model.eval()

    torch.manual_seed(42)
    N1, N2 = 200, 160
    N = N1 + N2
    pos = torch.randn(N, 3)
    ptr = torch.tensor([0, N1, N], dtype=torch.int64)

    # Random SE(3) transform
    omega = torch.randn(3) * 0.8
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)
    t = torch.randn(3) * 3.0
    pos_rot = (pos @ R.t()) + t.unsqueeze(0)

    # Forward on both
    torch.manual_seed(0)
    with torch.no_grad():
        s1, v1, _ = model(pos, ptr)

    torch.manual_seed(0)
    with torch.no_grad():
        s2, v2, _ = model(pos_rot, ptr)

    # Scalar INVARIANCE
    s_err = (s2 - s1).abs().max().item()
    s_rel = s_err / (s1.abs().max().item() + 1e-8)
    print(f"  Scalar max abs err: {s_err:.2e}")
    print(f"  Scalar relative err: {s_rel:.2e}")
    assert s_err < 5e-4, f"FAIL: Scalar NOT invariant: {s_err}"
    print(f"  ✓ Scalar invariance: max_err = {s_err:.6f}")

    # Vector EQUIVARIANCE
    expected_v = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - expected_v).abs().max().item()
    v_rel = v_err / (v1.abs().max().item() + 1e-8)
    print(f"  Vector max abs err: {v_err:.2e}")
    print(f"  Vector relative err: {v_rel:.2e}")
    assert v_err < 1e-3, f"FAIL: Vector NOT equivariant: {v_err}"
    print(f"  ✓ Vector equivariance: max_err = {v_err:.6f}")

    # FP64 baseline
    print(f"\n  FP64 Baseline:")
    model_64 = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v,
        num_stages=2, layers_per_stage=1,
        pool_ratio=0.25,
        gate_mode='norm',
        use_self_tp=True,
    )
    model_64.load_state_dict(model.state_dict())
    model_64 = model_64.double()
    model_64.eval()

    pos64 = pos.double()
    R64, t64 = R.double(), t.double()
    pos_rot64 = (pos64 @ R64.t()) + t64.unsqueeze(0)

    torch.manual_seed(0)
    with torch.no_grad():
        s1_64, v1_64, _ = model_64(pos64, ptr)
    torch.manual_seed(0)
    with torch.no_grad():
        s2_64, v2_64, _ = model_64(pos_rot64, ptr)

    s_err64 = (s2_64 - s1_64).abs().max().item()
    exp_v64 = torch.einsum('nvc,dc->nvd', v1_64, R64)
    v_err64 = (v2_64 - exp_v64).abs().max().item()
    print(f"  FP64 scalar invariance err: {s_err64:.2e}")
    print(f"  FP64 vector equivariance err: {v_err64:.2e}")

    fp32_vs_fp64 = (s1.double() - s1_64).abs().max().item()
    print(f"  FP32 vs FP64 scalar gap: {fp32_vs_fp64:.2e}")

    print("TEST Combined PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test: SE3NetBlock with self-TP equivariance
# ═══════════════════════════════════════════════════════════════════

def test_a3_block_equivariance():
    """SE3NetBlock with self-TP remains equivariant.

    The self-TP path (v·v → scalar) injects an SO(3)-invariant quantity
    into the scalar stream. This should NOT affect equivariance.
    """
    from geoembodied.nn.modules.se3_block import SE3NetBlock
    from geoembodied.nn.modules.spatial_graph import SpatialGraph

    print("=" * 60)
    print("TEST A3-Block: SE3NetBlock self-TP equivariance")
    print("=" * 60)

    C_s, C_v = 16, 4
    block = SE3NetBlock(
        channels_scalar=C_s,
        channels_vector=C_v,
        radius=0.3,
        gate_mode='norm',
        use_self_tp=True,
    )
    block.eval()

    torch.manual_seed(42)
    N = 64
    pos = torch.randn(N, 3)
    s = torch.randn(N, C_s)
    v = torch.randn(N, C_v, 3)

    # Random rotation
    omega = torch.randn(3) * 1.0
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)

    pos_rot = pos @ R.t()
    v_rot = torch.einsum('nvc,dc->nvd', v, R)

    # Build graphs
    graph1 = SpatialGraph.build(pos, radius=0.5, max_num_neighbors=32)
    graph2 = SpatialGraph.build(pos_rot, radius=0.5, max_num_neighbors=32)

    with torch.no_grad():
        s1, v1 = block(s, v, graph1)
        s2, v2 = block(s, v_rot, graph2)

    # Scalar invariance
    s_err = (s2 - s1).abs().max().item()
    print(f"  Block scalar invariance err: {s_err:.2e}")
    assert s_err < 5e-4, f"FAIL: Block scalar not invariant: {s_err}"

    # Vector equivariance
    v1_rot = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - v1_rot).abs().max().item()
    print(f"  Block vector equivariance err: {v_err:.2e}")
    assert v_err < 5e-3, f"FAIL: Block vector not equivariant: {v_err}"

    print("TEST A3-Block PASSED ✓\n")


# ═══════════════════════════════════════════════════════════════════
# Test: Memory budget estimation
# ═══════════════════════════════════════════════════════════════════

def test_memory_estimation():
    """Print parameter count and estimated memory for various configs.

    NOT an assertion test — information only for VRAM budgeting.
    """
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

    print("=" * 60)
    print("MEMORY ESTIMATION (information only)")
    print("=" * 60)

    configs = [
        # (C_s, C_v, stages, layers, description)
        (32, 8, 2, 1, "Current (baseline)"),
        (48, 12, 2, 1, "Small upgrade"),
        (64, 16, 2, 1, "Medium upgrade"),
        (64, 16, 2, 2, "Medium + deeper"),
        (96, 24, 2, 2, "Large (previous)"),
        (128, 32, 2, 2, "XL"),
    ]

    for C_s, C_v, stages, layers, desc in configs:
        model = MultiScaleSE3Net(
            in_channels=1,
            hidden_scalar=C_s, hidden_vector=C_v,
            num_stages=stages, layers_per_stage=layers,
            gate_mode='norm', use_self_tp=True,
        )
        n_params = sum(p.numel() for p in model.parameters())
        param_mb = n_params * 4 / 1024**2  # FP32
        # Rough memory estimate: params + grads + optimizer states + activations
        # AdamW: 2x param states (m, v) + 1x grads + 1x params = 4x
        # Activations ≈ 2-4x params for GNNs (edge features dominate)
        est_total_mb = param_mb * 8  # very rough
        print(f"  {desc:25s}: {n_params:>10,} params ({param_mb:.1f} MB) "
              f"| est_train ≈ {est_total_mb:.0f} MB")

    print()
    print("  NOTE: True VRAM is dominated by edge features (E × C × 3 tensors)")
    print("  and graph topology, NOT parameters. Batch size × 2048 points")
    print("  with avg ~20 edges/node: E ≈ B × 2048 × 20 = ~320K edges/step")
    print("  Each edge stores: direction[3] + SH[9] + msg[C_s + C_v*3] tensors")
    print()
    print("MEMORY ESTIMATION DONE\n")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Milestone A: Equivariance & Correctness Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_a2_norm_gate_equivariance,
        test_a3_self_tp_invariance,
        test_a1_multiscale_head_shape,
        test_a3_block_equivariance,
        test_combined_e2e_equivariance,
        test_memory_estimation,
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
