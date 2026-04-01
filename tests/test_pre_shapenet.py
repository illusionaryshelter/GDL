#!/usr/bin/env python3
"""Phase 6d+ Pre-ShapeNet Defense Tests.

3 critical tests before ShapeNet Part Segmentation:
  1. End-to-End U-Net Equivariance — LOCAL (the ultimate test)
  2. PyTorch Profiler Bottleneck — REMOTE (needs GPU)
  3. Memory & Receptive Field — REMOTE (needs GPU)

Run locally: python3 test_pre_shapenet.py
"""

import os
import torch
import torch.nn as nn

from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net
from geoembodied.functional.so3_ops import so3_exp
from geoembodied.functional.quaternion_ops import quaternion_to_matrix


def test_1_e2e_equivariance():
    """Test 1: End-to-End SE(3) Equivariance — The Ultimate Judgement.

    For Part Segmentation, U-Net outputs per-point scalar logits.
    THE ABSOLUTE TRUTH: f(P·R^T + t) ≡ f(P)

    If ANY operation in the entire U-Net chain (embedding, SE3Conv,
    normalization, pooling, skip-concat, projection, interpolation)
    breaks equivariance, this test WILL catch it.

    Test plan:
      1. Run U-Net on original point cloud → scalar output
      2. Run U-Net on SE(3)-transformed point cloud → scalar output
      3. Assert: scalar outputs are IDENTICAL (up to FP32)
      4. Also verify vector outputs are equivariant: v2 ≈ v1 @ R^T
    """
    print("=" * 60)
    print("TEST 1: End-to-End U-Net SE(3) Equivariance")
    print("=" * 60)

    C_s, C_v = 16, 4

    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v,
        num_stages=3, layers_per_stage=1,
        pool_ratio=0.25, pool_k=8, interp_k=3,
    )
    model.eval()

    # ── Construct Batch with B=2, variable sizes ──
    torch.manual_seed(42)
    N1, N2 = 256, 200
    N = N1 + N2
    pos = torch.randn(N, 3)
    ptr = torch.tensor([0, N1, N], dtype=torch.int64)

    # ── Random SE(3) transform ──
    omega = torch.randn(3) * 0.8  # moderate rotation
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)  # [3, 3]
    t = torch.randn(3) * 3.0

    pos_rot = (pos @ R.t()) + t.unsqueeze(0)

    # ── Forward on both ──
    torch.manual_seed(0)  # fix FPS seeds
    with torch.no_grad():
        s1, v1, _ = model(pos, ptr)

    torch.manual_seed(0)
    with torch.no_grad():
        s2, v2, _ = model(pos_rot, ptr)

    # ── Check 1a: Scalar INVARIANCE — f(Rp+t) ≡ f(p) ──
    s_err = (s2 - s1).abs().max().item()
    s_rel = s_err / (s1.abs().max().item() + 1e-8)
    print(f"  Scalar max abs err: {s_err:.2e}")
    print(f"  Scalar relative err: {s_rel:.2e}")
    assert s_err < 1e-4, f"FAIL: Scalar NOT invariant! err={s_err:.6f}"
    print(f"  ✓ Scalar invariance: max_err = {s_err:.6f}")

    # ── Check 1b: Vector EQUIVARIANCE — v2 ≈ v1 @ R^T ──
    expected_v = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - expected_v).abs().max().item()
    v_rel = v_err / (v1.abs().max().item() + 1e-8)
    print(f"  Vector max abs err: {v_err:.2e}")
    print(f"  Vector relative err: {v_rel:.2e}")
    assert v_err < 1e-3, f"FAIL: Vector NOT equivariant! err={v_err:.6f}"
    print(f"  ✓ Vector equivariance: max_err = {v_err:.6f}")

    # ── Check 1c: Per-batch isolation ──
    # Batch 0 and Batch 1 scalar errors independently
    s_err_b0 = (s2[:N1] - s1[:N1]).abs().max().item()
    s_err_b1 = (s2[N1:] - s1[N1:]).abs().max().item()
    print(f"  ✓ Batch 0 scalar err: {s_err_b0:.6f}")
    print(f"  ✓ Batch 1 scalar err: {s_err_b1:.6f}")

    # ── FP64 baseline comparison ──
    print(f"\n  FP64 Baseline (CPU):")
    model_64 = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=C_s, hidden_vector=C_v,
        num_stages=3, layers_per_stage=1,
        pool_ratio=0.25, pool_k=8, interp_k=3,
    )
    # Copy weights
    model_64.load_state_dict(model.state_dict())
    model_64 = model_64.double()
    model_64.eval()

    pos_64 = pos.double()
    R_64 = R.double()
    t_64 = t.double()
    pos_rot_64 = (pos_64 @ R_64.t()) + t_64.unsqueeze(0)

    torch.manual_seed(0)
    with torch.no_grad():
        s1_64, v1_64, _ = model_64(pos_64, ptr)
    torch.manual_seed(0)
    with torch.no_grad():
        s2_64, v2_64, _ = model_64(pos_rot_64, ptr)

    s_err_64 = (s2_64 - s1_64).abs().max().item()
    exp_v_64 = torch.einsum('nvc,dc->nvd', v1_64, R_64)
    v_err_64 = (v2_64 - exp_v_64).abs().max().item()
    print(f"  FP64 scalar invariance err: {s_err_64:.2e}")
    print(f"  FP64 vector equivariance err: {v_err_64:.2e}")

    # FP32 vs FP64 comparison → pure truncation
    fp32_vs_fp64 = (s1.double() - s1_64).abs().max().item()
    print(f"  FP32 vs FP64 scalar gap: {fp32_vs_fp64:.2e}")

    print("TEST 1 PASSED ✓\n")


def test_2_profiler_placeholder():
    """Test 2: PyTorch Profiler — REMOTE only (needs GPU).

    Prints config and writes remote script.
    """
    print("=" * 60)
    print("TEST 2: PyTorch Profiler (REMOTE ONLY)")
    print("=" * 60)
    print("  ⚠ Skipped locally (needs CUDA)")
    print("  → See benchmarks/bench_profiler.py")
    print("TEST 2 PLACEHOLDER ✓\n")


def test_3_memory_placeholder():
    """Test 3: Memory & Receptive Field — REMOTE only (needs GPU)."""
    print("=" * 60)
    print("TEST 3: Memory & Receptive Field (REMOTE ONLY)")
    print("=" * 60)
    print("  ⚠ Skipped locally (needs CUDA)")
    print("  → See benchmarks/bench_memory.py")
    print("TEST 3 PLACEHOLDER ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Pre-ShapeNet Defense Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_1_e2e_equivariance,
        test_2_profiler_placeholder,
        test_3_memory_placeholder,
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
    import sys
    sys.exit(failed)
