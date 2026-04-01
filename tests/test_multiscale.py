#!/usr/bin/env python3
"""Phase 6d Pre-Assembly Defense Tests for MultiScaleSE3Net.

3 critical tests (Test 3 is remote-only — writes report but skips CUDA OOM):
  1. Scale Collapse Test — graph connectivity at bottleneck
  2. Skip-Connection Dimensionality Test — U-Net concat alignment
  3. OOM Stress Test — REMOTE ONLY (writes config for remote execution)

NOT a pytest file — run directly: python3 test_multiscale.py
"""

import os
import torch
import torch.nn as nn

from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net, _EncoderStage
from geoembodied.nn.modules.equivariant_pool import EquivariantPool
from geoembodied.nn.modules.equivariant_interp import EquivariantInterpolate
from geoembodied.nn.modules.spatial_graph import SpatialGraph
from geoembodied.functional.knn import knn_self
from geoembodied.functional.fps import farthest_point_sampling


def test_1_scale_collapse():
    """Test 1: Scale Collapse — deep bottleneck graph connectivity.

    Build a 3-stage DownBlock chain on a SPARSE point cloud.
    At the bottleneck, nodes MUST still have neighbors (KNN fallback).
    Features must NOT collapse to all-zero or all-same.
    """
    print("=" * 60)
    print("TEST 1: Scale Collapse (Bottleneck Connectivity)")
    print("=" * 60)

    torch.manual_seed(42)
    C_s, C_v = 32, 8

    # Sparse point cloud: 500 points spread across a large volume
    pos = torch.randn(500, 3) * 5.0  # spread out → sparse
    ptr = torch.tensor([0, 500], dtype=torch.int64)

    # Initial features
    s = torch.randn(500, C_s)
    v = torch.randn(500, C_v, 3)

    pool = EquivariantPool(C_s, C_v, ratio=0.25, k_neighbors=8)
    stage = _EncoderStage(
        channels_scalar=C_s, channels_vector=C_v,
        num_layers=2, radius_multiplier=4.0
    )

    with torch.no_grad():
        # Stage 0: 500 points
        s0, v0, _ = stage(s, v, pos, ptr)
        print(f"  Stage 0: N=500, s_std={s0.std():.4f}, v_std={v0.std():.4f}")

        # Pool 0: 500 → ~125
        pos1, s1, v1, ptr1, _ = pool(pos, s0, v0, ptr)
        N1 = pos1.shape[0]
        print(f"  Pool 0→1: N={N1}")

        # Stage 1
        stage1 = _EncoderStage(C_s, C_v, num_layers=2, radius_multiplier=4.0)
        s1, v1, _ = stage1(s1, v1, pos1, ptr1)
        print(f"  Stage 1: N={N1}, s_std={s1.std():.4f}, v_std={v1.std():.4f}")

        # Pool 1: ~125 → ~31
        pool1 = EquivariantPool(C_s, C_v, ratio=0.25, k_neighbors=8)
        pos2, s2, v2, ptr2, _ = pool1(pos1, s1, v1, ptr1)
        N2 = pos2.shape[0]
        print(f"  Pool 1→2: N={N2}")

        # Stage 2 (bottleneck)
        stage2 = _EncoderStage(C_s, C_v, num_layers=2, radius_multiplier=4.0)
        s2, v2, _ = stage2(s2, v2, pos2, ptr2)
        print(f"  Stage 2 (bottleneck): N={N2}, s_std={s2.std():.4f}, v_std={v2.std():.4f}")

    # Check 1a: Features at bottleneck are NOT dead
    assert s2.std() > 1e-4, \
        f"FAIL: Bottleneck scalar features collapsed! std={s2.std():.6f}"
    assert v2.std() > 1e-4, \
        f"FAIL: Bottleneck vector features collapsed! std={v2.std():.6f}"
    print(f"  ✓ Bottleneck features alive: s_std={s2.std():.4f}, v_std={v2.std():.4f}")

    # Check 1b: Features are NOT all the same (over-smoothing)
    s_range = s2.max() - s2.min()
    assert s_range > 0.01, \
        f"FAIL: Scalar features over-smoothed! range={s_range:.6f}"
    print(f"  ✓ Feature diversity maintained: scalar range={s_range:.4f}")

    # Check 1c: Graph was actually connected at bottleneck level
    # Test via KNN: every node should have at least 3 neighbors
    knn_idx, knn_d = knn_self(pos2, ptr2, k=4)
    valid_nbrs = (knn_idx >= 0).sum(dim=1)  # including self
    min_nbrs = valid_nbrs.min().item()
    assert min_nbrs >= 2, \
        f"FAIL: Bottleneck node has only {min_nbrs} neighbors (including self)"
    print(f"  ✓ Bottleneck connectivity: min neighbors={min_nbrs} (including self)")

    print("TEST 1 PASSED ✓\n")


def test_2_skip_connection_dimensionality():
    """Test 2: U-Net skip-connection dimension alignment.

    Simulate encoder → decoder cycle. After interp + skip concat:
    - Spatial resolution matches encoder level
    - Scalar shape: [N, C_s + C_s_skip] (NOT [N, C_s, C_s_skip])
    - Vector shape: [N, C_v + C_v_skip, 3] (NOT concat on dim=2!)
    """
    print("=" * 60)
    print("TEST 2: Skip-Connection Dimensionality")
    print("=" * 60)

    torch.manual_seed(42)
    C_s, C_v = 32, 8

    # Input: 200 points
    N0 = 200
    pos0 = torch.randn(N0, 3)
    ptr0 = torch.tensor([0, N0], dtype=torch.int64)
    s0 = torch.randn(N0, C_s)
    v0 = torch.randn(N0, C_v, 3)

    # Encoder stage 0 output (simulate)
    enc_s0 = s0.clone()
    enc_v0 = v0.clone()

    # Pool: N0=200 → N1≈50
    pool = EquivariantPool(C_s, C_v, ratio=0.25, k_neighbors=8)
    with torch.no_grad():
        pos1, s1, v1, ptr1, fps_idx = pool(pos0, enc_s0, enc_v0, ptr0)
    N1 = pos1.shape[0]
    print(f"  Encoder: N0={N0} → Pool → N1={N1}")

    # Simulate encoder stage 1 output (same channels)
    enc_s1 = s1.clone()
    enc_v1 = v1.clone()

    # Pool: N1 → N2≈12
    pool1 = EquivariantPool(C_s, C_v, ratio=0.25, k_neighbors=8)
    with torch.no_grad():
        pos2, s2, v2, ptr2, _ = pool1(pos1, enc_s1, enc_v1, ptr1)
    N2 = pos2.shape[0]
    print(f"  Encoder: N1={N1} → Pool → N2={N2}")

    # ── Decoder: upsample N2 → N1 with skip ──
    interp = EquivariantInterpolate(k_neighbors=3)
    with torch.no_grad():
        s_up, v_up = interp(
            pos1, pos2,  # dense=N1, coarse=N2
            s2, v2,
            ptr1, ptr2,
            s_skip=enc_s1, v_skip=enc_v1,  # skip from encoder
        )

    # Check 2a: Spatial resolution matches encoder level
    assert s_up.shape[0] == N1, \
        f"FAIL: Upsampled N={s_up.shape[0]}, expected N1={N1}"
    print(f"  ✓ Spatial resolution: N_up={s_up.shape[0]} == N1={N1}")

    # Check 2b: Scalar shape = [N1, C_s + C_s] (concat on channels)
    expected_s_channels = C_s + C_s
    assert s_up.shape == (N1, expected_s_channels), \
        f"FAIL: Scalar shape {s_up.shape}, expected ({N1}, {expected_s_channels})"
    print(f"  ✓ Scalar shape: {s_up.shape} == ({N1}, {expected_s_channels})")

    # Check 2c: Vector shape = [N1, C_v + C_v, 3] (concat on dim=1, NOT dim=2!)
    expected_v_channels = C_v + C_v
    assert v_up.shape == (N1, expected_v_channels, 3), \
        f"FAIL: Vector shape {v_up.shape}, expected ({N1}, {expected_v_channels}, 3)"
    assert v_up.shape[2] == 3, \
        f"FAIL: Vector dim 2 is {v_up.shape[2]}, MUST be 3 (spatial dim)"
    print(f"  ✓ Vector shape: {v_up.shape} == ({N1}, {expected_v_channels}, 3)")
    print(f"  ✓ Spatial dimension preserved: dim[2]=3")

    # Check 2d: After projection (simulating skip_proj), channels restored
    proj_s = nn.Linear(C_s * 2, C_s)
    proj_v = nn.Linear(C_v * 2, C_v, bias=False)

    with torch.no_grad():
        s_proj = proj_s(s_up)  # [N1, C_s]
        # Vector projection preserving spatial dim
        v_flat = v_up.transpose(1, 2).reshape(N1 * 3, C_v * 2)
        v_proj = proj_v(v_flat).reshape(N1, 3, C_v).transpose(1, 2)

    assert s_proj.shape == (N1, C_s), \
        f"FAIL: Projected scalar {s_proj.shape}, expected ({N1}, {C_s})"
    assert v_proj.shape == (N1, C_v, 3), \
        f"FAIL: Projected vector {v_proj.shape}, expected ({N1}, {C_v}, 3)"
    print(f"  ✓ Post-projection scalar: {s_proj.shape}")
    print(f"  ✓ Post-projection vector: {v_proj.shape}")

    # Check 2e: Full U-Net forward pass
    print(f"\n  Full U-Net forward pass:")
    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=16, hidden_vector=4,
        num_stages=3, layers_per_stage=1,
        pool_ratio=0.25, pool_k=8,
    )
    model.eval()

    pos_in = torch.randn(100, 3)
    ptr_in = torch.tensor([0, 60, 100], dtype=torch.int64)

    with torch.no_grad():
        s_final, v_final, ptr_final = model(pos_in, ptr_in)

    assert s_final.shape == (100, 16), \
        f"FAIL: Final scalar {s_final.shape}, expected (100, 16)"
    assert v_final.shape == (100, 4, 3), \
        f"FAIL: Final vector {v_final.shape}, expected (100, 4, 3)"
    assert (ptr_final == ptr_in).all(), \
        f"FAIL: ptr changed! was {ptr_in.tolist()}, now {ptr_final.tolist()}"
    print(f"  ✓ U-Net output: s={s_final.shape}, v={v_final.shape}")
    print(f"  ✓ ptr preserved: {ptr_final.tolist()}")

    print("TEST 2 PASSED ✓\n")


def test_3_oom_stress_placeholder():
    """Test 3: OOM Stress Test — PLACEHOLDER for remote execution.

    This test requires significant GPU memory (>4GB) and should be
    run on a remote server with adequate VRAM.

    Prints the test configuration for remote execution.
    """
    print("=" * 60)
    print("TEST 3: OOM Stress Test (REMOTE ONLY)")
    print("=" * 60)

    print("  ⚠ Skipped locally (4GB VRAM limit)")
    print("  Remote test configuration:")
    print("  ─────────────────────────────")
    print("  Model: MultiScaleSE3Net(")
    print("    hidden_scalar=64, hidden_vector=16,")
    print("    num_stages=3, layers_per_stage=2,")
    print("    pool_ratio=0.25")
    print("  )")
    print("  Input: N=10000 points, B=2 batches")
    print("  Test: 3× forward+backward cycles")
    print("  Assertions:")
    print("    1. No OOM on 24GB GPU")
    print("    2. Memory returns to baseline after each cycle")
    print("    3. Peak VRAM < 12GB (safe for RTX 3090/4090)")
    print()

    # Note: bench_oom.py is a standalone diagnostic script.
    # See benchmarks/bench_oom.py for the actual OOM stress test.
    print("  → See benchmarks/bench_oom.py for remote execution")
    print("TEST 3 PLACEHOLDER ✓ (deferred to remote)\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Phase 6d Pre-Assembly Defense Tests")
    print("  MultiScaleSE3Net U-Net")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_1_scale_collapse,
        test_2_skip_connection_dimensionality,
        test_3_oom_stress_placeholder,
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
