#!/usr/bin/env python3
"""Tests for norm-preserving pool rescaling.

Validates that EquivariantPool's norm-preserving rescaling:
  1. Preserves l=1 (vector) and l=2 (type-2) norms through pooling
  2. Maintains strict SO(3) equivariance after rescaling
  3. Produces finite gradients (no NaN/Inf from division)
  4. Per-channel norm ratios bounded (no cancellation collapse)

NOT a pytest file — run directly: python tests/test_norm_preserving_pool.py
"""

import torch
import sys

from geoembodied.nn.modules.equivariant_pool import EquivariantPool
from geoembodied.functional.so3_ops import so3_exp
from geoembodied.functional.quaternion_ops import quaternion_to_matrix


def test_1_norm_preservation():
    """Verify post-pool norms ≈ weighted avg of pre-pool norms.

    With norm-preserving rescaling, output norms should be close to 1.0x
    of input norms (not 0.3-0.5x from cancellation).
    """
    print("=" * 60)
    print("TEST 1: Norm Preservation (v + t2)")
    print("=" * 60)

    torch.manual_seed(42)
    C_s, C_v, C_t2 = 16, 8, 4
    N = 200

    pool = EquivariantPool(C_s, C_v, type2_channels=C_t2,
                           ratio=0.25, k_neighbors=16)
    pool.eval()

    pos = torch.randn(N, 3)
    scalars = torch.randn(N, C_s)
    vectors = torch.randn(N, C_v, 3)
    type2 = torch.randn(N, C_t2, 5)
    ptr = torch.tensor([0, N], dtype=torch.int64)

    with torch.no_grad():
        seed_pos, s_out, v_out, t2_out, ptr_out, fps_idx = pool(
            pos, scalars, vectors, ptr, type2=type2)

    v_in_norm = vectors.norm(dim=-1).mean().item()
    t2_in_norm = type2.norm(dim=-1).mean().item()
    v_out_norm = v_out.norm(dim=-1).mean().item()
    t2_out_norm = t2_out.norm(dim=-1).mean().item()

    v_ratio = v_out_norm / max(v_in_norm, 1e-8)
    t2_ratio = t2_out_norm / max(t2_in_norm, 1e-8)

    print(f"  Vector: in={v_in_norm:.4f} out={v_out_norm:.4f} "
          f"ratio={v_ratio:.3f}")
    print(f"  Type-2: in={t2_in_norm:.4f} out={t2_out_norm:.4f} "
          f"ratio={t2_ratio:.3f}")

    # With norm-preserving: ratio ≈ 1.0
    # Without: ratio ≈ 0.3-0.5
    assert v_ratio > 0.7, (
        f"FAIL: Vector norm collapsed! ratio={v_ratio:.3f}")
    assert t2_ratio > 0.7, (
        f"FAIL: Type-2 norm collapsed! ratio={t2_ratio:.3f}")
    print(f"  ✓ Vector norm preserved (ratio={v_ratio:.3f})")
    print(f"  ✓ Type-2 norm preserved (ratio={t2_ratio:.3f})")

    print("TEST 1 PASSED ✓\n")


def test_2_equivariance_with_rescaling():
    """Strict SO(3) equivariance after norm-preserving rescaling.

    The rescaling multiplies by scale = target_norm/actual_norm.
    Both target and actual are SO(3)-invariant (norms),
    so scale is invariant → rescaling preserves equivariance.

    Uses the SAME tolerance as test_pool_interp.py (1e-4).
    """
    print("=" * 60)
    print("TEST 2: SO(3) Equivariance with Norm-Preserving")
    print("=" * 60)

    C_s, C_v = 8, 4
    pool = EquivariantPool(C_s, C_v, ratio=0.25, k_neighbors=8)
    pool.eval()

    torch.manual_seed(77)
    N = 100
    pos = torch.randn(N, 3)
    scalars = torch.randn(N, C_s)
    vectors = torch.randn(N, C_v, 3)
    ptr = torch.tensor([0, N], dtype=torch.int64)

    omega = torch.randn(3) * 0.5
    q = so3_exp(omega.unsqueeze(0))
    R = quaternion_to_matrix(q).squeeze(0)  # [3, 3]

    pos_rot = pos @ R.t()
    vectors_rot = torch.einsum('nvc,dc->nvd', vectors, R)

    torch.manual_seed(0)
    with torch.no_grad():
        sp1, s1, v1, pp1, idx1 = pool(pos, scalars, vectors, ptr)

    torch.manual_seed(0)
    with torch.no_grad():
        sp2, s2, v2, pp2, idx2 = pool(pos_rot, scalars, vectors_rot, ptr)

    # Position equivariance
    expected_pos = sp1 @ R.t()
    pos_err = (sp2 - expected_pos).abs().max().item()
    assert pos_err < 1e-4, f"FAIL: Position error = {pos_err}"
    print(f"  ✓ Position equivariance: err={pos_err:.6f}")

    # Scalar invariance
    s_err = (s2 - s1).abs().max().item()
    assert s_err < 1e-4, f"FAIL: Scalar error = {s_err}"
    print(f"  ✓ Scalar invariance: err={s_err:.6f}")

    # Vector equivariance — SAME threshold as test_pool_interp.py (1e-4)
    expected_v = torch.einsum('nvc,dc->nvd', v1, R)
    v_err = (v2 - expected_v).abs().max().item()
    assert v_err < 1e-4, f"FAIL: Vector equivariance error = {v_err}"
    print(f"  ✓ Vector equivariance: err={v_err:.6f}")

    print("TEST 2 PASSED ✓\n")


def test_3_gradient_stability():
    """No NaN/Inf gradients through norm-preserving rescaling.

    The rescaling involves division by actual_norm.
    Uses safe norm (x²→sum→clamp→sqrt) and torch.where guard.
    Verify backward pass is clean with adversarial inputs.
    """
    print("=" * 60)
    print("TEST 3: Gradient Stability")
    print("=" * 60)

    C_s, C_v, C_t2 = 8, 4, 4
    pool = EquivariantPool(C_s, C_v, type2_channels=C_t2,
                           ratio=0.25, k_neighbors=8)

    cases = {
        "normal": (1.0, 1.0, 1.0),
        "tiny_features": (1.0, 1e-6, 1e-6),
        "huge_features": (1.0, 1e3, 1e3),
    }

    for case_name, (pos_scale, feat_scale, t2_scale) in cases.items():
        pool.train()
        pool.zero_grad()
        torch.manual_seed(42)

        pos = torch.randn(100, 3) * pos_scale
        pos.requires_grad_(True)
        s = torch.randn(100, C_s) * feat_scale
        s.requires_grad_(True)
        v = torch.randn(100, C_v, 3) * feat_scale
        v.requires_grad_(True)
        t2 = torch.randn(100, C_t2, 5) * t2_scale
        t2.requires_grad_(True)
        ptr = torch.tensor([0, 100], dtype=torch.int64)

        sp, so, vo, t2o, pp, _ = pool(pos, s, v, ptr, type2=t2)
        loss = so.mean() + vo.mean() + t2o.mean()
        loss.backward()

        for name, tensor in [("pos", pos), ("s", s), ("v", v), ("t2", t2)]:
            g = tensor.grad
            assert g is not None, f"FAIL: {name}.grad is None in '{case_name}'"
            assert not torch.isnan(g).any(), \
                f"FAIL: {name} has NaN grad in '{case_name}'"
            assert not torch.isinf(g).any(), \
                f"FAIL: {name} has Inf grad in '{case_name}'"

        print(f"  ✓ {case_name}: grad norms — "
              f"pos={pos.grad.norm():.4f} "
              f"s={s.grad.norm():.4f} "
              f"v={v.grad.norm():.4f} "
              f"t2={t2.grad.norm():.4f}")

    # Special case: all-zero type-2 input
    pool.train()
    pool.zero_grad()
    torch.manual_seed(42)
    pos = torch.randn(100, 3, requires_grad=True)
    s = torch.randn(100, C_s, requires_grad=True)
    v = torch.randn(100, C_v, 3, requires_grad=True)
    t2 = torch.zeros(100, C_t2, 5, requires_grad=True)
    ptr = torch.tensor([0, 100], dtype=torch.int64)

    sp, so, vo, t2o, pp, _ = pool(pos, s, v, ptr, type2=t2)
    loss = so.mean() + vo.mean() + t2o.mean()
    loss.backward()

    assert not torch.isnan(t2.grad).any(), "FAIL: zero t2 input → NaN grad"
    print(f"  ✓ zero_t2: t2.grad norm={t2.grad.norm():.6f} (finite)")

    print("TEST 3 PASSED ✓\n")


def test_4_cancellation_reduction():
    """Per-channel norm ratio should be bounded (no collapse).

    With norm-preserving rescaling, per-channel norm ratio ≈ 1.0.
    Without rescaling, ratio ≈ 0.3-0.5.
    """
    print("=" * 60)
    print("TEST 4: Cancellation Reduction")
    print("=" * 60)

    torch.manual_seed(123)
    C_s, C_v, C_t2 = 16, 8, 8
    N = 512

    pool = EquivariantPool(C_s, C_v, type2_channels=C_t2,
                           ratio=0.25, k_neighbors=16)
    pool.eval()

    pos = torch.randn(N, 3)
    scalars = torch.randn(N, C_s)
    vectors = torch.randn(N, C_v, 3)
    type2 = torch.randn(N, C_t2, 5)
    ptr = torch.tensor([0, N], dtype=torch.int64)

    with torch.no_grad():
        sp, s_out, v_out, t2_out, pp, _ = pool(
            pos, scalars, vectors, ptr, type2=type2)

    v_in_ch = vectors.norm(dim=-1).mean(0)
    v_out_ch = v_out.norm(dim=-1).mean(0)
    t2_in_ch = type2.norm(dim=-1).mean(0)
    t2_out_ch = t2_out.norm(dim=-1).mean(0)

    v_ratios = v_out_ch / v_in_ch.clamp(min=1e-8)
    t2_ratios = t2_out_ch / t2_in_ch.clamp(min=1e-8)

    print(f"  Vector per-channel ratios: "
          f"min={v_ratios.min():.3f} "
          f"mean={v_ratios.mean():.3f} "
          f"max={v_ratios.max():.3f}")
    print(f"  Type-2 per-channel ratios: "
          f"min={t2_ratios.min():.3f} "
          f"mean={t2_ratios.mean():.3f} "
          f"max={t2_ratios.max():.3f}")

    v_min = v_ratios.min().item()
    t2_min = t2_ratios.min().item()
    assert v_min > 0.7, f"FAIL: Vector channel ratio too low: {v_min:.3f}"
    assert t2_min > 0.7, f"FAIL: Type-2 channel ratio too low: {t2_min:.3f}"

    print(f"  ✓ Vector min ratio = {v_min:.3f} > 0.7")
    print(f"  ✓ Type-2 min ratio = {t2_min:.3f} > 0.7")

    print("TEST 4 PASSED ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Norm-Preserving Pool Tests")
    print("  EquivariantPool with l=1 and l=2 rescaling")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_1_norm_preservation,
        test_2_equivariance_with_rescaling,
        test_3_gradient_stability,
        test_4_cancellation_reduction,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(failed)
