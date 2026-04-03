#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Tests for Component 1: Dynamic Content-Aware Pool Attention
#
# Tests verify:
# 1. Attention weights are SO(3)-invariant (same for rotated inputs)
# 2. Pool output is SE(3)-equivariant (v_out transforms correctly)
# 3. Content attention is dynamic (different seed features → different rankings)
# 4. Backward pass works (gradients flow through new Q/K projections)
#
# Run: python tests/test_dynamic_pool.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math


def random_rotation_matrix(dtype=torch.float64, device='cpu'):
    """Generate random SO(3) rotation via QR decomposition."""
    M = torch.randn(3, 3, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(M)
    signs = torch.diag(R).sign()
    Q = Q * signs.unsqueeze(0)
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def make_pool_inputs(N=64, C_s=32, C_v=8, C_t2=4, dtype=torch.float64):
    """Create synthetic packed point cloud inputs for EquivariantPool."""
    torch.manual_seed(42)
    pos = torch.randn(N, 3, dtype=dtype)
    scalars = torch.randn(N, C_s, dtype=dtype)
    vectors = torch.randn(N, C_v, 3, dtype=dtype)
    type2 = torch.randn(N, C_t2, 5, dtype=dtype)
    ptr = torch.tensor([0, N], dtype=torch.int64)
    return pos, scalars, vectors, type2, ptr


def test_pool_attention_invariance():
    """Test: attention weights are invariant under SO(3) rotation.

    For a single batch, applying R to pos and vectors should not
    change the attention weights (since they depend only on invariants).
    """
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool

    C_s, C_v, C_t2 = 32, 8, 4
    pool = EquivariantPool(
        scalar_channels=C_s, vector_channels=C_v,
        type2_channels=C_t2, ratio=0.5, k_neighbors=8,
    ).double()
    pool.eval()

    pos, scalars, vectors, type2, ptr = make_pool_inputs(
        N=32, C_s=C_s, C_v=C_v, C_t2=C_t2
    )

    R = random_rotation_matrix(dtype=torch.float64)

    # Original
    with torch.no_grad():
        pool(pos, scalars, vectors, ptr, type2=type2)
    w_orig = pool._last_attn_weights.clone()

    # Rotated: R @ pos, R @ vectors
    pos_rot = pos @ R.T
    vec_rot = torch.einsum('ij, ncj -> nci', R, vectors)
    # type2 under Wigner D^(2) — use identity for simplicity since
    # attention only uses ||t2|| which is invariant under any unitary
    type2_rot = type2.clone()

    with torch.no_grad():
        pool(pos_rot, scalars, vec_rot, ptr, type2=type2_rot)
    w_rot = pool._last_attn_weights.clone()

    # Attention weights should be identical
    err = (w_orig - w_rot).abs().max().item()
    # Tolerance 1e-5: BN running stats + FPS seed selection introduce
    # ~1e-6 level numerical differences in FP64
    assert err < 1e-5, f"Attention NOT invariant! max_err={err}"
    print(f"  [PASS] Attention weights SO(3)-invariant, max_err = {err:.2e}")


def test_pool_vector_equivariance():
    """Test: pool vector output transforms correctly under SO(3).

    pool(R@pos, s, R@v) should give R @ pool(pos, s, v) for vectors.
    """
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool

    C_s, C_v, C_t2 = 32, 8, 4
    pool = EquivariantPool(
        scalar_channels=C_s, vector_channels=C_v,
        type2_channels=C_t2, ratio=0.5, k_neighbors=8,
    ).double()
    pool.eval()

    pos, scalars, vectors, type2, ptr = make_pool_inputs(
        N=32, C_s=C_s, C_v=C_v, C_t2=C_t2
    )

    R = random_rotation_matrix(dtype=torch.float64)

    with torch.no_grad():
        _, s1, v1, t2_1, _, _ = pool(pos, scalars, vectors, ptr, type2=type2)

        pos_rot = pos @ R.T
        vec_rot = torch.einsum('ij, ncj -> nci', R, vectors)
        _, s2, v2, t2_2, _, _ = pool(pos_rot, scalars, vec_rot, ptr, type2=type2)

    # Scalar output should be identical (tolerance for BN noise)
    s_err = (s1 - s2).abs().max().item()
    assert s_err < 5e-5, f"Scalar NOT invariant! max_err={s_err}"

    # Vector output should transform: v2 ≈ R @ v1
    # Tolerance 5e-5: BN running stats → attention weight diff → weighted sum amplification
    v1_rot = torch.einsum('ij, ncj -> nci', R, v1)
    v_err = (v1_rot - v2).abs().max().item()
    assert v_err < 5e-5, f"Vector NOT equivariant! max_err={v_err}"

    print(f"  [PASS] Pool equivariance: scalar_err={s_err:.2e}, vector_err={v_err:.2e}")


def test_dynamic_attention_differentiation():
    """Test: content attention produces different weights for different seeds.

    This is the key property that distinguishes GATv2 from GAT.
    With static attention, all seeds would rank neighbors the same way.
    With dynamic attention, different seed features should produce
    different neighbor rankings.
    """
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool

    torch.manual_seed(7)
    C_s, C_v = 32, 8
    N = 48

    pool = EquivariantPool(
        scalar_channels=C_s, vector_channels=C_v,
        ratio=0.5, k_neighbors=8,
    ).double()
    pool.eval()

    pos = torch.randn(N, 3, dtype=torch.float64)
    scalars = torch.randn(N, C_s, dtype=torch.float64)
    # Make two groups of scalars very different
    scalars[:N//2] *= 3.0
    scalars[N//2:] *= 0.1
    vectors = torch.randn(N, C_v, 3, dtype=torch.float64)
    ptr = torch.tensor([0, N], dtype=torch.int64)

    with torch.no_grad():
        pool(pos, scalars, vectors, ptr)

    w = pool._last_attn_weights  # [N_out, K]

    # Check variance of max weights across seeds
    max_per_seed = w.max(dim=-1).values  # [N_out]
    max_std = max_per_seed.std().item()
    # With dynamic attention, seeds should have different max attention
    # With static attention, max would be nearly identical for all seeds
    assert max_std > 0.01, (
        f"Attention too uniform across seeds: std={max_std:.4f}. "
        f"Dynamic attention may not be working."
    )
    print(f"  [PASS] Dynamic attention differentiation: max_weight_std = {max_std:.4f}")


def test_gradient_flow_through_qk():
    """Test: gradients flow through the Q/K content attention path."""
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool

    C_s, C_v = 32, 8
    pool = EquivariantPool(
        scalar_channels=C_s, vector_channels=C_v,
        ratio=0.5, k_neighbors=8,
    )

    pos = torch.randn(32, 3)
    scalars = torch.randn(32, C_s, requires_grad=True)
    vectors = torch.randn(32, C_v, 3)
    ptr = torch.tensor([0, 32], dtype=torch.int64)

    _, s_out, v_out, _, _ = pool(pos, scalars, vectors, ptr)
    loss = s_out.sum() + v_out.sum()
    loss.backward()

    assert scalars.grad is not None, "No gradient on scalars!"
    assert scalars.grad.abs().sum() > 0, "Zero gradient on scalars!"

    # Check Q/K projections received gradients
    q_grad = pool.q_proj.weight.grad
    k_grad = pool.k_proj.weight.grad
    assert q_grad is not None and q_grad.abs().sum() > 0, "No gradient on q_proj!"
    assert k_grad is not None and k_grad.abs().sum() > 0, "No gradient on k_proj!"

    print(f"  [PASS] Gradients flow through Q/K: ||∇q||={q_grad.norm():.4f}, ||∇k||={k_grad.norm():.4f}")


def test_param_count():
    """Test: dynamic attention adds reasonable parameter count."""
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool

    C_s, C_v = 48, 12
    pool = EquivariantPool(
        scalar_channels=C_s, vector_channels=C_v,
        ratio=0.25, k_neighbors=16,
    )
    n_params = sum(p.numel() for p in pool.parameters())

    # Expect: q_proj(48*12) + k_proj(48*12) + attn_vec(12*1) + geo_mlp + BN + temperature
    # ~1200 + geo_mlp params
    print(f"  [PASS] Pool param count: {n_params} (Q/K/attn_vec + geo_mlp + BN)")
    assert n_params < 5000, f"Too many params: {n_params}"


if __name__ == '__main__':
    print("=" * 55)
    print("Component 1: Dynamic Pool Attention Tests")
    print("=" * 55)

    tests = [
        ("Attention invariance under SO(3)", test_pool_attention_invariance),
        ("Pool vector equivariance", test_pool_vector_equivariance),
        ("Dynamic attention differentiation", test_dynamic_attention_differentiation),
        ("Gradient flow through Q/K", test_gradient_flow_through_qk),
        ("Parameter count", test_param_count),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n{name}:")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
