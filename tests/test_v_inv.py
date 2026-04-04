#!/usr/bin/env python3
# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Tests for Component 2: Vector Feature Utilization in Head
#
# Tests verify the key invariant property:
#   ||R @ v|| == ||v|| for any rotation R in SO(3)
# This guarantees that v_inv_proj(||v||) is SE(3)-invariant,
# making it safe to feed into the invariant classification head.
#
# Run: python tests/test_v_inv.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math


def random_rotation_matrix(dtype=torch.float64, device='cpu'):
    """Generate random SO(3) rotation via QR decomposition.

    Returns:
        Q: shape: [3, 3], representation: SO(3) rotation matrix
    """
    M = torch.randn(3, 3, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(M)
    # Fix signs to ensure det(Q) = +1 (proper rotation)
    signs = torch.diag(R).sign()
    Q = Q * signs.unsqueeze(0)
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_vector_norm_invariance():
    """Test that ||R @ v|| == ||v|| for random SO(3) rotations.

    This is the fundamental property that makes v_inv safe for the head.
    """
    torch.manual_seed(42)
    N, C_v = 128, 16

    # Random vector features: [N, C_v, 3], representation: SO(3) type-1
    v = torch.randn(N, C_v, 3, dtype=torch.float64)

    # Random rotation
    R = random_rotation_matrix(dtype=torch.float64)  # [3, 3]

    # Rotate all vectors: v_rot[n, c] = R @ v[n, c]
    v_rot = torch.einsum('ij, ncj -> nci', R, v)  # [N, C_v, 3]

    # Per-channel norms should be identical
    norms_orig = v.norm(dim=-1)     # [N, C_v]
    norms_rot = v_rot.norm(dim=-1)  # [N, C_v]

    err = (norms_orig - norms_rot).abs().max().item()
    assert err < 1e-10, f"Vector norm is NOT rotation-invariant! max_err={err}"
    print(f"  [PASS] ||R@v|| == ||v||, max_err = {err:.2e}")


def test_v_inv_proj_invariance():
    """Test that Linear(||v||) is rotation-invariant end-to-end.

    Verifies: v_inv_proj(||R @ v||) == v_inv_proj(||v||)
    """
    torch.manual_seed(42)
    N, C_v, C_out = 128, 12, 24  # typical: hidden_vector=12, C_s//2=24

    # Create projection layer (same as in SE3PartSegNet)
    proj = torch.nn.Linear(C_v, C_out, bias=False).double()

    v = torch.randn(N, C_v, 3, dtype=torch.float64)
    R = random_rotation_matrix(dtype=torch.float64)
    v_rot = torch.einsum('ij, ncj -> nci', R, v)

    # Compute v_inv for original and rotated
    v_inv = proj(v.norm(dim=-1))         # [N, C_out]
    v_inv_rot = proj(v_rot.norm(dim=-1))  # [N, C_out]

    err = (v_inv - v_inv_rot).abs().max().item()
    assert err < 1e-10, f"v_inv_proj is NOT invariant! max_err={err}"
    print(f"  [PASS] v_inv_proj invariant, max_err = {err:.2e}")


def test_t2_norm_sq_invariance():
    """Test that ||t2_c||^2 = sum_m t2[c,m]^2 is invariant under orthogonal D.

    For type-2 (l=2) features with 2l+1=5 components, the Wigner D-matrix
    is unitary: D^(2)(R)^T D^(2)(R) = I, so ||D^(2)(R) t2||^2 = ||t2||^2.
    """
    torch.manual_seed(42)
    N, C_t2 = 128, 4

    t2 = torch.randn(N, C_t2, 5, dtype=torch.float64)

    # Generate random orthogonal matrix in 5x5 (simulating Wigner D^(2))
    M = torch.randn(5, 5, dtype=torch.float64)
    Q, R = torch.linalg.qr(M)
    signs = torch.diag(R).sign()
    Q = Q * signs.unsqueeze(0)
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]

    t2_rot = torch.einsum('ij, ncj -> nci', Q, t2)

    norm_sq = (t2 * t2).sum(dim=-1)             # [N, C_t2]
    norm_sq_rot = (t2_rot * t2_rot).sum(dim=-1)

    err = (norm_sq - norm_sq_rot).abs().max().item()
    assert err < 1e-10, f"||t2||^2 NOT invariant under D^(2)! max_err={err}"
    print(f"  [PASS] ||t2||^2 invariant under Wigner D^(2), max_err = {err:.2e}")


def test_head_input_dimension():
    """Test that the head input dimension calculation matches expected."""
    from geoembodied.nn.models.part_segmentation import SE3PartSegNet

    C_s, C_v, C_t2 = 48, 12, 4
    num_stages, num_cats = 2, 16

    model = SE3PartSegNet(
        num_categories=num_cats,
        num_parts=50,
        hidden_scalar=C_s,
        hidden_vector=C_v,
        hidden_type2=C_t2,
        num_stages=num_stages,
        head_hidden=128,
    )

    # t2_inv is projected: Linear(C_t2, C_s//2) + LayerNorm → output dim = C_s//2
    t2_inv_out = C_s // 2 if C_t2 > 0 else 0
    expected = C_s + C_s // 2 + t2_inv_out + C_s * num_stages + num_cats
    assert model._head_in_dim == expected, (
        f"Head dim mismatch: expected {expected}, got {model._head_in_dim}"
    )
    print(f"  [PASS] head_in_dim = {expected} (C_s={C_s} + v_inv={C_s // 2} + "
          f"t2_inv={t2_inv_out} + stages*C_s={num_stages * C_s} + cats={num_cats})")


def test_gradient_flow_through_v_inv():
    """Test that gradients flow through the v_inv projection back to v.

    If v_inv_proj kills gradients, the vector branch won't learn.
    """
    torch.manual_seed(42)
    N, C_v, C_out = 32, 12, 24

    proj = torch.nn.Linear(C_v, C_out, bias=False)
    v = torch.randn(N, C_v, 3, requires_grad=True)

    v_norms = v.norm(dim=-1)  # [N, C_v]
    v_inv = proj(v_norms)     # [N, C_out]
    loss = v_inv.sum()
    loss.backward()

    assert v.grad is not None, "No gradient on v!"
    assert v.grad.abs().sum() > 0, "Zero gradient on v!"
    grad_norm = v.grad.norm().item()
    print(f"  [PASS] Gradients flow through v_inv, ||grad_v|| = {grad_norm:.4f}")


if __name__ == '__main__':
    print("=" * 50)
    print("Component 2: Vector Feature Invariance Tests")
    print("=" * 50)

    tests = [
        ("Vector norm invariance", test_vector_norm_invariance),
        ("v_inv_proj end-to-end invariance", test_v_inv_proj_invariance),
        ("Type-2 ||t2||^2 invariance", test_t2_norm_sq_invariance),
        ("Head input dimension", test_head_input_dimension),
        ("Gradient flow through v_inv", test_gradient_flow_through_v_inv),
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
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
