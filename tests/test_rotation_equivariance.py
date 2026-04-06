#!/usr/bin/env python3
"""Rotation equivariance test for SE3PartSegNet.

Tests the fundamental GDL property: the model's output logits must be
*invariant* under arbitrary SO(3) rotations of the input point cloud
and normals. For a part segmentation model:
    f(R·pos, R·normals) == f(pos, normals)   ∀ R ∈ SO(3)

We test:
1. Logit invariance (max absolute error)
2. Prediction consistency (argmax match %)
3. Probability distribution shift (KL divergence)
4. Multiple random rotations to catch axis-specific issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../examples/shapenet_seg'))

import torch
import torch.nn.functional as F
import numpy as np

# ── Generate random SO(3) rotation matrix ──
def random_rotation_matrix(dtype=torch.float64) -> torch.Tensor:
    """Generate a random SO(3) rotation matrix via QR decomposition.
    
    Returns:
        R: [3, 3] orthogonal matrix with det=+1
    """
    # Random matrix → QR → ensure det = +1
    M = torch.randn(3, 3, dtype=dtype)
    Q, R_tri = torch.linalg.qr(M)
    # Fix sign to ensure det(Q) = +1
    d = torch.diag(R_tri)
    ph = d.sign()
    Q = Q * ph.unsqueeze(0)
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def rotation_from_axis_angle(axis: torch.Tensor, angle: float, 
                              dtype=torch.float64) -> torch.Tensor:
    """Rodrigues rotation formula for specific axis-angle."""
    axis = axis.to(dtype)
    axis = axis / axis.norm()
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=dtype)
    R = torch.eye(3, dtype=dtype) + torch.sin(torch.tensor(angle, dtype=dtype)) * K + \
        (1 - torch.cos(torch.tensor(angle, dtype=dtype))) * (K @ K)
    return R


def main():
    from model import SE3PartSegNet
    
    device = torch.device('cpu')  # CPU for FP64 precision
    
    # ── Load model ──
    ckpt_path = os.path.expanduser(
        os.path.join(os.path.dirname(__file__), 
                     '../examples/shapenet_seg/shapenet_best.pt')
    )
    if not os.path.exists(ckpt_path):
        # Try alternative path
        ckpt_path = '/home/shelter/Desktop/GDL/examples/shapenet_seg/shapenet_best.pt'
    
    print("=" * 72)
    print("  SO(3) Rotation Invariance Test — SE3PartSegNet")
    print("=" * 72)
    print(f"  Checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    
    model = SE3PartSegNet(
        in_channels=1, hidden_scalar=96, hidden_vector=24, hidden_type2=8,
        num_stages=3, layers_per_stage=2,
        gate_mode='norm', use_self_tp=True, use_bottleneck_attn=True,
        head_hidden=256,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    # Convert model to float64 for rigorous test
    model = model.double()
    
    # Total params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()
    
    # ── Test configurations ──
    test_configs = [
        # (name, N_points, rotation_generator)
        ("Random SO(3)", 256, lambda: random_rotation_matrix()),
        ("90° around Z", 256, lambda: rotation_from_axis_angle(
            torch.tensor([0., 0., 1.]), np.pi / 2)),
        ("180° around X", 256, lambda: rotation_from_axis_angle(
            torch.tensor([1., 0., 0.]), np.pi)),
        ("45° around (1,1,1)", 256, lambda: rotation_from_axis_angle(
            torch.tensor([1., 1., 1.]), np.pi / 4)),
        ("Random SO(3) #2", 512, lambda: random_rotation_matrix()),
        ("Random SO(3) #3 (large)", 1024, lambda: random_rotation_matrix()),
    ]
    
    results = []
    
    for name, N, rot_gen in test_configs:
        torch.manual_seed(42)  # Same input for reproducibility
        
        # ── Generate input ──
        pos = torch.randn(N, 3, dtype=torch.float64)
        normals = torch.randn(N, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        ptr = torch.tensor([0, N], dtype=torch.int64)
        cat_idx = torch.tensor([0])
        
        # ── Generate rotation ──
        R = rot_gen()
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12), \
            "R is not orthogonal!"
        assert torch.allclose(R.det(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12), \
            "det(R) != 1!"
        
        # ── Rotate inputs ──
        pos_rot = pos @ R.T         # [N, 3] @ [3, 3] = [N, 3]
        normals_rot = normals @ R.T  # Normals are type-1 vectors
        
        # ── Forward pass ──
        with torch.no_grad():
            logits_orig = model(pos, ptr, cat_idx, normals=normals)
            logits_rot = model(pos_rot, ptr, cat_idx, normals=normals_rot)
        
        # ── Analysis ──
        # 1. Raw logit difference
        diff = (logits_orig - logits_rot).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 2. Prediction match
        pred_orig = logits_orig.argmax(dim=1)
        pred_rot = logits_rot.argmax(dim=1)
        match_pct = (pred_orig == pred_rot).float().mean().item() * 100
        
        # 3. Relative error (compared to logit magnitude)
        logit_scale = logits_orig.abs().max().item()
        rel_error = max_diff / max(logit_scale, 1e-10)
        
        # 4. Softmax probability shift
        prob_orig = F.softmax(logits_orig, dim=1)
        prob_rot = F.softmax(logits_rot, dim=1)
        prob_diff = (prob_orig - prob_rot).abs().max().item()
        
        # 5. KL divergence (symmetrized)
        eps = 1e-12
        kl_fwd = (prob_orig * ((prob_orig + eps).log() - (prob_rot + eps).log())).sum(1).mean().item()
        kl_bwd = (prob_rot * ((prob_rot + eps).log() - (prob_orig + eps).log())).sum(1).mean().item()
        kl_sym = (kl_fwd + kl_bwd) / 2
        
        verdict = "✓ PASS" if max_diff < 1e-4 else ("⚠ WARN" if max_diff < 1e-2 else "✗ FAIL")
        
        results.append({
            'name': name, 'N': N,
            'max_diff': max_diff, 'mean_diff': mean_diff,
            'rel_error': rel_error, 'match_pct': match_pct,
            'prob_diff': prob_diff, 'kl_sym': kl_sym,
            'verdict': verdict,
        })
        
        print(f"  {name:30s} (N={N:4d}) | "
              f"max_Δ={max_diff:.2e} rel={rel_error:.2e} | "
              f"pred_match={match_pct:6.2f}% | "
              f"prob_Δ={prob_diff:.2e} | {verdict}")
    
    # ── Summary ──
    print()
    print("─" * 72)
    max_of_maxes = max(r['max_diff'] for r in results)
    min_match = min(r['match_pct'] for r in results)
    
    print(f"  Worst-case logit diff:  {max_of_maxes:.2e}")
    print(f"  Worst-case pred match:  {min_match:.2f}%")
    
    if max_of_maxes < 1e-6:
        print(f"\n  ✓ EXCELLENT — Logit error < 1e-6 (machine precision level)")
    elif max_of_maxes < 1e-4:
        print(f"\n  ✓ GOOD — Logit error < 1e-4 (acceptable for FP64)")
    elif max_of_maxes < 1e-2:
        print(f"\n  ⚠ WARNING — Logit error in [1e-4, 1e-2] range")
        print(f"    Predictions may still match (argmax robust), but")
        print(f"    SO(3) equivariance is not numerically exact.")
    else:
        print(f"\n  ✗ FAIL — Logit error > 1e-2")
        print(f"    MODEL IS NOT ROTATION INVARIANT!")
        print(f"    This likely indicates an architectural bug.")
    
    # ── Per-category rotation test ──
    print()
    print("=" * 72)
    print("  Per-Category Test: 5 random SO(3) rotations, N=256")
    print("=" * 72)
    
    n_cats = 16
    worst_per_cat = []
    for cat_id in range(min(n_cats, 4)):  # Test first 4 categories
        cat_diffs = []
        for trial in range(5):
            torch.manual_seed(100 + trial)
            N = 256
            pos = torch.randn(N, 3, dtype=torch.float64)
            normals = torch.randn(N, 3, dtype=torch.float64)
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            ptr = torch.tensor([0, N], dtype=torch.int64)
            cat_idx = torch.tensor([cat_id])
            
            R = random_rotation_matrix()
            pos_rot = pos @ R.T
            normals_rot = normals @ R.T
            
            with torch.no_grad():
                logits_orig = model(pos, ptr, cat_idx, normals=normals)
                logits_rot = model(pos_rot, ptr, cat_idx, normals=normals_rot)
            
            max_diff = (logits_orig - logits_rot).abs().max().item()
            cat_diffs.append(max_diff)
        
        worst = max(cat_diffs)
        mean = np.mean(cat_diffs)
        worst_per_cat.append(worst)
        print(f"  Category {cat_id:2d}: worst={worst:.2e}  mean={mean:.2e}")
    
    print()
    overall_worst = max(worst_per_cat)
    if overall_worst < 1e-4:
        print("  ✓ All categories pass rotation invariance (< 1e-4)")
    else:
        print(f"  ⚠ Some categories have invariance error > 1e-4 (worst: {overall_worst:.2e})")
    
    return 0 if max_of_maxes < 1e-2 else 1


if __name__ == '__main__':
    sys.exit(main())
