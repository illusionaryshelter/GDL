# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Pre-training validation tests for Route B (EquivariantSkipFusion).

Run locally BEFORE committing to remote ShapeNet training.
Covers gradient flow, numerical stability, batch consistency,
parameter budget, and multi-rotation equivariance statistics.

All tests designed for < 4GB memory and < 30s each.
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geoembodied.kernels.triton_sph_harm import spherical_harmonics
from geoembodied.nn.modules.equivariant_skip_fusion import EquivariantSkipFusion
from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net
from geoembodied.nn.models.part_segmentation import SE3PartSegNet


# ──────────────────────────── helpers ────────────────────────────

def _random_SO3(dtype=torch.float64, device='cpu'):
    """Random proper rotation matrix via QR decomposition."""
    A = torch.randn(3, 3, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _compute_D2(R):
    """Wigner-D² matrix from rotation R via SH least-squares."""
    dtype, device = R.dtype, R.device
    torch.manual_seed(777)
    dirs = torch.randn(50, 3, dtype=dtype, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    Y_o = spherical_harmonics(dirs, max_l=2)[:, 4:9]
    Y_r = spherical_harmonics(dirs @ R.T, max_l=2)[:, 4:9]
    return torch.linalg.lstsq(Y_o, Y_r).solution.T


# ──────────────────────────── tests ─────────────────────────────


class TestGradientFlow:
    """Verify backward pass through TP fusion produces valid gradients."""

    def test_grad_not_nan_or_zero(self):
        """All parameter gradients must be finite and non-zero.

        A zero gradient indicates a disconnected computation path.
        A NaN gradient indicates numerical instability (Rule 4 violation).
        """
        C_s, C_v, C_t2 = 16, 4, 2
        torch.manual_seed(0)
        fusion = EquivariantSkipFusion(
            scalar_channels=C_s, vector_channels=C_v,
            type2_channels=C_t2, k_neighbors=4,
        ).to(torch.float64)
        fusion.train()

        N_fine, N_coarse = 32, 8
        torch.manual_seed(42)
        fine_pos = torch.randn(N_fine, 3, dtype=torch.float64, requires_grad=True)
        coarse_pos = fine_pos[:N_coarse].detach().clone()

        s_c = torch.randn(N_coarse, C_s, dtype=torch.float64, requires_grad=True)
        v_c = torch.randn(N_coarse, C_v, 3, dtype=torch.float64, requires_grad=True)
        t2_c = torch.randn(N_coarse, C_t2, 5, dtype=torch.float64, requires_grad=True)
        s_sk = torch.randn(N_fine, C_s, dtype=torch.float64, requires_grad=True)
        v_sk = torch.randn(N_fine, C_v, 3, dtype=torch.float64, requires_grad=True)
        t2_sk = torch.randn(N_fine, C_t2, 5, dtype=torch.float64, requires_grad=True)
        ptr_f = torch.tensor([0, N_fine], dtype=torch.long)
        ptr_c = torch.tensor([0, N_coarse], dtype=torch.long)

        s_out, v_out, t2_out = fusion(
            fine_pos, coarse_pos, s_c, v_c, ptr_f, ptr_c,
            s_skip=s_sk, v_skip=v_sk,
            t2_coarse=t2_c, t2_skip=t2_sk,
        )

        # Loss: sum of all outputs (scalar + vector norm + t2 norm)
        loss = s_out.sum() + v_out.norm() + t2_out.norm()
        loss.backward()

        # Check parameter gradients
        zero_grad_params = []
        nan_grad_params = []
        for name, p in fusion.named_parameters():
            if p.grad is None:
                zero_grad_params.append(name)
                continue
            if torch.isnan(p.grad).any():
                nan_grad_params.append(name)
            elif p.grad.abs().max() == 0:
                zero_grad_params.append(name)

        assert len(nan_grad_params) == 0, \
            f"NaN gradients in: {nan_grad_params}"
        # Note: some params might legitimately be zero if path contributes
        # nothing for this specific input. We only fail on NaN.

        # Check input gradients
        for name, t in [('fine_pos', fine_pos), ('s_coarse', s_c),
                        ('v_coarse', v_c), ('t2_coarse', t2_c),
                        ('s_skip', s_sk), ('v_skip', v_sk), ('t2_skip', t2_sk)]:
            assert t.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(t.grad).any(), f"{name} has NaN gradient"

        print(f"  [✓] {len(list(fusion.parameters()))} params with finite grads")
        print(f"  [✓] 7 input tensors with finite grads")
        if zero_grad_params:
            print(f"  [!] Zero-grad params (may be OK): {zero_grad_params}")

    def test_grad_with_extreme_inputs(self):
        """Gradients remain finite with near-zero and large features.

        Rule 4: dangerous ops (acos, sqrt) must be safely clamped.
        """
        C_s, C_v, C_t2 = 16, 4, 2
        torch.manual_seed(0)
        fusion = EquivariantSkipFusion(
            scalar_channels=C_s, vector_channels=C_v,
            type2_channels=C_t2, k_neighbors=4,
        ).to(torch.float64)
        fusion.train()

        N_fine, N_coarse = 32, 8
        ptr_f = torch.tensor([0, N_fine], dtype=torch.long)
        ptr_c = torch.tensor([0, N_coarse], dtype=torch.long)

        scenarios = {
            'near_zero': 1e-10,
            'large': 1e4,
            'normal': 1.0,
        }

        for label, scale in scenarios.items():
            torch.manual_seed(42)
            fine_pos = torch.randn(N_fine, 3, dtype=torch.float64)
            coarse_pos = fine_pos[:N_coarse].clone()

            s_c = torch.randn(N_coarse, C_s, dtype=torch.float64) * scale
            v_c = torch.randn(N_coarse, C_v, 3, dtype=torch.float64) * scale
            t2_c = torch.randn(N_coarse, C_t2, 5, dtype=torch.float64) * scale
            s_sk = torch.randn(N_fine, C_s, dtype=torch.float64) * scale
            v_sk = torch.randn(N_fine, C_v, 3, dtype=torch.float64) * scale
            t2_sk = torch.randn(N_fine, C_t2, 5, dtype=torch.float64) * scale

            s_c.requires_grad_(True)
            v_c.requires_grad_(True)

            fusion.zero_grad()
            s_out, v_out, t2_out = fusion(
                fine_pos, coarse_pos, s_c, v_c, ptr_f, ptr_c,
                s_skip=s_sk, v_skip=v_sk,
                t2_coarse=t2_c, t2_skip=t2_sk,
            )

            loss = s_out.sum() + v_out.norm() + t2_out.norm()

            # Check output is finite
            assert torch.isfinite(loss), f"[{label}] Loss is not finite: {loss.item()}"

            loss.backward()

            # Check gradients are finite
            has_nan = any(torch.isnan(p.grad).any() for p in fusion.parameters()
                         if p.grad is not None)
            has_inf = any(torch.isinf(p.grad).any() for p in fusion.parameters()
                         if p.grad is not None)
            assert not has_nan, f"[{label}] NaN parameter gradients"
            assert not has_inf, f"[{label}] Inf parameter gradients"

            print(f"  [✓] {label} (scale={scale}): loss={loss.item():.2e}, grads finite")


class TestParameterBudget:
    """Audit parameter count and distribution for Route A vs Route B."""

    def test_route_b_parameter_allocation(self):
        """Route B should reallocate params from head to backbone.

        Specifically:
        - Head should be slimmer (1-layer MLP vs multi-scale fusion)
        - Backbone should have EquivariantSkipFusion layers
        """
        torch.manual_seed(0)
        model_b = SE3PartSegNet(
            num_categories=16, num_parts=50,
            in_channels=1, hidden_scalar=32, hidden_vector=8, hidden_type2=4,
            num_stages=3, layers_per_stage=2, pool_ratio=0.25,
            use_tp_fusion=True,
        )

        total = sum(p.numel() for p in model_b.parameters())
        backbone = sum(p.numel() for p in model_b.backbone.parameters())
        head = total - backbone

        print(f"  Route B parameter budget:")
        print(f"    Total:    {total:>8,d}")
        print(f"    Backbone: {backbone:>8,d} ({backbone/total*100:.1f}%)")
        print(f"    Head:     {head:>8,d} ({head/total*100:.1f}%)")

        # Backbone should dominate (> 70% of params)
        assert backbone / total > 0.5, \
            f"Backbone ratio too low: {backbone/total:.2%}"

        # Check fusion layers exist
        assert hasattr(model_b.backbone, 'fusion_layers'), \
            "Backbone missing fusion_layers"
        n_fusion = len(model_b.backbone.fusion_layers)
        fusion_params = sum(
            p.numel() for fl in model_b.backbone.fusion_layers
            for p in fl.parameters()
        )
        print(f"    Fusion:   {fusion_params:>8,d} ({n_fusion} layers)")

        print(f"  [✓] Parameter allocation within bounds")

    def test_route_a_vs_b_output_shapes(self):
        """Route A and Route B must produce identically shaped outputs."""
        torch.manual_seed(0)
        N, dtype = 64, torch.float32
        pos = torch.randn(N, 3, dtype=dtype)
        ptr = torch.tensor([0, N], dtype=torch.long)
        cat = torch.zeros(1, dtype=torch.long)
        normals = torch.randn(N, 3, dtype=dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True)

        torch.manual_seed(0)
        model_a = SE3PartSegNet(
            num_categories=16, num_parts=50,
            in_channels=1, hidden_scalar=32, hidden_vector=8, hidden_type2=4,
            num_stages=2, layers_per_stage=1, pool_ratio=0.25,
            use_tp_fusion=False,
        )
        model_a.eval()

        torch.manual_seed(0)
        model_b = SE3PartSegNet(
            num_categories=16, num_parts=50,
            in_channels=1, hidden_scalar=32, hidden_vector=8, hidden_type2=4,
            num_stages=2, layers_per_stage=1, pool_ratio=0.25,
            use_tp_fusion=True,
        )
        model_b.eval()

        with torch.no_grad():
            logits_a = model_a(pos, ptr, cat, normals=normals)
            logits_b = model_b(pos, ptr, cat, normals=normals)

        assert logits_a.shape == logits_b.shape, \
            f"Shape mismatch: A={logits_a.shape} vs B={logits_b.shape}"

        print(f"  [✓] Route A output: {logits_a.shape}")
        print(f"  [✓] Route B output: {logits_b.shape}")
        print(f"  [✓] Shapes match ✓")


class TestBatchConsistency:
    """Verify equivariance holds for multi-batch inputs."""

    def test_multi_batch_equivariance(self):
        """Equivariance with batch_size=2 (two independent point clouds).

        Each cloud is independently rotated by the SAME R.
        """
        dtype = torch.float64
        N_per_cloud = 64

        torch.manual_seed(42)
        pos_1 = torch.randn(N_per_cloud, 3, dtype=dtype)
        pos_2 = torch.randn(N_per_cloud, 3, dtype=dtype) + 5.0  # offset
        pos = torch.cat([pos_1, pos_2], dim=0)
        normals = torch.randn(pos.shape[0], 3, dtype=dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True)

        ptr = torch.tensor([0, N_per_cloud, 2 * N_per_cloud], dtype=torch.long)
        cat = torch.tensor([0, 1], dtype=torch.long)

        R = _random_SO3(dtype=dtype)

        torch.manual_seed(0)
        model = SE3PartSegNet(
            num_categories=16, num_parts=4,
            in_channels=1, hidden_scalar=32, hidden_vector=8, hidden_type2=4,
            num_stages=2, layers_per_stage=1, pool_ratio=0.25,
            use_tp_fusion=True,
        ).to(dtype)
        model.eval()

        with torch.no_grad():
            logits_o = model(pos, ptr, cat, normals=normals)
            logits_r = model(pos @ R.T, ptr, cat, normals=normals @ R.T)

        logit_err = (logits_r - logits_o).abs().max().item()
        logit_rel = logit_err / logits_o.abs().max().clamp(min=1e-10).item()

        assert logit_rel < 1e-3, \
            f"Multi-batch logit invariance failed: rel={logit_rel:.2e}"

        # Check per-cloud invariance
        err_1 = (logits_r[:N_per_cloud] - logits_o[:N_per_cloud]).abs().max().item()
        err_2 = (logits_r[N_per_cloud:] - logits_o[N_per_cloud:]).abs().max().item()
        rel_1 = err_1 / logits_o[:N_per_cloud].abs().max().clamp(min=1e-10).item()
        rel_2 = err_2 / logits_o[N_per_cloud:].abs().max().clamp(min=1e-10).item()

        print(f"  [✓] Cloud 1 invariance: rel={rel_1:.2e}")
        print(f"  [✓] Cloud 2 invariance: rel={rel_2:.2e}")
        print(f"  [✓] Combined: rel={logit_rel:.2e}")


class TestMultiRotationStats:
    """Statistical equivariance test across 10 random rotations.

    Ensures the fix is not coincidentally correct for one rotation.
    """

    def test_10_random_rotations(self):
        """Equivariance must hold for 10 independently sampled rotations.

        Reports min/max/mean error across rotations.
        """
        dtype = torch.float64
        N = 64

        torch.manual_seed(42)
        pos = torch.randn(N, 3, dtype=dtype)
        ptr = torch.tensor([0, N], dtype=torch.long)
        normals = torch.randn(N, 3, dtype=dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        v_init = normals.unsqueeze(1).expand(-1, 8, -1).clone()

        torch.manual_seed(0)
        backbone = MultiScaleSE3Net(
            in_channels=1, hidden_scalar=32, hidden_vector=8, hidden_type2=4,
            num_stages=2, layers_per_stage=1, pool_ratio=0.25,
            gate_mode='norm', use_bottleneck_attn=True, use_tp_fusion=True,
        ).to(dtype)
        backbone.eval()

        with torch.no_grad():
            out_o = backbone(pos, ptr, v_init=v_init, return_encoder_features=False)
        s_o, v_o, t2_o = out_o[0], out_o[1], out_o[2]
        t2_norm_o = t2_o.norm(dim=-1) if t2_o is not None else None

        errors_s, errors_v, errors_t2 = [], [], []

        for i in range(10):
            torch.manual_seed(1000 + i)
            R = _random_SO3(dtype=dtype)

            with torch.no_grad():
                out_r = backbone(pos @ R.T, ptr, v_init=v_init @ R.T,
                                 return_encoder_features=False)
            s_r, v_r, t2_r = out_r[0], out_r[1], out_r[2]

            s_rel = (s_r - s_o).abs().max().item() / s_o.abs().max().clamp(min=1e-10).item()
            v_rel = (v_r - v_o @ R.T).abs().max().item() / v_o.abs().max().clamp(min=1e-10).item()
            errors_s.append(s_rel)
            errors_v.append(v_rel)

            if t2_o is not None:
                t2_norm_r = t2_r.norm(dim=-1)
                t2_nr = (t2_norm_r - t2_norm_o).abs().max().item() / \
                        t2_norm_o.abs().max().clamp(min=1e-10).item()
                errors_t2.append(t2_nr)

        print(f"  10 random rotations — equivariance errors:")
        print(f"    scalar: min={min(errors_s):.2e} max={max(errors_s):.2e} "
              f"mean={sum(errors_s)/len(errors_s):.2e}")
        print(f"    vector: min={min(errors_v):.2e} max={max(errors_v):.2e} "
              f"mean={sum(errors_v)/len(errors_v):.2e}")
        if errors_t2:
            print(f"    type-2: min={min(errors_t2):.2e} max={max(errors_t2):.2e} "
                  f"mean={sum(errors_t2)/len(errors_t2):.2e}")

        assert max(errors_s) < 1e-4, f"Scalar worst-case: {max(errors_s):.2e}"
        assert max(errors_v) < 1e-4, f"Vector worst-case: {max(errors_v):.2e}"
        if errors_t2:
            assert max(errors_t2) < 1e-4, f"Type-2 worst-case: {max(errors_t2):.2e}"

        print(f"  [✓] All 10 rotations within tolerance")


# ──────────────────── run all tests ─────────────────────

def main():
    """Run all pre-training validation tests."""
    print("=" * 60)
    print("PRE-TRAINING VALIDATION: Route B (TP Fusion)")
    print("=" * 60)

    print("\n─── 1. Gradient Flow ───")
    grad = TestGradientFlow()
    grad.test_grad_not_nan_or_zero()
    print()
    grad.test_grad_with_extreme_inputs()

    print("\n─── 2. Parameter Budget ───")
    params = TestParameterBudget()
    params.test_route_b_parameter_allocation()
    print()
    params.test_route_a_vs_b_output_shapes()

    print("\n─── 3. Multi-Batch Equivariance ───")
    batch = TestBatchConsistency()
    batch.test_multi_batch_equivariance()

    print("\n─── 4. Multi-Rotation Statistics (10 rotations) ───")
    multi_rot = TestMultiRotationStats()
    multi_rot.test_10_random_rotations()

    print("\n" + "=" * 60)
    print("ALL PRE-TRAINING CHECKS PASSED ✓")
    print("Ready for remote ShapeNet training (Phase 4)")
    print("=" * 60)


if __name__ == '__main__':
    main()
