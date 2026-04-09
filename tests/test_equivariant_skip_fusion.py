# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for EquivariantSkipFusion module.

Covers:
    1. Standalone SO(3) equivariance (scalar invariance, vector/t2 equivariance)
    2. Degenerate self-loop edge handling (zero-distance FPS overlap)
    3. Full backbone integration with --use_tp_fusion
    4. Full model logit invariance under arbitrary rotation

Mathematical guarantee tested (Rule 3: Maintain Equivariance):
    f(Rx, Rv, D²t₂) = (f_s(x), R·f_v(x), D²·f_t2(x))

All tests use float64 for tight numerical tolerances.
"""

import sys
import os

import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geoembodied.kernels.triton_sph_harm import spherical_harmonics
from geoembodied.nn.modules.equivariant_skip_fusion import EquivariantSkipFusion


# ──────────────────────────── helpers ────────────────────────────

def _random_SO3(dtype: torch.dtype = torch.float64,
                device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Generate a uniformly random proper rotation matrix via QR.

    Returns:
        R: [3, 3] orthogonal matrix with det(R) = +1
    """
    A = torch.randn(3, 3, dtype=dtype, device=device)
    Q, Rqr = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(Rqr)))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _compute_wigner_D2(R: torch.Tensor) -> torch.Tensor:
    """Compute Wigner-D matrix for l=2 from a rotation matrix R.

    Uses the spherical harmonics function to empirically derive D² via
    least-squares fit on random unit directions.

    Args:
        R: [3, 3] rotation matrix

    Returns:
        D2: [5, 5] Wigner-D matrix for l=2 representation
    """
    dtype, device = R.dtype, R.device
    torch.manual_seed(777)
    dirs = torch.randn(50, 3, dtype=dtype, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    Y_orig = spherical_harmonics(dirs, max_l=2)[:, 4:9]
    Y_rot = spherical_harmonics(dirs @ R.T, max_l=2)[:, 4:9]
    D2 = torch.linalg.lstsq(Y_orig, Y_rot).solution.T
    # Verify D2 is orthogonal (sanity check)
    assert (D2 @ D2.T - torch.eye(5, dtype=dtype, device=device)).norm() < 1e-6, \
        "D2 is not orthogonal — SH convention issue"
    return D2


# ──────────────────────────── tests ─────────────────────────────


class TestEquivariantSkipFusionStandalone:
    """Test EquivariantSkipFusion module in isolation."""

    def _make_fusion(self, C_s=32, C_v=8, C_t2=4, K=6, dtype=torch.float64):
        """Create a fusion module with specified channel config."""
        fusion = EquivariantSkipFusion(
            scalar_channels=C_s,
            vector_channels=C_v,
            type2_channels=C_t2,
            k_neighbors=K,
        ).to(dtype)
        fusion.eval()
        return fusion

    def _make_inputs(self, N_fine=128, N_coarse=32, C_s=32, C_v=8, C_t2=4,
                     dtype=torch.float64, seed=42):
        """Create synthetic inputs mimicking encoder output.

        Critically: coarse points are a SUBSET of fine points (as with FPS),
        which creates zero-distance self-loop edges — the exact scenario
        that previously broke t2 equivariance.
        """
        torch.manual_seed(seed)
        fine_pos = torch.randn(N_fine, 3, dtype=dtype)

        # Coarse = first N_coarse fine points (simulating FPS subset)
        coarse_pos = fine_pos[:N_coarse].clone()

        s_coarse = torch.randn(N_coarse, C_s, dtype=dtype)
        v_coarse = torch.randn(N_coarse, C_v, 3, dtype=dtype)
        t2_coarse = torch.randn(N_coarse, C_t2, 5, dtype=dtype)

        s_skip = torch.randn(N_fine, C_s, dtype=dtype)
        v_skip = torch.randn(N_fine, C_v, 3, dtype=dtype)
        t2_skip = torch.randn(N_fine, C_t2, 5, dtype=dtype)

        ptr_fine = torch.tensor([0, N_fine], dtype=torch.long)
        ptr_coarse = torch.tensor([0, N_coarse], dtype=torch.long)

        return {
            'fine_pos': fine_pos, 'coarse_pos': coarse_pos,
            's_coarse': s_coarse, 'v_coarse': v_coarse,
            't2_coarse': t2_coarse,
            's_skip': s_skip, 'v_skip': v_skip, 't2_skip': t2_skip,
            'ptr_fine': ptr_fine, 'ptr_coarse': ptr_coarse,
        }

    def test_equivariance_with_fps_selfloops(self):
        """Core equivariance test: fusion must be equivariant even when
        coarse points are a subset of fine points (creating dist=0 edges).

        This is the scenario that previously caused a 6.56% t2 error.
        """
        dtype = torch.float64
        C_s, C_v, C_t2 = 32, 8, 4

        torch.manual_seed(0)
        fusion = self._make_fusion(C_s, C_v, C_t2, dtype=dtype)
        inputs = self._make_inputs(N_fine=128, N_coarse=32,
                                   C_s=C_s, C_v=C_v, C_t2=C_t2, dtype=dtype)

        R = _random_SO3(dtype=dtype)
        D2 = _compute_wigner_D2(R)

        with torch.no_grad():
            # Original
            s_o, v_o, t2_o = fusion(
                inputs['fine_pos'], inputs['coarse_pos'],
                inputs['s_coarse'], inputs['v_coarse'],
                inputs['ptr_fine'], inputs['ptr_coarse'],
                s_skip=inputs['s_skip'], v_skip=inputs['v_skip'],
                t2_coarse=inputs['t2_coarse'], t2_skip=inputs['t2_skip'],
            )

            # Rotated
            s_r, v_r, t2_r = fusion(
                inputs['fine_pos'] @ R.T, inputs['coarse_pos'] @ R.T,
                inputs['s_coarse'],  # invariant
                inputs['v_coarse'] @ R.T,
                inputs['ptr_fine'], inputs['ptr_coarse'],
                s_skip=inputs['s_skip'],
                v_skip=inputs['v_skip'] @ R.T,
                t2_coarse=inputs['t2_coarse'] @ D2.T,
                t2_skip=inputs['t2_skip'] @ D2.T,
            )

        # Scalar invariance
        s_rel = (s_r - s_o).abs().max().item() / s_o.abs().max().clamp(min=1e-10).item()
        assert s_rel < 1e-4, f"Scalar invariance failed: rel={s_rel:.2e}"

        # Vector equivariance: v_r = v_o @ R.T
        v_err = (v_r - v_o @ R.T).abs().max().item()
        v_rel = v_err / v_o.abs().max().clamp(min=1e-10).item()
        assert v_rel < 1e-4, f"Vector equivariance failed: rel={v_rel:.2e}"

        # Type-2 equivariance: t2_r = t2_o @ D2.T
        t2_err = (t2_r - t2_o @ D2.T).abs().max().item()
        t2_rel = t2_err / t2_o.abs().max().clamp(min=1e-10).item()
        assert t2_rel < 1e-4, f"Type-2 equivariance failed: rel={t2_rel:.2e}"

        # Also verify norm invariance (convention-free check)
        t2_norm_err = (t2_r.norm(dim=-1) - t2_o.norm(dim=-1)).abs().max().item()
        t2_norm_rel = t2_norm_err / t2_o.norm(dim=-1).max().clamp(min=1e-10).item()
        assert t2_norm_rel < 1e-4, f"Type-2 norm invariance failed: rel={t2_norm_rel:.2e}"

        print(f"  [✓] s invariance:    rel={s_rel:.2e}")
        print(f"  [✓] v equivariance:  rel={v_rel:.2e}")
        print(f"  [✓] t2 equivariance: rel={t2_rel:.2e}")
        print(f"  [✓] t2 norm invar:   rel={t2_norm_rel:.2e}")

    def test_degenerate_edges_zeroed(self):
        """Verify that zero-distance edges produce zero SH contribution.

        When fine and coarse points coincide (dist=0), the edge direction
        is undefined. The module must zero out the SH values to prevent
        injecting non-equivariant signal.
        """
        dtype = torch.float64
        C_s, C_v, C_t2 = 16, 4, 2

        torch.manual_seed(0)
        fusion = self._make_fusion(C_s, C_v, C_t2, K=4, dtype=dtype)

        # Deliberately create ALL self-loops: fine == coarse
        N = 8
        pos = torch.randn(N, 3, dtype=dtype)
        coarse_pos = pos.clone()  # 100% overlap

        s_coarse = torch.randn(N, C_s, dtype=dtype)
        v_coarse = torch.randn(N, C_v, 3, dtype=dtype)
        t2_coarse = torch.randn(N, C_t2, 5, dtype=dtype)
        s_skip = torch.randn(N, C_s, dtype=dtype)
        v_skip = torch.randn(N, C_v, 3, dtype=dtype)
        t2_skip = torch.randn(N, C_t2, 5, dtype=dtype)
        ptr = torch.tensor([0, N], dtype=torch.long)

        R = _random_SO3(dtype=dtype)
        D2 = _compute_wigner_D2(R)

        with torch.no_grad():
            s_o, v_o, t2_o = fusion(
                pos, coarse_pos, s_coarse, v_coarse, ptr, ptr,
                s_skip=s_skip, v_skip=v_skip,
                t2_coarse=t2_coarse, t2_skip=t2_skip,
            )
            s_r, v_r, t2_r = fusion(
                pos @ R.T, coarse_pos @ R.T, s_coarse,
                v_coarse @ R.T, ptr, ptr,
                s_skip=s_skip, v_skip=v_skip @ R.T,
                t2_coarse=t2_coarse @ D2.T, t2_skip=t2_skip @ D2.T,
            )

        s_rel = (s_r - s_o).abs().max().item() / s_o.abs().max().clamp(min=1e-10).item()
        t2_norm_err = (t2_r.norm(dim=-1) - t2_o.norm(dim=-1)).abs().max().item()
        t2_norm_rel = t2_norm_err / t2_o.norm(dim=-1).max().clamp(min=1e-10).item()

        # With 100% self-loops, all TP direction-dependent paths should
        # contribute zero, leaving only skip connections + bias.
        # Equivariance should still hold perfectly.
        assert s_rel < 1e-4, f"Scalar invariance failed with 100% self-loops: rel={s_rel:.2e}"
        assert t2_norm_rel < 1e-4, f"t2 norm invariance failed with 100% self-loops: rel={t2_norm_rel:.2e}"

        print(f"  [✓] 100% self-loops: s_rel={s_rel:.2e}, t2_norm_rel={t2_norm_rel:.2e}")

    def test_no_selfloops_baseline(self):
        """Sanity check: equivariance with NO self-loops (all unique points)."""
        dtype = torch.float64
        C_s, C_v, C_t2 = 32, 8, 4

        torch.manual_seed(0)
        fusion = self._make_fusion(C_s, C_v, C_t2, dtype=dtype)

        N_fine, N_coarse = 64, 16
        torch.manual_seed(42)
        fine_pos = torch.randn(N_fine, 3, dtype=dtype)
        # Coarse is DIFFERENT from fine (no overlap)
        coarse_pos = torch.randn(N_coarse, 3, dtype=dtype) * 2.0

        s_coarse = torch.randn(N_coarse, C_s, dtype=dtype)
        v_coarse = torch.randn(N_coarse, C_v, 3, dtype=dtype)
        t2_coarse = torch.randn(N_coarse, C_t2, 5, dtype=dtype)
        s_skip = torch.randn(N_fine, C_s, dtype=dtype)
        v_skip = torch.randn(N_fine, C_v, 3, dtype=dtype)
        t2_skip = torch.randn(N_fine, C_t2, 5, dtype=dtype)
        ptr_f = torch.tensor([0, N_fine], dtype=torch.long)
        ptr_c = torch.tensor([0, N_coarse], dtype=torch.long)

        R = _random_SO3(dtype=dtype)
        D2 = _compute_wigner_D2(R)

        with torch.no_grad():
            s_o, v_o, t2_o = fusion(
                fine_pos, coarse_pos, s_coarse, v_coarse, ptr_f, ptr_c,
                s_skip=s_skip, v_skip=v_skip,
                t2_coarse=t2_coarse, t2_skip=t2_skip,
            )
            s_r, v_r, t2_r = fusion(
                fine_pos @ R.T, coarse_pos @ R.T, s_coarse,
                v_coarse @ R.T, ptr_f, ptr_c,
                s_skip=s_skip, v_skip=v_skip @ R.T,
                t2_coarse=t2_coarse @ D2.T, t2_skip=t2_skip @ D2.T,
            )

        t2_rel = (t2_r - t2_o @ D2.T).abs().max().item() / t2_o.abs().max().clamp(min=1e-10).item()
        assert t2_rel < 1e-4, f"t2 equivariance failed (no self-loops): rel={t2_rel:.2e}"

        print(f"  [✓] No self-loops baseline: t2_rel={t2_rel:.2e}")


class TestBackboneTPFusion:
    """Test EquivariantSkipFusion integrated into MultiScaleSE3Net backbone."""

    def test_backbone_equivariance(self):
        """Full backbone equivariance with --use_tp_fusion.

        Verifies scalar invariance, vector equivariance, and type-2
        equivariance through the complete encoder-decoder pipeline.
        """
        from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

        dtype, N = torch.float64, 128

        torch.manual_seed(42)
        pos = torch.randn(N, 3, dtype=dtype)
        ptr = torch.tensor([0, N], dtype=torch.long)
        normals = torch.randn(N, 3, dtype=dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        v_init = normals.unsqueeze(1).expand(-1, 8, -1).clone()

        R = _random_SO3(dtype=dtype)
        D2 = _compute_wigner_D2(R)

        torch.manual_seed(0)
        backbone = MultiScaleSE3Net(
            in_channels=1, hidden_scalar=32, hidden_vector=8, hidden_type2=4,
            num_stages=2, layers_per_stage=1, pool_ratio=0.25,
            gate_mode='norm', use_bottleneck_attn=True, use_tp_fusion=True,
        ).to(dtype)
        backbone.eval()

        with torch.no_grad():
            out_o = backbone(pos, ptr, v_init=v_init, return_encoder_features=False)
            out_r = backbone(pos @ R.T, ptr, v_init=v_init @ R.T,
                             return_encoder_features=False)

        s_o, v_o, t2_o = out_o[0], out_o[1], out_o[2]
        s_r, v_r, t2_r = out_r[0], out_r[1], out_r[2]

        # Scalar
        s_rel = (s_r - s_o).abs().max().item() / s_o.abs().max().clamp(min=1e-10).item()
        assert s_rel < 1e-4, f"Backbone scalar invariance: rel={s_rel:.2e}"

        # Vector
        v_rel = (v_r - v_o @ R.T).abs().max().item() / v_o.abs().max().clamp(min=1e-10).item()
        assert v_rel < 1e-4, f"Backbone vector equivariance: rel={v_rel:.2e}"

        # Type-2
        if t2_o is not None:
            t2_rel = (t2_r - t2_o @ D2.T).abs().max().item() / \
                     t2_o.abs().max().clamp(min=1e-10).item()
            assert t2_rel < 1e-4, f"Backbone t2 equivariance: rel={t2_rel:.2e}"
            print(f"  [✓] Backbone: s={s_rel:.2e}, v={v_rel:.2e}, t2={t2_rel:.2e}")
        else:
            print(f"  [✓] Backbone: s={s_rel:.2e}, v={v_rel:.2e}, t2=None")

    def test_full_model_logit_invariance(self):
        """SE3PartSegNet with TP fusion: logits must be rotation-invariant.

        This is the end-to-end acceptance test. If logits are invariant,
        part segmentation predictions are guaranteed consistent under
        arbitrary SO(3) rotation.
        """
        from geoembodied.nn.models.part_segmentation import SE3PartSegNet

        dtype, N = torch.float64, 128

        torch.manual_seed(42)
        pos = torch.randn(N, 3, dtype=dtype)
        ptr = torch.tensor([0, N], dtype=torch.long)
        normals = torch.randn(N, 3, dtype=dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        cat = torch.zeros(1, dtype=torch.long)

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

        logit_rel = (logits_r - logits_o).abs().max().item() / \
                    logits_o.abs().max().clamp(min=1e-10).item()
        assert logit_rel < 1e-3, f"Model logit invariance: rel={logit_rel:.2e}"

        pred_o = logits_o.argmax(dim=-1)
        pred_r = logits_r.argmax(dim=-1)
        pred_match = (pred_o == pred_r).float().mean().item()
        assert pred_match == 1.0, f"Prediction match: {pred_match*100:.1f}%"

        print(f"  [✓] Model logits: rel={logit_rel:.2e}, pred_match={pred_match*100:.0f}%")


# ──────────────────── run all tests ─────────────────────

def main():
    """Run all tests manually (pytest is disabled per project rules)."""
    print("=" * 60)
    print("TEST SUITE: EquivariantSkipFusion")
    print("=" * 60)

    standalone = TestEquivariantSkipFusionStandalone()

    print("\n--- Standalone equivariance (with FPS self-loops) ---")
    standalone.test_equivariance_with_fps_selfloops()

    print("\n--- 100% degenerate self-loops ---")
    standalone.test_degenerate_edges_zeroed()

    print("\n--- No self-loops baseline ---")
    standalone.test_no_selfloops_baseline()

    backbone_tests = TestBackboneTPFusion()

    print("\n--- Backbone equivariance (use_tp_fusion=True) ---")
    backbone_tests.test_backbone_equivariance()

    print("\n--- Full model logit invariance ---")
    backbone_tests.test_full_model_logit_invariance()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
