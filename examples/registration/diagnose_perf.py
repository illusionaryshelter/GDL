#!/usr/bin/env python3
"""Remote GPU diagnostic — identify WHERE the 253ms/step is spent.

Run on the remote machine:
    python examples/registration/diagnose_perf.py --cache data/modelnet40_4096.pt

This prints a detailed timing breakdown per forward/backward step.
"""
import time
import sys
import os
import torch

# Ensure examples/registration is in path
sys.path.insert(0, os.path.dirname(__file__))


def sync_and_time() -> float:
    """Get current time after GPU sync (accurate timing)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        vram = getattr(props, "total_memory", None) or 0
        print(f"  VRAM: {vram / 1e9:.1f} GB")

    # ══════════════════════════════════════════════════════════
    # 1. Check CUDA backend availability
    # ══════════════════════════════════════════════════════════
    print("\n=== Backend Availability ===")

    try:
        from geoembodied.csrc import get_radius_graph_backend
        rg_backend = get_radius_graph_backend()
    except Exception as e:
        rg_backend = f"FAILED: {e}"
    print(f"  radius_graph backend: {rg_backend}")

    try:
        from geoembodied.csrc import segment_reduce_available
        sr_avail = segment_reduce_available()
    except Exception as e:
        sr_avail = f"FAILED: {e}"
    print(f"  segment_reduce CUDA: {sr_avail}")

    try:
        from geoembodied.kernels.triton_sph_harm import spherical_harmonics as sh
        # Test triton SH
        test_dirs = torch.randn(10, 3, device=device)
        test_dirs = test_dirs / test_dirs.norm(dim=-1, keepdim=True)
        _ = sh(test_dirs, max_l=2, normalize=False)
        sh_backend = "triton (OK)"
    except Exception as e:
        sh_backend = f"FAILED: {e}"
    print(f"  spherical_harmonics: {sh_backend}")

    # ══════════════════════════════════════════════════════════
    # 2. Micro-benchmark each component
    # ══════════════════════════════════════════════════════════
    print("\n=== Component Micro-Benchmarks (B=8, N=716) ===")
    B, N = 8, 716
    N_total = B * N

    pos = torch.randn(N_total, 3, device=device)
    batch = torch.arange(B, device=device).repeat_interleave(N)

    # 2a. SpatialGraph.build
    from geoembodied.nn.spatial_graph import SpatialGraph

    # Warmup
    for _ in range(3):
        g = SpatialGraph.build(pos, 0.3, max_num_neighbors=32,
                               batch=batch, num_batch_elements=B)

    t0 = sync_and_time()
    for _ in range(10):
        g = SpatialGraph.build(pos, 0.3, max_num_neighbors=32,
                               batch=batch, num_batch_elements=B)
    t1 = sync_and_time()
    graph_ms = (t1 - t0) / 10 * 1000
    print(f"  SpatialGraph.build: {graph_ms:.1f}ms  (E={g.E} edges)")

    # 2b. Sub-components of graph build
    from geoembodied.functional.radius_graph import radius_graph, compute_edge_vectors
    from geoembodied.functional.segment_reduce import sort_and_build_csr
    from geoembodied.kernels.triton_sph_harm import spherical_harmonics

    # radius_graph
    for _ in range(3):
        row, col = radius_graph(pos, 0.3, max_num_neighbors=32,
                                batch=batch, num_batch_elements=B)
    t0 = sync_and_time()
    for _ in range(10):
        row, col = radius_graph(pos, 0.3, max_num_neighbors=32,
                                batch=batch, num_batch_elements=B)
    t1 = sync_and_time()
    print(f"    └ radius_graph: {(t1-t0)/10*1000:.1f}ms")

    # compute_edge_vectors
    t0 = sync_and_time()
    for _ in range(10):
        diff, dist, direction = compute_edge_vectors(pos, row, col)
    t1 = sync_and_time()
    print(f"    └ compute_edge_vectors: {(t1-t0)/10*1000:.1f}ms")

    # spherical_harmonics
    t0 = sync_and_time()
    for _ in range(10):
        Y = spherical_harmonics(direction, max_l=2, normalize=False)
    t1 = sync_and_time()
    print(f"    └ spherical_harmonics: {(t1-t0)/10*1000:.1f}ms")

    # sort_and_build_csr
    t0 = sync_and_time()
    for _ in range(10):
        sort_and_build_csr(col, N_total)
    t1 = sync_and_time()
    print(f"    └ sort_and_build_csr: {(t1-t0)/10*1000:.1f}ms")

    # 2c. SE3Net forward (full backbone)
    from geoembodied.nn import SE3Net
    from geoembodied.functional.numeric_safe import lie_group_precision

    backbone = SE3Net(
        hidden_scalar=64, hidden_vector=16, num_layers=4, radius=0.3,
        max_num_neighbors=32,
    ).to(device)

    for _ in range(3):
        with lie_group_precision():
            s, v = backbone(pos, batch=batch, num_batch_elements=B)

    t0 = sync_and_time()
    for _ in range(5):
        with lie_group_precision():
            s, v = backbone(pos, batch=batch, num_batch_elements=B)
    t1 = sync_and_time()
    backbone_ms = (t1 - t0) / 5 * 1000
    print(f"\n  SE3Net forward (4 layers): {backbone_ms:.1f}ms")
    print(f"    └ graph_build: ~{graph_ms:.0f}ms"
          f"  neural_compute: ~{backbone_ms-graph_ms:.0f}ms")

    # 2d. Full registration model forward
    from geoembodied.nn import GeoRegistrationModel
    from geoembodied.data.batch import collate_point_clouds

    model = GeoRegistrationModel(
        backbone=SE3Net(
            hidden_scalar=64, hidden_vector=16, num_layers=4,
            radius=0.3, max_num_neighbors=32,
        ),
        cross_attention_layers=2,
        descriptor_dim=64,
        sinkhorn_iters=20,
    ).to(device)

    clouds_s = [{"pos": torch.randn(N, 3)} for _ in range(B)]
    clouds_t = [{"pos": torch.randn(N, 3)} for _ in range(B)]
    batch_s = collate_point_clouds(clouds_s).to(device)
    batch_t = collate_point_clouds(clouds_t).to(device)

    # Warmup
    for _ in range(3):
        with lie_group_precision():
            out = model(batch_s, batch_t)

    t0 = sync_and_time()
    for _ in range(5):
        with lie_group_precision():
            out = model(batch_s, batch_t)
    t1 = sync_and_time()
    fwd_ms = (t1 - t0) / 5 * 1000
    print(f"\n  Full model forward: {fwd_ms:.1f}ms")
    print(f"    └ 2x backbone: ~{2*backbone_ms:.0f}ms"
          f"  cross_attn+sinkhorn+SVD: ~{fwd_ms-2*backbone_ms:.0f}ms")

    # 2e. Backward
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        with lie_group_precision():
            out = model(batch_s, batch_t)
        loss = out["R"].sum() + out["t"].sum()
        loss.backward()

    t0 = sync_and_time()
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        with lie_group_precision():
            out = model(batch_s, batch_t)
        loss = out["R"].sum() + out["t"].sum()
        loss.backward()
    t1 = sync_and_time()
    fwd_bwd_ms = (t1 - t0) / 5 * 1000
    bwd_ms = fwd_bwd_ms - fwd_ms
    print(f"\n  Backward: {bwd_ms:.1f}ms")
    print(f"  Total fwd+bwd: {fwd_bwd_ms:.1f}ms")

    # ══════════════════════════════════════════════════════════
    # 3. Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SpatialGraph.build (×2): {2*graph_ms:.0f}ms "
          f"({2*graph_ms/fwd_bwd_ms*100:.0f}% of step)")
    print(f"  Neural network compute:  {fwd_bwd_ms-2*graph_ms:.0f}ms "
          f"({(fwd_bwd_ms-2*graph_ms)/fwd_bwd_ms*100:.0f}% of step)")
    print(f"  Total step:              {fwd_bwd_ms:.0f}ms")

    if 2 * graph_ms / fwd_bwd_ms > 0.5:
        print("\n  ⚠ BOTTLENECK: graph construction (>50% of step)")
        print("  Fix: pre-compute graphs or optimize radius_graph")
    elif bwd_ms / fwd_bwd_ms > 0.6:
        print("\n  ⚠ BOTTLENECK: backward pass (>60% of step)")
        print("  Check: SVD backward, Sinkhorn backward")
    else:
        print("\n  ✓ Balanced forward/backward")

    if rg_backend != "cuda":
        print(f"\n  ⚠ radius_graph NOT using CUDA: {rg_backend}")
        print("  Fix: compile CUDA extension with 'python setup.py build_ext'")

    if sr_avail is not True:
        print(f"\n  ⚠ segment_reduce NOT using CUDA: {sr_avail}")
        print("  The Python for-loop fallback is EXTREMELY SLOW!")


if __name__ == "__main__":
    main()
