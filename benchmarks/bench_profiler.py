#!/usr/bin/env python3
"""Compute Bottleneck Analysis — CUDA Event-based Profiling.

Run on REMOTE GPU server: python3 benchmarks/bench_profiler.py

WHY NOT torch.profiler?
  torch.profiler uses CUPTI which:
  1. Serializes ALL CUDA kernels (kills async execution → GPU util=0%)
  2. Allocates massive CPU-side metadata (GB of event records → OOM)
  3. Known PyTorch issue: pytorch/pytorch#144455

Instead, we use torch.cuda.Event for section-level timing and
torch.cuda.synchronize() for accurate wall-clock measurement.
This gives REAL-WORLD performance numbers, not profiler-distorted ones.

Reports:
  - Total forward+backward wall time
  - Per-section timing (graph construction, SE3Conv, pool, interp)
  - Throughput (points/second)
  - VRAM peak
"""
import torch
import gc
import time
from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

device = torch.device('cuda')


def cuda_timer():
    """Create a CUDA event pair for async-safe timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def main():
    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=64, hidden_vector=16,
        num_stages=3, layers_per_stage=2,
        pool_ratio=0.25,
    ).to(device)
    model.train()

    B, N_per = 16, 2048
    N = B * N_per
    ptr = torch.tensor(
        [i * N_per for i in range(B + 1)], dtype=torch.int64, device=device
    )
    pos = torch.randn(N, 3, device=device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Input: B={B}, N_per={N_per}, N_total={N}")
    print()

    # ══════════════════════════════════════════════════════════
    # Warmup (3 iterations — JIT compile, CUDA context init)
    # ══════════════════════════════════════════════════════════
    print("Warming up (3 iterations)...")
    for _ in range(3):
        pos_g = pos.clone().requires_grad_(True)
        s, v, _ = model(pos_g, ptr)
        loss = s.mean() + v.mean()
        loss.backward()
        del s, v, loss, pos_g
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    # ══════════════════════════════════════════════════════════
    # Section A: End-to-End Throughput (10 iterations)
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section A: End-to-End Throughput")
    print("=" * 60)

    N_ITERS = 10
    torch.cuda.reset_peak_memory_stats()

    # Forward timing
    fwd_start, fwd_end = cuda_timer()
    torch.cuda.synchronize()
    fwd_start.record()
    for _ in range(N_ITERS):
        pos_g = pos.clone().requires_grad_(True)
        s, v, _ = model(pos_g, ptr)
        del s, v, pos_g
    fwd_end.record()
    torch.cuda.synchronize()
    fwd_ms = fwd_start.elapsed_time(fwd_end)

    # Forward + Backward timing
    fwd_bwd_start, fwd_bwd_end = cuda_timer()
    torch.cuda.synchronize()
    fwd_bwd_start.record()
    for _ in range(N_ITERS):
        pos_g = pos.clone().requires_grad_(True)
        s, v, _ = model(pos_g, ptr)
        loss = s.mean() + v.mean()
        loss.backward()
        del s, v, loss, pos_g
    fwd_bwd_end.record()
    torch.cuda.synchronize()
    fwd_bwd_ms = fwd_bwd_start.elapsed_time(fwd_bwd_end)

    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    fwd_per_iter = fwd_ms / N_ITERS
    fwd_bwd_per_iter = fwd_bwd_ms / N_ITERS
    bwd_per_iter = fwd_bwd_per_iter - fwd_per_iter
    throughput = N / (fwd_bwd_per_iter / 1000)  # points/sec

    print(f"  Forward:           {fwd_per_iter:8.2f} ms/iter")
    print(f"  Backward:          {bwd_per_iter:8.2f} ms/iter")
    print(f"  Forward+Backward:  {fwd_bwd_per_iter:8.2f} ms/iter")
    print(f"  Throughput:        {throughput:,.0f} points/sec")
    print(f"  Peak VRAM:         {peak_vram:.2f} GB")
    print()

    # ══════════════════════════════════════════════════════════
    # Section B: GPU vs CPU time analysis
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section B: GPU vs CPU timeline (single iteration)")
    print("=" * 60)

    # Wall-clock time on CPU side
    torch.cuda.synchronize()
    cpu_start = time.perf_counter()

    pos_g = pos.clone().requires_grad_(True)
    s, v, _ = model(pos_g, ptr)
    loss = s.mean() + v.mean()
    loss.backward()

    torch.cuda.synchronize()
    cpu_end = time.perf_counter()
    cpu_wall_ms = (cpu_end - cpu_start) * 1000

    # GPU time via CUDA events (same iteration)
    gpu_start, gpu_end = cuda_timer()
    torch.cuda.synchronize()
    gpu_start.record()
    pos_g2 = pos.clone().requires_grad_(True)
    s2, v2, _ = model(pos_g2, ptr)
    loss2 = s2.mean() + v2.mean()
    loss2.backward()
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_wall_ms = gpu_start.elapsed_time(gpu_end)

    overhead_pct = (cpu_wall_ms - gpu_wall_ms) / cpu_wall_ms * 100

    print(f"  CPU wall time: {cpu_wall_ms:.2f} ms")
    print(f"  GPU time:      {gpu_wall_ms:.2f} ms")
    print(f"  Python/sync overhead: {overhead_pct:.1f}%")
    if overhead_pct > 50:
        print(f"  ⚠ Python overhead dominates — GPU is idle {overhead_pct:.0f}% of time")
        print(f"    Root cause: graph construction + GPU→CPU syncs")
        print(f"    Mitigation: pre-compute graphs, cache radii, use CUDA Graphs")
    elif overhead_pct > 20:
        print(f"  ⚠ Moderate overhead — consider reducing GPU→CPU syncs")
    else:
        print(f"  ✓ Good GPU utilization")
    print()

    # ══════════════════════════════════════════════════════════
    # Section C: Scaling analysis
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Section C: Scaling — time vs batch size")
    print("=" * 60)
    print(f"  {'B':>4s}  {'N_total':>8s}  {'fwd ms':>8s}  {'fwd+bwd ms':>11s}  {'pt/s':>12s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*12}")

    for B_test in [1, 2, 4, 8, 16]:
        N_test = B_test * N_per
        ptr_test = torch.tensor(
            [i * N_per for i in range(B_test + 1)], dtype=torch.int64, device=device
        )
        pos_test = torch.randn(N_test, 3, device=device)

        # Warmup
        pos_g = pos_test.clone().requires_grad_(True)
        s, v, _ = model(pos_g, ptr_test)
        (s.mean() + v.mean()).backward()
        del s, v, pos_g
        torch.cuda.synchronize()

        # Measure
        t_start, t_end = cuda_timer()
        torch.cuda.synchronize()
        t_start.record()
        for _ in range(5):
            pos_g = pos_test.clone().requires_grad_(True)
            s, v, _ = model(pos_g, ptr_test)
            (s.mean() + v.mean()).backward()
            del s, v, pos_g
        t_end.record()
        torch.cuda.synchronize()

        ms = t_start.elapsed_time(t_end) / 5
        pts = N_test / (ms / 1000)

        # Fwd only
        f_start, f_end = cuda_timer()
        torch.cuda.synchronize()
        f_start.record()
        for _ in range(5):
            with torch.no_grad():
                s, v, _ = model(pos_test, ptr_test)
            del s, v
        f_end.record()
        torch.cuda.synchronize()
        fwd_ms_test = f_start.elapsed_time(f_end) / 5

        print(f"  {B_test:4d}  {N_test:8d}  {fwd_ms_test:8.2f}  {ms:11.2f}  {pts:12,.0f}")

    # ══════════════════════════════════════════════════════════
    # Section D: Component-level breakdown (single forward)
    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Section D: Component breakdown (B=16, forward only)")
    print("=" * 60)
    print("  Timing each component with torch.cuda.synchronize()...")
    print()

    model.eval()
    N = B * N_per
    pos_d = torch.randn(N, 3, device=device)
    counts = ptr[1:] - ptr[:-1]
    batch_d = torch.arange(B, device=device).repeat_interleave(counts)

    # ── Components ──
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool
    from geoembodied.nn.modules.equivariant_interp import EquivariantInterpolate
    from geoembodied.functional.knn import knn_self

    C_s = model.hidden_scalar
    C_v = model.hidden_vector

    features = pos_d.new_ones(N, 1)
    s = model.embed_scalar(features)  # [N, C_s]
    v = pos_d.new_zeros(N, C_v, 3)

    results = {}

    # -- (1) Adaptive radius estimation --
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for stage in model.encoder_stages:
        _ = stage._estimate_adaptive_radius(pos_d, ptr)
    torch.cuda.synchronize()
    t_radius = (time.perf_counter() - t0) * 1000
    results['1. Adaptive radius (5 stages)'] = t_radius

    # -- (2) Graph construction (radius_graph + KNN fallback + SH) --
    radius = model.encoder_stages[0]._estimate_adaptive_radius(pos_d, ptr)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    graph = model.encoder_stages[0]._build_hybrid_graph(pos_d, radius, ptr=ptr, batch=batch_d)
    torch.cuda.synchronize()
    t_graph_s0 = (time.perf_counter() - t0) * 1000
    results['2a. Graph build (Stage 0, N=32K)'] = t_graph_s0

    # -- (3) SE3Conv blocks (Stage 0) --
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        s_out, v_out = s.clone(), v.clone()
        for block in model.encoder_stages[0].blocks:
            s_out, v_out = block(s_out, v_out, graph)
    torch.cuda.synchronize()
    t_se3conv = (time.perf_counter() - t0) * 1000
    results['2b. SE3Conv blocks (Stage 0, 2 layers)'] = t_se3conv

    # -- (4) Pool layer --
    pool = model.pool_layers[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pos_p, s_p, v_p, ptr_p, fps_idx = pool(pos_d, s_out, v_out, ptr)
    torch.cuda.synchronize()
    t_pool = (time.perf_counter() - t0) * 1000
    N_pooled = pos_p.shape[0]
    results[f'3. Pool (N={N}→{N_pooled})'] = t_pool

    # -- (5) Graph build at reduced scale --
    batch_p = batch_d[fps_idx]
    radius_p = model.encoder_stages[1]._estimate_adaptive_radius(pos_p, ptr_p)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    graph_p = model.encoder_stages[1]._build_hybrid_graph(
        pos_p, radius_p, ptr=ptr_p, batch=batch_p
    )
    torch.cuda.synchronize()
    t_graph_s1 = (time.perf_counter() - t0) * 1000
    results[f'4. Graph build (Stage 1, N={N_pooled})'] = t_graph_s1

    # -- (6) Interp layer --
    interp = model.interp_layers[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        s_up, v_up = interp(pos_d, pos_p, s_p, v_p, ptr, ptr_p)
    torch.cuda.synchronize()
    t_interp = (time.perf_counter() - t0) * 1000
    results[f'5. Interp (N={N_pooled}→{N})'] = t_interp

    # -- (7) Isolated cost: cdist alone --
    x_3d = pos_d.reshape(B, N_per, 3)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist_sq = torch.cdist(x_3d, x_3d, p=2.0).pow(2)
    torch.cuda.synchronize()
    t_cdist = (time.perf_counter() - t0) * 1000
    cdist_mem = dist_sq.nelement() * 4 / 1e6
    del dist_sq
    results[f'6. cdist alone ([{B},{N_per},{N_per}])'] = t_cdist

    # -- (8) Isolated cost: KNN alone --
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    knn_self(pos_d, ptr, k=4)
    torch.cuda.synchronize()
    t_knn = (time.perf_counter() - t0) * 1000
    results[f'7. KNN alone (k=4, N={N})'] = t_knn

    # -- Print results --
    total = sum(results.values())
    print(f"  {'Component':<42s}  {'Time ms':>8s}  {'%':>5s}")
    print(f"  {'─'*42}  {'─'*8}  {'─'*5}")
    for name, ms in results.items():
        pct = ms / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {name:<42s}  {ms:8.2f}  {pct:4.1f}% {bar}")
    print(f"  {'─'*42}  {'─'*8}")
    print(f"  {'TOTAL':<42s}  {total:8.2f}")
    print(f"\n  cdist memory: {cdist_mem:.0f}MB per [B, N, N] matrix")

    # -- Diagnosis --
    graph_total = t_radius + t_graph_s0 + t_graph_s1
    compute_total = t_se3conv + t_pool + t_interp
    print(f"\n  Graph construction: {graph_total:.0f}ms ({graph_total/total*100:.0f}%)")
    print(f"  SE3Conv compute:   {compute_total:.0f}ms ({compute_total/total*100:.0f}%)")
    if graph_total > compute_total * 3:
        print(f"  ⚠ Graph construction is {graph_total/compute_total:.1f}x slower than compute!")
        print(f"    → O(N²) cdist is the bottleneck")
        print(f"    → Solution: replace cdist with CUDA-accelerated radius search")
        print(f"       (e.g., torch_cluster.radius, grid hashing, or Triton kernel)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

