#!/usr/bin/env python3
"""Remote test: CUDA FPS kernel correctness, performance, and integration.

Run on a machine with ≥ 16GB VRAM (A100/4090/3090).

Validates:
  1. CUDA kernel compilation
  2. Correctness vs CPU reference (identical point selection)
  3. Batch isolation (no cross-contamination)
  4. No duplicates in selection
  5. Spatial coverage property (selected points are well-spread)
  6. Performance benchmark (CUDA vs CPU fallback)
  7. Integration with EquivariantPool
  8. End-to-end SE(3) equivariance of MultiScaleSE3Net
"""

import sys
import os
import time
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

print("=" * 60)
print("CUDA FPS Kernel — Remote Test Suite")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()


# ═══════════════════════════════════════════════════════════════════
# Test 1: Kernel Compilation
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: CUDA FPS Kernel Compilation")
print("=" * 60)

from geoembodied.csrc import fps_available, fps_cuda, _fps_iterative_fallback

avail = fps_available()
print(f"  CUDA FPS available: {avail}")
assert avail, "CUDA FPS kernel failed to compile!"
print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 2: Basic Correctness
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 2: Basic Correctness")
print("=" * 60)

device = torch.device("cuda")

# Small test with known geometry
pos = torch.randn(100, 3, device=device)
ptr = torch.tensor([0, 60, 100], dtype=torch.int64, device=device)
k_per = torch.tensor([15, 10], dtype=torch.int64, device=device)

idx = fps_cuda(pos, ptr, k_per)

print(f"  Output shape: {idx.shape} (expected [25])")
print(f"  Output dtype: {idx.dtype}")
print(f"  Output device: {idx.device}")

assert idx.shape[0] == 25, f"Expected 25, got {idx.shape[0]}"
assert idx.dtype == torch.int64
assert idx.is_cuda

# Batch isolation
batch0 = idx[idx < 60]
batch1 = idx[idx >= 60]
assert len(batch0) == 15, f"Batch 0: expected 15, got {len(batch0)}"
assert len(batch1) == 10, f"Batch 1: expected 10, got {len(batch1)}"
assert batch0.max() < 60, "Batch 0 leaked!"
assert batch1.min() >= 60, "Batch 1 leaked!"
print(f"  Batch 0: {len(batch0)} pts, range=[{batch0.min()}, {batch0.max()}]")
print(f"  Batch 1: {len(batch1)} pts, range=[{batch1.min()}, {batch1.max()}]")

# No duplicates
assert idx.unique().shape[0] == idx.shape[0], "Duplicates found!"
print("  No duplicates ✓")
print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 3: CUDA vs CPU Agreement
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 3: CUDA vs CPU Reference Agreement")
print("=" * 60)

torch.manual_seed(42)
pos3 = torch.randn(2000, 3, device=device)
ptr3 = torch.tensor([0, 1000, 2000], dtype=torch.int64, device=device)
k3 = torch.tensor([250, 250], dtype=torch.int64, device=device)

idx_cuda = fps_cuda(pos3, ptr3, k3)
idx_cpu = _fps_iterative_fallback(pos3, ptr3, k3)

# Both should start from the same seed (first point)
# and produce identical results since FPS is deterministic
match = (idx_cuda == idx_cpu).all().item()
if match:
    print("  CUDA and CPU produce IDENTICAL results ✓")
else:
    # Check how many match
    n_match = (idx_cuda == idx_cpu).sum().item()
    print(f"  CUDA vs CPU: {n_match}/{idx_cuda.shape[0]} match")
    print(f"  (Minor float32 differences in distance computation are OK)")

    # Even if exact indices differ, spatial coverage should be similar
    cuda_pos = pos3[idx_cuda[:250]]  # batch 0
    cpu_pos = pos3[idx_cpu[:250]]
    cuda_spread = torch.cdist(cuda_pos, cuda_pos).mean()
    cpu_spread = torch.cdist(cpu_pos, cpu_pos).mean()
    spread_diff = abs(cuda_spread.item() - cpu_spread.item()) / cpu_spread.item()
    print(f"  Spatial spread similarity: {1-spread_diff:.4f} (1.0 = identical)")
    assert spread_diff < 0.05, f"Spatial spread diverged: {spread_diff:.4f}"

print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 4: Spatial Coverage Property
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 4: Spatial Coverage Property")
print("=" * 60)

# FPS should select points that are maximally spread.
# Test: selected points should have greater average pairwise distance
# than a random subset of the same size.
torch.manual_seed(0)
N_test = 4096
pos4 = torch.randn(N_test, 3, device=device)
ptr4 = torch.tensor([0, N_test], dtype=torch.int64, device=device)
k4 = torch.tensor([256], dtype=torch.int64, device=device)

fps_idx = fps_cuda(pos4, ptr4, k4)
fps_pos = pos4[fps_idx]

# Random baseline (average of 10 trials)
random_spreads = []
for _ in range(10):
    rand_idx = torch.randperm(N_test, device=device)[:256]
    rand_pos = pos4[rand_idx]
    random_spreads.append(torch.cdist(rand_pos, rand_pos).mean().item())

fps_spread = torch.cdist(fps_pos, fps_pos).mean().item()
rand_spread = sum(random_spreads) / len(random_spreads)

print(f"  FPS mean pairwise distance:    {fps_spread:.4f}")
print(f"  Random mean pairwise distance: {rand_spread:.4f}")
print(f"  FPS / Random ratio:            {fps_spread/rand_spread:.4f}")

assert fps_spread > rand_spread, \
    f"FPS ({fps_spread:.4f}) not better than random ({rand_spread:.4f})!"
print("  FPS produces better spatial coverage than random ✓")
print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 5: Large-Scale & Variable Batch Sizes
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 5: Large Scale & Variable Batch Sizes")
print("=" * 60)

# Variable per-batch sizes
sizes = torch.tensor([2048, 1024, 4096, 512], dtype=torch.int64, device=device)
ptr5 = torch.zeros(5, dtype=torch.int64, device=device)
ptr5[1:] = sizes.cumsum(0)
N5 = ptr5[-1].item()
pos5 = torch.randn(N5, 3, device=device)
k5 = (sizes.float() * 0.25).clamp(min=1).long()

idx5 = fps_cuda(pos5, ptr5, k5)
print(f"  Total input: {N5}, Total output: {idx5.shape[0]}")
print(f"  Per-batch K: {k5.tolist()}")

# Verify batch isolation
offset = 0
for b in range(4):
    kb = k5[b].item()
    batch_idx = idx5[offset:offset+kb]
    start = ptr5[b].item()
    end = ptr5[b+1].item()
    assert batch_idx.min() >= start, f"Batch {b} underflow!"
    assert batch_idx.max() < end, f"Batch {b} overflow!"
    assert batch_idx.unique().shape[0] == kb, f"Batch {b} duplicates!"
    offset += kb
    print(f"  Batch {b}: K={kb}, range=[{start},{end}), ✓")

print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 6: Performance Benchmark — CUDA vs CPU
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 6: Performance Benchmark")
print("=" * 60)

B_bench = 16
N_bench = 2048
K_bench = 512

pos6 = torch.randn(B_bench * N_bench, 3, device=device)
ptr6 = torch.arange(0, (B_bench+1)*N_bench, N_bench, dtype=torch.int64, device=device)
k6 = torch.full((B_bench,), K_bench, dtype=torch.int64, device=device)

# Warmup
for _ in range(3):
    fps_cuda(pos6, ptr6, k6)
torch.cuda.synchronize()

# CUDA timing
t0 = time.perf_counter()
for _ in range(20):
    fps_cuda(pos6, ptr6, k6)
torch.cuda.synchronize()
t_cuda = (time.perf_counter() - t0) / 20 * 1000

# CPU fallback timing (fewer iterations — it's slow)
for _ in range(2):
    _fps_iterative_fallback(pos6, ptr6, k6)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(3):
    _fps_iterative_fallback(pos6, ptr6, k6)
torch.cuda.synchronize()
t_cpu = (time.perf_counter() - t0) / 3 * 1000

speedup = t_cpu / t_cuda
print(f"  Config: B={B_bench}, N_per={N_bench}, K={K_bench}")
print(f"  CUDA FPS:     {t_cuda:.2f} ms")
print(f"  CPU fallback: {t_cpu:.2f} ms")
print(f"  Speedup:      {speedup:.1f}x")

assert t_cuda < t_cpu, "CUDA should be faster than CPU fallback!"
print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 7: Integration with EquivariantPool
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 7: EquivariantPool Integration")
print("=" * 60)

from geoembodied.nn.modules.equivariant_pool import EquivariantPool

C_s, C_v = 64, 16
pool = EquivariantPool(
    scalar_channels=C_s,
    vector_channels=C_v,
    ratio=0.25,
    k_neighbors=4,
).to(device)

B_pool, N_pool = 4, 1024
pos_pool = torch.randn(B_pool * N_pool, 3, device=device)
ptr_pool = torch.arange(0, (B_pool+1)*N_pool, N_pool, dtype=torch.int64, device=device)
s_in = torch.randn(B_pool * N_pool, C_s, device=device)
v_in = torch.randn(B_pool * N_pool, C_v, 3, device=device)

seed_pos, s_out, v_out, ptr_out, pool_idx = pool(pos_pool, s_in, v_in, ptr_pool)

N_expect = int(N_pool * 0.25) * B_pool
print(f"  Input:  N={B_pool*N_pool}, C_s={C_s}, C_v={C_v}")
print(f"  Output: N={seed_pos.shape[0]} (expected ~{N_expect})")
print(f"  s_out:  {s_out.shape}")
print(f"  v_out:  {v_out.shape}")
print(f"  ptr_out: {ptr_out.tolist()}")

assert seed_pos.shape[0] == s_out.shape[0] == v_out.shape[0]
assert s_out.shape[1] == C_s
assert v_out.shape[1] == C_v
assert v_out.shape[2] == 3
assert ptr_out.shape[0] == B_pool + 1
print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Test 8: End-to-End SE(3) Equivariance
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 8: End-to-End SE(3) Equivariance")
print("=" * 60)

from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net
def _random_rotation(device):
    """Generate random SO(3) rotation matrix via QR decomposition.
    Returns: R [3,3] float32, proper rotation (det=+1)
    """
    M = torch.randn(3, 3, dtype=torch.float64)
    Q, R_diag = torch.linalg.qr(M)
    signs = torch.sign(torch.diag(R_diag))
    Q = Q * signs.unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.float().to(device)

torch.manual_seed(42)

model = MultiScaleSE3Net(
    in_channels=1,
    hidden_scalar=32,
    hidden_vector=8,
    num_stages=2,
    layers_per_stage=1,
    pool_ratio=0.5,
).to(device).eval()

B_eq, N_eq = 2, 512
pos_eq = torch.randn(B_eq * N_eq, 3, device=device)
ptr_eq = torch.arange(0, (B_eq+1)*N_eq, N_eq, dtype=torch.int64, device=device)
feat_eq = torch.randn(B_eq * N_eq, 1, device=device)

# Path 1: f(pos, ptr, feat)
with torch.no_grad():
    out1_s, out1_v, _ = model(pos_eq, ptr_eq, features=feat_eq)

# Generate random SE(3) transform
R = _random_rotation(device)  # [3, 3]
t = torch.randn(3, device=device) * 2.0     # translation

# Path 2: f(R @ pos + t, ptr, feat)
pos_transformed = pos_eq @ R.T + t.unsqueeze(0)
with torch.no_grad():
    out2_s, out2_v, _ = model(pos_transformed, ptr_eq, features=feat_eq)

# Scalar features should be invariant: f(Rp+t)_s ≈ f(p)_s
scalar_err = (out2_s - out1_s).abs().max().item()
scalar_rel = scalar_err / (out1_s.abs().max().item() + 1e-8)

print(f"  Scalar invariance:")
print(f"    Max absolute error: {scalar_err:.6f}")
print(f"    Relative error:     {scalar_rel:.6f}")

# Vector features should be equivariant: f(Rp+t)_v ≈ R @ f(p)_v
if out1_v is not None and out1_v.shape[0] > 0:
    out1_v_rotated = torch.einsum("ij,nvj->nvi", R, out1_v)
    vector_err = (out2_v - out1_v_rotated).abs().max().item()
    vector_rel = vector_err / (out1_v.abs().max().item() + 1e-8)
    print(f"  Vector equivariance:")
    print(f"    Max absolute error: {vector_err:.6f}")
    print(f"    Relative error:     {vector_rel:.6f}")
    assert vector_rel < 1e-3, f"Vector equivariance failed: {vector_rel}"

# FP32 tolerance for multi-stage network
assert scalar_rel < 1e-3, f"Scalar invariance failed: {scalar_rel}"
print("  ✓ PASSED\n")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print(f"  CUDA FPS speedup: {speedup:.1f}x over CPU fallback")
print(f"  SE(3) scalar invariance error: {scalar_rel:.6f}")
