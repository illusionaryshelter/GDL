#!/usr/bin/env python3
"""OOM Stress Test — run on remote GPU server.

Tests:
    1. No OOM on 24GB GPU with N=10000
    2. No memory leak across forward/backward cycles
    3. Peak VRAM within safe threshold

Note: A warmup cycle is required before measuring baseline because
the first forward pass allocates one-time CUDA infrastructure:
JIT kernel cache, cuDNN workspace, autograd metadata (~15-20MB).
This is expected PyTorch behavior, NOT a memory leak.
"""
import torch
import gc
from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

device = torch.device('cuda')
model = MultiScaleSE3Net(
    in_channels=1,
    hidden_scalar=64, hidden_vector=16,
    num_stages=3, layers_per_stage=2,
    pool_ratio=0.25,
).to(device)

N = 10000
ptr = torch.tensor([0, 5000, 10000], dtype=torch.int64, device=device)
pos = torch.randn(N, 3, device=device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Input: N={N}, B=2")

# ── Warmup cycle: allocates one-time CUDA infrastructure ──
print("Warmup cycle (allocating CUDA infrastructure)...")
pos_w = pos.clone().requires_grad_(True)
s_w, v_w, _ = model(pos_w, ptr)
loss_w = s_w.mean() + v_w.mean()
loss_w.backward()
del s_w, v_w, loss_w, pos_w
gc.collect()
torch.cuda.empty_cache()

# ── Baseline: measured AFTER warmup ──
torch.cuda.reset_peak_memory_stats()
baseline = torch.cuda.memory_allocated()
print(f"Post-warmup baseline: {baseline / 1e6:.1f}MB")

# ── 3 measurement cycles ──
for cycle in range(3):
    pos_g = pos.clone().requires_grad_(True)
    s, v, _ = model(pos_g, ptr)
    loss = s.mean() + v.mean()
    loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1e9
    current = torch.cuda.memory_allocated() / 1e9
    print(f"  Cycle {cycle}: peak={peak:.2f}GB, current={current:.2f}GB")

    del s, v, loss, pos_g
    gc.collect()
    torch.cuda.empty_cache()

    after = torch.cuda.memory_allocated()
    leak = (after - baseline) / 1e6
    # Allow 5MB tolerance for PyTorch internal caching jitter
    assert leak < 5, f"Memory leak: {leak:.1f}MB after cycle {cycle}"
    print(f"  ✓ No memory leak: delta={leak:.1f}MB")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPeak VRAM: {peak_gb:.2f}GB")
assert peak_gb < 12.0, f"Peak VRAM {peak_gb:.2f}GB exceeds 12GB safety threshold"
print("✓ Peak VRAM within 12GB safe threshold")
print("OOM STRESS TEST PASSED")
