#!/usr/bin/env python3
"""Memory & Receptive Field Check — Run on REMOTE GPU.

Tests:
  1. OOM limit: sweep B from 4 to 32, find max batch size
  2. Memory leak detection (with warmup)
  3. Receptive field connectivity: perturbation at node 0 → bottleneck
"""
import torch
import gc
from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net

device = torch.device('cuda')

# ══════════════════════════════════════════════════════════
# Test A: OOM Limit Sweep
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Test A: OOM Limit Sweep (B × 2048 points)")
print("=" * 60)

N_per = 2048
max_safe_b = 0

for B in [4, 8, 12, 16, 20, 24, 28, 32]:
    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=64, hidden_vector=16,
        num_stages=3, layers_per_stage=2,
        pool_ratio=0.25,
    ).to(device)
    model.train()

    N = B * N_per
    ptr = torch.tensor(
        [i * N_per for i in range(B + 1)], dtype=torch.int64, device=device
    )
    pos = torch.randn(N, 3, device=device, requires_grad=True)

    try:
        torch.cuda.reset_peak_memory_stats()
        s, v, _ = model(pos, ptr)
        loss = s.mean() + v.mean()
        loss.backward()

        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  B={B:3d}: ✓ peak={peak:.2f}GB")
        max_safe_b = B

        del s, v, loss, pos, model
        gc.collect()
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  B={B:3d}: ✗ OOM")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            break
        raise

print(f"\n→ Max safe batch size: B={max_safe_b} (N={max_safe_b * N_per})")

# ══════════════════════════════════════════════════════════
# Test B: Memory Leak Detection (with warmup)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test B: Memory Leak Detection")
print("=" * 60)

B = min(max_safe_b, 8)
N = B * N_per
model = MultiScaleSE3Net(
    in_channels=1,
    hidden_scalar=64, hidden_vector=16,
    num_stages=3, layers_per_stage=2,
    pool_ratio=0.25,
).to(device)
model.train()

ptr = torch.tensor(
    [i * N_per for i in range(B + 1)], dtype=torch.int64, device=device
)
pos = torch.randn(N, 3, device=device)

# Warmup cycle
pos_w = pos.clone().requires_grad_(True)
s_w, v_w, _ = model(pos_w, ptr)
(s_w.mean() + v_w.mean()).backward()
del s_w, v_w, pos_w
gc.collect()
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
baseline = torch.cuda.memory_allocated()
print(f"  Post-warmup baseline: {baseline / 1e6:.1f}MB")

for cycle in range(5):
    pos_g = pos.clone().requires_grad_(True)
    s, v, _ = model(pos_g, ptr)
    loss = s.mean() + v.mean()
    loss.backward()

    del s, v, loss, pos_g
    gc.collect()
    torch.cuda.empty_cache()

    after = torch.cuda.memory_allocated()
    leak = (after - baseline) / 1e6
    status = "✓" if leak < 5 else "✗"
    print(f"  Cycle {cycle}: {status} delta={leak:.1f}MB")
    assert leak < 5, f"Memory leak: {leak:.1f}MB after cycle {cycle}"

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  Peak VRAM: {peak:.2f}GB")
print("Test B PASSED ✓")

# ══════════════════════════════════════════════════════════
# Test C: Receptive Field Connectivity
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test C: Receptive Field Connectivity")
print("=" * 60)

model.eval()
N_rf = 512
pos_rf = torch.randn(N_rf, 3, device=device)
ptr_rf = torch.tensor([0, N_rf], dtype=torch.int64, device=device)

# Normal forward
with torch.no_grad():
    s_normal, _, _ = model(pos_rf, ptr_rf)

# Perturbed forward: add large perturbation to node 0
pos_perturbed = pos_rf.clone()
pos_perturbed[0] += 100.0  # massive displacement

with torch.no_grad():
    s_perturbed, _, _ = model(pos_perturbed, ptr_rf)

# Check: how far does the perturbation reach?
diff = (s_perturbed - s_normal).abs().max(dim=1).values  # [N]
affected = (diff > 1e-6).sum().item()
max_dist_affected = 0.0
if affected > 1:
    affected_idx = (diff > 1e-6).nonzero(as_tuple=True)[0]
    dists_from_0 = (pos_rf[affected_idx] - pos_rf[0]).norm(dim=1)
    max_dist_affected = dists_from_0.max().item()

print(f"  Nodes affected by perturbation at node 0: {affected}/{N_rf}")
print(f"  Farthest affected node distance: {max_dist_affected:.2f}")
print(f"  Receptive field coverage: {affected/N_rf*100:.1f}%")

# For a well-connected U-Net, perturbation should reach most nodes
if affected > N_rf * 0.1:
    print(f"  ✓ Good receptive field (>{N_rf*0.1:.0f} nodes affected)")
else:
    print(f"  ⚠ Limited receptive field — check adaptive radius")

print("\nAll remote tests complete.")
