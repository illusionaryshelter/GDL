"""Verify: disabling autocast inside _compute_messages fixes the overflow.

The fix wraps the message computation in autocast(enabled=False) so that
all matmul/@ operations run in genuine FP32, both forward and backward.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def test_fix_with_autocast_disabled():
    device = 'cuda'
    
    print("="*60)
    print("Test A: matmul INSIDE autocast (BROKEN)")
    print("="*60)
    
    for scale_val in [512, 1024, 4096, 16384]:
        a = torch.randn(200, 96, device=device, dtype=torch.float32, requires_grad=True)
        w = torch.randn(96, 96, device=device, dtype=torch.float32, requires_grad=True)
        
        with autocast('cuda', enabled=True):
            out = a @ w  # autocast forces FP16
            loss = out.sum()
        
        (loss * scale_val).backward()
        a_inf = torch.isinf(a.grad).any().item()
        w_inf = torch.isinf(w.grad).any().item()
        print(f"  scale={scale_val:6d}: a.grad inf={a_inf}, w.grad inf={w_inf}, "
              f"a.grad max={a.grad[torch.isfinite(a.grad)].abs().max():.1f}")
    
    print()
    print("="*60)
    print("Test B: matmul with autocast(enabled=False) (FIXED)")
    print("="*60)
    
    for scale_val in [512, 1024, 4096, 16384]:
        a = torch.randn(200, 96, device=device, dtype=torch.float32, requires_grad=True)
        w = torch.randn(96, 96, device=device, dtype=torch.float32, requires_grad=True)
        
        with autocast('cuda', enabled=True):
            # This is what we should do inside _compute_messages
            with autocast('cuda', enabled=False):
                out = a @ w  # genuine FP32!
            loss = out.sum()
        
        (loss * scale_val).backward()
        a_inf = torch.isinf(a.grad).any().item()
        w_inf = torch.isinf(w.grad).any().item()
        print(f"  scale={scale_val:6d}: a.grad inf={a_inf}, w.grad inf={w_inf}, "
              f"a.grad max={a.grad[torch.isfinite(a.grad)].abs().max():.1f}")

    print()
    print("="*60)
    print("Test C: Full model training loop comparison")
    print("="*60)
    
    from examples.shapenet_seg.model import SE3PartSegNet
    from examples.shapenet_seg.dataset import CATEGORY_PART_MASK
    
    # Test 1: Current (broken) — autocast wraps everything
    model = SE3PartSegNet(hidden_scalar=96, hidden_vector=24, 
                          num_stages=3, layers_per_stage=1).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scaler = GradScaler('cuda', init_scale=2**14)
    
    torch.manual_seed(42)
    overflow_count = 0
    for step in range(20):
        pos = torch.randn(200, 3, device=device)
        ptr = torch.tensor([0, 100, 200], dtype=torch.int64, device=device)
        cat = torch.tensor([0, 1], device=device)
        normals = torch.randn(200, 3, device=device)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        valid_parts_0 = CATEGORY_PART_MASK[0].nonzero().squeeze(-1).to(device)
        valid_parts_1 = CATEGORY_PART_MASK[1].nonzero().squeeze(-1).to(device)
        labels = torch.cat([
            valid_parts_0[torch.randint(0, len(valid_parts_0), (100,), device=device)],
            valid_parts_1[torch.randint(0, len(valid_parts_1), (100,), device=device)],
        ])
        
        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(pos, ptr, cat, normals=normals)
            loss = F.cross_entropy(logits, labels)
        
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        
        if scale_after < scale_before:
            overflow_count += 1
    
    print(f"  Current _compute_messages (no autocast guard):")
    print(f"    Overflow events in 20 steps: {overflow_count}")
    print(f"    Final scale: {scaler.get_scale()}")
    

if __name__ == '__main__':
    test_fix_with_autocast_disabled()
