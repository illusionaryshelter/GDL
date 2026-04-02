"""Post-fix verification: amp_scale should stay at 2^16 (65536) forever.

With the model internally disabling autocast, ALL operations run in FP32.
FP32 gradients never overflow FP32 range, so GradScaler will never reduce scale.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def main():
    device = 'cuda'
    
    from examples.shapenet_seg.model import SE3PartSegNet
    from examples.shapenet_seg.dataset import CATEGORY_PART_MASK
    
    model = SE3PartSegNet(hidden_scalar=96, hidden_vector=24, 
                          num_stages=3, layers_per_stage=1).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scaler = GradScaler('cuda', init_scale=2**16)
    
    print("Post-fix amp_scale stability test: 100 steps")
    print("="*60)
    print(f"{'Step':>5} {'Loss':>8} {'Scale':>8} {'Overflow':>10}")
    print("-"*40)
    
    overflow_count = 0
    for step in range(100):
        n_per_shape = 200
        B = 2
        N = n_per_shape * B
        pos = torch.randn(N, 3, device=device)
        ptr = torch.arange(0, N+1, n_per_shape, dtype=torch.int64, device=device)
        
        cat = torch.randint(0, 16, (B,), device=device)
        normals = torch.randn(N, 3, device=device)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        labels_list = []
        for i in range(B):
            valid = CATEGORY_PART_MASK[cat[i].item()].nonzero().squeeze(-1).to(device)
            labels_list.append(valid[torch.randint(0, len(valid), (n_per_shape,), device=device)])
        labels = torch.cat(labels_list)
        
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
            status = "OVERFLOW ⚠"
        elif scale_after > scale_before:
            status = "GROW ↑"
        else:
            status = "OK"
        
        if step % 10 == 0 or scale_after != scale_before:
            print(f"{step:>5} {loss.item():>8.4f} {scale_after:>8.0f} {status:>10}")
    
    print("-"*40)
    print(f"Total overflows: {overflow_count}/100")
    print(f"Final scale: {scaler.get_scale():.0f}")
    
    if overflow_count == 0:
        print("\n✅ SUCCESS: amp_scale stable at 65536 — no FP16 overflow!")
    else:
        print(f"\n❌ FAIL: {overflow_count} overflows detected")

if __name__ == '__main__':
    main()
