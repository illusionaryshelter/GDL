"""Extended test: 200 steps with real-sized batches to reproduce overflow."""
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
    scaler = GradScaler('cuda', init_scale=2**14)
    
    print("Extended overflow test: 200 steps, large batch")
    print("="*60)
    
    overflow_count = 0
    for step in range(200):
        # Larger batch: 4 shapes × ~256 pts each ≈ real training batch
        n_per_shape = 256
        B = 4
        N = n_per_shape * B
        pos = torch.randn(N, 3, device=device)
        ptr = torch.arange(0, N+1, n_per_shape, dtype=torch.int64, device=device)
        
        # Random categories  
        cat = torch.randint(0, 16, (B,), device=device)
        normals = torch.randn(N, 3, device=device)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Valid labels for each category
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
            # Find which params overflowed
            inf_params = []
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isinf(p.grad).any():
                    inf_params.append(name)
            print(f"  Step {step}: OVERFLOW scale {scale_before:.0f} -> {scale_after:.0f}")
            if inf_params:
                print(f"    Inf params: {inf_params[:5]}")
        elif step % 50 == 0:
            print(f"  Step {step}: OK scale={scale_after:.0f} loss={loss.item():.4f}")
    
    print(f"\nTotal overflows: {overflow_count}/200")
    print(f"Final scale: {scaler.get_scale():.0f}")

if __name__ == '__main__':
    main()
