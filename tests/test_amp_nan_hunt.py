import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from examples.shapenet_seg.model import SE3PartSegNet
from examples.shapenet_seg.dataset import CATEGORY_PART_MASK

def main():
    torch.manual_seed(42)
    device = torch.device('cuda')
    model = SE3PartSegNet(hidden_scalar=96, hidden_vector=24, num_stages=3, layers_per_stage=1).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    scaler = GradScaler('cuda', init_scale=32768)

    print('Hunting for AMP overflows on GPU...')
    
    for step in range(50):
        # Generate data with VALID labels on GPU
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
        
        with autocast('cuda', enabled=True):
            logits = model(pos, ptr, cat, normals=normals)
            loss = F.cross_entropy(logits, labels)
            
        scaler.scale(loss).backward()
        
        # Check for INF/NaN in gradients
        overflow = False
        max_grad = 0
        overflow_params = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad
                has_inf = torch.isinf(g).any().item()
                has_nan = torch.isnan(g).any().item()
                if has_inf or has_nan:
                    overflow = True
                    overflow_params.append((name, has_inf, has_nan))
                
                finite_g = g[torch.isfinite(g)]
                if len(finite_g) > 0:
                    max_grad = max(max_grad, finite_g.abs().max().item())
        
        # Check unscale behavior
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        
        if scale_after < scale_before:
            print(f'Step {step}: Scale DROPPED {scale_before} -> {scale_after}')
            print(f'  Loss: {loss.item():.4f}, Max finite grad: {max_grad:.2f}')
            if overflow:
                print(f'  Overflowed params: {overflow_params[:3]}')
        else:
            if step % 10 == 0:
                print(f'Step {step}: OK. Scale={scale_after}, Max grad={max_grad:.2f}, Loss={loss.item():.4f}')

if __name__ == '__main__':
    main()
