"""Test: trace gradient dtype and magnitude per-layer under autocast.

This reveals which layers produce FP16 gradients and how large they get.
Run on GPU to detect the actual FP16 backward paths.
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
    
    pos = torch.randn(200, 3, device=device)
    ptr = torch.tensor([0, 100, 200], dtype=torch.int64, device=device)
    cat = torch.tensor([0, 1], device=device)
    normals = torch.randn(200, 3, device=device)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    valid0 = CATEGORY_PART_MASK[0].nonzero().squeeze(-1).to(device)
    valid1 = CATEGORY_PART_MASK[1].nonzero().squeeze(-1).to(device)
    labels = torch.cat([
        valid0[torch.randint(0, len(valid0), (100,), device=device)],
        valid1[torch.randint(0, len(valid1), (100,), device=device)],
    ])
    
    # Forward with autocast
    with autocast('cuda'):
        logits = model(pos, ptr, cat, normals=normals)
        loss = F.cross_entropy(logits, labels)
    
    loss.backward()
    
    print("="*70)
    print("Per-parameter gradient dtype and magnitude")
    print("="*70)
    print(f"{'Parameter':<50} {'dtype':>8} {'max_abs':>12} {'overflow?':>10}")
    print("-"*80)
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad
            max_abs = g.abs().max().item() if torch.isfinite(g).any() else float('inf')
            overflow = "YES" if torch.isinf(g).any() or torch.isnan(g).any() else "no"
            # How much headroom before FP16 overflow at different scales?
            if g.dtype == torch.float16:
                headroom_512 = 65504 / (max_abs * 512) if max_abs > 0 else float('inf')
                headroom_16k = 65504 / (max_abs * 16384) if max_abs > 0 else float('inf')
                extra = f" | FP16! headroom@512={headroom_512:.1f}x @16K={headroom_16k:.2f}x"
            else:
                extra = ""
            print(f"  {name:<48} {str(g.dtype):>8} {max_abs:>12.4f} {overflow:>10}{extra}")

    # Also check intermediate tensor dtypes
    print()
    print("="*70)
    print("Intermediate tensor dtypes under autocast")
    print("="*70)
    
    # Hook into model to capture tensor dtypes
    dtypes = {}
    def make_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        dtypes[f"{layer_name}.out[{i}]"] = o.dtype
            elif isinstance(output, torch.Tensor):
                dtypes[f"{layer_name}.out"] = output.dtype
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        dtypes[f"{layer_name}.in[{i}]"] = inp.dtype
        return hook
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.BatchNorm1d)):
            handles.append(module.register_forward_hook(make_hook(name)))
    
    with autocast('cuda'):
        logits2 = model(pos, ptr, cat, normals=normals)
    
    for h in handles:
        h.remove()
    
    for name, dtype in sorted(dtypes.items()):
        marker = " ⚠ FP16" if dtype == torch.float16 else ""
        print(f"  {name:<50} {str(dtype)}{marker}")


if __name__ == '__main__':
    main()
