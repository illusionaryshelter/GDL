"""Test: Does autocast force FP32 matmul back to FP16?

This is the smoking gun test for the persistent amp_scale decrease.
Under CUDA autocast, torch.mm/@ is on the FP16 list, meaning even
explicitly FP32 inputs get cast to FP16 for the matmul.

If SE3Conv's _compute_messages does `s_src @ w_ss.t()` inside an
active autocast region, it will run in FP16 despite explicit .float() casts.
"""
import torch
from torch.amp import autocast

def test_matmul_dtype_under_autocast():
    """Does autocast downcast explicitly-FP32 matmul to FP16?"""
    device = 'cuda'
    
    # Two FP32 matrices
    a = torch.randn(100, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 64, device=device, dtype=torch.float32)
    
    # Without autocast: FP32
    c_no_amp = a @ b
    print(f"No autocast:   a.dtype={a.dtype}, b.dtype={b.dtype}, (a@b).dtype={c_no_amp.dtype}")
    
    # With autocast: what happens?
    with autocast('cuda', enabled=True):
        c_amp = a @ b
        print(f"With autocast: a.dtype={a.dtype}, b.dtype={b.dtype}, (a@b).dtype={c_amp.dtype}")
        
        # What about torch.mm?
        c_mm = torch.mm(a, b)
        print(f"With autocast: torch.mm dtype={c_mm.dtype}")
        
        # What about explicit .float() inside autocast?
        a_f = a.float()  # already float32
        b_f = b.float()  # already float32
        c_ff = a_f @ b_f
        print(f"With autocast: .float() + @ dtype={c_ff.dtype}")
        
        # Does disabling autocast inside the region help?
        with autocast('cuda', enabled=False):
            c_disabled = a @ b
            print(f"autocast(False): (a@b).dtype={c_disabled.dtype}")
    
    assert c_amp.dtype == torch.float16, "autocast should downcast FP32 matmul to FP16"


def test_se3conv_internal_dtype():
    """Check what dtype SE3Conv actually computes in under autocast."""
    from geoembodied.nn.modules.se3_conv import _compute_messages
    
    device = 'cuda'
    E = 200  # edges
    C_s = 96
    C_v = 24
    
    # Simulate FP32 (explicitly cast) inputs inside _compute_messages
    s_src = torch.randn(E, C_s, device=device, dtype=torch.float32)
    v_src = torch.randn(E, C_v, 3, device=device, dtype=torch.float32)
    direction = torch.randn(E, 3, device=device, dtype=torch.float32)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    Y_0 = torch.ones(E, 1, device=device, dtype=torch.float32) * 0.282
    R = torch.randn(E, 5, device=device, dtype=torch.float32)
    
    w_ss = torch.randn(C_s, C_s, device=device, dtype=torch.float32)
    w_sv = torch.randn(C_v, C_s, device=device, dtype=torch.float32)
    
    # Without autocast
    s1, v1 = _compute_messages(
        s_src, v_src, direction, Y_0, R,
        w_ss, w_sv, None, None, None, C_s, C_v,
    )
    print(f"\nNo autocast: s_out.dtype={s1.dtype}, v_out.dtype={v1.dtype}")
    print(f"  s_out range: [{s1.min():.2f}, {s1.max():.2f}]")
    
    # With autocast — THIS IS WHAT HAPPENS DURING TRAINING
    with autocast('cuda', enabled=True):
        s2, v2 = _compute_messages(
            s_src, v_src, direction, Y_0, R,
            w_ss, w_sv, None, None, None, C_s, C_v,
        )
        print(f"With autocast: s_out.dtype={s2.dtype}, v_out.dtype={v2.dtype}")
        print(f"  s_out range: [{s2.min():.2f}, {s2.max():.2f}]")
        
        # FP16 range check
        if s2.dtype == torch.float16:
            print(f"  ⚠ OUTPUT IS FP16! Max FP16 = 65504")
            print(f"  ⚠ If any message > 65504, it overflows to Inf")
            overflow = (s2.abs() > 65000).sum().item()
            print(f"  ⚠ Values near FP16 limit: {overflow}")


def test_backward_overflow():
    """Check if backward through autocast matmul produces FP16 Inf gradients."""
    device = 'cuda'
    
    a = torch.randn(100, 64, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, 64, device=device, dtype=torch.float32, requires_grad=True)
    
    scale = 16384.0  # GradScaler scale factor
    
    with autocast('cuda', enabled=True):
        out = a @ w
        loss = out.sum() * scale  # simulate scaled loss
    
    loss.backward()
    
    print(f"\nBackward test (scale={scale}):")
    print(f"  a.grad dtype={a.grad.dtype}, max={a.grad.abs().max():.1f}")
    print(f"  w.grad dtype={w.grad.dtype}, max={w.grad.abs().max():.1f}")
    print(f"  a.grad has inf: {torch.isinf(a.grad).any().item()}")
    print(f"  w.grad has inf: {torch.isinf(w.grad).any().item()}")
    
    # The gradient of (a @ w).sum() w.r.t. a is: ones @ w^T
    # The gradient of (a @ w).sum() w.r.t. w is: a^T @ ones
    # These are matmuls — which under autocast may run in FP16!
    # scaled grad_a = scale * (ones @ w^T) — if any entry > 65504/scale ≈ 4 → overflow
    

if __name__ == '__main__':
    test_matmul_dtype_under_autocast()
    test_se3conv_internal_dtype()
    test_backward_overflow()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The root cause of persistent amp_scale decrease:

autocast('cuda') puts torch.mm/@ on the FP16 promotion list.
Even though SE3Conv explicitly casts inputs to FP32 via .to(float32),
the @/matmul operations INSIDE _compute_messages run in FP16
because autocast is still active in the calling scope.

This means:
1. s_src @ w_ss.t()  → FP16 matmul (forward AND backward)
2. If any gradient × scale > 65504 → Inf → GradScaler halves scale
3. Scale never recovers because overflow happens every ~100 steps
   but recovery needs 2000 consecutive clean steps

Fix: Either:
A. Disable autocast inside _compute_messages (torch.cuda.amp.autocast(enabled=False))
B. Disable AMP entirely for this model (most of computation is FP32 anyway)
""")
