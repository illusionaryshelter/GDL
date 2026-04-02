"""Verify: FP64 equivariance path is preserved after AMP guard fix.

The autocast guard must:
1. Upcast FP16/BF16 → FP32 (AMP safety)
2. Preserve FP64 (equivariance tests need this)
3. NOT introduce mixed dtypes (BatchNorm requires uniform dtype)
"""
import torch

def test_fp64_preserved_in_se3conv():
    """SE3Conv must work with FP64 inputs (not downcast to FP32)."""
    from geoembodied.nn.modules.se3_conv import SE3Conv
    from geoembodied.nn.modules.spatial_graph import SpatialGraph
    
    conv = SE3Conv(
        in_scalar_channels=16, in_vector_channels=4,
        out_scalar_channels=16, out_vector_channels=4,
        radius=2.0,
    ).double()  # FP64 module
    
    pos = torch.randn(50, 3, dtype=torch.float64)
    s = torch.randn(50, 16, dtype=torch.float64)
    v = torch.randn(50, 4, 3, dtype=torch.float64)
    
    graph = SpatialGraph.build(pos, radius=2.0, max_num_neighbors=32)
    s_out, v_out = conv(s, v, graph)
    
    assert s_out.dtype == torch.float64, f"Expected FP64 output, got {s_out.dtype}"
    assert v_out.dtype == torch.float64, f"Expected FP64 output, got {v_out.dtype}"
    print("✅ SE3Conv FP64 preservation: PASS")


def test_fp64_preserved_in_pool():
    """EquivariantPool must work with FP64 inputs."""
    from geoembodied.nn.modules.equivariant_pool import EquivariantPool
    
    pool = EquivariantPool(
        scalar_channels=16, vector_channels=4,
        ratio=0.5, k_neighbors=8,
    ).double()  # FP64 module
    
    pos = torch.randn(50, 3, dtype=torch.float64)
    s = torch.randn(50, 16, dtype=torch.float64)
    v = torch.randn(50, 4, 3, dtype=torch.float64)
    ptr = torch.tensor([0, 50], dtype=torch.int64)
    
    seed_pos, s_out, v_out, ptr_out, fps_idx = pool(pos, s, v, ptr)
    
    assert s_out.dtype == torch.float64, f"Expected FP64 s_out, got {s_out.dtype}"
    assert v_out.dtype == torch.float64, f"Expected FP64 v_out, got {v_out.dtype}"
    print("✅ EquivariantPool FP64 preservation: PASS")


def test_fp64_preserved_in_multiscale():
    """MultiScaleSE3Net must work with FP64 inputs end-to-end."""
    from geoembodied.nn.modules.multi_scale_se3_net import MultiScaleSE3Net
    
    model = MultiScaleSE3Net(
        in_channels=1,
        hidden_scalar=16, hidden_vector=4,
        num_stages=2, layers_per_stage=1,
    ).double()  # FP64 module
    
    pos = torch.randn(50, 3, dtype=torch.float64)
    ptr = torch.tensor([0, 25, 50], dtype=torch.int64)
    
    s_out, v_out, _ = model(pos, ptr)
    
    assert s_out.dtype == torch.float64, f"Expected FP64 s_out, got {s_out.dtype}"
    assert v_out.dtype == torch.float64, f"Expected FP64 v_out, got {v_out.dtype}"
    print("✅ MultiScaleSE3Net FP64 preservation: PASS")


def test_fp32_under_autocast():
    """FP32 inputs under autocast must NOT be downcast (the AMP guard works)."""
    from torch.amp import autocast
    from geoembodied.nn.modules.se3_conv import SE3Conv
    from geoembodied.nn.modules.spatial_graph import SpatialGraph
    
    if not torch.cuda.is_available():
        print("⏭ Skipping CUDA autocast test (no GPU)")
        return
    
    device = 'cuda'
    conv = SE3Conv(
        in_scalar_channels=16, in_vector_channels=4,
        out_scalar_channels=16, out_vector_channels=4,
        radius=2.0,
    ).to(device)
    
    pos = torch.randn(50, 3, device=device)
    s = torch.randn(50, 16, device=device)
    v = torch.randn(50, 4, 3, device=device)
    
    graph = SpatialGraph.build(pos, radius=2.0, max_num_neighbors=32)
    
    with autocast('cuda'):
        s_out, v_out = conv(s, v, graph)
    
    # Even under autocast, output should be FP32 (not FP16)
    assert s_out.dtype == torch.float32, f"Expected FP32 under autocast, got {s_out.dtype}"
    assert v_out.dtype == torch.float32, f"Expected FP32 under autocast, got {v_out.dtype}"
    print("✅ SE3Conv FP32 under autocast: PASS (not demoted to FP16)")


if __name__ == '__main__':
    test_fp64_preserved_in_se3conv()
    test_fp64_preserved_in_pool()
    test_fp64_preserved_in_multiscale()
    test_fp32_under_autocast()
    print("\n✅ All dtype preservation tests passed!")
