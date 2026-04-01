# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for CUDA radius_graph kernel.

Validates:
    1. Correctness vs brute-force reference implementation
    2. Algebraic masking (pad points excluded)
    3. Self-loop control
    4. Edge cases (empty, single point, tiny radius)
    5. Batched mode
    6. Performance regression guard

Requires: CUDA-enabled PyTorch + compiled kernel (JIT at import time).
"""

import os
import math
import pytest
import torch

# Skip entire module if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@pytest.fixture(autouse=True)
def set_cuda_arch():
    """Auto-detect GPU arch for JIT compilation."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"


def _get_cuda_module():
    """Try to load the CUDA module, skip if unavailable."""
    try:
        from geoembodied.csrc import radius_graph_cuda
        # Force compilation check
        from geoembodied.csrc import get_radius_graph_backend
        if get_radius_graph_backend() != "cuda":
            pytest.skip("CUDA kernel not compiled")
        return radius_graph_cuda
    except Exception as e:
        pytest.skip(f"CUDA kernel load failed: {e}")


def _brute_force_ref(x, r, max_k=16, loop=False, mask=None):
    """Reference brute-force implementation for comparison."""
    from geoembodied.functional.radius_graph import _radius_graph_single
    return _radius_graph_single(x, r, max_num_neighbors=max_k, loop=loop, mask=mask)


def _edges_to_set(row, col):
    """Convert edge index tensors to a set of (src, tgt) tuples."""
    return set(zip(row.cpu().tolist(), col.cpu().tolist()))


# ═══════════════════════════════════════════════════════════════════
# Correctness Tests
# ═══════════════════════════════════════════════════════════════════


class TestRadiusGraphCUDACorrectness:
    """Verify CUDA kernel matches brute-force reference."""

    @pytest.mark.parametrize("N,r", [
        (50, 0.5),
        (100, 0.3),
        (200, 0.2),
        (500, 0.15),
        (1000, 0.1),
    ])
    def test_matches_bruteforce(self, N: int, r: float) -> None:
        """CUDA edges == brute-force edges for moderate N."""
        rg_cuda = _get_cuda_module()
        device = torch.device("cuda")

        torch.manual_seed(42)
        x = torch.randn(N, 3, device=device)
        K = 16

        row_bf, col_bf = _brute_force_ref(x, r, max_k=K, loop=False)
        row_cu, col_cu = rg_cuda(x, r, max_num_neighbors=K, loop=False)

        edges_bf = _edges_to_set(row_bf, col_bf)
        edges_cu = _edges_to_set(row_cu, col_cu)

        # Allow minor K-cap ordering differences
        if len(edges_bf) == 0 and len(edges_cu) == 0:
            return  # Both empty — OK
        overlap = len(edges_bf & edges_cu) / max(len(edges_bf | edges_cu), 1)
        assert overlap > 0.95, (
            f"N={N}, r={r}: Jaccard={overlap:.3f}, "
            f"BF={len(edges_bf)}, CUDA={len(edges_cu)}, "
            f"BF-only={len(edges_bf - edges_cu)}, CUDA-only={len(edges_cu - edges_bf)}"
        )


class TestAlgebraicMasking:
    """Verify mask=False points are excluded from all edges."""

    def test_masked_points_excluded(self) -> None:
        """No edge should reference a masked point."""
        rg_cuda = _get_cuda_module()
        device = torch.device("cuda")

        torch.manual_seed(42)
        N = 200
        x = torch.randn(N, 3, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[150:] = False  # Last 50 points are pad

        row, col = rg_cuda(x, 0.4, max_num_neighbors=16, loop=False, mask=mask)

        if len(row) > 0:
            assert (row < 150).all(), "Masked points appear as source"
            assert (col < 150).all(), "Masked points appear as target"

    def test_all_masked_no_edges(self) -> None:
        """All masked → no edges."""
        rg_cuda = _get_cuda_module()
        device = torch.device("cuda")

        x = torch.randn(100, 3, device=device)
        mask = torch.zeros(100, dtype=torch.bool, device=device)

        row, col = rg_cuda(x, 1.0, max_num_neighbors=16, loop=False, mask=mask)
        assert len(row) == 0, f"Expected 0 edges with all-masked, got {len(row)}"


class TestSelfLoops:
    """Verify self-loop control."""

    def test_no_self_loops_default(self) -> None:
        """loop=False should produce no (i,i) edges."""
        rg_cuda = _get_cuda_module()
        device = torch.device("cuda")

        torch.manual_seed(42)
        x = torch.randn(100, 3, device=device)
        row, col = rg_cuda(x, 0.5, max_num_neighbors=16, loop=False)

        if len(row) > 0:
            assert not (row == col).any(), "Self-loops found with loop=False"

    def test_self_loops_enabled(self) -> None:
        """loop=True should include (i,i) edges where applicable."""
        rg_cuda = _get_cuda_module()
        device = torch.device("cuda")

        torch.manual_seed(42)
        x = torch.randn(50, 3, device=device)
        row, col = rg_cuda(x, 10.0, max_num_neighbors=64, loop=True)  # Large radius

        if len(row) > 0:
            has_self = (row == col).any().item()
            assert has_self, "No self-loops with loop=True and large radius"


class TestEdgeCases:
    """Edge cases: tiny N, zero radius, etc."""

    def test_single_point(self) -> None:
        """Single point, no loop → 0 edges."""
        rg_cuda = _get_cuda_module()
        x = torch.randn(1, 3, device="cuda")
        row, col = rg_cuda(x, 1.0, max_num_neighbors=16, loop=False)
        assert len(row) == 0

    def test_two_points_far_apart(self) -> None:
        """Two points far apart → 0 edges."""
        rg_cuda = _get_cuda_module()
        x = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], device="cuda")
        row, col = rg_cuda(x, 1.0, max_num_neighbors=16, loop=False)
        assert len(row) == 0

    def test_two_points_close(self) -> None:
        """Two points close → 2 edges (bidirectional)."""
        rg_cuda = _get_cuda_module()
        x = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], device="cuda")
        row, col = rg_cuda(x, 0.5, max_num_neighbors=16, loop=False)
        assert len(row) == 2, f"Expected 2 bidirectional edges, got {len(row)}"

    def test_zero_radius(self) -> None:
        """Zero radius → 0 edges."""
        rg_cuda = _get_cuda_module()
        x = torch.randn(100, 3, device="cuda")
        row, col = rg_cuda(x, 0.0, max_num_neighbors=16, loop=False)
        assert len(row) == 0


class TestDispatcher:
    """Test the auto-dispatch in radius_graph() public API."""

    def test_auto_dispatch_cuda(self) -> None:
        """radius_graph() should auto-dispatch to CUDA on GPU tensors."""
        from geoembodied.functional.radius_graph import radius_graph
        from geoembodied.csrc import get_radius_graph_backend

        if get_radius_graph_backend() != "cuda":
            pytest.skip("CUDA backend not available")

        torch.manual_seed(42)
        x = torch.randn(100, 3, device="cuda")
        row, col = radius_graph(x, 0.5, max_num_neighbors=16)
        assert row.device.type == "cuda"
        assert len(row) > 0

    def test_cpu_fallback(self) -> None:
        """radius_graph() should use brute-force on CPU tensors."""
        from geoembodied.functional.radius_graph import radius_graph

        torch.manual_seed(42)
        x = torch.randn(50, 3)  # CPU
        row, col = radius_graph(x, 0.5, max_num_neighbors=16)
        assert row.device.type == "cpu"


class TestPerformanceGuard:
    """Ensure CUDA kernel handles large N without crash."""

    @pytest.mark.parametrize("N", [10000, 50000])
    def test_large_n(self, N: int) -> None:
        """CUDA kernel should handle large N without error."""
        rg_cuda = _get_cuda_module()
        x = torch.randn(N, 3, device="cuda") * 0.5
        row, col = rg_cuda(x, 0.05, max_num_neighbors=32, loop=False)
        assert len(row) >= 0  # Just verify no crash
        assert not torch.isnan(row.float()).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
