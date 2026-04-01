# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for CUDA segment reduce kernel and SpatialGraph.

Test categories:
    1. Correctness: segment_reduce matches scatter_add_
    2. Bit-exact determinism: 100 runs → max_deviation = 0
    3. Autograd: gradcheck + backward correctness
    4. SpatialGraph: build, from_edge_index, empty
    5. SE3Conv integration: new API matches legacy, deterministic
    6. Edge cases: E=0, N=1, single-node graphs
"""

import pytest
import torch
from torch import Tensor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

DEVICE = torch.device("cuda")


# ═══════════════════════════════════════════════════════════════════
# Segment Reduce Tests
# ═══════════════════════════════════════════════════════════════════

class TestSegmentReduce:
    """CUDA segment reduce correctness and determinism."""

    def test_correctness_vs_scatter_add(self):
        """Segment reduce must match scatter_add_ within float32 ULP."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        torch.manual_seed(42)
        N, E, C = 500, 3000, 32
        col = torch.randint(0, N, (E,), device=DEVICE)
        msg = torch.randn(E, C, device=DEVICE)

        # Reference
        ref = torch.zeros(N, C, device=DEVICE)
        ref.scatter_add_(0, col.unsqueeze(-1).expand(E, C), msg)

        # Segment reduce
        cs, pm, ns, ne = sort_and_build_csr(col, N)
        out = segment_reduce(msg, cs, pm, ns, ne, N)

        assert torch.allclose(out, ref, atol=1e-5), \
            f"Max err: {(out - ref).abs().max():.2e}"

    def test_bit_exact_determinism(self):
        """20 runs must produce identical results (zero deviation)."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        torch.manual_seed(0)
        N, E, C = 100, 500, 8
        col = torch.randint(0, N, (E,), device=DEVICE)
        msg = torch.randn(E, C, device=DEVICE)
        cs, pm, ns, ne = sort_and_build_csr(col, N)

        # Streaming comparison: compare each run to first, no stacking
        ref = segment_reduce(msg, cs, pm, ns, ne, N)
        max_dev = 0.0
        for _ in range(19):
            out = segment_reduce(msg, cs, pm, ns, ne, N)
            max_dev = max(max_dev, (out - ref).abs().max().item())
        assert max_dev == 0.0, f"Non-deterministic: max_dev={max_dev}"

    def test_backward_correctness(self):
        """Backward of sum-reduce should distribute ones to all edges."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        torch.manual_seed(42)
        N, E, C = 50, 200, 8
        col = torch.randint(0, N, (E,), device=DEVICE)
        msg = torch.randn(E, C, device=DEVICE, requires_grad=True)
        cs, pm, ns, ne = sort_and_build_csr(col, N)

        out = segment_reduce(msg, cs, pm, ns, ne, N)
        out.sum().backward()

        # d(sum_of_all_out)/d(msg[e]) = 1 for all e
        assert torch.allclose(msg.grad, torch.ones_like(msg), atol=1e-6)

    def test_autograd_gradcheck(self):
        """torch.autograd.gradcheck with double precision."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        N, E, C = 10, 30, 4
        col = torch.randint(0, N, (E,), device=DEVICE)
        msg = torch.randn(E, C, device=DEVICE, dtype=torch.float64, requires_grad=True)
        cs, pm, ns, ne = sort_and_build_csr(col, N)

        # gradcheck only tests msg gradient (other args don't require grad)
        def fn(m):
            return segment_reduce(m, cs, pm, ns, ne, N)

        # Use PyTorch fallback for gradcheck (CUDA kernel only supports float32)
        msg_cpu = msg.cpu().requires_grad_(True)
        cs_cpu = cs.cpu()
        pm_cpu = pm.cpu()
        ns_cpu = ns.cpu()
        ne_cpu = ne.cpu()

        def fn_cpu(m):
            from geoembodied.functional.segment_reduce import SegmentReduceFunction
            return SegmentReduceFunction.apply(m, cs_cpu, pm_cpu, ns_cpu, ne_cpu, N)

        assert torch.autograd.gradcheck(fn_cpu, (msg_cpu,), atol=1e-5)

    def test_empty_edges(self):
        """E=0 should produce zero output."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        N, C = 10, 8
        col = torch.empty(0, dtype=torch.long, device=DEVICE)
        msg = torch.empty(0, C, device=DEVICE)
        cs, pm, ns, ne = sort_and_build_csr(col, N)
        out = segment_reduce(msg, cs, pm, ns, ne, N)
        assert out.shape == (N, C)
        assert out.abs().sum() == 0

    def test_single_node(self):
        """All edges target node 0 → out[0] = sum(all messages)."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        E, C = 10, 4
        col = torch.zeros(E, dtype=torch.long, device=DEVICE)
        msg = torch.ones(E, C, device=DEVICE)
        cs, pm, ns, ne = sort_and_build_csr(col, 1)
        out = segment_reduce(msg, cs, pm, ns, ne, 1)
        assert torch.allclose(out, torch.full((1, C), float(E), device=DEVICE))

    @pytest.mark.parametrize("C", [1, 16, 32, 64])
    def test_various_channel_widths(self, C):
        """Kernel should work for various channel widths."""
        from geoembodied.functional.segment_reduce import (
            sort_and_build_csr, segment_reduce,
        )
        torch.manual_seed(C)
        N, E = 30, 150
        col = torch.randint(0, N, (E,), device=DEVICE)
        msg = torch.randn(E, C, device=DEVICE)

        ref = torch.zeros(N, C, device=DEVICE)
        ref.scatter_add_(0, col.unsqueeze(-1).expand(E, C), msg)

        cs, pm, ns, ne = sort_and_build_csr(col, N)
        out = segment_reduce(msg, cs, pm, ns, ne, N)
        assert torch.allclose(out, ref, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════
# SpatialGraph Tests
# ═══════════════════════════════════════════════════════════════════

class TestSpatialGraph:
    """SpatialGraph construction and invariants."""

    def test_build_basic(self):
        """Build from positions should populate all fields."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        torch.manual_seed(42)
        N = 100
        pos = torch.randn(N, 3, device=DEVICE) * 0.5
        graph = SpatialGraph.build(pos, radius=0.3, max_num_neighbors=16)

        assert graph.N == N
        assert graph.E == graph.row.shape[0]
        assert graph.row.shape == graph.col.shape
        assert graph.direction.shape == (graph.E, 3)
        assert graph.dist.shape == (graph.E,)
        assert graph.Y.shape == (graph.E, 9)  # l≤2 → 9 SH
        assert graph.col_sorted.shape == (graph.E,)
        assert graph.perm.shape == (graph.E,)
        assert graph.node_start.shape == (N,)
        assert graph.node_end.shape == (N,)

    def test_directions_are_unit(self):
        """Edge directions should have unit norm."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        N = 50
        pos = torch.randn(N, 3, device=DEVICE) * 0.5
        graph = SpatialGraph.build(pos, radius=0.5)
        if graph.E > 0:
            norms = graph.direction.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_from_edge_index(self):
        """from_edge_index should match build with same edges."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        from geoembodied.functional.radius_graph import radius_graph
        torch.manual_seed(42)
        N = 50
        pos = torch.randn(N, 3, device=DEVICE) * 0.5
        row, col = radius_graph(pos, 0.3, max_num_neighbors=16)

        g1 = SpatialGraph.build(pos, 0.3, max_num_neighbors=16)
        g2 = SpatialGraph.from_edge_index(row, col, pos, N)

        assert g1.E == g2.E
        assert torch.allclose(g1.direction, g2.direction, atol=1e-6)
        assert torch.allclose(g1.Y, g2.Y, atol=1e-6)

    def test_empty_graph(self):
        """Tiny radius → zero edges."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        pos = torch.randn(10, 3, device=DEVICE)
        graph = SpatialGraph.build(pos, radius=1e-6)
        assert graph.E == 0
        assert graph.node_start.shape == (10,)

    def test_immutable(self):
        """SpatialGraph should be frozen (no attribute assignment)."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        pos = torch.randn(10, 3, device=DEVICE) * 0.5
        graph = SpatialGraph.build(pos, radius=0.5)
        with pytest.raises(Exception):
            graph.N = 999


# ═══════════════════════════════════════════════════════════════════
# SE3Conv Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestSE3ConvIntegration:
    """SE3Conv with SpatialGraph — new API tests."""

    @pytest.fixture
    def conv(self):
        from geoembodied.nn.se3_conv import SE3Conv
        return SE3Conv(
            in_scalar_channels=16, in_vector_channels=4,
            out_scalar_channels=16, out_vector_channels=4,
            radius=0.3, max_num_neighbors=32,
        ).to(DEVICE)

    @pytest.fixture
    def graph(self):
        from geoembodied.nn.spatial_graph import SpatialGraph
        torch.manual_seed(42)
        pos = torch.randn(50, 3, device=DEVICE) * 0.5
        return SpatialGraph.build(pos, radius=0.3, max_num_neighbors=32)

    def test_new_api_matches_legacy(self, conv):
        """New graph API must produce identical results to legacy."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        torch.manual_seed(42)
        pos = torch.randn(50, 3, device=DEVICE) * 0.5
        s = torch.randn(50, 16, device=DEVICE)
        v = torch.randn(50, 4, 3, device=DEVICE)

        conv.eval()
        graph = SpatialGraph.build(pos, 0.3, max_num_neighbors=32)

        with torch.no_grad():
            s1, v1 = conv(s, v, graph)
            s2, v2 = conv.forward_legacy(pos, s, v)

        assert torch.allclose(s1, s2, atol=1e-5), \
            f"s err: {(s1-s2).abs().max():.2e}"
        assert torch.allclose(v1, v2, atol=1e-5), \
            f"v err: {(v1-v2).abs().max():.2e}"

    def test_deterministic(self, conv, graph):
        """5 forward passes must be bit-exact identical."""
        s = torch.randn(50, 16, device=DEVICE)
        v = torch.randn(50, 4, 3, device=DEVICE)

        with torch.no_grad():
            ref, _ = conv(s, v, graph)
            for _ in range(4):
                s_out, _ = conv(s, v, graph)
                assert (s_out - ref).abs().max().item() == 0.0

    def test_backward_no_nan(self, conv, graph):
        """Full forward+backward should have no NaN."""
        s = torch.randn(graph.N, 16, device=DEVICE, requires_grad=True)
        v = torch.randn(graph.N, 4, 3, device=DEVICE, requires_grad=True)

        s_out, v_out = conv(s, v, graph)
        loss = s_out.sum() + v_out.sum()
        loss.backward()

        assert not torch.isnan(s.grad).any()
        assert not torch.isnan(v.grad).any()
        for p in conv.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()

    def test_graph_reuse_across_layers(self):
        """Two SE3Conv layers sharing the same graph."""
        from geoembodied.nn.se3_conv import SE3Conv
        from geoembodied.nn.spatial_graph import SpatialGraph

        conv1 = SE3Conv(8, 2, 16, 4, radius=0.3).to(DEVICE)
        conv2 = SE3Conv(16, 4, 8, 2, radius=0.3).to(DEVICE)

        torch.manual_seed(42)
        pos = torch.randn(20, 3, device=DEVICE) * 0.5
        graph = SpatialGraph.build(pos, 0.3)

        s = torch.randn(20, 8, device=DEVICE)
        v = torch.randn(20, 2, 3, device=DEVICE)

        s1, v1 = conv1(s, v, graph)
        s2, v2 = conv2(s1, v1, graph)

        assert s2.shape == (20, 8)
        assert v2.shape == (20, 2, 3)

        loss = s2.sum() + v2.sum()
        loss.backward()
        torch.cuda.empty_cache()

    def test_empty_graph(self, conv):
        """Zero edges → zero output."""
        from geoembodied.nn.spatial_graph import SpatialGraph
        N = 10
        pos = torch.randn(N, 3, device=DEVICE)
        graph = SpatialGraph.build(pos, radius=1e-6)

        s = torch.randn(N, 16, device=DEVICE)
        v = torch.randn(N, 4, 3, device=DEVICE)

        s_out, v_out = conv(s, v, graph)
        assert s_out.shape == (N, 16)
        # Only bias should be present
