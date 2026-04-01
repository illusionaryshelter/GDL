# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""torch.compile end-to-end verification for SE3Conv.

This is the Phase 2 final deliverable: prove that the entire
cuBLAS + CUDA segment_reduce + _compute_messages pipeline is
torch.compile transparent with ZERO graph breaks.

Test categories:
    1. Graph break count = 0 (torch._dynamo.explain)
    2. Numerical parity: compiled output == eager output
    3. Gradient parity: compiled backward == eager backward

Memory budget: < 200MB VRAM (micro-test safe for 4GB devices).
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

DEVICE = torch.device("cuda")


# ═══════════════════════════════════════════════════════════════════
# Minimal SE3Net: 2-layer backbone for compile testing
# ═══════════════════════════════════════════════════════════════════

class MiniSE3Net(nn.Module):
    """Minimal 2-layer SE3Conv backbone for torch.compile testing.

    This tests the full pipeline:
        gather → _compute_messages (cuBLAS) → segment_reduce (custom_op)
        → bias → next layer
    """

    def __init__(self) -> None:
        super().__init__()
        from geoembodied.nn.se3_conv import SE3Conv
        self.conv1 = SE3Conv(
            in_scalar_channels=8, in_vector_channels=4,
            out_scalar_channels=8, out_vector_channels=4,
            radius=3.0, max_num_neighbors=16,
        )
        self.conv2 = SE3Conv(
            in_scalar_channels=8, in_vector_channels=4,
            out_scalar_channels=4, out_vector_channels=2,
            radius=3.0, max_num_neighbors=16,
        )

    def forward(
        self, scalars: Tensor, vectors: Tensor, graph,
    ) -> tuple[Tensor, Tensor]:
        """Two-layer forward.

        Args:
            scalars: [N, 8] float32
            vectors: [N, 4, 3] float32
            graph: SpatialGraph

        Returns:
            (scalar_out, vector_out): [N, 4], [N, 2, 3]
        """
        s, v = self.conv1(scalars, vectors, graph)
        s, v = self.conv2(s, v, graph)
        return s, v


# ═══════════════════════════════════════════════════════════════════
# Test: Graph Break Analysis
# ═══════════════════════════════════════════════════════════════════

class TestCompileGraphBreaks:
    """Verify torch.compile produces ZERO graph breaks."""

    def test_zero_graph_breaks(self) -> None:
        """torch._dynamo.explain must report 0 graph breaks.

        This is the crown jewel assertion: the entire SE3Conv pipeline —
        including our custom CUDA segment_reduce registered via
        torch.library.custom_op — is fully traceable by Dynamo.
        """
        from geoembodied.nn.spatial_graph import SpatialGraph

        model = MiniSE3Net().to(DEVICE)
        model.eval()

        # Build graph (this is OUTSIDE the compiled region)
        torch.manual_seed(42)
        pos = torch.randn(15, 3, device=DEVICE) * 0.5
        s = torch.randn(15, 8, device=DEVICE)
        v = torch.randn(15, 4, 3, device=DEVICE)
        graph = SpatialGraph.build(pos, radius=3.0, max_num_neighbors=16)

        # Use torch._dynamo.explain to count graph breaks
        explanation = torch._dynamo.explain(model)(s, v, graph)

        # The key assertion
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}.\n"
            f"Break reasons: {explanation.break_reasons}"
        )


# ═══════════════════════════════════════════════════════════════════
# Test: Numerical Parity (Eager vs Compiled)
# ═══════════════════════════════════════════════════════════════════

class TestCompileNumericalParity:
    """Verify compiled model produces bit-close output to eager mode."""

    def test_forward_parity(self) -> None:
        """Compiled forward output matches eager forward output."""
        from geoembodied.nn.spatial_graph import SpatialGraph

        model = MiniSE3Net().to(DEVICE)
        model.eval()

        torch.manual_seed(42)
        pos = torch.randn(15, 3, device=DEVICE) * 0.5
        s = torch.randn(15, 8, device=DEVICE)
        v = torch.randn(15, 4, 3, device=DEVICE)
        graph = SpatialGraph.build(pos, radius=3.0, max_num_neighbors=16)

        # Eager forward
        with torch.no_grad():
            s_eager, v_eager = model(s, v, graph)

        # Compiled forward (dynamic=True for variable E)
        compiled_model = torch.compile(model, dynamic=True)
        with torch.no_grad():
            s_compiled, v_compiled = compiled_model(s, v, graph)

        assert torch.allclose(s_eager, s_compiled, atol=1e-5), (
            f"Scalar parity failed: max err = "
            f"{(s_eager - s_compiled).abs().max():.2e}"
        )
        assert torch.allclose(v_eager, v_compiled, atol=1e-5), (
            f"Vector parity failed: max err = "
            f"{(v_eager - v_compiled).abs().max():.2e}"
        )

    def test_backward_parity(self) -> None:
        """Compiled backward gradients match eager backward gradients."""
        from geoembodied.nn.spatial_graph import SpatialGraph

        model_eager = MiniSE3Net().to(DEVICE)
        model_compiled = MiniSE3Net().to(DEVICE)
        # Copy weights to ensure identical models
        model_compiled.load_state_dict(model_eager.state_dict())

        torch.manual_seed(42)
        pos = torch.randn(15, 3, device=DEVICE) * 0.5
        graph = SpatialGraph.build(pos, radius=3.0, max_num_neighbors=16)

        # Eager
        s1 = torch.randn(15, 8, device=DEVICE, requires_grad=True)
        v1 = torch.randn(15, 4, 3, device=DEVICE, requires_grad=True)
        s_e, v_e = model_eager(s1, v1, graph)
        loss_e = s_e.sum() + v_e.sum()
        loss_e.backward()

        # Compiled
        compiled = torch.compile(model_compiled, dynamic=True)
        s2 = s1.data.clone().requires_grad_(True)
        v2 = v1.data.clone().requires_grad_(True)
        s_c, v_c = compiled(s2, v2, graph)
        loss_c = s_c.sum() + v_c.sum()
        loss_c.backward()

        # Compare input gradients
        assert torch.allclose(s1.grad, s2.grad, atol=1e-4), (
            f"Scalar grad parity failed: max err = "
            f"{(s1.grad - s2.grad).abs().max():.2e}"
        )
        assert torch.allclose(v1.grad, v2.grad, atol=1e-4), (
            f"Vector grad parity failed: max err = "
            f"{(v1.grad - v2.grad).abs().max():.2e}"
        )

        # Compare parameter gradients
        for (n1, p1), (n2, p2) in zip(
            model_eager.named_parameters(),
            model_compiled.named_parameters(),
        ):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-4), (
                    f"Param grad parity failed for {n1}: max err = "
                    f"{(p1.grad - p2.grad).abs().max():.2e}"
                )

