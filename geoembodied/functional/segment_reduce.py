# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Deterministic segment reduce for GNN message aggregation.

Replaces ``scatter_add_`` with a CUDA-backed segment reduce that:
    1. Sorts edges by target node (CUB RadixSort)
    2. Builds CSR offsets
    3. Warp-level reduce per node — zero atomic operations

Properties:
    - Bit-exact deterministic (reproducible across runs)
    - Backward is a trivial gather (grad_msg[e] = grad_out[col[e]])
    - Auto-fallback to PyTorch scatter_add on CPU
    - Registered as ``torch.library.custom_op`` for ``torch.compile``
      transparency (no graph breaks)

Usage::

    from geoembodied.functional.segment_reduce import segment_reduce, sort_and_build_csr

    # Build CSR structure (once per graph)
    col_sorted, perm, node_start, node_end = sort_and_build_csr(col, N)

    # Reduce (per SE3Conv forward)
    out = segment_reduce(msg, col_sorted, perm, node_start, node_end, N)
"""

from typing import Tuple

import torch
from torch import Tensor

from geoembodied.csrc import (
    segment_reduce_available,
    sort_and_build_csr_cuda,
    segment_reduce_sum_cuda,
)


def sort_and_build_csr(
    col: Tensor,
    N: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sort edges by target and build CSR offsets.

    Auto-dispatches to CUDA (CUB sort) or PyTorch fallback.

    Args:
        col: [E] int64, target node indices
        N: Number of nodes

    Returns:
        col_sorted: [E] int32 — sorted target indices
        perm: [E] int32 — permutation mapping sorted→original edge order
        node_start: [N] int32 — CSR start pointer per node
        node_end: [N] int32 — CSR end pointer per node
    """
    E = col.shape[0]

    if E == 0:
        device = col.device
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(N, dtype=torch.int32, device=device),
            torch.zeros(N, dtype=torch.int32, device=device),
        )

    if col.is_cuda and segment_reduce_available():
        return sort_and_build_csr_cuda(col, N)
    else:
        # PyTorch fallback
        return _sort_and_build_csr_pytorch(col, N)


def _sort_and_build_csr_pytorch(
    col: Tensor,
    N: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pure-PyTorch fallback for sort + CSR build.

    Args:
        col: [E] int64, target node indices
        N: Number of nodes

    Returns:
        Same as sort_and_build_csr
    """
    device = col.device
    E = col.shape[0]

    perm = col.argsort()
    col_sorted = col[perm]

    # Build CSR by counting bin sizes
    counts = torch.bincount(col.int(), minlength=N)
    node_end = counts.cumsum(0).int()
    node_start = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        node_end[:-1],
    ])

    return (
        col_sorted.int(),
        perm.int(),
        node_start,
        node_end,
    )


class SegmentReduceFunction(torch.autograd.Function):
    """Deterministic segment-reduce with CUDA backend.

    Forward: out[n] = Σ_{e ∈ neighbors(n)} msg[e]
    Backward: grad_msg[e] = grad_out[col_sorted[e]]  (trivial gather)

    The sort permutation and CSR offsets are cached in ctx
    for backward reuse (zero re-computation).
    """

    @staticmethod
    def forward(
        ctx,
        msg: Tensor,            # [E, C] messages
        col_sorted: Tensor,     # [E] int32 sorted target indices
        perm: Tensor,           # [E] int32 permutation
        node_start: Tensor,     # [N] int32 CSR start
        node_end: Tensor,       # [N] int32 CSR end
        N: int,                 # number of nodes
    ) -> Tensor:
        # type: (Any, Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tensor
        """Forward: accumulated messages per target node."""
        ctx.save_for_backward(col_sorted, perm)
        ctx.N = N

        if msg.is_cuda and segment_reduce_available():
            return segment_reduce_sum_cuda(msg, perm, node_start, node_end, N)
        else:
            return _segment_reduce_pytorch(msg, perm, node_start, node_end, N)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        """Backward: gather grad_out at target node for each edge.

        grad_msg[orig_edge_idx] = grad_out[target_node_of_edge]
        """
        col_sorted, perm = ctx.saved_tensors

        # Gather: expand grad_out to edge-level
        # grad_out[col_sorted[e]] gives gradient for sorted edge e
        grad_msg_sorted = grad_out[col_sorted.long()]  # [E, C]

        # Un-permute: map from sorted order back to original edge order
        grad_msg = torch.empty_like(grad_msg_sorted)
        grad_msg[perm.long()] = grad_msg_sorted

        return grad_msg, None, None, None, None, None


def _segment_reduce_pytorch(
    msg: Tensor,
    perm: Tensor,
    node_start: Tensor,
    node_end: Tensor,
    N: int,
) -> Tensor:
    """Pure-PyTorch fallback for segment reduce.

    Uses scatter_add_ (non-deterministic on GPU but works on CPU).

    Args:
        msg: [E, C] messages
        perm: [E] int32 permutation — not used in fallback
        node_start: [N] int32 — not used in fallback
        node_end: [N] int32 — not used in fallback
        N: number of nodes

    Returns:
        out: [N, C] float32
    """
    # Reconstruct col from col_sorted by inverting CSR
    # But simpler: just use the sorted structure directly
    E, C = msg.shape
    device = msg.device

    out = torch.zeros(N, C, device=device, dtype=msg.dtype)
    if E == 0:
        return out

    # Iterate CSR segments (CPU-friendly, vectorized on small segments)
    for n in range(N):
        s = node_start[n].item()
        e = node_end[n].item()
        if s < e:
            edge_ids = perm[s:e].long()
            out[n] = msg[edge_ids].sum(dim=0)

    return out


# ═══════════════════════════════════════════════════════════════════
# torch.library.custom_op registration for torch.compile transparency
# ═══════════════════════════════════════════════════════════════════

# Register the low-level forward as a custom op so torch.compile
# can trace through it without graph breaks.
_CUSTOM_OP_REGISTERED = False


def _register_custom_op() -> None:
    """Register segment_reduce as torch.library.custom_op (once).

    This allows torch.compile to trace through the segment reduce
    call without inserting a graph break. The backward is defined
    via setup_context + backward decorators.
    """
    global _CUSTOM_OP_REGISTERED
    if _CUSTOM_OP_REGISTERED:
        return
    _CUSTOM_OP_REGISTERED = True

    @torch.library.custom_op(
        "geoembodied::segment_reduce_sum", mutates_args=()
    )
    def _segment_reduce_op(
        msg: Tensor,
        col_sorted: Tensor,
        perm: Tensor,
        node_start: Tensor,
        node_end: Tensor,
        N: int,
    ) -> Tensor:
        """Forward: deterministic segment reduce (CUDA or fallback)."""
        if msg.is_cuda and segment_reduce_available():
            return segment_reduce_sum_cuda(msg, perm, node_start, node_end, N)
        else:
            return _segment_reduce_pytorch(msg, perm, node_start, node_end, N)

    @_segment_reduce_op.register_fake
    def _segment_reduce_fake(
        msg: Tensor,
        col_sorted: Tensor,
        perm: Tensor,
        node_start: Tensor,
        node_end: Tensor,
        N: int,
    ) -> Tensor:
        """Shape inference for torch.compile tracing."""
        C = msg.shape[1] if msg.dim() > 1 else 1
        return msg.new_empty(N, C)

    def _segment_reduce_setup_context(ctx, inputs, output) -> None:
        """Save tensors needed for backward."""
        msg, col_sorted, perm, node_start, node_end, N = inputs
        ctx.save_for_backward(col_sorted, perm)
        ctx.N = N

    def _segment_reduce_backward(ctx, grad_out: Tensor):
        """Backward: gather grad_out at target node for each edge.

        grad_msg[orig_edge_idx] = grad_out[target_node_of_edge]
        """
        col_sorted, perm = ctx.saved_tensors
        # Gather: expand grad_out to edge-level
        grad_msg_sorted = grad_out[col_sorted.long()]  # [E, C]
        # Un-permute: sorted → original edge order
        grad_msg = torch.empty_like(grad_msg_sorted)
        grad_msg[perm.long()] = grad_msg_sorted
        return grad_msg, None, None, None, None, None

    _segment_reduce_op.register_autograd(
        _segment_reduce_backward,
        setup_context=_segment_reduce_setup_context,
    )


# Eagerly register at import time
_register_custom_op()


def segment_reduce(
    msg: Tensor,
    col_sorted: Tensor,
    perm: Tensor,
    node_start: Tensor,
    node_end: Tensor,
    N: int,
) -> Tensor:
    """Deterministic aggregation of edge messages to nodes.

    Replaces ``scatter_add_`` with zero atomic contention.
    Registered as ``torch.library.custom_op`` for ``torch.compile``
    transparency — no graph breaks at this boundary.

    Forward: CUDA warp-level reduce (shared memory, no atomics)
    Backward: trivial gather (grad_msg[e] = grad_out[col[e]])

    Args:
        msg: [E, C] float32, per-edge messages (original edge ordering)
        col_sorted: [E] int32, sorted target indices
        perm: [E] int32, permutation from sort_and_build_csr
        node_start: [N] int32, CSR start offsets
        node_end: [N] int32, CSR end offsets
        N: Number of target nodes

    Returns:
        out: [N, C] float32, aggregated features
            out[n] = Σ_{edges to n} msg[e]
    """
    return torch.ops.geoembodied.segment_reduce_sum(
        msg, col_sorted, perm, node_start, node_end, N,
    )
