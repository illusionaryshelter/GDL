# Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Backend dispatcher for CUDA kernels — radius_graph, segment_reduce, knn, fps.

Architecture:
    1. On first call, attempts JIT compilation of CUDA kernels
    2. If CUDA available and compilation succeeds → CUDA backend
    3. Otherwise → falls back to PyTorch CPU/autograd backends
    4. Backend selection is cached for the lifetime of the process

Kernels:
    - radius_graph: Hash-grid O(N log N + NK) neighbor search
    - segment_reduce: CUB sort + CSR + warp-level reduce (no atomics)
    - knn: Brute-force O(N_q * N_s) with MinK heap, ptr-based batching
    - fps: Farthest Point Sampling O(N × K) with block-level argmax reduce

Usage:
    from geoembodied.csrc import get_radius_graph_backend
    from geoembodied.csrc import segment_reduce_available
    from geoembodied.csrc import knn_available, knn_cuda, knn_self_cuda
    from geoembodied.csrc import fps_available, fps_cuda
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Radius Graph Backend
# ═══════════════════════════════════════════════════════════════════

# Cached module reference
_cuda_module = None
_backend_checked = False
_backend_available = False

# Path to CUDA source
_CSRC_DIR = Path(__file__).parent
_KERNEL_PATH = _CSRC_DIR / "radius_graph_kernel.cu"


def _get_cuda_arch_list() -> str:
    """Auto-detect GPU compute capability for JIT compilation.

    Returns:
        CUDA arch list string (e.g. '8.7' for Orin)
    """
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    try:
        cap = torch.cuda.get_device_capability(0)
        return f"{cap[0]}.{cap[1]}"
    except Exception:
        return "7.0;7.5;8.0;8.6;8.7;8.9;9.0+PTX"


def _try_load_cuda_module():
    """Attempt JIT compilation of the CUDA radius_graph kernel.

    Returns:
        The compiled module, or None if compilation fails.
    """
    global _cuda_module, _backend_checked, _backend_available

    if _backend_checked:
        return _cuda_module

    _backend_checked = True

    # Check prerequisites
    if not torch.cuda.is_available():
        logger.info("radius_graph: CUDA not available, using brute-force backend")
        return None

    if not _KERNEL_PATH.exists():
        logger.warning(
            f"radius_graph: CUDA kernel source not found at {_KERNEL_PATH}, "
            "using brute-force backend"
        )
        return None

    try:
        from torch.utils.cpp_extension import load

        arch_list = _get_cuda_arch_list()
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        _cuda_module = load(
            name="geoembodied_radius_graph_cuda",
            sources=[str(_KERNEL_PATH)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",  # For profiling with nsight
            ],
            verbose=False,
        )

        _backend_available = True
        logger.info(
            f"radius_graph: CUDA kernel compiled successfully "
            f"(arch_list={arch_list})"
        )
        return _cuda_module

    except Exception as e:
        logger.warning(
            f"radius_graph: CUDA kernel compilation failed: {e}\n"
            "Falling back to brute-force backend."
        )
        return None


def get_radius_graph_backend() -> str:
    """Get the active radius_graph backend name.

    Returns:
        'cuda' if compiled CUDA kernel is available, else 'bruteforce'
    """
    _try_load_cuda_module()
    return "cuda" if _backend_available else "bruteforce"


def radius_graph_cuda(
    x: torch.Tensor,
    r: float,
    *,
    max_num_neighbors: int = 32,
    loop: bool = False,
    mask: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    points_per_batch: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to CUDA radius_graph kernel.

    Args:
        x: Point positions [N, 3] or [B*N, 3], float32, CUDA
        r: Radius cutoff
        max_num_neighbors: Max neighbors per query point
        loop: Include self-loops
        mask: [N] or [B*N] bool mask (True=valid, False=pad)
        batch_size: If provided, treat as batched with B=batch_size
        points_per_batch: Points per batch element (required if batch_size > 1)

    Returns:
        (row, col) edge index tensors in global indices
    """
    mod = _try_load_cuda_module()
    if mod is None:
        raise RuntimeError(
            "CUDA radius_graph kernel not available. "
            "Install PyTorch with CUDA support."
        )

    # Prepare mask tensor (empty if None)
    if mask is None:
        mask_t = torch.empty(0, dtype=torch.bool, device=x.device)
    else:
        mask_t = mask

    if batch_size is not None and batch_size > 1:
        assert points_per_batch is not None, \
            "points_per_batch required when batch_size > 1"
        return mod.radius_graph_cuda_batched(
            x.contiguous(),
            float(r),
            max_num_neighbors,
            mask_t.contiguous(),
            loop,
            batch_size,
            points_per_batch,
        )
    else:
        return mod.radius_graph_cuda(
            x.contiguous(),
            float(r),
            max_num_neighbors,
            mask_t.contiguous(),
            loop,
        )


# ═══════════════════════════════════════════════════════════════════
# Segment Reduce Backend
# ═══════════════════════════════════════════════════════════════════

_seg_reduce_module = None
_seg_reduce_checked = False
_seg_reduce_available = False

_SEG_REDUCE_KERNEL_PATH = _CSRC_DIR / "segment_reduce_kernel.cu"


def _try_load_segment_reduce_module():
    """Attempt JIT compilation of the CUDA segment_reduce kernel.

    Returns:
        The compiled module, or None if compilation fails.
    """
    global _seg_reduce_module, _seg_reduce_checked, _seg_reduce_available

    if _seg_reduce_checked:
        return _seg_reduce_module

    _seg_reduce_checked = True

    if not torch.cuda.is_available():
        logger.info("segment_reduce: CUDA not available")
        return None

    if not _SEG_REDUCE_KERNEL_PATH.exists():
        logger.warning(
            f"segment_reduce: CUDA kernel not found at {_SEG_REDUCE_KERNEL_PATH}"
        )
        return None

    try:
        from torch.utils.cpp_extension import load

        arch_list = _get_cuda_arch_list()
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        _seg_reduce_module = load(
            name="geoembodied_segment_reduce_cuda",
            sources=[str(_SEG_REDUCE_KERNEL_PATH)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )

        _seg_reduce_available = True
        logger.info(
            f"segment_reduce: CUDA kernel compiled successfully "
            f"(arch_list={arch_list})"
        )
        return _seg_reduce_module

    except Exception as e:
        logger.warning(
            f"segment_reduce: CUDA kernel compilation failed: {e}\n"
            "Falling back to PyTorch scatter_add."
        )
        return None


def segment_reduce_available() -> bool:
    """Check if CUDA segment reduce kernel is available."""
    _try_load_segment_reduce_module()
    return _seg_reduce_available


def sort_and_build_csr_cuda(
    col: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort edges by target and build CSR offsets (CUDA).

    Args:
        col: [E] int64, target node indices
        N: Number of nodes

    Returns:
        (col_sorted, perm, node_start, node_end) all int32 on CUDA
    """
    mod = _try_load_segment_reduce_module()
    if mod is None:
        raise RuntimeError("segment_reduce CUDA kernel not available")
    return mod.sort_and_build_csr(col.contiguous(), N)


def segment_reduce_sum_cuda(
    msg: torch.Tensor,
    perm: torch.Tensor,
    node_start: torch.Tensor,
    node_end: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Deterministic segment reduce sum (CUDA, no atomics).

    Args:
        msg: [E, C] float32, per-edge messages
        perm: [E] int32, sort permutation from sort_and_build_csr
        node_start: [N] int32, CSR start offsets
        node_end: [N] int32, CSR end offsets
        N: Number of nodes

    Returns:
        out: [N, C] float32, aggregated node features
    """
    mod = _try_load_segment_reduce_module()
    if mod is None:
        raise RuntimeError("segment_reduce CUDA kernel not available")
    return mod.segment_reduce_sum(msg.contiguous(), perm, node_start, node_end, N)


# ═══════════════════════════════════════════════════════════════════
# KNN Backend
# ═══════════════════════════════════════════════════════════════════

_knn_module = None
_knn_checked = False
_knn_available = False

_KNN_KERNEL_PATH = _CSRC_DIR / "knn_kernel.cu"


def _try_load_knn_module():
    """Attempt JIT compilation of the CUDA KNN kernel.

    Returns:
        The compiled module, or None if compilation fails.
    """
    global _knn_module, _knn_checked, _knn_available

    if _knn_checked:
        return _knn_module

    _knn_checked = True

    if not torch.cuda.is_available():
        logger.info("knn: CUDA not available")
        return None

    if not _KNN_KERNEL_PATH.exists():
        logger.warning(
            f"knn: CUDA kernel not found at {_KNN_KERNEL_PATH}"
        )
        return None

    try:
        from torch.utils.cpp_extension import load

        arch_list = _get_cuda_arch_list()
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        _knn_module = load(
            name="geoembodied_knn_cuda",
            sources=[str(_KNN_KERNEL_PATH)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )

        _knn_available = True
        logger.info(
            f"knn: CUDA kernel compiled successfully (arch_list={arch_list})"
        )
        return _knn_module

    except Exception as e:
        logger.warning(
            f"knn: CUDA kernel compilation failed: {e}\n"
            "Falling back to PyTorch brute-force backend."
        )
        return None


def knn_available() -> bool:
    """Check if CUDA KNN kernel is available."""
    _try_load_knn_module()
    return _knn_available


def knn_cuda(
    query: torch.Tensor,
    source: torch.Tensor,
    ptr_q: torch.Tensor,
    ptr_s: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-Nearest Neighbor search with ptr-based batch segmentation.

    Falls back to brute-force PyTorch if CUDA kernel unavailable.

    Args:
        query: Query points [N_q, 3], float32
        source: Source points [N_s, 3], float32
        ptr_q: [B+1] int64, CSR offsets for query batches
        ptr_s: [B+1] int64, CSR offsets for source batches
        k: Number of nearest neighbors

    Returns:
        indices: [N_q, k] int64, global indices into source
        dists: [N_q, k] float32, squared L2 distances
    """
    mod = _try_load_knn_module()

    if mod is not None and query.is_cuda:
        return mod.knn_cuda(
            query.contiguous().float(),
            source.contiguous().float(),
            ptr_q.contiguous().to(torch.int64),
            ptr_s.contiguous().to(torch.int64),
            k,
        )

    # CPU fallback: brute-force per batch
    return _knn_bruteforce(query, source, ptr_q, ptr_s, k)


def knn_self_cuda(
    points: torch.Tensor,
    ptr: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Self-KNN: find k nearest neighbors within each batch element.

    Note: Results include self (distance=0) as the first neighbor.
    Slice [:, 1:] to exclude self-loops.

    Args:
        points: [N_total, 3] packed point cloud
        ptr: [B+1] CSR offsets
        k: Number of neighbors (including self)

    Returns:
        indices: [N_total, k] int64
        dists: [N_total, k] float32
    """
    return knn_cuda(points, points, ptr, ptr, k)


def _knn_bruteforce(
    query: torch.Tensor,
    source: torch.Tensor,
    ptr_q: torch.Tensor,
    ptr_s: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch brute-force KNN fallback.

    Operates per-batch using ptr boundaries. No CUDA required.

    Args:
        query: [N_q, 3]
        source: [N_s, 3]
        ptr_q: [B+1]
        ptr_s: [B+1]
        k: neighbors

    Returns:
        indices: [N_q, k] int64, global source indices
        dists: [N_q, k] float32, squared L2 distances
    """
    B = ptr_q.shape[0] - 1
    N_q = query.shape[0]
    device = query.device

    all_indices = torch.full((N_q, k), -1, dtype=torch.int64, device=device)
    all_dists = torch.full((N_q, k), 1e30, dtype=torch.float32, device=device)

    for b in range(B):
        q_start, q_end = int(ptr_q[b]), int(ptr_q[b + 1])
        s_start, s_end = int(ptr_s[b]), int(ptr_s[b + 1])

        if q_end <= q_start or s_end <= s_start:
            continue

        q = query[q_start:q_end]     # [n_q, 3]
        s = source[s_start:s_end]    # [n_s, 3]

        # Pairwise squared distances: [n_q, n_s]
        # IMPORTANT: compute squared L2 directly, NOT via cdist(p=2).pow(2).
        # cdist(p=2) computes sqrt(Σ(xi-yi)²) internally, then .pow(2)
        # squares it back — this sqrt→square round-trip introduces ~1e-6
        # asymmetric FP error between original and rotated coordinates,
        # which gets amplified to ~1e-3 after downstream sqrt in pool.
        q_f = q.float().unsqueeze(1)   # [n_q, 1, 3]
        s_f = s.float().unsqueeze(0)   # [1, n_s, 3]
        diff = q_f - s_f               # [n_q, n_s, 3]
        dist = (diff * diff).sum(-1)   # [n_q, n_s]

        k_actual = min(k, dist.shape[1])
        topk_dists, topk_local = dist.topk(k_actual, dim=1, largest=False)

        # Convert local indices to global source indices
        all_indices[q_start:q_end, :k_actual] = topk_local + s_start
        all_dists[q_start:q_end, :k_actual] = topk_dists

    return all_indices, all_dists


# ═══════════════════════════════════════════════════════════════════
# FPS Backend
# ═══════════════════════════════════════════════════════════════════

_fps_module = None
_fps_checked = False
_fps_available = False

_FPS_KERNEL_PATH = _CSRC_DIR / "fps_kernel.cu"


def _try_load_fps_module():
    """Attempt JIT compilation of the CUDA FPS kernel.

    Returns:
        The compiled module, or None if compilation fails.
    """
    global _fps_module, _fps_checked, _fps_available

    if _fps_checked:
        return _fps_module

    _fps_checked = True

    if not torch.cuda.is_available():
        logger.info("fps: CUDA not available")
        return None

    if not _FPS_KERNEL_PATH.exists():
        logger.warning(
            f"fps: CUDA kernel not found at {_FPS_KERNEL_PATH}"
        )
        return None

    try:
        from torch.utils.cpp_extension import load

        arch_list = _get_cuda_arch_list()
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        _fps_module = load(
            name="geoembodied_fps_cuda",
            sources=[str(_FPS_KERNEL_PATH)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )

        _fps_available = True
        logger.info(
            f"fps: CUDA kernel compiled successfully (arch_list={arch_list})"
        )
        return _fps_module

    except Exception as e:
        logger.warning(
            f"fps: CUDA kernel compilation failed: {e}\n"
            "Falling back to PyTorch iterative backend."
        )
        return None


def fps_available() -> bool:
    """Check if CUDA FPS kernel is available."""
    _try_load_fps_module()
    return _fps_available


def fps_cuda(
    pos: torch.Tensor,
    ptr: torch.Tensor,
    k_per_batch: torch.Tensor,
) -> torch.Tensor:
    """Farthest Point Sampling with ptr-based batch segmentation.

    Selects k_per_batch[b] points from each batch element using FPS.
    Falls back to PyTorch iterative if CUDA kernel unavailable.

    GPU→CPU sync: ONE (.item() for output size allocation inside kernel).
    The FPS iteration loop itself is ZERO sync.

    Args:
        pos: [N_total, 3] float32, packed point cloud positions
        ptr: [B+1] int64, CSR batch offsets
        k_per_batch: [B] int64, number of samples per batch element

    Returns:
        indices: [N_out_total] int64, global indices of selected points
    """
    mod = _try_load_fps_module()

    if mod is not None and pos.is_cuda:
        return mod.fps_cuda(
            pos.contiguous().float(),
            ptr.contiguous().to(torch.int64),
            k_per_batch.contiguous().to(torch.int64),
        )

    # CPU fallback: iterative per-batch FPS
    return _fps_iterative_fallback(pos, ptr, k_per_batch)


def _fps_iterative_fallback(
    pos: torch.Tensor,
    ptr: torch.Tensor,
    k_per_batch: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch FPS fallback (iterative, per-batch).

    This is the O(N*K) iterative algorithm with GPU→CPU syncs.
    Used only when CUDA kernel is unavailable (CPU or compile failure).

    Args:
        pos: [N_total, 3]
        ptr: [B+1]
        k_per_batch: [B]

    Returns:
        indices: [N_out_total] int64
    """
    B = ptr.shape[0] - 1
    device = pos.device
    indices_list = []

    for b in range(B):
        start = int(ptr[b])
        end = int(ptr[b + 1])
        n_b = end - start
        k_b = int(k_per_batch[b])
        k_b = min(k_b, n_b)

        if k_b <= 0:
            continue

        local_pos = pos[start:end]  # [n_b, 3]

        if k_b >= n_b:
            indices_list.append(torch.arange(start, end, device=device))
            continue

        # Standard iterative FPS
        selected = torch.zeros(k_b, dtype=torch.long, device=device)
        selected[0] = 0  # first point
        min_dists = torch.full((n_b,), float('inf'), device=device)

        for i in range(k_b):
            current = local_pos[selected[i]]
            dist = (local_pos - current.unsqueeze(0)).pow(2).sum(-1)
            min_dists = torch.minimum(min_dists, dist)
            if i + 1 < k_b:
                selected[i + 1] = min_dists.argmax()

        indices_list.append(selected + start)

    if not indices_list:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(indices_list)

