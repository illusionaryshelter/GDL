// Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

// segment_reduce_kernel.cu — Deterministic segment reduce for GNN aggregation
//
// Replaces PyTorch scatter_add_ with a three-stage pipeline:
//   1. Sort edges by target node (CUB RadixSort)
//   2. Compute CSR node_start/end offsets
//   3. Warp-level segment reduce: one block per node, no atomics
//
// Key properties:
//   - Bit-exact deterministic (no floating-point atomic contention)
//   - Coalesced memory access (sorted order)
//   - Zero shared memory needed for degree ≤ 64
//   - Backward is trivial gather (no kernel needed)
//
// Target: sm_70+ (V100, T4, A100, Orin, H100)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <tuple>

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Compute CSR offsets from sorted target indices
// ═══════════════════════════════════════════════════════════════════

__global__ void compute_node_offsets_kernel(
    const int* __restrict__ col_sorted,   // [E] sorted target indices (int32)
    int* __restrict__ node_start,          // [N] output
    int* __restrict__ node_end,            // [N] output
    int E,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    int node = col_sorted[idx];
    if (node < 0 || node >= N) return;  // safety

    // Transition detection: mark segment boundaries
    if (idx == 0 || col_sorted[idx - 1] != node) {
        node_start[node] = idx;
    }
    if (idx == E - 1 || col_sorted[idx + 1] != node) {
        node_end[node] = idx + 1;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Segment reduce sum — one thread block per target node
// ═══════════════════════════════════════════════════════════════════
//
// Each block:
//   - Reads the segment [node_start[node], node_end[node]) of sorted edges
//   - Each thread accumulates a subset of channels
//   - Writes to out[node, c] — exclusive write, no atomics
//
// Block size = min(C, 32) for C ≤ 32 (one warp, hardware lockstep)
// For C > 32, multiple warps cooperate (still no conflict since
// each thread writes to disjoint out[node, c]).

__global__ void segment_reduce_sum_kernel(
    const float* __restrict__ msg,      // [E, C] pre-computed messages
    const int*   __restrict__ perm,     // [E] permutation: sorted→original edge order
    const int*   __restrict__ node_start,  // [N]
    const int*   __restrict__ node_end,    // [N]
    float*       __restrict__ out,      // [N, C] output (pre-zeroed)
    int N,
    int C
) {
    int node = blockIdx.x;
    if (node >= N) return;

    int start = node_start[node];
    int end   = node_end[node];
    if (start >= end) return;  // no edges to this node

    // Each thread handles channels [tid, tid+blockDim, tid+2*blockDim, ...]
    int tid = threadIdx.x;

    for (int c = tid; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int e = start; e < end; e++) {
            int orig_edge = perm[e];  // map back to original edge ordering
            acc += msg[orig_edge * C + c];
        }
        out[node * C + c] = acc;  // Exclusive write — NO atomic!
    }
}


// ═══════════════════════════════════════════════════════════════════
// Host function: sort_and_build_csr
// ═══════════════════════════════════════════════════════════════════

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
sort_and_build_csr(
    at::Tensor col,   // [E] int64, target node indices (COO)
    int N              // number of nodes
) {
    TORCH_CHECK(col.is_cuda(), "col must be on CUDA");
    TORCH_CHECK(col.dim() == 1, "col must be 1D");

    const int E = col.size(0);
    const auto device = col.device();

    if (E == 0) {
        auto empty_i32 = torch::empty({0}, torch::dtype(torch::kInt32).device(device));
        auto node_s = torch::zeros({N}, torch::dtype(torch::kInt32).device(device));
        auto node_e = torch::zeros({N}, torch::dtype(torch::kInt32).device(device));
        return std::make_tuple(empty_i32, empty_i32, node_s, node_e);
    }

    // Convert col from int64 to int32 for CUB sort
    auto col_i32 = col.to(torch::kInt32).contiguous();

    // Create identity permutation [0, 1, 2, ..., E-1]
    auto perm_in = torch::arange(E, torch::dtype(torch::kInt32).device(device));

    // Allocate outputs
    auto col_sorted = torch::empty({E}, torch::dtype(torch::kInt32).device(device));
    auto perm_out   = torch::empty({E}, torch::dtype(torch::kInt32).device(device));

    // ── CUB RadixSort: sort col (keys), carry perm (values) ──
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        col_i32.data_ptr<int>(),
        col_sorted.data_ptr<int>(),
        perm_in.data_ptr<int>(),
        perm_out.data_ptr<int>(),
        E
    );

    auto temp_storage = torch::empty(
        {static_cast<long>(temp_bytes)},
        torch::dtype(torch::kUInt8).device(device)
    );

    cub::DeviceRadixSort::SortPairs(
        static_cast<void*>(temp_storage.data_ptr<uint8_t>()),
        temp_bytes,
        col_i32.data_ptr<int>(),
        col_sorted.data_ptr<int>(),
        perm_in.data_ptr<int>(),
        perm_out.data_ptr<int>(),
        E
    );

    // ── Compute CSR offsets ──
    auto node_start = torch::zeros({N}, torch::dtype(torch::kInt32).device(device));
    auto node_end   = torch::zeros({N}, torch::dtype(torch::kInt32).device(device));

    int threads = 256;
    int blocks = (E + threads - 1) / threads;

    compute_node_offsets_kernel<<<blocks, threads>>>(
        col_sorted.data_ptr<int>(),
        node_start.data_ptr<int>(),
        node_end.data_ptr<int>(),
        E, N
    );

    return std::make_tuple(col_sorted, perm_out, node_start, node_end);
}


// ═══════════════════════════════════════════════════════════════════
// Host function: segment_reduce_sum
// ═══════════════════════════════════════════════════════════════════

at::Tensor segment_reduce_sum(
    at::Tensor msg,           // [E, C] float32 messages
    at::Tensor perm,          // [E] int32, sort permutation
    at::Tensor node_start,    // [N] int32
    at::Tensor node_end,      // [N] int32
    int N
) {
    TORCH_CHECK(msg.is_cuda(), "msg must be on CUDA");
    TORCH_CHECK(msg.dim() == 2, "msg must be [E, C]");
    TORCH_CHECK(msg.dtype() == torch::kFloat32, "msg must be float32");

    const int E = msg.size(0);
    const int C = msg.size(1);

    auto out = torch::zeros({N, C}, torch::dtype(torch::kFloat32).device(msg.device()));

    if (E == 0 || N == 0) return out;

    // Choose block size: one warp (32) for C ≤ 32, scale up for larger C
    int block_size = 32;  // one warp
    if (C > 32) block_size = 64;
    if (C > 64) block_size = 128;
    if (C > 128) block_size = 256;

    segment_reduce_sum_kernel<<<N, block_size>>>(
        msg.data_ptr<float>(),
        perm.data_ptr<int>(),
        node_start.data_ptr<int>(),
        node_end.data_ptr<int>(),
        out.data_ptr<float>(),
        N, C
    );

    return out;
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11 module
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort_and_build_csr", &sort_and_build_csr,
          "Sort edges by target + build CSR offsets (CUDA)",
          py::arg("col"),
          py::arg("N"));
    m.def("segment_reduce_sum", &segment_reduce_sum,
          "Deterministic segment reduce sum (CUDA, no atomics)",
          py::arg("msg"),
          py::arg("perm"),
          py::arg("node_start"),
          py::arg("node_end"),
          py::arg("N"));
}
