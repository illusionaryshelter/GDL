// Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

// fps_kernel.cu — Farthest Point Sampling for packed point clouds
//
// Algorithm: Iterative FPS with block-level argmax reduction
//   1. Start from first point in batch (deterministic seed)
//   2. For each iteration k:
//      a. Each thread computes dist(selected[k], point[p]) for its
//         assigned points via grid-stride loop
//      b. Updates min_distances[p] = min(min_distances[p], new_dist)
//      c. Finds local argmax(min_distances) for its assigned points
//      d. Warp-level __shfl_down_sync to find warp-max
//      e. Block-level shared memory reduction to find global argmax
//      f. Thread 0 writes selected[k+1] to global memory (GPU-only!)
//
// Memory: O(N_total) — only min_distances[N] + shared_mem per block
//   NEVER allocates [N, N] distance matrix.
//
// GPU→CPU sync: ZERO in the FPS loop itself.
//
// Design choices:
//   - One block per batch element: blockIdx.x = batch index
//   - block_size threads per block (template parameter, power of 2)
//   - Grid-stride loop for N > block_size
//   - All indices use int (32-bit) internally for __shfl_down_sync
//     compatibility. int64_t only at I/O boundaries.
//   - ptr-based batch segmentation: ptr[b]..ptr[b+1] defines batch b
//
// Target: sm_70+ (V100, T4, A100, 4090, H100, Orin)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════
// FPS Kernel — uses int (32-bit) for all internal indices
// ═══════════════════════════════════════════════════════════════════

template <int block_size>
__global__ void fps_kernel(
    const float* __restrict__ pos,         // [N_total, 3]
    const int* __restrict__ ptr,           // [B+1] CSR offsets (int32)
    const int* __restrict__ k_per_batch,   // [B] samples per batch (int32)
    float* __restrict__ min_distances,     // [N_total] temp (init to inf)
    int* __restrict__ out_indices,         // [N_out_total] output (int32)
    const int* __restrict__ out_ptr        // [B+1] output CSR offsets (int32)
) {
    // Shared memory for block-level reduction
    extern __shared__ char shared_buf[];
    float* smem_dist = (float*)shared_buf;             // [block_size]
    int*   smem_idx  = (int*)&smem_dist[block_size];   // [block_size]

    const int b = blockIdx.x;        // Batch index
    const int tid = threadIdx.x;

    // Batch boundaries (all from GPU global memory, no CPU sync)
    const int start = ptr[b];
    const int end   = ptr[b + 1];
    const int n_b   = end - start;
    const int k_b   = k_per_batch[b];

    // Output offset
    const int out_start = out_ptr[b];

    // Degenerate
    if (n_b <= 0 || k_b <= 0) return;

    const int actual_k = (k_b < n_b) ? k_b : n_b;

    // Deterministic seed: first point in batch
    int selected = start;

    if (tid == 0) {
        out_indices[out_start] = selected;
    }

    // ── Main FPS loop ──
    for (int k = 1; k < actual_k; ++k) {
        // Selected point coordinates
        float sx = pos[selected * 3 + 0];
        float sy = pos[selected * 3 + 1];
        float sz = pos[selected * 3 + 2];

        // Thread-local best
        float thread_max_dist = -1.0f;
        int   thread_max_idx  = start;

        // Grid-stride loop
        for (int p = start + tid; p < end; p += block_size) {
            float dx = pos[p * 3 + 0] - sx;
            float dy = pos[p * 3 + 1] - sy;
            float dz = pos[p * 3 + 2] - sz;
            float dist2 = dx * dx + dy * dy + dz * dz;

            // Update min distance to nearest selected point
            float prev_min = min_distances[p];
            float new_min = fminf(dist2, prev_min);
            min_distances[p] = new_min;

            // Argmax of min_distances
            if (new_min > thread_max_dist) {
                thread_max_dist = new_min;
                thread_max_idx  = p;
            }
        }

        // ── Warp-level reduction ──
        // All 32 threads in each warp participate → safe 0xFFFFFFFF mask
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_dist = __shfl_down_sync(0xFFFFFFFF, thread_max_dist, offset);
            int   other_idx  = __shfl_down_sync(0xFFFFFFFF, thread_max_idx,  offset);
            if (other_dist > thread_max_dist) {
                thread_max_dist = other_dist;
                thread_max_idx  = other_idx;
            }
        }

        // ── Block-level reduction via shared memory binary tree ──
        // (NOT warp shuffle — avoids deadlock when num_warps < 32)
        //
        // Step 1: Lane 0 of each warp writes its result to shared memory
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) {
            smem_dist[warp_id] = thread_max_dist;
            smem_idx[warp_id]  = thread_max_idx;
        }
        __syncthreads();

        // Step 2: Binary tree reduction in shared memory
        // All threads participate in sync, but only active ones do work
        for (int s = block_size / 64; s > 0; s >>= 1) {
            // block_size/64 = num_warps/2 = first reduction step
            if (tid < s) {
                if (smem_dist[tid + s] > smem_dist[tid]) {
                    smem_dist[tid] = smem_dist[tid + s];
                    smem_idx[tid]  = smem_idx[tid + s];
                }
            }
            __syncthreads();
        }

        // All threads read next selected from shared memory (GPU-only!)
        selected = smem_idx[0];

        // Thread 0 writes to output
        if (tid == 0) {
            out_indices[out_start + k] = selected;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Host function: fps_cuda
// ═══════════════════════════════════════════════════════════════════

at::Tensor fps_cuda(
    at::Tensor pos,           // [N_total, 3] float32
    at::Tensor ptr,           // [B+1] int64
    at::Tensor k_per_batch    // [B] int64
) {
    TORCH_CHECK(pos.is_cuda(), "pos must be on CUDA");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, "pos must be [N, 3]");
    TORCH_CHECK(pos.dtype() == torch::kFloat32, "pos must be float32");

    const int N_total = pos.size(0);
    const int B = ptr.size(0) - 1;
    const auto device = pos.device();

    // Convert to int32 for kernel (ptr and k_per_batch are small)
    auto ptr_i32 = ptr.to(torch::kInt32);
    auto k_i32 = k_per_batch.to(torch::kInt32);

    // Compute output offsets on GPU (int32)
    auto out_ptr = torch::zeros({B + 1}, torch::dtype(torch::kInt32).device(device));
    out_ptr.index({torch::indexing::Slice(1, torch::indexing::None)}) =
        k_i32.cumsum(0);

    // ONE GPU→CPU sync: get total output size
    const int N_out = out_ptr[-1].item<int>();

    if (N_out <= 0 || N_total <= 0) {
        return torch::empty({0}, torch::dtype(torch::kInt64).device(device));
    }

    // Allocate output (int32 internally) and temp buffer
    auto out_i32 = torch::empty({N_out}, torch::dtype(torch::kInt32).device(device));
    auto min_distances = torch::full({N_total}, 1e30f,
        torch::dtype(torch::kFloat32).device(device));

    // Launch: one block per batch element, 512 threads
    constexpr int BLOCK_SIZE = 512;
    size_t shared_mem = BLOCK_SIZE * sizeof(float) + BLOCK_SIZE * sizeof(int);

    fps_kernel<BLOCK_SIZE><<<B, BLOCK_SIZE, shared_mem>>>(
        pos.data_ptr<float>(),
        ptr_i32.data_ptr<int>(),
        k_i32.data_ptr<int>(),
        min_distances.data_ptr<float>(),
        out_i32.data_ptr<int>(),
        out_ptr.data_ptr<int>()
    );

    // Convert output to int64 (PyTorch convention for indices)
    return out_i32.to(torch::kInt64);
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11 module
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fps_cuda", &fps_cuda,
          "Farthest Point Sampling with ptr-based batch segmentation (CUDA)\n"
          "\n"
          "Args:\n"
          "    pos: [N_total, 3] float32, CUDA — packed point positions\n"
          "    ptr: [B+1] int64, CUDA — CSR batch offsets\n"
          "    k_per_batch: [B] int64, CUDA — number of samples per batch\n"
          "\n"
          "Returns:\n"
          "    indices: [N_out_total] int64 — global indices of selected points",
          py::arg("pos"),
          py::arg("ptr"),
          py::arg("k_per_batch"));
}
