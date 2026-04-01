// Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

// radius_graph_kernel.cu — Hash-grid based fixed-radius nearest neighbor search
//
// Algorithm (inspired by FRNN / torchCompactRadius / Hoetzlein):
//   1. Compute grid cell index for each point: cell_idx = hash(floor(pos/cell_size))
//   2. Sort points by cell_idx using CUB radix sort
//   3. Compute cell start/end offsets via prefix scan
//   4. For each query point, iterate neighboring 3x3x3 cells, check distances
//   5. Maintain a per-thread MinK heap of K nearest within radius
//
// Design choices for GeoEmbodied:
//   - Native algebraic masking: mask=False points have cell_idx = INT_MAX (sorted to end)
//   - Batched: batch dimension handled via per-batch grid parameters
//   - Output: COO (row, col) format for direct consumption by SE3Conv
//   - 3D-only (our domain is always R³)
//
// Target: sm_70+ (V100, T4, A100, 4090, H100)
// Build: TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cmath>
#include <tuple>

// ═══════════════════════════════════════════════════════════════════
// MinK: thread-local min-heap for K nearest neighbors (register-only)
// ═══════════════════════════════════════════════════════════════════

template <int K>
struct MinK {
    float dists[K];
    int   idxs[K];
    int   count;

    __device__ __forceinline__ MinK() : count(0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            dists[i] = 1e30f;
            idxs[i]  = -1;
        }
    }

    __device__ __forceinline__ void add(float d, int idx) {
        // If full and d >= max, skip
        if (count >= K && d >= dists[0]) return;

        // Replace max (root of max-heap at index 0)
        if (count >= K) {
            dists[0] = d;
            idxs[0]  = idx;
            sift_down(0);
        } else {
            dists[count] = d;
            idxs[count]  = idx;
            ++count;
            if (count == K) build_heap();
        }
    }

    __device__ __forceinline__ void build_heap() {
        for (int i = K / 2 - 1; i >= 0; --i) {
            sift_down(i);
        }
    }

    __device__ __forceinline__ void sift_down(int i) {
        while (true) {
            int left  = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;
            if (left  < K && dists[left]  > dists[largest]) largest = left;
            if (right < K && dists[right] > dists[largest]) largest = right;
            if (largest == i) break;
            // swap
            float td = dists[i]; dists[i] = dists[largest]; dists[largest] = td;
            int   ti = idxs[i];  idxs[i]  = idxs[largest];  idxs[largest]  = ti;
            i = largest;
        }
    }
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Compute cell indices for each point
// ═══════════════════════════════════════════════════════════════════

__global__ void compute_cell_indices_kernel(
    const float* __restrict__ points,    // [N, 3]
    const bool*  __restrict__ mask,      // [N] or nullptr
    int*   __restrict__ cell_indices,    // [N]  output
    int*   __restrict__ point_indices,   // [N]  output: 0..N-1
    int N,
    float inv_cell_size,
    float min_x, float min_y, float min_z,
    int grid_res_x, int grid_res_y, int grid_res_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Initialize point index (before sort)
    point_indices[idx] = idx;

    // Masked points → send to end of sorted order
    if (mask != nullptr && !mask[idx]) {
        cell_indices[idx] = INT_MAX;
        return;
    }

    float px = points[idx * 3 + 0];
    float py = points[idx * 3 + 1];
    float pz = points[idx * 3 + 2];

    int cx = __float2int_rd((px - min_x) * inv_cell_size);
    int cy = __float2int_rd((py - min_y) * inv_cell_size);
    int cz = __float2int_rd((pz - min_z) * inv_cell_size);

    // Clamp to grid bounds
    cx = max(0, min(cx, grid_res_x - 1));
    cy = max(0, min(cy, grid_res_y - 1));
    cz = max(0, min(cz, grid_res_z - 1));

    cell_indices[idx] = (cx * grid_res_y + cy) * grid_res_z + cz;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Compute cell start/end offsets from sorted cell indices
// ═══════════════════════════════════════════════════════════════════

__global__ void compute_cell_offsets_kernel(
    const int* __restrict__ sorted_cell_indices,   // [N]
    int* __restrict__ cell_start,                   // [num_cells]
    int* __restrict__ cell_end,                     // [num_cells]
    int N,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int cell = sorted_cell_indices[idx];

    // Skip masked points (INT_MAX)
    if (cell >= num_cells) return;

    // If first element or different from previous, mark start
    if (idx == 0 || sorted_cell_indices[idx - 1] != cell) {
        cell_start[cell] = idx;
    }
    // If last element or different from next, mark end
    if (idx == N - 1 || sorted_cell_indices[idx + 1] != cell) {
        cell_end[cell] = idx + 1;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Query neighbors — the core radius search kernel
// ═══════════════════════════════════════════════════════════════════

template <int MAX_K>
__global__ void find_neighbors_kernel(
    const float* __restrict__ points,              // [N, 3] ORIGINAL positions
    const float* __restrict__ sorted_points,       // [N, 3] sorted by cell
    const int*   __restrict__ sorted_point_indices, // [N] original idx of sorted pts
    const int*   __restrict__ cell_start,           // [num_cells]
    const int*   __restrict__ cell_end,             // [num_cells]
    const bool*  __restrict__ mask,                 // [N] or nullptr
    int* __restrict__ out_row,                      // [N * MAX_K] output
    int* __restrict__ out_col,                      // [N * MAX_K] output
    int* __restrict__ out_count,                    // [1] atomic counter
    int N,
    float radius_sq,
    float inv_cell_size,
    float min_x, float min_y, float min_z,
    int grid_res_x, int grid_res_y, int grid_res_z,
    bool include_self,
    int max_total_edges
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= N) return;

    // Skip masked query points
    if (mask != nullptr && !mask[query_idx]) return;

    float qx = points[query_idx * 3 + 0];
    float qy = points[query_idx * 3 + 1];
    float qz = points[query_idx * 3 + 2];

    // Compute cell of query point
    int qcx = __float2int_rd((qx - min_x) * inv_cell_size);
    int qcy = __float2int_rd((qy - min_y) * inv_cell_size);
    int qcz = __float2int_rd((qz - min_z) * inv_cell_size);

    // Local MinK heap (in registers)
    MinK<MAX_K> mink;

    // Iterate 3x3x3 neighboring cells
    for (int dx = -1; dx <= 1; ++dx) {
        int cx = qcx + dx;
        if (cx < 0 || cx >= grid_res_x) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int cy = qcy + dy;
            if (cy < 0 || cy >= grid_res_y) continue;
            for (int dz = -1; dz <= 1; ++dz) {
                int cz = qcz + dz;
                if (cz < 0 || cz >= grid_res_z) continue;

                int cell = (cx * grid_res_y + cy) * grid_res_z + cz;
                int start = cell_start[cell];
                int end   = cell_end[cell];

                for (int j = start; j < end; ++j) {
                    int orig_idx = sorted_point_indices[j];

                    // Skip self if no self-loops
                    if (!include_self && orig_idx == query_idx) continue;

                    float sx = sorted_points[j * 3 + 0];
                    float sy = sorted_points[j * 3 + 1];
                    float sz = sorted_points[j * 3 + 2];

                    float ddx = qx - sx;
                    float ddy = qy - sy;
                    float ddz = qz - sz;
                    float dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                    if (dist_sq < radius_sq) {
                        mink.add(dist_sq, orig_idx);
                    }
                }
            }
        }
    }

    // Write results using atomic counter for compacted output
    for (int k = 0; k < mink.count; ++k) {
        if (mink.idxs[k] >= 0) {
            int pos = atomicAdd(out_count, 1);
            if (pos < max_total_edges) {
                out_row[pos] = mink.idxs[k];   // source (neighbor)
                out_col[pos] = query_idx;        // target (query point)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Host function: radius_graph_cuda
// ═══════════════════════════════════════════════════════════════════

std::tuple<at::Tensor, at::Tensor> radius_graph_cuda(
    at::Tensor points,       // [N, 3] float32, CUDA
    double radius,
    int max_num_neighbors,
    at::Tensor mask,         // [N] bool, CUDA (or empty)
    bool loop
) {
    TORCH_CHECK(points.is_cuda(), "points must be on CUDA");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be [N, 3]");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");

    const int N = points.size(0);
    const auto device = points.device();
    const float r = static_cast<float>(radius);

    // Early return for degenerate cases
    if (N <= 1 || r <= 0.0f) {
        auto empty = torch::empty({0}, torch::dtype(torch::kLong).device(device));
        return std::make_tuple(empty, empty);
    }

    const float r_sq = r * r;
    const float cell_size = r;  // cell_size = radius → only check 3x3x3 neighbors
    const float inv_cell_size = 1.0f / cell_size;

    const bool has_mask = mask.numel() > 0;

    // ── Step 1: Compute bounding box (on GPU, one sync) ──
    auto pts_min = std::get<0>(points.min(0));  // [3]
    auto pts_max = std::get<0>(points.max(0));  // [3]
    float min_x = pts_min[0].item<float>() - 1e-5f;
    float min_y = pts_min[1].item<float>() - 1e-5f;
    float min_z = pts_min[2].item<float>() - 1e-5f;
    float max_x = pts_max[0].item<float>() + 1e-5f;
    float max_y = pts_max[1].item<float>() + 1e-5f;
    float max_z = pts_max[2].item<float>() + 1e-5f;

    int grid_res_x = static_cast<int>(std::ceil((max_x - min_x) * inv_cell_size)) + 1;
    int grid_res_y = static_cast<int>(std::ceil((max_y - min_y) * inv_cell_size)) + 1;
    int grid_res_z = static_cast<int>(std::ceil((max_z - min_z) * inv_cell_size)) + 1;

    // Safety cap for degenerate point clouds
    grid_res_x = std::min(grid_res_x, 1024);
    grid_res_y = std::min(grid_res_y, 1024);
    grid_res_z = std::min(grid_res_z, 1024);

    int num_cells = grid_res_x * grid_res_y * grid_res_z;

    // ── Step 2: Compute cell indices ──
    auto cell_indices   = torch::empty({N}, torch::dtype(torch::kInt32).device(device));
    auto point_indices  = torch::empty({N}, torch::dtype(torch::kInt32).device(device));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_cell_indices_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        has_mask ? mask.data_ptr<bool>() : nullptr,
        cell_indices.data_ptr<int>(),
        point_indices.data_ptr<int>(),
        N,
        inv_cell_size,
        min_x, min_y, min_z,
        grid_res_x, grid_res_y, grid_res_z
    );

    // ── Step 3: Sort by cell index using CUB radix sort ──
    auto sorted_cell_indices  = torch::empty_like(cell_indices);
    auto sorted_point_indices = torch::empty_like(point_indices);

    // CUB sort needs temp storage
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        cell_indices.data_ptr<int>(),
        sorted_cell_indices.data_ptr<int>(),
        point_indices.data_ptr<int>(),
        sorted_point_indices.data_ptr<int>(),
        N
    );

    auto temp_storage = torch::empty(
        {static_cast<long>(temp_storage_bytes)},
        torch::dtype(torch::kUInt8).device(device)
    );

    cub::DeviceRadixSort::SortPairs(
        static_cast<void*>(temp_storage.data_ptr<uint8_t>()),
        temp_storage_bytes,
        cell_indices.data_ptr<int>(),
        sorted_cell_indices.data_ptr<int>(),
        point_indices.data_ptr<int>(),
        sorted_point_indices.data_ptr<int>(),
        N
    );

    // ── Step 4: Gather sorted points for coalesced memory access ──
    auto sorted_points = points.index_select(0, sorted_point_indices.to(torch::kLong));

    // ── Step 5: Compute cell start/end offsets ──
    auto cell_start = torch::full({num_cells}, 0, torch::dtype(torch::kInt32).device(device));
    auto cell_end   = torch::full({num_cells}, 0, torch::dtype(torch::kInt32).device(device));

    compute_cell_offsets_kernel<<<blocks, threads>>>(
        sorted_cell_indices.data_ptr<int>(),
        cell_start.data_ptr<int>(),
        cell_end.data_ptr<int>(),
        N,
        num_cells
    );

    // ── Step 6: Find neighbors ──
    int max_total_edges = N * max_num_neighbors;  // Worst case
    auto out_row   = torch::empty({max_total_edges}, torch::dtype(torch::kInt32).device(device));
    auto out_col   = torch::empty({max_total_edges}, torch::dtype(torch::kInt32).device(device));
    auto out_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(device));

    // Dispatch based on max_num_neighbors (template K)
    // We use K=32 as default, with fallback for other values
    if (max_num_neighbors <= 16) {
        find_neighbors_kernel<16><<<blocks, threads>>>(
            points.data_ptr<float>(),
            sorted_points.data_ptr<float>(),
            sorted_point_indices.data_ptr<int>(),
            cell_start.data_ptr<int>(),
            cell_end.data_ptr<int>(),
            has_mask ? mask.data_ptr<bool>() : nullptr,
            out_row.data_ptr<int>(),
            out_col.data_ptr<int>(),
            out_count.data_ptr<int>(),
            N, r_sq, inv_cell_size,
            min_x, min_y, min_z,
            grid_res_x, grid_res_y, grid_res_z,
            loop, max_total_edges
        );
    } else if (max_num_neighbors <= 32) {
        find_neighbors_kernel<32><<<blocks, threads>>>(
            points.data_ptr<float>(),
            sorted_points.data_ptr<float>(),
            sorted_point_indices.data_ptr<int>(),
            cell_start.data_ptr<int>(),
            cell_end.data_ptr<int>(),
            has_mask ? mask.data_ptr<bool>() : nullptr,
            out_row.data_ptr<int>(),
            out_col.data_ptr<int>(),
            out_count.data_ptr<int>(),
            N, r_sq, inv_cell_size,
            min_x, min_y, min_z,
            grid_res_x, grid_res_y, grid_res_z,
            loop, max_total_edges
        );
    } else {
        find_neighbors_kernel<64><<<blocks, threads>>>(
            points.data_ptr<float>(),
            sorted_points.data_ptr<float>(),
            sorted_point_indices.data_ptr<int>(),
            cell_start.data_ptr<int>(),
            cell_end.data_ptr<int>(),
            has_mask ? mask.data_ptr<bool>() : nullptr,
            out_row.data_ptr<int>(),
            out_col.data_ptr<int>(),
            out_count.data_ptr<int>(),
            N, r_sq, inv_cell_size,
            min_x, min_y, min_z,
            grid_res_x, grid_res_y, grid_res_z,
            loop, max_total_edges
        );
    }

    // ── Step 7: Compact output ──
    int num_edges = out_count.item<int>();
    num_edges = std::min(num_edges, max_total_edges);

    auto row = out_row.narrow(0, 0, num_edges).to(torch::kLong);
    auto col = out_col.narrow(0, 0, num_edges).to(torch::kLong);

    return std::make_tuple(row, col);
}


// ═══════════════════════════════════════════════════════════════════
// Batched version: handles [B*N, 3] with per-batch grid
// ═══════════════════════════════════════════════════════════════════

std::tuple<at::Tensor, at::Tensor> radius_graph_cuda_batched(
    at::Tensor points,       // [B*N, 3]
    double radius,
    int max_num_neighbors,
    at::Tensor mask,         // [B*N] bool (or empty)
    bool loop,
    int batch_size,
    int points_per_batch
) {
    // Process each batch element and concatenate
    // This is simple but correct. Optimization: single-kernel with batch-offset.
    std::vector<at::Tensor> all_rows, all_cols;

    for (int b = 0; b < batch_size; ++b) {
        int offset = b * points_per_batch;
        auto pts_b = points.narrow(0, offset, points_per_batch);
        auto mask_b = mask.numel() > 0
            ? mask.narrow(0, offset, points_per_batch)
            : mask;  // empty

        auto [row_local, col_local] = radius_graph_cuda(
            pts_b, radius, max_num_neighbors, mask_b, loop
        );

        // Offset to global indices
        all_rows.push_back(row_local + offset);
        all_cols.push_back(col_local + offset);
    }

    return std::make_tuple(torch::cat(all_rows), torch::cat(all_cols));
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11 module
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radius_graph_cuda", &radius_graph_cuda,
          "Radius graph using hash-grid (single point cloud, CUDA)",
          py::arg("points"),
          py::arg("radius"),
          py::arg("max_num_neighbors"),
          py::arg("mask"),
          py::arg("loop"));
    m.def("radius_graph_cuda_batched", &radius_graph_cuda_batched,
          "Batched radius graph using hash-grid (CUDA)",
          py::arg("points"),
          py::arg("radius"),
          py::arg("max_num_neighbors"),
          py::arg("mask"),
          py::arg("loop"),
          py::arg("batch_size"),
          py::arg("points_per_batch"));
}
