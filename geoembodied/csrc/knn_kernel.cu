// Copyright 2026 GeoEmbodied Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

// knn_kernel.cu — Batched K-Nearest Neighbor search for packed point clouds
//
// Algorithm: Brute-force column-wise distance + per-query max-heap (MinK)
//   For each query point, compute squared distances to all source points
//   within the same batch element (delimited by ptr), maintain a
//   register-resident MinK heap of the K nearest.
//
// Design choices:
//   - ptr-based batch segmentation: ptr[b]..ptr[b+1] defines batch b
//   - Thread-per-query: one CUDA thread handles one query point
//   - MinK<K> heap (same as radius_graph) for register-only top-K
//   - Output: [N_q, K] indices into source (global indices)
//   - Distances output: [N_q, K] squared L2 distances
//   - Supports asymmetric query/source (e.g. FPS downsampled queries)
//
// Complexity: O(N_q * N_s_per_batch) — optimal for N_s ≤ 100K per batch
//   For larger point clouds, hash-grid KNN should be used instead.
//
// Target: sm_70+ (V100, T4, A100, 4090, H100)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>

// ═══════════════════════════════════════════════════════════════════
// MinK: thread-local max-heap for K nearest neighbors (register-only)
// Same structure as radius_graph_kernel.cu for consistency.
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

        if (count >= K) {
            // Replace max (root of max-heap)
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
            float td = dists[i]; dists[i] = dists[largest]; dists[largest] = td;
            int   ti = idxs[i];  idxs[i]  = idxs[largest];  idxs[largest]  = ti;
            i = largest;
        }
    }

    // Sort heap results by distance (ascending) for deterministic output
    __device__ __forceinline__ void sort_ascending() {
        // Simple insertion sort — K is small (≤64), runs in registers
        for (int i = 1; i < count; ++i) {
            float key_d = dists[i];
            int   key_i = idxs[i];
            int j = i - 1;
            while (j >= 0 && dists[j] > key_d) {
                dists[j + 1] = dists[j];
                idxs[j + 1]  = idxs[j];
                --j;
            }
            dists[j + 1] = key_d;
            idxs[j + 1]  = key_i;
        }
    }
};


// ═══════════════════════════════════════════════════════════════════
// KNN Kernel: brute-force with ptr-based batch segmentation
//
// Each thread processes one query point.
// ptr_q and ptr_s define batch boundaries for query and source.
// batch_map_q[i] = b maps query point i to batch b.
// ═══════════════════════════════════════════════════════════════════

template <int MAX_K>
__global__ void knn_kernel(
    const float* __restrict__ query,      // [N_q, 3]
    const float* __restrict__ source,     // [N_s, 3]
    const int64_t* __restrict__ ptr_q,    // [B+1] CSR offsets for query
    const int64_t* __restrict__ ptr_s,    // [B+1] CSR offsets for source
    const int*  __restrict__ batch_map_q, // [N_q] batch index per query
    int64_t* __restrict__ out_indices,    // [N_q, K] output: global source indices
    float*   __restrict__ out_dists,      // [N_q, K] output: squared distances
    int N_q,
    int K,
    int B
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= N_q) return;

    // Determine which batch this query belongs to
    int b = batch_map_q[qi];

    // Source range for this batch
    int64_t s_start = ptr_s[b];
    int64_t s_end   = ptr_s[b + 1];

    float qx = query[qi * 3 + 0];
    float qy = query[qi * 3 + 1];
    float qz = query[qi * 3 + 2];

    // Local MinK heap
    MinK<MAX_K> mink;

    // Brute-force distance to all source points in this batch
    for (int64_t si = s_start; si < s_end; ++si) {
        float sx = source[si * 3 + 0];
        float sy = source[si * 3 + 1];
        float sz = source[si * 3 + 2];

        float dx = qx - sx;
        float dy = qy - sy;
        float dz = qz - sz;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        mink.add(dist_sq, static_cast<int>(si));  // global source index
    }

    // Sort by distance (ascending) for deterministic output
    mink.sort_ascending();

    // Write output [N_q, K] — row-major
    int base = qi * K;
    for (int k = 0; k < K; ++k) {
        if (k < mink.count) {
            out_indices[base + k] = static_cast<int64_t>(mink.idxs[k]);
            out_dists[base + k]   = mink.dists[k];
        } else {
            // Fewer than K points in this batch — pad with -1 / inf
            out_indices[base + k] = -1;
            out_dists[base + k]   = 1e30f;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Build batch_map: expand ptr → per-point batch assignment
// Runs on GPU, one thread per source range.
// ═══════════════════════════════════════════════════════════════════

__global__ void build_batch_map_kernel(
    const int64_t* __restrict__ ptr,  // [B+1]
    int*   __restrict__ batch_map,    // [N_total]
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int64_t start = ptr[b];
    int64_t end   = ptr[b + 1];
    for (int64_t i = start; i < end; ++i) {
        batch_map[i] = b;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Host function: knn_cuda
// ═══════════════════════════════════════════════════════════════════

std::tuple<at::Tensor, at::Tensor> knn_cuda(
    at::Tensor query,       // [N_q, 3] float32, CUDA
    at::Tensor source,      // [N_s, 3] float32, CUDA
    at::Tensor ptr_q,       // [B+1] int64, CUDA — CSR offsets for query
    at::Tensor ptr_s,       // [B+1] int64, CUDA — CSR offsets for source
    int64_t k               // Number of nearest neighbors
) {
    TORCH_CHECK(query.is_cuda(), "query must be on CUDA");
    TORCH_CHECK(source.is_cuda(), "source must be on CUDA");
    TORCH_CHECK(query.dim() == 2 && query.size(1) == 3, "query must be [N_q, 3]");
    TORCH_CHECK(source.dim() == 2 && source.size(1) == 3, "source must be [N_s, 3]");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(source.dtype() == torch::kFloat32, "source must be float32");
    TORCH_CHECK(ptr_q.dtype() == torch::kInt64, "ptr_q must be int64");
    TORCH_CHECK(ptr_s.dtype() == torch::kInt64, "ptr_s must be int64");
    TORCH_CHECK(ptr_q.size(0) == ptr_s.size(0),
        "ptr_q and ptr_s must have same length (B+1)");

    const int N_q = query.size(0);
    const int B = ptr_q.size(0) - 1;
    const int K = static_cast<int>(k);
    const auto device = query.device();

    // Handle degenerate cases
    if (N_q == 0 || K <= 0) {
        auto empty_idx = torch::empty({0, K}, torch::dtype(torch::kInt64).device(device));
        auto empty_dist = torch::empty({0, K}, torch::dtype(torch::kFloat32).device(device));
        return std::make_tuple(empty_idx, empty_dist);
    }

    // Build batch_map for query points: [N_q] → batch index
    auto batch_map_q = torch::empty({N_q}, torch::dtype(torch::kInt32).device(device));
    {
        int threads = 256;
        int blocks = (B + threads - 1) / threads;
        build_batch_map_kernel<<<blocks, threads>>>(
            ptr_q.data_ptr<int64_t>(),
            batch_map_q.data_ptr<int>(),
            B
        );
    }

    // Allocate output
    auto out_indices = torch::full(
        {N_q, K}, -1, torch::dtype(torch::kInt64).device(device));
    auto out_dists = torch::full(
        {N_q, K}, 1e30f, torch::dtype(torch::kFloat32).device(device));

    // Launch KNN kernel
    int threads = 256;
    int blocks = (N_q + threads - 1) / threads;

    // Dispatch based on K (template parameter determines register usage)
    if (K <= 4) {
        knn_kernel<4><<<blocks, threads>>>(
            query.data_ptr<float>(), source.data_ptr<float>(),
            ptr_q.data_ptr<int64_t>(), ptr_s.data_ptr<int64_t>(),
            batch_map_q.data_ptr<int>(),
            out_indices.data_ptr<int64_t>(), out_dists.data_ptr<float>(),
            N_q, K, B);
    } else if (K <= 8) {
        knn_kernel<8><<<blocks, threads>>>(
            query.data_ptr<float>(), source.data_ptr<float>(),
            ptr_q.data_ptr<int64_t>(), ptr_s.data_ptr<int64_t>(),
            batch_map_q.data_ptr<int>(),
            out_indices.data_ptr<int64_t>(), out_dists.data_ptr<float>(),
            N_q, K, B);
    } else if (K <= 16) {
        knn_kernel<16><<<blocks, threads>>>(
            query.data_ptr<float>(), source.data_ptr<float>(),
            ptr_q.data_ptr<int64_t>(), ptr_s.data_ptr<int64_t>(),
            batch_map_q.data_ptr<int>(),
            out_indices.data_ptr<int64_t>(), out_dists.data_ptr<float>(),
            N_q, K, B);
    } else if (K <= 32) {
        knn_kernel<32><<<blocks, threads>>>(
            query.data_ptr<float>(), source.data_ptr<float>(),
            ptr_q.data_ptr<int64_t>(), ptr_s.data_ptr<int64_t>(),
            batch_map_q.data_ptr<int>(),
            out_indices.data_ptr<int64_t>(), out_dists.data_ptr<float>(),
            N_q, K, B);
    } else {
        knn_kernel<64><<<blocks, threads>>>(
            query.data_ptr<float>(), source.data_ptr<float>(),
            ptr_q.data_ptr<int64_t>(), ptr_s.data_ptr<int64_t>(),
            batch_map_q.data_ptr<int>(),
            out_indices.data_ptr<int64_t>(), out_dists.data_ptr<float>(),
            N_q, K, B);
    }

    return std::make_tuple(out_indices, out_dists);
}


// ═══════════════════════════════════════════════════════════════════
// Self-KNN convenience: query == source, ptr_q == ptr_s
// ═══════════════════════════════════════════════════════════════════

std::tuple<at::Tensor, at::Tensor> knn_self_cuda(
    at::Tensor points,     // [N, 3] float32, CUDA
    at::Tensor ptr,        // [B+1] int64, CUDA
    int64_t k
) {
    // k+1 because we skip self in post-processing
    // But for simplicity, we just call knn with query=source
    // and the result will include self (distance=0) at index 0.
    // The caller can slice [:, 1:] if self-loops are unwanted.
    return knn_cuda(points, points, ptr, ptr, k);
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11 module
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_cuda", &knn_cuda,
          "K-Nearest Neighbor search with ptr-based batch segmentation (CUDA)",
          py::arg("query"),
          py::arg("source"),
          py::arg("ptr_q"),
          py::arg("ptr_s"),
          py::arg("k"));
    m.def("knn_self_cuda", &knn_self_cuda,
          "Self-KNN (query=source) with ptr-based batch segmentation (CUDA)",
          py::arg("points"),
          py::arg("ptr"),
          py::arg("k"));
}
