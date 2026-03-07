#include "common.cuh"
#include "ggml-quants.h"
#include "vecdotq.cuh"
#include <cuda_fp16.h>

static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");
static_assert(sizeof(block_q8_1) == 36,  "block_q8_1 size mismatch");

// SUGGESTED KERNEL: "Pack-2" Strategy
// - 100% Warp Utilization (All 32 threads active)
// - Processes 2 blocks per loop iteration
// - Stride = 2

template <int NWARPS, int NCOLS_DST>
__global__ void custom_q4k_gemv(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,        // quant_blocks per row
    const int stride_col_y_blocks,
    const int stride_col_dst
) {
    // Thread indexing
    const int lane_id = threadIdx.x;   // 0..31
    const int warp_id = threadIdx.y;   // 0..NWARPS-1

    // Output row this CTA is responsible for
    const int row = blockIdx.x;

    // Cast pointers
    const block_q4_K * W = reinterpret_cast<const block_q4_K *>(vx);
    const block_q8_1 * Y = reinterpret_cast<const block_q8_1 *>(vy);

    // Constants for Q4_K
    constexpr int qi  = QI4_K;                      // 32
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;         // 2

    // K-dimension partitioning across warps
    // We want to align to 2-block boundaries for our Pack-2 logic
    int blocks_per_warp = (num_blocks_total + NWARPS - 1) / NWARPS;
    
    // Align blocks_per_warp to multiple of 2 to ensure clean striding
    if (blocks_per_warp % 2 != 0) blocks_per_warp++;

    const int warp_k_start = warp_id * blocks_per_warp;
    const int warp_k_end   = min(warp_k_start + blocks_per_warp, num_blocks_total);

    // Each thread accumulates partial sums for each output column
    float sums[NCOLS_DST];
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) sums[j] = 0.0f;

    // Base pointer for this row in W
    const int row_base = row * num_blocks_total;

    // -----------------------------------------------------------------------
    // PACK-2 LOGIC
    // -----------------------------------------------------------------------
    // Lane 0-15:  Logical Lane 0-15, Offset 0 (Process kbx)
    // Lane 16-31: Logical Lane 0-15, Offset 1 (Process kbx+1)
    
    const int logical_lane = lane_id % 16;  // 0..15 for everyone
    const int block_offset = lane_id / 16;  // 0 for lower half, 1 for upper half
    
    // Correct iqs for the vec_dot function (0, 2, ..., 30)
    const int iqs = vdr * logical_lane; 

    // Loop Stride is 2 because the warp consumes 2 blocks at once
    for (int kbx = warp_k_start; kbx < warp_k_end; kbx += 2) {
        
        // Calculate the specific block this specific lane works on
        const int my_kbx = kbx + block_offset;

        // Boundary check: ensure we don't read past the end (if N is odd)
        if (my_kbx < warp_k_end) {
            const int kby = my_kbx * (QK_K / QK8_1);   // Y block index

            #pragma unroll
            for (int j = 0; j < NCOLS_DST; ++j) {
                const block_q8_1 * y_col  = Y + j * stride_col_y_blocks;
                const block_q8_1 * bq8_ptr = y_col + kby;

                // Call the dot product
                // No "if (active)" check needed! All 32 threads contribute.
                sums[j] += vec_dot_q4_K_q8_1(
                    (const void *)(W + row_base),
                    bq8_ptr,
                    my_kbx,
                    iqs
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // REDUCTION
    // -----------------------------------------------------------------------
    // Step 1: Fold the "Upper Half" (Block K+1 results) into "Lower Half" (Block K)
    // After this, Lanes 0-15 hold the sum of BOTH blocks.
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        sums[j] += __shfl_xor_sync(0xffffffff, sums[j], 16);
    }

    // Step 2: Standard reduction within the lower 16 lanes
    // We only need to reduce 16 -> 1 now.
    const unsigned mask16 = 0x0000FFFFu; // Mask for lower 16 threads
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        // 16 -> 8
        sums[j] += __shfl_down_sync(mask16, sums[j], 8);
        // 8 -> 4
        sums[j] += __shfl_down_sync(mask16, sums[j], 4);
        // 4 -> 2
        sums[j] += __shfl_down_sync(mask16, sums[j], 2);
        // 2 -> 1
        sums[j] += __shfl_down_sync(mask16, sums[j], 1);
    }

    // Shared storage for warp partials (only lane 0 of each warp writes)
    __shared__ float warp_partials[NWARPS][NCOLS_DST];

    if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            warp_partials[warp_id][j] = sums[j];
        }
    }

    __syncthreads();

    // Final reduction across warps by warp 0 lane 0
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                total += warp_partials[w][j];
            }
            dst[j * stride_col_dst + row] = total;
        }
    }
}

// Explicit instantiations for NWARPS=4
template __global__ void custom_q4k_gemv<4, 1>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);

template __global__ void custom_q4k_gemv<4, 2>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);

template __global__ void custom_q4k_gemv<4, 3>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);

template __global__ void custom_q4k_gemv<4, 4>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);
