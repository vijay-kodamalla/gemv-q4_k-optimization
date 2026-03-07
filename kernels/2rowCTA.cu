#include "common.cuh"
#include "ggml-quants.h"
#include "vecdotq.cuh"
#include <cuda_fp16.h>

static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");
static_assert(sizeof(block_q8_1) == 36,  "block_q8_1 size mismatch");

__device__ __forceinline__ void load_tile_q4k_coalesced(
    block_q4_K * smem_dst,
    const block_q4_K * global_src,
    int blocks_to_load,
    int tid,
    int total_threads
) {
    const int4* src_vec = reinterpret_cast<const int4*>(global_src);
    int4* dst_vec       = reinterpret_cast<int4*>(smem_dst);
    int total_int4s = blocks_to_load * 9;

    for (int i = tid; i < total_int4s; i += total_threads) {
        dst_vec[i] = src_vec[i];
    }
}

// 4 WARPS, 2 ROWS PER CTA
// Warp 0-1: Process Row 0
// Warp 2-3: Process Row 1
template <int BLOCK_SIZE, int NWARPS, int ROWS_PER_CTA, int TILE_SIZE, int NCOLS_DST>
__global__ void custom_q4k_gemv(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
) {
    // Shared memory: 2 tile buffers (one per row)
    extern __shared__ char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15) & ~uintptr_t(15);
    
    // Each row gets its own tile buffer
    block_q4_K * smem_tiles = reinterpret_cast<block_q4_K *>(p);

    const block_q4_K * W = reinterpret_cast<const block_q4_K *>(vx);
    const block_q8_1 * Y = reinterpret_cast<const block_q8_1 *>(vy);

    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * BLOCK_SIZE + lane_id;
    const int total_threads = NWARPS * BLOCK_SIZE;

    // Which row does this warp work on?
    const int warps_per_row = NWARPS / ROWS_PER_CTA;  // 4/2 = 2 warps per row
    const int local_row = warp_id / warps_per_row;    // 0 or 1
    const int warp_in_row = warp_id % warps_per_row;  // 0 or 1

    const int row0 = ROWS_PER_CTA * blockIdx.x;
    const int row = row0 + local_row;

    // Bounds check
    if (row >= stride_col_dst) return;  // Grid might be oversized

    const int row_base = row * num_blocks_total;

    constexpr int qi  = QI4_K;
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int blocks_per_iter = (vdr * BLOCK_SIZE) / qi;

    const int kqs = vdr * (lane_id % (qi / vdr));
    const int kbx_lane = lane_id / (qi / vdr);

    float sums[NCOLS_DST];
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        sums[j] = 0.0f;
    }

    // K-dimension partitioning across warps WITHIN each row
    const int blocks_per_warp = (num_blocks_total + warps_per_row - 1) / warps_per_row;
    const int warp_k_start = warp_in_row * blocks_per_warp;
    const int warp_k_end = min(warp_k_start + blocks_per_warp, num_blocks_total);

    // Pointer to this row's tile buffer
    block_q4_K * my_smem_tile = smem_tiles + (local_row * TILE_SIZE);

    // Tiled loop
    for (int tile_start = 0; tile_start < num_blocks_total; tile_start += TILE_SIZE) {
        
        const int remaining = num_blocks_total - tile_start;
        const int blocks_to_load = remaining < TILE_SIZE ? remaining : TILE_SIZE;

        // COOPERATIVE LOAD: Each row's warps load their own tile
        // Warps 0-1 load for row 0, Warps 2-3 load for row 1
        const int threads_per_row = warps_per_row * BLOCK_SIZE;  // 64 threads
        const int tid_in_row = warp_in_row * BLOCK_SIZE + lane_id;

        load_tile_q4k_coalesced(
            my_smem_tile,
            &W[row_base + tile_start],
            blocks_to_load,
            tid_in_row,
            threads_per_row
        );
        
        __syncthreads();  // BARRIER: Ensure both rows loaded

        // COMPUTE: Each warp works on its K-slice
        if (tile_start < warp_k_end && (tile_start + TILE_SIZE) > warp_k_start) {
            
            const int warp_tile_start = max(0, warp_k_start - tile_start);
            const int warp_tile_end = min(blocks_to_load, warp_k_end - tile_start);

            #pragma unroll
            for (int j = 0; j < NCOLS_DST; ++j) {
                const block_q8_1 * y_col = Y + j * stride_col_y_blocks;

                for (int k = warp_tile_start; k < warp_tile_end; k += blocks_per_iter) {
                    const int local_kbx = k + kbx_lane;
                    const int global_kbx = tile_start + local_kbx;

                    if (local_kbx < warp_tile_end) {
                        const int kby = global_kbx * (QK_K / QK8_1);
                        const block_q8_1 * bq8_ptr = y_col + kby;

                        sums[j] += vec_dot_q4_K_q8_1(
                            (const void *)my_smem_tile,
                            bq8_ptr,
                            local_kbx,
                            kqs
                        );
                    }
                }
            }
        }

        __syncthreads();  // BARRIER: Before next tile
    }

    // Warp-level reduction
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sums[j] += __shfl_xor_sync(0xffffffff, sums[j], mask);
        }
    }

    // Cross-warp reduction (per row)
    __shared__ float warp_partials[NWARPS][NCOLS_DST];
    
    if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            warp_partials[warp_id][j] = sums[j];
        }
    }
    
    __syncthreads();

    // Final reduction: First warp of each row writes result
    if (warp_in_row == 0 && lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            float total = 0.0f;
            // Sum across warps for this row
            for (int w = 0; w < warps_per_row; ++w) {
                total += warp_partials[local_row * warps_per_row + w][j];
            }
            dst[j * stride_col_dst + row] = total;
        }
    }
}

// Instantiations: 4 warps, 2 rows, TILE_SIZE=64
template __global__ void custom_q4k_gemv<32, 4, 2, 64, 1>(
    const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 4, 2, 64, 2>(
    const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 4, 2, 64, 3>(
    const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 4, 2, 64, 4>(
    const void *, const void *, float *, int, int, int);
