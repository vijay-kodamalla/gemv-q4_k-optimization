#include "common.cuh"
#include "ggml-quants.h"
#include "vecdotq.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

// -----------------------------------------------------------------------------
// Async Loader
// -----------------------------------------------------------------------------
__device__ __forceinline__ void load_chunk_async(
    block_q4_K * smem_dst,
    const block_q4_K * global_src,
    int blocks_to_load,
    int tid
) {
    const int4* src_vec = reinterpret_cast<const int4*>(global_src);
    int4* dst_vec       = reinterpret_cast<int4*>(smem_dst);
    int total_int4s = blocks_to_load * 9; 
    int lane = tid % 32;

    for (int i = lane; i < total_int4s; i += 32) {
        __pipeline_memcpy_async(&dst_vec[i], &src_vec[i], 16);
    }
}

// -----------------------------------------------------------------------------
// MAIN KERNEL: DOUBLE BUFFERED, 1 ROW/CTA
// -----------------------------------------------------------------------------
template <int NWARPS, int TILE_SIZE, int NCOLS_DST>
__global__ void custom_q4k_gemv(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int row = blockIdx.x;

    const block_q4_K * W = reinterpret_cast<const block_q4_K *>(vx);
    const block_q8_1 * Y = reinterpret_cast<const block_q8_1 *>(vy);
    const int row_base = row * num_blocks_total;

    // --- Partition K (Split-K) ---
    int blocks_per_warp = (num_blocks_total + NWARPS - 1) / NWARPS;
    if (blocks_per_warp % 2 != 0) blocks_per_warp++; 

    const int warp_k_start = warp_id * blocks_per_warp;
    int warp_k_end_actual = warp_k_start + blocks_per_warp;
    if (warp_k_end_actual > num_blocks_total) warp_k_end_actual = num_blocks_total;

    // --- Shared Memory (Double Buffered) ---
    // Total Size = NWARPS * (2 * TILE_SIZE)
    extern __shared__ char smem_raw[];
    block_q4_K* smem_base = reinterpret_cast<block_q4_K*>(smem_raw);
    
    // Each warp gets a PRIVATE 2-tile chunk
    block_q4_K* my_smem_base = smem_base + (warp_id * 2 * TILE_SIZE);
    
    // Pointers for "Ping-Pong"
    block_q4_K* buff0 = my_smem_base;
    block_q4_K* buff1 = my_smem_base + TILE_SIZE;

    // State Pointers
    block_q4_K* compute_buff = buff0; // We compute from here
    block_q4_K* load_buff    = buff1; // We load into here

    // --- Init Sums ---
    float sums[NCOLS_DST];
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) sums[j] = 0.0f;

    // Constants
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ; 
    const int iqs = vdr * (lane_id % 16);
    const int block_offset = lane_id / 16; 

    // =========================================================================
    // PIPELINE PROLOG (Load First Tile)
    // =========================================================================
    int tile_start = warp_k_start;
    int remaining = warp_k_end_actual - tile_start;
    
    if (remaining > 0) {
        int blocks = (remaining < TILE_SIZE) ? remaining : TILE_SIZE;
        const block_q4_K* src = W + row_base + tile_start;
        
        // Load into 'compute_buff' initially (it becomes valid after wait)
        load_chunk_async(compute_buff, src, blocks, tid);
        __pipeline_commit(); 
    }

    // Tracking vars for the "Current Computing Tile"
    int compute_blocks = (remaining < TILE_SIZE) ? remaining : TILE_SIZE;
    int compute_tile_idx = tile_start;

    // =========================================================================
    // PIPELINE MAIN LOOP
    // =========================================================================
    // Start loop from SECOND tile
    for (int next_tile_start = tile_start + TILE_SIZE; next_tile_start < warp_k_end_actual; next_tile_start += TILE_SIZE) {
        
        // 1. Issue Load for Tile N+1 (into 'load_buff')
        int next_remaining = warp_k_end_actual - next_tile_start;
        int next_blocks = (next_remaining < TILE_SIZE) ? next_remaining : TILE_SIZE;
        const block_q4_K* next_src = W + row_base + next_tile_start;

        load_chunk_async(load_buff, next_src, next_blocks, tid);
        __pipeline_commit(); 

        // 2. Wait for Tile N
        // "Wait until only 1 batch (the one we just started) is pending."
        // This guarantees the PREVIOUS batch (Tile N) is done.
        __pipeline_wait_prior(1);

        // 3. Compute Tile N (Read from 'compute_buff')
        for (int k = 0; k < compute_blocks; k += 2) {
            int my_local_k = k + block_offset; 
            if (my_local_k < compute_blocks) {
                int global_k_idx = compute_tile_idx + my_local_k;
                const int kby = global_k_idx * (QK_K / QK8_1);

                #pragma unroll
                for (int j = 0; j < NCOLS_DST; ++j) {
                    const block_q8_1 * bq8_ptr = Y + (j * stride_col_y_blocks) + kby;
                    sums[j] += vec_dot_q4_K_q8_1((const void *)compute_buff, bq8_ptr, my_local_k, iqs);
                }
            }
        }

        // 4. Swap Buffers
        block_q4_K* temp = compute_buff;
        compute_buff = load_buff;
        load_buff = temp;

        compute_blocks = next_blocks;
        compute_tile_idx = next_tile_start;
    }

    // =========================================================================
    // PIPELINE EPILOG (Drain Last Tile)
    // =========================================================================
    if (warp_k_end_actual > warp_k_start) {
        // Wait for the very last load to finish
        __pipeline_wait_prior(0);
        
        // Compute the final tile
        for (int k = 0; k < compute_blocks; k += 2) {
            int my_local_k = k + block_offset; 
            if (my_local_k < compute_blocks) {
                int global_k_idx = compute_tile_idx + my_local_k;
                const int kby = global_k_idx * (QK_K / QK8_1);

                #pragma unroll
                for (int j = 0; j < NCOLS_DST; ++j) {
                    const block_q8_1 * bq8_ptr = Y + (j * stride_col_y_blocks) + kby;
                    sums[j] += vec_dot_q4_K_q8_1((const void *)compute_buff, bq8_ptr, my_local_k, iqs);
                }
            }
        }
    }

    // =========================================================================
    // REDUCTION (Split-K Logic)
    // =========================================================================
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) sums[j] += __shfl_xor_sync(0xffffffff, sums[j], 16);
    
    const unsigned mask16 = 0x0000FFFFu;
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        sums[j] += __shfl_down_sync(mask16, sums[j], 8);
        sums[j] += __shfl_down_sync(mask16, sums[j], 4);
        sums[j] += __shfl_down_sync(mask16, sums[j], 2);
        sums[j] += __shfl_down_sync(mask16, sums[j], 1);
    }

    // Shared partials for 4 Warps
    __shared__ float warp_partials[NWARPS][NCOLS_DST];
    if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) warp_partials[warp_id][j] = sums[j];
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) total += warp_partials[w][j];
            dst[j * stride_col_dst + row] = total;
        }
    }
}

// Instantiations
template __global__ void custom_q4k_gemv<4, 32, 1>(const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<4, 32, 2>(const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<4, 32, 3>(const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<4, 32, 4>(const void *, const void *, float *, int, int, int);
