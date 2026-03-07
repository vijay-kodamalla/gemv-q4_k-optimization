#include "common.cuh"
#include "ggml-quants.h"
#include "vecdotq.cuh"
#include <cuda_fp16.h>

static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");
static_assert(sizeof(block_q8_1) == 36,  "block_q8_1 size mismatch");

// Coalesced 128-bit loader (your optimized version)
__device__ __forceinline__ void load_tile_q4k_coalesced(
    block_q4_K * smem_dst,
    const block_q4_K * global_src,
    int blocks_to_load,
    int tid,
    int total_threads  // ← Now uses all 128 threads (4 warps)
) {
    const int4* src_vec = reinterpret_cast<const int4*>(global_src);
    int4* dst_vec       = reinterpret_cast<int4*>(smem_dst);
    int total_int4s = blocks_to_load * 9;

    // All threads cooperatively load
    for (int i = tid; i < total_int4s; i += total_threads) {
        dst_vec[i] = src_vec[i];
    }
}

// MULTI-WARP W-TILING: Strategy 1 (4 Warps on 1 Row)
template <int BLOCK_SIZE, int NWARPS, int TILE_SIZE, int NCOLS_DST>
__global__ void custom_q4k_gemv(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
) {
    // Shared memory: ONE tile buffer shared by all warps
    extern __shared__ char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15) & ~uintptr_t(15);
    block_q4_K * smem_tile = reinterpret_cast<block_q4_K *>(p);

    const block_q4_K * W = reinterpret_cast<const block_q4_K *>(vx);
    const block_q8_1 * Y = reinterpret_cast<const block_q8_1 *>(vy);

    // Thread indexing
    const int lane_id = threadIdx.x;   // 0-31
    const int warp_id = threadIdx.y;   // 0-(NWARPS-1)
    const int tid = warp_id * BLOCK_SIZE + lane_id;  // 0-127 for NWARPS=4
    const int total_threads = NWARPS * BLOCK_SIZE;

    const int row = blockIdx.x;
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

    // K-dimension partitioning across warps
    const int blocks_per_warp = (num_blocks_total + NWARPS - 1) / NWARPS;
    const int warp_k_start = warp_id * blocks_per_warp;
    const int warp_k_end = min(warp_k_start + blocks_per_warp, num_blocks_total);

    // Tiled loop: All warps cooperate on loading, then work on their K-slice
    for (int tile_start = 0; tile_start < num_blocks_total; tile_start += TILE_SIZE) {
        
        const int remaining = num_blocks_total - tile_start;
        const int blocks_to_load = remaining < TILE_SIZE ? remaining : TILE_SIZE;

        // COOPERATIVE LOAD: All 4 warps load together (128 threads!)
        load_tile_q4k_coalesced(
            smem_tile, 
            &W[row_base + tile_start], 
            blocks_to_load, 
            tid,
            total_threads  // ← 128 threads = 4× faster loading!
        );
        
        __syncthreads();  // BARRIER 1: Ensure load completes

        // COMPUTE: Each warp works on its K-slice within the tile
        // Only process if this tile overlaps with warp's K-range
        if (tile_start < warp_k_end && (tile_start + TILE_SIZE) > warp_k_start) {
            
            // Determine this warp's work range within the tile
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
                            (const void *)smem_tile,
                            bq8_ptr,
                            local_kbx,  // Index into tile (0 to TILE_SIZE)
                            kqs
                        );
                    }
                }
            }
        }

        __syncthreads();  // BARRIER 2: Before loading next tile
    }

    // Warp-level reduction
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sums[j] += __shfl_xor_sync(0xffffffff, sums[j], mask);
        }
    }

    // Cross-warp reduction
    __shared__ float warp_partials[NWARPS][NCOLS_DST];
    
    if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            warp_partials[warp_id][j] = sums[j];
        }
    }
    
    __syncthreads();  // BARRIER 3: Ensure all warps wrote partials

    // Final reduction
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

// Instantiations for NWARPS=4
template __global__ void custom_q4k_gemv<32, 4, 64, 1>(
    const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 4, 64, 2>(
    const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 4, 64, 3>(
    const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 4, 64, 4>(
    const void *, const void *, float *, int, int, int);
