#include "common.cuh"
#include "ggml-quants.h"
#include "vecdotq.cuh"
#include <cuda_fp16.h>

// -----------------------------------------------------------------------------
// Loader: Activations (Q8_1) Only
// -----------------------------------------------------------------------------
__device__ __forceinline__ void load_tile_q8_1_coalesced(
    block_q8_1 * smem_dst,
    const block_q8_1 * global_src,
    int q8_blocks_to_load, 
    int tid
) {
    const int4* src_vec = reinterpret_cast<const int4*>(global_src);
    int4* dst_vec       = reinterpret_cast<int4*>(smem_dst);
    
    // 36 bytes per block. 
    // Total size (blocks * 36) is divisible by 16 if blocks is a multiple of 8 (which it is).
    int total_int4s = (q8_blocks_to_load * 36) / 16; 

    for (int i = tid; i < total_int4s; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
}

// -----------------------------------------------------------------------------
// Main Kernel: Activations Tiled, Weights Direct
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE, int TILE_SIZE, int NCOLS_DST>
__global__ void custom_q4k_gemv(
    const void * __restrict__ vx, // Weights (Global)
    const void * __restrict__ vy, // Activations (Global source)
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
) {
    // --- 1. Shared Memory Setup (Activations Only) ---
    extern __shared__ char smem_raw[];
    
    // Align Y pointer (Starts at beginning of SMEM now)
    uintptr_t p_y = reinterpret_cast<uintptr_t>(smem_raw);
    p_y = (p_y + 15) & ~uintptr_t(15);
    block_q8_1 * smem_Y_base = reinterpret_cast<block_q8_1 *>(p_y);

    const block_q4_K * W = reinterpret_cast<const block_q4_K *>(vx);
    const block_q8_1 * Y = reinterpret_cast<const block_q8_1 *>(vy);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int row_base = row * num_blocks_total;

    constexpr int qi  = QI4_K; 
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int q8_per_q4 = QK_K / QK8_1; // 8

    constexpr int blocks_per_iter = (vdr * 32) / qi;
    const int kqs = vdr * (tid % (qi / vdr));
    const int kbx_lane = tid / (qi / vdr);

    float sums[NCOLS_DST];
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        sums[j] = 0.0f;
    }

    // --- 2. Main Loop ---
    for (int tile_start = 0; tile_start < num_blocks_total; tile_start += TILE_SIZE) {
        
        const int remaining = num_blocks_total - tile_start;
        const int blocks_to_load = remaining < TILE_SIZE ? remaining : TILE_SIZE;
        const int q8_blocks_to_load = blocks_to_load * q8_per_q4;

        // A. Load Activations (Loop over Columns)
        #pragma unroll
        for(int j = 0; j < NCOLS_DST; ++j) {
            // SMEM Offset: stack tiles sequentially
            block_q8_1* smem_Y_col = smem_Y_base + (j * TILE_SIZE * q8_per_q4);
            
            // Global Offset: Jump by stride_col_y_blocks
            const block_q8_1* global_Y_col = Y + (j * stride_col_y_blocks) + (tile_start * q8_per_q4);

            load_tile_q8_1_coalesced(smem_Y_col, global_Y_col, q8_blocks_to_load, tid);
        }

        // Note: No weight loading here.

        __syncthreads();

        // B. Compute (Loop over Columns)
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            // Point to the correct SMEM tile for this column
            const block_q8_1 * y_col_smem = smem_Y_base + (j * TILE_SIZE * q8_per_q4);

            // Point to Global Memory for weights (current tile window)
            // Note: vec_dot expects the base pointer, it adds kbx internally. 
            // So we pass the pointer to the START of the current tile in Global Memory.
            const void* w_global_tile = (const void*)&W[row_base + tile_start];

            for (int k = 0; k < blocks_to_load; k += blocks_per_iter) {
                const int local_kbx = k + kbx_lane;

                if (local_kbx < blocks_to_load) {
                    const int kby = local_kbx * q8_per_q4;
                    const block_q8_1 * bq8_ptr = y_col_smem + kby;

                    sums[j] += vec_dot_q4_K_q8_1(
                        w_global_tile, // Weights from Global
                        bq8_ptr,       // Activations from Shared
                        local_kbx,
                        kqs
                    );
                }
            }
        }
        __syncthreads();
    }

    // --- 3. Reduction & Write ---
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sums[j] += __shfl_xor_sync(0xffffffff, sums[j], mask);
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            dst[j * stride_col_dst + row] = sums[j];
        }
    }
}

// Instantiations
template __global__ void custom_q4k_gemv<32, 32, 1>(const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 32, 2>(const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 32, 3>(const void *, const void *, float *, int, int, int);
template __global__ void custom_q4k_gemv<32, 32, 4>(const void *, const void *, float *, int, int, int);
