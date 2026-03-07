#include "common.cuh"
#include "ggml-quants.h"
#include "vecdotq.cuh"
#include <cuda_fp16.h>

static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");
static_assert(sizeof(block_q8_1) == 36,  "block_q8_1 size mismatch");

   // OPTIMIZED LOADER: Coalesced 128-bit (16-byte) access
// Drastically reduces instruction count compared to uint32_t
__device__ __forceinline__ void load_tile_q4k_coalesced(
    block_q4_K * smem_dst,
    const block_q4_K * global_src,
    int blocks_to_load,
    int tid
) {
    // 1. Reinterpret pointers as int4 (16 bytes wide)
    const int4* src_vec = reinterpret_cast<const int4*>(global_src);
    int4* dst_vec       = reinterpret_cast<int4*>(smem_dst);

    // 2. Calculate Total Chunks
    // 1 block_q4_K = 144 bytes.
    // 1 int4       = 16 bytes.
    // Total int4s per block = 144 / 16 = 9.
    int total_int4s = blocks_to_load * 9;

    // 3. Grid-Stride Loop (Vectorized)
    // All 32 threads load 128-bit chunks in parallel.
    for (int i = tid; i < total_int4s; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
 
}

template <int BLOCK_SIZE, int TILE_SIZE, int NCOLS_DST>
__global__ void custom_q4k_gemv(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
) {
    clock_t start_load, end_load, start_compute, end_compute;	
    extern __shared__ char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15) & ~uintptr_t(15);
    block_q4_K * smem_tile = reinterpret_cast<block_q4_K *>(p);

    const block_q4_K * W = reinterpret_cast<const block_q4_K *>(vx);
    const block_q8_1 * Y = reinterpret_cast<const block_q8_1 *>(vy);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int row_base = row * num_blocks_total;

    constexpr int qi  = QI4_K;
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int blocks_per_iter = (vdr * 32) / qi;

    const int kqs = vdr * (tid % (qi / vdr));
    const int kbx_lane = tid / (qi / vdr);

    // Multiple columns - separate sum for each
    float sums[NCOLS_DST];
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        sums[j] = 0.0f;
    }

    

    // Tile loop
    for (int tile_start = 0; tile_start < num_blocks_total; tile_start += TILE_SIZE) {

	//if (tid==0 && row==0) start_load = clock64();    
        const int remaining = num_blocks_total - tile_start;
        const int blocks_to_load = remaining < TILE_SIZE ? remaining : TILE_SIZE;

        // Zero-init shared memory
       // for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
       //     memset(&smem_tile[i], 0, sizeof(block_q4_K));
       // }
       // __syncthreads();

        // Load tile
        load_tile_q4k_coalesced(smem_tile, &W[row_base + tile_start], blocks_to_load, tid);
        __syncthreads();

	//if (tid == 0 && row == 0) {
//		end_load = clock64();
//		start_compute = clock64();
//	}

        // Compute for all columns
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            const block_q8_1 * y_col = Y + j * stride_col_y_blocks;

            for (int k = 0; k < blocks_to_load; k += blocks_per_iter) {
                const int local_kbx  = k + kbx_lane;
                const int global_kbx = tile_start + local_kbx;

                if (local_kbx < blocks_to_load) {
                    const int kby = global_kbx * (QK_K / QK8_1);
                    const block_q8_1 * bq8_ptr = y_col + kby;

                    sums[j] += vec_dot_q4_K_q8_1(
                        (const void *)smem_tile,
                        bq8_ptr,
                        local_kbx,
                        kqs
                    );
                }
            }
        }

//	if (tid == 0 && row == 0) {
 //           end_compute = clock64();
  //          printf("Tile %d: load=%lld cycles, compute=%lld cycles\n",
   //                tile_start, end_load - start_load, end_compute - start_compute);
    //    }

        __syncthreads();
    }

    // Warp reduce for each column
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sums[j] += __shfl_xor_sync(0xffffffff, sums[j], mask);
        }
    }

    // Write results
    if (tid == 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            dst[j * stride_col_dst + row] = sums[j];
        }
    }
}

// Explicit instantiations for different ncols_dst
template __global__ void custom_q4k_gemv<32, 32, 1>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);

template __global__ void custom_q4k_gemv<32, 32, 2>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);

template __global__ void custom_q4k_gemv<32, 32, 3>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);

template __global__ void custom_q4k_gemv<32, 32, 4>(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int num_blocks_total,
    const int stride_col_y_blocks,
    const int stride_col_dst
);
