// Dispatch block for 2pack.cu
// Pairs with: kernels/2pack.cu
// Verified against the actual kernel file — NWARPS=4 confirmed by 2pack.cu's
// own explicit template instantiations. One row per CTA (kernel indexes
// dst with row = blockIdx.x). No dynamic shared memory — warp_partials is
// statically declared.

        case GGML_TYPE_Q4_K: {
            const int device = ggml_cuda_get_device();
            const int cc = ggml_cuda_info().devices[device].cc;

            const bool simple_ok =
                (ncols_dst <= 4) &&
                (ids == nullptr) &&
                (fusion.gate == nullptr && fusion.x_bias == nullptr) &&
                (nchannels_x == 1 && nchannels_y == 1 && nchannels_dst == 1) &&
                (nsamples_x == 1 && nsamples_dst == 1);

            if (simple_ok && cc >= 80) {
                const int num_blocks_total = stride_row_x;
                constexpr int NWARPS = 4;

                dim3 grid(nrows_x, 1, 1);
                dim3 block(32, NWARPS);

                const size_t smem_size = 0;

                switch (ncols_dst) {
                    case 1: custom_q4k_gemv<NWARPS, 1><<<grid, block, smem_size, stream>>>(
                                vx, vy, dst, num_blocks_total, stride_col_y, stride_col_dst); break;
                    case 2: custom_q4k_gemv<NWARPS, 2><<<grid, block, smem_size, stream>>>(
                                vx, vy, dst, num_blocks_total, stride_col_y, stride_col_dst); break;
                    case 3: custom_q4k_gemv<NWARPS, 3><<<grid, block, smem_size, stream>>>(
                                vx, vy, dst, num_blocks_total, stride_col_y, stride_col_dst); break;
                    case 4: custom_q4k_gemv<NWARPS, 4><<<grid, block, smem_size, stream>>>(
                                vx, vy, dst, num_blocks_total, stride_col_y, stride_col_dst); break;
                }
            } else {
                mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                    (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                     nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            }
            break;
        }
