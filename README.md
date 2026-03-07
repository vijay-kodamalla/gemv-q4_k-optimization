# I Built a Custom Q4_K GEMV Kernel for llama.cpp. Here is What I Learned!

Hey! Welcome to my repo. This is a deep dive into my journey of optimizing the memory-bound token generation phase (GEMV) for Large Language Models using `llama.cpp`. 

I started this project on an NVIDIA H200 to see how far I could push the HBM3e bandwidth, and ended up doing obsessive micro-benchmarking on an A100 to figure out why textbook optimizations sometimes just completely fail in the real world.

**📖 Full Technical Write-up:** [I Built a Quantized GEMV Kernel from Scratch](https://vijayprabhas9.github.io/gemv_optimization/)

## The Story: The "Bandwidth Paradox"

When I first spun up the H200, my Mistral 7B throughput was a dismal 34.7 tokens/second. After comparing it to an A100 (which was hitting 142.9 t/s), I realized the H200 wasn't hardware-limited, it was system-limited. The CPU process was running on a distant socket, creating massive NUMA interconnect latency. Fixing the process affinity made the throughput explode to ~240 t/s!

But then I looked at the actual kernels. Comparing the H100 to the H200, the Matrix-Vector times were almost identical (14.71 μs vs 14.51 μs). The H200 has 43% more memory bandwidth, but zero speedup! That is when it hit me: the current kernels are entirely **latency bound**, not bandwidth bound. 

## What I Tried (The Crazy Stuff)

I went to work building my own custom kernels to beat the upstream `llama.cpp` implementation. I threw everything at it:
* **Vectorized "Shovel" Loads:** I replaced the scalar 4-byte loads with vectorized 16-byte `int4` loads to pull in 512 bytes per cycle per warp, cutting instruction counts drastically.
* **Nuclear Debugging:** Early asynchronous pipeline tests failed silently, so I had to inject synchronous error traps directly into the production hot path just to catch the GPU driver dropping my launch commands.
* **Dynamic Batching:** I used a 2D Grid Launch (`blockIdx.y`) so the exact same kernel binary could handle single-token decoding or small batch prefill (Batch 1, 2, 3, 4) without falling back to a slower path. 

## The 5 Kernel Variants Explored

I wrote 5 distinct kernel architectures to try and break the memory wall. You can find all the raw CUDA files in the `kernels/` directory.

1. **`y_tile.cu` (Activation Tiling):** Staged the activation vector in shared memory. Catastrophic failure. Nsight Compute proved my L1 cache hit rate was already 82.5% on global loads. By forcing the data into shared memory, I bypassed the hardware cache and introduced 1.7-way bank conflicts across 66% of my requests, tanking performance for zero benefit.
2. **`w_tile.cu` (Weight Tiling):** Staged weights in shared memory using coalesced 128-bit loads. Fails because GEMV has zero data reuse (you touch a weight once and never again). Shared memory overhead was a pure penalty.
3. **`2pack.cu` (100% Warp Utilization):** Q4_K math only needs 16 threads. To stop half the warp from idling, I forced the upper 16 threads to process a *second* block simultaneously and merged them with warp shuffles. Fastest custom variant, but indexing overhead was too high.
4. **`tile_multiwarp.cu` & `2rowCTA.cu`:** Combined cooperative warp loading with shared memory tiles, testing both single-row and multi-row CTA geometry to maximize intra-CTA data reuse.
5. **`double_buff.cu` (`cp.async` Pipeline):** Used hardware double-buffering to hide latency by fetching the next tile from global memory while computing the current tile.

## The Hard Truth: Why Upstream Wins

I locked down a checkpoint on an A100 to compare my best custom variant (`2pack.cu`) against the naive upstream `llama.cpp` kernel. 

My custom kernel was 11% slower (22.27 μs vs 20.03 μs). Why? I fired up Nsight Compute to find out:

1. **Instruction Overhead:** My kernel executed 7.33M instructions compared to upstream's 4.78M. In a memory-starved regime, that 53% instruction bloat is an absolute killer. 
2. **Scheduler Health:** Upstream is just better at hiding latency. It maintains more eligible warps per scheduler, preventing memory scoreboard stalls from propagating into idle execution cycles. 

**The ultimate lesson:** Decode-path GEMV performance is governed by scheduler health, not raw bandwidth or micro-tiling. Upstream wins because it keeps its instruction footprint incredibly small, tolerating uncoalesced accesses better than my complex indexing ever could.

## How to Run It

Because I built these kernels as isolated, hardcore experiments across different clusters, I am providing them as standalone drop-in files instead of a massive fork. 

If you want to compile and test a variant:
1. Clone the upstream `llama.cpp` repository.
2. Navigate to the CUDA backend directory (e.g., `ggml/src/ggml-cuda/`).
3. Replace the upstream GEMV implementation with `kernels/mmvq.cu`.
4. Drop your chosen kernel variant (e.g., `kernels/2pack.cu`) into the same directory and ensure it is included/called by the wrapper.
5. Recompile the project using standard CMake instructions.

## Let's Connect!
**Vijay Prabhas Kodamalla** Graduate Research Assistant, GPU Performance Engineering | M.S. Computational Science & Engineering @ Georgia Tech  
[LinkedIn](https://linkedin.com/in/vijaykodamalla)
