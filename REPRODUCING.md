# Reproducing these results

Performance was measured on an NVIDIA A100 (80GB PCIe) in early 2026.
Correctness and build reproducibility were verified separately on an NVIDIA
RTX PRO 6000 Blackwell in August 2026. This document covers both, and keeps
them clearly separate — the two sections report different things on
different hardware.

---

## 1. Pinned upstream revision

```
repo:   https://github.com/ggml-org/llama.cpp
commit: c00ff929dcfd150234e62f30e863bca4f1337aee
build:  b7389
```

All six kernel variants are patched against this commit.

---

## 2. What gets modified

Each variant touches two files inside a stock `llama.cpp` checkout:

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/<kernel>.cu` | the kernel itself |
| `ggml/src/ggml-cuda/mmvq.cu` | `#include "<kernel>.cu"`, a forward declaration matching that kernel's template signature, and a `GGML_TYPE_Q4_K` dispatch branch with a fallback to stock upstream |

Each variant ships as a single patch containing both files. The kernels have
different template signatures, grid geometry, and shared-memory requirements,
so a kernel file from one variant will not compile against another variant's
dispatch.

The upstream fallback path (`mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>`)
is preserved unmodified in every patch. A patched build handles every
workload correctly, not just the benchmarked one — any tensor shape or
configuration outside the fast path returns to stock behavior automatically.

---

## 3. Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout c00ff929dcfd150234e62f30e863bca4f1337aee

export CUDA_ARCH=80          # A100. Use 120 for RTX PRO 6000 Blackwell
export NVCC=/usr/local/cuda/bin/nvcc   # adjust to your CUDA install

bash repro-scripts/build.sh upstream    # baseline, no patch
bash repro-scripts/build.sh 2pack       # or: 2rowCTA, tile_multiwarp, w_tile, y_tile, double_buff
```

Each variant builds into its own `build_<variant>/` directory, so all seven
(six variants plus upstream) can coexist. `repro-scripts/build.sh` applies
`patches/<variant>.patch` to a clean checkout, configures CMake, and builds.

**Build notes:**

- Use `-lineinfo` (single dash), not `--lineinfo`. The double-dash form is
  forwarded past `nvcc` to the host compiler and rejected.
- `CMAKE_CUDA_ARCHITECTURES` and `CMAKE_CUDA_COMPILER` must match your
  hardware and toolchain. Check with
  `nvidia-smi --query-gpu=name,compute_cap --format=csv` and `which nvcc`.
- A stale `build_<variant>/` directory configured for a different kernel or
  architecture fails with an `undefined reference` linker error rather than a
  clear message. If a build behaves unexpectedly, delete the build directory
  and reconfigure rather than debug in place.

---

## 4. Correctness

```bash
./build_<variant>/bin/test-backend-ops -o MUL_MAT
```

Validates the CUDA `MUL_MAT` path, including the custom Q4_K dispatch,
against a CPU reference across many shapes, using numeric tolerance.

```bash
./build_<variant>/bin/llama-completion -m <model.gguf> -p "<prompt>" \
    -n 64 -ngl 99 -t 1 --seed 1234 --temp 0 --no-display-prompt \
    < /dev/null
```

Fixed-seed greedy generation, diffed against the same command run on
`build_upstream`. Use `llama-completion`, not `llama-cli` — `llama-cli` in
this build defaults to an interactive REPL and does not write output in a way
that's reliably capturable via stdout/stderr redirection. Close stdin
(`< /dev/null`) as well, or `llama-completion` waits for a second prompt after
finishing generation.

### Results — RTX PRO 6000 Blackwell, single prompt

| Variant | `test-backend-ops` | Generation vs upstream |
|---|---|---|
| `2rowCTA` | PASS (985/985) | Differs |
| `2pack` | PASS (985/985) | Differs |
| `tile_multiwarp` | PASS (985/985) | Differs |
| `w_tile` | PASS (985/985) | Identical |
| `y_tile` | PASS (985/985) | Identical |
| `double_buff` | FAIL for `ncols_dst > 1` | n/a |

`test-backend-ops` cases with `type_b=f16` reporting "not supported" reflect
a pre-existing upstream limitation (Q4_K × f16 is not implemented on CUDA)
and are unrelated to these kernels.

See [FINDINGS.md](FINDINGS.md) for the generation-diff divergence across
multiple prompts and the `double_buff` limitation in detail.

---

## 5. Performance

The performance table in the write-up — instruction counts, cycles,
wall-clock time, the 11% gap — is from the original A100 measurement
campaign, early 2026. RTX PRO 6000 Blackwell is a different GPU generation,
memory subsystem, and SM count, so its numbers would not be a meaningful
comparison against the A100 table and are not included here.

Original A100 build environment:

| | |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| Compute capability | 8.0 |
| CUDA toolkit | 13.0.88 |
| Host compiler | GCC 12.3.0 |
| Build | Release, `-O3 -DNDEBUG` |

Raw `.ncu-rep` artifacts backing the original table are in `profiling/`.

---

## 6. Verification environment

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| CUDA toolkit | 12.9 |
| Model | Mistral 7B v0.3, Q4_K, `mistral-7b-v0.3-q4.gguf` |

The dispatch condition is `cc >= 80`, so the correctness gate should run on
any `sm_80`+ GPU.

---

## 7. Kernel status

| Kernel | Published row | Status |
|---|---|---|
| `2rowCTA` | 2-Row/CTA | Correctness-verified |
| `2pack` | 2-Pack | Correctness-verified |
| `tile_multiwarp` | W-Tile + 4W/2CTA | Correctness-verified |
| `w_tile` | W-Tiling | Correctness-verified |
| `y_tile` | Y-Tiling | Correctness-verified |
| `double_buff` | Double-Buffered | Verified for `ncols_dst=1` only — see FINDINGS.md |
| `both_tile` | *(not published)* | Exploratory; tiles both operands; not independently profiled |

---

## 8. Model

```
model:   Mistral 7B v0.3
quant:   Q4_K (M/S — confirm against your actual file's tensor naming)
file:    mistral-7b-v0.3-q4.gguf
```

---

## 9. License

Kernels and the modified `mmvq.cu` derive from and link against `llama.cpp`
(MIT licensed). See `LICENSE` and `NOTICE`.
