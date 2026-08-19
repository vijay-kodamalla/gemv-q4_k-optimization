# Findings

Two things surfaced while verifying correctness on real hardware, separate
from the original performance study.

---

## 1. Generation divergence across prompts

`test-backend-ops` checks `MUL_MAT` output against a CPU reference using
numeric tolerance, and passes 985/985 for five of the six kernels
(`double_buff` excepted — see below). A stricter check was also run:
fixed-seed, greedy (`--temp 0`) generation through the full model, text
diffed exactly against the same run on stock upstream. This catches a
difference of a single floating-point ULP if it happens to flip which token
wins an argmax.

Across 8 varied prompts, on an RTX PRO 6000 Blackwell:

| Variant | Diverged | Rate |
|---|---|---|
| `2rowCTA` | 3/8 | 38% |
| `2pack` | 2/8 | 25% |
| `tile_multiwarp` | 2/8 | 25% |
| `w_tile` | 2/8 | 25% |
| `y_tile` | 2/8 | 25% |

Every divergence was small — one or two words differing partway through a
64-token generation, with everything before that point byte-identical. This
is consistent with floating-point non-associativity: each kernel sums the
same ~4096-element dot product in a different order than upstream (different
tiling, warp partitioning, reduction tree), and the final sum can differ by a
few ULP. At most tokens this has no effect. Occasionally, at a token where
the top two logits are close, it flips the argmax, and greedy decoding
follows a different path from there since each token depends on the ones
before it.

### A hypothesis that didn't hold up

The first four prompts fit a clean pattern: kernels that split reduction
work across multiple warps (`2pack`, `2rowCTA`, `tile_multiwarp`) diverged;
kernels that keep everything in a single warp (`w_tile`, `y_tile`) — closer
to upstream's own accumulation order — did not.

The full 8-prompt set didn't support it. On prompt 6 ("In a conversation
between two friends, one says"), `w_tile` and `y_tile` both diverged while
the three multi-warp kernels didn't — the reverse of what the pattern
predicted. Across all 8 prompts, all five kernels land in a narrow 25-38%
band with no separation by architecture.

Divergence rate doesn't track kernel architecture at this sample size. It's
more likely tied to the prompt — how many tokens in a given generation land
near a genuine tie in the logits — than to which kernel is doing the
summing. That wasn't tested further and remains open.

### Practical implications

- Numeric closeness (`test-backend-ops` passing) does not imply exact output
  match under greedy decoding. This is expected for any kernel with a
  different reduction order, not specific to these kernels or to Q4_K.
- None of the five passing kernels should be assumed bit-exact with upstream.
  Verify directly against your own workload if that matters for your use
  case.
- 8 prompts is a small sample. A proper characterization would need more
  prompts and logit-level inspection, not just text diffing, to confirm the
  near-tied-logit explanation directly.

---

## 2. `double_buff`: correctness failure for batched decode

`double_buff` (`double_buff.cu`) is a 4-warp, split-K kernel using
`cp.async`-based software pipelining — ping-pong shared memory buffers,
`__pipeline_memcpy_async` / `__pipeline_commit` / `__pipeline_wait_prior`.

### The failure

`test-backend-ops -o MUL_MAT` fails for `ncols_dst = 2, 3, 4`:

```
MUL_MAT(type_a=q4_K, type_b=f32, m=16, n=2, k=256, ...): FAIL
MUL_MAT(type_a=q4_K, type_b=f32, m=16, n=3, k=256, ...): FAIL
MUL_MAT(type_a=q4_K, type_b=f32, m=16, n=4, k=256, ...): FAIL
```

`ncols_dst = 1` passes. `ncols_dst` is the number of output columns per
launch — the number of tokens processed per forward pass. `ncols_dst = 1` is
standard single-token autoregressive decode, the configuration the original
performance numbers were measured under. `ncols_dst > 1` is batched or
speculative decode, a configuration this kernel was not exercised against
during the original work.

### Likely cause

The kernel's final reduction stage:

```cpp
sums[j] += __shfl_xor_sync(0xffffffff, sums[j], 16);   // full mask, correct

const unsigned mask16 = 0x0000FFFFu;
sums[j] += __shfl_down_sync(mask16, sums[j], 8);        // mask excludes calling lane
sums[j] += __shfl_down_sync(mask16, sums[j], 4);
sums[j] += __shfl_down_sync(mask16, sums[j], 2);
sums[j] += __shfl_down_sync(mask16, sums[j], 1);
```

`mask16 = 0x0000FFFF` covers lanes 0-15. A calling lane in 16-31 is not in
its own mask — undefined behavior under the CUDA programming model,
independent of `NCOLS_DST`. The same pattern exists in `2pack.cu` without
producing a `test-backend-ops` failure there, plausibly because at
`ncols_dst = 1` the undefined values involved don't affect the result on this
hardware/compiler combination, while looping over multiple columns changes
the register and execution pattern enough to surface it. This is a plausible
explanation, not a confirmed one — confirming it would mean fixing the mask
and re-testing, which would change the kernel's behavior and break its link
to the original performance measurement.

### Scope

`double_buff` is correctness-verified for `ncols_dst = 1` — the configuration
the reported performance numbers were measured under — and fails for
`ncols_dst > 1`. It should not be used in a batched or speculative decode
configuration without further work. The kernel logic is unchanged from the
version that was originally profiled.

---

## 3. Using these kernels

None of the six kernels should be treated as drop-in replacements for
upstream in production without verifying against your own workload — see the
main write-up for why the project stopped at this stage. `double_buff`'s
`ncols_dst=1` scope in particular should be checked before reuse in any
batched setting.
