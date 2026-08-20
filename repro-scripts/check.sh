#!/usr/bin/env bash
# check.sh <variant> — correctness gate. Run this before any timing.
#
# Two checks:
#   1. test-backend-ops: validates the CUDA MUL_MAT path against a CPU reference
#      across many shapes and quant types. This is llama.cpp's own test, not
#      something hand-rolled, which is the point.
#   2. fixed-seed greedy generation, diffed against the upstream build. Coarse,
#      but it catches dispatch and reduction wiring errors that unit tests on
#      isolated shapes can miss.

set -euo pipefail

VARIANT="${1:?usage: check.sh <variant>}"
BUILD="build_${VARIANT}"
BASE="build_upstream"
GGUF="${GGUF:-../mistral-7b-v0.3-q4.gguf}"

[[ -x "${BUILD}/bin/test-backend-ops" ]] || {
    echo "error: ${BUILD}/bin/test-backend-ops not found — build with -DLLAMA_BUILD_TESTS=ON" >&2
    exit 1; }

echo "==> [1/2] backend op tests (MUL_MAT) — ${VARIANT}"
if "${BUILD}/bin/test-backend-ops" -o MUL_MAT; then
    echo "    PASS"
else
    echo "    FAIL — the kernel produces incorrect results. Stop here; do not benchmark." >&2
    exit 1
fi

echo
echo "==> [2/2] fixed-seed generation diff vs upstream"

if [[ ! -x "${BASE}/bin/llama-completion" ]]; then
    echo "    skipped: ${BASE}/ not built (run ./scripts/build.sh upstream first)"
    exit 0
fi
if [[ ! -f "${GGUF}" ]]; then
    echo "    skipped: model not found at ${GGUF}"
    exit 0
fi

PROMPT="The architecture of a GPU is"
run() {
    "$1/bin/llama-completion" -m "${GGUF}" -p "${PROMPT}" \
        -n 64 -ngl 99 -t 1 --seed 1234 --temp 0 --no-display-prompt < /dev/null 2>/dev/null
}

run "${BASE}"  > "/tmp/gen_upstream.txt"
run "${BUILD}" > "/tmp/gen_${VARIANT}.txt"

if diff -q "/tmp/gen_upstream.txt" "/tmp/gen_${VARIANT}.txt" >/dev/null; then
    echo "    PASS — greedy output identical to upstream"
else
    echo "    DIFFERS from upstream:" >&2
    diff "/tmp/gen_upstream.txt" "/tmp/gen_${VARIANT}.txt" | head -20 >&2
    echo
    echo "    Small floating-point divergence can be legitimate if the reduction" >&2
    echo "    order changed, but investigate before reporting timings." >&2
    exit 1
fi
