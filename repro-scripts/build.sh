#!/usr/bin/env bash
# build.sh <variant> — apply a kernel variant to a pinned llama.cpp checkout and
# build it into its own tree.
#
#   ./scripts/build.sh upstream    # baseline, no patch applied
#   ./scripts/build.sh 2pack
#
# Run from inside the llama.cpp clone (checked out at the pinned SHA).
# Each variant gets build_<variant>/, so all variants coexist and there is never
# any ambiguity about which kernel a binary contains.

set -euo pipefail

VARIANT="${1:?usage: build.sh <upstream|2pack|w_tile|y_tile|tile_multiwarp|2rowCTA|double_buff>}"

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="build_${VARIANT}"
ARCH="${CUDA_ARCH:-80}"          # 80 = A100. Hopper: 90
JOBS="${JOBS:-16}"

# --- sanity: are we in a llama.cpp checkout at the expected revision? ---------
[[ -f ggml/src/ggml-cuda/mmvq.cu ]] || {
    echo "error: run this from the root of a llama.cpp checkout" >&2; exit 1; }

if [[ -n "${UPSTREAM_SHA:-}" ]]; then
    HAVE="$(git rev-parse HEAD)"
    if [[ "${HAVE}" != "${UPSTREAM_SHA}" ]]; then
        echo "warning: HEAD is ${HAVE}" >&2
        echo "         patches were generated against ${UPSTREAM_SHA}" >&2
        echo "         they may not apply cleanly" >&2
    fi
fi

# --- apply the variant --------------------------------------------------------
if [[ "${VARIANT}" != "upstream" ]]; then
    PATCH="${KIT}/patches/${VARIANT}.patch"
    [[ -f "${PATCH}" ]] || { echo "error: no such patch: ${PATCH}" >&2; exit 1; }

    # start from a clean tree so variants never stack on top of each other
    git checkout -- ggml/src/ggml-cuda/mmvq.cu 2>/dev/null || true
    rm -f ggml/src/ggml-cuda/custom_q4k.cu

    echo "==> applying patches/${VARIANT}.patch"
    git apply --check "${PATCH}" || {
        echo "error: patch does not apply — check that HEAD matches the pinned SHA" >&2
        exit 1; }
    git apply "${PATCH}"
else
    echo "==> upstream baseline (no patch)"
    git checkout -- ggml/src/ggml-cuda/mmvq.cu 2>/dev/null || true
    rm -f ggml/src/ggml-cuda/custom_q4k.cu
fi

# --- configure ----------------------------------------------------------------
# NOTE: --lineinfo is required for NCU to map counters back to source lines.
#       GGML_NATIVE=OFF keeps host codegen identical across machines.
echo "==> configuring ${BUILD}/ (sm_${ARCH})"
rm -rf "${BUILD}"
cmake -B "${BUILD}" -S . \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
    -DCMAKE_CUDA_COMPILER="${NVCC:-/usr/local/cuda/bin/nvcc}" \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-lineinfo"

echo "==> building"
cmake --build "${BUILD}" -j "${JOBS}"

echo
echo "built: ${BUILD}/bin/"
echo "next:  ./scripts/check.sh ${VARIANT}     # correctness, run this first"
echo "       ./scripts/bench.sh ${VARIANT}     # timing"
