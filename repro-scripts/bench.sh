#!/usr/bin/env bash
# bench.sh <variant> [reps] — kernel timing via Nsight Systems.
#
# Reports the MEDIAN over `reps` runs with observed min/max. Single-shot numbers
# at the ~20 microsecond scale are not reproducible and should not be published.
#
# Clocks MUST be locked first — see REPRODUCING.md section 7. Boost drift alone
# moves these timings by more than the effect being measured.

set -euo pipefail

VARIANT="${1:?usage: bench.sh <variant> [reps]}"
REPS="${2:-5}"
BUILD="build_${VARIANT}"
GGUF="${GGUF:-../mistral-7b-v0.3-q4.gguf}"
OUTDIR="results/${VARIANT}"

mkdir -p "${OUTDIR}"

# --- warn if clocks are not locked -------------------------------------------
if command -v nvidia-smi >/dev/null; then
    if ! nvidia-smi -q -d CLOCK 2>/dev/null | grep -qi "Applications Clocks"; then
        echo "warning: could not confirm locked clocks. Run:" >&2
        echo "  sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc <MHz>" >&2
    fi
fi

echo "==> nsys, ${REPS} reps, variant=${VARIANT}"

for i in $(seq 1 "${REPS}"); do
    echo "    run ${i}/${REPS}"
    nsys profile \
        --force-overwrite=true \
        --stats=true \
        -o "${OUTDIR}/nsys_run${i}" \
        "${BUILD}/bin/llama-completion" \
            -m "${GGUF}" \
            -p "The architecture of a GPU is" \
            -n 20 -ngl 99 -t 1 --single-turn \
        > "${OUTDIR}/nsys_run${i}.log" < /dev/null 2>&1
done

echo
echo "==> extracting GEMV kernel durations"
# Kernel name differs by variant: custom_q4k_gemv for patched builds,
# mul_mat_vec_q for the upstream baseline.
if [[ "${VARIANT}" == "upstream" ]]; then
    PATTERN="mul_mat_vec_q"
else
    PATTERN="custom_q4k_gemv"
fi

for i in $(seq 1 "${REPS}"); do
    nsys stats --report cuda_gpu_kern_sum --format csv \
        --output - "${OUTDIR}/nsys_run${i}.nsys-rep" 2>/dev/null \
        | grep "${PATTERN}" || true
done | tee "${OUTDIR}/kernel_times_raw.csv"

echo
echo "raw per-run data: ${OUTDIR}/kernel_times_raw.csv"
echo "report the MEDIAN of the average-duration column, with min/max as the spread."
echo
echo "State in the write-up that these are nsys wall-clock durations. NCU"
echo "serializes launches and flushes caches, so its durations are NOT"
echo "comparable to these and the two must not be mixed in one table."
