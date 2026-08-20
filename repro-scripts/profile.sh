#!/usr/bin/env bash
# profile.sh <variant> — Nsight Compute counter collection.
#
# The counters behind the results table (instructions executed, occupancy,
# scheduler stats, excessive sectors) come from here.
#
# LAUNCH SELECTION MATTERS. Layer shapes vary through a 7B model, so the first
# matching launch is not representative of steady-state decode. --launch-skip is
# used to land in the middle of decode, and the skip value used for the published
# numbers is recorded alongside the report so the selection is not a hidden
# parameter.

set -euo pipefail

VARIANT="${1:?usage: profile.sh <variant>}"
BUILD="build_${VARIANT}"
GGUF="${GGUF:-../mistral-7b-v0.3-q4.gguf}"
SKIP="${LAUNCH_SKIP:-200}"     # TODO(vijay): set to the value used for the published table
OUTDIR="profiling"

mkdir -p "${OUTDIR}"

if [[ "${VARIANT}" == "upstream" ]]; then
    KERNEL="regex:mul_mat_vec_q"
else
    KERNEL="regex:custom_q4k_gemv"
fi

echo "==> ncu: variant=${VARIANT} kernel=${KERNEL} skip=${SKIP}"

ncu \
    --set full \
    --target-processes application-only \
    --kernel-name "${KERNEL}" \
    --launch-skip "${SKIP}" \
    --launch-count 1 \
    --force-overwrite \
    -o "${OUTDIR}/ncu_${VARIANT}" \
    "${BUILD}/bin/llama-cli" \
        -m "${GGUF}" \
        -p "The architecture of a GPU is" \
        -n 20 -ngl 99 -t 1 --single-turn

# Record how this report was produced, next to the report itself.
cat > "${OUTDIR}/ncu_${VARIANT}.provenance.txt" <<EOF
variant:      ${VARIANT}
kernel:       ${KERNEL}
launch-skip:  ${SKIP}
launch-count: 1
metric set:   full
model:        ${GGUF}
command:      -p "The architecture of a GPU is" -n 20 -ngl 99 -t 1 --single-turn
ncu version:  $(ncu --version 2>/dev/null | head -1)
date:         $(date -Is)
EOF

echo
echo "==> exporting CSV for the results table"
ncu --import "${OUTDIR}/ncu_${VARIANT}.ncu-rep" --csv --page raw \
    > "results/ncu_${VARIANT}.csv"

echo "report:     ${OUTDIR}/ncu_${VARIANT}.ncu-rep"
echo "provenance: ${OUTDIR}/ncu_${VARIANT}.provenance.txt"
echo "csv:        results/ncu_${VARIANT}.csv"
echo
echo "Commit the .ncu-rep files. They are the evidence for every counter in the"
echo "write-up, and they also record driver version, GPU SKU, and clock rates."
