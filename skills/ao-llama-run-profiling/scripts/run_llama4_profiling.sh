#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")"; pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
RESULT_DIR="${WORKSPACE_ROOT}/unitrace_results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

LLAMA_DIR="${WORKSPACE_ROOT}/frameworks.ai.pytorch.gpu-models/presi-models/reduced-llama4"
UNITRACE_OPTS="-d -v -s --chrome-kernel-logging"
WARMUP=2
MEASURE=3

echo "============================================="
echo " Llama4 Unitrace Profiling: ${TIMESTAMP}"
echo " Results: ${RESULT_DIR}"
echo " Warmup: ${WARMUP}, Measure: ${MEASURE}"
echo "============================================="

run_profile_eager() {
    local name="$1" workdir="$2"; shift 2
    local outdir="${RESULT_DIR}/${name}"
    mkdir -p "${outdir}"
    echo ">>> [$(date +%H:%M:%S)] ${name} (eager)"
    (cd "${workdir}" && unitrace ${UNITRACE_OPTS} \
        --output-dir-path "${outdir}" -o "${outdir}/${name}.csv" \
        "$@") 2>&1 | tee "${outdir}/console.log"
    echo "<<< [$(date +%H:%M:%S)] Done: ${name}"
}

run_profile_compile() {
    local name="$1"; shift
    local outdir="${RESULT_DIR}/${name}"
    mkdir -p "${outdir}"
    echo ">>> [$(date +%H:%M:%S)] ${name} (compile, warmup=${WARMUP})"
    (cd "${WORKSPACE_ROOT}" && unitrace ${UNITRACE_OPTS} --start-paused \
        --output-dir-path "${outdir}" -o "${outdir}/${name}.csv" \
        "$@") 2>&1 | tee "${outdir}/console.log"
    echo "<<< [$(date +%H:%M:%S)] Done: ${name}"
}

# ===== Llama4 MXFP8 =====
run_profile_eager "llama4_mxfp8_eager" "${LLAMA_DIR}" python llama4-FP8.py

run_profile_compile "llama4_mxfp8_compile" \
    env USE_COMPILE=true python "${WORKSPACE_ROOT}/profile_llama4.py" \
        --script llama4-FP8.py --warmup ${WARMUP} --measure ${MEASURE}

# ===== Llama4 MXFP4 =====
run_profile_eager "llama4_mxfp4_eager" "${LLAMA_DIR}" python llama4-FP4.py

run_profile_compile "llama4_mxfp4_compile" \
    env USE_COMPILE=true python "${WORKSPACE_ROOT}/profile_llama4.py" \
        --script llama4-FP4.py --warmup ${WARMUP} --measure ${MEASURE}

# ===== Summary =====
echo ""
echo "============================================="
echo " Llama4 profiling complete! Results: ${RESULT_DIR}"
echo "============================================="
for d in "${RESULT_DIR}"/*/; do
    echo "  $(basename "$d"): $(find "$d" -type f | wc -l) files"
done
