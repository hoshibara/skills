#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")"; pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
RESULT_DIR="${WORKSPACE_ROOT}/unitrace_results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

FLUX_DIR="${WORKSPACE_ROOT}/frameworks.ai.pytorch.gpu-models/presi-models/CRI/reduced-flux"
UNITRACE_OPTS="-d -v -s --chrome-kernel-logging"
WARMUP=2
MEASURE=3

echo "============================================="
echo " FLUX Unitrace Profiling: ${TIMESTAMP}"
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

run_compile_debug() {
    local name="$1" workdir="$2"; shift 2
    local outdir="${RESULT_DIR}/${name}"
    mkdir -p "${outdir}"
    echo ">>> [$(date +%H:%M:%S)] ${name} (TORCH_COMPILE_DEBUG)"
    (cd "${workdir}" && TORCH_COMPILE_DEBUG=1 \
        TORCH_COMPILE_DEBUG_DIR="${outdir}/torch_compile_debug" \
        "$@") 2>&1 | tee "${outdir}/console.log"
    echo "<<< [$(date +%H:%M:%S)] Done: ${name}"
}

# ===== FLUX MXFP8 =====
run_profile_eager "flux_mxfp8_eager" "${FLUX_DIR}" python flux_dev_mxfp8.py

run_profile_compile "flux_mxfp8_compile" \
    env COMPILE=true python "${WORKSPACE_ROOT}/profile_flux.py" \
        --script flux_dev_mxfp8.py --warmup ${WARMUP} --measure ${MEASURE}

# ===== FLUX MXFP4 =====
run_profile_eager "flux_mxfp4_eager" "${FLUX_DIR}" python flux_dev_mxfp4.py

run_profile_compile "flux_mxfp4_compile" \
    env COMPILE=true python "${WORKSPACE_ROOT}/profile_flux.py" \
        --script flux_dev_mxfp4.py --warmup ${WARMUP} --measure ${MEASURE}

# ===== TORCH_COMPILE_DEBUG =====
echo ""
echo "===== TORCH_COMPILE_DEBUG runs ====="
run_compile_debug "flux_mxfp8_compile_debug" "${FLUX_DIR}" env COMPILE=true python flux_dev_mxfp8.py
run_compile_debug "flux_mxfp4_compile_debug" "${FLUX_DIR}" env COMPILE=true python flux_dev_mxfp4.py

# ===== Summary =====
echo ""
echo "============================================="
echo " FLUX profiling complete! Results: ${RESULT_DIR}"
echo "============================================="
for d in "${RESULT_DIR}"/*/; do
    echo "  $(basename "$d"): $(find "$d" -type f | wc -l) files"
done
