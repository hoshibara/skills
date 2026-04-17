#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")"; pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
RESULT_DIR="${WORKSPACE_ROOT}/unitrace_results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

FLUX_DIR="${WORKSPACE_ROOT}/frameworks.ai.pytorch.gpu-models/presi-models/CRI/reduced-flux"
LLAMA_DIR="${WORKSPACE_ROOT}/frameworks.ai.pytorch.gpu-models/presi-models/reduced-llama4"

# unitrace options: device timing + verbose (kernel shapes) + submission timing + chrome timeline
UNITRACE_OPTS="-d -v -s --chrome-kernel-logging"

# Warmup / measurement iterations for compile mode
WARMUP=2
MEASURE=3

echo "============================================="
echo " Unitrace Profiling Run: ${TIMESTAMP}"
echo " Results: ${RESULT_DIR}"
echo " Compile warmup: ${WARMUP} iters, measure: ${MEASURE} iters"
echo "============================================="

# For eager mode: profile everything (no warmup needed)
run_profile_eager() {
    local name="$1"
    local workdir="$2"
    shift 2
    local outdir="${RESULT_DIR}/${name}"
    mkdir -p "${outdir}"

    echo ""
    echo ">>> [$(date +%H:%M:%S)] Running: ${name} (eager)"
    echo "    Cmd: $@"
    echo "    Out: ${outdir}"

    (cd "${workdir}" && unitrace ${UNITRACE_OPTS} \
        --output-dir-path "${outdir}" \
        -o "${outdir}/${name}.csv" \
        "$@") 2>&1 | tee "${outdir}/console.log"

    echo "<<< [$(date +%H:%M:%S)] Done: ${name}"
    echo ""
}

# For compile mode: use --start-paused, wrapper enables PTI_ENABLE_COLLECTION after warmup
run_profile_compile() {
    local name="$1"
    shift
    local outdir="${RESULT_DIR}/${name}"
    mkdir -p "${outdir}"

    echo ""
    echo ">>> [$(date +%H:%M:%S)] Running: ${name} (compile, warmup=${WARMUP}, measure=${MEASURE})"
    echo "    Cmd: $@"
    echo "    Out: ${outdir}"

    (cd "${WORKSPACE_ROOT}" && unitrace ${UNITRACE_OPTS} \
        --start-paused \
        --output-dir-path "${outdir}" \
        -o "${outdir}/${name}.csv" \
        "$@") 2>&1 | tee "${outdir}/console.log"

    echo "<<< [$(date +%H:%M:%S)] Done: ${name}"
    echo ""
}

# For compile debug: TORCH_COMPILE_DEBUG=1, no unitrace (perf not meaningful)
run_compile_debug() {
    local name="$1"
    local workdir="$2"
    shift 2
    local outdir="${RESULT_DIR}/${name}"
    mkdir -p "${outdir}"

    echo ""
    echo ">>> [$(date +%H:%M:%S)] Running: ${name} (TORCH_COMPILE_DEBUG=1, no unitrace)"
    echo "    Dir: ${workdir}"
    echo "    Cmd: $@"
    echo "    Out: ${outdir}"

    (cd "${workdir}" && \
        TORCH_COMPILE_DEBUG=1 \
        TORCH_COMPILE_DEBUG_DIR="${outdir}/torch_compile_debug" \
        "$@") 2>&1 | tee "${outdir}/console.log"

    echo "<<< [$(date +%H:%M:%S)] Done: ${name}"
    echo ""
}

# =============================================
# FLUX MXFP8
# =============================================
run_profile_eager "flux_mxfp8_eager" "${FLUX_DIR}" \
    python flux_dev_mxfp8.py

run_profile_compile "flux_mxfp8_compile" \
    env COMPILE=true python "${WORKSPACE_ROOT}/profile_flux.py" \
        --script flux_dev_mxfp8.py --warmup ${WARMUP} --measure ${MEASURE}

# =============================================
# FLUX MXFP4
# =============================================
run_profile_eager "flux_mxfp4_eager" "${FLUX_DIR}" \
    python flux_dev_mxfp4.py

run_profile_compile "flux_mxfp4_compile" \
    env COMPILE=true python "${WORKSPACE_ROOT}/profile_flux.py" \
        --script flux_dev_mxfp4.py --warmup ${WARMUP} --measure ${MEASURE}

# =============================================
# Llama4 MXFP8
# =============================================
run_profile_eager "llama4_mxfp8_eager" "${LLAMA_DIR}" \
    python llama4-FP8.py

run_profile_compile "llama4_mxfp8_compile" \
    env USE_COMPILE=true python "${WORKSPACE_ROOT}/profile_llama4.py" \
        --script llama4-FP8.py --warmup ${WARMUP} --measure ${MEASURE}

# =============================================
# Llama4 MXFP4
# =============================================
run_profile_eager "llama4_mxfp4_eager" "${LLAMA_DIR}" \
    python llama4-FP4.py

run_profile_compile "llama4_mxfp4_compile" \
    env USE_COMPILE=true python "${WORKSPACE_ROOT}/profile_llama4.py" \
        --script llama4-FP4.py --warmup ${WARMUP} --measure ${MEASURE}

# =============================================
# TORCH_COMPILE_DEBUG runs (kernel fusion analysis only)
# No unitrace, perf data not meaningful with this flag
# =============================================
echo ""
echo "============================================="
echo " TORCH_COMPILE_DEBUG runs (fusion analysis)"
echo "============================================="

run_compile_debug "flux_mxfp8_compile_debug" "${FLUX_DIR}" \
    env COMPILE=true python flux_dev_mxfp8.py

run_compile_debug "flux_mxfp4_compile_debug" "${FLUX_DIR}" \
    env COMPILE=true python flux_dev_mxfp4.py

run_compile_debug "llama4_mxfp8_compile_debug" "${LLAMA_DIR}" \
    env USE_COMPILE=true python llama4-FP8.py

run_compile_debug "llama4_mxfp4_compile_debug" "${LLAMA_DIR}" \
    env USE_COMPILE=true python llama4-FP4.py

# =============================================
# Summary
# =============================================
echo ""
echo "============================================="
echo " All profiling complete!"
echo " Results saved to: ${RESULT_DIR}"
echo "============================================="
echo ""
echo "Directory listing:"
for d in "${RESULT_DIR}"/*/; do
    name=$(basename "$d")
    file_count=$(find "$d" -type f | wc -l)
    echo "  ${name}: ${file_count} files"
done
echo ""
echo "Contents:"
find "${RESULT_DIR}" -type f | sort
