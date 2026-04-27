#!/bin/bash
# Run ONE (model, precision, preset) combo under unitrace.
#
# Usage:
#   bash run_one.sh <model> <precision> <preset>
#     model     : llama31 | llama4 | flux
#     precision : mxfp8 | mxfp4
#     preset    : baseline | combo_kernels | benchmark_combo |
#                 epilogue_first | prologue_fusion | realize_reads |
#                 expand_dim | aggressive | all
#
# Output: $WS/results/<model>_<precision>/<preset>/
#   ├── <name>.csv     unitrace device-timing CSV
#   ├── <name>.json    chrome trace
#   ├── console.log    full stdout
#   └── kernel_summary.md
set -eo pipefail

ENV_SH="/root/xingyuan/projects/20260416-bmg-ao/env.sh"
WS="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${WS}/scripts"
RESULTS="${WS}/results"
TMPDIR_LOCAL="${WS}/tmp"
mkdir -p "${RESULTS}" "${TMPDIR_LOCAL}"
export TMPDIR="${TMPDIR_LOCAL}"
# TORCHINDUCTOR_CACHE_DIR is set per-run (per preset) below to avoid kernel
# cache leaking across presets — see "fresh cache" block.

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <model:llama31|llama4|flux> <precision:mxfp8|mxfp4> <preset>"
    exit 2
fi
MODEL="$1"
PRECISION="$2"
PRESET="$3"

WARMUP="${WARMUP:-}"
MEASURE="${MEASURE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2}"
CONFIG="${CONFIG:-}"
MODE="${MODE:-compile}"

UNITRACE_OPTS="-d -v -s --chrome-kernel-logging"

# shellcheck disable=SC1090
source "${ENV_SH}" >/dev/null

NAME="${MODEL}_${PRECISION}_${PRESET}"
OUTDIR="${RESULTS}/${MODEL}_${PRECISION}/${PRESET}"
mkdir -p "${OUTDIR}"

# ── Fresh Inductor cache per run ──────────────────────────────────────
# Per-preset cache dir under the run's OUTDIR, wiped before every run so
# previous compiled kernels from other presets / earlier attempts cannot
# leak in and skew compile time, autotune, or fusion decisions.
# Also wipe the user-default cache (~/.cache/torch_inductor*, /tmp/torchinductor_$USER)
# in case Triton / TorchInductor falls back there for any reason.
export TORCHINDUCTOR_CACHE_DIR="${OUTDIR}/inductor_cache"
rm -rf "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
rm -rf "${HOME}/.cache/torch_inductor"* "/tmp/torchinductor_${USER}" 2>/dev/null || true

echo "============================================================"
echo "  ${NAME}    mode=${MODE}    preset=${PRESET}"
echo "  outdir=${OUTDIR}"
echo "  cache  =${TORCHINDUCTOR_CACHE_DIR}  (wiped)"
echo "  warmup=${WARMUP}  measure=${MEASURE}"
echo "============================================================"

export INDUCTOR_CFG_PRESET="${PRESET}"
export PYTHONPATH="${SCRIPTS}:${PYTHONPATH:-}"

case "${MODEL}" in
    llama31)
        CFG_NAME="${CONFIG:-4-Func}"
        WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}"
        CMD=( python "${SCRIPTS}/profile_llama31.py"
              --precision "${PRECISION}" --mode "${MODE}"
              --warmup "${WARMUP_L}" --measure "${MEASURE_L}"
              --config-name "${CFG_NAME}"
              --max-new-tokens "${MAX_NEW_TOKENS}" )
        ;;
    llama4)
        CFG_NAME="${CONFIG:-4-Func}"
        # llama4 measure is very fast (3-6ms/iter); bump iters so PTI
        # has enough wall-clock to log per-kernel device timings.
        WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}"
        CMD=( python "${SCRIPTS}/profile_llama4.py"
              --precision "${PRECISION}" --mode "${MODE}"
              --warmup "${WARMUP_L}" --measure "${MEASURE_L}"
              --config-name "${CFG_NAME}" )
        ;;
    flux)
        CFG_NAME="${CONFIG:-4-Func}"
        WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}"
        CMD=( python "${SCRIPTS}/profile_flux.py"
              --precision "${PRECISION}" --mode "${MODE}"
              --warmup "${WARMUP_L}" --measure "${MEASURE_L}"
              --config-name "${CFG_NAME}" )
        ;;
    *)
        echo "unknown model ${MODEL}"; exit 2 ;;
esac

# ── Pass A — clean latency, no unitrace overhead ──────────────────────
# unitrace's PTI hooks add per-kernel-launch host overhead (us-scale per
# call → noticeable on host-bound paths like Llama generate). To get a
# trustworthy wall-clock latency, run the workload once without unitrace
# and capture its `avg latency : X ms` line.
echo
echo "--- Pass A (clean latency, no unitrace) ---"
# Pass A may segfault during interpreter cleanup (XPU/MXTensor finalize bug)
# but the `avg latency : X ms` line is already printed before that — so we
# tolerate non-zero exit. set -e is locally suppressed.
set +e
PTI_ENABLE_COLLECTION=0 "${CMD[@]}" 2>&1 | tee "${OUTDIR}/console_clean.log"
PASS_A_RC=${PIPESTATUS[0]}
set -e
if [[ ${PASS_A_RC} -ne 0 ]]; then
    if grep -q "avg latency\|\[measure" "${OUTDIR}/console_clean.log"; then
        echo "  Pass A exited rc=${PASS_A_RC} after measure — tolerated (latency captured)"
    else
        echo "  Pass A FAILED (rc=${PASS_A_RC}) before measure — no clean latency"
    fi
fi

# ── Pass B — kernel timing, under unitrace ────────────────────────────
echo
echo "--- Pass B (kernel timing, under unitrace) ---"
unitrace ${UNITRACE_OPTS} --start-paused \
    --output-dir-path "${OUTDIR}" \
    -o "${OUTDIR}/${NAME}.csv" \
    "${CMD[@]}" 2>&1 | tee "${OUTDIR}/console.log"

python "${SCRIPTS}/analyze_unitrace_csv.py" "${OUTDIR}" \
    > "${OUTDIR}/kernel_summary.md" || true

echo "done -> ${OUTDIR}"
