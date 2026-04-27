#!/bin/bash
# Sweep all (model, precision) combos × all presets.
#
# Default: 6 models × `default` preset group (9 presets) = 54 runs.
# Selectable with env vars:
#   MODELS="llama31_mxfp8 llama31_mxfp4 llama4_mxfp8 llama4_mxfp4 flux_mxfp8 flux_mxfp4"
#   PRESETS="baseline combo_kernels …"            # explicit list
#   PRESET_GROUP="default|tier2|tier3_pair|tier3_trip|tier3_abl|full"
#                                                  # resolves via inductor_cfg.PRESET_GROUPS
#
# Failures are logged but do not abort the sweep.
set -u

WS="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${WS}/scripts"
LOGS="${WS}/logs"
mkdir -p "${LOGS}"

MODELS="${MODELS:-llama31_mxfp8 llama31_mxfp4 llama4_mxfp8 llama4_mxfp4 flux_mxfp8 flux_mxfp4}"

# Resolve PRESETS: explicit list wins; else PRESET_GROUP via Python; else 'default' group.
if [[ -z "${PRESETS:-}" ]]; then
    PG="${PRESET_GROUP:-default}"
    PRESETS="$(python -c "
import sys, os, io
sys.path.insert(0, '${SCRIPTS}')
# suppress inductor_cfg's auto-apply print, capture stdout
_orig=sys.stdout; sys.stdout=io.StringIO()
from inductor_cfg import PRESET_GROUPS
sys.stdout=_orig
g = PRESET_GROUPS.get('${PG}')
if g is None:
    print('UNKNOWN_GROUP', file=sys.stderr); sys.exit(2)
print(' '.join(g))
")"
fi

date_tag="$(date +%Y%m%d_%H%M%S)"
sweep_log="${LOGS}/sweep_${date_tag}.log"

echo "sweep started ${date_tag}" | tee "${sweep_log}"
echo "  models  : ${MODELS}"   | tee -a "${sweep_log}"
echo "  presets : ${PRESETS}"  | tee -a "${sweep_log}"

failed=()

for combo in ${MODELS}; do
    model="${combo%%_*}"      # llama31 / llama4 / flux
    precision="${combo#*_}"   # mxfp8 / mxfp4
    for preset in ${PRESETS}; do
        echo
        echo "--- [$(date +%H:%M:%S)] ${combo} / ${preset} ---" | tee -a "${sweep_log}"
        if bash "${SCRIPTS}/run_one.sh" "${model}" "${precision}" "${preset}" \
                >> "${sweep_log}" 2>&1; then
            echo "    ok" | tee -a "${sweep_log}"
        else
            echo "    FAILED" | tee -a "${sweep_log}"
            failed+=("${combo}/${preset}")
        fi
    done
done

echo | tee -a "${sweep_log}"
echo "sweep complete: ${#failed[@]} failures" | tee -a "${sweep_log}"
for f in "${failed[@]}"; do echo "  FAILED: $f" | tee -a "${sweep_log}"; done

# Generate aggregate table
python "${SCRIPTS}/summarize.py" --results "${WS}/results" \
    --out "${WS}/results/SWEEP_SUMMARY.md" || true
echo "summary -> ${WS}/results/SWEEP_SUMMARY.md" | tee -a "${sweep_log}"

