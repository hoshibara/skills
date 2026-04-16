#!/bin/bash
# Generate the env activation script for Flex Attention development
# Usage: generate_env.sh <output_file> <conda_env_name> <proxy_url> <oneapi_home> <gpu_arch> <huggingface_token>
#
# Arguments:
#   output_file      - Path to write the env script
#   conda_env_name   - Name of the conda environment
#   proxy_url        - Working proxy URL
#   oneapi_home      - Path to oneAPI installation
#   gpu_arch         - GPU architecture target (e.g., bmg, pvc, or empty)
#   huggingface_token - HuggingFace token (optional)

set -e

OUTPUT_FILE="${1:?Usage: generate_env.sh <output_file> <conda_env_name> <proxy_url> <oneapi_home> <gpu_arch> [huggingface_token]}"
CONDA_ENV_NAME="${2:?Missing conda_env_name}"
PROXY_URL="${3:?Missing proxy_url}"
ONEAPI_HOME="${4:?Missing oneapi_home}"
GPU_ARCH="${5:-}"
HF_TOKEN="${6:-}"

# Build AOT/ARCH lines
AOT_LINE=""
ARCH_LINE=""
if [[ -n "$GPU_ARCH" ]]; then
    AOT_LINE="export USE_AOT_DEVLIST='${GPU_ARCH}'"
    ARCH_LINE="export TORCH_XPU_ARCH_LIST='${GPU_ARCH}'"
else
    AOT_LINE="# export USE_AOT_DEVLIST=''  # TODO: Set GPU arch (bmg for B580/B60, pvc for 1550/1100)"
    ARCH_LINE="# export TORCH_XPU_ARCH_LIST=''  # TODO: Set GPU arch (bmg for B580/B60, pvc for 1550/1100)"
fi

# Build HF token line
HF_LINE=""
if [[ -n "$HF_TOKEN" ]]; then
    HF_LINE="export HUGGING_FACE_HUB_TOKEN=\"${HF_TOKEN}\""
else
    HF_LINE="# export HUGGING_FACE_HUB_TOKEN=\"\"  # TODO: Set your HuggingFace token"
fi

cat > "$OUTPUT_FILE" << ENVEOF
#!/bin/bash
# Flex Attention Environment - Auto-generated $(date +%Y-%m-%d)
# Conda env: ${CONDA_ENV_NAME}

# ===== Proxy Settings =====
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export no_proxy="localhost,127.0.0.1,intel.com"
export NO_PROXY="localhost,127.0.0.1,intel.com"

# ===== HuggingFace =====
${HF_LINE}

# ===== Conda Environment =====
CONDA_ENV_NAME=${CONDA_ENV_NAME}

# Initialize conda for non-interactive shells (e.g., scripts run with bash)
CONDA_BASE=\$(conda info --base 2>/dev/null || echo "\${CONDA_PREFIX:-/mnt/miniforge3}")
if [[ -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "\$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda deactivate 2>/dev/null || true
conda activate \$CONDA_ENV_NAME || true

SCRIPT_PATH="\${BASH_SOURCE[0]:-\$0}"
SCRIPT_DIR=\$(cd \$(dirname "\$SCRIPT_PATH"); pwd)
echo "Script dir: \$SCRIPT_DIR"

# ===== oneAPI =====
ONEAPI_HOME=${ONEAPI_HOME}
# Support both setvars.sh (older) and oneapi-vars.sh (newer)
if [[ -f "\${ONEAPI_HOME}/oneapi-vars.sh" ]]; then
    source \${ONEAPI_HOME}/oneapi-vars.sh --force
elif [[ -f "\${ONEAPI_HOME}/setvars.sh" ]]; then
    source \${ONEAPI_HOME}/setvars.sh --force
else
    echo "ERROR: No oneAPI init script found in \${ONEAPI_HOME}"
fi

# ===== XPU Build Settings =====
export USE_XPU=1
export PYTORCH_ENABLE_XPU_FALLBACK=0
export PYTORCH_DEBUG_XPU_FALLBACK=1
export USE_KINETO=1
export BUILD_SEPARATE_OPS=ON

${AOT_LINE}
${ARCH_LINE}

# ===== TorchInductor Cache =====
export TORCHINDUCTOR_CACHE_DIR="\${SCRIPT_DIR}/torchinductor_cache"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0
echo "TORCHINDUCTOR_CACHE_DIR: \$TORCHINDUCTOR_CACHE_DIR"

# ===== Build Toolchain =====
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH="\${CONDA_PREFIX:-'\$(dirname \$(which conda))/../'}:\${CMAKE_PREFIX_PATH}"

# ===== GPU Frequency Pinning =====
# Uncomment and adjust renderD* path for your system:
# echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# echo 2683 | sudo tee /sys/class/drm/renderD129/device/tile0/gt0/freq0/min_freq
# echo 2683 | sudo tee /sys/class/drm/renderD129/device/tile0/gt0/freq0/max_freq

echo "============================================="
echo " Successfully Activated Env"
echo "  \$CONDA_ENV_NAME"
echo "============================================="
ENVEOF

chmod +x "$OUTPUT_FILE"
echo "Generated env script: $OUTPUT_FILE"
