#!/bin/bash
# Install PyTorch + Triton from source with XPU support
# Usage: install_pytorch.sh [pytorch_dir]
#
# IMPORTANT: This script MUST be run after sourcing env.sh so the correct
# conda environment is active. It will refuse to run otherwise to prevent
# polluting other Python environments.
#
# Arguments:
#   pytorch_dir - Directory to clone/use pytorch (default: ./pytorch)

set -e

# ===== Environment Guard: Refuse to run in base/wrong env =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ENV_SCRIPT="${WORKSPACE_ROOT}/env.sh"

if [[ -f "$ENV_SCRIPT" ]]; then
    echo "===== Activating environment from $ENV_SCRIPT ====="
    source "$ENV_SCRIPT"
else
    echo "ERROR: env.sh not found at $ENV_SCRIPT"
    echo "Run Phase 1 (generate env.sh) first."
    exit 1
fi

# Verify we are NOT in base or empty env
CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ -z "$CURRENT_ENV" || "$CURRENT_ENV" == "base" ]]; then
    echo "ERROR: Conda environment is '${CURRENT_ENV:-<none>}'. Refusing to continue."
    echo "Activate the correct FA conda env first (source env.sh)."
    exit 1
fi

# Verify env.sh was fully sourced (not just conda activate)
if [[ "${USE_XPU:-}" != "1" ]]; then
    echo "ERROR: USE_XPU is not set to 1. env.sh was not sourced properly."
    exit 1
fi
if ! command -v icpx &>/dev/null; then
    echo "ERROR: icpx not found on PATH. oneAPI was not sourced."
    exit 1
fi
if [[ -z "${http_proxy:-}" ]]; then
    echo "WARNING: http_proxy is not set. Proxy may be needed for downloads."
fi
echo "===== Conda env: $CURRENT_ENV (python: $(python --version)) ====="
echo "===== USE_XPU=$USE_XPU, ARCH=${TORCH_XPU_ARCH_LIST:-unset}, icpx=$(which icpx) ====="

# ===== Parse arguments =====
CLEAN_BUILD=false
PYTORCH_DIR="pytorch"
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN_BUILD=true ;;
        *) PYTORCH_DIR="$arg" ;;
    esac
done

# ===== Step 0: Hard reset if --clean or stale CMake cache detected =====
if [[ -d "$PYTORCH_DIR" ]]; then
    NEEDS_CLEAN=false
    if [[ "$CLEAN_BUILD" == true ]]; then
        NEEDS_CLEAN=true
        echo "===== Step 0: Hard reset (--clean requested) ====="
    elif [[ -f "$PYTORCH_DIR/build/CMakeCache.txt" ]]; then
        # Check if CMakeCache has paths from a different oneAPI than current CMPLR_ROOT
        CACHED_SYCL=$(grep '^SYCL_COMPILER:' "$PYTORCH_DIR/build/CMakeCache.txt" 2>/dev/null | head -1)
        if [[ -n "$CACHED_SYCL" && "$CACHED_SYCL" != *"$CMPLR_ROOT"* ]]; then
            NEEDS_CLEAN=true
            echo "===== Step 0: Hard reset (stale SYCL cache detected) ====="
            echo "  Cached: $CACHED_SYCL"
            echo "  Current CMPLR_ROOT: $CMPLR_ROOT"
        fi
    fi
    if [[ "$NEEDS_CLEAN" == true ]]; then
        pushd "$PYTORCH_DIR" > /dev/null
        git reset --hard HEAD
        git clean -ffdx
        git submodule deinit -f .
        popd > /dev/null
        echo "===== Hard reset complete ====="
    fi
fi

echo "===== Step 1: Clone PyTorch ====="
if [[ ! -d "$PYTORCH_DIR" ]]; then
    git clone https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
    cd "$PYTORCH_DIR"
    git remote add hoshibara https://github.com/hoshibara/pytorch.git
    git fetch hoshibara
else
    echo "PyTorch directory already exists: $PYTORCH_DIR"
    cd "$PYTORCH_DIR"
    if ! git remote | grep -q hoshibara; then
        git remote add hoshibara https://github.com/hoshibara/pytorch.git
        git fetch hoshibara
    fi
fi

echo "===== Step 2: Install build dependencies ====="
conda install mkl-static mkl-include -y
conda install cmake ninja -y
# Skip conda gcc/gxx — conda GCC 15.2 sysroot layout conflicts with icpx SYCL compilation.
# System GCC (13.x) works correctly with icpx. CC/CXX should be set in env.sh.
if command -v gcc &>/dev/null && command -v g++ &>/dev/null; then
    echo "Using system compiler: $(gcc --version | head -1)"
else
    echo "ERROR: No system gcc/g++ found. Install gcc/g++ (e.g. apt install gcc g++)."
    exit 1
fi

echo "===== Step 3: Clean previous installs ====="
pip uninstall torch -y 2>/dev/null || true
pip uninstall torch -y 2>/dev/null || true
pip uninstall torch -y 2>/dev/null || true

echo "===== Step 4: Sync submodules ====="
git submodule sync
git submodule update --init --recursive

echo "===== Step 5: Install requirements ====="
pip install -r requirements.txt

echo "===== Step 6: Clean previous triton installs ====="
pip uninstall pytorch-triton-xpu triton-xpu triton -y 2>/dev/null || true
pip uninstall pytorch-triton-xpu triton-xpu triton -y 2>/dev/null || true
pip uninstall pytorch-triton-xpu triton-xpu triton -y 2>/dev/null || true

echo "===== Step 7: Build Triton ====="
make triton

echo "===== Step 8: Build and install PyTorch ====="
python -m pip install --no-build-isolation -v -e .

echo "===== Installation Complete ====="
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}')" || echo "WARNING: torch import check failed"
