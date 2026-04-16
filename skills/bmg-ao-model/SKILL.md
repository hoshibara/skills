---
name: bmg-ao-model
description: "Set up Intel BMG/B60 environment and run FLUX MXFP8/MXFP4 with conda + oneAPI, including proxy detection, env.sh generation, package installation, and runtime verification."
argument-hint: "Optional: conda env name, oneAPI home, GPU arch"
---

# Intel BMG AO Model Setup (FLUX CRI on B60)

Automates end-to-end setup for FLUX on Intel BMG/B60: proxy detection, conda env creation, oneAPI configuration, env.sh generation, package build/install, and MXFP8/MXFP4 runs.

## When to Use
- Setting up a new FLUX CRI runtime on Intel BMG/B60
- Creating a reproducible conda + oneAPI environment for BMG model runs
- Building the validated PyTorch/Triton/TorchAO stack required by FLUX MXFP8/MXFP4

## CRITICAL: Single Terminal Rule — ABSOLUTE, NO EXCEPTIONS

**ALL commands in this skill — from Phase 1 through Phase 3 — MUST execute in the SAME terminal session.** This is non-negotiable:

- Use `run_in_terminal` **ONLY ONCE** to create the terminal at the very start. Record the terminal ID.
- **ALL subsequent commands MUST use `send_to_terminal`** with that same terminal ID. This includes build, install, and run commands.
- **NEVER call `run_in_terminal` a second time. NEVER.** Each `run_in_terminal` invocation creates a NEW terminal that does NOT inherit env vars, conda activation, or oneAPI paths from previous terminals. Even if the script sources env.sh internally, opening a second terminal violates this rule.
- If the user specifies an existing terminal to use, use `send_to_terminal` with that terminal — do NOT create a new one.
- For long-running builds (e.g., `make triton`, `pip install -e .`), use `get_terminal_output` to poll for completion instead of setting a timeout.

**Why this matters:** conda activate, oneAPI source, proxy env vars, and build flags are all shell state. A new terminal loses ALL of this. Sourcing `env.sh` later in another script is NOT a substitute for maintaining a single terminal session.

## CRITICAL: Environment Isolation Rules

**Every command that touches Python packages (pip, conda install, compile scripts, python setup.py, etc.) MUST run inside the activated BMG conda environment.** Never run these in `base` or any other env.

1. **At the very beginning of the terminal session**, run `source ./env.sh` BEFORE doing anything else. This applies to build terminals, test terminals, Python REPL sessions, etc.
2. Verify with `echo $CONDA_DEFAULT_ENV` — it must NOT be `base` or empty.
3. If `CONDA_DEFAULT_ENV` is empty or `base`, stop and reactivate the target conda env before continuing.
4. **`source ./env.sh` MUST be executed as a standalone command.** Do not chain it with `&&`, `;`, pipes, subshells, or inline command groups.
5. For agent execution with `send_to_terminal`, first send only `source ./env.sh`, wait for prompt return, then send the next command.

## Procedure

### Phase 1: Generate Environment Activation Script

#### Step 1 — Detect Working Proxy

Detect a working Intel proxy from all known options:

```bash
PROXIES=(
    "http://proxy.ims.intel.com:911"
    "http://proxy-prc.intel.com:913"
    "http://child-ir.intel.com:912"
    "http://proxy-chain.intel.com:912"
    "http://proxy-dmz.intel.com:911"
    "http://proxy-us.intel.com:912"
)
TEST_URL="https://github.com"
WORKING_PROXY=""

for proxy in "${PROXIES[@]}"; do
    if curl -s -o /dev/null -w "%{http_code}" --proxy "$proxy" --connect-timeout 2 --max-time 3 "$TEST_URL" 2>/dev/null | grep -q "^[23]"; then
        WORKING_PROXY="$proxy"
        break
    fi
done

echo "WORKING_PROXY=$WORKING_PROXY"
```

If `WORKING_PROXY` is empty, ask the user to provide a manually verified proxy URL.

#### Step 2 — Detect oneAPI Installation

Search for oneAPI installations in **both system and user home** directories, list all versions, and auto-select the best candidate:

```bash
BEST_ONEAPI=""
BEST_SCRIPT=""
BEST_VERSION="0"
BEST_COUNT=-1

for base in /opt/intel/oneapi "$HOME/intel/oneapi"; do
    for script in setvars.sh oneapi-vars.sh; do
        if [[ -f "$base/$script" ]]; then
            comp_dirs=$(find "$base" -maxdepth 2 -type d -name "20*" 2>/dev/null || true)
            count=$(printf '%s\n' "$comp_dirs" | sed '/^$/d' | wc -l | tr -d ' ')
            version=$(printf '%s\n' "$comp_dirs" | sed '/^$/d' | awk -F/ '{print $NF}' | sort -V | tail -n1)
            version=${version:-0}
            echo "Found oneAPI at: $base (via $script, version=$version, count=$count)"

            newest=$(printf '%s\n%s\n' "$BEST_VERSION" "$version" | sort -V | tail -n1)
            if [[ "$newest" == "$version" && "$version" != "$BEST_VERSION" ]]; then
                BEST_ONEAPI="$base"
                BEST_SCRIPT="$script"
                BEST_VERSION="$version"
                BEST_COUNT="$count"
            elif [[ "$version" == "$BEST_VERSION" && "$count" -gt "$BEST_COUNT" ]]; then
                BEST_ONEAPI="$base"
                BEST_SCRIPT="$script"
                BEST_COUNT="$count"
            fi
        fi
    done
done

if [[ -z "$BEST_ONEAPI" ]]; then
    echo "ERROR: no oneAPI installation found under /opt/intel/oneapi or $HOME/intel/oneapi"
    false
fi

export ONEAPI_HOME="$BEST_ONEAPI"
export ONEAPI_SCRIPT="$BEST_SCRIPT"
echo "Selected ONEAPI_HOME=$ONEAPI_HOME"
echo "Selected ONEAPI_SCRIPT=$ONEAPI_SCRIPT"
```

Selection logic:
1. Prefer the **latest version number** (e.g., `2025.10` > `2025.3`)
2. If tied, prefer the one with **more component directories**

`ONEAPI_HOME` points to the root containing the init script (e.g., `/opt/intel/oneapi`). `ONEAPI_SCRIPT` is `setvars.sh` or `oneapi-vars.sh`.

**Important:** Always use the concrete versioned path, never a `latest` symlink. Verify the init script exists before using it.

#### Step 3 — Detect GPU Model

Determine the Intel GPU model and export architecture target:

```bash
gpu_info="$(xpu-smi discovery 2>/dev/null || true)"
if [[ -z "$gpu_info" ]]; then
    gpu_info="$(lspci | grep -i "vga\|display\|3d" | grep -i intel || true)"
fi
echo "$gpu_info"

if echo "$gpu_info" | grep -Eiq 'B580|B60|bmg'; then
    export GPU_ARCH="bmg"
elif echo "$gpu_info" | grep -Eiq '1550|1100|pvc|Max'; then
    export GPU_ARCH="pvc"
else
    export GPU_ARCH=""
    echo "WARNING: could not determine GPU_ARCH automatically"
fi

echo "GPU_ARCH=${GPU_ARCH}"
```

Map GPU model to architecture:
| GPU | Architecture |
|-----|-------------|
| Arc B580, B60 | `bmg` |
| Data Center GPU Max 1550, 1100 | `pvc` |

If GPU detection is unclear, keep `GPU_ARCH` empty and set `USE_AOT_DEVLIST` / `TORCH_XPU_ARCH_LIST` manually in `env.sh`.

#### Step 4 — Create Conda Environment

Generate a name following the convention `BMG-MODEL-<date>` (e.g., `BMG-MODEL-20260417`) unless the user specifies one:

```bash
CONDA_ENV_NAME="BMG-MODEL-$(date +%Y%m%d)"
conda create -n $CONDA_ENV_NAME python=3.12 -y
conda activate $CONDA_ENV_NAME
pip install uv
```

**Why uv?** `uv` is a fast Python package installer. Once installed, use `uv pip install` instead of `pip install` for all subsequent batch package installations to significantly speed up dependency resolution and downloads.

#### Step 5 — Generate env.sh

Generate a local `env.sh` using the detected values:

```bash
cat << ENVEOF > env.sh
#!/bin/bash
# Conda env: ${CONDA_ENV_NAME}

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
    echo "ERROR: run with 'source ./env.sh'" >&2
    exit 1
fi

export http_proxy="${WORKING_PROXY}"
export https_proxy="${WORKING_PROXY}"
export HTTP_PROXY="${WORKING_PROXY}"
export HTTPS_PROXY="${WORKING_PROXY}"
export no_proxy="localhost,127.0.0.1,intel.com"
export NO_PROXY="localhost,127.0.0.1,intel.com"

# Optional HF token passthrough
# export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Initialize conda
CONDA_BASE=\$(conda info --base 2>/dev/null || echo "\${CONDA_PREFIX:-/mnt/miniforge3}")
if [[ -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "\$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda deactivate 2>/dev/null || true
conda activate ${CONDA_ENV_NAME} || { echo "ERROR: failed to activate ${CONDA_ENV_NAME}"; return 1; }

# Source oneAPI
source ${ONEAPI_HOME}/${ONEAPI_SCRIPT} --force

# XPU build/runtime settings
export USE_XPU=1
export PYTORCH_ENABLE_XPU_FALLBACK=0
export PYTORCH_DEBUG_XPU_FALLBACK=1
export USE_KINETO=1
export BUILD_SEPARATE_OPS=ON
export USE_AOT_DEVLIST="${GPU_ARCH}"
export TORCH_XPU_ARCH_LIST="${GPU_ARCH}"

# TorchInductor cache
SCRIPT_PATH="\${BASH_SOURCE[0]:-\$0}"
SCRIPT_DIR=\$(cd \$(dirname "\$SCRIPT_PATH"); pwd)
export TORCHINDUCTOR_CACHE_DIR="\${SCRIPT_DIR}/torchinductor_cache"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0

echo "============================================="
echo " Successfully Activated Env: ${CONDA_ENV_NAME}"
echo " ONEAPI_HOME: ${ONEAPI_HOME}"
echo " ONEAPI_SCRIPT: ${ONEAPI_SCRIPT}"
echo " GPU_ARCH: ${GPU_ARCH}"
echo "============================================="
ENVEOF

chmod +x env.sh
```

The generated `env.sh` is saved to the workspace root. Review it with the user and ask if any adjustments are needed (e.g., GPU frequency pinning, ZE_AFFINITY_MASK).

#### Step 6 — Activate and Verify

Source env.sh and verify **all critical environment variables** are set (not just conda):

Use this strict pattern:
- Valid: run `source ./env.sh` alone, then run verification commands.
- Invalid: `source ./env.sh && <other_command>`.

```bash
source ./env.sh

# Verify all key env vars are set
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "USE_XPU: $USE_XPU"
echo "ONEAPI sourced: $(which icpx 2>/dev/null || echo 'NOT FOUND')"
echo "Proxy: $http_proxy"
echo "TORCH_XPU_ARCH_LIST: $TORCH_XPU_ARCH_LIST"
python -c "import os; print('USE_XPU:', os.environ.get('USE_XPU', 'NOT SET'))"
```

All of the following must be true:
- `CONDA_DEFAULT_ENV` is the BMG env name (not `base`)
- `USE_XPU` is `1`
- `icpx` (oneAPI compiler) is on PATH
- `http_proxy` is set
- `TORCH_XPU_ARCH_LIST` is set (e.g., `bmg`)

If any check fails, env.sh was not sourced correctly.

### Phase 2: Install Packages for FLUX CRI on B60

#### Step 1 — Build and Install PyTorch (private-gpu)

Use the CRI-validated branch and commit from README:

```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
cd frameworks.ai.pytorch.private-gpu
git checkout v0.1.0_next
git checkout 113fe338498df16d8b830ce45c5618c84905b82d
git submodule sync
git submodule update --init --recursive
uv pip install -r requirements.txt
python setup.py install >log.txt 2>&1
cd ..
```

#### Step 2 — Build and Install intel-xpu-backend-for-triton

```bash
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton
git checkout ea1cad403cca274b3cd6a97ce9c2ff467b640d39
scripts/compile-triton.sh
cd ..
```

#### Step 3 — Build and Install TorchAO

```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ao
cd frameworks.ai.pytorch.torch-ao
git checkout 27b169f2cb0ae866756faabc72325239171a7d70
python setup.py install
cd ..
```

#### Step 4 — Install FLUX Runtime Packages

```bash
uv pip install transformers sentencepiece protobuf accelerate diffusers
```

#### Step 5 — Verify Runtime Stack

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU available:', torch.xpu.is_available()); print('XPU device:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')"
python -c "import diffusers, transformers, accelerate, sentencepiece, google.protobuf; print('Runtime packages import OK')"
```

### Phase 3: Run FLUX Models

#### Step 1 — Enter Reduced FLUX Directory

```bash
cd frameworks.ai.pytorch.gpu-models/presi-models/CRI/reduced-flux
```

#### Step 2 — Run MXFP8

```bash
# Functional check (128x128, eager)
python flux_dev_mxfp8.py

# Performance run (1024x1024, eager)
MODEL_CONFIG_NAME=48-Perf python flux_dev_mxfp8.py

# Compile path
COMPILE=true python flux_dev_mxfp8.py

# Performance + compile
MODEL_CONFIG_NAME=48-Perf COMPILE=true python flux_dev_mxfp8.py
```

#### Step 3 — Run MXFP4 (Optional Baseline Comparison)

```bash
# Functional check (128x128, eager)
python flux_dev_mxfp4.py

# Performance run (1024x1024, eager)
MODEL_CONFIG_NAME=48-Perf python flux_dev_mxfp4.py

# Compile path
COMPILE=true python flux_dev_mxfp4.py

# Performance + compile
MODEL_CONFIG_NAME=48-Perf COMPILE=true python flux_dev_mxfp4.py
```

#### Step 4 — Run Full Model Wrapper (Optional)

```bash
./run.sh --device xpu --mode eager
./run.sh --device xpu --mode compile --use_profiler
```

#### Step 5 — Hugging Face Access Check

FLUX models are gated. If model download fails, request access and login:

```bash
huggingface-cli login
```

## Completion Checklist

- [ ] Proxy detected and working
- [ ] oneAPI found and sourced
- [ ] GPU arch identified (or user warned)
- [ ] Conda env created and activated
- [ ] env.sh generated and saved to workspace root
- [ ] PyTorch private-gpu built and installed
- [ ] intel-xpu-backend-for-triton built
- [ ] TorchAO installed
- [ ] FLUX runtime packages installed
- [ ] `python flux_dev_mxfp8.py` runs successfully
