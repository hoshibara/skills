---
name: bmg-ao-env-setup
description: "Set up Intel BMG/B60 XPU environment with DPCPP_JGS compiler, PTI profiling tools, conda, env.sh, and source builds of PyTorch/Triton/TorchAO. Use when setting up a new BMG development environment, building PyTorch XPU stack from source, or preparing for any Intel XPU model inference task. Triggers on mentions of BMG setup, DPCPP, Intel XPU environment, or building PyTorch/Triton/TorchAO from source."
argument-hint: "Optional: conda env name, GPU arch"
---

# Intel BMG/B60 Environment Setup

End-to-end environment setup for Intel BMG/B60 XPU: proxy detection, DPCPP/PTI compiler setup, conda env creation, env.sh generation, and local source builds (PyTorch/Triton/TorchAO).

## When to Use
- Setting up a new development environment on Intel BMG/B60
- Creating a reproducible conda + DPCPP_JGS environment
- Building PyTorch/Triton/TorchAO from source for Intel XPU
- Preparing the environment before running any model (FLUX, Llama4, etc.)

## Bundled Resources

| Script | Purpose |
|--------|---------|
| `scripts/analyze_unitrace.py` | Model-agnostic unitrace CSV parser with category breakdown and comparative summary |

The analysis script is shared by model-specific skills (`ao-flux-run-profiling`, `ao-llama-run-profiling`).

---

## CRITICAL: Single Terminal Rule — ABSOLUTE, NO EXCEPTIONS

**ALL commands — from env setup through builds — MUST execute in the SAME terminal session.** This is non-negotiable:

- Use `run_in_terminal` **ONLY ONCE** to create the terminal at the very start. Record the terminal ID.
- **ALL subsequent commands MUST use `send_to_terminal`** with that same terminal ID.
- **NEVER call `run_in_terminal` a second time. NEVER.** Each invocation creates a NEW terminal that does NOT inherit env vars, conda activation, or compiler paths.
- For long-running builds, use `get_terminal_output` to poll for completion.

**Why:** conda activate, DPCPP source, proxy env vars, and build flags are all shell state. A new terminal loses ALL of this.

## CRITICAL: Long-Running Command Output — DO NOT PIPE TO tail

**For long-running commands (builds, installs), NEVER pipe output through `| tail` or `| head`.** This hides errors.

- **FORBIDDEN:** `python -m pip install -e . 2>&1 | tail -20`
- **CORRECT:** `python -m pip install -e . 2>&1 | tee /tmp/build.log`, then `tail -50 /tmp/build.log`

**Why:** Build commands take 30+ minutes. Piping through tail loses the full log.

## CRITICAL: Environment Activation — MANDATORY FIRST STEP

**`source ./env.sh` MUST be the VERY FIRST command in EVERY terminal session, BEFORE any other operation.**

### Activation Rules

1. **ALWAYS activate first.** Before `cd`, `git`, `pip`, `python`, or ANY other command.
2. **`source ./env.sh` MUST be executed ALONE.**
   - **FORBIDDEN:** `source ./env.sh && pip install ...` — NEVER chain with `&&`
   - **FORBIDDEN:** `source ./env.sh; python setup.py install` — NEVER chain with `;`
   - **CORRECT:** Send `source ./env.sh` alone, wait for completion, then send the next command separately.
3. **Verify after activation:** `echo $CONDA_DEFAULT_ENV` must NOT be `base` or empty.
4. **Environment activation is allowed only via `source ./env.sh`.** Do not manually run `conda activate`.

**Why:** env.sh sets conda, DPCPP compiler, PTI paths, proxy, and build flags. Without it, builds silently use wrong Python/compiler.

## CRITICAL: XPU-Only Package Policy

- Target platform is Intel XPU only, not CUDA.
- Do not install `nvidia-*` or `cuda-*` Python packages.
- Do not install generic PyPI `torch` wheels.
- If CUDA packages detected, remove them:

```bash
CUDA_PKGS=$(pip list | awk '{print $1}' | grep -Ei '^(nvidia-|cuda-)' || true)
if [[ -n "$CUDA_PKGS" ]]; then
    echo "$CUDA_PKGS" | xargs -r pip uninstall -y
fi
```

---

## Procedure

### Phase 1: Generate Environment Activation Script

#### Step 1 — Detect Working Proxy

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

If empty, ask the user for a proxy URL.

#### Step 2 — Prepare DPCPP Compiler & PTI

`habana-labs.com` is Intel internal — accessible without proxy. Set `no_proxy` accordingly.

```bash
mkdir -p deps && cd deps

wget --no-check-certificate https://artifactory-kfs.habana-labs.com/artifactory/bin-generic-dev-local/DPCPP_JGS/latest/DPCPP_JGS-v0.1.0.tgz
mkdir -p DPCPP_JGS-v0.1.0 && tar -xf DPCPP_JGS-v0.1.0.tgz -C DPCPP_JGS-v0.1.0

wget --no-check-certificate https://artifactory-kfs.habana-labs.com/artifactory/bin-generic-dev-local/PROFILING_TOOLS_JGS/latest/PROFILING_TOOLS_JGS-v0.1.0.tgz
mkdir -p PROFILING_TOOLS_JGS-v0.1.0 && tar -xf PROFILING_TOOLS_JGS-v0.1.0.tgz -C PROFILING_TOOLS_JGS-v0.1.0

cd ..
```

Verify: `ls deps/DPCPP_JGS-v0.1.0/setvars.sh` and `ls deps/PROFILING_TOOLS_JGS-v0.1.0/bin/`

#### Step 3 — Detect GPU Model

```bash
gpu_info="$(xpu-smi discovery 2>/dev/null || lspci | grep -i 'vga\|display\|3d' | grep -i intel || true)"
if echo "$gpu_info" | grep -Eiq 'B580|B60|bmg'; then export GPU_ARCH="bmg"
elif echo "$gpu_info" | grep -Eiq '1550|1100|pvc|Max'; then export GPU_ARCH="pvc"
else export GPU_ARCH=""; echo "WARNING: could not determine GPU_ARCH"; fi
```

| GPU | Architecture |
|-----|-------------|
| Arc B580, B60 | `bmg` |
| Data Center GPU Max 1550, 1100 | `pvc` |

#### Step 4 — Create Conda Environment

```bash
CONDA_ENV_NAME="BMG-MODEL-$(date +%Y%m%d)"
conda create -n $CONDA_ENV_NAME python=3.12 -y
conda run -n $CONDA_ENV_NAME pip install uv
```

#### Step 5 — Generate env.sh

```bash
WORKSPACE_ROOT="$(pwd)"

cat << ENVEOF > env.sh
#!/bin/bash
# Conda env: ${CONDA_ENV_NAME}

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
    echo "ERROR: run with 'source ./env.sh'" >&2; exit 1
fi

export http_proxy="${WORKING_PROXY}"
export https_proxy="${WORKING_PROXY}"
export HTTP_PROXY="${WORKING_PROXY}"
export HTTPS_PROXY="${WORKING_PROXY}"
export no_proxy="localhost,127.0.0.1,intel.com,habana-labs.com"
export NO_PROXY="localhost,127.0.0.1,intel.com,habana-labs.com"

CONDA_BASE=\$(conda info --base 2>/dev/null || echo "\${CONDA_PREFIX:-/mnt/miniforge3}")
if [[ -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "\$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda deactivate 2>/dev/null || true
conda activate ${CONDA_ENV_NAME} || { echo "ERROR: failed to activate ${CONDA_ENV_NAME}"; return 1; }

DPCPP_INSTALL_PATH="${WORKSPACE_ROOT}/deps/DPCPP_JGS-v0.1.0/"
PTI_INSTALL_PATH="${WORKSPACE_ROOT}/deps/PROFILING_TOOLS_JGS-v0.1.0/"
source \${DPCPP_INSTALL_PATH}/setvars.sh --force
export PATH=\${PTI_INSTALL_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=\${PTI_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=\${CONDA_PREFIX:-\$(dirname "\$(which conda)")/../}:\${PTI_INSTALL_PATH}:\${CMAKE_PREFIX_PATH}

export TORCH_XPU_ARCH_LIST="${GPU_ARCH}"
export USE_STATIC_MKL=1
export USE_XCCL=1
export USE_ONEMKL_XPU=0

SCRIPT_DIR=\$(cd \$(dirname "\${BASH_SOURCE[0]:-\$0}"); pwd)
export TORCHINDUCTOR_CACHE_DIR="\${SCRIPT_DIR}/torchinductor_cache"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0

echo "============================================="
echo " Env: ${CONDA_ENV_NAME} | GPU: ${GPU_ARCH}"
echo " DPCPP: \${DPCPP_INSTALL_PATH}"
echo " PTI:   \${PTI_INSTALL_PATH}"
echo "============================================="
ENVEOF
chmod +x env.sh
```

#### Step 6 — Activate and Verify

```bash
source ./env.sh

echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "DPCPP: $(which icpx 2>/dev/null || echo 'NOT FOUND')"
echo "Proxy: $http_proxy"
echo "TORCH_XPU_ARCH_LIST: $TORCH_XPU_ARCH_LIST"
```

All must be set correctly before proceeding.

### Phase 2: Build Packages from Source

#### Step 1 — Build PyTorch (private-gpu)

```bash
if [[ ! -d "frameworks.ai.pytorch.private-gpu" ]]; then
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
fi
cd frameworks.ai.pytorch.private-gpu
git checkout v0.1.0_next && git pull
git submodule sync && git submodule update --init --recursive
uv pip install -r requirements.txt
python -m pip install --no-build-isolation -v -e . 2>&1 | tee log.txt
cd ..
```

Long-running build — poll with `get_terminal_output`.

#### Step 2 — Build intel-xpu-backend-for-triton

```bash
if [[ ! -d "intel-xpu-backend-for-triton" ]]; then
    git clone https://github.com/intel/intel-xpu-backend-for-triton.git
fi
cd intel-xpu-backend-for-triton
git checkout main && git pull
scripts/compile-triton.sh
cd ..
```

#### Step 3 — Build TorchAO

```bash
if [[ ! -d "frameworks.ai.pytorch.torch-ao" ]]; then
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ao
fi
cd frameworks.ai.pytorch.torch-ao
git checkout main && git pull
python setup.py install
cd ..
```

#### Step 4 — Install Runtime Packages

```bash
python -c "import torch; print('torch:', torch.__version__); print('xpu:', hasattr(torch, 'xpu'))"
uv pip install transformers sentencepiece protobuf accelerate diffusers
```

#### Step 5 — Verify Stack

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available()); print('Device:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')"
python -c "import diffusers, transformers, accelerate, sentencepiece, google.protobuf; print('OK')"
```

## Completion Checklist

- [ ] Proxy detected and working
- [ ] DPCPP compiler extracted and setvars.sh sourced
- [ ] PTI profiling tools extracted and on PATH/LD_LIBRARY_PATH
- [ ] GPU arch identified
- [ ] Conda env created
- [ ] env.sh generated and activated
- [ ] PyTorch private-gpu built from source
- [ ] intel-xpu-backend-for-triton built from source
- [ ] TorchAO installed from source
- [ ] Runtime packages installed
- [ ] No `nvidia-*`/`cuda-*` packages in env
- [ ] `torch.xpu.is_available()` returns True
