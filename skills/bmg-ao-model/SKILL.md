---
name: bmg-ao-model
description: "Set up Intel BMG/B60 environment, run FLUX/Llama4 MXFP8/MXFP4 with DPCPP_JGS compiler + PTI, local repo builds, unitrace profiling with warmup, and performance analysis."
argument-hint: "Optional: conda env name, GPU arch"
---

# Intel BMG AO Model Setup & Profiling (FLUX/Llama4 on B60)

Automates end-to-end setup for FLUX and Llama4 on Intel BMG/B60: proxy detection, DPCPP/PTI compiler setup, conda env creation, env.sh generation, local source builds (PyTorch/Triton/TorchAO), MXFP8/MXFP4 runs, unitrace profiling with compile-mode warmup, and performance analysis.

## When to Use
- Setting up a new FLUX/Llama4 CRI runtime on Intel BMG/B60
- Creating a reproducible conda + DPCPP_JGS environment for BMG model runs
- Building the validated PyTorch/Triton/TorchAO stack required by FLUX/Llama4 MXFP8/MXFP4
- Profiling model inference with unitrace (eager and compile modes)
- Analyzing kernel-level performance (device time, category breakdown, fusion analysis)

## Bundled Resources

Profiling scripts are bundled in this skill's `scripts/` directory. Copy them to the workspace root before use:

```bash
SKILL_SCRIPTS=".agents/skills/bmg-ao-model/scripts"
cp "${SKILL_SCRIPTS}/profile_flux.py" .
cp "${SKILL_SCRIPTS}/profile_llama4.py" .
cp "${SKILL_SCRIPTS}/run_unitrace_all.sh" .
cp "${SKILL_SCRIPTS}/analyze_unitrace.py" .
```

| Script | Purpose |
|--------|---------|
| `profile_flux.py` | FLUX profiling wrapper with warmup + `PTI_ENABLE_COLLECTION` control |
| `profile_llama4.py` | Llama4 profiling wrapper with warmup + `PTI_ENABLE_COLLECTION` control |
| `run_unitrace_all.sh` | Master script: runs all models × modes (eager/compile/debug) |
| `analyze_unitrace.py` | Parses unitrace CSV output, produces category breakdown and comparative summary |

These scripts may need adaptation for new models or changed script paths. The key patterns (warmup exclusion via `PTI_ENABLE_COLLECTION`, `--start-paused`, TORCH_COMPILE_DEBUG) are reusable.

## CRITICAL: Single Terminal Rule — ABSOLUTE, NO EXCEPTIONS

**ALL commands in this skill — from Phase 1 through Phase 3 — MUST execute in the SAME terminal session.** This is non-negotiable:

- Use `run_in_terminal` **ONLY ONCE** to create the terminal at the very start. Record the terminal ID.
- **ALL subsequent commands MUST use `send_to_terminal`** with that same terminal ID. This includes build, install, and run commands.
- **NEVER call `run_in_terminal` a second time. NEVER.** Each `run_in_terminal` invocation creates a NEW terminal that does NOT inherit env vars, conda activation, or compiler paths from previous terminals. Even if the script sources env.sh internally, opening a second terminal violates this rule.
- If the user specifies an existing terminal to use, use `send_to_terminal` with that terminal — do NOT create a new one.
- For long-running builds (e.g., `make triton`, `pip install -e .`), use `get_terminal_output` to poll for completion instead of setting a timeout.

**Why this matters:** conda activate, DPCPP source, proxy env vars, and build flags are all shell state. A new terminal loses ALL of this. Sourcing `env.sh` later in another script is NOT a substitute for maintaining a single terminal session.

## CRITICAL: Long-Running Command Output — DO NOT PIPE TO tail

**For long-running commands (builds, installs, compile scripts), NEVER pipe output through `| tail`, `| head`, or any filter that hides the full log.** This makes it impossible to diagnose failures.

- **FORBIDDEN:** `python -m pip install -e . 2>&1 | tail -20` — hides build errors
- **FORBIDDEN:** `scripts/compile-triton.sh | tail` — hides compilation progress
- **CORRECT:** Redirect output to a log file, then read the log file to check status:

```bash
# Redirect to a log file
python -m pip install -e . 2>&1 | tee /tmp/build_pytorch.log

# Check progress or errors by reading the log file
tail -50 /tmp/build_pytorch.log
```

**Why this matters:** Build commands can take 30+ minutes. If the output is piped through `tail` directly, the full log is lost and errors mid-build become invisible. Writing to a log file preserves the complete output for troubleshooting while still allowing `tail` on the file to check the latest status.

## CRITICAL: Environment Activation — MANDATORY FIRST STEP, NO EXCEPTIONS

**`source ./env.sh` MUST be the VERY FIRST command in EVERY terminal session, BEFORE any other operation.** This is non-negotiable and applies to ALL phases (build, install, test, run).

### Activation Rules

1. **ALWAYS activate first.** Before `cd`, `git`, `pip`, `python`, `cmake`, `scripts/compile-triton.sh`, or ANY other command — run `source ./env.sh` first. No exceptions.
2. **`source ./env.sh` MUST be executed ALONE as a standalone command.**
   - **FORBIDDEN:** `source ./env.sh && pip install ...` — NEVER chain with `&&`
   - **FORBIDDEN:** `source ./env.sh; python setup.py install` — NEVER chain with `;`
   - **FORBIDDEN:** `cd dir && source ../env.sh` — NEVER chain before it either
   - **FORBIDDEN:** `(source ./env.sh)` — NEVER in a subshell
   - **CORRECT:** Send `source ./env.sh` alone, wait for it to complete, then send the next command separately.
3. **For agent execution with `send_to_terminal`:** Send ONLY `source ./env.sh`, wait for prompt return, THEN send the next command as a separate `send_to_terminal` call.
4. **Verify after activation:** Run `echo $CONDA_DEFAULT_ENV` — it must NOT be `base` or empty. If it is, env.sh was not sourced correctly. Do NOT proceed.
5. **Environment activation is allowed only via `source ./env.sh`.** Do not manually run `conda activate <env>`.
6. **If you `cd` to a subdirectory**, use the relative path back to workspace root: `source ../env.sh` or `source ../../env.sh` etc. The env.sh path must be correct for the current working directory.

### Why This Matters

`env.sh` sets up conda env, DPCPP compiler, PTI paths, proxy, and build flags. Without it, builds will silently use the wrong Python (e.g., base conda Python 3.13 instead of the target env Python 3.12), wrong compiler, or missing env vars — causing cryptic build failures.

## CRITICAL: XPU-Only Package Policy

- Target platform is Intel XPU only (`xpu`), not CUDA.
- Do not install `nvidia-*` or `cuda-*` Python packages.
- Do not install generic PyPI `torch` wheels in this skill path.
- If a prebuilt torch is needed for troubleshooting, use XPU wheels only:

```bash
uv pip install --pre -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

- If CUDA packages are detected, remove them before continuing:

```bash
CUDA_PKGS=$(pip list | awk '{print $1}' | grep -Ei '^(nvidia-|cuda-)' || true)
if [[ -n "$CUDA_PKGS" ]]; then
    echo "$CUDA_PKGS" | xargs -r pip uninstall -y
    echo "ERROR: CUDA packages were detected and removed. Re-run installation in a clean XPU env." >&2
    exit 1
fi
```

## Local Source Build with Auto-Clone

**PyTorch, Triton, and TorchAO are built from source repos in the workspace.** If a repo directory already exists locally, use it directly (no re-clone). If a directory is missing, `git clone` it from the network.

| Component | Local Path | Clone URL | Branch |
|-----------|-----------|-----------|--------|
| PyTorch (private-gpu) | `frameworks.ai.pytorch.private-gpu/` | `https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu` | `v0.1.0_next` |
| TorchAO | `frameworks.ai.pytorch.torch-ao/` | `https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ao` | `main` |
| Triton (intel-xpu-backend) | `intel-xpu-backend-for-triton/` | `https://github.com/intel/intel-xpu-backend-for-triton.git` | `main` |
| GPU Models (FLUX scripts) | `frameworks.ai.pytorch.gpu-models/` | `https://github.com/intel-innersource/frameworks.ai.pytorch.gpu-models` | *(default)* |

Clone pattern (use for each repo if directory is absent):

```bash
if [[ ! -d "<local-path>" ]]; then
    git clone <clone-url>
fi
```

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

#### Step 2 — Prepare DPCPP Compiler & PTI

The DPCPP compiler and PTI profiling tools are distributed as tarballs. They should be placed under the `deps/` directory in the workspace root.

**Download (if not already present):**

Note: `habana-labs.com` is an Intel internal host accessible via direct connection (no proxy). Set `no_proxy` to include `habana-labs.com` or unset proxy env vars before downloading.

```bash
mkdir -p deps && cd deps

# Download DPCPP compiler
wget --no-check-certificate https://artifactory-kfs.habana-labs.com/artifactory/bin-generic-dev-local/DPCPP_JGS/latest/DPCPP_JGS-v0.1.0.tgz
mkdir -p DPCPP_JGS-v0.1.0
tar -xf DPCPP_JGS-v0.1.0.tgz -C DPCPP_JGS-v0.1.0

# Download PTI profiling tools
wget --no-check-certificate https://artifactory-kfs.habana-labs.com/artifactory/bin-generic-dev-local/PROFILING_TOOLS_JGS/latest/PROFILING_TOOLS_JGS-v0.1.0.tgz
mkdir -p PROFILING_TOOLS_JGS-v0.1.0
tar -xf PROFILING_TOOLS_JGS-v0.1.0.tgz -C PROFILING_TOOLS_JGS-v0.1.0

cd ..
```

**Verify extraction:**

```bash
ls deps/DPCPP_JGS-v0.1.0/setvars.sh       # Must exist
ls deps/PROFILING_TOOLS_JGS-v0.1.0/bin/    # Must exist
ls deps/PROFILING_TOOLS_JGS-v0.1.0/lib/    # Must exist
```

Expected layout:
```
deps/
├── DPCPP_JGS-v0.1.0/
│   ├── setvars.sh          # Compiler environment init
│   ├── compiler/
│   └── ...
└── PROFILING_TOOLS_JGS-v0.1.0/
    ├── bin/                 # unitrace, etc.
    ├── lib/                 # libpti*.so, etc.
    └── include/
```

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

If GPU detection is unclear, default to `bmg` for this skill or ask the user.

#### Step 4 — Create Conda Environment

Generate a name following the convention `BMG-MODEL-<date>` (e.g., `BMG-MODEL-20260417`) unless the user specifies one:

```bash
CONDA_ENV_NAME="BMG-MODEL-$(date +%Y%m%d)"
conda create -n $CONDA_ENV_NAME python=3.12 -y
conda run -n $CONDA_ENV_NAME pip install uv
```

Do not activate with `conda activate` here. Activation must be done later by `source ./env.sh` only.

**Why uv?** `uv` is a fast Python package installer. Once installed, use `uv pip install` instead of `pip install` for all subsequent batch package installations to significantly speed up dependency resolution and downloads.

#### Step 5 — Generate env.sh

Generate a local `env.sh` using the detected values. The env.sh sources the DPCPP compiler (not system oneAPI) and sets PTI paths:

```bash
WORKSPACE_ROOT="$(pwd)"

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
export no_proxy="localhost,127.0.0.1,intel.com,habana-labs.com"
export NO_PROXY="localhost,127.0.0.1,intel.com,habana-labs.com"

# Optional HF token passthrough
# export HUGGING_FACE_HUB_TOKEN="<your-token>"

# Initialize conda
CONDA_BASE=\$(conda info --base 2>/dev/null || echo "\${CONDA_PREFIX:-/mnt/miniforge3}")
if [[ -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "\$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda deactivate 2>/dev/null || true
conda activate ${CONDA_ENV_NAME} || { echo "ERROR: failed to activate ${CONDA_ENV_NAME}"; return 1; }

# Source DPCPP compiler & set PTI paths
DPCPP_INSTALL_PATH="${WORKSPACE_ROOT}/deps/DPCPP_JGS-v0.1.0/"
PTI_INSTALL_PATH="${WORKSPACE_ROOT}/deps/PROFILING_TOOLS_JGS-v0.1.0/"

source \${DPCPP_INSTALL_PATH}/setvars.sh --force
_profiling_tools_root=\${PTI_INSTALL_PATH}
export PATH=\${_profiling_tools_root}/bin:\$PATH
export LD_LIBRARY_PATH=\${_profiling_tools_root}/lib:\$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=\${CONDA_PREFIX:-\$(dirname "\$(which conda)")/../}:\${_profiling_tools_root}:\${CMAKE_PREFIX_PATH}

# XPU build/runtime settings
export TORCH_XPU_ARCH_LIST="${GPU_ARCH}"
export USE_STATIC_MKL=1
export USE_XCCL=1
export USE_ONEMKL_XPU=0

# TorchInductor cache
SCRIPT_PATH="\${BASH_SOURCE[0]:-\$0}"
SCRIPT_DIR=\$(cd \$(dirname "\$SCRIPT_PATH"); pwd)
export TORCHINDUCTOR_CACHE_DIR="\${SCRIPT_DIR}/torchinductor_cache"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0

# Paths
export FLUX_ROOT="\$PWD/frameworks.ai.pytorch.gpu-models/presi-models/CRI/reduced-flux"
export LLAMA_ROOT="\$PWD/frameworks.ai.pytorch.gpu-models/TorchBench-Llama"

echo "============================================="
echo " Successfully Activated Env: ${CONDA_ENV_NAME}"
echo " DPCPP: \${DPCPP_INSTALL_PATH}"
echo " PTI:   \${PTI_INSTALL_PATH}"
echo " GPU_ARCH: ${GPU_ARCH}"
echo "============================================="
ENVEOF

chmod +x env.sh
```

The generated `env.sh` is saved to the workspace root. Review it with the user and ask if any adjustments are needed (e.g., GPU frequency pinning, ZE_AFFINITY_MASK, HF token).

#### Step 6 — Activate and Verify

Source env.sh and verify **all critical environment variables** are set (not just conda):

Use this strict pattern:
- Valid: run `source ./env.sh` alone, then run verification commands.
- Invalid: `source ./env.sh && <other_command>`.

```bash
source ./env.sh

# Verify all key env vars are set
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "DPCPP sourced: $(which icpx 2>/dev/null || echo 'NOT FOUND')"
echo "Proxy: $http_proxy"
echo "TORCH_XPU_ARCH_LIST: $TORCH_XPU_ARCH_LIST"
echo "USE_STATIC_MKL: $USE_STATIC_MKL"
echo "USE_XCCL: $USE_XCCL"
```

All of the following must be true:
- `CONDA_DEFAULT_ENV` is the BMG env name (not `base`)
- `icpx` (DPCPP compiler) is on PATH
- `http_proxy` is set
- `TORCH_XPU_ARCH_LIST` is set (e.g., `bmg`)

If any check fails, env.sh was not sourced correctly.

### Phase 2: Build Packages from Source

For each repo: if the directory exists locally, use it; otherwise clone it first.

#### Step 1 — Build and Install PyTorch (private-gpu)

```bash
if [[ ! -d "frameworks.ai.pytorch.private-gpu" ]]; then
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu
fi
cd frameworks.ai.pytorch.private-gpu
git checkout v0.1.0_next
git pull
git submodule sync
git submodule update --init --recursive
uv pip install -r requirements.txt
python -m pip install --no-build-isolation -v -e . 2>&1 | tee log.txt
cd ..
```

Key notes:
- Uses editable install (`-e .`) with `--no-build-isolation` for development workflow.
- `git pull` ensures the branch tip is current.
- Build output is logged to `log.txt` inside the repo directory.
- This is a long-running build. Use `get_terminal_output` to poll for completion.

#### Step 2 — Build and Install intel-xpu-backend-for-triton

```bash
if [[ ! -d "intel-xpu-backend-for-triton" ]]; then
    git clone https://github.com/intel/intel-xpu-backend-for-triton.git
fi
cd intel-xpu-backend-for-triton
git checkout main
git pull
scripts/compile-triton.sh
cd ..
```

Key notes:
- Uses `main` branch (latest).
- `compile-triton.sh` handles the full build and install.

#### Step 3 — Build and Install TorchAO

```bash
if [[ ! -d "frameworks.ai.pytorch.torch-ao" ]]; then
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ao
fi
cd frameworks.ai.pytorch.torch-ao
git checkout main
git pull
python setup.py install
cd ..
```

Key notes:
- Uses `main` branch (latest).

#### Step 4 — Install FLUX Runtime Packages

These are small packages that can be fetched from PyPI:

```bash
# Ensure Step 1/2/3 already installed XPU torch stack before runtime package install.
python -c "import torch; print('torch:', torch.__version__); print('xpu attr:', hasattr(torch, 'xpu'))"

uv pip install transformers sentencepiece protobuf accelerate diffusers

CUDA_PKGS=$(pip list | awk '{print $1}' | grep -Ei '^(nvidia-|cuda-)' || true)
if [[ -n "$CUDA_PKGS" ]]; then
    echo "ERROR: CUDA packages are not allowed in this XPU skill:" >&2
    echo "$CUDA_PKGS" >&2
    exit 1
fi
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

#### Step 4 — Run Llama4 Models

```bash
cd ../../../presi-models/reduced-llama4

# Llama4 MXFP8 (eager)
python llama4-FP8.py

# Llama4 MXFP4 (eager)
python llama4-FP4.py
```

Note: Llama4 original scripts do NOT support `USE_COMPILE=true` — the flag is checked but `torch.compile()` is never called. Use `profile_llama4.py` wrapper for compile mode.

#### Step 5 — Run Full Model Wrapper (Optional)

```bash
./run.sh --device xpu --mode eager
./run.sh --device xpu --mode compile --use_profiler
```

#### Step 5 — Hugging Face Access Check

FLUX models are gated. If model download fails, request access and login:

```bash
huggingface-cli login
```

### Phase 4: Unitrace Profiling

#### CRITICAL: torch.compile Warmup — Exclude Compilation Overhead

**`torch.compile` is lazy.** The first inference triggers JIT compilation, which is 50-4000x slower than steady-state. Profiling must exclude this warmup:

| Model | 1st Iteration (compile) | 2nd Iteration (steady) | Ratio |
|-------|------------------------:|----------------------:|------:|
| FLUX MXFP8 | 2.48s | 0.05s | 50x |
| FLUX MXFP4 | 2.79s | 0.04s | 65x |
| Llama4 MXFP8 | 12.79s | 0.003s | 4264x |
| Llama4 MXFP4 | 11.09s | 0.003s | 3696x |

**Solution:** Use `--start-paused` with unitrace + `PTI_ENABLE_COLLECTION` env var in a profiling wrapper:

1. unitrace launches with `--start-paused` (no data collection initially)
2. Wrapper script runs N warmup iterations (compilation happens here, not traced)
3. Wrapper sets `os.environ["PTI_ENABLE_COLLECTION"] = "1"` to start collection
4. Wrapper runs M measured iterations (only these are traced)
5. Wrapper sets `os.environ["PTI_ENABLE_COLLECTION"] = "0"` to stop collection

This is the ONLY correct way to profile compile mode. Without this, compile-mode device time includes compilation overhead and will appear much slower or identical to eager.

#### CRITICAL: Llama4 Original Scripts Do NOT Call torch.compile

The original `llama4-FP8.py` and `llama4-FP4.py` scripts check the `USE_COMPILE` env var but **never actually call `torch.compile()`**. The compile code path only exists in the `profile_llama4.py` wrapper, which explicitly does `model = torch.compile(model)`. This means:

- TORCH_COMPILE_DEBUG runs using original scripts will produce **empty debug directories** for Llama4
- To get Llama4 compile behavior, you MUST use the profiling wrapper (`profile_llama4.py`)
- FLUX scripts (`flux_dev_mxfp8.py`, `flux_dev_mxfp4.py`) DO have a working compile path via `COMPILE=true`

#### Step 1 — Profiling Wrapper Scripts

Create separate wrapper scripts for each model family that implement the warmup pattern:

**profile_flux.py** — Key design:
```python
# Load model → compile (lazy wrap) → quantize → warmup → enable PTI → measure → disable PTI
pipe = FluxPipeline.from_pretrained(...)
if COMPILE:
    pipe.transformer.compile(fullgraph=True)  # lazy, no compilation yet
quantize_(pipe.transformer, config=config, filter_fn=_filter)

# Warmup (compilation happens here, not profiled)
for i in range(args.warmup):
    _ = do_inference(pipe, prompt)
    torch.xpu.synchronize()

# Enable profiling for measurement only
os.environ["PTI_ENABLE_COLLECTION"] = "1"
for i in range(args.measure):
    _ = do_inference(pipe, prompt)
    torch.xpu.synchronize()
os.environ["PTI_ENABLE_COLLECTION"] = "0"
```

**profile_llama4.py** — Key differences from FLUX:
```python
# For Llama4 compile, use torch.compile(model) NOT model.compile()
model = Llama4ForCausalLM._from_config(text_config)
model = model.to(torch.bfloat16).to(DEVICE).eval()
model = replace_experts_with_sequential(model, text_config)
quantize_(model, config=config, filter_fn=_is_expert_linear)
if COMPILE:
    model = torch.compile(model)  # NOT model.compile()
```

#### Step 2 — Unitrace Options

```bash
UNITRACE_OPTS="-d -v -s --chrome-kernel-logging"
```

| Flag | Purpose |
|------|---------|
| `-d` | Device timing (L0 kernel execution time) |
| `-v` | Verbose mode (kernel shapes) |
| `-s` | Submission timing |
| `--chrome-kernel-logging` | Chrome trace output for visualization |
| `--start-paused` | For compile mode: start with collection paused |
| `--output-dir-path DIR` | Output directory |
| `-o FILE.csv` | CSV output path |

#### Step 3 — Run Script Structure

The master profiling script (`run_unitrace_all.sh`) uses three function types:

1. **`run_profile_eager()`** — Direct unitrace on original scripts (no warmup needed for eager)
2. **`run_profile_compile()`** — unitrace with `--start-paused` + profiling wrapper
3. **`run_compile_debug()`** — `TORCH_COMPILE_DEBUG=1` runs (no unitrace, for fusion analysis only)

Total runs per model config: 3 (eager + compile + debug) × number of models

#### Step 4 — TORCH_COMPILE_DEBUG for Fusion Analysis

```bash
TORCH_COMPILE_DEBUG=1 \
TORCH_COMPILE_DEBUG_DIR="${outdir}/torch_compile_debug" \
COMPILE=true python flux_dev_mxfp8.py
```

This generates `output_code.py` files containing all fused Triton kernels. Analyze them for:
- Total Triton kernel count (by prefix: `triton_poi_` = pointwise, `triton_per_` = persistent reduction, `triton_red_` = reduction)
- Extern kernel calls count (`extern_kernels.`)
- Fused patterns (`_scaled_mm` = quantized matmul, `norm` = fused normalization, `lshift/rshift/bitwise` = FP4 pack/unpack)

### Phase 5: Performance Analysis

#### Step 1 — analyze_unitrace.py

Parse unitrace CSV output with `analyze_unitrace.py`:

```bash
python analyze_unitrace.py unitrace_results/<TIMESTAMP>
```

The script produces:
- **Per-run analysis**: Top-15 kernels by device time, category breakdown
- **Comparative summary**: Eager vs compile speedup, MXFP8 vs MXFP4 comparison

#### Step 2 — Key Analysis Insights (Lessons Learned)

**Eager vs Compile: What changes**
- **Memory transfers disappear**: Compile mode keeps tensors on device; eager has ~40% time in MemCopy(M2D)
- **Elementwise ops fuse into Triton kernels**: Eager FLUX MXFP4 is 58% elementwise → compile drops to 14%
- **GEMM becomes dominant**: Ideal GPU utilization — compute-bound, not memory/overhead-bound
- **Kernel count drops significantly**: FLUX MXFP8 goes from 505 → 182 unique kernels (64% reduction)

**MXFP4 vs MXFP8 behavior:**
- MXFP4 is 1.5-1.75x slower than MXFP8 in eager mode (heavy bitwise pack/unpack operations)
- Compile mode fuses bitwise ops into Triton kernels, nearly closing the gap (1.13x for FLUX, 0.94x for Llama4)
- MXFP4 generates more Triton kernels than MXFP8 (66 vs 54 for FLUX) due to FP4 bit manipulation

**Category classification for Intel XPU kernels:**
- `gemm_kernel`, `_scaled_mm` → GEMM/MatMul
- `MemCopy(M2D/D2M/D2D)` → Memory Transfer
- `gen_conv`, `conv_reorder` → Other (convolution reformat)
- `micro_sdpa` → Attention/FMHA
- `triton_poi_`, `triton_per_`, `triton_red_` → Triton Kernels (fused ops)
- `__sycl_kernel` with norm context → Reduce/Norm

**GPU Utilization appears low** (~0.6-3.5%) because the reduced/functional models are tiny. This is expected — the metric is meaningful only for full-size models with Perf configs.

## Completion Checklist

- [ ] Proxy detected and working
- [ ] DPCPP compiler extracted and setvars.sh sourced
- [ ] PTI profiling tools extracted and on PATH/LD_LIBRARY_PATH
- [ ] GPU arch identified (or user warned)
- [ ] Conda env created
- [ ] Environment activated only via `source ./env.sh`
- [ ] env.sh generated and saved to workspace root
- [ ] PyTorch private-gpu built from local source (editable install)
- [ ] intel-xpu-backend-for-triton built from local source
- [ ] TorchAO installed from local source
- [ ] FLUX runtime packages installed
- [ ] No `nvidia-*` / `cuda-*` Python packages remain in the env
- [ ] `python flux_dev_mxfp8.py` runs successfully
- [ ] Profiling wrappers created (profile_flux.py, profile_llama4.py) with warmup + PTI_ENABLE_COLLECTION
- [ ] Unitrace profiling run completed (eager + compile with warmup + TORCH_COMPILE_DEBUG)
- [ ] Results analyzed with analyze_unitrace.py
