---
name: pytorch-upstream-env
description: "Set up Intel XPU Flex Attention test environment. Use when: configuring FA dev env, installing PyTorch+Triton from source for XPU, setting up Intel GPU proxy/oneAPI/conda env, benchmarking flex attention on Intel GPUs (BMG/PVC)."
argument-hint: "Optional: conda env name, GPU arch (bmg/pvc), pytorch branch"
---

# PyTorch Upstream Environment Setup

Automates the full setup of an Intel XPU Flex Attention development and test environment: proxy detection, conda env creation, oneAPI configuration, env script generation, and PyTorch + Triton source build.

## When to Use
- Setting up a new Flex Attention test/dev environment from scratch
- Rebuilding PyTorch with XPU support on a new machine
- Creating a reproducible env script for Intel GPU work

## CRITICAL: Single Terminal Rule — ABSOLUTE, NO EXCEPTIONS

**ALL commands in this skill — from Phase 1 through Phase 2 — MUST execute in the SAME terminal session.** This is non-negotiable:

- Use `run_in_terminal` **ONLY ONCE** to create the terminal at the very start. Record the terminal ID.
- **ALL subsequent commands MUST use `send_to_terminal`** with that same terminal ID. This includes `install_pytorch.sh`, `make triton`, verification commands — everything.
- **NEVER call `run_in_terminal` a second time. NEVER.** Each `run_in_terminal` invocation creates a NEW terminal that does NOT inherit env vars, conda activation, or oneAPI paths from previous terminals. Even if the script sources env.sh internally, opening a second terminal violates this rule.
- If the user specifies an existing terminal to use, use `send_to_terminal` with that terminal — do NOT create a new one.
- For long-running builds (e.g., `make triton`, `pip install -e .`), use `get_terminal_output` to poll for completion instead of setting a timeout.

**Why this matters:** conda activate, oneAPI source, proxy env vars, and build flags are all shell state. A new terminal loses ALL of this. The install script's internal `source env.sh` is NOT a substitute — it may behave differently without the prior shell state (e.g., conda init may not have run in the new shell).

## CRITICAL: Environment Isolation Rules

**Every command that touches Python packages (pip, conda install, make triton, python setup.py, etc.) MUST run inside the activated FA conda environment.** Never run these in `base` or any other env.

1. **At the very beginning of the terminal session**, run `source ./env.sh` BEFORE doing anything else. This applies to build terminals, test terminals, Python REPL sessions, etc.
2. Verify with `echo $CONDA_DEFAULT_ENV` — it must NOT be `base` or empty.
3. The `install_pytorch.sh` script enforces this automatically and will abort if the wrong env is active.

## Procedure

### Phase 1: Generate Environment Activation Script

#### Step 1 — Detect Working Proxy

Run [detect_proxy.sh](./scripts/detect_proxy.sh) to find a working Intel proxy:

```bash
bash .github/skills/flex-attention-env/scripts/detect_proxy.sh
```

Capture the `WORKING_PROXY=` output for the next step. If no proxy works, ask the user to provide one manually.

#### Step 2 — Detect oneAPI Installation

Search for oneAPI installations in **both system and user home** directories, list all versions, and pick the one with the most components (most complete installation):

```bash
# Search all candidate locations
for base in /opt/intel/oneapi "$HOME/intel/oneapi"; do
    # oneAPI uses either setvars.sh (older) or oneapi-vars.sh (newer)
    for script in setvars.sh oneapi-vars.sh; do
        if [[ -f "$base/$script" ]]; then
            echo "Found oneAPI at: $base (via $script)"
            echo "  Components:"
            find "$base" -maxdepth 2 -type d -name "20*" 2>/dev/null | sort
            echo "  Component count: $(find "$base" -maxdepth 2 -type d -name "20*" 2>/dev/null | wc -l)"
        fi
    done
done
```

If multiple installations exist, choose the one with:
1. The **latest version number** (e.g., `2025.10` > `2025.3`)
2. The **most component directories** (most complete)

The `ONEAPI_HOME` should point to the root containing the init script (e.g., `/opt/intel/oneapi`). The init script may be named `setvars.sh` (older releases) or `oneapi-vars.sh` (newer releases). Record the version from component directories for reference.

**Important:** Always use the concrete versioned path, never a `latest` symlink. Verify the init script exists before using it.

#### Step 3 — Detect GPU Model

Determine the Intel GPU model to set the correct architecture target:

```bash
# Check for Intel GPUs
lspci | grep -i "vga\|display\|3d" | grep -i intel
# Or use xpu-smi if available
xpu-smi discovery 2>/dev/null || true
```

Map GPU model to architecture:
| GPU | Architecture |
|-----|-------------|
| Arc B580, B60 | `bmg` |
| Data Center GPU Max 1550, 1100 | `pvc` |

If the GPU model is unclear, **leave architecture empty and warn the user** to set `USE_AOT_DEVLIST` and `TORCH_XPU_ARCH_LIST` manually.

#### Step 4 — Create Conda Environment

Generate a name following the convention `FA-<date>` (e.g., `FA-20260415`) unless the user specifies one:

```bash
CONDA_ENV_NAME="FA-$(date +%Y%m%d)"
conda create -n $CONDA_ENV_NAME python=3.12 -y
conda activate $CONDA_ENV_NAME
pip install uv
```

**Why uv?** `uv` is a fast Python package installer. Once installed, use `uv pip install` instead of `pip install` for all subsequent batch package installations to significantly speed up dependency resolution and downloads.

#### Step 5 — Generate env.sh

Run [generate_env.sh](./scripts/generate_env.sh) with the detected values:

```bash
bash .github/skills/flex-attention-env/scripts/generate_env.sh \
    ./env.sh \
    "$CONDA_ENV_NAME" \
    "$WORKING_PROXY" \
    "$ONEAPI_HOME" \
    "$GPU_ARCH" \
    "$HF_TOKEN"
```

The generated `env.sh` is saved to the workspace root. Review it with the user and ask if any adjustments are needed (e.g., GPU frequency pinning, ZE_AFFINITY_MASK).

#### Step 6 — Activate and Verify

Source env.sh and verify **all critical environment variables** are set (not just conda):

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
- `CONDA_DEFAULT_ENV` is the FA env name (not `base`)
- `USE_XPU` is `1`
- `icpx` (oneAPI compiler) is on PATH
- `http_proxy` is set
- `TORCH_XPU_ARCH_LIST` is set (e.g., `pvc`)

If any check fails, env.sh was not sourced correctly.

### Phase 2: Install PyTorch + Triton

#### Alternative: Install Pre-built Nightly Wheels (user-initiated only)

**Do NOT use this path by default.** Only when the user explicitly requests installing PyTorch from nightly wheels (instead of building from source), use this path.

**CRITICAL: oneAPI must NOT be active when using pip-installed PyTorch.** The nightly wheels ship their own bundled SYCL runtime; sourcing oneAPI introduces conflicting libraries and will cause runtime errors. Before installing, comment out the oneAPI block in `env.sh`:

```bash
# Comment out the oneAPI section in env.sh
sed -i '/^# ===== oneAPI =====/,/^fi$/s/^/#/' ./env.sh
```

This comments out everything from `# ===== oneAPI =====` through the closing `fi`. After this change, `source ./env.sh` will no longer activate oneAPI. **Do not re-enable oneAPI in this environment — it is permanently incompatible with pip-installed PyTorch.**

Then re-source env.sh (without oneAPI) and install:

```bash
source ./env.sh
uv pip install --pre -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

This is much faster than building from source and is suitable for quick testing or when source builds are not needed. After installation, skip directly to **Step 2 — Verify Installation**.

#### Step 0 — Hard Reset PyTorch (optional, user-initiated only)

**Do NOT run these commands automatically.** Only execute when the user explicitly requests a full reset. This is useful when switching oneAPI versions, as stale CMakeCache.txt can point to the wrong SYCL/compiler paths.

```bash
cd pytorch
git reset --hard HEAD
git clean -ffdx
git submodule deinit -f .
cd ..
```

**When the user might request this:** CMakeCache.txt persists paths (e.g., `SYCL_COMPILER`, `CMAKE_PREFIX_PATH`) from a previous build's oneAPI. Even with the correct `CMPLR_ROOT` env var, CMake reuses cached paths and finds the wrong SYCL, causing `SYCL_CMPLR_TEST` compile failures. A lighter alternative is to just remove `build/CMakeCache.txt` and `build/CMakeFiles/`.

#### Step 1 — Run the Install Script

Run [install_pytorch.sh](./scripts/install_pytorch.sh). The script will **automatically source `env.sh`**, hard-reset the repo if `--clean` is passed, and verify the correct conda env is active before proceeding. If env.sh is missing or the env is `base`, it will abort.

```bash
# Clean build (recommended when switching oneAPI versions):
bash .github/skills/flex-attention-env/scripts/install_pytorch.sh --clean

# Incremental build (only if same oneAPI version):
bash .github/skills/flex-attention-env/scripts/install_pytorch.sh
```

This will:
1. **Source env.sh and verify conda env** (abort if base/wrong env)
2. Clone pytorch and add the hoshibara remote
3. Install conda build deps (mkl, cmake, ninja, gcc)
4. Clean previous torch/triton installs (uses `uv pip uninstall` when available)
5. Sync and update submodules
6. Build Triton (`make triton`)
7. Build and install PyTorch in editable mode

**Note:** The build can take 30+ minutes. If the user needs a specific branch (e.g., from hoshibara), they should checkout before running the install:

```bash
cd pytorch
git checkout hoshibara/<branch-name>
cd ..
bash .github/skills/flex-attention-env/scripts/install_pytorch.sh
```

#### Step 2 — Verify Installation

**Important:** Use a single-line `python -c` command to avoid indentation errors when executing via `send_to_terminal`:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU: {torch.xpu.is_available()}'); print(f'Devices: {torch.xpu.device_count()}') if torch.xpu.is_available() else None; print(f'GPU: {torch.xpu.get_device_name(0)}') if torch.xpu.is_available() else None; x=torch.rand(3,device='xpu'); print(f'rand(3) on XPU: {x}')"
```

## Completion Checklist

- [ ] Proxy detected and working
- [ ] oneAPI found and sourced
- [ ] GPU arch identified (or user warned)
- [ ] Conda env created and activated
- [ ] env.sh generated and saved to workspace root
- [ ] PyTorch built from source with XPU support
- [ ] Triton built
- [ ] `torch.xpu.is_available()` returns True
