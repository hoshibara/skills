---
name: ao-flux-run-profiling
description: "Run and profile FLUX.1-dev with MXFP8/MXFP4 quantization on Intel BMG/B60. Covers eager and torch.compile modes, unitrace profiling with compile warmup exclusion, TORCH_COMPILE_DEBUG fusion analysis, and performance analysis. Use when running FLUX models, profiling FLUX inference, or analyzing FLUX performance on Intel XPU. Requires bmg-ao-env-setup to be completed first."
argument-hint: "Optional: precision (mxfp8/mxfp4), mode (eager/compile)"
---

# FLUX MXFP8/MXFP4 Run & Profiling on BMG/B60

Run FLUX.1-dev with TorchAO MX quantization (MXFP8/MXFP4) on Intel BMG/B60, profile with unitrace, and analyze kernel-level performance.

**Prerequisite:** `bmg-ao-env-setup` skill must be completed first (env.sh, PyTorch/Triton/TorchAO built).

## When to Use
- Running FLUX model with MXFP8 or MXFP4 quantization
- Profiling FLUX inference (eager and compile modes)
- Analyzing FLUX kernel performance and torch.compile fusion patterns

## Bundled Resources

| Script | Purpose |
|--------|---------|
| `scripts/profile_flux.py` | Profiling wrapper with warmup + `PTI_ENABLE_COLLECTION` control |
| `scripts/run_flux_profiling.sh` | Master script: runs eager + compile + debug for FLUX MXFP8/MXFP4 |

The shared `analyze_unitrace.py` script is in the `bmg-ao-env-setup` skill's `scripts/` directory.

```bash
# Copy scripts to workspace
SKILL_DIR=".agents/skills/ao-flux-run-profiling/scripts"
ENV_SKILL_DIR=".agents/skills/bmg-ao-env-setup/scripts"
cp "${SKILL_DIR}/profile_flux.py" .
cp "${SKILL_DIR}/run_flux_profiling.sh" .
cp "${ENV_SKILL_DIR}/analyze_unitrace.py" .
```

---

## Phase 1: Run FLUX Models

### Step 1 — Enter FLUX Directory

```bash
cd frameworks.ai.pytorch.gpu-models/presi-models/CRI/reduced-flux
```

### Step 2 — Run MXFP8

```bash
# Functional check (128×128, eager)
python flux_dev_mxfp8.py

# Performance run (1024×1024, eager)
MODEL_CONFIG_NAME=48-Perf python flux_dev_mxfp8.py

# Compile mode
COMPILE=true python flux_dev_mxfp8.py

# Performance + compile
MODEL_CONFIG_NAME=48-Perf COMPILE=true python flux_dev_mxfp8.py
```

### Step 3 — Run MXFP4

```bash
python flux_dev_mxfp4.py
MODEL_CONFIG_NAME=48-Perf python flux_dev_mxfp4.py
COMPILE=true python flux_dev_mxfp4.py
MODEL_CONFIG_NAME=48-Perf COMPILE=true python flux_dev_mxfp4.py
```

### Step 4 — Hugging Face Access

FLUX models are gated. If download fails: `huggingface-cli login`

---

## Phase 2: Unitrace Profiling

### CRITICAL: torch.compile Warmup — Exclude Compilation Overhead

**`torch.compile` is lazy.** The first inference triggers JIT compilation, which is 50–65× slower than steady-state for FLUX:

| Model | 1st Iteration (compile) | 2nd Iteration (steady) | Ratio |
|-------|------------------------:|----------------------:|------:|
| FLUX MXFP8 | 2.48s | 0.05s | 50× |
| FLUX MXFP4 | 2.79s | 0.04s | 65× |

**Solution:** `--start-paused` with unitrace + `PTI_ENABLE_COLLECTION` in a profiling wrapper:

1. unitrace starts with `--start-paused` (no collection)
2. Wrapper runs N warmup iterations (compilation happens here, not traced)
3. Wrapper sets `os.environ["PTI_ENABLE_COLLECTION"] = "1"` → collection starts
4. Wrapper runs M measured iterations (only these are traced)
5. Wrapper sets `os.environ["PTI_ENABLE_COLLECTION"] = "0"` → collection stops

### Step 1 — Profiling Wrapper (profile_flux.py)

Key design pattern:
```python
# Load model → compile (lazy) → quantize → warmup → enable PTI → measure → disable PTI
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.transformer.transformer_blocks = nn.Sequential(pipe.transformer.transformer_blocks[0])
pipe.transformer.single_transformer_blocks = nn.Sequential(pipe.transformer.single_transformer_blocks[0])
pipe.text_encoder_2.encoder.block = nn.ModuleList([pipe.text_encoder_2.encoder.block[0]])
pipe.to(DEVICE)

if COMPILE:
    pipe.transformer.compile(fullgraph=True)  # lazy, no compilation yet

config = MXDynamicActivationMXWeightConfig(
    activation_dtype=act_dtype, weight_dtype=wt_dtype,
    kernel_preference=KernelPreference.AUTO,
)
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

FLUX-specific notes:
- Uses `pipe.transformer.compile(fullgraph=True)` (not `torch.compile(pipe)`)
- Model is reduced: single transformer block, single encoder block
- MXFP8 uses `torch.float8_e4m3fn`, MXFP4 uses `torch.float4_e2m1fn_x2`
- Quantization filter: all `nn.Linear` except `lm_head`

### Step 2 — Unitrace Options

```bash
UNITRACE_OPTS="-d -v -s --chrome-kernel-logging"
```

| Flag | Purpose |
|------|---------|
| `-d` | Device timing (L0 kernel execution time) |
| `-v` | Verbose mode (kernel shapes) |
| `-s` | Submission timing |
| `--chrome-kernel-logging` | Chrome trace output |
| `--start-paused` | For compile mode: start paused |

### Step 3 — Run Profiling

Three run types per precision (MXFP8 + MXFP4 = 6 runs total):

```bash
# Eager: direct unitrace on original script
unitrace -d -v -s --chrome-kernel-logging \
    --output-dir-path "${OUTDIR}" -o "${OUTDIR}/flux_mxfp8_eager.csv" \
    python flux_dev_mxfp8.py

# Compile: unitrace --start-paused + wrapper
unitrace -d -v -s --chrome-kernel-logging --start-paused \
    --output-dir-path "${OUTDIR}" -o "${OUTDIR}/flux_mxfp8_compile.csv" \
    env COMPILE=true python profile_flux.py \
        --script flux_dev_mxfp8.py --warmup 2 --measure 3

# TORCH_COMPILE_DEBUG: no unitrace, for fusion analysis
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="${OUTDIR}/torch_compile_debug" \
    COMPILE=true python flux_dev_mxfp8.py
```

### Step 4 — TORCH_COMPILE_DEBUG Fusion Analysis

The debug run generates `output_code.py` with all fused Triton kernels. Analyze:

```python
# Count Triton kernel types
grep -c '^def triton_poi_' output_code.py  # pointwise
grep -c '^def triton_per_' output_code.py  # persistent reduction
grep -c '^def triton_red_' output_code.py  # reduction
grep -c 'extern_kernels\.' output_code.py  # extern calls

# FLUX-specific fusion patterns
grep '^def triton_' output_code.py | grep -c '_scaled_mm'   # quantized matmul fusion
grep '^def triton_' output_code.py | grep -c 'norm'          # normalization fusion
grep '^def triton_' output_code.py | grep -c 'lshift\|rshift\|bitwise'  # FP4 pack/unpack (MXFP4 only)
```

MXFP4 generates ~12 more Triton kernels than MXFP8 (66 vs 54) due to FP4 bitwise pack/unpack operations.

---

## Phase 3: Performance Analysis

### Step 1 — Run Analysis

```bash
python analyze_unitrace.py unitrace_results/<TIMESTAMP>
```

### Step 2 — Key FLUX Performance Insights

**Eager vs Compile speedup (device time):**
- FLUX MXFP8: **2.72×** (206ms → 76ms)
- FLUX MXFP4: **4.22×** (361ms → 85ms)

**Wall-clock throughput:**
- Eager: ~5–6 it/s → Compile: ~22–24 it/s (**4–5× real-world speedup**)

**Memory reduction:**
- MXFP8: 2.79 GB → 1.74 GB (38% reduction)
- MXFP4: 3.82 GB → 1.49 GB (61% reduction)

**Category shifts (Eager → Compile):**

| Category | MXFP8 Eager | MXFP8 Compile | MXFP4 Eager | MXFP4 Compile |
|----------|------------:|--------------:|------------:|--------------:|
| Memory Transfer | 40.1% | 0.08% | 23.0% | 0.08% |
| Elementwise | 28.8% | 16.2% | **58.3%** | 14.3% |
| GEMM/MatMul | 6.9% | **58.8%** | 4.8% | **62.6%** |
| Reduce/Norm | 21.4% | 2.1% | 12.3% | 1.9% |

Key patterns:
- **Memory transfers eliminated**: Compile keeps tensors on device
- **Elementwise ops fused**: MXFP4 eager is 58% elementwise (bit pack/unpack) → compile drops to 14%
- **GEMM becomes dominant**: 58–63% in compile = ideal compute-bound profile
- **MXFP4 benefits more** from compile (4.22× vs 2.72×) because FP4 bitwise ops get fused

**MXFP8 vs MXFP4:**
- Eager: MXFP4 is 1.75× slower (heavy elementwise)
- Compile: gap closes to 1.13× (bitwise ops fused away)
- MXFP4 + compile is viable: lower bit-width with minimal performance penalty

**Kernel count reduction:** 505 → 182 (MXFP8), 748 → 193 (MXFP4)

**Top kernels in compile mode:** `gemm_kernel` alone accounts for ~48% of device time → further optimization should target GEMM.

## Completion Checklist

- [ ] FLUX MXFP8 eager runs successfully
- [ ] FLUX MXFP4 eager runs successfully
- [ ] FLUX MXFP8 compile runs successfully
- [ ] FLUX MXFP4 compile runs successfully
- [ ] Unitrace profiling completed (eager + compile with warmup)
- [ ] TORCH_COMPILE_DEBUG fusion analysis completed
- [ ] Results analyzed with analyze_unitrace.py
