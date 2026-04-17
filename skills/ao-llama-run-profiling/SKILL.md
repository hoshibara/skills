---
name: ao-llama-run-profiling
description: "Run and profile Llama4 Maverick-17B with MXFP8/MXFP4 quantization on Intel BMG/B60. Covers eager and torch.compile modes, unitrace profiling with compile warmup exclusion, and performance analysis. Use when running Llama4 models, profiling Llama inference, or analyzing Llama4 performance on Intel XPU. Requires bmg-ao-env-setup to be completed first."
argument-hint: "Optional: precision (mxfp8/mxfp4), mode (eager/compile)"
---

# Llama4 MXFP8/MXFP4 Run & Profiling on BMG/B60

Run Llama4 Maverick-17B (reduced) with TorchAO MX quantization (MXFP8/MXFP4) on Intel BMG/B60, profile with unitrace, and analyze kernel-level performance.

**Prerequisite:** `bmg-ao-env-setup` skill must be completed first (env.sh, PyTorch/Triton/TorchAO built).

## When to Use
- Running Llama4 model with MXFP8 or MXFP4 quantization
- Profiling Llama4 inference (eager and compile modes)
- Analyzing Llama4 kernel performance

## Bundled Resources

| Script | Purpose |
|--------|---------|
| `scripts/profile_llama4.py` | Profiling wrapper with warmup + `PTI_ENABLE_COLLECTION` control |
| `scripts/run_llama4_profiling.sh` | Master script: runs eager + compile for Llama4 MXFP8/MXFP4 |

The shared `analyze_unitrace.py` script is in the `bmg-ao-env-setup` skill's `scripts/` directory.

```bash
SKILL_DIR=".agents/skills/ao-llama-run-profiling/scripts"
ENV_SKILL_DIR=".agents/skills/bmg-ao-env-setup/scripts"
cp "${SKILL_DIR}/profile_llama4.py" .
cp "${SKILL_DIR}/run_llama4_profiling.sh" .
cp "${ENV_SKILL_DIR}/analyze_unitrace.py" .
```

---

## CRITICAL: Llama4 Original Scripts Do NOT Call torch.compile

The original `llama4-FP8.py` and `llama4-FP4.py` scripts check `USE_COMPILE` env var but **never actually call `torch.compile()`**. The compile code path only exists in the `profile_llama4.py` wrapper. This means:

- Setting `USE_COMPILE=true` on original scripts has **no effect** on execution
- TORCH_COMPILE_DEBUG runs using original scripts produce **empty debug directories**
- To get Llama4 compile behavior, you MUST use `profile_llama4.py`

---

## Phase 1: Run Llama4 Models

### Step 1 — Enter Llama4 Directory

```bash
cd frameworks.ai.pytorch.gpu-models/presi-models/reduced-llama4
```

### Step 2 — Run MXFP8 (Eager)

```bash
python llama4-FP8.py
```

Config options: `MODEL_CONFIG_NAME=4-Func-FP8` (default, 128 tokens) or `MODEL_CONFIG_NAME=48-Perf-FP8` (1024 tokens).

### Step 3 — Run MXFP4 (Eager)

```bash
python llama4-FP4.py
```

### Step 4 — Run Compile Mode (via wrapper only)

```bash
cd <workspace_root>
USE_COMPILE=true python profile_llama4.py --script llama4-FP8.py --warmup 2 --measure 3
USE_COMPILE=true python profile_llama4.py --script llama4-FP4.py --warmup 2 --measure 3
```

---

## Phase 2: Unitrace Profiling

### CRITICAL: torch.compile Warmup — Exclude Compilation Overhead

Llama4 has **extreme** compile warmup overhead because `torch.compile(model)` compiles the entire model:

| Model | 1st Iteration (compile) | 2nd Iteration (steady) | Ratio |
|-------|------------------------:|----------------------:|------:|
| Llama4 MXFP8 | 12.79s | 0.003s | 4264× |
| Llama4 MXFP4 | 11.09s | 0.003s | 3696× |

The warmup/measurement pattern is identical to FLUX — see the profiling wrapper.

### Step 1 — Profiling Wrapper (profile_llama4.py)

Key design — differs from FLUX in several ways:

```python
# Load model directly from config (no pipeline)
text_config = MODEL_CONFIG.text_config
model = Llama4ForCausalLM._from_config(text_config)
model = model.to(torch.bfloat16).to(DEVICE).eval()

# Replace experts with sequential (required for quantization)
model = replace_experts_with_sequential(model, text_config)

# Quantize ONLY expert linear layers
def _is_expert_linear(mod, fqn):
    return (isinstance(mod, torch.nn.Linear)
            and ".feed_forward.experts." in fqn
            and "shared_expert" not in fqn
            and (fqn.endswith(".gate_proj") or fqn.endswith(".up_proj") or fqn.endswith(".down_proj")))

quantize_(model, config=config, filter_fn=_is_expert_linear)

# CRITICAL: Use torch.compile(model), NOT model.compile()
if COMPILE:
    model = torch.compile(model)
```

Llama4-specific notes:
- Uses `torch.compile(model)` (not `model.compile()`) — reassigns the model variable
- Quantization filter targets only MoE expert linears (`gate_proj`, `up_proj`, `down_proj`)
- Requires `SequentialLlama4TextExperts` replacement for proper weight structure
- Input: batch of tokenized text (not image generation like FLUX)
- `input_ids` must be truncated to `vocab_size` via `torch.remainder`

### Step 2 — Run Profiling

```bash
# Eager: direct unitrace on original scripts
unitrace -d -v -s --chrome-kernel-logging \
    --output-dir-path "${OUTDIR}" -o "${OUTDIR}/llama4_mxfp8_eager.csv" \
    python llama4-FP8.py

# Compile: unitrace --start-paused + wrapper
unitrace -d -v -s --chrome-kernel-logging --start-paused \
    --output-dir-path "${OUTDIR}" -o "${OUTDIR}/llama4_mxfp8_compile.csv" \
    env USE_COMPILE=true python profile_llama4.py \
        --script llama4-FP8.py --warmup 2 --measure 3
```

Note: TORCH_COMPILE_DEBUG for Llama4 requires using the wrapper script, not the originals.

---

## Phase 3: Performance Analysis

### Step 1 — Run Analysis

```bash
python analyze_unitrace.py unitrace_results/<TIMESTAMP>
```

### Step 2 — Key Llama4 Performance Insights

**Eager vs Compile speedup (device time):**
- Llama4 MXFP8: **1.91×** (1.618ms → 846.7μs)
- Llama4 MXFP4: **3.09×** (2.461ms → 796.5μs)

**Compile-mode latency:**
- MXFP8: 4.1 ms/iter average
- MXFP4: 4.3 ms/iter average

**MXFP8 vs MXFP4:**
- Eager: MXFP4 is 1.52× slower (heavy elementwise bit ops)
- Compile: MXFP4 is **0.94× (faster!)** — bitwise ops completely fused away

**Category shifts (Eager → Compile):**

| Category | MXFP8 Eager | MXFP8 Compile | MXFP4 Eager | MXFP4 Compile |
|----------|------------:|--------------:|------------:|--------------:|
| Elementwise | 62.7% | 4.1% | **75.1%** | 4.4% |
| Memory Transfer | 18.9% | — | 13.0% | — |
| GEMM/MatMul | 4.5% | **63.4%** | 2.7% | **58.0%** |
| Triton Kernels | — | 12.1% | — | 16.5% |
| Attention/FMHA | 1.8% | 9.6% | 1.1% | 10.0% |

Key patterns:
- **Elementwise dominated in eager** (63–75%) → drops to ~4% in compile
- **MXFP4 benefits more** from compile (3.09× vs 1.91×) due to FP4 bitwise op fusion
- **Compile-mode MXFP4 actually beats MXFP8** (796.5μs vs 846.7μs) — unique to Llama4
- **Triton kernels appear** in compile: 12–17% of time (fused ops from TorchInductor)
- **Kernel count drops**: 67→39 (MXFP8), 83→41 (MXFP4)

**Llama4 MXFP4 compile kernel names include FP4-specific patterns:**
```
triton_per_fused___lshift_____rshift____scaled_mm_...
triton_poi_fused___rshift____to_copy_add_bitwise_and_...
```
These indicate successful fusion of bitwise pack/unpack with matmul operations.

## Completion Checklist

- [ ] Llama4 MXFP8 eager runs successfully
- [ ] Llama4 MXFP4 eager runs successfully
- [ ] Llama4 MXFP8 compile runs via wrapper
- [ ] Llama4 MXFP4 compile runs via wrapper
- [ ] Unitrace profiling completed (eager + compile with warmup)
- [ ] Results analyzed with analyze_unitrace.py
