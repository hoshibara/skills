---
name: inductor-config-sweep
description: "Systematically sweep torch._inductor.config knobs (fusion / combo_kernels / realize_reads / etc.) across one or more torch.compile'd models on Intel BMG/B60 with unitrace profiling. Provides a 3-tier preset library (baseline / single-knob / combinations & ablations), per-run isolation, two-channel (wall + device) measurement, and an aggregated Markdown summary. Use when you want to (a) quantify the speedup of an Inductor config change, (b) find the best knob combination for a model, or (c) build a publishable performance regression matrix. Requires bmg-ao-env-setup."
argument-hint: "Optional: model list, preset group (default|tier2|tier3_pair|tier3_trip|tier3_abl|full)"
---

# Inductor Config Sweep on BMG/B60

Run a controlled grid sweep `models × inductor-config presets`, profile each
combo with unitrace, and aggregate results into a single Markdown summary.

**Prerequisites:**
- `bmg-ao-env-setup` skill completed (env.sh sources DPC++ / oneAPI / conda /
  unitrace; PyTorch+Triton+TorchAO built from source).
- A working `profile_<model>.py` script that:
  1. `import inductor_cfg` *before* any `torch.compile` call
  2. Has a 3-phase BUILD / WARMUP / MEASURE structure with PTI gating
  3. Prints `avg latency : X ms` to stdout
  Three reference impls are bundled — see `scripts/profile_examples/`.

---

## When to use

- Quantify the speedup of an `Inductor` config change (e.g. `combo_kernels=True`)
- Find the best **combination** of knobs for a specific model (super-additive
  / cancelling effects only show up in pairwise/triplet sweeps)
- Build a regression matrix (model × config) for tracking across PyTorch /
  TorchAO versions
- Investigate fusion-analyzer recommendations end-to-end (see also
  `fusion-analyzer-use-debug-output` skill — that skill *recommends* knobs;
  this skill *measures* them)

---

## Bundled resources

```
scripts/
├── inductor_cfg.py              # 26+ preset library, 3-tier groups, env-driven
├── run_one.sh                   # single-run wrapper: env + unitrace + python
├── sweep_all.sh                 # MODELS × PRESETS grid runner
├── summarize.py                 # aggregate console.log + unitrace CSV → Markdown
├── analyze_unitrace_csv.py      # helper: per-kernel breakdown from one CSV
└── profile_examples/
    ├── profile_flux.py          # FluxPipeline reference (self-contained)
    ├── profile_llama31.py       # Llama-3.1 generate(1 token) reference
    └── profile_llama4.py        # Llama-4 single forward (delegates via runpy)
```

---

## Preset library (`scripts/inductor_cfg.py`)

Three tiers. Select via `INDUCTOR_CFG_PRESET=<name>` per run, or via
`PRESET_GROUP=<group>` to run a whole group in one sweep.

### Tier 1 — control
| Preset | Knobs |
|---|---|
| `baseline` | (none — Inductor defaults) |

### Tier 2 — single-knob sensitivity (8)
| Preset | Knob flipped |
|---|---|
| `combo_kernels` | `combo_kernels=True` |
| `benchmark_combo` | `combo_kernels` + `benchmark_combo_kernel=True` |
| `epilogue_first` | `epilogue_fusion_first=True` |
| `prologue_fusion` | `prologue_fusion=True` |
| `realize_reads` | `realize_reads_threshold=16` |
| `expand_dim` | `expand_dimension_for_pointwise_nodes=True` |
| `aggressive` | `aggressive_fusion=True` |

### Tier 3a — pairwise combinations (11)
`combo+rr` `combo+epi_first` `combo+prologue` `combo+expand` `combo+aggr`
`rr+epi_first` `rr+prologue` `rr+expand` `rr+aggr` `prologue+expand`
`epi_first+aggr`

### Tier 3b — triplets (4)
`combo+rr+epi_first` `combo+rr+prologue` `combo+rr+expand` `combo+rr+bench`

### Tier 3c — quadruplet (1)
`combo+rr+epi_first+expand`

### Tier 3d — all-on (1)
`all` — every recommended knob simultaneously

### Tier 3e — leave-one-out ablation (4)
`all-no_combo` `all-no_prologue` `all-no_aggressive` `all-no_expand`
— pinpoints which member of `all` is helping vs hurting

### Preset groups (collections you can request as a unit)
| Group | Contains | Run count per model |
|---|---|---:|
| `tier1` | baseline only | 1 |
| `tier2` | baseline + 7 single-knob | 8 |
| `tier3_pair` | the 11 pairwise | 11 |
| `tier3_trip` | the 4 triplets | 4 |
| `tier3_abl` | all + 4 ablations | 5 |
| **`default`** | tier2 + `all` (the canonical fast sweep) | 9 |
| `full` | every preset (~26) | 26 |

### Custom (build your own at runtime)
```bash
INDUCTOR_CFG_PRESET=custom \
INDUCTOR_CFG_KNOBS="combo_kernels=True,realize_reads_threshold=8,prologue_fusion=True" \
  bash scripts/run_one.sh llama31 mxfp8 custom
```

---

## How to use this skill

### 1. Stage the harness into your workspace
```bash
SKILL_DIR=".agents/skills/inductor-config-sweep/scripts"
WS="<your_sweep_workspace>"               # e.g. unitrace_results/<date>_my_sweep
mkdir -p "$WS"/{scripts,tmp,results,logs}
cp -r "$SKILL_DIR"/{inductor_cfg.py,run_one.sh,sweep_all.sh,summarize.py,analyze_unitrace_csv.py} \
       "$WS/scripts/"
```

### 2. Provide a `profile_<model>.py` per model

Required interface contract:
- Module `inductor_cfg` imported **before** any `torch.compile` (it auto-applies
  the preset on import via `INDUCTOR_CFG_PRESET`)
- 3-phase structure (skip BUILD if dataset/model loading is cheap):
  ```python
  # Phase A: BUILD
  model = build_model(); quantize_(model, mx_cfg); model = torch.compile(model, ...)

  # Phase B: WARMUP — PTI off
  for _ in range(args.warmup):
      model(**inputs); torch.xpu.synchronize()

  # Phase C: MEASURE — PTI on
  os.environ["PTI_ENABLE_COLLECTION"] = "1"
  for i in range(args.measure):
      t0 = time.time(); model(**inputs); torch.xpu.synchronize()
      times.append(time.time() - t0)
  os.environ["PTI_ENABLE_COLLECTION"] = "0"

  print(f"avg latency : {1000*sum(times)/len(times):.1f} ms")  # parsed by summarize.py
  ```
- Argparse: `--warmup`, `--measure`, `--precision {mxfp8,mxfp4}` (others optional)

Three working examples are bundled — see `scripts/profile_examples/`. Copy and
adapt one of them for a new model.

### 3. Wire the model name into `run_one.sh`

Edit the per-model branch in `run_one.sh` to point at your `profile_<model>.py`:
```bash
case "${MODEL}" in
    llama31) PROFILE="profile_llama31.py"; WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}" ;;
    llama4)  PROFILE="profile_llama4.py";  WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}" ;;
    flux)    PROFILE="profile_flux.py";    WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}" ;;
    YOURS)   PROFILE="profile_yours.py";   WARMUP_L="${WARMUP:-3}"; MEASURE_L="${MEASURE:-20}" ;;
esac
```
**Tip on WARMUP/MEASURE**: dynamic shapes + autotune may bleed compilation /
autotuning into the first measure iter. Use **WARMUP ≥ 3** for `dynamic=True`
models. **MEASURE ≥ 20** is recommended unless per-iter time is > 100 ms; the
default sweep used `3` for llama31/flux and that yielded ±3-5% noise (too high
for differentiating sub-2% knob effects).

### 4. Run the sweep
```bash
cd "$WS"

# fastest: 9 presets (the default group)
bash scripts/sweep_all.sh

# more thorough: include all combinations & ablations
PRESET_GROUP=full bash scripts/sweep_all.sh

# only pairwise of one model
MODELS="llama31_mxfp8" PRESET_GROUP=tier3_pair bash scripts/sweep_all.sh

# explicit override
PRESETS="baseline combo+rr combo+rr+expand all" \
  bash scripts/sweep_all.sh
```

### 5. Read the aggregated table
`results/SWEEP_SUMMARY.md` — one table per `(model, precision)` with rows for
every preset that produced a result, sorted with known presets first.

---

## Two-channel measurement (important!)

Each run produces **two independent timing signals** with **different units**:

| Signal | Source | Captures | Unit per row |
|---|---|---|---|
| `latency (ms)` | **Pass A** clean run (no unitrace) — Python `time.time() + torch.xpu.synchronize()` | host overhead + device + sync, **without PTI hook overhead** | per-iter (already averaged) |
| `kernel/iter (ms)` | **Pass B** unitrace CSV `Total Device Time for L0 backend (ns)` ÷ MEASURE | only L0 kernels in MEASURE window | per-iter |

**Why two passes?** unitrace's PTI hooks add per-kernel-launch host overhead
(us-scale per call). On host-bound paths (e.g. Llama generate loop with
hundreds of small kernels), this can inflate measured wall-clock latency by
**30–60%**. To get a trustworthy latency number we run each preset twice:
- **Pass A** — no unitrace, just `python profile_*.py …`. Captures clean
  `avg latency : X ms`. Output: `console_clean.log`.
- **Pass B** — `unitrace --start-paused … python profile_*.py …`. Captures
  the kernel CSV. Output: `console.log` + `<run>.csv`.

`summarize.py` prefers `console_clean.log` for the latency column; if
missing (legacy single-pass runs), it falls back to `console.log` with a
note in the per-section header.

**Δ vs baseline columns are valid in both directions** because the baseline
shares the same MEASURE.

**Common patterns**
- latency Δ ≈ kernel Δ ≈ same sign and ≥ 5%: real win
- latency Δ ≈ 0 but kernel Δ much better: host-bound — preset is helping the
  GPU but Python/dispatch is the bottleneck
- kernel Δ better but latency Δ worse: preset added host overhead (recompile,
  extra dispatch); investigate before adopting
- `n_kernels` ↓ + kernel ↓: real fusion happened (e.g. `combo_kernels` on llama)
- `n_kernels` flat + kernel ↓: per-kernel improvement (e.g. `realize_reads`)

---

## Per-run isolation

Every run gets:
- Fresh Python process (no dynamo / autotune cache leak)
- **Fresh `TORCHINDUCTOR_CACHE_DIR` per (model, preset)**, wiped before each
  run (`rm -rf` then re-create). Also wipes default fallback locations
  (`~/.cache/torch_inductor*`, `/tmp/torchinductor_$USER`). Guarantees every
  measurement starts from a cold compile so previous presets / re-runs
  cannot leak compiled kernels, autotune choices, or fusion artefacts.
- Per-sweep `TMPDIR=<workspace>/tmp` (kept out of `/tmp`)
- `unitrace --start-paused` + `PTI_ENABLE_COLLECTION` two-stage gating so
  warmup (compile + autotune) is **never** in the device time

---

## Best practices

- **Always run `baseline` first** in any new workspace — it anchors all Δ values
- **Run an ablation tier** (`PRESET_GROUP=tier3_abl`) before recommending `all`;
  some knobs cancel each other on certain models (e.g. on FLUX `prologue_fusion`
  *hurts*, so `all` is worse than `realize_reads` alone)
- **Increase MEASURE** when per-iter time is < 5 ms; otherwise CSV's L0 record
  may have only a few rows and statistics get noisy
- For models with KV-cache / dynamic shapes, **WARMUP ≥ 3** to absorb per-shape
  autotune
- **Don't horizontally compare** wall-clock latency across different model
  workloads (e.g. llama4 single-forward vs llama31 generate-1-token aren't
  apples-to-apples)
- After the sweep, feed the *winning* preset back to
  `fusion-analyzer-use-debug-output` to verify the fusion-graph change matches
  expectations

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `0 rows` in unitrace CSV | MEASURE window too short (< ~30 ms total) | bump `MEASURE=20+` |
| First measure 50% slower than others | Per-shape autotune leaked from warmup | bump `WARMUP=3+` |
| `unknown INDUCTOR_CFG_PRESET=...` | Typo or stale preset name | `python -c 'import inductor_cfg; print(list(inductor_cfg.PRESETS))'` |
| Latency goes up with `all` but down with single knobs | A member of `all` is harmful for this model | run `PRESET_GROUP=tier3_abl` |
| `n_kernels` column inflated | unitrace counted multiple launches of same kernel | this column is a coarse proxy; trust kernel-time Δ instead |
| `set -u: unbound variable` from env.sh | env.sh has unbound vars | `run_one.sh` uses `set -eo pipefail` only — keep that |

---

## Reference

Full methodology document with phase-by-phase timing diagrams, host-gap
estimation, and known limitations is in
`reproducer/METHODOLOGY.md` of any sweep workspace generated by this skill,
or in the source workspace at
`unitrace_results/20260424_1840_inductor_config_sweep/reproducer/METHODOLOGY.md`.
