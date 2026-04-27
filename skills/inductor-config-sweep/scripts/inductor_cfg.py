"""Single-file source of truth for Inductor config sweep presets.

Imported (or exec'd) at the very start of every profile script. The active
preset is selected by the `INDUCTOR_CFG_PRESET` environment variable.

Three tiers of presets are bundled:
  1. baseline                 — all defaults (control)
  2. single-knob (8 presets)  — flip exactly one config off baseline; for
                                 attribution / per-knob sensitivity
  3. combinations             — pairwise / triplet / all-on; for finding
                                 super-additive or cancelling interactions

You can also build presets on the fly via env vars:
  INDUCTOR_CFG_KNOBS="combo_kernels=True,realize_reads_threshold=16"
  INDUCTOR_CFG_PRESET=custom

Knob recommendation source: see the fusion-analyzer-use-debug-output skill's
fusion_opportunity_analyzer.py output (CONFIG fix tier).
"""
import os


# ── building blocks: each "knob" is a single (attr, value) ─────────────
KNOBS: dict[str, tuple[str, object]] = {
    "combo":       ("combo_kernels", True),
    "bench_combo": ("benchmark_combo_kernel", True),
    "epi_first":   ("epilogue_fusion_first", True),
    "prologue":    ("prologue_fusion", True),
    "rr16":        ("realize_reads_threshold", 16),
    "expand_dim":  ("expand_dimension_for_pointwise_nodes", True),
    "aggressive":  ("aggressive_fusion", True),
    "epi":         ("epilogue_fusion", True),  # already default-True; for explicit listing
}


def _build(*knob_names: str) -> list[tuple[str, object]]:
    return [KNOBS[k] for k in knob_names]


# ── PRESETS ────────────────────────────────────────────────────────────
PRESETS: dict[str, list[tuple[str, object]]] = {
    # tier 1 — control
    "baseline": [],

    # tier 2 — single-knob sensitivity (8)
    "combo_kernels":   _build("combo"),
    "benchmark_combo": _build("combo", "bench_combo"),
    "epilogue_first":  _build("epi_first"),
    "prologue_fusion": _build("prologue"),
    "realize_reads":   _build("rr16"),
    "expand_dim":      _build("expand_dim"),
    "aggressive":      _build("aggressive"),

    # tier 3a — pairwise combinations of the typical "winners"
    # (combo_kernels and realize_reads usually carry; combine each with the rest)
    "combo+rr":        _build("combo", "rr16"),
    "combo+epi_first": _build("combo", "epi_first"),
    "combo+prologue":  _build("combo", "prologue"),
    "combo+expand":    _build("combo", "expand_dim"),
    "combo+aggr":      _build("combo", "aggressive"),
    "rr+epi_first":    _build("rr16", "epi_first"),
    "rr+prologue":     _build("rr16", "prologue"),
    "rr+expand":       _build("rr16", "expand_dim"),
    "rr+aggr":         _build("rr16", "aggressive"),
    "prologue+expand": _build("prologue", "expand_dim"),
    "epi_first+aggr":  _build("epi_first", "aggressive"),

    # tier 3b — triplets of the consistently best knobs
    "combo+rr+epi_first": _build("combo", "rr16", "epi_first"),
    "combo+rr+prologue":  _build("combo", "rr16", "prologue"),
    "combo+rr+expand":    _build("combo", "rr16", "expand_dim"),
    "combo+rr+bench":     _build("combo", "rr16", "bench_combo"),

    # tier 3c — quadruplet (top-4)
    "combo+rr+epi_first+expand":
        _build("combo", "rr16", "epi_first", "expand_dim"),

    # tier 3d — all-on
    "all": _build("combo", "bench_combo", "epi", "epi_first",
                  "prologue", "rr16", "expand_dim", "aggressive"),

    # tier 3e — "all minus one" ablation (to find harmful members of `all`)
    "all-no_combo":     _build("bench_combo", "epi", "epi_first",
                                "prologue", "rr16", "expand_dim", "aggressive"),
    "all-no_prologue":  _build("combo", "bench_combo", "epi", "epi_first",
                                "rr16", "expand_dim", "aggressive"),
    "all-no_aggressive": _build("combo", "bench_combo", "epi", "epi_first",
                                 "prologue", "rr16", "expand_dim"),
    "all-no_expand":    _build("combo", "bench_combo", "epi", "epi_first",
                                "prologue", "rr16", "aggressive"),
}


# Convenience preset groups (use as the PRESETS env var in run_one.sh /
# sweep_all.sh; values must be space-separated preset names).
PRESET_GROUPS = {
    "tier1":      ["baseline"],
    "tier2":      ["baseline", "combo_kernels", "benchmark_combo",
                   "epilogue_first", "prologue_fusion", "realize_reads",
                   "expand_dim", "aggressive"],
    "tier3_pair": ["combo+rr", "combo+epi_first", "combo+prologue",
                   "combo+expand", "combo+aggr",
                   "rr+epi_first", "rr+prologue", "rr+expand", "rr+aggr",
                   "prologue+expand", "epi_first+aggr"],
    "tier3_trip": ["combo+rr+epi_first", "combo+rr+prologue",
                   "combo+rr+expand", "combo+rr+bench"],
    "tier3_abl":  ["all", "all-no_combo", "all-no_prologue",
                   "all-no_aggressive", "all-no_expand"],
    # the canonical "fast" group: control + all single-knob + all-on
    "default":    ["baseline", "combo_kernels", "benchmark_combo",
                   "epilogue_first", "prologue_fusion", "realize_reads",
                   "expand_dim", "aggressive", "all"],
    # the "thorough" group: everything above
    "full":       None,  # filled at end of file = list(PRESETS)
}


def _parse_custom_knobs(env_val: str) -> list[tuple[str, object]]:
    """Parse `INDUCTOR_CFG_KNOBS=key1=val1,key2=val2` env into knob list."""
    out: list[tuple[str, object]] = []
    for part in env_val.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"INDUCTOR_CFG_KNOBS entry missing '=': {part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # type coerce: True/False/int/float else str
        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out.append((k, v))
    return out


def apply_preset(name: str | None = None) -> tuple[str, list[tuple[str, object]]]:
    """Apply the preset selected by `name` (or env). Returns (name, applied)."""
    import torch._inductor.config as cfg  # lazy: lets PRESET_GROUPS be queried without torch
    name = name or os.environ.get("INDUCTOR_CFG_PRESET", "baseline")

    # 1. custom preset built from env
    if name == "custom":
        env_val = os.environ.get("INDUCTOR_CFG_KNOBS", "")
        knob_list = _parse_custom_knobs(env_val)
    elif name in PRESETS:
        knob_list = PRESETS[name]
    else:
        raise ValueError(f"unknown INDUCTOR_CFG_PRESET={name!r}; "
                         f"available: {sorted(PRESETS)} or 'custom'")

    applied: list[tuple[str, object]] = []
    for attr, value in knob_list:
        if not hasattr(cfg, attr):
            print(f"  [inductor_cfg] WARNING: torch._inductor.config has no "
                  f"attribute '{attr}', skipping", flush=True)
            continue
        setattr(cfg, attr, value)
        applied.append((attr, value))
    print(f"  [inductor_cfg] preset='{name}' applied {len(applied)} settings: "
          f"{applied}", flush=True)
    return name, applied


PRESET_GROUPS["full"] = list(PRESETS)


# Auto-apply on import — but only when torch is importable (i.e. inside a
# profile script). When sweep_all.sh queries PRESET_GROUPS without env.sh
# sourced, this block must not crash.
try:
    import torch  # noqa: F401
    apply_preset()
except ImportError:
    pass
