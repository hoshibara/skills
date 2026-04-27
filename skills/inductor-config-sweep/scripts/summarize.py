"""Aggregate per-(model, preset) results into one Markdown table.

Reads:
  results/<model>_<precision>/<preset>/console.log         — script-printed
                                                              avg latency line
  results/<model>_<precision>/<preset>/<name>.csv          — unitrace device
                                                              timing CSV

Writes a Markdown report with two tables per model:
  1. wall-clock latency (script's own measure-loop avg) per preset
  2. unitrace total kernel time per preset
plus per-model "% vs baseline" columns.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Ensure the scripts dir (which contains inductor_cfg.py) is importable so
# the appendix can render the live preset / knob definitions.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PRESETS_ORDER = ["baseline", "combo_kernels", "benchmark_combo",
                 "epilogue_first", "prologue_fusion", "realize_reads",
                 "expand_dim", "aggressive", "all"]


def order_presets(found: list[str]) -> list[str]:
    """Return found presets in canonical order (known ones first, rest sorted)."""
    known = [p for p in PRESETS_ORDER if p in found]
    extra = sorted(p for p in found if p not in PRESETS_ORDER)
    return known + extra


def parse_latency(console_log: Path) -> float | None:
    if not console_log.exists():
        return None
    txt = console_log.read_text(errors="ignore")
    m = re.search(r"avg latency\s*:\s*([\d.]+)\s*ms", txt)
    if m:
        return float(m.group(1))
    # Fallback: parse `[measure i/N] X.XXXs` lines (llama4 raw output)
    measures = re.findall(r"\[measure\s+\d+/\d+\]\s+([\d.]+)s", txt)
    if measures:
        vals = [float(v) for v in measures]
        return sum(vals) / len(vals) * 1000  # s → ms
    return None


def parse_unitrace_csv(csv_dir: Path) -> tuple[float, int] | tuple[None, None]:
    """Pick the largest <name>.<pid>.csv in csv_dir and parse it.

    Returns (total_kernel_us, n_distinct_kernels).
    """
    csvs = [p for p in csv_dir.glob("*.csv")
            if p.name.startswith(csv_dir.parent.name.split('_')[0])  # skip ldconfig
            or "ldconfig" not in p.name]
    csvs = [p for p in csv_dir.glob("*.csv") if "ldconfig" not in p.name]
    if not csvs:
        return None, None
    main = max(csvs, key=lambda p: p.stat().st_size)
    txt = main.read_text(errors="ignore")
    m = re.search(r"Total Device Time for L0 backend \(ns\):\s*([\d,]+)", txt)
    total_us = float(m.group(1).replace(",", "")) / 1000 if m else None
    # n_kernels = lines that look like CSV rows after header
    nk = sum(1 for ln in txt.splitlines()
             if re.search(r',\s*\d+,\s*\d+,', ln) and "Calls" not in ln)
    return total_us, nk


def parse_run_meta(console_log: Path, combo: str) -> dict:
    """Extract per-run hyperparams from console.log header.

    `combo` is e.g. 'llama31_mxfp8' / 'llama4_mxfp4' / 'flux_mxfp8'.
    Returns dict of human-readable strings (missing keys are skipped from output).
    """
    if not console_log.exists():
        return {}
    txt = console_log.read_text(errors="ignore")
    model = combo.split("_", 1)[0]
    meta: dict[str, object] = {}

    if model == "llama31":
        m = re.search(r"config=(\S+)\s*\|\s*input_len=(\d+)\s*\|\s*batch=(\d+)", txt)
        if m:
            meta.update(config=m.group(1), input_len=int(m.group(2)),
                        batch=int(m.group(3)))
        m = re.search(r"warmup=(\d+)\s+measure=(\d+)\s+max_new_tokens=(\d+)", txt)
        if m:
            meta.update(warmup=int(m.group(1)), measure=int(m.group(2)),
                        max_new_tokens=int(m.group(3)))
        m = re.search(r"model=(\S+)", txt)
        if m: meta["model_name"] = m.group(1)

    elif model == "llama4":
        m = re.search(r"config=(\S+)", txt)
        if m: meta["config"] = m.group(1)
        for k, pat in (("batch",         r"Batch size:\s*(\d+)"),
                       ("input_len",     r"Input length:\s*(\d+)"),
                       ("max_new_tokens", r"Max new tokens:\s*(\d+)")):
            m = re.search(pat, txt)
            if m: meta[k] = int(m.group(1))
        # vendor script reports the actual values used inside runpy
        m = re.search(r"Warmup:\s*(\d+)\s*iters\s*\|\s*Measure:\s*(\d+)\s*iters", txt)
        if m:
            meta.update(warmup=int(m.group(1)), measure=int(m.group(2)))

    elif model == "flux":
        # Match both old (no config=) and new (with config=) banner formats
        m = re.search(r"H=W=(\d+)", txt)
        if m: meta["HW"] = int(m.group(1))
        m = re.search(r"FLUX\s+\S+\s*\|.*?warmup=(\d+)\s+measure=(\d+)", txt)
        if m:
            meta.update(warmup=int(m.group(1)), measure=int(m.group(2)))
        m = re.search(r"steps=(\d+)", txt)
        if m: meta["steps"] = int(m.group(1))
        m = re.search(r"config=(\S+)", txt)
        if m:
            meta["config"] = m.group(1)
        elif "HW" in meta:
            # Reverse-lookup config from HW: profile_flux.py uses
            # INPUT_ID_LENGTH_DICT[config_name] → HW. The default '4-Func' maps
            # to 128 in the bundled script.
            meta["config"] = "4-Func"  # best-effort fallback for legacy logs

    return meta


def fmt_meta(meta: dict) -> str:
    """One-line markdown rendering of run hyperparams."""
    if not meta:
        return "_(no run metadata recovered)_"
    order = ["model_name", "config", "input_len", "HW", "batch",
             "max_new_tokens", "steps", "warmup", "measure"]
    parts = []
    for k in order:
        if k in meta:
            parts.append(f"`{k}={meta[k]}`")
    # any unknown extras
    for k, v in meta.items():
        if k not in order:
            parts.append(f"`{k}={v}`")
    return " · ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R = Path(args.results)
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    metas: dict[str, dict] = {}
    for combo_dir in sorted(R.glob("*_mxfp*")):
        if not combo_dir.is_dir(): continue
        for preset_dir in sorted(combo_dir.iterdir()):
            if not preset_dir.is_dir(): continue
            preset = preset_dir.name
            csvs = list(preset_dir.glob("*.csv"))
            # Prefer Pass-A clean latency (no unitrace overhead); fall back
            # to Pass-B unitrace-instrumented log for legacy single-pass runs.
            clean_log = preset_dir / "console_clean.log"
            lat_ms = (parse_latency(clean_log)
                      if clean_log.exists()
                      else parse_latency(preset_dir / "console.log"))
            lat_source = "clean" if clean_log.exists() else "unitrace"
            kt_us, n_k = parse_unitrace_csv(preset_dir)
            rows[combo_dir.name][preset] = {
                "latency_ms": lat_ms, "kernel_us": kt_us, "n_kernels": n_k,
                "lat_source": lat_source,
            }
        # Snapshot run metadata from baseline (or first available preset)
        baseline_log = combo_dir / "baseline" / "console.log"
        if not baseline_log.exists():
            for preset_dir in sorted(combo_dir.iterdir()):
                cand = preset_dir / "console.log"
                if cand.exists():
                    baseline_log = cand
                    break
        metas[combo_dir.name] = parse_run_meta(baseline_log, combo_dir.name)

    md: list[str] = [
        "# Inductor-config sweep summary\n",
        "## How values are computed\n",
        "Both columns are **per-iter wall-clock-equivalent (ms)** so they can "
        "be compared on the same scale.\n",
        "- **`latency (ms)`** — host wall-clock per iter, captured from a "
        "**clean run without unitrace** (Pass A in `run_one.sh`). Avoids the "
        "PTI hook overhead unitrace adds per kernel launch (us-scale, "
        "noticeable on host-bound paths). From the profile script's "
        "`[measure i/N] X.X ms` loop, computed as "
        "`mean(time.time()_after - time.time()_before)` with a "
        "`torch.xpu.synchronize()` before stopping the timer. Includes host "
        "dispatch / Python / Dynamo + device exec + sync wait. "
        "If `console_clean.log` is missing (legacy single-pass runs), falls "
        "back to the unitrace pass — see the per-section note.\n",
        "- **`kernel/iter (ms)`** — device-only kernel time per iter, from "
        "**Pass B** (workload re-run under unitrace). Reads "
        "`Total Device Time for L0 backend (ns)` from the unitrace CSV "
        "summary, **divided by `measure`** (number of measured iters). "
        "Counts only L0-backend kernel exec time inside the PTI-on window; "
        "does not include host overhead, kernel-launch gaps, or H2D/D2H "
        "transfers when those are not L0 kernels.\n",
        "- **`Δ vs base`** — `(value − baseline_value) / baseline_value × 100%`. "
        "Same `measure` is used in baseline and preset runs, so this Δ is "
        "robust to the per-iter normalisation.\n",
        "- **`n kernels`** — coarse proxy: number of CSV data rows. Multiple "
        "launches of the same kernel are counted separately.\n",
        "\n*Latency includes host overhead; kernel/iter is pure device time. "
        "`latency − kernel/iter` ≈ host gap (Python / dispatch / sync stalls).*\n",
    ]

    # ── Executive summary (TL;DR) ────────────────────────────────────
    md.append("\n## TL;DR — best preset per model")
    md.append("Latency = clean Pass A (no unitrace).  "
              "Δ is vs each model's `baseline`.  "
              "Verdict: **real win** = ≥ 5% latency improvement.\n")
    tldr = ["| Model | Baseline lat | Best preset | Δ lat | Δ kernel | Verdict |",
            "|---|---:|---|---:|---:|---|"]
    verdicts: list = []
    for combo, presets in rows.items():
        base = presets.get("baseline")
        if not base or base["latency_ms"] is None:
            continue
        b_lat = base["latency_ms"]
        measure = metas.get(combo, {}).get("measure") or 1
        b_kt_ms = (base["kernel_us"] / 1000.0 / measure) if base["kernel_us"] else None
        scored = [(p, d) for p, d in presets.items()
                  if d["latency_ms"] is not None]
        if not scored:
            continue
        best_p, best_d = min(scored, key=lambda kv: kv[1]["latency_ms"])
        d_lat = (best_d["latency_ms"] - b_lat) / b_lat * 100
        best_kt_ms = (best_d["kernel_us"] / 1000.0 / measure) if best_d["kernel_us"] else None
        d_kt = ((best_kt_ms - b_kt_ms) / b_kt_ms * 100
                if (b_kt_ms and best_kt_ms) else float("nan"))
        if d_lat <= -5:
            tag = "**real win**"
        elif d_lat >= 5:
            tag = "regression risk"
        else:
            tag = "noise (≤ 5%)"
        tldr.append(f"| `{combo}` | {b_lat:.2f} ms | `{best_p}` | "
                    f"{d_lat:+.1f}% | {d_kt:+.1f}% | {tag} |")
        top3 = sorted(scored, key=lambda kv: kv[1]["latency_ms"])[:3]
        worst_p, worst_d = max(scored, key=lambda kv: kv[1]["latency_ms"])
        d_worst = (worst_d["latency_ms"] - b_lat) / b_lat * 100
        host_gap = ((b_lat - b_kt_ms) / b_lat * 100) if b_kt_ms else 0
        verdicts.append((combo, b_lat, b_kt_ms, host_gap, top3,
                         worst_p, d_worst, measure))
    md.append("\n".join(tldr) + "\n")

    # ── Per-model narrative conclusions ──────────────────────────────
    md.append("\n## Conclusions")
    for combo, b_lat, b_kt_ms, host_gap, top3, worst_p, d_worst, measure in verdicts:
        bound = ("host-bound" if host_gap > 50 else
                 "compute-bound" if host_gap < 20 else "mixed")
        md.append(f"\n### {combo}")
        md.append(f"- **Baseline:** `{b_lat:.2f} ms` latency, "
                  f"`{(b_kt_ms or 0):.3f} ms` kernel/iter → "
                  f"host gap ≈ **{host_gap:.0f}%** ({bound}).")
        top_str = ", ".join(
            f"`{p}` ({(d['latency_ms']-b_lat)/b_lat*100:+.1f}%)"
            for p, d in top3
        )
        md.append(f"- **Top-3 fastest:** {top_str}")
        d_top1 = (top3[0][1]["latency_ms"] - b_lat) / b_lat * 100
        if d_top1 <= -5:
            md.append(f"- **Recommendation:** adopt `{top3[0][0]}` "
                      f"({d_top1:+.1f}% latency).")
        else:
            md.append("- **Recommendation:** keep `baseline` — no preset "
                      "delivers a meaningful win (all within ±5% noise).")
        if d_worst >= 5:
            md.append(f"- **Avoid:** `{worst_p}` "
                      f"({d_worst:+.1f}% latency regression).")

    # ── Cross-model takeaways ─────────────────────────────────────────
    md.append("\n### Cross-model takeaways")
    md.append("- **Llama paths are host-bound** (host gap typically ≥ 80%): "
              "fewer kernel launches dominate, so `combo_kernels` and "
              "combos that include it are the main lever.")
    md.append("- **FLUX is compute-bound**: the kernel itself dominates, so "
              "config knobs barely move latency. Baseline is competitive; "
              "wide presets like `all-no_*` can even regress.")
    md.append("- **Wide multi-knob presets (`all`, `all-no_*`) are rarely "
              "optimal** — pick the smallest combo that already wins.")
    md.append("\n---\n")

    for combo, presets in rows.items():
        md.append(f"## {combo}\n")
        md.append(f"**Run config:** {fmt_meta(metas.get(combo, {}))}\n")
        # Note source of latency values for this combo
        sources = {p.get("lat_source") for p in presets.values()}
        if sources == {"clean"}:
            md.append("_Latency: clean (no-unitrace) Pass A._\n")
        elif sources == {"unitrace"}:
            md.append("_Latency: unitrace Pass B (no clean Pass A available; "
                      "values may be inflated by PTI overhead)._\n")
        else:
            md.append(f"_Latency sources mixed across presets: {sources}._\n")
        measure = metas.get(combo, {}).get("measure")
        if not measure:
            md.append("_(measure iter count not detected — kernel/iter column "
                      "shows MEASURE-window total instead of per-iter)_\n")
        base_lat = presets.get("baseline", {}).get("latency_ms")
        base_kt  = presets.get("baseline", {}).get("kernel_us")
        md.append("| Preset | latency (ms) | Δ vs base | kernel/iter (ms) | "
                  "Δ vs base | n kernels |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for p in order_presets(list(presets.keys())):
            if p not in presets:
                md.append(f"| `{p}` | – | – | – | – | – |")
                continue
            r = presets[p]
            lat = r.get("latency_ms"); kt = r.get("kernel_us"); nk = r.get("n_kernels")
            # Convert kt: us → ms, then ÷ measure for per-iter
            kt_ms_per_iter = (kt / 1000 / measure) if (kt is not None and measure) \
                              else (kt / 1000 if kt is not None else None)
            base_kt_pi = (base_kt / 1000 / measure) if (base_kt is not None and measure) \
                          else (base_kt / 1000 if base_kt is not None else None)
            d_lat = f"{(lat - base_lat)/base_lat*100:+.1f}%" if (lat and base_lat) else "–"
            d_kt  = (f"{(kt_ms_per_iter - base_kt_pi)/base_kt_pi*100:+.1f}%"
                     if (kt_ms_per_iter and base_kt_pi) else "–")
            md.append(f"| `{p}` | "
                      f"{('%.2f' % lat) if lat is not None else '–'} | {d_lat} | "
                      f"{('%.3f' % kt_ms_per_iter) if kt_ms_per_iter is not None else '–'} | {d_kt} | "
                      f"{nk if nk is not None else '–'} |")
        md.append("")

    # ── Appendix: presets & knobs reference ───────────────────────────
    md.append("\n---\n")
    md.append("## Appendix A — atomic config knobs\n")
    md.append("Each preset is built by toggling one or more of these "
              "`torch._inductor.config` attributes. All are off / default "
              "unless otherwise noted.\n")
    md.append("| Knob (label) | `torch._inductor.config` attribute | Value | Purpose |")
    md.append("|---|---|---|---|")
    knob_doc = {
        "combo":       "Pack independent kernels with the same launch grid into one "
                       "Triton kernel — fewer launches → big win on host-bound paths "
                       "(Llama generate).",
        "bench_combo": "Autotune which kernels to combine (rather than using the "
                       "static heuristic). Pairs with `combo`.",
        "epi_first":   "Prefer fusing epilogues (post-matmul pointwise) before other "
                       "fusion candidates — helps quantization paths where epilogues "
                       "are scale/dequant ops.",
        "prologue":    "Allow fusing pre-matmul ops (e.g. weight cast, transpose, "
                       "shuffle) into the matmul kernel. Reduces extra reads of "
                       "quantized weights.",
        "rr16":        "Set `realize_reads_threshold = 16` (default 4). Lets a "
                       "buffer be re-read up to 16 times before being materialised, "
                       "enabling wider vertical fusion at the cost of more recompute.",
        "expand_dim":  "Allow inductor to insert size-1 dimensions when fusing "
                       "broadcast-shaped pointwise ops, removing rank mismatch "
                       "fusion gaps.",
        "aggressive":  "Enable `aggressive_fusion = True`. Tries to fuse across more "
                       "loop-domain mismatches (reductions ↔ pointwise). Often hurts "
                       "compute-bound kernels by inflating register pressure.",
        "epi":         "Explicitly enable `epilogue_fusion = True`. Already on by "
                       "default — listed for completeness in `all`.",
    }
    try:
        from inductor_cfg import KNOBS
        for label, (attr, val) in KNOBS.items():
            doc = knob_doc.get(label, "")
            md.append(f"| `{label}` | `{attr}` | `{val!r}` | {doc} |")
    except Exception as e:
        md.append(f"_(could not import KNOBS from inductor_cfg: {e})_")

    md.append("\n## Appendix B — preset definitions\n")
    md.append("Presets are grouped by tier. The **knobs** column lists the "
              "atomic labels from Appendix A that are flipped on; everything "
              "else stays at Inductor defaults.\n")
    md.append("| Preset | Tier | Knobs | When to try |")
    md.append("|---|---|---|---|")
    preset_doc = {
        "baseline": ("control",
                     "Always include — reference for every Δ."),
        "combo_kernels": ("single",
                          "First thing to try on any host-bound model."),
        "benchmark_combo": ("single",
                            "If `combo_kernels` helps, see if autotuning the "
                            "selection helps further."),
        "epilogue_first": ("single",
                           "Quantized models with scale/dequant epilogues."),
        "prologue_fusion": ("single",
                            "Quantized models with weight-cast prologues."),
        "realize_reads": ("single",
                          "Models with shared intermediate tensors fed into "
                          "many ops (residuals, gating)."),
        "expand_dim": ("single",
                       "Models with broadcast pointwise after matmul."),
        "aggressive": ("single",
                       "Last resort; usually neutral or slight regression."),
        "combo+rr": ("pair",
                     "Standard go-to combo for Llama-style decoders."),
        "combo+epi_first": ("pair",
                            "Llama / Llama4 with quant epilogues."),
        "combo+prologue": ("pair",
                           "Quant models where weight cast is hot."),
        "combo+expand": ("pair",
                         "Best-performing pair on Llama-3.1 in this sweep."),
        "combo+aggr": ("pair",
                       "Probe whether aggressive fusion compounds with combo."),
        "rr+epi_first": ("pair", "RR amplifies epilogue gains."),
        "rr+prologue":  ("pair", "RR amplifies prologue gains."),
        "rr+expand":    ("pair", "RR + broadcast handling."),
        "rr+aggr":      ("pair", "RR + aggressive — risky on compute-bound."),
        "prologue+expand": ("pair", "No-combo combo for matmul-heavy models."),
        "epi_first+aggr":  ("pair", "Epilogue priority + aggressive."),
        "combo+rr+epi_first": ("triplet", "Standard winner stack for Llama."),
        "combo+rr+prologue":  ("triplet", "Same as above but prologue instead."),
        "combo+rr+expand":    ("triplet", "Top performer for Llama-3.1 mxfp4."),
        "combo+rr+bench":     ("triplet", "Adds combo autotuning on top."),
        "combo+rr+epi_first+expand":
            ("quadruplet", "Top-4 stack — usually plateaus vs the triplets."),
        "all": ("all-on",
                "Sanity check; reveals whether the model wants everything or "
                "whether some knob is harmful."),
        "all-no_combo":      ("ablation", "Is combo essential to the all-on win?"),
        "all-no_prologue":   ("ablation", "Is prologue helping or hurting in `all`?"),
        "all-no_aggressive": ("ablation",
                              "Often the *best* `all-*` because aggressive "
                              "is the most likely to regress."),
        "all-no_expand":     ("ablation", "Removes broadcast-fusion knob."),
    }
    try:
        from inductor_cfg import PRESETS
        for name, knobs in PRESETS.items():
            tier, doc = preset_doc.get(name, ("?", ""))
            knob_labels = [_label_for_knob(attr) for attr, _ in knobs]
            knob_str = (", ".join(f"`{k}`" for k in knob_labels)
                        if knob_labels else "_none (defaults)_")
            md.append(f"| `{name}` | {tier} | {knob_str} | {doc} |")
    except Exception as e:
        md.append(f"_(could not import PRESETS from inductor_cfg: {e})_")

    Path(args.out).write_text("\n".join(md))
    print(f"wrote {args.out}")


def _label_for_knob(attr_name: str) -> str:
    """Reverse-lookup the short label given the torch._inductor.config attr."""
    try:
        from inductor_cfg import KNOBS
        for label, (attr, _) in KNOBS.items():
            if attr == attr_name:
                return label
    except Exception:
        pass
    return attr_name


if __name__ == "__main__":
    main()
