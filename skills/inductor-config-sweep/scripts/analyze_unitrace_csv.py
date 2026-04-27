#!/usr/bin/env python3
"""Summarise a unitrace device-timing CSV produced by:
    unitrace -d -v -s --chrome-kernel-logging -o <name>.csv ...

Usage:
    python analyze_unitrace_csv.py <result_dir>          # picks largest *.csv
    python analyze_unitrace_csv.py <path/to/file.csv>

Output (stdout, markdown):
    * Overall device time + per-iteration estimate
    * Top-N kernels by total time
    * Aggregate per *kernel category* (gemm / triton_poi / triton_per /
      triton_red / sdpa / memcpy / other)
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict


def find_csv(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    if os.path.isdir(arg):
        cands = glob.glob(os.path.join(arg, "*.csv"))
        if not cands:
            sys.exit(f"no .csv in {arg}")
        return max(cands, key=os.path.getsize)
    sys.exit(f"not found: {arg}")


def parse_total_ns(path: str):
    """Return (l0_total_ns, summary_total_ns) from the header section."""
    l0 = total = None
    with open(path) as fh:
        for line in fh:
            if "Total Device Time for L0 backend" in line:
                m = re.search(r"(\d+)", line);  l0    = int(m.group(1)) if m else None
            elif "Total Execution Time" in line and total is None:
                m = re.search(r"(\d+)", line);  total = int(m.group(1)) if m else None
    return l0, total


def parse_rows(path: str):
    """Yield (kernel_name, calls, time_ns, pct, avg_ns, min_ns, max_ns).
    Only rows from the ``== L0 Backend ==`` device-timing table.
    """
    rows = []
    in_block = False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("== L0 Backend =="):
                in_block = True
                continue
            if not in_block:
                continue
            if line.startswith("==="):
                break
            if not line:
                continue
            if line.lstrip().startswith("Kernel,"):
                continue
            try:
                rdr = csv.reader([line], skipinitialspace=True)
                fields = next(rdr)
            except csv.Error:
                continue
            if len(fields) < 7:
                continue
            try:
                rows.append((
                    fields[0].strip().strip('"'),
                    int(fields[1]),
                    int(fields[2]),
                    float(fields[3]),
                    int(fields[4]),
                    int(fields[5]),
                    int(fields[6]),
                ))
            except ValueError:
                continue
    return rows


def categorise(name: str) -> str:
    if "triton_poi_" in name:           return "triton_poi (pointwise)"
    if "triton_per_" in name:           return "triton_per (persistent_reduction)"
    if "triton_red_" in name:           return "triton_red (reduction)"
    if "triton_tem_" in name:           return "triton_tem (template)"
    if "triton_"     in name:           return "triton_other"
    if "gemm_kernel" in name:           return "gemm (XeTLA/oneDNN)"
    if "micro_sdpa"  in name or "sdpa" in name.lower(): return "sdpa"
    if "MemoryCopy"  in name or "memcpy" in name.lower(): return "memcpy"
    if "MemoryFill"  in name or "Barrier" in name: return "barrier/fill"
    if "elementwise" in name:           return "aten_eltwise"
    if "Reduce" in name or "Radix" in name: return "aten_reduce/sort"
    return "other"


def short(name: str, n: int = 90) -> str:
    name = re.sub(r"\s+", " ", name)
    # strip SIMD/grid suffix for readability in summary
    name = re.sub(r"\[SIMD\d+ .*?\]$", "", name).strip()
    if len(name) > n:
        name = name[:n - 1] + "…"
    return name


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_unitrace_csv.py <csv_or_dir>")
    path = find_csv(sys.argv[1])
    rows = parse_rows(path)
    l0_total, total = parse_total_ns(path)
    if not rows:
        sys.exit(f"no kernel rows parsed from {path}")

    sum_ns   = sum(r[2] for r in rows)
    sum_calls = sum(r[1] for r in rows)
    rows.sort(key=lambda r: r[2], reverse=True)

    # category aggregate
    cat = defaultdict(lambda: [0, 0])  # cat -> [calls, ns]
    for name, calls, ns, *_ in rows:
        c = categorise(name)
        cat[c][0] += calls
        cat[c][1] += ns

    print(f"# Unitrace kernel summary — `{os.path.basename(path)}`\n")
    print(f"- file: `{path}`")
    if l0_total is not None:
        print(f"- L0 device time : **{l0_total/1e6:.3f} ms** "
              f"({l0_total} ns)")
    if total is not None:
        print(f"- total wall (incl. host) : {total/1e6:.3f} ms")
    print(f"- distinct kernels : {len(rows)}     total launches : {sum_calls}")
    print(f"- sum of per-kernel time : {sum_ns/1e6:.3f} ms\n")

    print("## Per category\n")
    print("| Category | Launches | Time (ms) | Share |")
    print("|---|---:|---:|---:|")
    for c, (calls, ns) in sorted(cat.items(), key=lambda kv: -kv[1][1]):
        pct = ns / sum_ns * 100 if sum_ns else 0
        print(f"| {c} | {calls} | {ns/1e6:.3f} | {pct:.1f}% |")
    print()

    TOPN = 25
    print(f"## Top {TOPN} kernels by total device time\n")
    print("| # | Time (ms) | Share | Calls | Avg (us) | Kernel |")
    print("|---:|---:|---:|---:|---:|---|")
    for i, (name, calls, ns, pct, avg, _mn, _mx) in enumerate(rows[:TOPN], 1):
        print(f"| {i} | {ns/1e6:.3f} | {pct:.2f}% | {calls} "
              f"| {avg/1000:.2f} | `{short(name)}` |")
    print()

    print("## Quick-look fusion / overhead notes\n")
    triton_ns = sum(v[1] for k, v in cat.items() if k.startswith("triton_"))
    gemm_ns   = cat.get("gemm (XeTLA/oneDNN)", [0, 0])[1]
    sdpa_ns   = cat.get("sdpa", [0, 0])[1]
    print(f"- Triton (epilogue + pointwise) : **{triton_ns/1e6:.2f} ms** "
          f"({triton_ns/sum_ns*100:.1f}%)")
    print(f"- GEMM (`gemm_kernel`)           : **{gemm_ns/1e6:.2f} ms** "
          f"({gemm_ns/sum_ns*100:.1f}%)")
    print(f"- SDPA                           : **{sdpa_ns/1e6:.2f} ms** "
          f"({sdpa_ns/sum_ns*100:.1f}%)")
    if triton_ns > gemm_ns:
        print("- ⚠ Triton epilogue/pointwise time exceeds GEMM time — "
              "likely a fusion-gap signal (see `extern_chain_fusion.md` if "
              "you also captured TORCH_COMPILE_DEBUG output).")


if __name__ == "__main__":
    main()
