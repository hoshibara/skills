#!/usr/bin/env python3
"""Analyze unitrace profiling results and produce a summary."""
import os
import csv
import sys
import re
from collections import defaultdict

RESULT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/xingyuan/projects/20260416-bmg-ao/unitrace_results/20260418_0142"

RUNS = [
    "flux_mxfp8_eager", "flux_mxfp8_compile",
    "flux_mxfp4_eager", "flux_mxfp4_compile",
    "llama4_mxfp8_eager", "llama4_mxfp8_compile",
    "llama4_mxfp4_eager", "llama4_mxfp4_compile",
]

def simplify_kernel_name(name):
    """Shorten verbose SYCL kernel names to readable form."""
    name = name.strip('"')
    # Memory copy operations
    if "zeCommandListAppendMemoryCopy" in name:
        m = re.search(r'zeCommandListAppendMemoryCopy\(([^)]+)\)\[(\d+)\]', name)
        if m:
            size_bytes = int(m.group(2))
            if size_bytes >= 1024*1024:
                return f"MemCopy({m.group(1)})[{size_bytes/(1024*1024):.1f}MB]"
            elif size_bytes >= 1024:
                return f"MemCopy({m.group(1)})[{size_bytes/1024:.1f}KB]"
            return f"MemCopy({m.group(1)})[{size_bytes}B]"
    if "zeCommandListAppendMemoryFill" in name:
        return "MemFill"
    if "zeCommandListAppendBarrier" in name:
        return "Barrier"
    
    # Triton kernels
    if "triton_" in name or "_kernel0d1d2d3" in name:
        m = re.search(r'(triton_\w+)', name)
        if m:
            return m.group(1)[:60]
        return "triton_kernel"
    
    # cutlass FMHA
    if "cutlass::fmha" in name:
        return "cutlass_fmha_forward"
    
    # Extract the innermost meaningful function name
    # Look for common patterns
    for pattern in [
        r'(\w+Kernel)',
        r'(\w+_kernel)',
        r'at::native::xpu::(\w+)',
        r'void __sycl_kernel_(\w+)',
    ]:
        m = re.search(pattern, name)
        if m:
            result = m.group(1)
            # Further simplify
            result = result.replace("__sycl_kernel_", "")
            if len(result) > 60:
                result = result[:57] + "..."
            return result
    
    # Fallback: first 60 chars
    if len(name) > 60:
        return name[:57] + "..."
    return name

def categorize_kernel(name):
    """Categorize kernel into high-level categories."""
    n = name.lower()
    if "memcopy" in n or "memorycopy" in n or "memoryfill" in n:
        return "Memory Transfer"
    if "fmha" in n or "attention" in n or "sdpa" in n:
        return "Attention/FMHA"
    if "gemm" in n or "matmul" in n or "mm_" in n or "xetla" in n:
        return "GEMM/MatMul"
    if "triton" in n:
        return "Triton Kernels"
    if "reduce" in n or "norm" in n or "layernorm" in n or "rmsnorm" in n:
        return "Reduce/Norm"
    if "elementwise" in n or "copy_kernel" in n or "unrolled" in n:
        return "Elementwise"
    if "softmax" in n:
        return "Softmax"
    if "index" in n or "scatter" in n or "gather" in n:
        return "Index/Gather"
    if "barrier" in n:
        return "Barrier"
    return "Other"

def parse_csv(filepath):
    """Parse unitrace CSV and extract device timing data."""
    kernels = []
    total_exec_time = 0
    total_device_time = 0
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract total times
    m = re.search(r'Total Execution Time \(ns\):\s+(\d+)', content)
    if m:
        total_exec_time = int(m.group(1))
    m = re.search(r'Total Device Time for L0 backend \(ns\):\s+(\d+)', content)
    if m:
        total_device_time = int(m.group(1))
    
    # Parse device timing section (lines starting with ")
    in_device_timing = False
    for line in content.split('\n'):
        if '=== Device Timing Summary ===' in line:
            in_device_timing = True
            continue
        if in_device_timing and '=== Kernel Properties ===' in line:
            break
        if in_device_timing and line.strip().startswith('"'):
            # Parse: "kernel_name", calls, time_ns, time_pct, avg_ns, min_ns, max_ns
            # The kernel name may contain commas, so we need careful parsing
            parts = line.strip()
            # Find the closing quote of kernel name
            if parts.startswith('"'):
                end_quote = parts.index('"', 1)
                kernel_name = parts[1:end_quote]
                rest = parts[end_quote+1:].strip().lstrip(',').strip()
                fields = [x.strip() for x in rest.split(',')]
                if len(fields) >= 6:
                    try:
                        calls = int(fields[0])
                        time_ns = int(fields[1])
                        time_pct = float(fields[2])
                        avg_ns = int(fields[3])
                        min_ns = int(fields[4])
                        max_ns = int(fields[5])
                        kernels.append({
                            'name': kernel_name,
                            'short_name': simplify_kernel_name(kernel_name),
                            'category': categorize_kernel(kernel_name),
                            'calls': calls,
                            'time_ns': time_ns,
                            'time_pct': time_pct,
                            'avg_ns': avg_ns,
                            'min_ns': min_ns,
                            'max_ns': max_ns,
                        })
                    except (ValueError, IndexError):
                        pass
    
    return {
        'total_exec_time_ns': total_exec_time,
        'total_device_time_ns': total_device_time,
        'kernels': sorted(kernels, key=lambda x: x['time_ns'], reverse=True),
    }

def find_main_csv(run_dir, run_name):
    """Find the main process CSV file."""
    # Look for python.*.json to identify main PID
    json_files = sorted([f for f in os.listdir(run_dir) if f.startswith('python.') and f.endswith('.json')])
    if not json_files:
        # Try env.*.json (for env-wrapped commands)
        json_files = sorted([f for f in os.listdir(run_dir) if f.endswith('.json') and not f.startswith('ldconfig')])
    
    for jf in json_files:
        pid = jf.replace('.json', '').split('.')[-1]
        csv_path = os.path.join(run_dir, f"{run_name}.{pid}.csv")
        if os.path.exists(csv_path):
            return csv_path
    
    # Fallback: find any CSV with the run name
    csv_files = [f for f in os.listdir(run_dir) if f.startswith(run_name) and f.endswith('.csv')]
    if csv_files:
        return os.path.join(run_dir, sorted(csv_files)[0])
    return None

def ns_to_human(ns):
    if ns >= 1e9:
        return f"{ns/1e9:.3f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.3f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.1f}us"
    return f"{ns}ns"

def print_separator(char='=', width=120):
    print(char * width)

# ============================================
# Main Analysis
# ============================================
print_separator()
print(f" UNITRACE PROFILING ANALYSIS")
print(f" Results: {RESULT_DIR}")
print_separator()

all_results = {}

for run in RUNS:
    run_dir = os.path.join(RESULT_DIR, run)
    if not os.path.isdir(run_dir):
        print(f"\n[SKIP] {run}: directory not found")
        continue
    
    csv_path = find_main_csv(run_dir, run)
    if not csv_path:
        print(f"\n[SKIP] {run}: no CSV found")
        continue
    
    data = parse_csv(csv_path)
    all_results[run] = data
    
    # Determine model and mode from run name
    parts = run.rsplit('_', 1)
    mode = parts[-1]  # eager or compile
    model = parts[0]   # e.g. flux_mxfp8
    
    print(f"\n{'='*120}")
    print(f" {run.upper()}")
    print(f"{'='*120}")
    print(f"  Total Execution Time:    {ns_to_human(data['total_exec_time_ns'])}")
    print(f"  Total Device Time (L0):  {ns_to_human(data['total_device_time_ns'])}")
    if data['total_exec_time_ns'] > 0:
        gpu_util = data['total_device_time_ns'] / data['total_exec_time_ns'] * 100
        print(f"  GPU Utilization:         {gpu_util:.2f}%")
    print(f"  Total Unique Kernels:    {len(data['kernels'])}")
    total_calls = sum(k['calls'] for k in data['kernels'])
    print(f"  Total Kernel Calls:      {total_calls}")
    
    # Top 15 kernels by time
    print(f"\n  {'Top 15 Kernels by Device Time':}")
    print(f"  {'Rank':<5} {'Calls':>7} {'Total Time':>12} {'Time%':>7} {'Avg':>12} {'Category':<18} {'Kernel'}")
    print(f"  {'-'*5} {'-'*7} {'-'*12} {'-'*7} {'-'*12} {'-'*18} {'-'*40}")
    for i, k in enumerate(data['kernels'][:15]):
        print(f"  {i+1:<5} {k['calls']:>7} {ns_to_human(k['time_ns']):>12} {k['time_pct']:>6.2f}% {ns_to_human(k['avg_ns']):>12} {k['category']:<18} {k['short_name'][:50]}")
    
    # Category breakdown
    cat_time = defaultdict(int)
    cat_calls = defaultdict(int)
    for k in data['kernels']:
        cat_time[k['category']] += k['time_ns']
        cat_calls[k['category']] += k['calls']
    
    print(f"\n  Category Breakdown:")
    print(f"  {'Category':<20} {'Time':>12} {'Time%':>7} {'Calls':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*7} {'-'*8}")
    for cat, time in sorted(cat_time.items(), key=lambda x: x[1], reverse=True):
        pct = time / data['total_device_time_ns'] * 100 if data['total_device_time_ns'] > 0 else 0
        print(f"  {cat:<20} {ns_to_human(time):>12} {pct:>6.2f}% {cat_calls[cat]:>8}")

# ============================================
# Comparative Summary
# ============================================
print(f"\n\n{'='*120}")
print(f" COMPARATIVE SUMMARY")
print(f"{'='*120}")

# Group by model
models = ["flux_mxfp8", "flux_mxfp4", "llama4_mxfp8", "llama4_mxfp4"]
print(f"\n  {'Model':<20} {'Mode':<10} {'Exec Time':>12} {'Device Time':>12} {'GPU Util%':>10} {'Kernels':>8} {'Calls':>8}")
print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

for model in models:
    for mode in ["eager", "compile"]:
        run = f"{model}_{mode}"
        if run in all_results:
            d = all_results[run]
            gpu_util = d['total_device_time_ns'] / d['total_exec_time_ns'] * 100 if d['total_exec_time_ns'] > 0 else 0
            total_calls = sum(k['calls'] for k in d['kernels'])
            print(f"  {model:<20} {mode:<10} {ns_to_human(d['total_exec_time_ns']):>12} {ns_to_human(d['total_device_time_ns']):>12} {gpu_util:>9.2f}% {len(d['kernels']):>8} {total_calls:>8}")

# Eager vs Compile comparison
print(f"\n  Eager vs Compile Speedup (Device Time):")
print(f"  {'Model':<20} {'Eager Device':>14} {'Compile Device':>16} {'Speedup':>10}")
print(f"  {'-'*20} {'-'*14} {'-'*16} {'-'*10}")
for model in models:
    eager_run = f"{model}_eager"
    compile_run = f"{model}_compile"
    if eager_run in all_results and compile_run in all_results:
        eager_dt = all_results[eager_run]['total_device_time_ns']
        compile_dt = all_results[compile_run]['total_device_time_ns']
        if compile_dt > 0:
            speedup = eager_dt / compile_dt
            print(f"  {model:<20} {ns_to_human(eager_dt):>14} {ns_to_human(compile_dt):>16} {speedup:>9.2f}x")

# MXFP8 vs MXFP4 comparison
print(f"\n  MXFP8 vs MXFP4 (Device Time):")
print(f"  {'Base Model':<15} {'Mode':<10} {'MXFP8':>14} {'MXFP4':>14} {'Ratio(FP4/FP8)':>16}")
print(f"  {'-'*15} {'-'*10} {'-'*14} {'-'*14} {'-'*16}")
for base in ["flux", "llama4"]:
    for mode in ["eager", "compile"]:
        fp8_run = f"{base}_mxfp8_{mode}"
        fp4_run = f"{base}_mxfp4_{mode}"
        if fp8_run in all_results and fp4_run in all_results:
            fp8_dt = all_results[fp8_run]['total_device_time_ns']
            fp4_dt = all_results[fp4_run]['total_device_time_ns']
            if fp8_dt > 0:
                ratio = fp4_dt / fp8_dt
                print(f"  {base:<15} {mode:<10} {ns_to_human(fp8_dt):>14} {ns_to_human(fp4_dt):>14} {ratio:>15.2f}x")

print(f"\n{'='*120}")
print(f" Analysis complete.")
print(f"{'='*120}")
