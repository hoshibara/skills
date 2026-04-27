"""Profiling wrapper for Llama-3.1 MXFP8 / MXFP4 with warmup support.

Mirrors profile_llama4.py but uses the existing llama31_FP8.py / llama31_FP4.py
model-construction code path (re-importing it as a library would also work; we
re-implement here so PTI start/stop bracket only the measured iterations).

Usage:
  python profile_llama31.py --precision mxfp8 [--mode eager|compile] \
      [--warmup 1] [--measure 3] [--config_name 4-Func] [--prompt-tokens 128] \
      [--max-new-tokens 2]

Env passthroughs (compatible with original script):
  LLAMA_DEVICE      (default xpu)
  MODEL_NAME        (default meta-llama/Meta-Llama-3.1-70B)
  MODEL_CONFIG_NAME (default 4-Func)
"""
import argparse
import os
import sys
import time
from datetime import datetime

import torch
import inductor_cfg  # noqa: F401  — auto-applies INDUCTOR_CFG_PRESET on import
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
)

# Reach into the existing llama3.1 directory to read configs/<NAME>.json
LLAMA31_DIR = os.environ.get(
    "LLAMA31_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "vendor", "reduced-llama"),
)

PROMPT = (
    "It is done, and submitted. You can play 'Survival of the Tastiest' on "
    "Android, and on the web. Playing on the web works, but you have to "
    "simulate multiple touch for table moving and that can be a bit confusing. "
) * 32  # plenty of tokens; will be truncated to --prompt-tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", required=True, choices=["mxfp8", "mxfp4"])
    p.add_argument("--mode", default="compile", choices=["eager", "compile"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--measure", type=int, default=3)
    p.add_argument("--config-name", default=os.environ.get("MODEL_CONFIG_NAME", "4-Func"))
    p.add_argument("--prompt-tokens", type=int, default=None,
                   help="Override input length; otherwise picked per config.")
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


INPUT_ID_LENGTH_DICT = {"4-Func": 128, "48-Perf": 1024,
                         "Llama-3.1-70B": 8192, "2-Blocks-8B": 1024}
BATCH_SIZE_DICT      = {"4-Func": 4,   "48-Perf": 4,
                         "Llama-3.1-70B": 4, "2-Blocks-8B": 1}


def main():
    args = parse_args()

    DEVICE     = os.environ.get("LLAMA_DEVICE", "xpu")
    MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B")
    cfg_name   = args.config_name
    cfg_path   = os.path.join(LLAMA31_DIR, f"configs/{cfg_name}.json")
    MODEL_CONFIG = LlamaConfig.from_json_file(cfg_path)

    input_len  = args.prompt_tokens or INPUT_ID_LENGTH_DICT.get(cfg_name, 128)
    batch_size = args.batch_size    or BATCH_SIZE_DICT.get(cfg_name, 4)

    if args.precision == "mxfp8":
        act_dtype = wt_dtype = torch.float8_e4m3fn
    else:
        act_dtype = wt_dtype = torch.float4_e2m1fn_x2

    print("=" * 70)
    print(f" Llama-3.1 {args.precision.upper()} | mode={args.mode} "
          f"| config={cfg_name} | input_len={input_len} | batch={batch_size}")
    print(f" warmup={args.warmup}  measure={args.measure}  "
          f"max_new_tokens={args.max_new_tokens}")
    print(f" device={DEVICE}  model={MODEL_NAME}")
    print("=" * 70)

    if DEVICE == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()

    # Build model from config (no weights download needed — random init suffices
    # for kernel-time profiling, matches what llama31_FP8.py does).
    model = AutoModelForCausalLM.from_config(MODEL_CONFIG).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    single = tokenizer(PROMPT, return_tensors="pt", max_length=input_len,
                       truncation=True, padding="max_length").to(DEVICE)
    single["input_ids"] = torch.remainder(single["input_ids"], MODEL_CONFIG.vocab_size)
    batch_inputs = {k: v.expand(batch_size, -1).contiguous().to(DEVICE)
                    for k, v in single.items()}
    print(f"  input_ids shape: {tuple(batch_inputs['input_ids'].shape)}")

    # Quantize
    qconfig = MXDynamicActivationMXWeightConfig(
        activation_dtype=act_dtype, weight_dtype=wt_dtype,
        kernel_preference=KernelPreference.AUTO,
    )
    def _is_linear_but_not_lm_head(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and "lm_head" not in fqn
    t0 = time.time()
    quantize_(model, config=qconfig, filter_fn=_is_linear_but_not_lm_head)
    print(f"  quantize time: {time.time()-t0:.2f} s")

    # Compile (wraps model.forward, like llama31_FP8.py does)
    if args.mode == "compile":
        model.forward = torch.compile(model.forward, dynamic=True)
        print("  model.forward wrapped with torch.compile(dynamic=True)")
    model.eval()

    def gen_once():
        with torch.no_grad():
            return model.generate(
                **batch_inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=False,
            )

    # ── Warmup (untraced — PTI off) ──
    print("\n--- WARMUP (PTI disabled) ---")
    for i in range(args.warmup):
        t = time.perf_counter()
        gen_once()
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        print(f"  [warmup {i+1}/{args.warmup}] {time.perf_counter()-t:.4f}s "
              f"({datetime.now().strftime('%H:%M:%S')})")

    # ── Measured (PTI on) ──
    os.environ["PTI_ENABLE_COLLECTION"] = "1"
    print("\n--- MEASURE (PTI enabled) ---")
    if hasattr(torch, "xpu"):
        torch.xpu.reset_peak_memory_stats("xpu")

    times = []
    for i in range(args.measure):
        t = time.perf_counter()
        gen_once()
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        dt = time.perf_counter() - t
        times.append(dt)
        print(f"  [measure {i+1}/{args.measure}] {dt*1000:.4f} ms")

    os.environ["PTI_ENABLE_COLLECTION"] = "0"

    # ── Summary ──
    if hasattr(torch, "xpu"):
        peak_gb = torch.xpu.max_memory_allocated() / 1024**3
    else:
        peak_gb = -1.0
    avg = sum(times) / len(times)
    print("\n" + "=" * 70)
    print(f" RESULT  Llama-3.1 {args.precision.upper()} {args.mode}")
    print(f"   peak XPU memory : {peak_gb:.3f} GB")
    print(f"   avg latency     : {avg*1000:.4f} ms / generate({args.max_new_tokens} tok)")
    print(f"   per-iter (ms)   : " + ", ".join(f"{t*1000:.4f}" for t in times))
    print("=" * 70)


if __name__ == "__main__":
    main()
