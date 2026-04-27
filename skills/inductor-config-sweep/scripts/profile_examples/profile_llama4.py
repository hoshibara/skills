"""Profile Llama-4 MoE MXFP8 / MXFP4 with inductor-config sweep support.

Re-implements the load+quant+forward path from
  presi-models/reduced-llama4/llama4-FP8.py  (and -FP4.py)
adding a clean WARMUP / MEASURE / PTI-toggle bracket so unitrace measures
only the steady-state forward, not torch.compile cost.

Activated inductor config knob is selected by env INDUCTOR_CFG_PRESET (see
inductor_cfg.py).
"""
import argparse
import importlib.util
import os
import runpy
import sys
import time
from datetime import datetime

import torch
import inductor_cfg  # noqa: F401 — auto-applies preset

LLAMA4_DIR = os.environ.get(
    "LLAMA4_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "vendor", "reduced-llama4"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", required=True, choices=["mxfp8", "mxfp4"])
    p.add_argument("--mode", default="compile", choices=["eager", "compile"])
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "1")))
    p.add_argument("--measure", type=int, default=int(os.environ.get("MEASURE", "3")))
    p.add_argument("--config-name", default=os.environ.get("MODEL_CONFIG_NAME", "4-Func"))
    return p.parse_args()


def main():
    args = parse_args()

    # Tell the underlying llama4 script: do compile if requested, no PTI gating
    # — we manage PTI here.
    os.environ["LLAMA_DEVICE"] = os.environ.get("LLAMA_DEVICE", "xpu")
    os.environ["USE_COMPILE"] = "True" if args.mode == "compile" else "False"
    os.environ["LLAMA_PRECISION"] = args.precision
    os.environ["MODEL_CONFIG_NAME"] = args.config_name
    # Force the script's internal warmup/measure to 0/1 so we can do our own
    # bracketing — but llama4-FP8 already exposes WARMUP/MEASURE/PTI gating,
    # so we can let it drive everything.
    os.environ["WARMUP"]  = str(args.warmup)
    os.environ["MEASURE"] = str(args.measure)
    # Make sure PTI starts off; the script flips it on around its own measure
    # loop.
    os.environ["PTI_ENABLE_COLLECTION"] = "0"

    target = "llama4-FP8.py" if args.precision == "mxfp8" else "llama4-FP4.py"
    target_path = os.path.join(LLAMA4_DIR, target)
    print("=" * 70)
    print(f" Llama-4 {args.precision.upper()} | mode={args.mode} "
          f"| config={args.config_name} | warmup={args.warmup} measure={args.measure}")
    print(f" target script: {target_path}")
    print("=" * 70)

    # Run the underlying script as __main__. It does its own PTI gating.
    sys.argv = [target_path]
    sys.path.insert(0, LLAMA4_DIR)
    runpy.run_path(target_path, run_name="__main__")


if __name__ == "__main__":
    main()
