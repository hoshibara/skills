"""Profile FLUX MXFP8 / MXFP4 with the inductor-config sweep harness.

Re-implements presi-models/reduced-flux/flux_dev_mxfp{8,4}.py adding clean
WARMUP / MEASURE / PTI bracketing.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import inductor_cfg  # noqa: F401  — auto-applies preset

from diffusers import FluxPipeline
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
)

DEVICE = "xpu"
PROMPT = "A cat holding a sign that says hello world"
INPUT_ID_LENGTH_DICT = {"4-Func": 128, "48-Perf": 1024}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", required=True, choices=["mxfp8", "mxfp4"])
    p.add_argument("--mode", default="compile", choices=["eager", "compile"])
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "1")))
    p.add_argument("--measure", type=int, default=int(os.environ.get("MEASURE", "3")))
    p.add_argument("--config-name", default=os.environ.get("MODEL_CONFIG_NAME", "4-Func"))
    p.add_argument("--num-inference-steps", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    hw = INPUT_ID_LENGTH_DICT[args.config_name]
    print("=" * 70)
    print(f" FLUX {args.precision.upper()} | mode={args.mode} "
          f"| config={args.config_name} | H=W={hw} "
          f"| warmup={args.warmup} measure={args.measure} "
          f"| steps={args.num_inference_steps}")
    print("=" * 70)

    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats("xpu")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
    )
    pipe.transformer.transformer_blocks = nn.Sequential(
        pipe.transformer.transformer_blocks[0]
    )
    pipe.transformer.single_transformer_blocks = nn.Sequential(
        pipe.transformer.single_transformer_blocks[0]
    )
    pipe.text_encoder_2.encoder.block = nn.ModuleList(
        [pipe.text_encoder_2.encoder.block[0]]
    )
    pipe.to(DEVICE)
    print(f"  loaded; mem={torch.xpu.memory_allocated()/1024**3:.2f} GB")

    if args.mode == "compile":
        print("  compiling pipe.transformer …")
        pipe.transformer.compile(fullgraph=True)

    if args.precision == "mxfp8":
        act_dtype = wt_dtype = torch.float8_e4m3fn
    else:
        act_dtype = wt_dtype = torch.float4_e2m1fn_x2

    qcfg = MXDynamicActivationMXWeightConfig(
        activation_dtype=act_dtype, weight_dtype=wt_dtype,
        kernel_preference=KernelPreference.AUTO,
    )

    def _is_linear_but_not_lm_head(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and "lm_head" not in fqn

    quantize_(pipe.transformer, config=qcfg, filter_fn=_is_linear_but_not_lm_head)

    def gen_once():
        with torch.no_grad():
            return pipe(
                PROMPT, height=hw, width=hw,
                guidance_scale=3.5,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0),
            ).images

    print("\n--- WARMUP (PTI disabled) ---")
    for i in range(args.warmup):
        t = time.time()
        gen_once()
        torch.xpu.synchronize()
        print(f"  [warmup {i+1}/{args.warmup}] {time.time()-t:.3f}s")

    os.environ["PTI_ENABLE_COLLECTION"] = "1"
    print("\n--- MEASURE (PTI enabled) ---")
    times = []
    for i in range(args.measure):
        t = time.time()
        gen_once()
        torch.xpu.synchronize()
        dt = time.time() - t
        times.append(dt)
        print(f"  [measure {i+1}/{args.measure}] {dt*1000:.1f} ms")
    os.environ["PTI_ENABLE_COLLECTION"] = "0"

    avg = sum(times) / len(times)
    peak_gb = torch.xpu.max_memory_allocated() / 1024**3
    print("\n" + "=" * 70)
    print(f" RESULT  FLUX {args.precision.upper()} {args.mode}")
    print(f"   peak XPU memory : {peak_gb:.3f} GB")
    print(f"   avg latency     : {avg*1000:.1f} ms / pipe-call")
    print(f"   per-iter (ms)   : " + ", ".join(f"{t*1000:.1f}" for t in times))


if __name__ == "__main__":
    main()
