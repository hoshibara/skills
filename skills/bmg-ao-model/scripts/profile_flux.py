"""Profiling wrapper for FLUX with warmup support.
Usage:
  python profile_flux.py --script flux_dev_mxfp8.py [--warmup N] [--measure N]
  
When COMPILE mode is on, the first inference triggers torch.compile.
This wrapper runs warmup iterations first (untraced), then enables
PTI_ENABLE_COLLECTION for measured iterations.
"""
import torch
import torch.nn as nn
import os
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, choices=["flux_dev_mxfp8.py", "flux_dev_mxfp4.py"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    args = parser.parse_args()

    from diffusers import FluxPipeline
    from torchao.quantization import quantize_
    import torchao.prototype.mx_formats
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.prototype.mx_formats.inference_workflow import (
        MXDynamicActivationMXWeightConfig,
    )

    DEVICE = "xpu"
    COMPILE = os.environ.get('COMPILE', 'False').lower() in ('true', '1', 'y', 'yes')
    MODEL_CONFIG_NAME = os.environ.get('MODEL_CONFIG_NAME', "4-Func")
    INPUT_ID_LENGTH_DICT = {"4-Func": 128, "48-Perf": 1024}

    # Determine precision from script name
    if "mxfp4" in args.script:
        act_dtype = torch.float4_e2m1fn_x2
        wt_dtype = torch.float4_e2m1fn_x2
        precision_tag = "MXFP4"
    else:
        act_dtype = torch.float8_e4m3fn
        wt_dtype = torch.float8_e4m3fn
        precision_tag = "MXFP8"

    def do_inference(pipe, prompt):
        return pipe(
            prompt,
            height=INPUT_ID_LENGTH_DICT[MODEL_CONFIG_NAME],
            width=INPUT_ID_LENGTH_DICT[MODEL_CONFIG_NAME],
            guidance_scale=3.5,
            num_inference_steps=1,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images

    # ── Load model ──
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats("xpu")
    torch.xpu.reset_accumulated_memory_stats("xpu")

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.transformer.transformer_blocks = nn.Sequential(pipe.transformer.transformer_blocks[0])
    pipe.transformer.single_transformer_blocks = nn.Sequential(pipe.transformer.single_transformer_blocks[0])
    pipe.text_encoder_2.encoder.block = nn.ModuleList([pipe.text_encoder_2.encoder.block[0]])
    pipe.to(DEVICE)

    memory_after_loading = torch.xpu.memory_allocated() / 1024**3
    prompt = "A cat holding a sign that says hello world"
    print(f"Memory after loading: {memory_after_loading:.2f} GB")

    # ── Compile (lazy) ──
    if COMPILE:
        print("Compiling the model (lazy wrap)...")
        pipe.transformer.compile(fullgraph=True)
        print("Compile wrap done.")

    # ── Quantize ──
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=act_dtype,
        weight_dtype=wt_dtype,
        kernel_preference=KernelPreference.AUTO,
    )
    def _filter(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and "lm_head" not in fqn
    quantize_(pipe.transformer, config=config, filter_fn=_filter)

    # ── Warmup (not profiled) ──
    print(f"\n{'='*60}")
    print(f" FLUX {precision_tag} | compile={COMPILE} | config={MODEL_CONFIG_NAME}")
    print(f" Warmup: {args.warmup} iters | Measure: {args.measure} iters")
    print(f"{'='*60}")

    with torch.no_grad():
        for i in range(args.warmup):
            t0 = time.time()
            _ = do_inference(pipe, prompt)
            torch.xpu.synchronize()
            elapsed = time.time() - t0
            print(f"  [warmup {i+1}/{args.warmup}] {elapsed:.3f}s")

    # ── Enable profiling collection ──
    os.environ["PTI_ENABLE_COLLECTION"] = "1"
    print(f"\n>>> PTI collection ENABLED — measuring {args.measure} iterations")

    torch.xpu.reset_peak_memory_stats("xpu")
    times = []
    with torch.no_grad():
        for i in range(args.measure):
            t0 = time.time()
            image = do_inference(pipe, prompt)
            torch.xpu.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  [measure {i+1}/{args.measure}] {elapsed:.3f}s  ({1/elapsed:.2f} it/s)")

    # ── Disable profiling collection ──
    os.environ["PTI_ENABLE_COLLECTION"] = "0"
    print(f"<<< PTI collection DISABLED")

    # ── Summary ──
    inference_memory = torch.xpu.max_memory_allocated() / 1024**3
    avg_time = sum(times) / len(times)
    print(f"\n{'='*60}")
    print(f" RESULTS: FLUX {precision_tag} {'compile' if COMPILE else 'eager'}")
    print(f"{'='*60}")
    print(f"  Max memory allocated: {inference_memory:.3f} GB")
    print(f"  Avg latency:  {avg_time*1000:.1f} ms/iter")
    print(f"  Avg speed:    {1/avg_time:.2f} it/s")
    print(f"  All latencies: {[f'{t*1000:.1f}ms' for t in times]}")
    print(f"{'='*60}")

    image[0].save("flux-dev.png")
    print("Image saved as flux-dev.png")

if __name__ == "__main__":
    main()
