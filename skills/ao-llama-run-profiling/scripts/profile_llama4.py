"""Profiling wrapper for Llama4 with warmup support.
Usage:
  python profile_llama4.py --script llama4-FP8.py [--warmup N] [--measure N]

Runs warmup iterations first (untraced), then enables
PTI_ENABLE_COLLECTION for measured iterations.
"""
import torch
import os
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, choices=["llama4-FP8.py", "llama4-FP4.py"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    args = parser.parse_args()

    from transformers import Llama4Config, Llama4ForCausalLM, AutoTokenizer
    from transformers.quantizers.base import SequentialLlama4TextExperts
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
    from torchao.quantization import quantize_
    import torchao.prototype.mx_formats
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.prototype.mx_formats.inference_workflow import (
        MXDynamicActivationMXWeightConfig,
    )

    DEVICE = "xpu"
    MODEL_NAME = os.environ.get('MODEL_NAME', "meta-llama/Llama-4-Maverick-17B-128E-Instruct")
    COMPILE = os.environ.get('USE_COMPILE', 'False').lower() in ('1', 'true', 'yes', 'on')
    MODEL_CONFIG_NAME = os.environ.get('MODEL_CONFIG_NAME', "4-Func-FP8")

    SHELL_DIR = os.path.dirname(os.path.abspath(__file__))
    # Config files are in the llama4 directory
    LLAMA_DIR = os.path.join(SHELL_DIR, "frameworks.ai.pytorch.gpu-models/presi-models/reduced-llama4")
    MODEL_CONFIG = Llama4Config.from_json_file(os.path.join(LLAMA_DIR, f'configs/{MODEL_CONFIG_NAME}.json'))

    INPUT_ID_LENGTH = {"4-Func-FP8": 128, "48-Perf-FP8": 1024}.get(MODEL_CONFIG_NAME, 128)
    BATCH_SIZE = {"4-Func-FP8": 4, "48-Perf-FP8": 4}.get(MODEL_CONFIG_NAME, 4)

    PROMPT = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web."

    # Determine precision
    if "FP4" in args.script:
        act_dtype = torch.float4_e2m1fn_x2
        wt_dtype = torch.float4_e2m1fn_x2
        precision_tag = "MXFP4"
    else:
        act_dtype = torch.float8_e4m3fn
        wt_dtype = torch.float8_e4m3fn
        precision_tag = "MXFP8"

    def replace_experts_with_sequential(model, config):
        for name, module in list(model.named_modules()):
            if isinstance(module, Llama4TextExperts):
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    child_name = parts[1]
                else:
                    parent = model
                    child_name = name
                sequential = SequentialLlama4TextExperts(config)
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                sequential = sequential.to(device=device, dtype=dtype)
                setattr(parent, child_name, sequential)
                print(f"  Replaced {name}: Llama4TextExperts -> SequentialLlama4TextExperts")
        return model

    # ── Load model ──
    torch.xpu.empty_cache()
    text_config = MODEL_CONFIG.text_config
    model = Llama4ForCausalLM._from_config(text_config)
    model = model.to(torch.bfloat16).to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}, dtype: {next(model.parameters()).dtype}")

    model = replace_experts_with_sequential(model, text_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    single_input = tokenizer(PROMPT, return_tensors="pt", max_length=INPUT_ID_LENGTH,
                             padding="max_length", truncation=True).to(DEVICE)
    vocab_size = text_config.vocab_size
    single_input["input_ids"] = torch.remainder(single_input["input_ids"], vocab_size)
    batch_inputs = {k: v.expand(BATCH_SIZE, -1).contiguous() for k, v in single_input.items()}
    print(f"Input shape: {batch_inputs['input_ids'].shape}")

    # ── Quantize ──
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=act_dtype, weight_dtype=wt_dtype,
        kernel_preference=KernelPreference.AUTO,
    )
    def _is_expert_linear(mod, fqn):
        if not isinstance(mod, torch.nn.Linear):
            return False
        is_expert = (".feed_forward.experts." in fqn and "shared_expert" not in fqn
                     and (fqn.endswith(".gate_proj") or fqn.endswith(".up_proj") or fqn.endswith(".down_proj")))
        if is_expert:
            print(f"  [QUANT] {fqn}")
        return is_expert
    quantize_(model, config=config, filter_fn=_is_expert_linear)

    # ── Compile ──
    if COMPILE:
        print("Compiling model...")
        model = torch.compile(model)
        print("Compile wrap done.")

    # ── Warmup ──
    print(f"\n{'='*60}")
    print(f" Llama4 {precision_tag} | compile={COMPILE} | config={MODEL_CONFIG_NAME}")
    print(f" Warmup: {args.warmup} iters | Measure: {args.measure} iters")
    print(f"{'='*60}")

    with torch.no_grad():
        for i in range(args.warmup):
            t0 = time.time()
            _ = model(**batch_inputs)
            torch.xpu.synchronize()
            elapsed = time.time() - t0
            print(f"  [warmup {i+1}/{args.warmup}] {elapsed:.3f}s")

    # ── Enable profiling ──
    os.environ["PTI_ENABLE_COLLECTION"] = "1"
    print(f"\n>>> PTI collection ENABLED — measuring {args.measure} iterations")

    torch.xpu.reset_peak_memory_stats("xpu")
    times = []
    with torch.no_grad():
        for i in range(args.measure):
            t0 = time.time()
            _ = model(**batch_inputs)
            torch.xpu.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  [measure {i+1}/{args.measure}] {elapsed:.3f}s")

    os.environ["PTI_ENABLE_COLLECTION"] = "0"
    print(f"<<< PTI collection DISABLED")

    # ── Summary ──
    inference_memory = torch.xpu.max_memory_allocated() / 1024**3
    avg_time = sum(times) / len(times)
    print(f"\n{'='*60}")
    print(f" RESULTS: Llama4 {precision_tag} {'compile' if COMPILE else 'eager'}")
    print(f"{'='*60}")
    print(f"  Max memory allocated: {inference_memory:.3f} GB")
    print(f"  Avg latency:  {avg_time*1000:.1f} ms/iter")
    print(f"  All latencies: {[f'{t*1000:.1f}ms' for t in times]}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
