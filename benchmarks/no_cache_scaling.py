"""M2 measurement: how long does ONE no-cache forward step take as the sequence length T grows?

This is exactly the step `generate_no_cache` repeats (forward over T tokens, logits for the last
position only). Sweeping T finds where the step stops being flat (memory-bound) and starts growing
(compute-bound): the knee of the roofline.

Device and dtype are labeled in the output. This is an Apple M2 / MPS result and says nothing about
NVIDIA GPUs.

Usage: uv run python benchmarks/no_cache_scaling.py
"""

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "benchmarks" / "results" / "no_cache_scaling_fp16.json"
OUT_TXT = ROOT / "benchmarks" / "results" / "no_cache_scaling_fp16.txt"

LENGTHS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
WARMUP = 3
REPS = 15


def import_script():
    spec = importlib.util.spec_from_file_location("hf_reference", ROOT / "scripts" / "hf_reference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


@torch.inference_mode()
def time_forward(model, T: int, device: str, hf) -> list[float]:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (1, T), generator=g).to(device)
    times = []
    for i in range(WARMUP + REPS):
        hf.sync(device)
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        hf.sync(device)
        if i >= WARMUP:
            times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    hf = import_script()
    dtype = getattr(torch, hf.CONFIG["bench_dtype"])
    tok, model, device = hf.load(dtype)

    emb = model.get_input_embeddings().weight.numel()  # V * d; tied with lm_head
    body_params = sum(p.numel() for p in model.parameters()) - emb
    lm_head_flops = 2 * emb  # one position only, because logits_to_keep=1

    rows = []
    for T in LENGTHS:
        ts = time_forward(model, T, device, hf)
        med = statistics.median(ts)
        flops = 2 * body_params * T + lm_head_flops
        rows.append({
            "T": T,
            "median_ms": med * 1e3,
            "min_ms": min(ts) * 1e3,
            "max_ms": max(ts) * 1e3,
            "implied_tflops": flops / med / 1e12,
        })
        print(f"T={T:4d}  median {med * 1e3:7.2f} ms", flush=True)

    base = rows[0]["median_ms"]
    knee = next((r["T"] for r in rows if r["median_ms"] > 1.25 * base), None)

    lines = [
        f"No-cache forward step vs sequence length T",
        f"{hf.CONFIG['model_id']}  device={device} (Apple M2, 16 GB)  dtype={hf.CONFIG['bench_dtype']}  "
        f"warmup={WARMUP} reps={REPS}  torch={torch.__version__}",
        f"FLOPs model: 2 * {body_params / 1e6:.0f}M body params * T + one lm_head row. "
        f"'knee' = first T whose median exceeds 1.25x the T=1 median.",
        "",
        "     T   median ms    min ms    max ms   ms/token   implied TFLOPS",
    ]
    for r in rows:
        lines.append(f"  {r['T']:4d}  {r['median_ms']:9.2f} {r['min_ms']:9.2f} {r['max_ms']:9.2f}"
                     f"  {r['median_ms'] / r['T']:9.3f}  {r['implied_tflops']:10.2f}")
    lines += ["", f"knee (first T > 1.25x T=1 time): {knee}"]
    text = "\n".join(lines)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"rows": rows, "knee_T": knee, "device": device,
                                    "dtype": hf.CONFIG["bench_dtype"]}, indent=2))
    OUT_TXT.write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
