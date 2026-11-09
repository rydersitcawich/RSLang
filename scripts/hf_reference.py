"""M1: HF reference run.

Produces two artifacts (see CLAUDE.md dtype rules):
  1. golden tokens  (fp32, greedy)  -> tests/golden/hf_greedy_fp32.json   (M2 must match exactly)
  2. perf results   (fp16, MPS)     -> benchmarks/results/hf_reference_fp16.json

Usage:
    uv run python scripts/hf_reference.py golden
    uv run python scripts/hf_reference.py bench [--runs 10] [--new-tokens 256] [--out PATH]
    uv run python scripts/hf_reference.py stats [--out PATH]    # re-analyze saved raw timings
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.streamers import BaseStreamer


ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "configs" / "qwen3_0.6b.yaml").read_text())
GOLDEN_PATH = ROOT / "tests" / "golden" / "hf_greedy_fp32.json"
PERF_PATH = ROOT / "benchmarks" / "results" / "hf_reference_fp16.json"

PROMPTS: list[str] = ["Explain the difference between supervised and unsupervised learning in machine learning.", "Finish the following story: Once upon a time, in a land far away, there lived a young princess who loved to explore the forests and mountains surrounding her kingdom. One day, she stumbled upon a hidden cave that led to a magical world filled with talking animals and enchanted creatures. As she ventured deeper into this mystical realm, she discovered a secret that would change her life forever. She saw a shimmering portal that glowed with an otherworldly light, beckoning her to step through and uncover the mysteries that lay beyond. With a mix of excitement and trepidation, she took a deep breath and stepped into the unknown, ready to embrace the adventure that awaited her.",]

NUM_RUNS = 10  # warm runs per prompt, after one extra "cold" run that is recorded separately
BENCH_NEW_TOKENS = 256  # long enough that ITL-vs-position shows the KV-cache effect
GOLDEN_NEW_TOKENS = 64  # tokens per prompt in the golden file M2 must reproduce
WINDOW = 32  # decode tokens per window in the ITL-vs-position table

# Spec-sheet peak for Apple M2 unified memory (NOT measured; real achievable is lower).
PEAK_BANDWIDTH_GBPS = 100.0


def get_device() -> str:
    if CONFIG["device"] == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Drain queued GPU work so a perf_counter() reading means 'the GPU has finished'."""
    if device == "mps":
        torch.mps.synchronize()


def load(dtype: torch.dtype):
    """Load tokenizer + model. Time this separately: it is the cold *process* cost."""
    device = get_device()
    tok = AutoTokenizer.from_pretrained(CONFIG["model_id"])
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model_id"], dtype=dtype)
    model.to(device).eval()
    return tok, model, device


def summarize(xs: list[float]) -> dict:
    """Keep chronological order too: sorting hides thermal drift."""
    return {
        "chronological": xs,
        "sorted": sorted(xs),
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "min": min(xs),
        "max": max(xs),
    }


def kv_bytes_per_token(model) -> int:
    """K and V, for every layer, for one token of one sequence."""
    return model.config.num_hidden_layers * model.config.num_key_value_heads * 2 * model.config.head_dim * torch.finfo(model.dtype).bits // 8


def weight_bytes(model) -> int:
    # model.parameters() yields a tied tensor (embedding == lm_head) once.
    return sum(p.numel() * p.element_size() for p in model.parameters())


def generate_greedy(model: PreTrainedModel, tok: PreTrainedTokenizerBase, prompt: str, max_new_tokens: int, device: str, streamer: BaseStreamer | None = None):
    """Greedy generate. Returns the new token ids (not the prompt), exactly max_new_tokens of them."""
    model_inputs = tok([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        min_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample = False,
        use_cache = True,
        streamer=streamer,
    )
    return generated_ids[0][len(model_inputs.input_ids[0]):].tolist()


# ---------------------------------------------------------------------------
# Timing: collect raw data
# ---------------------------------------------------------------------------


class TimingStreamer(BaseStreamer):
    """Records a perf_counter() timestamp every time generate() calls put().

    generate() calls put() once with the prompt ids, then once per new token. We deliberately
    do NOT call sync() in here: HF already copies each token to the host before streaming it,
    and in this transformers version on MPS it does so one step *behind* the GPU (so the host
    can queue step k+1 while reading step k). An extra synchronize() here would stall that
    pipeline and make every timestamp wait for the *next* step's compute.
    """

    def __init__(self) -> None:
        self.stamps: list[float] = []

    def put(self, value) -> None:
        self.stamps.append(time.perf_counter())

    def end(self) -> None:
        pass


def time_one_run(model, tok, prompt: str, max_new_tokens: int, device: str) -> dict:
    """One greedy generation. Returns raw timings in seconds (analysis happens later).

    ttft_s : t0 -> first new token reaches the host. Includes tokenization and generate() setup.
    itl_s  : gaps between consecutive new tokens (length max_new_tokens - 1).
    """
    streamer = TimingStreamer()
    sync(device)  # nothing from a previous run may still be queued
    t0 = time.perf_counter()
    generate_greedy(model, tok, prompt, max_new_tokens, device, streamer=streamer)
    sync(device)
    t_done = time.perf_counter()

    stamps = streamer.stamps
    assert len(stamps) == max_new_tokens + 1, (
        f"expected 1 prompt put + {max_new_tokens} token puts, got {len(stamps)} puts"
    )
    prompt_stamp, token_stamps = stamps[0], stamps[1:]
    return {
        "setup_s": prompt_stamp - t0,
        "ttft_s": token_stamps[0] - t0,
        "itl_s": [b - a for a, b in zip(token_stamps, token_stamps[1:])],
        "total_s": t_done - t0,
    }


# ---------------------------------------------------------------------------
# Parsing: raw timings -> statistics
# ---------------------------------------------------------------------------


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.9 * len(s)))]


def _run_stats(run: dict) -> dict:
    itl = run["itl_s"]
    return {
        "ttft_ms": run["ttft_s"] * 1e3,
        "itl_mean_ms": statistics.fmean(itl) * 1e3,
        "itl_median_ms": statistics.median(itl) * 1e3,
        "itl_p90_ms": _p90(itl) * 1e3,
        "decode_tok_s": 1.0 / statistics.fmean(itl),
        "wall_tok_s": (len(itl) + 1) / run["total_s"],
    }


def _windows(warm_runs: list[dict], prompt_tokens: int, meta: dict) -> list[dict]:
    """Median ITL per WINDOW-token slice, taken across warm runs, vs the roofline prediction.

    ITL index i is the gap before new token i+2, so a slice [a, b) of ITLs happens at a context of
    roughly prompt_tokens + (a+b)/2 cached tokens.
    """
    n = len(warm_runs[0]["itl_s"])
    out = []
    for a in range(0, n, WINDOW):
        b = min(a + WINDOW, n)
        per_run = [statistics.median(r["itl_s"][a:b]) for r in warm_runs]
        itl = statistics.median(per_run)
        ctx = prompt_tokens + (a + b) / 2
        predicted_tok_s = PEAK_BANDWIDTH_GBPS * 1e9 / (meta["weight_bytes"] + ctx * meta["kv_bytes_per_token"])
        measured_tok_s = 1.0 / itl
        out.append({
            "itl_index_range": [a, b],
            "context_tokens": ctx,
            "itl_ms": itl * 1e3,
            "measured_tok_s": measured_tok_s,
            "roofline_tok_s": predicted_tok_s,
            "measured_over_roofline": measured_tok_s / predicted_tok_s,
        })
    return out


def analyze(raw: dict) -> dict:
    """Pure function: raw timings in, statistics out. Run 0 of each prompt is the cold run."""
    meta = raw["meta"]
    result = []
    for p in raw["prompts"]:
        cold, warm = p["runs"][0], p["runs"][1:]
        entry = {
            "prompt_idx": p["prompt_idx"],
            "prompt_tokens": p["prompt_tokens"],
            "cold": _run_stats(cold),
        }
        if warm:
            ws = [_run_stats(r) for r in warm]
            entry["warm"] = {
                "ttft_ms": summarize([s["ttft_ms"] for s in ws]),
                "itl_median_ms": summarize([s["itl_median_ms"] for s in ws]),
                "decode_tok_s": summarize([s["decode_tok_s"] for s in ws]),
                "wall_tok_s": summarize([s["wall_tok_s"] for s in ws]),
            }
            entry["itl_by_position"] = _windows(warm, p["prompt_tokens"], meta)
        result.append(entry)
    return {"prompts": result}


def print_summary(raw: dict, stats: dict) -> None:
    m = raw["meta"]
    print(f"\n{m['model_id']}  device={m['device']} ({m['hardware']})  dtype={m['dtype']}  "
          f"new_tokens={m['new_tokens']}  warm_runs={m['warm_runs']}")
    print(f"weights {m['weight_bytes'] / 1e9:.2f} GB, KV {m['kv_bytes_per_token']} B/token, "
          f"roofline uses SPEC peak {PEAK_BANDWIDTH_GBPS:.0f} GB/s (not measured), load {m['load_s']:.2f}s")
    for e in stats["prompts"]:
        print(f"\n--- prompt {e['prompt_idx']} ({e['prompt_tokens']} tokens) ---")
        c = e["cold"]
        print(f"cold  : TTFT {c['ttft_ms']:8.1f} ms | ITL mean {c['itl_mean_ms']:6.1f} ms | {c['decode_tok_s']:6.1f} tok/s")
        if "warm" not in e:
            continue
        w = e["warm"]
        print(f"warm  : TTFT median {w['ttft_ms']['median']:8.1f} ms (mean {w['ttft_ms']['mean']:.1f})")
        print(f"        ITL  median {w['itl_median_ms']['median']:6.2f} ms (mean of medians {w['itl_median_ms']['mean']:.2f})")
        print(f"        decode tok/s median {w['decode_tok_s']['median']:.1f} (mean {w['decode_tok_s']['mean']:.1f}) | "
              f"wall tok/s median {w['wall_tok_s']['median']:.1f}")
        print("        decode tok/s per run, chronological:", [round(x, 1) for x in w["decode_tok_s"]["chronological"]])
        print("        decode tok/s per run, sorted       :", [round(x, 1) for x in w["decode_tok_s"]["sorted"]])
        print("        ITL by position (warm, median across runs):")
        print("          ITL idx    ctx    ITL ms   tok/s   roofline   measured/roofline")
        for win in e["itl_by_position"]:
            a, b = win["itl_index_range"]
            print(f"          {a:3d}-{b:<4d} {win['context_tokens']:6.0f} {win['itl_ms']:8.2f} {win['measured_tok_s']:7.1f} "
                  f"{win['roofline_tok_s']:9.1f} {win['measured_over_roofline']:12.2f}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_golden(new_tokens: int) -> None:
    """fp32 greedy token ids for each prompt -> GOLDEN_PATH. This is the M2 correctness target.

    fp32 (not fp16) so near-tied logits don't flip the argmax from rounding noise.
    Generated on MPS; a different device/torch build could in principle break a tie differently.
    """
    dtype = getattr(torch, CONFIG["test_dtype"])
    tok, model, device = load(dtype)
    cases = []
    for prompt in PROMPTS:
        cases.append({
            "prompt": prompt,
            "prompt_token_ids": tok(prompt)["input_ids"],
            "new_token_ids": generate_greedy(model, tok, prompt, new_tokens, device),
        })
    golden = {
        "model_id": CONFIG["model_id"],
        # The cached snapshot dir is named after the HF commit hash it was downloaded at.
        "model_revision": Path(snapshot_download(CONFIG["model_id"], local_files_only=True)).name,
        "dtype": CONFIG["test_dtype"],
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "max_new_tokens": new_tokens,
        "generation": {"do_sample": False, "min_new_tokens": new_tokens, "max_new_tokens": new_tokens},
        "cases": cases,
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2))
    print(f"saved {GOLDEN_PATH}")
    for i, c in enumerate(cases):
        print(f"prompt {i}: {tok.decode(c['new_token_ids'])!r}")


def cmd_bench(runs: int, new_tokens: int, out: Path) -> None:
    """fp16 timing: model load time, then per prompt 1 cold run + `runs` warm runs.

    Note: prompt 0's cold run is the truly cold one (first kernels ever compiled in this process);
    prompt 1's cold run is cold only for its new shapes.
    """
    dtype = getattr(torch, CONFIG["bench_dtype"])
    t = time.perf_counter()
    tok, model, device = load(dtype)
    sync(device)
    load_s = time.perf_counter() - t

    meta = {
        "model_id": CONFIG["model_id"],
        "device": device,
        "hardware": "Apple M2, 16 GB unified memory" if device == "mps" else "cpu",
        "dtype": CONFIG["bench_dtype"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "new_tokens": new_tokens,
        "warm_runs": runs,
        "load_s": load_s,
        "weight_bytes": weight_bytes(model),
        "kv_bytes_per_token": kv_bytes_per_token(model),
        "peak_bandwidth_gbps_spec": PEAK_BANDWIDTH_GBPS,
    }

    prompts = []
    for i, prompt in enumerate(PROMPTS):
        prompt_tokens = len(tok(prompt)["input_ids"])
        results = []
        for r in range(runs + 1):
            results.append(time_one_run(model, tok, prompt, new_tokens, device))
            label = "cold" if r == 0 else f"warm {r}/{runs}"
            print(f"prompt {i} {label}: TTFT {results[-1]['ttft_s'] * 1e3:.0f} ms, total {results[-1]['total_s']:.2f} s", flush=True)
        prompts.append({"prompt_idx": i, "prompt_tokens": prompt_tokens, "runs": results})

    raw = {"meta": meta, "prompts": prompts}
    stats = analyze(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"raw": raw, "stats": stats}, indent=2))
    print(f"\nsaved {out}")
    print_summary(raw, stats)


def cmd_stats(out: Path) -> None:
    """Re-analyze previously saved raw timings (so analysis changes don't need a re-run)."""
    saved = json.loads(out.read_text())
    stats = analyze(saved["raw"])
    out.write_text(json.dumps({"raw": saved["raw"], "stats": stats}, indent=2))
    print_summary(saved["raw"], stats)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["golden", "bench", "stats"])
    p.add_argument("--runs", type=int, default=NUM_RUNS)
    p.add_argument("--new-tokens", type=int, default=BENCH_NEW_TOKENS)
    p.add_argument("--out", type=Path, default=PERF_PATH)
    p.add_argument("--golden-tokens", type=int, default=GOLDEN_NEW_TOKENS)
    args = p.parse_args()
    if args.mode == "golden":
        cmd_golden(args.golden_tokens)
    elif args.mode == "bench":
        cmd_bench(args.runs, args.new_tokens, args.out)
    else:
        cmd_stats(args.out)


if __name__ == "__main__":
    main()
