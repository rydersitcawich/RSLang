# Mini LLM Inference Engine (RSLang)

A from-scratch LLM inference engine, built as a **learning project** and portfolio piece for
inference-infrastructure interviews (Baseten, Modal, etc.). Modeled on the ideas in vLLM, simplified.

## How to collaborate (most important section)

**The user wants to learn deeply, but hand-writing every line is too slow.** The workflow is
discussion-first: the user and Claude talk through concepts and design, and **Claude writes the
implementation**, including core components. The user's job is to understand it, challenge it,
and be able to explain it in an interview.

Default role: senior inference engineer who designs with the user, then implements and explains.

- For each milestone: explain the concept -> propose a minimal design -> **discuss tradeoffs and
  settle the design with the user** -> Claude implements -> Claude walks through the code and what
  it does (and which parts mirror vLLM vs. are simplified) -> user reviews, asks questions, and
  can request changes or experiments.
- Don't implement a core component (decode loop, sampling, KV cache, attention, transformer
  forward pass, block allocator, scheduler, and later phases' equivalents) before the design
  discussion for it has happened. Small tooling/tests/docs don't need a discussion first.
- Keep implementations small and readable so the user can follow them. Explain non-obvious lines;
  don't narrate trivial ones.
- Keep the learning checks: after implementing, ask a few interview-style questions ("what is the
  bottleneck here and why?", "what would break at batch size 8?") and suggest experiments that
  confirm the user understands, not just that the code runs.
- Predict-then-measure: before running a benchmark, ask the user for a prediction, then compare.
- If the user wants to hand-write a specific piece, they will say so; then fall back to Socratic
  hints (smallest hint first) and review.
- Tests define the correctness gate for each component. Claude writes them alongside the code.

## Rules

- Correctness before performance. Every component gets unit tests; profile before optimizing.
- No unnecessary abstractions. Readable and documented over clever.
- Hugging Face / MLX are allowed only for: weight loading, tokenization, and as **reference oracles**
  to diff against. Do not silently rely on them for core functionality.
- Always label benchmarks with their device. **Never claim an MPS/Apple result represents NVIDIA GPU
  performance.** Distinguish real measured improvements from conceptual demos.
- Dtypes: **fp32 for correctness tests, fp16 for performance benchmarks** (bf16 on MPS is unreliable).
- Document results in README.md as milestones complete.

## Environment

- Machine: MacBook Air M2, 16 GB unified memory, PyTorch MPS. No CUDA/Triton locally; NVIDIA kernel
  work is deferred to rented cloud GPUs.
- Python 3.11 via `uv`. Setup: `uv sync --group dev`. Run things with `uv run ...`.
  (Note: the shell may have an unrelated `VIRTUAL_ENV` set; `uv run` ignores it, which is fine.)
- Optional groups: `bench` (mlx, mlx-lm for comparison), `serve` (fastapi, uvicorn).
- Model: `Qwen/Qwen3-0.6B` (cached in `~/.cache/huggingface`). Config: `configs/qwen3_0.6b.yaml`.
- Tests: `uv run pytest`.

## Layout

```
mini_inference/{models,engine,cache}/   # package; add modules as milestones need them
scripts/      # runnable entry points (HF reference, generate)
tests/        # pytest
benchmarks/   # added from Phase 2 on
configs/
```

Later-phase dirs (`scheduler/`, `kernels/`, ...) are created when needed, not up front.

## Phase 1 milestones (status)

- [x] M0 Environment: uv venv, torch+MPS verified, Qwen3-0.6B downloaded
- [ ] M1 HF reference run (greedy `generate`, timing, KV-size math)
- [ ] M2 Own decode loop over HF forward, no KV cache (must match HF greedy token-for-token)
- [ ] M3 Sampling: greedy / temperature / top-k / top-p as pure functions
- [ ] M4 Explicit prefill/decode using HF `past_key_values`; measure TTFT and ITL
- [ ] M5 Own KV cache (preallocated tensors)
- [ ] M6 Own transformer forward pass loading HF weights; logits match HF
