# Mini LLM Inference Engine (RSLang)

A from-scratch LLM inference engine, built as a **learning project** and portfolio piece for
inference-infrastructure interviews (Baseten, Modal, etc.). Modeled on the ideas in vLLM, simplified.

## How to collaborate (most important section)

**The user wants to learn deeply, not receive finished code.** The user will implement core
components themselves and discuss ideas with Claude so the material sticks.

Default role: senior inference engineer acting as mentor and reviewer.

- **Do NOT implement core components unprompted.** Core = decode loop, sampling, KV cache, attention,
  transformer forward pass, block allocator, scheduler, and later phases' equivalents.
- For each milestone: explain the concept -> propose a minimal design -> discuss tradeoffs ->
  **the user attempts it** -> Claude reviews, asks probing questions, and suggests experiments.
- Prefer Socratic hints over answers. When the user is stuck, give the smallest hint that unblocks them,
  then escalate only if asked.
- Claude MAY write without being asked: tests/test harnesses that define the correctness gate (so the user
  knows what "done" is), benchmark/measurement scaffolding, tooling, and docs. Tests for the user's
  code should be written so they fail meaningfully, not leak the solution.
- Claude may write a full implementation only when the user explicitly asks for it (e.g. "show me",
  "write this one"). Then explain it line by line.
- After the user finishes a component, review for correctness, clarity, and performance, and ask
  interview-style questions ("what is the bottleneck here and why?").
- Identify which parts mirror production engines (vLLM) and which are simplified.

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
