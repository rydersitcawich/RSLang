"""M1 correctness gate for scripts/hf_reference.py.

Run: uv run pytest tests/test_hf_reference.py -v

Model-dependent tests use fp32 (CLAUDE.md: fp32 for correctness) and share one loaded
model per session. Golden-file tests are skipped until `hf_reference.py golden` has been
run. The golden schema asserted here is a proposal; change both together if you change it:

    {
      "model_id": str,
      "dtype": "float32",
      "max_new_tokens": int,
      "cases": [{"prompt": str, "new_token_ids": [int, ...]}, ...]
    }
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
N = 16


def _import_script():
    spec = importlib.util.spec_from_file_location("hf_reference", ROOT / "scripts" / "hf_reference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


hf = _import_script()


def ids(out) -> list[int]:
    """Normalize list / [L] tensor / [1, L] tensor to a flat list of ints."""
    if isinstance(out, torch.Tensor):
        assert out.dim() in (1, 2), f"unexpected tensor shape {tuple(out.shape)}"
        if out.dim() == 2:
            assert out.shape[0] == 1, f"expected batch size 1, got {tuple(out.shape)}"
            out = out[0]
        return out.tolist()
    assert isinstance(out, list), f"expected list or tensor, got {type(out)}"
    if out and isinstance(out[0], list):
        assert len(out) == 1, "expected batch size 1"
        return out[0]
    return out


@pytest.fixture(scope="session")
def fp32():
    tok, model, device = hf.load(torch.float32)
    return tok, model, device


# ---------- plumbing ----------


def test_prompts_defined():
    assert len(hf.PROMPTS) >= 2, "need at least a short and a long prompt"


def test_prompts_differ_in_length(fp32):
    tok, _, _ = fp32
    lens = [len(tok(p)["input_ids"]) for p in hf.PROMPTS]
    assert max(lens) >= 4 * min(lens), f"prompt lengths {lens} too close to see prefill scaling"


def test_kv_bytes_per_token_matches_hand_derivation(fp32):
    """Hand math: 28 layers * 8 kv heads * 128 head_dim * 2 (K and V) * 2 bytes = 114,688 in fp16."""
    _, model, _ = fp32
    fp32_bytes = hf.kv_bytes_per_token(model)  # model is fp32 here
    assert fp32_bytes == 2 * 114_688, f"fp32 KV bytes/token: got {fp32_bytes}, expected {2 * 114_688}"
    assert fp32_bytes // 2 == 114_688  # i.e. fp16 would be the hand-derived number


# ---------- generate_greedy ----------


@pytest.mark.parametrize("prompt_idx", [0, 1])
def test_exact_length(fp32, prompt_idx):
    tok, model, device = fp32
    out = ids(hf.generate_greedy(model, tok, hf.PROMPTS[prompt_idx], N, device))
    plen = len(tok(hf.PROMPTS[prompt_idx])["input_ids"])
    assert len(out) == N, (
        f"got {len(out)} ids, expected exactly {N} new tokens "
        f"(prompt is {plen} tokens, so prompt+new would be {plen + N})"
    )


@pytest.mark.parametrize("prompt_idx", [0, 1])
def test_deterministic(fp32, prompt_idx):
    tok, model, device = fp32
    a = ids(hf.generate_greedy(model, tok, hf.PROMPTS[prompt_idx], N, device))
    b = ids(hf.generate_greedy(model, tok, hf.PROMPTS[prompt_idx], N, device))
    assert a == b, "greedy decoding gave different tokens on identical inputs"


def test_ids_are_valid_vocab_entries(fp32):
    tok, model, device = fp32
    out = ids(hf.generate_greedy(model, tok, hf.PROMPTS[0], N, device))
    vocab = model.config.vocab_size
    assert all(isinstance(i, int) and 0 <= i < vocab for i in out)


def test_different_prompts_give_different_outputs(fp32):
    tok, model, device = fp32
    a = ids(hf.generate_greedy(model, tok, hf.PROMPTS[0], N, device))
    b = ids(hf.generate_greedy(model, tok, hf.PROMPTS[1], N, device))
    assert a != b, "two very different prompts produced identical continuations; is the prompt being used?"


def test_longer_generation_extends_shorter(fp32):
    """Greedy is a pure function of the prefix, so the first N of a 2N run must equal the N run."""
    tok, model, device = fp32
    short = ids(hf.generate_greedy(model, tok, hf.PROMPTS[0], N, device))
    long = ids(hf.generate_greedy(model, tok, hf.PROMPTS[0], 2 * N, device))
    assert long[:N] == short


# ---------- golden file (gate for M2) ----------

GOLDEN = ROOT / "tests" / "golden" / "hf_greedy_fp32.json"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="run `hf_reference.py golden` first")


@needs_golden
def test_golden_schema():
    g = json.loads(GOLDEN.read_text())
    assert g["model_id"] == hf.CONFIG["model_id"]
    assert g["dtype"] == "float32", "golden tokens must come from fp32 (CLAUDE.md dtype rules)"
    assert isinstance(g["max_new_tokens"], int) and g["max_new_tokens"] > 0
    assert len(g["cases"]) == len(hf.PROMPTS)
    for case, prompt in zip(g["cases"], hf.PROMPTS):
        assert case["prompt"] == prompt, "golden prompts are stale vs PROMPTS in hf_reference.py"
        assert len(case["new_token_ids"]) == g["max_new_tokens"]
        assert all(isinstance(i, int) for i in case["new_token_ids"])


@needs_golden
def test_golden_reproduces(fp32):
    """HF reproduces its own golden file. If this fails, M2 has no stable target."""
    tok, model, device = fp32
    g = json.loads(GOLDEN.read_text())
    for case in g["cases"]:
        out = ids(hf.generate_greedy(model, tok, case["prompt"], g["max_new_tokens"], device))
        assert out == case["new_token_ids"]
