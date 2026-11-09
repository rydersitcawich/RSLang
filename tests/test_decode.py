"""M2 correctness gate: our no-cache decode loop must reproduce HF's greedy tokens exactly.

Run: uv run pytest tests/test_decode.py -v
"""

import json
from pathlib import Path

import pytest
import torch

from mini_inference.engine.decode import generate_no_cache

GOLDEN = Path(__file__).resolve().parent / "golden" / "hf_greedy_fp32.json"
pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="run `hf_reference.py golden` first")


def golden_cases():
    return json.loads(GOLDEN.read_text())["cases"] if GOLDEN.exists() else []


def prompt_tensor(case, device):
    return torch.tensor([case["prompt_token_ids"]], device=device)


def first_divergence(got: list[int], want: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return i
    return None if len(got) == len(want) else min(len(got), len(want))


@pytest.mark.parametrize("idx", range(len(golden_cases())))
def test_matches_golden(fp32, idx):
    _, model, device = fp32
    case = golden_cases()[idx]
    want = case["new_token_ids"]
    got = generate_no_cache(model, prompt_tensor(case, device), len(want))
    d = first_divergence(got, want)
    assert d is None, (
        f"prompt {idx}: first divergence at new-token index {d} "
        f"(got {got[d:d + 3]}, want {want[d:d + 3]}); a bug here usually shows up at index 0 or 1"
    )


def test_returns_plain_int_list_of_exact_length(fp32):
    _, model, device = fp32
    case = golden_cases()[0]
    got = generate_no_cache(model, prompt_tensor(case, device), 5)
    assert isinstance(got, list) and len(got) == 5
    assert all(type(i) is int for i in got)


def test_does_not_modify_input(fp32):
    _, model, device = fp32
    ids = prompt_tensor(golden_cases()[0], device)
    before = ids.clone()
    generate_no_cache(model, ids, 3)
    assert torch.equal(ids, before), "the caller's prompt tensor was mutated"


def test_stops_after_eos_and_includes_it(fp32):
    _, model, device = fp32
    case = golden_cases()[0]
    want = case["new_token_ids"]
    eos = want[3]
    expected = want[: want.index(eos) + 1]  # stops at the FIRST occurrence of that id
    got = generate_no_cache(model, prompt_tensor(case, device), len(want), eos_token_id=eos)
    assert got == expected


def test_eos_not_hit_runs_full_length(fp32):
    _, model, device = fp32
    case = golden_cases()[0]
    got = generate_no_cache(model, prompt_tensor(case, device), 8, eos_token_id=-1)  # never matches
    assert len(got) == 8


def test_rejects_batched_input(fp32):
    _, model, device = fp32
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
    with pytest.raises(AssertionError):
        generate_no_cache(model, ids, 1)
