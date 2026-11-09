"""Shared fixtures. One fp32 model is loaded per test session and reused by every test file."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def _import_script():
    spec = importlib.util.spec_from_file_location("hf_reference", ROOT / "scripts" / "hf_reference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def fp32():
    """(tokenizer, model, device) in fp32, per CLAUDE.md's dtype rules for correctness tests."""
    tok, model, device = _import_script().load(torch.float32)
    return tok, model, device
