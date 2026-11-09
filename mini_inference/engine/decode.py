"""M2: our own decode loop over the HF forward pass, with NO KV cache.

Every step re-runs the model over the whole sequence so far. That is wasteful on purpose: it is the
baseline that the KV cache (M4/M5) is measured against.

What is ours here: the loop, the choice of next token, stopping.
What is still HF's: `model(...)`, the transformer forward pass (replaced in M6).
"""

import torch


@torch.inference_mode()
def generate_no_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Greedy decode. `input_ids` is the prompt, shape [1, prompt_len], on the model's device.

    Returns only the new token ids. If `eos_token_id` is given, stops after emitting it (the EOS
    token is included in the result); otherwise always produces exactly `max_new_tokens`.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "single sequence only (batching comes later)"

    tokens = input_ids
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        # Forward over ALL tokens so far (no cache). logits_to_keep=1 makes the lm_head project only
        # the last position: we would throw the other T-1 rows away, and that is a T x vocab matmul.
        logits = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits  # [1, 1, vocab]

        # Pick the next token. Greedy for now; M3 replaces this one line with sampling.
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]

        tokens = torch.cat([tokens, next_token], dim=1)
        new_ids.append(next_token.item())  # .item() syncs with the GPU every step; fine for a baseline
        if eos_token_id is not None and new_ids[-1] == eos_token_id:
            break
    return new_ids
