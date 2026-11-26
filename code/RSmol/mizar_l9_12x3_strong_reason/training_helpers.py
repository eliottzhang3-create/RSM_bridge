"""Unchanged Mizar S1 loss/move/RNG/digest functions, copied in isolation."""
import hashlib,random
import numpy as np
import torch
import torch.nn.functional as F
from contract import PREFIX_TOKENS,ANSWER_TOKENS

def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)

def tensor_digest(items):
    result = hashlib.sha256()
    for name, tensor in items:
        result.update(name.encode("utf-8") + b"\0")
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()

def move_batch(batch, device):
    return {"audio1": batch["audio1"].to(device), "audio2": batch["audio2"].to(device),
            "input": {k: v.to(device) for k, v in batch["input"].items()},
            "answer": {k: v.to(device) for k, v in batch["answer"].items()}}

def answer_loss(output, inputs, ignore_index):
    logits = output.logits
    prefix_len = logits.shape[1] - inputs["answer"]["input_ids"].shape[1]
    if prefix_len != PREFIX_TOKENS or logits.shape[1] != PREFIX_TOKENS + ANSWER_TOKENS:
        raise ValueError(f"unexpected prefix/total sequence: {prefix_len}, {logits.shape[1]}")
    # Match Mizar's _answer_token_loss: the tokenizer attention mask and the
    # PAD target check jointly define valid answer targets. Prompt/audio
    # contribute context only; each rank averages its own valid answer tokens.
    logits = logits[:, prefix_len - 1:-1]
    answer = inputs["answer"]
    targets = answer["input_ids"]
    attention_mask = answer.get("attention_mask")
    if attention_mask is None or attention_mask.shape != targets.shape:
        raise ValueError("Mizar answer padding requires an attention_mask matching input_ids")
    valid = attention_mask.to(dtype=torch.bool) & targets.ne(ignore_index)
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=ignore_index
    )
    valid_count = valid.sum()
    if valid_count.item() == 0:
        raise ValueError("answer batch contains no valid target tokens")
    return (token_loss * valid).sum() / valid_count
