import torch


def logit_diff(logits, target_id, pos=-1):
    """Logit of the target token minus the best competing token at `pos`.
    Positive = model prefers target, negative = something else wins."""
    if logits.dim() == 3:
        logits = logits[0]

    row = logits[pos]
    target_val = row[target_id].item()

    # mask the target out so max() gives us the runner-up
    masked = row.clone()
    masked[target_id] = float("-inf")
    runner_up = masked.max().item()

    return target_val - runner_up


def logit_diff_batch(logits, target_ids, positions):
    """Mean logit diff across a batch — used when averaging over multiple
    IOI examples to reduce per-prompt noise."""
    total = 0.0
    n = logits.shape[0]
    for i in range(n):
        row = logits[i, positions[i]]
        t = row[target_ids[i]].item()
        masked = row.clone()
        masked[target_ids[i]] = float("-inf")
        total += t - masked.max().item()
    return total / n
