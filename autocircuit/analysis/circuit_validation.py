import torch
from transformer_lens.hook_points import HookPoint

from autocircuit.core.patching import (
    _attn_hook_name, _mlp_hook_name,
    cache_clean_run, corrupted_baseline, patch_components,
)
from autocircuit.core.scoring import logit_diff


def _zero_head_hook(layer, head):
    def hook(act, hook=None):
        act[:, :, head, :] = 0.0
        return act
    return hook

def _zero_mlp_hook(layer):
    def hook(act, hook=None):
        act[:] = 0.0
        return act
    return hook


def run_ablation(model, clean_tok, target_id, components, pos=-1):
    """Zeros out the given components during a clean forward pass.
    If the components are causally necessary, the logit diff should
    drop sharply - flipping from correct to incorrect prediction."""
    with torch.no_grad():
        clean_logits = model(clean_tok)
    clean_score = logit_diff(clean_logits, target_id, pos)

    hooks = []
    for c in components:
        if c["type"] == "attn_head":
            hooks.append((_attn_hook_name(c["layer"]),
                          _zero_head_hook(c["layer"], c["head"])))
        elif c["type"] == "mlp":
            hooks.append((_mlp_hook_name(c["layer"]),
                          _zero_mlp_hook(c["layer"])))

    with torch.no_grad():
        ablated_logits = model.run_with_hooks(clean_tok, fwd_hooks=hooks)
    ablated_score = logit_diff(ablated_logits, target_id, pos)

    drop = clean_score - ablated_score
    pct = (drop / abs(clean_score) * 100) if abs(clean_score) > 0.01 else 0.0
    return {
        "clean_score": round(clean_score, 4),
        "ablated_score": round(ablated_score, 4),
        "drop": round(drop, 4),
        "drop_pct": round(pct, 1),
    }


def run_sufficiency(model, clean_tok, corrupt_tok, target_id, components, pos=-1):
    """Restores only the given components from clean into corrupted.
    High recovery % means these components alone are sufficient to
    reproduce the model's correct IOI behavior."""
    clean_logits, clean_cache = cache_clean_run(model, clean_tok)
    clean_score = logit_diff(clean_logits, target_id, pos)
    corrupt_score = logit_diff(corrupted_baseline(model, corrupt_tok), target_id, pos)

    specs = []
    for c in components:
        s = {"type": c["type"], "layer": c["layer"]}
        if c["type"] == "attn_head":
            s["head"] = c["head"]
        specs.append(s)

    recovered = logit_diff(
        patch_components(model, corrupt_tok, clean_cache, specs),
        target_id, pos
    )

    total_drop = clean_score - corrupt_score
    recovery = recovered - corrupt_score
    pct = (recovery / total_drop * 100) if abs(total_drop) > 0.01 else 0.0
    return {
        "clean_score": round(clean_score, 4),
        "corrupted_score": round(corrupt_score, 4),
        "recovered_score": round(recovered, 4),
        "recovery_pct": round(pct, 1),
    }
