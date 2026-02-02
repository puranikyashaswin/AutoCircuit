import torch
from tqdm import tqdm

from autocircuit.core.patching import (
    cache_clean_run, corrupted_baseline, patch_head, patch_mlp,
)
from autocircuit.core.scoring import logit_diff


def component_label(ctype, layer, head=-1):
    if ctype == "attn_head":
        return f"attn_L{layer}_H{head}"
    return f"mlp_L{layer}"


def parse_label(name):
    parts = name.split("_")
    if parts[0] == "attn":
        return {"type": "attn_head", "layer": int(parts[1][1:]), "head": int(parts[2][1:])}
    return {"type": "mlp", "layer": int(parts[1][1:])}


def score_all(model, clean_tok, corrupt_tok, target_id, pos=-1):
    """Scores every attention head and MLP layer by patching each one
    from clean into corrupted and measuring how much the logit diff recovers.

    Importance(C) = |patched_score - corrupted_baseline_score|

    High score => restoring that component's clean activation substantially
    recovers the model's correct behavior on the IOI task."""

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    _, clean_cache = cache_clean_run(model, clean_tok)
    baseline = logit_diff(corrupted_baseline(model, corrupt_tok), target_id, pos)

    scores = {}
    total = n_layers * n_heads + n_layers
    print(f"Scoring {total} components ({n_layers}x{n_heads} heads + {n_layers} MLPs)...")

    for layer in tqdm(range(n_layers), desc="heads", leave=False):
        for head in range(n_heads):
            patched = logit_diff(
                patch_head(model, corrupt_tok, clean_cache, layer, head),
                target_id, pos
            )
            scores[component_label("attn_head", layer, head)] = abs(patched - baseline)

    for layer in tqdm(range(n_layers), desc="mlps", leave=False):
        patched = logit_diff(
            patch_mlp(model, corrupt_tok, clean_cache, layer),
            target_id, pos
        )
        scores[component_label("mlp", layer)] = abs(patched - baseline)

    return scores


def select_top_k(scores, k=20):
    """Returns the top-k components sorted by importance, with scores
    normalized to [0, 1] for visualization."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    if not ranked:
        return []

    hi, lo = ranked[0][1], ranked[-1][1]
    span = hi - lo if hi != lo else 1.0

    out = []
    for name, raw in ranked:
        comp = parse_label(name)
        comp["name"] = name
        comp["score"] = round((raw - lo) / span, 4)
        comp["raw_score"] = round(raw, 4)
        out.append(comp)
    return out
