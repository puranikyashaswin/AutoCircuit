#!/usr/bin/env python3
"""
Diagnostic: investigate K=1 sufficiency anomaly on examples 6 and 7.
Print raw scores at every step to find the bug.
"""
import torch
import json

from autocircuit.core.model_utils import load_model, tokenize, resolve_target_token
from autocircuit.core.scoring import logit_diff
from autocircuit.core.patching import cache_clean_run, corrupted_baseline, patch_components
from autocircuit.analysis.node_selection import score_all, select_top_k, parse_label


def diagnose_example(model, ex, idx):
    print(f"\n{'='*70}")
    print(f"EXAMPLE {idx}")
    print(f"  clean:     {ex['clean']}")
    print(f"  corrupted: {ex['corrupted']}")
    print(f"  target:    {ex['target']}")

    clean_tok = tokenize(model, ex["clean"])
    corrupt_tok = tokenize(model, ex["corrupted"])
    target_id = resolve_target_token(model, ex["target"])
    print(f"  target_id: {target_id}")

    # Baselines
    clean_logits, clean_cache = cache_clean_run(model, clean_tok)
    corrupt_logits = corrupted_baseline(model, corrupt_tok)
    clean_score = logit_diff(clean_logits, target_id)
    corrupt_score = logit_diff(corrupt_logits, target_id)
    total_drop = clean_score - corrupt_score
    print(f"  clean_score:   {clean_score:+.4f}")
    print(f"  corrupt_score: {corrupt_score:+.4f}")
    print(f"  total_drop:    {total_drop:+.4f}")

    # Score all components for this example
    scores = score_all(model, clean_tok, corrupt_tok, target_id)

    # What does select_top_k return for K=1?
    top1 = select_top_k(scores, k=1)
    print(f"\n  select_top_k(k=1) returns:")
    for c in top1:
        print(f"    {c}")

    # What is the actual #1 component by raw score?
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 components by raw score:")
    for i, (name, raw) in enumerate(ranked[:5], 1):
        print(f"    {i}. {name:18s} raw={raw:.4f}")

    # Now do the sufficiency test for K=1 manually
    print(f"\n  --- K=1 sufficiency (manual) ---")
    comp_name = top1[0]["name"]
    comp_dict = parse_label(comp_name)
    specs = [{"type": comp_dict["type"], "layer": comp_dict["layer"]}]
    if comp_dict["type"] == "attn_head":
        specs[0]["head"] = comp_dict["head"]

    print(f"  Patching component: {comp_name}")
    print(f"  Spec: {specs}")

    restored_logits = patch_components(model, corrupt_tok, clean_cache, specs)
    restored_score = logit_diff(restored_logits, target_id)

    recovery = restored_score - corrupt_score
    pct = (recovery / total_drop * 100) if abs(total_drop) > 0.01 else 0.0

    print(f"  restored_score:  {restored_score:+.4f}")
    print(f"  recovery (abs):  {recovery:+.4f}")
    print(f"  recovery_pct:    {pct:.1f}%")
    print(f"  Interpretation:  restored ({restored_score:+.4f}) vs corrupt ({corrupt_score:+.4f})")
    if restored_score < corrupt_score:
        print(f"  *** RESTORED IS WORSE THAN CORRUPT ***")
        print(f"      This means patching {comp_name} alone into the corrupted run")
        print(f"      actually makes things worse, not better.")

    # Also test: what if we patch mlp_L0 specifically (the claimed K=1 winner)?
    if comp_name != "mlp_L0":
        print(f"\n  --- Also testing mlp_L0 (README's claimed K=1 winner) ---")
        mlp0_spec = [{"type": "mlp", "layer": 0}]
        mlp0_logits = patch_components(model, corrupt_tok, clean_cache, mlp0_spec)
        mlp0_score = logit_diff(mlp0_logits, target_id)
        mlp0_recovery = mlp0_score - corrupt_score
        mlp0_pct = (mlp0_recovery / total_drop * 100) if abs(total_drop) > 0.01 else 0.0
        print(f"  mlp_L0 raw importance score: {scores.get('mlp_L0', 0):.4f}")
        print(f"  mlp_L0 restored_score: {mlp0_score:+.4f}")
        print(f"  mlp_L0 recovery_pct:   {mlp0_pct:.1f}%")

    # Cross-check: what does score_all measure vs what sufficiency measures?
    # score_all uses abs(patched_score - baseline) where baseline = corrupt_score
    # So the "score" IS the same as |recovery| for that single component.
    print(f"\n  --- Cross-check: score_all vs sufficiency for {comp_name} ---")
    print(f"  score_all raw score:          {scores[comp_name]:.4f}")
    print(f"  |restored - corrupt| (= |recovery|): {abs(recovery):.4f}")
    # These should match. If they don't, there's a caching bug.

    # K=3 for comparison (this should be fine)
    print(f"\n  --- K=3 sufficiency ---")
    top3 = select_top_k(scores, k=3)
    specs3 = []
    for c in top3:
        s = {"type": c["type"], "layer": c["layer"]}
        if c["type"] == "attn_head":
            s["head"] = c["head"]
        specs3.append(s)
    print(f"  Components: {[c['name'] for c in top3]}")
    restored3 = logit_diff(
        patch_components(model, corrupt_tok, clean_cache, specs3),
        target_id
    )
    rec3 = (restored3 - corrupt_score) / total_drop * 100
    print(f"  restored_score: {restored3:+.4f}, recovery: {rec3:.1f}%")


def main():
    device = "cpu"
    model = load_model("gpt2", device=device)
    with open("autocircuit/data/ioi_task.json") as f:
        dataset = json.load(f)

    # Focus on examples 6 and 7 (the negative outliers), plus example 0 as control
    for idx in [0, 6, 7]:
        diagnose_example(model, dataset[idx], idx)

    # Also show the raw per-example K=1 selected component for ALL 8 examples
    print(f"\n{'='*70}")
    print("SUMMARY: K=1 selected component per example")
    print("="*70)
    for idx, ex in enumerate(dataset):
        clean_tok = tokenize(model, ex["clean"])
        corrupt_tok = tokenize(model, ex["corrupted"])
        target_id = resolve_target_token(model, ex["target"])
        scores = score_all(model, clean_tok, corrupt_tok, target_id)
        top1 = select_top_k(scores, k=1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"  Ex {idx}: select_top_k selects '{top1[0]['name']}' "
              f"(raw={top1[0]['raw_score']:.4f}), "
              f"actual #1 by score: '{ranked[0][0]}' (raw={ranked[0][1]:.4f})")


if __name__ == "__main__":
    main()
