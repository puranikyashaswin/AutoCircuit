#!/usr/bin/env python3
"""
Run the full pipeline on ALL 8 IOI examples and print per-example + averaged
results for verification against Readme.md claims (98.2% recovery at K=3,
99.6% search space reduction, averaged baselines +1.33/-2.40).
"""
import torch
import json
from collections import defaultdict
from pathlib import Path

from autocircuit.core.model_utils import load_model, tokenize, resolve_target_token
from autocircuit.core.scoring import logit_diff
from autocircuit.core.patching import cache_clean_run, corrupted_baseline
from autocircuit.analysis.node_selection import score_all, select_top_k
from autocircuit.analysis.circuit_validation import run_sufficiency, run_ablation
from autocircuit.analysis.edge_analysis import analyze_edges


def main():
    device = "cpu"
    model = load_model("gpt2", device=device)
    with open("autocircuit/data/ioi_task.json") as f:
        dataset = json.load(f)

    n = len(dataset)
    total_attn = model.cfg.n_layers * model.cfg.n_heads
    total_mlp = model.cfg.n_layers
    total_components = total_attn + total_mlp
    print(f"Components: {total_attn} attention heads + {total_mlp} MLPs = {total_components}")
    print(f"Dataset: {n} examples")
    print("=" * 70)

    # Accumulators
    all_clean_scores = []
    all_corrupt_scores = []
    # component_name -> list of scores across examples
    component_scores_all = defaultdict(list)
    # k -> list of (recovery_pct, drop_pct, ablated_score) across examples
    ks = [1, 3, 5, 10, 15]
    k_results = {k: {"recovery": [], "drop": [], "abl_score": []} for k in ks}

    for idx, ex in enumerate(dataset):
        print(f"\n--- Example {idx} ---")
        print(f"  clean:     {ex['clean']}")
        print(f"  corrupted: {ex['corrupted']}")
        print(f"  target:    {ex['target']}")

        clean_tok = tokenize(model, ex["clean"])
        corrupt_tok = tokenize(model, ex["corrupted"])
        target_id = resolve_target_token(model, ex["target"])

        # Baselines
        clean_logits, _ = cache_clean_run(model, clean_tok)
        corrupt_logits = corrupted_baseline(model, corrupt_tok)
        clean_score = logit_diff(clean_logits, target_id)
        corrupt_score = logit_diff(corrupt_logits, target_id)
        all_clean_scores.append(clean_score)
        all_corrupt_scores.append(corrupt_score)
        print(f"  Clean logit diff: {clean_score:+.4f}")
        print(f"  Corrupted logit diff: {corrupt_score:+.4f}")
        print(f"  Drop: {clean_score - corrupt_score:+.4f}")

        # Score all components
        scores = score_all(model, clean_tok, corrupt_tok, target_id)
        for name, val in scores.items():
            component_scores_all[name].append(val)

        # Sufficiency / ablation per K
        for k in ks:
            top_k_ids = select_top_k(scores, k=k)
            suff = run_sufficiency(model, clean_tok, corrupt_tok, target_id, top_k_ids)
            abl = run_ablation(model, clean_tok, target_id, top_k_ids)
            k_results[k]["recovery"].append(suff["recovery_pct"])
            k_results[k]["drop"].append(abl["drop_pct"])
            k_results[k]["abl_score"].append(abl["ablated_score"])
            print(f"  K={k:02d}  recovery={suff['recovery_pct']:.1f}%  "
                  f"ablation_drop={abl['drop_pct']:.1f}%  "
                  f"abl_score={abl['ablated_score']:+.4f}")

    # ===== AVERAGED RESULTS =====
    print("\n" + "=" * 70)
    print(f"AVERAGED RESULTS (n={n})")
    print("=" * 70)

    avg_clean = sum(all_clean_scores) / n
    avg_corrupt = sum(all_corrupt_scores) / n
    avg_drop = avg_clean - avg_corrupt
    print(f"\nBaselines:")
    print(f"  Avg clean logit diff:     {avg_clean:+.2f}  (README: +1.33)")
    print(f"  Avg corrupted logit diff: {avg_corrupt:+.2f}  (README: -2.40)")
    print(f"  Avg drop:                 {avg_drop:+.2f}  (README: +3.73)")

    # Averaged top-15 component scores
    avg_scores = {name: sum(vals) / len(vals) for name, vals in component_scores_all.items()}
    avg_sorted = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 15 components (averaged over {n} examples):")
    for i, (name, score) in enumerate(avg_sorted[:15], 1):
        print(f"  {i:2d}. {name:18s} score = {score:.3f}")

    # Averaged sufficiency / ablation
    print(f"\nSufficiency (averaged over {n} examples):")
    print(f"  {'K':>3s}  {'Avg Recovery':>14s}  {'README':>8s}")
    readme_recovery = {1: "95.4%", 3: "98.2%", 5: "102.5%", 10: "103.7%", 15: "103.1%"}
    for k in ks:
        avg_rec = sum(k_results[k]["recovery"]) / n
        readme_val = readme_recovery.get(k, "")
        print(f"  K={k:02d}  recovery={avg_rec:6.1f}%     ({readme_val})")

    print(f"\nAblation (averaged over {n} examples):")
    print(f"  {'K':>3s}  {'Avg Drop% (mean)':>17s}  {'Group Drop% (direct)':>20s}  {'Avg Abl Score':>14s}  {'README':>8s}")
    readme_ablation = {1: "416%", 3: "537%", 5: "543%", 10: "624%", 15: "659%"}
    for k in ks:
        avg_drop_k = sum(k_results[k]["drop"]) / n
        avg_abl = sum(k_results[k]["abl_score"]) / n
        direct_drop = (avg_clean - avg_abl) / abs(avg_clean) * 100
        readme_val = readme_ablation.get(k, "")
        print(f"  K={k:02d}  drop_mean={avg_drop_k:6.1f}%   drop_direct={direct_drop:9.1f}%   abl_score={avg_abl:+.2f}     ({readme_val})")
    print("\nNote: 'Group Drop% (direct)' is computed from the averaged scores, which is robust against")
    print("      near-zero clean scores on individual prompts. 'Avg Drop% (mean)' is skewed by outliers.")

    # Edge analysis
    all_n = len(avg_scores)
    exhaustive = all_n * (all_n - 1)
    pairwise = 10 * 9
    reduction = 1 - pairwise / exhaustive
    print(f"\nEdge analysis (K=10):")
    print(f"  Pairwise edges: {pairwise}")
    print(f"  Exhaustive:     {exhaustive}")
    print(f"  Reduction:      {reduction:.4f} = {reduction * 100:.1f}%  (README: 99.6%)")


if __name__ == "__main__":
    main()
