#!/usr/bin/env python3
"""
Dump raw ablation and sufficiency numbers per example to verify formulas
and explain the K=15 dip.
"""
import torch, json
from collections import defaultdict

from autocircuit.core.model_utils import load_model, tokenize, resolve_target_token
from autocircuit.core.scoring import logit_diff
from autocircuit.core.patching import cache_clean_run, corrupted_baseline
from autocircuit.analysis.node_selection import score_all, select_top_k
from autocircuit.analysis.circuit_validation import run_sufficiency, run_ablation


def main():
    model = load_model("gpt2", device="cpu")
    with open("autocircuit/data/ioi_task.json") as f:
        dataset = json.load(f)

    ks = [1, 3, 5, 10, 15]

    # ---- ABLATION deep-dive ----
    print("=" * 80)
    print("ABLATION RAW NUMBERS")
    print("=" * 80)
    print(f"\nFormula: drop_pct = (clean_score - ablated_score) / abs(clean_score) * 100")
    print(f"This is a ratio against the clean baseline, NOT against the clean-corrupt gap.\n")

    abl_results = {k: [] for k in ks}
    suf_results = {k: [] for k in ks}

    for idx, ex in enumerate(dataset):
        clean_tok = tokenize(model, ex["clean"])
        corrupt_tok = tokenize(model, ex["corrupted"])
        target_id = resolve_target_token(model, ex["target"])

        scores = score_all(model, clean_tok, corrupt_tok, target_id)

        print(f"\n--- Example {idx}: target='{ex['target']}' ---")
        for k in ks:
            top_k = select_top_k(scores, k=k)
            abl = run_ablation(model, clean_tok, target_id, top_k)
            suf = run_sufficiency(model, clean_tok, corrupt_tok, target_id, top_k)

            abl_results[k].append(abl)
            suf_results[k].append(suf)

            print(f"  K={k:2d}  ABL: clean={abl['clean_score']:+.4f}  "
                  f"ablated={abl['ablated_score']:+.4f}  "
                  f"drop={abl['drop']:+.4f}  drop_pct={abl['drop_pct']:.1f}%")

    # ---- ABLATION averages ----
    print("\n" + "=" * 80)
    print("ABLATION AVERAGES")
    print("=" * 80)
    n = len(dataset)
    for k in ks:
        avg_clean = sum(r["clean_score"] for r in abl_results[k]) / n
        avg_ablated = sum(r["ablated_score"] for r in abl_results[k]) / n
        avg_drop_pct = sum(r["drop_pct"] for r in abl_results[k]) / n
        # Also compute: what if we average the raw scores first, THEN compute pct?
        direct_pct = (avg_clean - avg_ablated) / abs(avg_clean) * 100
        print(f"  K={k:2d}  avg_clean={avg_clean:+.4f}  avg_ablated={avg_ablated:+.4f}  "
              f"avg_drop_pct(per-ex mean)={avg_drop_pct:.1f}%  "
              f"direct_pct(from avg scores)={direct_pct:.1f}%")

    # ---- SUFFICIENCY deep-dive for K=10 vs K=15 ----
    print("\n" + "=" * 80)
    print("SUFFICIENCY: K=10 vs K=15 per example")
    print("=" * 80)
    for idx in range(n):
        s10 = suf_results[10][idx]
        s15 = suf_results[15][idx]
        print(f"  Ex {idx}: K=10 recovery={s10['recovery_pct']:6.1f}%  "
              f"K=15 recovery={s15['recovery_pct']:6.1f}%  "
              f"diff={s15['recovery_pct'] - s10['recovery_pct']:+.1f}%  "
              f"(clean={s10['clean_score']:+.4f} corrupt={s10['corrupted_score']:+.4f} "
              f"restored10={s10['recovered_score']:+.4f} restored15={s15['recovered_score']:+.4f})")

    # ---- SUFFICIENCY averages ----
    print("\n" + "=" * 80)
    print("SUFFICIENCY AVERAGES")
    print("=" * 80)
    for k in ks:
        avg_rec = sum(r["recovery_pct"] for r in suf_results[k]) / n
        print(f"  K={k:2d}  avg_recovery={avg_rec:.1f}%")

    # ---- What components are in top-15 but not top-10? ----
    print("\n" + "=" * 80)
    print("COMPONENTS IN TOP-15 BUT NOT TOP-10 (per example)")
    print("=" * 80)
    for idx, ex in enumerate(dataset):
        clean_tok = tokenize(model, ex["clean"])
        corrupt_tok = tokenize(model, ex["corrupted"])
        target_id = resolve_target_token(model, ex["target"])
        scores = score_all(model, clean_tok, corrupt_tok, target_id)
        t10 = {c["name"] for c in select_top_k(scores, k=10)}
        t15 = select_top_k(scores, k=15)
        extras = [(c["name"], c["raw_score"]) for c in t15 if c["name"] not in t10]
        print(f"  Ex {idx}: extras at K=15: {extras}")


if __name__ == "__main__":
    main()
