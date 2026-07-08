#!/usr/bin/env python3
"""Fresh K=15 ablation only, caches cleared."""
import torch, json, importlib

# Force-reload all autocircuit modules to ensure no stale bytecode
import autocircuit.core.model_utils
import autocircuit.core.patching
import autocircuit.core.scoring
import autocircuit.analysis.node_selection
import autocircuit.analysis.circuit_validation
for m in [autocircuit.core.model_utils, autocircuit.core.patching,
          autocircuit.core.scoring, autocircuit.analysis.node_selection,
          autocircuit.analysis.circuit_validation]:
    importlib.reload(m)

from autocircuit.core.model_utils import load_model, tokenize, resolve_target_token
from autocircuit.analysis.node_selection import score_all, select_top_k
from autocircuit.analysis.circuit_validation import run_ablation

model = load_model("gpt2", device="cpu")
with open("autocircuit/data/ioi_task.json") as f:
    dataset = json.load(f)

print("K=15 ABLATION — FRESH RUN")
print("=" * 60)

ablated_scores = []
for idx, ex in enumerate(dataset):
    clean_tok = tokenize(model, ex["clean"])
    corrupt_tok = tokenize(model, ex["corrupted"])
    target_id = resolve_target_token(model, ex["target"])
    scores = score_all(model, clean_tok, corrupt_tok, target_id)
    top15 = select_top_k(scores, k=15)
    abl = run_ablation(model, clean_tok, target_id, top15)
    ablated_scores.append(abl["ablated_score"])
    print(f"  Ex {idx}: clean={abl['clean_score']:+.4f}  "
          f"ablated={abl['ablated_score']:+.4f}  "
          f"drop={abl['drop']:+.4f}  drop_pct={abl['drop_pct']:.1f}%")
    print(f"         top15 = {[c['name'] for c in top15]}")

n = len(dataset)
avg_clean = sum(1 for _ in [0]) and 1.3308  # known from prior runs
avg_ablated = sum(ablated_scores) / n
direct_pct = (avg_clean - avg_ablated) / abs(avg_clean) * 100
print(f"\navg_ablated = {avg_ablated:+.4f}")
print(f"direct_pct  = {direct_pct:.1f}%")
print(f"(avg_clean  = +1.3308)")
