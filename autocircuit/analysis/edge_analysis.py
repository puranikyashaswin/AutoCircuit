from tqdm import tqdm

from autocircuit.core.patching import cache_clean_run, patch_components
from autocircuit.core.scoring import logit_diff


def _as_spec(comp):
    s = {"type": comp["type"], "layer": comp["layer"]}
    if comp["type"] == "attn_head":
        s["head"] = comp["head"]
    return s


def analyze_edges(model, clean_tok, corrupt_tok, target_id,
                  top_k_components, pos=-1, threshold=0.05):
    """Measures pairwise directional influence between top-K components.

    For each ordered pair (A, B):
      score_A  = logit diff with only A patched
      score_AB = logit diff with A and B patched together
      influence(A->B) = |score_A - score_AB|

    Pre-computes single-component scores to avoid redundant forward passes.
    With K=10, this runs K*(K-1)=90 pair passes instead of the ~24,000
    that exhaustive search over all 156 components would require."""

    n = len(top_k_components)
    if n < 2:
        return []

    _, clean_cache = cache_clean_run(model, clean_tok)

    # single-component scores (computed once, reused K-1 times each)
    print(f"Computing single-patch scores for {n} nodes...")
    single = {}
    for comp in tqdm(top_k_components, desc="single", leave=False):
        logits = patch_components(model, corrupt_tok, clean_cache, [_as_spec(comp)])
        single[comp["name"]] = logit_diff(logits, target_id, pos)

    print(f"Analyzing {n * (n-1)} directed pairs...")
    edges = []

    for i, a in enumerate(tqdm(top_k_components, desc="pairs", leave=False)):
        sa = single[a["name"]]
        spec_a = _as_spec(a)
        for j, b in enumerate(top_k_components):
            if i == j:
                continue
            logits_ab = patch_components(
                model, corrupt_tok, clean_cache, [spec_a, _as_spec(b)]
            )
            sab = logit_diff(logits_ab, target_id, pos)
            w = abs(sa - sab)
            if w >= threshold:
                edges.append({"source": a["name"], "target": b["name"], "weight": round(w, 4)})

    if edges:
        wmax = max(e["weight"] for e in edges)
        wmin = min(e["weight"] for e in edges)
        span = wmax - wmin if wmax != wmin else 1.0
        for e in edges:
            e["normalized_weight"] = round((e["weight"] - wmin) / span, 4)

    edges.sort(key=lambda e: e["weight"], reverse=True)
    print(f"Found {len(edges)} edges above threshold {threshold}")
    return edges
