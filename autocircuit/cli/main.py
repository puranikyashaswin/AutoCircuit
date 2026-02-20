import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="autocircuit",
        description="Causal circuit discovery for transformer models",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the full discovery pipeline")
    run.add_argument("--model", default="gpt2")
    run.add_argument("--dataset", required=True, help="JSON file with clean/corrupted pairs")
    run.add_argument("--top_k", type=int, default=20)
    run.add_argument("--output", default="results")
    run.add_argument("--edge_threshold", type=float, default=0.05)
    run.add_argument("--example_index", type=int, default=0,
                     help="Which example to analyze (-1 = average all)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "run":
        _run(args)


def _run(args):
    t0 = time.time()

    # lazy imports - TransformerLens + torch are heavy; skip them on --help
    import torch
    from autocircuit.core.model_utils import load_model, tokenize, resolve_target_token
    from autocircuit.core.scoring import logit_diff
    from autocircuit.core.patching import cache_clean_run, corrupted_baseline
    from autocircuit.analysis.node_selection import score_all, select_top_k
    from autocircuit.analysis.edge_analysis import analyze_edges
    from autocircuit.visualization.graph_builder import build_graph, export_png

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found")
        sys.exit(1)

    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"Dataset: {dataset_path} ({len(dataset)} examples)")

    model = load_model(args.model)

    if args.example_index >= 0:
        if args.example_index >= len(dataset):
            print(f"Error: index {args.example_index} out of range")
            sys.exit(1)
        examples = [dataset[args.example_index]]
    else:
        examples = dataset

    ex = examples[0]
    print(f"  clean:     {ex['clean']}")
    print(f"  corrupted: {ex['corrupted']}")
    print(f"  target:    {ex['target']}")

    clean_tok = tokenize(model, ex["clean"])
    corrupt_tok = tokenize(model, ex["corrupted"])
    target_id = resolve_target_token(model, ex["target"])

    # baselines
    clean_logits, _ = cache_clean_run(model, clean_tok)
    corrupt_logits = corrupted_baseline(model, corrupt_tok)
    clean_ld = logit_diff(clean_logits, target_id)
    corrupt_ld = logit_diff(corrupt_logits, target_id)
    print(f"\n  clean logit diff:     {clean_ld:+.4f}")
    print(f"  corrupted logit diff: {corrupt_ld:+.4f}")
    print(f"  drop:                 {clean_ld - corrupt_ld:+.4f}")

    # score all components
    all_scores = score_all(model, clean_tok, corrupt_tok, target_id)
    top_k = select_top_k(all_scores, k=args.top_k)

    print(f"\nTop-{args.top_k} components:")
    for i, c in enumerate(top_k[:10]):
        print(f"  {i+1:2d}. {c['name']:18s} score={c['raw_score']:.4f}")
    if len(top_k) > 10:
        print(f"  ... and {len(top_k) - 10} more")

    # edge analysis
    edges = analyze_edges(model, clean_tok, corrupt_tok, target_id,
                          top_k, threshold=args.edge_threshold)
    print(f"\nTop edges:")
    for i, e in enumerate(edges[:10]):
        print(f"  {i+1:2d}. {e['source']:18s} -> {e['target']:18s} w={e['weight']:.4f}")
    if len(edges) > 10:
        print(f"  ... and {len(edges) - 10} more")

    # graph
    G = build_graph(top_k, edges)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_png(G, str(out_dir / "circuit_graph.png"))

    # save results
    results = {
        "model": args.model,
        "dataset": str(dataset_path),
        "example": ex,
        "clean_logit_diff": round(clean_ld, 4),
        "corrupted_logit_diff": round(corrupt_ld, 4),
        "score_drop": round(clean_ld - corrupt_ld, 4),
        "top_k_components": top_k,
        "edges": edges,
        "config": {"top_k": args.top_k, "edge_threshold": args.edge_threshold},
    }
    with open(out_dir / "scores.json", "w") as f:
        json.dump(results, f, indent=2)

    sorted_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))
    with open(out_dir / "all_component_scores.json", "w") as f:
        json.dump(sorted_scores, f, indent=2)

    print(f"\nDone in {time.time() - t0:.1f}s - results in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
