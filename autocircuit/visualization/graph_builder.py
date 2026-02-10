from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def build_graph(top_k_components, edges):
    G = nx.DiGraph()
    for c in top_k_components:
        G.add_node(c["name"], score=c["score"], raw_score=c["raw_score"],
                   component_type=c["type"], layer=c["layer"],
                   head=c.get("head", -1))
    for e in edges:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"],
                       weight=e["weight"],
                       normalized_weight=e.get("normalized_weight", 0.0))
    return G


def _layer_positions(G):
    """Groups nodes by transformer layer for a roughly hierarchical layout."""
    by_layer = {}
    for node in G.nodes():
        layer = G.nodes[node].get("layer", 0)
        by_layer.setdefault(layer, []).append(node)

    pos = {}
    for y, layer_num in enumerate(sorted(by_layer)):
        nodes = sorted(by_layer[layer_num])
        n = len(nodes)
        for x, node in enumerate(nodes):
            pos[node] = ((x - n/2 + 0.5) * 2.0, y * 2.0)
    return pos


def export_png(G, path, title="AutoCircuit — Causal Graph", figsize=(18, 14)):
    if not G.number_of_nodes():
        return path

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    pos = _layer_positions(G)

    # node colors: blue for attention heads, orange for MLPs
    colors, sizes = [], []
    for node in G.nodes():
        d = G.nodes[node]
        colors.append("#4A90D9" if d.get("component_type") == "attn_head" else "#E8913A")
        sizes.append(400 + d.get("score", 0.5) * 2100)

    widths = [0.5 + G.edges[u, v].get("normalized_weight", 0.5) * 3.5
              for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors="#333", linewidths=1.2, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555", width=widths,
                           alpha=0.5, arrows=True, arrowsize=15, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=15, min_target_margin=15)

    labels = {n: f"{n}\n({G.nodes[n].get('raw_score', 0):.2f})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                            font_weight="bold", font_color="#111")

    ax.legend(handles=[
        mpatches.Patch(facecolor="#4A90D9", edgecolor="#333", label="Attention Head"),
        mpatches.Patch(facecolor="#E8913A", edgecolor="#333", label="MLP Layer"),
    ], loc="upper left", fontsize=10, framealpha=0.9)

    ax.axis("off")
    plt.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Graph saved: {out}")
    return str(out.resolve())
