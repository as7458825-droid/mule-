"""network.py — NetworkX-based mule ring detection.

Builds a directed transaction graph and runs community detection to
surface tightly connected mule rings. Uses Louvain if available,
otherwise falls back to greedy modularity.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

LOG = logging.getLogger("muleshield.network")


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from transactions.

    Nodes are accounts (nameOrig -> nameDest). Edges carry total amount
    transferred, count, and the channel mix as attributes.
    """
    LOG.info("Building transaction graph from %d rows ...", len(df))
    G = nx.DiGraph()

    grouped = df.groupby(["nameOrig", "nameDest", "channel"])
    for (src, dst, ch), g in grouped:
        if G.has_edge(src, dst):
            G[src][dst]["amount"] += g["amount"].sum()
            G[src][dst]["count"] += len(g)
        else:
            G.add_edge(src, dst, amount=float(g["amount"].sum()),
                       count=int(len(g)), channel=ch)

    LOG.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def detect_rings(G: nx.DiGraph, top_n: int = 10) -> list[dict]:
    """Detect mule rings via Louvain community detection on the
    underlying undirected graph.

    Returns the top_n rings ranked by total in/out flow.
    """
    LOG.info("Running community detection ...")
    U = G.to_undirected()

    # Try Louvain (needs python-louvain), fall back to greedy_modularity
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(U, random_state=42)
    except ImportError:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(U))
        partition = {}
        for i, comm in enumerate(comms):
            for node in comm:
                partition[node] = i

    communities: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    rings = []
    for comm_id, members in communities.items():
        if len(members) < 3:
            continue
        subg = G.subgraph(members)
        total_amount = sum(d.get("amount", 0) for _, _, d in subg.edges(data=True))
        page_rank = nx.pagerank(subg) if subg.number_of_nodes() > 0 else {}
        top_hub = max(page_rank, key=page_rank.get) if page_rank else None
        rings.append({
            "community_id": comm_id,
            "n_accounts": len(members),
            "n_edges": subg.number_of_edges(),
            "total_flow": float(total_amount),
            "top_hub": top_hub,
            "members": members[:50],  # cap for log readability
        })

    rings.sort(key=lambda r: r["total_flow"], reverse=True)
    LOG.info("Detected %d rings (>= 3 accounts)", len(rings))
    return rings[:top_n]


def render_ring_html(rings: list[dict], out_path: Path) -> None:
    """Render a small HTML page with the top rings (for the dashboard)."""
    rows = []
    for r in rings:
        rows.append(
            f"<tr><td>{r['community_id']}</td><td>{r['n_accounts']}</td>"
            f"<td>{r['n_edges']}</td><td>{r['total_flow']:,.0f}</td>"
            f"<td>{r['top_hub']}</td></tr>"
        )
    html = f"""
<html><head><title>MuleShield — Mule Rings</title>
<style>body{{font-family:sans-serif}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:6px}}th{{background:#0B3D91;color:white}}</style>
</head><body>
<h2>Top Mule Rings Detected</h2>
<table><tr><th>Community</th><th>Accounts</th><th>Edges</th><th>Total Flow</th><th>Top Hub</th></tr>
{''.join(rows)}
</table></body></html>
"""
    out_path.write_text(html)
    LOG.info("Wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_paysim

    df = load_paysim("data", sample_n=50_000)
    G = build_graph(df)
    rings = detect_rings(G, top_n=5)
    for r in rings:
        print(r["community_id"], r["n_accounts"], r["n_edges"], f"{r['total_flow']:,.0f}", r["top_hub"])
    render_ring_html(rings, Path("reports/rings.html"))
