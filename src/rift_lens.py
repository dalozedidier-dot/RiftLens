import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("riftlens")


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    corr: float


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_graph(df: pd.DataFrame, corr_threshold: float) -> Tuple[List[str], List[Edge]]:
    numeric = df.select_dtypes(include=["number"])
    cols = list(numeric.columns)
    if len(cols) < 2:
        return cols, []
    corr = numeric.corr(numeric_only=True)
    edges: List[Edge] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = float(corr.loc[a, b])
            if pd.notna(c) and abs(c) >= corr_threshold:
                edges.append(Edge(a=a, b=b, corr=c))
    return cols, edges


def visualize_graph(nodes: List[str], edges: List[Edge], out_png: str) -> bool:
    try:
        import networkx as nx  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning("Visualization skipped: %s", e)
        return False

    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e in edges:
        g.add_edge(e.a, e.b, weight=abs(e.corr), corr=e.corr)

    plt.figure()
    pos = nx.spring_layout(g, seed=42)
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    return True


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="RiftLens: descriptive coherence graph")
    ap.add_argument("csv", help="Input CSV")
    ap.add_argument("--corr-threshold", type=float, default=0.6)
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_csv(args.csv)
    nodes, edges = build_graph(df, args.corr_threshold)

    report = {
        "tool": "RiftLens",
        "version": "0.1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": args.csv, "rows": int(len(df)), "columns": list(df.columns)},
        "params": {"corr_threshold": args.corr_threshold},
        "graph": {"nodes": nodes, "edges": [{"a": e.a, "b": e.b, "corr": e.corr} for e in edges]},
    }

    out_report = os.path.join(args.output_dir, "graph_report.json")
    write_json(out_report, report)

    out_png = os.path.join(args.output_dir, "coherence_graph.png")
    visualize_graph(nodes, edges, out_png)

    logger.info("Wrote %s", out_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
