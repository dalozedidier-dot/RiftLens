"""
DD-Graph - Prototype v0.1
Observation descriptive de cohérence multivariée via graphes
Ne prédit rien, n'explique pas pourquoi, constate seulement
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path: Path) -> pd.DataFrame:
    """Charge un CSV et conserve uniquement les colonnes numériques.

    Si la première colonne est une colonne temporelle classique (date/time/timestamp),
    elle est utilisée comme index. Le parsing reste volontairement léger.
    """
    df = pd.read_csv(path)

    if df.shape[1] == 0:
        return df

    first = df.columns[0]
    if str(first).lower() in ["date", "time", "timestamp"]:
        df[first] = pd.to_datetime(df[first], errors="coerce")
        df = df.set_index(first)

    df = df.select_dtypes(include=[np.number])
    df = df.dropna(how="all")
    return df


def coherence_score_series(series: pd.Series) -> float:
    """Score descriptif simple de cohérence interne.

    Convention:
    - 0.0 : très faible cohérence descriptive
    - 1.0 : forte cohérence descriptive
    """
    series = series.dropna()
    if len(series) < 3:
        return 0.0

    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        return 1.0

    roll_std = series.rolling(5, min_periods=1).std().mean()
    roll_std = float(roll_std) if np.isfinite(roll_std) else std

    autocorr = series.autocorr(lag=1)
    autocorr = float(autocorr) if np.isfinite(autocorr) else 0.0

    score = 1.0 - (roll_std / (std + 1e-12)) + autocorr
    score = float(np.clip(score, 0.0, 1.0))
    return round(score, 4)


def build_graph(df: pd.DataFrame, corr_threshold: float = 0.6) -> nx.Graph:
    """Construit un graphe descriptif: noeuds = colonnes, aretes = cohérence entre colonnes."""
    G = nx.Graph()

    for col in df.columns:
        score = coherence_score_series(df[col])
        G.add_node(col, score=score, size=300 + score * 700)

    corr_matrix = df.corr(numeric_only=True).abs()
    cols = list(df.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            corr = float(corr_matrix.loc[c1, c2])
            if np.isfinite(corr) and corr >= corr_threshold:
                G.add_edge(c1, c2, weight=corr, label=f"{corr:.2f}")

    return G


def detect_graph_changes(G1: nx.Graph, G2: nx.Graph, node_delta_threshold: float = 0.05) -> Dict[str, Any]:
    """Compare deux graphes (par exemple deux fenêtres temporelles) de façon descriptive."""
    e1 = set((min(u, v), max(u, v)) for (u, v) in G1.edges())
    e2 = set((min(u, v), max(u, v)) for (u, v) in G2.edges())

    added = sorted(list(e2 - e1))
    removed = sorted(list(e1 - e2))

    node_changes: Dict[str, Any] = {}
    for node in set(G1.nodes) & set(G2.nodes):
        s1 = float(G1.nodes[node].get("score", 0.0))
        s2 = float(G2.nodes[node].get("score", 0.0))
        if abs(s1 - s2) > node_delta_threshold:
            node_changes[node] = {"before": s1, "after": s2, "delta": round(s2 - s1, 6)}

    return {
        "added_edges": added,
        "removed_edges": removed,
        "node_score_changes": node_changes,
    }


def graph_to_report(G: nx.Graph) -> Dict[str, Any]:
    """Export JSON explicite: noeuds + aretes + attributs."""
    nodes = []
    for n, attrs in G.nodes(data=True):
        clean = {}
        for k, v in attrs.items():
            if isinstance(v, (int, float, np.number)):
                clean[k] = float(v)
            else:
                clean[k] = v
        nodes.append({"id": n, **clean})

    edges = []
    for u, v, attrs in G.edges(data=True):
        clean = {}
        for k, v0 in attrs.items():
            if isinstance(v0, (int, float, np.number)):
                clean[k] = float(v0)
            else:
                clean[k] = v0
        edges.append({"source": u, "target": v, **clean})

    return {"nodes": nodes, "edges": edges, "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}


def visualize_graph(G: nx.Graph, output_path: Path, title: str = "Graphe descriptif de cohérence") -> None:
    """Visualisation simple et descriptive."""
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [G.nodes[n].get("size", 300) for n in G.nodes]
    node_colors = [G.nodes[n].get("score", 0.0) for n in G.nodes]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.9
    )

    widths = [float(d.get("weight", 0.0)) * 3.0 for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="gray", alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(title)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def split_windows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe simple en deux fenêtres successives (moitié 1, moitié 2)."""
    n = len(df)
    mid = max(1, n // 2)
    return df.iloc[:mid], df.iloc[mid:]


def run(csv_path: Path, corr_threshold: float, output_dir: Path, split_compare: bool = False) -> Dict[str, Any]:
    """API minimale pour greffe: run() retourne un dict descriptif (report, changes optionnel)."""
    output_dir.mkdir(exist_ok=True, parents=True)
    df = load_csv(csv_path)

    if df is None or df.empty or len(df.columns) < 2:
        raise ValueError("Au moins 2 colonnes numériques non vides sont requises")

    G = build_graph(df, corr_threshold)

    nx.write_gexf(G, output_dir / "coherence_graph.gexf")
    report = graph_to_report(G)
    (output_dir / "graph_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    visualize_graph(G, output_dir / "coherence_graph.png")

    result: Dict[str, Any] = {"report": report}

    if split_compare and len(df) >= 6:
        w1, w2 = split_windows(df)
        G1 = build_graph(w1, corr_threshold)
        G2 = build_graph(w2, corr_threshold)
        changes = detect_graph_changes(G1, G2)
        (output_dir / "graph_changes.json").write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
        visualize_graph(G1, output_dir / "coherence_graph_window1.png", title="Fenêtre 1")
        visualize_graph(G2, output_dir / "coherence_graph_window2.png", title="Fenêtre 2")
        result["changes"] = changes

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DD-Graph: observation descriptive en graphe")
    parser.add_argument("csv_path", type=str, help="Chemin vers un CSV multivarié")
    parser.add_argument("--corr_threshold", type=float, default=0.6, help="Seuil de corrélation pour créer une arête")
    parser.add_argument("--output_dir", type=str, default="dd_graph_output", help="Dossier de sortie")
    parser.add_argument("--split_compare", action="store_true", help="Compare deux fenêtres (première moitié vs seconde moitié)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    run(csv_path=csv_path, corr_threshold=args.corr_threshold, output_dir=output_dir, split_compare=args.split_compare)
    print(f"Résultats dans: {output_dir}")


if __name__ == "__main__":
    main()
