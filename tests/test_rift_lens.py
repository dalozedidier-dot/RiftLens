from pathlib import Path

from riftlens.rift_lens import build_graph, load_csv


def test_build_graph_returns_graph():
    repo_root = Path(__file__).resolve().parents[1]
    df = load_csv(repo_root / "tests" / "data" / "test_multi.csv")
    G = build_graph(df, corr_threshold=0.1)
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 0
