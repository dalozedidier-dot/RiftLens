from src.rift_lens import build_graph, load_csv


def test_build_graph_returns_nodes():
    df = load_csv("tests/data/test_multi.csv")
    nodes, edges = build_graph(df, corr_threshold=0.1)
    assert len(nodes) >= 2
    assert isinstance(edges, list)
