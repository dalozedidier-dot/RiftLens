from pathlib import Path
import json

from riftlens.rift_lens import run


def test_riftlens_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "multi.csv"
    out_dir = tmp_path / "riftlens_out"

    res = run(csv_path=csv_path, corr_threshold=0.5, output_dir=out_dir, split_compare=True)

    assert (out_dir / "coherence_graph.png").exists()
    assert (out_dir / "graph_report.json").exists()
    assert (out_dir / "coherence_graph.gexf").exists()

    report = json.loads((out_dir / "graph_report.json").read_text(encoding="utf-8"))
    assert report["n_nodes"] >= 2
    assert "report" in res

    # split_compare outputs
    assert (out_dir / "graph_changes.json").exists()
    assert (out_dir / "coherence_graph_window1.png").exists()
    assert (out_dir / "coherence_graph_window2.png").exists()
