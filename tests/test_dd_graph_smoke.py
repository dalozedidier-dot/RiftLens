from pathlib import Path
import json
import subprocess
import sys


def test_dd_graph_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "test_data" / "multi.csv"
    out_dir = tmp_path / "dd_graph_out"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_dd_graph.py"),
        str(csv_path),
        "--corr_threshold",
        "0.5",
        "--out",
        str(out_dir),
        "--split_compare",
    ]
    subprocess.check_call(cmd)

    assert (out_dir / "coherence_graph.png").exists()
    assert (out_dir / "graph_report.json").exists()
    assert (out_dir / "coherence_graph.gexf").exists()

    report = json.loads((out_dir / "graph_report.json").read_text(encoding="utf-8"))
    assert report["n_nodes"] >= 2

    # split_compare outputs
    assert (out_dir / "graph_changes.json").exists()
    assert (out_dir / "coherence_graph_window1.png").exists()
    assert (out_dir / "coherence_graph_window2.png").exists()
