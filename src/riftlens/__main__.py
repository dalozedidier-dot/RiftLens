from __future__ import annotations

import argparse
from pathlib import Path

from .rift_lens import run


def main() -> None:
    p = argparse.ArgumentParser(description="RiftLens: observation descriptive en graphe")
    p.add_argument("csv_path", type=str, help="Chemin vers un CSV multivarié")
    p.add_argument("--corr-threshold", type=float, default=0.6, help="Seuil de corrélation pour créer une arête")
    p.add_argument("--output-dir", type=str, default="outputs/riftlens", help="Dossier de sortie")
    p.add_argument("--split-compare", action="store_true", help="Compare deux fenêtres (moitié 1 vs moitié 2)")
    args = p.parse_args()

    run(
        csv_path=Path(args.csv_path),
        corr_threshold=float(args.corr_threshold),
        output_dir=Path(args.output_dir),
        split_compare=bool(args.split_compare),
    )


if __name__ == "__main__":
    main()
