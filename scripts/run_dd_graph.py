#!/usr/bin/env python3
from pathlib import Path
import argparse

from dd_graph_tool.dd_graph import run


def main() -> None:
    p = argparse.ArgumentParser(description="Run DD-Graph (wrapper)")
    p.add_argument("csv_path", type=str)
    p.add_argument("--corr_threshold", type=float, default=0.6)
    p.add_argument("--out", type=str, default="_ci_out/dd_graph")
    p.add_argument("--split_compare", action="store_true")
    args = p.parse_args()

    run(
        csv_path=Path(args.csv_path),
        corr_threshold=args.corr_threshold,
        output_dir=Path(args.out),
        split_compare=args.split_compare,
    )


if __name__ == "__main__":
    main()
