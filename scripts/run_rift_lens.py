#!/usr/bin/env python3
from pathlib import Path
import argparse

from riftlens.rift_lens import run


def main() -> None:
    p = argparse.ArgumentParser(description="Run RiftLens (wrapper)")
    p.add_argument("csv_path", type=str)
    p.add_argument("--corr-threshold", type=float, default=0.6)
    p.add_argument("--output-dir", type=str, default="outputs/riftlens")
    p.add_argument("--split-compare", action="store_true")
    args = p.parse_args()

    run(
        csv_path=Path(args.csv_path),
        corr_threshold=args.corr_threshold,
        output_dir=Path(args.output_dir),
        split_compare=args.split_compare,
    )


if __name__ == "__main__":
    main()
