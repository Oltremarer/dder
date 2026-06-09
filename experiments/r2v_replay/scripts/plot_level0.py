#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_replay.plotting import plot_level0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    diagnostic_dir = Path(args.diagnostic_dir)
    plot_level0(
        diagnostic_dir / "score_table.csv",
        diagnostic_dir / "composition_summary.json",
        args.output_dir,
    )
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
