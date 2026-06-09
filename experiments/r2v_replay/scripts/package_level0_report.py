#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    diagnostic_dir = Path(args.diagnostic_dir)
    figures_dir = Path(args.figures_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in list(diagnostic_dir.glob("*")) + list(figures_dir.glob("*")):
            if path.is_file():
                zf.write(path, arcname=path.name)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
