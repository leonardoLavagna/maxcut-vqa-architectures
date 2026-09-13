"""Run or resume the frozen benchmark reported in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

from qaoa_structure.runner import run_benchmark


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "config" / "final.yaml"
OUTPUT_DIR = ROOT / "results" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted benchmark without duplicating completed runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(CONFIG_PATH, OUTPUT_DIR, resume=args.resume)


if __name__ == "__main__":
    main()
