"""Regenerate every figure and the verdicts from ``results/``: ``uv run python scripts/make_figures.py [results]``."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hwm.eval.figures import make_all
from hwm.eval.hypotheses import write_verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results")
    args = ap.parse_args()
    write_verdicts(args.results)  # figures read verdicts.json (H4 curve)
    for p in make_all(args.results):
        print(p)


if __name__ == "__main__":
    main()
